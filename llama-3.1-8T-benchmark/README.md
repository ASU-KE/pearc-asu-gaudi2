# Llama-3.1 inference benchmark (Gaudi2 vs A100 / H100 / GH200)

vLLM is the common serving engine on **every** platform, so the comparison is
apples-to-apples. Each run does in-process warmup (graph capture/compile happens
*before* timing), forces an exact output length, measures prefill separately from
decode, repeats and reports the **median**, and samples board power live for a
per-config energy number. Implemented per `plan.txt`.

> Replaces the earlier setup (optimum-habana `run_generation.py` on Gaudi vs a
> hand-rolled `cuda_benchmark.py` on CUDA). That CUDA path counted `torch.compile`
> time as inference (≈8 s "first token" on A100) and produced an implausible ~25×
> Gaudi speedup. Both `cuda_benchmark.py` and the duplicate `plot.py` were removed.

## Files

| File | Role |
|---|---|
| `run_vllm_cuda.py` | vLLM driver for A100 (BF16) / H100 / GH200 (BF16, FP8) |
| `run_vllm_gaudi.py` | vLLM driver for Gaudi2 HPU (BF16, FP8 via INC) |
| `run_optimum_gaudi.py` | Optimum-Habana driver for Gaudi2 — **lazy mode + HPU graphs** (Feb baseline), same schema |
| `gaudi_{8b,70b}_ohf.sh` | Gaudi2 **lazy-path** jobs → series `gaudi2-ohf` |
| `plot_results_ohf.py` | overlay plots that add the `gaudi2-ohf` (lazy) series in a distinct colour |
| `bench_common.sh` | shared sweep loop, sourced by every sbatch script |
| `{gaudi,a100,h100,gh200}_8b.sh` | per-device **8B** jobs (single card, concurrency 1–128) |
| `{gaudi,a100,h100,gh200}_70b.sh` | per-device **70B** jobs (multi-card, concurrency 1–32) |
| `parse_log.py` | one run log → one CSV row (owns the CSV header) |
| `merge_results.sh` | merge every `bench_results/<device>/results.csv` → `merged.csv` |
| `plot_results.py` | figures (color = device, linestyle = precision) |
| `quantization_config/` | Gaudi INC calibration configs (FP8) |

## Workflow — 8B first, then 70B

Set `BENCH_ROOT` (top of each sbatch) and choose `RUN_BACKEND=mamba` (default) or
`RUN_BACKEND=apptainer` (set `MAMBA_ENV` / `APPTAINER_SIF`).

```bash
# 1) 8B — single card everywhere, fast; validates the whole pipeline
sbatch gaudi_8b.sh ; sbatch a100_8b.sh ; sbatch h100_8b.sh ; sbatch gh200_8b.sh

# 2) 70B — later, when you have time / the multi-card allocations
sbatch gaudi_70b.sh ; sbatch a100_70b.sh ; sbatch h100_70b.sh ; sbatch gh200_70b.sh

# 3) merge + plot (run anywhere with pandas+matplotlib; not on the accelerators)
bash merge_results.sh "$BENCH_ROOT"
python plot_results.py "$BENCH_ROOT/bench_results/merged.csv"
```

Each device appends both models to one `bench_results/<device>/results.csv`
(`rm` it to start clean). The exact vLLM version is recorded per row.

## Gaudi2 lazy-mode track (`gaudi2-ohf`)

The vLLM-Gaudi plugin image runs `torch.compile`/eager (its upstream-bridge torch
can't do lazy mode). The classic, traditionally-tuned Habana path is **lazy mode +
HPU graphs** via Optimum-Habana `run_generation.py` — the original Feb baseline.
`run_optimum_gaudi.py` reproduces it behind the *same* CLI + printed schema, so
`bench_common.sh` and `parse_log.py` are reused unchanged. It runs as a **separate
series** (`device = gaudi2-ohf`) at the same workload shape as the vLLM sweep, so
lazy-vs-compile lines up point-for-point.

```bash
# needs the SynapseAI-1.24-matched mamba env (`gaudi-1.24.0`; lazy verified) +
# optimum-habana installed into it (see below)
export OHF_TEXTGEN_DIR=/path/to/optimum-habana/examples/text-generation
sbatch gaudi_8b_ohf.sh         # 8B BF16 + FP8, single card
sbatch gaudi_70b_ohf.sh        # 70B BF16 (tp=2 via gaudi_spawn/DeepSpeed) + FP8

bash merge_results.sh "$BENCH_ROOT"     # picks up bench_results/gaudi2-ohf/ too
python plot_results_ohf.py "$BENCH_ROOT/bench_results/merged.csv"
```

`optimum-habana` must be **installed into the active env** (`pip install -e
/path/to/optimum-habana`) — that pulls the base `optimum` package + matching
`transformers`. A bare source checkout on `PYTHONPATH` does **not** work: it
ships only `optimum/habana/` and shadows the base `optimum` namespace, breaking
imports like `optimum.configuration_utils`. The driver imports the installed
package first and only falls back to the checkout path as a last resort.

FP8 here has **no `--fp8` flag** — this optimum-habana build drives INC purely off
the `QUANT_CONFIG` env var (the model still runs `--bf16`; INC quantizes on top).
Run the measurement pass **through `run_generation.py`** (not the vLLM driver — the
layer-name mapping differs) from `OHF_TEXTGEN_DIR`, so its `hqt_output/` lands there:

```bash
cd "$OHF_TEXTGEN_DIR"
QUANT_CONFIG=…/quantization_config/maxabs_measure.json \
  python run_generation.py --model_name_or_path NousResearch/Meta-Llama-3.1-8B-Instruct \
    --bf16 --use_hpu_graphs --use_kv_cache --batch_size 1 --max_new_tokens 32
# benchmark runs then read maxabs_quant.json via QUANT_CONFIG_FP8 (wired in the sbatch)
```

TTFT/ITL are blank for this series (`run_generation.py` reports aggregate
throughput, not a prefill/decode split), so those two panels omit it; throughput
also has no error bar (one aggregate figure over `--n_iterations`).

### Changes from the original Feb baseline (`gaudi_bench.sh`)

`run_optimum_gaudi.py` keeps the Feb lazy + HPU-graphs invocation
(`--use_hpu_graphs --use_kv_cache --trim_logits --reuse_cache --bf16
--trust_remote_code`, `PT_HPU_LAZY_MODE=1`) but differs as follows:

| Aspect | Feb `gaudi_bench.sh` | Now |
|---|---|---|
| Model | `Meta-Llama-3-8B` (base) | `Meta-Llama-3.1-8B-Instruct` |
| Input length | uncontrolled (default prompt) | `--max_input_tokens 512` |
| Output length | swept `--max_new_tokens` ∈ {128, 512, 2048} | fixed `--max_new_tokens 256` |
| Batch | {1, 8, 32, 128} | {1, 8, 32, 64, 128} |
| Repeats | 3 separate processes (cold init each); run_generation defaults warmup=3/n_iters=5 | 1 process, `--warmup 2 --n_iterations 3` (graphs compiled once) |
| Precision | one config, `--bf16` **with `QUANT_CONFIG` exported globally** | explicit `bf16` (no `QUANT_CONFIG`) **and** `fp8` (`QUANT_CONFIG` set) |

**Important — the old "BF16" was probably FP8.** Feb exported
`QUANT_CONFIG=…/maxabs_quant.json` for *every* run while passing `--bf16`.
`run_generation.py` activates INC quantization whenever `QUANT_CONFIG` is in the
environment, so those Feb runs were almost certainly **INC-FP8**, not pure BF16
(provided a measurement pass had populated `hqt_output/`). The new track sets
`QUANT_CONFIG` **only** for the `fp8` precision, so its `bf16` series is genuinely
BF16 and is expected to be slower than the Feb "bf16" numbers — that gap is the
quantization effect, now reported explicitly rather than hidden in the baseline.

## Precision per platform

Two precisions: **BF16** (16-bit baseline) and **FP8 (E4M3)** (8-bit). INT8 is
out of scope. A100 has no FP8 tensor cores, so it runs **BF16 only** — that's a
hardware limitation (it appears in the BF16 comparison, not the FP8 one), not an
unfair exclusion.

| Precision | A100 | H100 / GH200 | Gaudi2 |
|---|---|---|---|
| BF16 | native | native | native |
| FP8 (E4M3) | — (no HW) | vLLM online dynamic W8A8 (`quantization=fp8`, `kv_cache_dtype=fp8`) | INC, `quantization=inc` + calibration |

* **CUDA FP8** is online/dynamic from the BF16 checkpoint — no separate model.
* **Gaudi FP8** goes through Intel Neural Compressor (INC) on the base checkpoint.
  Record the exact vLLM + vllm-gaudi/INC versions (captured per CSV row).

### FP8 calibration on Gaudi (one-time, per model)

INC needs a measurement pass (collects max-abs scales) before quantized serving.
Run once per model; the sbatch scripts then point `QUANT_CONFIG_FP8` at the quant
config for the measured runs.

```bash
# measurement pass (writes hqt_output/… next to CWD)
QUANT_CONFIG=quantization_config/maxabs_measure.json \
  python run_vllm_gaudi.py --model meta-llama/Llama-3.1-8B-Instruct \
    --precision fp8 --batch-size 1 --output-len 32 --warmup-iters 0 --num-iters 1
# the benchmark then runs with maxabs_quant.json (already wired in the sbatch scripts)
```

## Topology (70B; see `plan.txt`)

| Platform | 70B BF16 | 70B FP8 |
|---|---|---|
| A100 80 GB | tp=4 (4× comfortable, 2× tight) | — (no FP8) |
| H100 80 GB | tp=2 (near edge; bump to 4 if OOM) | tp=2 |
| Gaudi2 96 GB | tp=2 | tp=1 (single card) |
| GH200 96 GB | **tp=1 + `--cpu-offload-gb 70`** → Grace LPDDR | tp=1 (in HBM) |

GH200 70B BF16 is offloaded, so it's recorded with `cpu_offload_gb>0` and plotted
as **"GH200 bf16 (offload)"** — never compared as an in-HBM result.

## Metrics (CSV columns)

`throughput` (median tok/s) + `throughput_std`, `first_token_ms` (TTFT),
`rest_token_ms` (ITL/TPOT), `end2end_ms`, `mem_max_gb` (peak), and
`avg_power_w` / `energy_per_token` from live `nvidia-smi` / `hl-smi` sampling.

## Caveats

* Single workload slice (512 in / 256 out) — doesn't generalize to long context/output.
* vLLM feature/version parity isn't guaranteed across CUDA and HPU; the version is
  recorded per row — cite it.
* Energy assumes the job sees only its own accelerators (cgroup isolation); the
  70B jobs use `--exclusive`. On GH200, `nvidia-smi` power may not include the
  Grace CPU during offloaded BF16.
