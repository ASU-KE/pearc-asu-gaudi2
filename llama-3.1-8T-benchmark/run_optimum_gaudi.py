#!/usr/bin/env python3
"""
run_optimum_gaudi.py — Optimum-Habana inference benchmark for Intel Gaudi2 (HPU)
in the classic LAZY mode + HPU-graphs path (the Feb baseline), exposed with the
SAME CLI and SAME printed-metric schema as run_vllm_gaudi.py so the shared
bench_common.sh sweep harness and parse_log.py both handle it unchanged.

Why this exists
---------------
The vLLM-Gaudi plugin image ships the upstream plugin-bridge torch (2.10+cpu),
which only runs torch.compile/eager — lazy mode aborts on import. The original
(Feb) Gaudi2 numbers were produced by Optimum-Habana's text-generation example
(`run_generation.py`) under PT_HPU_LAZY_MODE=1 with `--use_hpu_graphs`, i.e. the
traditionally tuned Habana path. This driver reproduces THAT path as a separate
series (device label `gaudi2-ohf`) so it can be overlaid against the vLLM runs.

How it works
------------
Rather than re-implement Habana generation, we shell out to the proven
`run_generation.py` once per measured iteration (each child does its own internal
warmup so HPU graphs are compiled before the timed pass), parse its throughput /
memory / graph-compile stats, and re-emit them in the common schema. Board power
is sampled live over the timed children via hl-smi and folded into an
energy-per-token estimate.

Limitations vs the vLLM driver
------------------------------
`run_generation.py` reports aggregate throughput, not a separate prefill (TTFT)
vs decode (TPOT) split, so first/rest-token latencies are left blank (those plot
panels simply omit the gaudi2-ohf series). End-to-end latency is derived from
throughput so it stays consistent with the vLLM definition.

Environment
-----------
  OHF_TEXTGEN_DIR  path to optimum-habana/examples/text-generation (has
                   run_generation.py). REQUIRED. If this points to a source
                   checkout, the driver auto-adds the checkout root to
                   PYTHONPATH so optimum.habana can be imported.
  OHF_SPAWN        path to optimum-habana/examples/gaudi_spawn.py (multi-card,
                   only needed when --tensor-parallel-size > 1). Defaults to
                   ../gaudi_spawn.py relative to OHF_TEXTGEN_DIR.
  QUANT_CONFIG_FP8 (or QUANT_CONFIG) maxabs_quant.json for FP8; a one-time INC
                   measurement pass (QUANT_CONFIG=maxabs_measure.json) must have
                   been run first so hqt_output/ exists. See README.
"""

import argparse
import contextlib
import importlib
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading

ENGINE_VERSION = "optimum-habana"


def _prepend_env_path(var_name, path):
    cur = os.environ.get(var_name, "")
    parts = [p for p in cur.split(os.pathsep) if p]
    if path in parts:
        return
    os.environ[var_name] = path if not cur else f"{path}{os.pathsep}{cur}"


def _ensure_optimum_habana_importable(textgen):
    """Return an engine-version string, importing optimum.habana.

    Prefer a properly installed package: a clean import resolves both
    optimum.habana AND the base `optimum` namespace (configuration_utils, etc.).
    Only if that fails do we add the source checkout to the path as a last
    resort — a bare checkout root on sys.path SHADOWS the installed base
    `optimum` (it ships only optimum/habana/), which breaks imports like
    optimum.configuration_utils. So we never inject when a real install works.
    """
    repo_root = os.path.abspath(os.path.join(textgen, os.pardir, os.pardir))
    pkg_init = os.path.join(repo_root, "optimum", "habana", "__init__.py")

    def _do_import():
        oh = importlib.import_module("optimum.habana")
        return f"optimum-habana {getattr(oh, '__version__', '?')}"

    # 1) clean import from the active environment (the correct path).
    try:
        return _do_import()
    except ModuleNotFoundError:
        pass

    # 2) fallback: source checkout on the path (best-effort; only helps if its
    #    deps — base optimum, transformers — are already installed).
    if os.path.isfile(pkg_init) and repo_root not in sys.path:
        _prepend_env_path("PYTHONPATH", repo_root)
        sys.path.insert(0, repo_root)
        try:
            return _do_import()
        except ModuleNotFoundError:
            pass

    hint = (f"  pip install -e {repo_root}\n" if os.path.isfile(pkg_init)
            else "  point OHF_TEXTGEN_DIR at .../optimum-habana/examples/"
                 "text-generation,\n  or activate an env where optimum-habana "
                 "is installed.\n")
    raise SystemExit(
        "Cannot import optimum.habana (or its base `optimum` dependency).\n"
        "Install optimum-habana INTO the active env — a bare source checkout on\n"
        "PYTHONPATH shadows the base `optimum` package and breaks imports such as\n"
        "optimum.configuration_utils. Recommended:\n"
        f"{hint}"
        f"OHF_TEXTGEN_DIR={textgen}\n")


class HlSmiPowerSampler:
    """Poll hl-smi power.draw / memory in a background thread over the timed
    region, summed across the first `num_devices` AIPs (the cards a tp run uses).
    Assumes an exclusive allocation. No-op if hl-smi is absent. Identical
    methodology to run_vllm_gaudi.py so the energy metric is comparable."""

    def __init__(self, interval_ms=100, num_devices=1):
        self.interval = interval_ms / 1000.0
        self.num_devices = num_devices
        self.samples, self.mem_used = [], []
        self.mem_total = None
        self._stop = threading.Event()
        self._thread = None
        self._have = shutil.which("hl-smi") is not None

    def _poll(self):
        cmd = ["hl-smi", "-Q", "power.draw,memory.used,memory.total",
               "-f", "csv,noheader,nounits"]
        try:
            out = subprocess.check_output(cmd, text=True, timeout=5,
                                          stderr=subprocess.DEVNULL)
            psum = msum = mtot = 0.0
            for line in out.strip().splitlines()[:self.num_devices]:
                p, mu, mt = [x.strip() for x in line.split(",")]
                psum += float(p)
                msum += float(mu) / 1024.0    # MiB -> GiB
                mtot += float(mt) / 1024.0
            if mtot > 0:
                self.samples.append(psum)
                self.mem_used.append(msum)
                self.mem_total = mtot
        except Exception:
            pass

    def _loop(self):
        while not self._stop.is_set():
            self._poll()
            self._stop.wait(self.interval)

    def __enter__(self):
        if self._have:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    @property
    def avg_power_w(self):
        return statistics.mean(self.samples) if self.samples else None

    @property
    def peak_mem_gb(self):
        return max(self.mem_used) if self.mem_used else None


def _f(pattern, text, default=None):
    m = re.search(pattern, text, re.I)
    return float(m.group(1)) if m else default


def build_command(args):
    """Map the common CLI onto an optimum-habana run_generation.py invocation.
    For tp>1 (70B) we route through gaudi_spawn.py with DeepSpeed inference."""
    textgen = os.environ.get("OHF_TEXTGEN_DIR")
    if not textgen:
        raise SystemExit("OHF_TEXTGEN_DIR must point to "
                         "optimum-habana/examples/text-generation")
    gen_py = os.path.join(textgen, "run_generation.py")
    if not os.path.isfile(gen_py):
        raise SystemExit(f"run_generation.py not found at {gen_py}")
    global ENGINE_VERSION
    ENGINE_VERSION = _ensure_optimum_habana_importable(textgen)

    # One process does warmup + all measured iterations (graphs compile once);
    # restarting per repeat would re-pay the multi-minute init/compile each time.
    gen = [
        gen_py,
        "--model_name_or_path", args.model,
        "--batch_size", str(args.batch_size),
        "--max_input_tokens", str(args.input_len),
        "--max_new_tokens", str(args.output_len),
        "--warmup", str(max(args.warmup_iters, 1)),
        "--n_iterations", str(max(args.num_iters, 1)),
        "--bf16",                       # base compute dtype (FP8 layers on via INC)
        "--use_hpu_graphs",
        "--use_kv_cache",
        "--trim_logits",
        "--reuse_cache",
        "--trust_remote_code",
    ]
    if args.precision == "fp8":
        cfg = os.environ.get("QUANT_CONFIG_FP8") or os.environ.get("QUANT_CONFIG")
        if not cfg:
            raise SystemExit(
                "FP8 on Gaudi needs QUANT_CONFIG_FP8 (or QUANT_CONFIG) set to a "
                "maxabs config AND a prior INC measurement pass. See README: "
                "'FP8 calibration on Gaudi'.")
        # This optimum-habana build has NO --fp8 flag: INC quantizes to FP8 from
        # the bf16 run purely off the QUANT_CONFIG env var (child inherits it).
        os.environ["QUANT_CONFIG"] = cfg

    if args.tensor_parallel_size > 1:
        spawn = os.environ.get(
            "OHF_SPAWN", os.path.join(textgen, os.pardir, "gaudi_spawn.py"))
        if not os.path.isfile(spawn):
            raise SystemExit(f"OHF_SPAWN not found at {spawn} (needed for tp>1)")
        return [sys.executable, spawn, "--use_deepspeed",
                "--world_size", str(args.tensor_parallel_size)] + gen
    return [sys.executable] + gen


def run_child(cmd, cwd):
    out = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)
    sys.stderr.write(out.stdout)                 # surface child log for debugging
    if out.returncode != 0:
        raise RuntimeError(f"run_generation.py exited {out.returncode}")
    text = out.stdout
    thr = _f(r"throughput[^=]*=\s*([0-9.]+)\s*tokens", text)
    if thr is None:
        raise RuntimeError("could not parse throughput from run_generation.py")
    return {
        "throughput": thr,
        "mem_alloc": _f(r"Memory allocated\s*=\s*([0-9.]+)\s*GB", text),
        "mem_max": _f(r"Max memory allocated\s*=\s*([0-9.]+)\s*GB", text),
        "mem_total": _f(r"Total memory available\s*=\s*([0-9.]+)\s*GB", text),
        "graph_s": _f(r"Graph compilation duration\s*=\s*([0-9.]+)", text),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--precision", choices=["bf16", "fp8"], default="bf16")
    p.add_argument("--input-len", type=int, default=512)
    p.add_argument("--output-len", type=int, default=256,
                   help="generated tokens (recorded as seq_len)")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-iters", type=int, default=3)
    p.add_argument("--warmup-iters", type=int, default=2)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                   help="accepted for CLI parity; unused by optimum-habana")
    p.add_argument("--max-model-len", type=int, default=0,
                   help="accepted for CLI parity; unused")
    p.add_argument("--enforce-eager", action="store_true",
                   help="accepted for CLI parity; this driver is lazy+graphs")
    p.add_argument("--device-label", default="gaudi2-ohf")
    p.add_argument("--no-power", action="store_true")
    p.add_argument("--power-interval-ms", type=int, default=100)
    p.add_argument("--cpu-offload-gb", type=float, default=0)  # parity; ignored
    args = p.parse_args()

    cmd = build_command(args)
    cwd = os.environ.get("OHF_TEXTGEN_DIR")       # so hqt_output/ etc. resolve

    print(f"vLLM version = {ENGINE_VERSION}")     # provenance (engine, not vLLM)
    print(f"Device = {args.device_label}")
    print(f"Model = {args.model}")
    print(f"Precision = {args.precision}")
    print(f"Tensor parallel size = {args.tensor_parallel_size}")
    print(f"CPU offload = 0 GB")
    print(f"Input length = {args.input_len}")
    print(f"Output length = {args.output_len}")
    print(f"Batch size = {args.batch_size}")
    sys.stdout.flush()

    sampler_cm = (contextlib.nullcontext() if args.no_power
                  else HlSmiPowerSampler(args.power_interval_ms,
                                         args.tensor_parallel_size))
    with sampler_cm as sampler:
        r = run_child(cmd, cwd)               # warmup + all measured iters in one go
    mem_max, mem_alloc, mem_total = r["mem_max"], r["mem_alloc"], r["mem_total"]
    if r["graph_s"] is not None:
        print(f"Graph compilation duration = {r['graph_s']:.2f} s")

    thr_med = r["throughput"]
    # run_generation.py reports one aggregate throughput over --n_iterations, so
    # there is no per-iteration spread to report (the OHF series has no error bar).
    thr_std = 0.0
    # end-to-end latency for the batch, consistent with the vLLM driver's
    # definition (wall to generate output_len tokens for the whole batch).
    e2e_ms = (args.batch_size * args.output_len) / max(thr_med, 1e-9) * 1000.0

    print(f"Throughput (output tokens) = {thr_med:.2f} tokens/s")
    print(f"Throughput std = {thr_std:.2f} tokens/s")
    print(f"Average end to end latency = {e2e_ms:.2f} ms")

    # NOTE: hl-smi `memory.used` does NOT track Synapse/HBM tensor allocations on
    # Gaudi (it stays near the idle ~0.75 GB even with ~19 GB of live tensors), so
    # sampler.peak_mem_gb badly under-reports. Trust run_generation.py's
    # torch.hpu.max_memory_allocated() value (mem_max) for peak memory — the direct
    # analog of torch.cuda.max_memory_allocated() on the NVIDIA side. hl-smi is
    # still used for board power/energy below, where it is accurate.
    if sampler and sampler.mem_total is not None:
        mem_total = sampler.mem_total or mem_total
    if mem_alloc is not None:
        print(f"Memory allocated = {mem_alloc:.2f} GB")
    if mem_max is not None:
        print(f"Max memory allocated = {mem_max:.2f} GB")
    if mem_total is not None:
        print(f"Total memory available = {mem_total:.2f} GB")

    if sampler and sampler.avg_power_w is not None:
        # steady-state energy: avg board power / throughput avoids charging the
        # per-child graph-compile time to the energy-per-token figure.
        epm = sampler.avg_power_w / max(thr_med, 1e-9)
        tokens = args.num_iters * args.batch_size * args.output_len
        print(f"Average power = {sampler.avg_power_w:.2f} W")
        print(f"Total energy = {epm * tokens:.2f} J")
        print(f"Energy per token = {epm:.6f} J")


if __name__ == "__main__":
    main()
