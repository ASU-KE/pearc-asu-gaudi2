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
import time

ENGINE_VERSION = "optimum-habana"


def _prepend_env_path(var_name, path):
    cur = os.environ.get(var_name, "")
    parts = [p for p in cur.split(os.pathsep) if p]
    if path in parts:
        return
    os.environ[var_name] = path if not cur else f"{path}{os.pathsep}{cur}"


def _ensure_optimum_habana_importable(textgen):
    """Ensure optimum.habana resolves in this process and in child launches."""
    repo_root = os.path.abspath(os.path.join(textgen, os.pardir, os.pardir))
    pkg_init = os.path.join(repo_root, "optimum", "habana", "__init__.py")
    if os.path.isfile(pkg_init):
        _prepend_env_path("PYTHONPATH", repo_root)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
    try:
        oh = importlib.import_module("optimum.habana")
        return f"optimum-habana {getattr(oh, '__version__', '?')}"
    except ModuleNotFoundError as exc:
        if exc.name not in {"optimum", "optimum.habana"}:
            raise
        if os.path.isfile(pkg_init):
            hint = (
                f"Detected source checkout root: {repo_root}\n"
                "Try one of:\n"
                f"  export PYTHONPATH={repo_root}:$PYTHONPATH\n"
                f"  pip install -e {repo_root}\n"
            )
        else:
            hint = (
                "OHF_TEXTGEN_DIR does not look like an optimum-habana checkout.\n"
                "Point OHF_TEXTGEN_DIR to "
                ".../optimum-habana/examples/text-generation\n"
                "or activate an environment where optimum-habana is installed.\n"
            )
        raise SystemExit(
            "Cannot import optimum.habana required by run_generation.py.\n"
            f"OHF_TEXTGEN_DIR={textgen}\n"
            f"{hint}"
        ) from exc


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

    gen = [
        gen_py,
        "--model_name_or_path", args.model,
        "--batch_size", str(args.batch_size),
        "--max_input_tokens", str(args.input_len),
        "--max_new_tokens", str(args.output_len),
        "--warmup", str(max(args.warmup_iters, 1)),
        "--n_iterations", "1",          # one measured pass per child; we repeat
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
        os.environ["QUANT_CONFIG"] = cfg     # INC reads this at load time
        gen += ["--fp8"]
    else:
        gen += ["--bf16"]

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

    throughputs, graph_s_first = [], None
    mem_max = mem_alloc = mem_total = None
    sampler_cm = (contextlib.nullcontext() if args.no_power
                  else HlSmiPowerSampler(args.power_interval_ms,
                                         args.tensor_parallel_size))
    compile_start = time.perf_counter()
    with sampler_cm as sampler:
        for i in range(args.num_iters):
            r = run_child(cmd, cwd)
            throughputs.append(r["throughput"])
            if graph_s_first is None:
                graph_s_first = r["graph_s"]      # first child pays compilation
            for k, dst in (("mem_max", "mem_max"), ("mem_alloc", "mem_alloc"),
                           ("mem_total", "mem_total")):
                if r[k] is not None:
                    if k == "mem_max":
                        mem_max = max(mem_max or 0, r[k])
                    elif k == "mem_alloc":
                        mem_alloc = max(mem_alloc or 0, r[k])
                    else:
                        mem_total = r[k]
    if graph_s_first is not None:
        print(f"Graph compilation duration = {graph_s_first:.2f} s")

    thr_med = statistics.median(throughputs)
    thr_std = statistics.stdev(throughputs) if len(throughputs) > 1 else 0.0
    # end-to-end latency for the batch, consistent with the vLLM driver's
    # definition (wall to generate output_len tokens for the whole batch).
    e2e_ms = (args.batch_size * args.output_len) / max(thr_med, 1e-9) * 1000.0

    print(f"Throughput (output tokens) = {thr_med:.2f} tokens/s")
    print(f"Throughput std = {thr_std:.2f} tokens/s")
    print(f"Average end to end latency = {e2e_ms:.2f} ms")

    # prefer hl-smi peak mem; fall back to run_generation.py's reported numbers
    if sampler and sampler.peak_mem_gb is not None:
        mem_max = sampler.peak_mem_gb
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
