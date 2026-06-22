#!/usr/bin/env python3
"""
run_vllm_gaudi.py — vLLM inference benchmark for Intel Gaudi2 (HPU).

Self-contained twin of run_vllm_cuda.py: same methodology, same printed metric
schema (so parse_log.py handles both), but the HPU-specific bits live here so it
can be copied into the vLLM-Gaudi mamba env or apptainer image without dragging
in any CUDA code.

Precision:
  bf16  -> dtype=bfloat16
  fp8   -> quantization=inc (Intel Neural Compressor), QUANT_CONFIG_FP8 (or
           QUANT_CONFIG) -> maxabs_quant.json (E4M3). Needs a one-time INC
           measurement pass first (see README: "FP8 calibration on Gaudi").

Energy comes from live hl-smi power sampling over the timed region.
"""

import argparse
import contextlib
import os
import random
import shutil
import statistics
import subprocess
import threading
import time

import vllm
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

VLLM_VERSION = getattr(vllm, "__version__", "unknown")


class HlSmiPowerSampler:
    """Poll hl-smi power.draw / memory in a background thread over the timed
    region, summed across the first `num_devices` AIPs (the cards used by a
    tensor-parallel run). Assumes an exclusive allocation. No-op if hl-smi
    is absent."""

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


def build_engine(args):
    kwargs = dict(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        enforce_eager=args.enforce_eager,
        disable_log_stats=True,
    )
    if args.max_model_len:
        kwargs["max_model_len"] = args.max_model_len
    if args.precision == "fp8":
        cfg = os.environ.get("QUANT_CONFIG_FP8") or os.environ.get("QUANT_CONFIG")
        if not cfg:
            raise SystemExit(
                "FP8 on Gaudi needs QUANT_CONFIG_FP8 (or QUANT_CONFIG) set to "
                "a maxabs config AND a prior INC measurement pass. "
                "See README: 'FP8 calibration on Gaudi'."
            )
        os.environ["QUANT_CONFIG"] = cfg          # INC reads this at load time
        kwargs["quantization"] = "inc"
        kwargs["kv_cache_dtype"] = "fp8_inc"
    return LLM(**kwargs)


def make_prompts(llm, batch_size, input_len, seed=1234):
    tok = llm.get_tokenizer()
    rng = random.Random(seed)
    hi = max(tok.vocab_size - 100, 100)
    return [TokensPrompt(prompt_token_ids=[rng.randint(10, hi) for _ in range(input_len)])
            for _ in range(batch_size)]


def timed_generate(llm, prompts, out_len):
    sp = SamplingParams(temperature=0.0, max_tokens=out_len,
                        min_tokens=out_len, ignore_eos=True)
    t0 = time.perf_counter()
    llm.generate(prompts, sp, use_tqdm=False)
    return time.perf_counter() - t0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--precision", choices=["bf16", "fp8"], default="bf16")
    p.add_argument("--input-len", type=int, default=512)
    p.add_argument("--output-len", type=int, default=256,
                   help="generated tokens (recorded as seq_len)")
    p.add_argument("--batch-size", type=int, default=1,
                   help="concurrent prompts (recorded as batch_size)")
    p.add_argument("--num-iters", type=int, default=3)
    p.add_argument("--warmup-iters", type=int, default=2)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--max-model-len", type=int, default=0)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--device-label", default="gaudi2")
    p.add_argument("--no-power", action="store_true")
    p.add_argument("--power-interval-ms", type=int, default=100)
    args = p.parse_args()

    print(f"vLLM version = {VLLM_VERSION}")
    print(f"Device = {args.device_label}")
    print(f"Model = {args.model}")
    print(f"Precision = {args.precision}")
    print(f"Tensor parallel size = {args.tensor_parallel_size}")
    print(f"CPU offload = 0 GB")
    print(f"Input length = {args.input_len}")
    print(f"Output length = {args.output_len}")
    print(f"Batch size = {args.batch_size}")

    compile_start = time.perf_counter()
    llm = build_engine(args)
    print(f"Graph compilation duration = {time.perf_counter() - compile_start:.2f} s")

    prompts = make_prompts(llm, args.batch_size, args.input_len)

    # in-process warmup (NOT timed) — builds HPU graphs / bucket recipes
    for _ in range(args.warmup_iters):
        timed_generate(llm, prompts, min(args.output_len, 16))

    total_new = args.batch_size * args.output_len
    full_times, prefill_times = [], []
    sampler_cm = (contextlib.nullcontext() if args.no_power
                  else HlSmiPowerSampler(args.power_interval_ms,
                                         args.tensor_parallel_size))
    with sampler_cm as sampler:
        for _ in range(args.num_iters):
            prefill_times.append(timed_generate(llm, prompts, 1))
            full_times.append(timed_generate(llm, prompts, args.output_len))

    # report MEDIAN over repeats (plan.txt: median + variability)
    full_s = statistics.median(full_times)
    prefill_s = statistics.median(prefill_times)
    rest_token_s = max(full_s - prefill_s, 1e-9) / max(args.output_len - 1, 1)

    thr_per_iter = [total_new / t for t in full_times]
    thr_std = statistics.stdev(thr_per_iter) if len(thr_per_iter) > 1 else 0.0

    print(f"Throughput (output tokens) = {statistics.median(thr_per_iter):.2f} tokens/s")
    print(f"Throughput std = {thr_std:.2f} tokens/s")
    print(f"Average first token latency = {prefill_s * 1000:.2f} ms")
    print(f"Average rest token latency = {rest_token_s * 1000:.2f} ms")
    print(f"Average end to end latency = {full_s * 1000:.2f} ms")

    if sampler and sampler.peak_mem_gb is not None:
        print(f"Memory allocated = {sampler.peak_mem_gb:.2f} GB")
        print(f"Max memory allocated = {sampler.peak_mem_gb:.2f} GB")
        if sampler.mem_total is not None:
            print(f"Total memory available = {sampler.mem_total:.2f} GB")
    if sampler and sampler.avg_power_w is not None:
        measured_wall = sum(full_times) + sum(prefill_times)
        tokens = args.num_iters * total_new + args.num_iters * args.batch_size
        energy_j = sampler.avg_power_w * measured_wall
        print(f"Average power = {sampler.avg_power_w:.2f} W")
        print(f"Total energy = {energy_j:.2f} J")
        print(f"Energy per token = {energy_j / tokens:.6f} J")


if __name__ == "__main__":
    main()
