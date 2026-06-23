#!/usr/bin/env python3
"""
convert_legacy_csv.py — Up-convert a legacy benchmark CSV (the Feb Gaudi2 OHF
baseline, written before parse_log.py grew the model/precision/energy columns)
into the CURRENT parse_log.py schema so it merges and plots alongside fresh runs.

The legacy file has no model/precision/vllm_version/input_len/tp_size/
cpu_offload_gb/throughput_std/avg_power_w columns. Everything we can carry over
is carried over verbatim; everything the legacy run never recorded is left blank
(so plot_results.load() coerces it to NaN) rather than invented. model,
precision and device are supplied on the command line because they are not
recoverable from the legacy rows.

Usage:
    python convert_legacy_csv.py IN.csv OUT.csv \
        --device gaudi2-ohf \
        --model NousResearch/Meta-Llama-3.1-8B-Instruct \
        --precision fp8
"""

import argparse
import csv
import sys

# The authoritative target schema — keep in lockstep with parse_log.py's row.
NEW_HEADER = [
    "device", "model", "precision", "vllm_version", "timestamp", "seq_len",
    "batch_size", "input_len", "tp_size", "cpu_offload_gb", "run_id",
    "slurm_jobid", "wall_time_s", "throughput", "throughput_std",
    "first_token_ms", "rest_token_ms", "end2end_ms", "mem_alloc_gb",
    "mem_max_gb", "total_mem_gb", "graph_compile_s", "avg_power_w",
    "energy_j_est", "energy_per_token", "logfile",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("outfile")
    ap.add_argument("--device", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--precision", required=True)
    args = ap.parse_args()

    with open(args.infile, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit(f"no data rows in {args.infile}")

    n = 0
    with open(args.outfile, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=NEW_HEADER)
        w.writeheader()
        for r in rows:
            out = {k: "" for k in NEW_HEADER}
            # values supplied (not recoverable from legacy rows)
            out["device"] = args.device
            out["model"] = args.model
            out["precision"] = args.precision
            # values carried over verbatim where the legacy file has them
            for col in ("timestamp", "seq_len", "batch_size", "run_id",
                        "slurm_jobid", "wall_time_s", "throughput",
                        "first_token_ms", "rest_token_ms", "end2end_ms",
                        "mem_alloc_gb", "mem_max_gb", "total_mem_gb",
                        "graph_compile_s", "energy_j_est", "energy_per_token",
                        "logfile"):
                if r.get(col, "") != "":
                    out[col] = r[col]
            w.writerow(out)
            n += 1

    print(f"✓ {n} rows  {args.infile} → {args.outfile}  (device={args.device}, "
          f"model={args.model}, precision={args.precision})")


if __name__ == "__main__":
    main()
