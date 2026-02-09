#!/usr/bin/env python3
"""
parse_log.py — Parse a benchmark log file and append one CSV row.
Called by all three sbatch scripts (gaudi2 / a100 / h100).

Usage:
    python parse_log.py <logfile> <device> <timestamp> <seq_len> \
                        <batch_size> <run_id> <wall_time_s> <csv_path>
"""

import os, sys, re, csv

logpath   = sys.argv[1]
device    = sys.argv[2]
stamp     = sys.argv[3]
seq       = sys.argv[4]
bs        = sys.argv[5]
run_id    = sys.argv[6]
wall_s    = sys.argv[7]
csv_path  = sys.argv[8]
jobid     = os.environ.get("SLURM_JOB_ID", "")

text = open(logpath, "r", errors="ignore").read()


def grab(pattern):
    m = re.search(pattern, text, re.M)
    return float(m.group(1)) if m else ""


metrics = {
    "throughput":      grab(r"Throughput.*?=\s*([0-9]+\.[0-9]+)"),
    "first_token_ms":  grab(r"Average first token latency\s*=\s*([0-9]+\.[0-9]+)"),
    "rest_token_ms":   grab(r"Average rest token latency\s*=\s*([0-9]+\.[0-9]+)"),
    "end2end_ms":      grab(r"Average end to end latency\s*=\s*([0-9]+\.[0-9]+)"),
    "mem_alloc_gb":    grab(r"Memory allocated\s*=\s*([0-9]+\.[0-9]+)\s*GB"),
    "mem_max_gb":      grab(r"Max memory allocated\s*=\s*([0-9]+\.[0-9]+)\s*GB"),
    "total_mem_gb":    grab(r"Total memory available\s*=\s*([0-9]+\.[0-9]+)\s*GB"),
    "graph_compile_s": grab(r"Graph compilation duration\s*=\s*([0-9]+\.[0-9]+)"),
}

row = [
    device, stamp, seq, bs, run_id, jobid, wall_s,
    metrics["throughput"],
    metrics["first_token_ms"],
    metrics["rest_token_ms"],
    metrics["end2end_ms"],
    metrics["mem_alloc_gb"],
    metrics["mem_max_gb"],
    metrics["total_mem_gb"],
    metrics["graph_compile_s"],
    logpath,
]

with open(csv_path, "a") as fh:
    csv.writer(fh).writerow(row)

print(f"APPENDED  {run_id}  thr={metrics['throughput']}  e2e={metrics['end2end_ms']}")
