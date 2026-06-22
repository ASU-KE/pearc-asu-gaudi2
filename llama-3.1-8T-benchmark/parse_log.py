#!/usr/bin/env python3
"""
parse_log.py — Parse one benchmark log (run_vllm_cuda.py / run_vllm_gaudi.py)
and append one CSV row. Called by bench_common.sh for every device.

Usage:
    python parse_log.py <logfile> <device> <timestamp> <seq_len> \
                        <batch_size> <run_id> <wall_time_s> <csv_path>

<seq_len> is the output length and <batch_size> the concurrency passed to the
driver; everything else (model, precision, tp, latencies, power, energy) is read
out of the log so the row is self-describing.
"""

import os
import sys
import re
import csv

logpath, device, stamp, seq, bs, run_id, wall_s, csv_path = sys.argv[1:9]
jobid = os.environ.get("SLURM_JOB_ID", "")

text = open(logpath, "r", errors="ignore").read()

NUM = r"([0-9]+(?:\.[0-9]+)?)"


def fnum(pattern):
    m = re.search(pattern, text, re.M)
    return float(m.group(1)) if m else ""


def fstr(pattern):
    m = re.search(pattern, text, re.M)
    return m.group(1).strip() if m else ""


row = {
    "device": device,
    "model": fstr(r"^Model\s*=\s*(.+)$"),
    "precision": fstr(r"^Precision\s*=\s*(\w+)"),
    "vllm_version": fstr(r"^vLLM version\s*=\s*(.+)$"),
    "timestamp": stamp,
    "seq_len": seq,
    "batch_size": bs,
    "input_len": fnum(r"Input length\s*=\s*" + NUM),
    "tp_size": fnum(r"Tensor parallel size\s*=\s*" + NUM),
    "cpu_offload_gb": fnum(r"CPU offload\s*=\s*" + NUM + r"\s*GB"),
    "run_id": run_id,
    "slurm_jobid": jobid,
    "wall_time_s": wall_s,
    "throughput": fnum(r"Throughput \(output tokens\)\s*=\s*" + NUM),
    "throughput_std": fnum(r"Throughput std\s*=\s*" + NUM),
    "first_token_ms": fnum(r"Average first token latency\s*=\s*" + NUM),
    "rest_token_ms": fnum(r"Average rest token latency\s*=\s*" + NUM),
    "end2end_ms": fnum(r"Average end to end latency\s*=\s*" + NUM),
    "mem_alloc_gb": fnum(r"Memory allocated\s*=\s*" + NUM + r"\s*GB"),
    "mem_max_gb": fnum(r"Max memory allocated\s*=\s*" + NUM + r"\s*GB"),
    "total_mem_gb": fnum(r"Total memory available\s*=\s*" + NUM + r"\s*GB"),
    "graph_compile_s": fnum(r"Graph compilation duration\s*=\s*" + NUM),
    "avg_power_w": fnum(r"Average power\s*=\s*" + NUM + r"\s*W"),
    "energy_j_est": fnum(r"Total energy\s*=\s*" + NUM + r"\s*J"),
    "energy_per_token": fnum(r"Energy per token\s*=\s*" + NUM + r"\s*J"),
    "logfile": logpath,
}

header = list(row.keys())
write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
with open(csv_path, "a", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=header)
    if write_header:
        w.writeheader()
    w.writerow(row)

print(f"APPENDED  {run_id}  thr={row['throughput']}  e2e={row['end2end_ms']}  "
      f"J/tok={row['energy_per_token']}")
