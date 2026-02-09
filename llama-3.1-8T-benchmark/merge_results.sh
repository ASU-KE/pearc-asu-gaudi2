#!/bin/bash
# ============================================================
#  merge_results.sh — Merge the three per-device CSVs
#  Usage:  bash merge_results.sh
# ============================================================
set -eo pipefail

BENCH_ROOT="/scratch/username/gaudi_bench"
OUTROOT="${BENCH_ROOT}/bench_results"
MERGED="${OUTROOT}/merged.csv"

GAUDI_CSV="${OUTROOT}/gaudi2/results.csv"
#A100_CSV="${OUTROOT}/a100/results.csv"
A100_CSV="${OUTROOT}/a100/results.csv"
H100_CSV="${OUTROOT}/h100/results.csv"

# header from first file found
for f in "${GAUDI_CSV}" "${A100_CSV}" "${H100_CSV}"; do
  if [ -f "$f" ]; then
    head -1 "$f" > "${MERGED}"
    break
  fi
done

# append data rows from all three
for f in "${GAUDI_CSV}" "${A100_CSV}" "${H100_CSV}"; do
  if [ -f "$f" ]; then
    tail -n +2 "$f" >> "${MERGED}"
    echo "  ✓ $(tail -n +2 "$f" | wc -l) rows from $f"
  else
    echo "  ⚠ Missing: $f – skipping"
  fi
done

echo "✓ Merged CSV → ${MERGED}  ($(( $(wc -l < "${MERGED}") - 1 )) data rows)"
