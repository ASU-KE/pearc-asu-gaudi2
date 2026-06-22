#!/bin/bash
# ============================================================
#  merge_results.sh — Merge every per-device results.csv into one merged.csv.
#  Finds bench_results/<device>/results.csv for all devices automatically,
#  so adding a new card (e.g. gh200) needs no edit here.
#  Usage:  bash merge_results.sh [BENCH_ROOT]
# ============================================================
set -eo pipefail

BENCH_ROOT="${1:-/scratch/tianche5/gaudi_bench}"
OUTROOT="${BENCH_ROOT}/bench_results"
MERGED="${OUTROOT}/merged.csv"

shopt -s nullglob
CSVS=("${OUTROOT}"/*/results.csv)
if [ ${#CSVS[@]} -eq 0 ]; then
  echo "No per-device results.csv found under ${OUTROOT}" >&2
  exit 1
fi

# header from the first file
head -1 "${CSVS[0]}" > "${MERGED}"

# append data rows from every device
for f in "${CSVS[@]}"; do
  n=$(tail -n +2 "$f" | wc -l | tr -d ' ')
  tail -n +2 "$f" >> "${MERGED}"
  echo "  ✓ ${n} rows from ${f}"
done

echo "✓ Merged CSV → ${MERGED}  ($(( $(wc -l < "${MERGED}") - 1 )) data rows)"
