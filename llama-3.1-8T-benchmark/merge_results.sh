#!/bin/bash
# ============================================================
#  merge_results.sh — Merge every per-device results.csv into one merged.csv.
#  Finds bench_results/<device>/results.csv for all devices automatically
#  (including the Optimum-Habana gaudi2-ohf series), so adding a new card
#  needs no edit here — every device writes the same parse_log.py schema.
#  Usage:  bash merge_results.sh [BENCH_ROOT]
# ============================================================
set -eo pipefail

BENCH_ROOT="${1:-/scratch/tianche5/pearc-asu-gaudi2/llama-3.1-8T-benchmark}"
OUTROOT="${BENCH_ROOT}/bench_results"
MERGED="${OUTROOT}/merged.csv"

shopt -s nullglob
CSVS=("${OUTROOT}"/*/results.csv)
if [ ${#CSVS[@]} -eq 0 ]; then
  echo "No per-device results.csv found under ${OUTROOT}" >&2
  exit 1
fi

# header from the first file
HEADER=$(head -1 "${CSVS[0]}")
printf '%s\n' "${HEADER}" > "${MERGED}"

# append data rows from every device, refusing to merge a CSV whose columns
# don't match — appending mismatched rows would silently misalign every field.
for f in "${CSVS[@]}"; do
  if [ "$(head -1 "$f")" != "${HEADER}" ]; then
    echo "  ✗ header mismatch in ${f} (skipped — columns differ from ${CSVS[0]})" >&2
    continue
  fi
  n=$(tail -n +2 "$f" | wc -l | tr -d ' ')
  tail -n +2 "$f" >> "${MERGED}"
  echo "  ✓ ${n} rows from ${f}"
done

echo "✓ Merged CSV → ${MERGED}  ($(( $(wc -l < "${MERGED}") - 1 )) data rows)"
