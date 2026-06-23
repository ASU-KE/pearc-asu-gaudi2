#!/bin/bash
# ============================================================
#  backfill_gh200_logs.sh — Re-parse already-collected GH200 run logs into a
#  results CSV without re-running any benchmark.
#
#  The GH200 sweep writes its per-run logs fine, but its parse step runs on the
#  arm node whose host python3 is wrong-arch, so no CSV rows get written. This
#  helper does just the parse, on any box with a normal python3 (the logs live on
#  shared /scratch, so run it wherever is convenient).
#
#  Every metric is read out of the log; device / output-len / batch-size / run_id
#  are recovered from the filename, which bench_common.sh writes as:
#      run_<device>_<label>_<prec>_o<out>_b<bs>_<epoch>.log
#  The trailing <epoch> (run start, from date +%s) becomes the timestamp.
#  wall_time is not recoverable from a log, so it is left blank.
#
#  Usage:
#     ./backfill_gh200_logs.sh [logdir] [csv_path]
#  Defaults:
#     logdir   = bench_results/gh200/logs   (relative to this script)
#     csv_path = bench_results/gh200/results.csv
#  Override the interpreter with PARSE_PY if python3 isn't your parser:
#     PARSE_PY=/path/to/python ./backfill_gh200_logs.sh
# ============================================================
set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LOGDIR="${1:-${HERE}/bench_results/gh200/logs}"
CSV_PATH="${2:-${HERE}/bench_results/gh200/results.csv}"
PARSE_PY="${PARSE_PY:-python3}"

shopt -s nullglob
logs=( "${LOGDIR}"/run_*.log )
if [ ${#logs[@]} -eq 0 ]; then
  echo "No run_*.log files in ${LOGDIR}" >&2
  exit 1
fi

parsed=0 skipped=0
for f in "${logs[@]}"; do
  base="$(basename "${f}" .log)"
  run_id="${base#run_}"                 # strip leading "run_"

  # Only backfill completed runs; a crashed run has no throughput line and would
  # otherwise append a near-empty row.
  if ! grep -q "Throughput (output tokens)" "${f}"; then
    echo "  skip (no throughput): ${base}"
    skipped=$((skipped + 1))
    continue
  fi

  device="${run_id%%_*}"                # first field
  epoch="${run_id##*_}"                 # trailing date +%s
  out=""; bs=""
  [[ "${run_id}" =~ _o([0-9]+)_ ]] && out="${BASH_REMATCH[1]}"
  [[ "${run_id}" =~ _b([0-9]+)_ ]] && bs="${BASH_REMATCH[1]}"

  # epoch -> ISO-8601 UTC (GNU date on the Linux nodes; harmless if it fails).
  stamp="$(date -u -d "@${epoch}" +"%Y-%m-%dT%H:%M:%SZ" 2>/dev/null || echo "")"

  "${PARSE_PY}" "${HERE}/parse_log.py" \
    "${f}" "${device}" "${stamp}" "${out}" "${bs}" \
    "${run_id}" "" "${CSV_PATH}"
  parsed=$((parsed + 1))
done

echo "✓ backfill done: ${parsed} parsed, ${skipped} skipped → ${CSV_PATH}"
