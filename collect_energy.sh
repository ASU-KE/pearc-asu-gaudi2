#!/usr/bin/env bash
# collect_energy.sh
# Query sacct for a single job and annotate merged.csv for all device-name variants
# Usage: bash collect_energy.sh <JOBID> <device_alias>
# Example: bash collect_energy.sh 46852369 a100

set -eo pipefail

# Path to merged CSV (adjust if different)
BENCH_ROOT="/scratch/username/gaudi_bench"
MERGED="${BENCH_ROOT}/bench_results/merged.csv"

if [ ! -f "${MERGED}" ]; then
  echo "ERROR: merged CSV not found at ${MERGED}. Run merge_results.sh first."
  exit 1
fi

if [ $# -lt 2 ]; then
  echo "Usage: bash collect_energy.sh <JOBID> <device_alias>"
  echo "Example: bash collect_energy.sh 46852369 a100"
  exit 1
fi

JOBID="$1"
ALIAS_RAW="$2"
ALIAS="$(echo "$ALIAS_RAW" | tr '[:upper:]' '[:lower:]')"

echo "Querying sacct for energy data for job ${JOBID}..."

# Query sacct: produce JobID|ConsumedEnergyRaw|ElapsedRaw
SACCT_LINE=$(sacct -j "${JOBID}" -n -P --format=JobID,ConsumedEnergyRaw,ElapsedRaw 2>/dev/null | grep "^${JOBID}|" | head -1 || true)

if [ -z "${SACCT_LINE}" ]; then
  echo "WARNING: sacct returned no energy/elapsed info for job ${JOBID}."
  echo "Check if energy accounting is enabled or the job ID is correct."
  exit 0
fi

ENERGY_J=$(echo "${SACCT_LINE}" | cut -d'|' -f2)
ELAPSED_S=$(echo "${SACCT_LINE}" | cut -d'|' -f3)

# safe numeric parsing
if [[ "${ENERGY_J}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  ENERGY_J_NUM="${ENERGY_J}"
else
  ENERGY_J_NUM="0"
fi

if [[ "${ELAPSED_S}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  ELAPSED_S_NUM="${ELAPSED_S}"
else
  ELAPSED_S_NUM="0"
fi

AVG_W=""
if [ "$(echo "${ELAPSED_S_NUM} > 0" | bc -l)" -eq 1 2>/dev/null; then
  AVG_W=$(awk -v e="${ENERGY_J_NUM}" -v t="${ELAPSED_S_NUM}" 'BEGIN{printf("%.2f", (t>0)? e/t : 0)}')
fi

echo "  sacct: ${ENERGY_J_NUM} J over ${ELAPSED_S_NUM}s (avg ${AVG_W:-N/A} W)"

# call python to update merged.csv
python3 - "${MERGED}" "${JOBID}" "${ENERGY_J_NUM}" "${ELAPSED_S_NUM}" "${AVG_W}" "${ALIAS}" <<'PYEOF'
import sys, csv, math

merged_path = sys.argv[1]
jobid = sys.argv[2]
try:
    energy_j_total = float(sys.argv[3])
except:
    energy_j_total = 0.0
try:
    elapsed_s_job = float(sys.argv[4])
except:
    elapsed_s_job = 0.0
avg_power_w = sys.argv[5] if sys.argv[5] != "None" else ""
alias = sys.argv[6].strip().lower()

# read merged CSV
with open(merged_path, newline='') as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    header = reader.fieldnames[:] if reader.fieldnames else []

# find all distinct device strings present
devices_present = sorted({ (r.get('device') or '').strip() for r in rows if (r.get('device') or '').strip() })

# select matched devices: substring match (case-insensitive) either way
matched_devices = []
for d in devices_present:
    d_low = d.strip().lower()
    if not d_low:
        continue
    if alias in d_low or d_low in alias:
        matched_devices.append(d)

if not matched_devices:
    print(f"WARNING: No device names in {merged_path} matched alias '{alias}'. No changes made.")
    sys.exit(0)

print("Matched device names in merged.csv:", matched_devices)

# ensure energy columns exist: energy_jobid, avg_power_w, energy_j_est, energy_per_token
extra_cols = ["energy_jobid", "avg_power_w", "energy_j_est", "energy_per_token"]
for c in extra_cols:
    if c not in header:
        header.append(c)

# compute total wall_time for *all matched devices*
total_wall = 0.0
for r in rows:
    dev = (r.get('device') or '').strip()
    if dev in matched_devices:
        try:
            total_wall += float(r.get('wall_time_s') or 0.0)
        except:
            total_wall += 0.0

if total_wall <= 0.0:
    print("WARNING: total wall_time for matched devices is zero or missing; cannot proportionally allocate energy.")
    # Still add avg_power_w & jobid to matched rows (so the job is recorded), but leave energy_j_est/energy_per_token blank
    for r in rows:
        dev = (r.get('device') or '').strip()
        if dev in matched_devices:
            r['energy_jobid'] = jobid
            r['avg_power_w'] = avg_power_w
            r['energy_j_est'] = ""
            r['energy_per_token'] = ""
    # write back
    with open(merged_path, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print("Wrote jobid and avg_power_w to matched rows (no energy distributed).")
    sys.exit(0)

# allocate energy proportionally across matched-device rows
for r in rows:
    dev = (r.get('device') or '').strip()
    # default empty
    r.setdefault('energy_jobid', "")
    r.setdefault('avg_power_w', "")
    r.setdefault('energy_j_est', "")
    r.setdefault('energy_per_token', "")

    if dev in matched_devices:
        try:
            wall = float(r.get('wall_time_s') or 0.0)
        except:
            wall = 0.0
        share = (wall / total_wall) if total_wall > 0 and wall > 0 else 0.0
        run_energy = energy_j_total * share
        # compute tokens = throughput * wall_time (if throughput present)
        try:
            thr = float(r.get('throughput') or 0.0)
        except:
            thr = 0.0
        tokens = thr * wall if thr and wall else 0.0
        energy_per_token = (run_energy / tokens) if tokens > 0 else ""
        r['energy_jobid'] = jobid
        r['avg_power_w'] = avg_power_w
        r['energy_j_est'] = f"{run_energy:.2f}" if run_energy else "0.00"
        r['energy_per_token'] = f"{energy_per_token:.6f}" if energy_per_token != "" else ""

# write updated merged.csv (preserve header order + extras)
with open(merged_path, "w", newline='') as f:
    writer = csv.DictWriter(f, fieldnames=header, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)

print(f"✓ Distributed {energy_j_total:.0f} J for job {jobid} across devices: {matched_devices}")
PYEOF

echo "Done."

