#!/bin/bash
#SBATCH --job-name=bench-a100
#SBATCH --output=/scratch/username/gaudi_bench/bench_results/a100/logs/a100_%j.out
#SBATCH --error=/scratch/username/gaudi_bench/bench_results/a100/logs/a100_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH -G a100:1
#SBATCH --time=01:00:00
#SBATCH -p htc

# ============================================================
#  A100 GPU  —  Llama-3-8B BF16 benchmark
# ============================================================
set -eo pipefail

DEVICE="a100"
BENCH_ROOT="/scratch/username/gaudi_bench"
OUTROOT="${BENCH_ROOT}/bench_results"
DEVDIR="${OUTROOT}/${DEVICE}"
LOGDIR="${DEVDIR}/logs"
mkdir -p "${LOGDIR}"

MODEL="meta-llama/Meta-Llama-3-8B"
SEQ_LIST=(128 512 2048)
BATCH_LIST=(1 8 32 128)
REPEATS=3
WARMUP_RUNS=2

# ---- environment -----------------------------------------------------------
module purge
ml mamba
source activate /scratch/username/gaudi_bench/env/a100

export HF_HOME="/scratch/username/huggingface"
export CUDA_VISIBLE_DEVICES=0
if [ -f "${HF_HOME}/token" ]; then
  export HUGGINGFACE_HUB_TOKEN="$(< "${HF_HOME}/token")"
fi

# ---- CSV header (written once) ---------------------------------------------
RESULTS_CSV="${DEVDIR}/results.csv"
if [ ! -f "${RESULTS_CSV}" ]; then
  echo "device,timestamp,seq_len,batch_size,run_id,slurm_jobid,wall_time_s,throughput,first_token_ms,rest_token_ms,end2end_ms,mem_alloc_gb,mem_max_gb,total_mem_gb,graph_compile_s,logfile" \
    > "${RESULTS_CSV}"
fi

# ---- warmup + measured loops ------------------------------------------------
for seq in "${SEQ_LIST[@]}"; do
  for bs in "${BATCH_LIST[@]}"; do

    # --- warmup (not recorded) ---
    for w in $(seq 1 $WARMUP_RUNS); do
      echo "  warmup  seq=${seq} bs=${bs} [${w}/${WARMUP_RUNS}]"
      python "${BENCH_ROOT}/cuda_benchmark.py" \
        --model_name_or_path "$MODEL" --bf16 --trust_remote_code \
        --max_new_tokens "$seq" --batch_size "$bs" \
        > "${LOGDIR}/warm_s${seq}_b${bs}_w${w}.log" 2>&1 || true
      sleep 1
    done

    # --- measured runs ---
    for run_idx in $(seq 1 $REPEATS); do
      stamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
      run_id="${DEVICE}_s${seq}_b${bs}_r${run_idx}_$(date +%s)"
      logfile="${LOGDIR}/run_${run_id}.log"
      echo "▶ measured  seq=${seq} bs=${bs} run=${run_idx}"

      wall_start=$SECONDS
      python "${BENCH_ROOT}/cuda_benchmark.py" \
        --model_name_or_path "$MODEL" --bf16 --trust_remote_code \
        --max_new_tokens "$seq" --batch_size "$bs" \
        > "$logfile" 2>&1 || echo "⚠ run failed – see $logfile"
      wall_time=$(( SECONDS - wall_start ))

      python "${BENCH_ROOT}/parse_log.py" \
        "$logfile" "$DEVICE" "$stamp" "$seq" "$bs" "$run_id" "$wall_time" "$RESULTS_CSV" \
        || echo "⚠ parse failed for $run_id"

      sleep 1
    done
  done
done

echo "✓ A100 job ${SLURM_JOB_ID} done.  CSV → ${RESULTS_CSV}"
