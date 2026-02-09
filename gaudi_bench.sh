#!/bin/bash
#SBATCH --job-name=bench-gaudi2
#SBATCH --output=/scratch/username/gaudi_bench/bench_results/gaudi2/logs/gaudi2_%j.out
#SBATCH --error=/scratch/username/gaudi_bench/bench_results/gaudi2/logs/gaudi2_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH -p gaudi
#SBATCH -G 1
#SBATCH --time=04:00:00

# ============================================================
#  Gaudi2 HPU  —  Llama-3-8B FP8 benchmark
# ============================================================
set -eo pipefail

DEVICE="gaudi2"
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
source activate gaudi

export HF_HOME="/scratch/username/huggingface"
if [ -f "${HF_HOME}/token" ]; then
  export HUGGINGFACE_HUB_TOKEN="$(< "${HF_HOME}/token")"
fi

# absolute paths to the optimum-habana example directory
HABANA_DIR="/home/username/gaudi2/optimum-habana/examples/text-generation"
export QUANT_CONFIG="./quantization_config/maxabs_quant.json"
export PT_HPU_LAZY_MODE=1

# ---- CSV header (written once) ---------------------------------------------
RESULTS_CSV="${DEVDIR}/results.csv"
if [ ! -f "${RESULTS_CSV}" ]; then
  echo "device,timestamp,seq_len,batch_size,run_id,slurm_jobid,wall_time_s,throughput,first_token_ms,rest_token_ms,end2end_ms,mem_alloc_gb,mem_max_gb,total_mem_gb,graph_compile_s,logfile" \
    > "${RESULTS_CSV}"
fi

# ---- cd into the habana dir so all relative paths resolve ------------------
# run_generation.py + neural_compressor expect hqt_output/, .graph_dumps/,
# and quantization_config/ relative to CWD.
cd "${HABANA_DIR}"

GEN_CMD_BASE=(
  python run_generation.py
  --model_name_or_path "$MODEL"
  --use_hpu_graphs
  --use_kv_cache
  --trim_logits
  --reuse_cache
  --bf16
  --trust_remote_code
)

# ---- warmup + measured loops ------------------------------------------------
for seq in "${SEQ_LIST[@]}"; do
  for bs in "${BATCH_LIST[@]}"; do

    # --- warmup (not recorded) ---
    for w in $(seq 1 $WARMUP_RUNS); do
      echo "  warmup  seq=${seq} bs=${bs} [${w}/${WARMUP_RUNS}]"
      "${GEN_CMD_BASE[@]}" --max_new_tokens "$seq" --batch_size "$bs" \
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
      "${GEN_CMD_BASE[@]}" --max_new_tokens "$seq" --batch_size "$bs" \
        > "$logfile" 2>&1 || echo "⚠ run failed – see $logfile"
      wall_time=$(( SECONDS - wall_start ))

      python "${BENCH_ROOT}/parse_log.py" \
        "$logfile" "$DEVICE" "$stamp" "$seq" "$bs" "$run_id" "$wall_time" "$RESULTS_CSV"

      sleep 1
    done
  done
done

echo "✓ Gaudi2 job ${SLURM_JOB_ID} done.  CSV → ${RESULTS_CSV}"
