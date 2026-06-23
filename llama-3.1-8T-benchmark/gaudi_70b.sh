#!/bin/bash
#SBATCH --job-name=bench-gaudi2-70b
#SBATCH --output=/scratch/tianche5/pearc-asu-gaudi2/llama-3.1-8T-benchmark/bench_results/gaudi2/logs/gaudi2_70b_%j.out
#SBATCH --error=/scratch/tianche5/pearc-asu-gaudi2/llama-3.1-8T-benchmark/bench_results/gaudi2/logs/gaudi2_70b_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH -p gaudi
#SBATCH -G 2
#SBATCH --time=08:00:00

# ============================================================
#  Gaudi2 HPU — vLLM Llama-3.1-70B (BF16 + FP8).
#  70B BF16 (~140 GB) needs tp=2 across two 96 GB cards; FP8 (~70 GB)
#  runs on a single card (the single-card headline).
# ============================================================
set -eo pipefail

DEVICE="gaudi2"
BENCH_ROOT="/scratch/tianche5/pearc-asu-gaudi2/llama-3.1-8T-benchmark"
DEVDIR="${BENCH_ROOT}/bench_results/${DEVICE}"
LOGDIR="${DEVDIR}/logs"
RESULTS_CSV="${DEVDIR}/results.csv"
mkdir -p "${LOGDIR}"

RUNS=(
  "llama70b:meta-llama/Llama-3.1-70B-Instruct:bf16:2"
  "llama70b:meta-llama/Llama-3.1-70B-Instruct:fp8:1"
)
IN_LEN=512
OUT_LIST=(256)
BATCH_LIST=(1 8 32)            # add 64 if you want one more load point
REPEATS=3
WARMUP_RUNS=1

RUN_BACKEND="${RUN_BACKEND:-mamba}"
MAMBA_ENV="${MAMBA_ENV:-gaudi}"
APPTAINER_SIF="${APPTAINER_SIF:-/scratch/tianche5/sif/vllm-gaudi.sif}"
DRIVER="${BENCH_ROOT}/run_vllm_gaudi.py"

export HF_HOME="/scratch/tianche5/huggingface"
[ -f "${HF_HOME}/token" ] && export HUGGINGFACE_HUB_TOKEN="$(< "${HF_HOME}/token")"
export PT_HPU_LAZY_MODE=1
export QUANT_CONFIG_FP8="${BENCH_ROOT}/quantization_config/maxabs_quant.json"

if [ "${RUN_BACKEND}" = "mamba" ]; then
  module purge; ml mamba; source activate "${MAMBA_ENV}"
fi

launch_one () {
  if [ "${RUN_BACKEND}" = "apptainer" ]; then
    apptainer exec \
      --bind /dev/hl0:/dev/hl0 \
      --bind /dev/hl1:/dev/hl1 \
      --bind /usr/lib/habanalabs:/usr/lib/habanalabs \
      --bind /sys:/sys \
      --env HF_HOME="${HF_HOME}" \
      --env HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN}" \
      --env PT_HPU_LAZY_MODE="${PT_HPU_LAZY_MODE}" \
      --env QUANT_CONFIG_FP8="${QUANT_CONFIG_FP8}" \
      --env HABANA_VISIBLE_MODULES="${HABANA_VISIBLE_MODULES}" \
      "${APPTAINER_SIF}" python3 "${DRIVER}" "$@"
  else
    python "${DRIVER}" "$@"
  fi
}

source "${BENCH_ROOT}/bench_common.sh"
run_sweep
echo "✓ Gaudi2 70B job ${SLURM_JOB_ID} done.  CSV → ${RESULTS_CSV}"
