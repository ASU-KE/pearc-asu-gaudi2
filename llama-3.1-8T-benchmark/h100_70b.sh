#!/bin/bash
#SBATCH --job-name=bench-h100-70b
#SBATCH --output=/scratch/tianche5/gaudi_bench/bench_results/h100/logs/h100_70b_%j.out
#SBATCH --error=/scratch/tianche5/gaudi_bench/bench_results/h100/logs/h100_70b_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH -G h100:2
#SBATCH --time=08:00:00
#SBATCH -p general
#SBATCH -q private

# ============================================================
#  H100 — vLLM Llama-3.1-70B (BF16 + FP8).
#  tp=2 across two 80 GB H100s (plan.txt: 2x works at this short context,
#  near the memory edge — bump to -G h100:4 + tp=4 if you OOM).
# ============================================================
set -eo pipefail

DEVICE="h100"
BENCH_ROOT="/scratch/tianche5/gaudi_bench"
DEVDIR="${BENCH_ROOT}/bench_results/${DEVICE}"
LOGDIR="${DEVDIR}/logs"
RESULTS_CSV="${DEVDIR}/results.csv"
mkdir -p "${LOGDIR}"

RUNS=(
  "llama70b:meta-llama/Llama-3.1-70B-Instruct:bf16:2"
  "llama70b:meta-llama/Llama-3.1-70B-Instruct:fp8:2"
)
IN_LEN=512
OUT_LIST=(256)
BATCH_LIST=(1 8 32)
REPEATS=3
WARMUP_RUNS=1

RUN_BACKEND="${RUN_BACKEND:-mamba}"
MAMBA_ENV="${MAMBA_ENV:-vllm-cuda}"
APPTAINER_SIF="${APPTAINER_SIF:-/scratch/tianche5/sif/vllm-cuda.sif}"
DRIVER="${BENCH_ROOT}/run_vllm_cuda.py"

export HF_HOME="/scratch/tianche5/huggingface"
[ -f "${HF_HOME}/token" ] && export HUGGINGFACE_HUB_TOKEN="$(< "${HF_HOME}/token")"

if [ "${RUN_BACKEND}" = "mamba" ]; then
  module purge; ml mamba; source activate "${MAMBA_ENV}"
fi

launch_one () {
  if [ "${RUN_BACKEND}" = "apptainer" ]; then
    apptainer exec --nv \
      --env HF_HOME="${HF_HOME}" \
      --env HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN}" \
      --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
      "${APPTAINER_SIF}" python "${DRIVER}" "$@"
  else
    python "${DRIVER}" "$@"
  fi
}

source "${BENCH_ROOT}/bench_common.sh"
run_sweep
echo "✓ H100 70B job ${SLURM_JOB_ID} done.  CSV → ${RESULTS_CSV}"
