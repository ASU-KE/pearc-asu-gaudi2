#!/bin/bash
#SBATCH --job-name=bench-a100-70b
#SBATCH --output=/scratch/tianche5/pearc-asu-gaudi2/llama-3.1-8T-benchmark/bench_results/a100/logs/a100_70b_%j.out
#SBATCH --error=/scratch/tianche5/pearc-asu-gaudi2/llama-3.1-8T-benchmark/bench_results/a100/logs/a100_70b_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH -G a100:4
#SBATCH --time=08:00:00
#SBATCH -p htc

# ============================================================
#  A100 — vLLM Llama-3.1-70B (BF16 only; no FP8 on Ampere, INT8 out of scope).
#  70B BF16 (~140 GB) runs tp=4 across four 80 GB A100s (plan.txt: 4x
#  comfortable, 2x tight).
# ============================================================
set -eo pipefail

DEVICE="a100"
BENCH_ROOT="/scratch/tianche5/pearc-asu-gaudi2/llama-3.1-8T-benchmark"
DEVDIR="${BENCH_ROOT}/bench_results/${DEVICE}"
LOGDIR="${DEVDIR}/logs"
RESULTS_CSV="${DEVDIR}/results.csv"
mkdir -p "${LOGDIR}"

RUNS=(
  "llama70b:meta-llama/Llama-3.1-70B-Instruct:bf16:4"
)
IN_LEN=512
OUT_LIST=(256)
BATCH_LIST=(1 8 32)
REPEATS=3
WARMUP_RUNS=1

RUN_BACKEND=apptainer
MAMBA_ENV="${MAMBA_ENV:-vllm-cuda}"
APPTAINER_SIF="${APPTAINER_SIF:-/packages/apps/simg/vllm-cu129-nightly-0426.sif}"
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
      "${APPTAINER_SIF}" python3 "${DRIVER}" "$@"
  else
    python "${DRIVER}" "$@"
  fi
}

source "${BENCH_ROOT}/bench_common.sh"
run_sweep
echo "✓ A100 70B job ${SLURM_JOB_ID} done.  CSV → ${RESULTS_CSV}"
