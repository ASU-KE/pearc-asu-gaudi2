#!/bin/bash
#SBATCH --job-name=bench-gh200-8b
#SBATCH --output=/scratch/tianche5/pearc-asu-gaudi2/llama-3.1-8T-benchmark/bench_results/gh200/logs/gh200_8b_%j.out
#SBATCH --error=/scratch/tianche5/pearc-asu-gaudi2/llama-3.1-8T-benchmark/bench_results/gh200/logs/gh200_8b_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH -p arm
#SBATCH --exclusive

# ============================================================
#  GH200 — vLLM Llama-3.1-8B (BF16 + FP8), single card.
#  8B fits entirely in HBM, so no offload here — the BF16 offload caveat
#  applies only to 70B.
# ============================================================
set -eo pipefail

DEVICE="gh200"
BENCH_ROOT="/scratch/tianche5/pearc-asu-gaudi2/llama-3.1-8T-benchmark"
DEVDIR="${BENCH_ROOT}/bench_results/${DEVICE}"
LOGDIR="${DEVDIR}/logs"
RESULTS_CSV="${DEVDIR}/results.csv"
mkdir -p "${LOGDIR}"

RUNS=(
  "llama8b:NousResearch/Meta-Llama-3.1-8B-Instruct:bf16:1"
  "llama8b:NousResearch/Meta-Llama-3.1-8B-Instruct:fp8:1"
)
IN_LEN=512
OUT_LIST=(256)
BATCH_LIST=(1 8 32 64 128)
REPEATS=3
WARMUP_RUNS=1

RUN_BACKEND=apptainer
MAMBA_ENV="${MAMBA_ENV:-vllm-cuda-arm}"
APPTAINER_SIF="${APPTAINER_SIF:-/packages/aarch64/simg/vllm-26.05.post1-py3.sif}"
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
echo "✓ GH200 8B job ${SLURM_JOB_ID} done.  CSV → ${RESULTS_CSV}"
