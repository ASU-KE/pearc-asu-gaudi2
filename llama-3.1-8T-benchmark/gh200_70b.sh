#!/bin/bash
#SBATCH --job-name=bench-gh200-70b
#SBATCH --output=/scratch/tianche5/pearc-asu-gaudi2/llama-3.1-8T-benchmark/bench_results/gh200/logs/gh200_70b_%j.out
#SBATCH --error=/scratch/tianche5/pearc-asu-gaudi2/llama-3.1-8T-benchmark/bench_results/gh200/logs/gh200_70b_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH -p arm
#SBATCH --exclusive

# ============================================================
#  GH200 — vLLM Llama-3.1-70B (BF16 + FP8), SINGLE card (plan.txt).
#  FP8 (~70 GB) fits in 96 GB HBM cleanly. BF16 (~140 GB) does NOT fit and
#  runs only by offloading ~70 GB to Grace LPDDR over NVLink-C2C — that run is
#  recorded with cpu_offload_gb>0 and plotted as "bf16 (offload)" so it is NOT
#  read as an in-HBM result. Raise the offload if you OOM (144 GB parts: set 0).
# ============================================================
set -eo pipefail

DEVICE="gh200"
BENCH_ROOT="/scratch/tianche5/pearc-asu-gaudi2/llama-3.1-8T-benchmark"
DEVDIR="${BENCH_ROOT}/bench_results/${DEVICE}"
LOGDIR="${DEVDIR}/logs"
RESULTS_CSV="${DEVDIR}/results.csv"
mkdir -p "${LOGDIR}"

# 5th field = GB of weights to offload to Grace memory (BF16 only).
RUNS=(
  "llama70b:NousResearch/Meta-Llama-3.1-70B-Instruct:fp8:1"
  "llama70b:NousResearch/Meta-Llama-3.1-70B-Instruct:bf16:1:70"
)
IN_LEN=512
OUT_LIST=(256)
BATCH_LIST=(1 8 32)
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
echo "✓ GH200 70B job ${SLURM_JOB_ID} done.  CSV → ${RESULTS_CSV}"
