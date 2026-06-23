#!/bin/bash
#SBATCH --job-name=bench-gaudi2-8b
#SBATCH --output=/scratch/tianche5/pearc-asu-gaudi2/llama-3.1-8T-benchmark/bench_results/gaudi2/logs/gaudi2_8b_%j.out
#SBATCH --error=/scratch/tianche5/pearc-asu-gaudi2/llama-3.1-8T-benchmark/bench_results/gaudi2/logs/gaudi2_8b_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -p gaudi
#SBATCH -G 1
#SBATCH --time=04:00:00
#SBATCH --exclusive

# ============================================================
#  Gaudi2 HPU — vLLM Llama-3.1-8B (BF16 + FP8), single card.
#  Run this FIRST (clean single-card baseline / pipeline validation).
#  NOTE: energy assumes the job sees only its own HPU (cgroup isolation);
#  add --exclusive if your gaudi nodes are not device-isolated.
# ============================================================
set -eo pipefail

DEVICE="gaudi2"
BENCH_ROOT="/scratch/tianche5/pearc-asu-gaudi2/llama-3.1-8T-benchmark"
DEVDIR="${BENCH_ROOT}/bench_results/${DEVICE}"
LOGDIR="${DEVDIR}/logs"
RESULTS_CSV="${DEVDIR}/results.csv"
mkdir -p "${LOGDIR}"

# 8B fits one card at both precisions. FP8 via INC on the base checkpoint.
RUNS=(
  "llama8b:NousResearch/Meta-Llama-3.1-8B-Instruct:bf16:1"
  "llama8b:NousResearch/Meta-Llama-3.1-8B-Instruct:fp8:1"
)
IN_LEN=512
OUT_LIST=(256)
BATCH_LIST=(1 8 32 64 128)     # 8B needs a high ceiling to reach saturation
REPEATS=3
WARMUP_RUNS=1

RUN_BACKEND=apptainer
RUN_BACKEND="${RUN_BACKEND:-mamba}"
MAMBA_ENV="${MAMBA_ENV:-gaudi}"
APPTAINER_SIF="${APPTAINER_SIF:-/packages/apps/simg/vllm-gaudi-1.24.0-1007.sif}"
DRIVER="${BENCH_ROOT}/run_vllm_gaudi.py"

export HF_HOME="/scratch/tianche5/huggingface"
[ -f "${HF_HOME}/token" ] && export HUGGINGFACE_HUB_TOKEN="$(< "${HF_HOME}/token")"
export PT_HPU_LAZY_MODE=1
# FP8 (INC): calibrated maxabs config (see README → calibration).
export QUANT_CONFIG_FP8="${BENCH_ROOT}/quantization_config/maxabs_quant.json"

if [ "${RUN_BACKEND}" = "mamba" ]; then
  module purge; ml mamba; source activate "${MAMBA_ENV}"
fi

launch_one () {
  if [ "${RUN_BACKEND}" = "apptainer" ]; then
    # Bind whatever HPU device nodes this node actually exposes. SynapseAI 1.24
    # uses the kernel accel subsystem (/dev/accel/*); older stacks use /dev/hl*.
    # HABANA_VISIBLE_MODULES (set per-run) still selects which card vLLM uses.
    local dev_binds=() d
    [ -d /dev/accel ] && dev_binds+=( --bind /dev/accel )
    for d in /dev/hl[0-9]* /dev/hl_controlD*; do
      [ -e "$d" ] && dev_binds+=( --bind "$d:$d" )
    done
    apptainer exec \
      "${dev_binds[@]}" \
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
echo "✓ Gaudi2 8B job ${SLURM_JOB_ID} done.  CSV → ${RESULTS_CSV}"
