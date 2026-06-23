#!/bin/bash
#SBATCH --job-name=bench-gaudi2ohf-70b
#SBATCH --output=/scratch/tianche5/pearc-asu-gaudi2/llama-3.1-8T-benchmark/bench_results/gaudi2-ohf/logs/gaudi2ohf_70b_%j.out
#SBATCH --error=/scratch/tianche5/pearc-asu-gaudi2/llama-3.1-8T-benchmark/bench_results/gaudi2-ohf/logs/gaudi2ohf_70b_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH -p gaudi
#SBATCH -G 2
#SBATCH --time=08:00:00

# ============================================================
#  Gaudi2 HPU — Optimum-Habana Llama-3.1-70B (BF16 + FP8).
#  Classic lazy-mode + HPU-graphs path (Feb baseline), series "gaudi2-ohf".
#  70B BF16 (~140 GB) needs tp=2 across two 96 GB cards (routed through
#  gaudi_spawn.py + DeepSpeed by the driver); FP8 (~70 GB) fits one card.
# ============================================================
set -eo pipefail

DEVICE="gaudi2-ohf"
BENCH_ROOT="/scratch/tianche5/pearc-asu-gaudi2/llama-3.1-8T-benchmark"
DEVDIR="${BENCH_ROOT}/bench_results/${DEVICE}"
LOGDIR="${DEVDIR}/logs"
RESULTS_CSV="${DEVDIR}/results.csv"
mkdir -p "${LOGDIR}"

RUNS=(
  "llama70b:NousResearch/Meta-Llama-3.1-70B-Instruct:bf16:2"
  "llama70b:NousResearch/Meta-Llama-3.1-70B-Instruct:fp8:1"
)
IN_LEN=512
OUT_LIST=(256)
BATCH_LIST=(1 8 32)            # add 64 if you want one more load point
REPEATS=3
WARMUP_RUNS=2                  # internal optimum-habana warmup per measured child

MAMBA_ENV="${MAMBA_ENV:-gaudi-1.24.0}"
DRIVER="${BENCH_ROOT}/run_optimum_gaudi.py"

export OHF_TEXTGEN_DIR="${OHF_TEXTGEN_DIR:-/scratch/tianche5/gaudi2/optimum-habana/examples/text-generation}"
# gaudi_spawn.py (DeepSpeed launcher) for the tp=2 BF16 run; defaults to one
# level up from OHF_TEXTGEN_DIR. Override if your layout differs.
export OHF_SPAWN="${OHF_SPAWN:-/scratch/tianche5/gaudi2/optimum-habana/examples/gaudi_spawn.py}"

export HF_HOME="/scratch/tianche5/huggingface"
[ -f "${HF_HOME}/token" ] && export HUGGINGFACE_HUB_TOKEN="$(< "${HF_HOME}/token")"
export PT_HPU_LAZY_MODE=1
export QUANT_CONFIG_FP8="${BENCH_ROOT}/quantization_config/maxabs_quant.json"

module purge; ml mamba; source activate "${MAMBA_ENV}"

launch_one () {
  python "${DRIVER}" "$@"
}

source "${BENCH_ROOT}/bench_common.sh"
run_sweep
echo "✓ Gaudi2 (OHF/lazy) 70B job ${SLURM_JOB_ID} done.  CSV → ${RESULTS_CSV}"
