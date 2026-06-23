#!/bin/bash
#SBATCH --job-name=bench-gaudi2ohf-8b
#SBATCH --output=/scratch/tianche5/pearc-asu-gaudi2/llama-3.1-8T-benchmark/bench_results/gaudi2-ohf/logs/gaudi2ohf_8b_%j.out
#SBATCH --error=/scratch/tianche5/pearc-asu-gaudi2/llama-3.1-8T-benchmark/bench_results/gaudi2-ohf/logs/gaudi2ohf_8b_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -p gaudi
#SBATCH -G 1
#SBATCH --time=04:00:00
#SBATCH --exclusive

# ============================================================
#  Gaudi2 HPU — Optimum-Habana Llama-3.1-8B (BF16 + FP8), single card.
#  This is the CLASSIC lazy-mode + HPU-graphs path (the Feb baseline), kept as a
#  SEPARATE series (device label "gaudi2-ohf") so it can be overlaid against the
#  vLLM-Gaudi (torch.compile) runs in plot_results_ohf.py.
#  Same workload shape as the vLLM sweep (in=512, out=256, batch 1..128) so the
#  points line up for a direct lazy-vs-compile comparison.
# ============================================================
set -eo pipefail

DEVICE="gaudi2-ohf"
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
WARMUP_RUNS=2                  # internal optimum-habana warmup per measured child

MAMBA_ENV="${MAMBA_ENV:-gaudi-1.24.0}"
DRIVER="${BENCH_ROOT}/run_optimum_gaudi.py"

# Optimum-Habana text-generation example (has run_generation.py + gaudi_spawn.py
# one level up). Override if your checkout lives elsewhere.
export OHF_TEXTGEN_DIR="${OHF_TEXTGEN_DIR:-/scratch/tianche5/gaudi2/optimum-habana/examples/text-generation}"

export HF_HOME="/scratch/tianche5/huggingface"
[ -f "${HF_HOME}/token" ] && export HUGGINGFACE_HUB_TOKEN="$(< "${HF_HOME}/token")"
# Env `gaudi-1.24.0` has the SynapseAI-1.24-matched Habana torch (lazy still works,
# verified). The driver requests lazy mode + HPU graphs via run_generation.py flags.
export PT_HPU_LAZY_MODE=1
# FP8 (INC): calibrated maxabs config. Requires a prior measurement pass that
# wrote hqt_output/ under OHF_TEXTGEN_DIR (see README → "FP8 calibration").
export QUANT_CONFIG_FP8="${BENCH_ROOT}/quantization_config/maxabs_quant.json"

module purge; ml mamba; source activate "${MAMBA_ENV}"

launch_one () {
  python "${DRIVER}" "$@"
}

source "${BENCH_ROOT}/bench_common.sh"
run_sweep
echo "✓ Gaudi2 (OHF/lazy) 8B job ${SLURM_JOB_ID} done.  CSV → ${RESULTS_CSV}"
