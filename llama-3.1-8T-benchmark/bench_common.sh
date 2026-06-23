#!/bin/bash
# ============================================================
#  bench_common.sh — shared benchmark sweep, sourced by each *_8b.sh / *_70b.sh.
#
#  One vLLM process per (model, precision, output_len, batch_size) config:
#  the Python driver does its own in-process warmup + repeats and prints the
#  median metrics, so the shell does NOT spawn a fresh process per repeat
#  (that was the old design whose cross-process warmup never warmed anything).
#
#  The caller (a *_8b.sh / *_70b.sh) must define BEFORE sourcing:
#     DEVICE         e.g. "h100"
#     BENCH_ROOT     dir holding the run_vllm_*.py + parse_log.py
#     RESULTS_CSV    output CSV path
#     LOGDIR         dir for per-run logs
#     RUNS           array of "label:hf_id:precision:tp[:cpu_offload_gb]" entries —
#                    each device declares exactly the (model, precision,
#                    tensor-parallel) combos it can actually run (tp depends on
#                    model+precision+VRAM). The optional 5th field offloads that
#                    many GB of weights to host memory (GH200 single-card 70B).
#     IN_LEN         fixed prompt length (tokens)
#     OUT_LIST       array of output lengths   (recorded as seq_len)
#     BATCH_LIST     array of concurrencies    (recorded as batch_size)
#     REPEATS        measured iters per config (driver --num-iters)
#     WARMUP_RUNS    warmup iters per config   (driver --warmup-iters)
#     launch_one()   function that runs the driver with "$@" (mamba or apptainer)
# ============================================================

run_sweep () {
  local entry mlabel mid mtp prec out bs run_id logfile stamp wall_start wall_time
  local offload extra

  # Capture the full set of accelerators Slurm exposed (with --exclusive this is
  # every GPU/HPU on the node) so each run can be restricted to just the first
  # $tp of them — vLLM and the power sampler then use/measure exactly the
  # intended devices while --exclusive still keeps co-tenant jobs off the node.
  local slurm_cvd="${CUDA_VISIBLE_DEVICES:-}"
  local slurm_hvm="${HABANA_VISIBLE_MODULES:-}"

  # huggingface_hub's get_token() auto-reads HF_TOKEN, NOT HUGGINGFACE_HUB_TOKEN,
  # so a gated download runs anonymously (401) unless HF_TOKEN is set. Mirror the
  # token to HF_TOKEN for the mamba path and inject it into every apptainer
  # container via APPTAINERENV_/SINGULARITYENV_ (works regardless of --env flags).
  if [ -n "${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}" ]; then
    export HF_TOKEN="${HF_TOKEN:-$HUGGINGFACE_HUB_TOKEN}"
    export APPTAINERENV_HF_TOKEN="${HF_TOKEN}"
    export SINGULARITYENV_HF_TOKEN="${HF_TOKEN}"
  fi

  # parse_log.py owns the CSV header (writes it on the first append).
  for entry in "${RUNS[@]}"; do
    IFS=':' read -r mlabel mid prec mtp offload <<< "${entry}"
    extra=()
    if [ -n "${offload}" ] && [ "${offload}" != "0" ]; then
      extra+=(--cpu-offload-gb "${offload}")    # CUDA driver only (GH200)
    fi

    # restrict this run to exactly $mtp devices (first $mtp of the node)
    if [ -n "${slurm_cvd}" ]; then
      export CUDA_VISIBLE_DEVICES="$(printf '%s' "${slurm_cvd}" | tr ',' '\n' | head -n "${mtp}" | paste -sd, -)"
    fi
    if [ -n "${slurm_hvm}" ]; then
      export HABANA_VISIBLE_MODULES="$(printf '%s' "${slurm_hvm}" | tr ',' '\n' | head -n "${mtp}" | paste -sd, -)"
    fi
    for out in "${OUT_LIST[@]}"; do
      for bs in "${BATCH_LIST[@]}"; do
        stamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
        run_id="${DEVICE}_${mlabel}_${prec}_o${out}_b${bs}_$(date +%s)"
        logfile="${LOGDIR}/run_${run_id}.log"
        echo "▶ ${DEVICE}  ${mlabel}  ${prec}  out=${out}  bs=${bs}  (tp=${mtp}${offload:+ offload=${offload}G})"

        wall_start=${SECONDS}
        launch_one \
          --model "${mid}" \
          --precision "${prec}" \
          --input-len "${IN_LEN}" \
          --output-len "${out}" \
          --batch-size "${bs}" \
          --num-iters "${REPEATS}" \
          --warmup-iters "${WARMUP_RUNS}" \
          --tensor-parallel-size "${mtp}" \
          --device-label "${DEVICE}" \
          "${extra[@]}" \
          > "${logfile}" 2>&1 || echo "  ⚠ run failed – see ${logfile}"
        wall_time=$(( SECONDS - wall_start ))

        python "${BENCH_ROOT}/parse_log.py" \
          "${logfile}" "${DEVICE}" "${stamp}" "${out}" "${bs}" \
          "${run_id}" "${wall_time}" "${RESULTS_CSV}" \
          || echo "  ⚠ parse failed for ${run_id}"
      done
    done
  done

  echo "✓ ${DEVICE} sweep done.  CSV → ${RESULTS_CSV}"
}
