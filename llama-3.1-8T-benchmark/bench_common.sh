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

  # parse_log.py owns the CSV header (writes it on the first append).
  for entry in "${RUNS[@]}"; do
    IFS=':' read -r mlabel mid prec mtp offload <<< "${entry}"
    extra=()
    if [ -n "${offload}" ] && [ "${offload}" != "0" ]; then
      extra+=(--cpu-offload-gb "${offload}")    # CUDA driver only (GH200)
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
