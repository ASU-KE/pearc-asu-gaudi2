#!/usr/bin/env python3
"""
plot_results_ohf.py — Comparison plots that ADD the Optimum-Habana lazy-mode +
HPU-graphs Gaudi2 series (device label `gaudi2-ohf`, the Feb baseline) on top of
the existing devices, in a distinct colour.

It reuses every plotting routine from plot_results.py unchanged; it only:
  * registers the `gaudi2-ohf` device (purple) in the style/label tables,
  * relabels the vLLM Gaudi2 run as "Gaudi2 (compile)" so the two Gaudi2 paths
    are unambiguous on the same axes,
  * writes into a separate output directory so the main figures are never
    overwritten.

The throughput / memory / energy / speedup panels gain the lazy series; the
TTFT and ITL panels simply omit it (run_generation.py reports aggregate
throughput, not a prefill/decode split — see run_optimum_gaudi.py).

Usage:
    python plot_results_ohf.py [path/to/merged.csv] [out_subdir]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_results as base   # noqa: E402

# New series: lazy + HPU graphs. Purple, distinct from the existing palette.
base.DEVICE_STYLE["gaudi2-ohf"] = {"color": "#9334E6", "marker": "P"}
base.DEVICE_LABEL["gaudi2-ohf"] = "Gaudi2 (lazy)"
# Disambiguate the two Gaudi2 execution modes on shared axes.
base.DEVICE_LABEL["gaudi2"] = "Gaudi2 (compile)"


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "bench_results/merged.csv"
    out_subdir = sys.argv[2] if len(sys.argv) > 2 else "ohf_plots"
    df = base.load(csv_path)

    outdir = os.path.join(os.path.dirname(csv_path) or ".", out_subdir)
    os.makedirs(outdir, exist_ok=True)

    for model in [m for m in ["8B", "70B"] if (df.model_label == m).any()]:
        base.metric_panels(df, model, "throughput", "Throughput (tokens/s)",
                           f"throughput_{model}.pdf", outdir, logy=True)
        base.metric_panels(df, model, "first_token_ms", "Time to first token (ms)",
                           f"ttft_{model}.pdf", outdir, logy=True)
        base.metric_panels(df, model, "rest_token_ms",
                           "Inter-token latency / TPOT (ms)",
                           f"itl_{model}.pdf", outdir, logy=True)
        base.metric_panels(df, model, "mem_max_gb", "Peak memory (GB)",
                           f"memory_{model}.pdf", outdir, logy=False)
        base.metric_panels(df, model, "energy_per_token", "Energy per token (J)",
                           f"energy_{model}.pdf", outdir, logy=True)
        for prec in ("bf16", "fp8"):
            base.speedup_panels(df, model, prec, outdir)

    print(f"All OHF-overlay plots saved → {outdir}")


if __name__ == "__main__":
    main()
