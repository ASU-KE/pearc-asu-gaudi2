#!/usr/bin/env python3
"""
plot_results_ohf.py — Comparison plots that ADD the Optimum-Habana lazy-mode +
HPU-graphs Gaudi2 series (device label `gaudi2-ohf`, the Feb baseline) on top of
the existing devices, in a distinct colour.

It reuses the plotting primitives from plot_results.py; it only:
  * registers the `gaudi2-ohf` device (purple) in the style/label tables,
  * relabels the vLLM Gaudi2 run as "Gaudi2 (compile)" so the two Gaudi2 paths
    are unambiguous on the same axes,
  * writes into a separate output directory so the main figures are never
    overwritten.

The core 3 metrics where gaudi2-ohf actually appears — throughput, peak memory
and energy/token — are stacked into a SINGLE combined PNG per model (rows =
metric, columns = output length, one shared legend on top). TTFT/ITL are dropped
here because run_generation.py reports aggregate throughput, not a prefill/decode
split (see run_optimum_gaudi.py). The bf16/fp8 speedup figures stay separate.

Usage:
    python plot_results_ohf.py [path/to/merged.csv] [out_subdir]
"""

import os
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_results as base   # noqa: E402

# New series: lazy + HPU graphs. Purple, distinct from the existing palette.
base.DEVICE_STYLE["gaudi2-ohf"] = {"color": "#9334E6", "marker": "P"}
base.DEVICE_LABEL["gaudi2-ohf"] = "Gaudi2 (lazy)"
# Disambiguate the two Gaudi2 execution modes on shared axes.
base.DEVICE_LABEL["gaudi2"] = "Gaudi2 (compile)"

# Rows of the combined figure: (metric column, y-axis label, log-y?).
COMBINED_METRICS = [
    ("throughput",       "Throughput (tokens/s)", True),
    ("mem_max_gb",       "Peak memory (GB)",      False),
    ("energy_per_token", "Energy per token (J)",  True),
]


def combined_panels(df, model, fname, outdir):
    """One PNG per model: rows = metric, columns = output length, sharing a
    single legend on top. Per-panel titles only (no figure suptitle)."""
    sub = df[df.model_label == model]
    # Column axis (output lengths) and series (device×precision) are taken from
    # throughput — the metric with the widest device coverage.
    g_ref = base.agg(sub, "throughput")
    if g_ref.empty:
        return
    seqs = sorted(g_ref.seq_len.dropna().unique())
    offp = base.offload_pairs(sub)
    nrows, ncols = len(COMBINED_METRICS), len(seqs)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(max(5.5, 4.8 * ncols), 3.6 * nrows),
                             sharey="row", squeeze=False)

    handles, seen_labels = [], set()       # dedup legend across every panel
    for r, (metric, ylabel, logy) in enumerate(COMBINED_METRICS):
        g = base.agg(sub, metric)
        pairs = base._dev_prec_pairs(g)
        for c, seq in enumerate(seqs):
            ax = axes[r][c]
            for dev, prec in pairs:
                s = g[(g.device == dev) & (g.precision == prec) &
                      (g.seq_len == seq)].sort_values("batch_size")
                if s.empty:
                    continue
                lbl = base._label(dev, prec, offp)
                line = ax.errorbar(s.batch_size, s["mean"], yerr=s["err"],
                                   color=base.DEVICE_STYLE[dev]["color"],
                                   marker=base.DEVICE_STYLE[dev]["marker"],
                                   linestyle=base.PRECISION_STYLE.get(prec, "-"),
                                   linewidth=2, markersize=6, capsize=3,
                                   label=lbl)
                if lbl not in seen_labels:
                    seen_labels.add(lbl)
                    handles.append((lbl, line))
            # Per-panel title only on the top row (column = output length).
            if r == 0:
                ax.set_title(f"output len = {int(seq)}", fontweight="bold")
            # Metric name on the leftmost column only.
            ax.set_ylabel(ylabel if c == 0 else "")
            # x-axis label only on the bottom row.
            ax.set_xlabel("Batch size (concurrency)" if r == nrows - 1 else "")
            ax.set_xscale("log", base=2)
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            if logy:
                ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.3)

    # Single legend, centred above the whole grid.
    labels = [l for l, _ in handles]
    lines = [h for _, h in handles]
    ncol = min(len(labels), 4)
    fig.legend(lines, labels, loc="upper center", ncol=ncol, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = os.path.join(outdir, fname)
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"✓ {out}")


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "bench_results/merged.csv"
    out_subdir = sys.argv[2] if len(sys.argv) > 2 else "ohf_plots"
    df = base.load(csv_path)

    outdir = os.path.join(os.path.dirname(csv_path) or ".", out_subdir)
    os.makedirs(outdir, exist_ok=True)

    for model in [m for m in ["8B", "70B"] if (df.model_label == m).any()]:
        combined_panels(df, model, f"combined_{model}.png", outdir)
        # Prefer the lazy-mode Gaudi2 run as the speedup baseline; fall back to
        # the compile run when no lazy-mode data is present.
        has_lazy = ((df.model_label == model) & (df.device == "gaudi2-ohf")).any()
        baseline = "gaudi2-ohf" if has_lazy else "gaudi2"
        for prec in ("bf16", "fp8"):
            base.speedup_panels(df, model, prec, outdir, baseline=baseline)

    print(f"All OHF-overlay plots saved → {outdir}")


if __name__ == "__main__":
    main()
