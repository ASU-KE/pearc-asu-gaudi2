#!/usr/bin/env python3
"""
plot_results_combined.py — All metrics in ONE multi-panel figure (per model),
with every device on every panel. Replaces both the per-output-length panel split
of plot_results.py and the one-figure-per-metric layout.

Why this exists: the gaudi2-ohf (lazy) baseline was swept at output lengths
128/512/2048, while every vLLM device was swept only at 256. plot_results.py draws
one panel per output length, which orphans each series into its own panel and
builds the legend from the first panel only, so most devices never appear in it.
New 256-length lazy data isn't available, so instead of aligning panels we collapse
them: one curve per (device, precision, output-length) group vs batch size, and we
lay the five metrics out as titled sub-panels of a single figure that shares one
legend.

Caveat the reader must keep in mind: curves at different output lengths are NOT
apples-to-apples (longer outputs amortise prefill differently), so the marker
encodes the output length to keep that explicit. This is a "see everything on one
chart" view, not a strict same-workload comparison.

Encoding:  color = device,  linestyle = precision (bf16 solid / fp8 dashed),
           marker = output length.

Usage:
    python plot_results_combined.py [merged.csv] [out_subdir]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt          # noqa: E402
import matplotlib.ticker as ticker       # noqa: E402
import plot_results as base              # noqa: E402

# Gaudi is red. The lazy series is the only Gaudi path present, so give it the
# Google-red the paper uses for Gaudi2; keep the label disambiguated.
base.DEVICE_STYLE["gaudi2-ohf"] = {"color": "#EA4335", "marker": "P"}
base.DEVICE_LABEL["gaudi2-ohf"] = "Gaudi2 (lazy)"
base.DEVICE_LABEL["gaudi2"] = "Gaudi2 (compile)"

# Output length → marker. Device colour still identifies the card; the marker
# tells the three same-coloured lazy curves apart and flags the 256-only cards.
SEQ_MARKER = {128: "v", 256: "o", 512: "s", 1024: "X", 2048: "D"}

# (column, panel title, log-y?) — title names the metric, so no per-axis ylabel.
METRICS = [
    ("throughput", "Throughput (tokens/s)", True),
    ("first_token_ms", "Time to first token (ms)", True),
    ("rest_token_ms", "Inter-token latency / TPOT (ms)", True),
    ("mem_max_gb", "Peak memory (GB)", False),
    ("energy_per_token", "Energy per token (J)", True),
]


def _draw(ax, g, offp, logy, title):
    for dev, prec in base._dev_prec_pairs(g):
        sd = g[(g.device == dev) & (g.precision == prec)]
        for seq in sorted(sd.seq_len.dropna().unique()):
            s = sd[sd.seq_len == seq].sort_values("batch_size")
            if s.empty:
                continue
            tag = " (offload)" if (dev, prec) in offp else ""
            ax.errorbar(s.batch_size, s["mean"], yerr=s["err"],
                        color=base.DEVICE_STYLE[dev]["color"],
                        marker=SEQ_MARKER.get(int(seq), "o"),
                        linestyle=base.PRECISION_STYLE.get(prec, "-"),
                        linewidth=2, markersize=6, capsize=3,
                        label=f"{base.DEVICE_LABEL[dev]} {prec} · out={int(seq)}{tag}")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Batch size (concurrency)")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    if logy:
        ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)


def combined_figure(df, model, fname, outdir):
    sub = df[df.model_label == model]
    if sub.empty:
        return
    offp = base.offload_pairs(sub)

    ncols = 3
    nrows = -(-len(METRICS) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.7, nrows * 4.1),
                             squeeze=False)
    axes = axes.flatten()

    # One shared legend: collect handles across every panel and dedupe by label,
    # so a series present in only some metrics still shows up exactly once.
    by_label = {}
    for ax, (metric, title, logy) in zip(axes, METRICS):
        g = base.agg(sub, metric)
        if g.empty:
            ax.set_title(title, fontweight="bold")
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="grey")
            continue
        _draw(ax, g, offp, logy, title)
        h, l = ax.get_legend_handles_labels()
        by_label.update(dict(zip(l, h)))

    for ax in axes[len(METRICS):]:        # hide unused grid cells
        ax.set_visible(False)

    labels = list(by_label)
    ncol = min(4, max(1, len(labels)))
    nrows_leg = -(-len(labels) // ncol)
    fig.legend(by_label.values(), labels, loc="lower center", ncol=ncol,
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0))
    # No suptitle by request. Reserve bottom space for the shared legend.
    fig.tight_layout(rect=(0, 0.04 + 0.028 * nrows_leg, 1, 1))

    out = os.path.join(outdir, fname)
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"✓ {out}")


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "bench_results/merged.csv"
    out_subdir = sys.argv[2] if len(sys.argv) > 2 else "combined_plots"
    df = base.load(csv_path)

    outdir = os.path.join(os.path.dirname(csv_path) or ".", out_subdir)
    os.makedirs(outdir, exist_ok=True)

    for model in [m for m in ["8B", "70B"] if (df.model_label == m).any()]:
        combined_figure(df, model, f"combined_{model}.png", outdir)

    print(f"All combined plots saved → {outdir}")


if __name__ == "__main__":
    main()
