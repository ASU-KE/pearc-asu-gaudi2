#!/usr/bin/env python3
"""
plot_results_combined.py — One axes per metric with EVERY series on it, instead of
one sub-panel per output length.

Why this exists: the gaudi2-ohf (lazy) baseline was swept at output lengths
128/512/2048, while every vLLM device was swept only at 256. plot_results.py draws
one panel per output length, which (a) orphans each series into its own panel —
the lazy curves alone on the 128/512/2048 panels, the vLLM cards alone on the 256
panel — and (b) builds the legend from the first panel only, so most devices never
appear in it. New 256-length lazy data isn't available, so instead of trying to
align panels we collapse them: a single axes per metric, one curve per
(device, precision, output-length) group vs batch size.

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

# Register the lazy series + disambiguate the two Gaudi2 execution modes, exactly
# as plot_results_ohf.py does, so labels/colours match the rest of the paper.
base.DEVICE_STYLE.setdefault("gaudi2-ohf", {"color": "#9334E6", "marker": "P"})
base.DEVICE_LABEL["gaudi2-ohf"] = "Gaudi2 (lazy)"
base.DEVICE_LABEL["gaudi2"] = "Gaudi2 (compile)"

# Output length → marker. Device colour still identifies the card; the marker
# tells the three same-coloured lazy curves apart and flags the 256-only cards.
SEQ_MARKER = {128: "v", 256: "o", 512: "s", 1024: "X", 2048: "D"}


def combined_panel(df, model, metric, ylabel, fname, outdir, logy=False):
    sub = df[df.model_label == model]
    g = base.agg(sub, metric)
    if g.empty:
        return
    offp = base.offload_pairs(sub)

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
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

    ax.set_xlabel("Batch size (concurrency)")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    if logy:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.3)

    # Legend below the axes so a long device×precision×outlen list never sits on
    # top of the data; columns capped at 3 to stay legible.
    h, l = ax.get_legend_handles_labels()
    ncol = min(3, max(1, len(l)))
    fig.legend(h, l, loc="upper center", ncol=ncol, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 0.02))
    fig.tight_layout(rect=(0, 0.06 + 0.03 * (-(-len(l) // ncol)), 1, 1))

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
        combined_panel(df, model, "throughput", "Throughput (tokens/s)",
                       f"throughput_{model}.pdf", outdir, logy=True)
        combined_panel(df, model, "first_token_ms", "Time to first token (ms)",
                       f"ttft_{model}.pdf", outdir, logy=True)
        combined_panel(df, model, "rest_token_ms", "Inter-token latency / TPOT (ms)",
                       f"itl_{model}.pdf", outdir, logy=True)
        combined_panel(df, model, "mem_max_gb", "Peak memory (GB)",
                       f"memory_{model}.pdf", outdir, logy=False)
        combined_panel(df, model, "energy_per_token", "Energy per token (J)",
                       f"energy_{model}.pdf", outdir, logy=True)

    print(f"All combined plots saved → {outdir}")


if __name__ == "__main__":
    main()
