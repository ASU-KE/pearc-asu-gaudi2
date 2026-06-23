#!/usr/bin/env python3
"""
plot_results.py — Comparison plots from the merged benchmark CSV.

Addresses the Fig-2 reviewer comments:
  * Shared y-axis across the seq-length panels (sharey=True) so panels are
    directly comparable; log-y where the dynamic range spans batch sizes.
  * A consolidated single-figure "summary" per model to free page space.
  * Precision is now first-class: color = device, linestyle = precision
    (bf16 solid, fp8 dashed). That lets the reader read FP8-vs-FP8 and
    BF16-vs-BF16 off the same axes — the apples-to-apples comparison the
    reviewer asked for. Dedicated speedup figures hold precision constant.

Plots are produced per model (8B and 70B separately — never mixed on one axis).

Usage:
    python plot_results.py [path/to/merged.csv]
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# device → colour + marker;  precision → linestyle
DEVICE_STYLE = {
    "gaudi2": {"color": "#EA4335", "marker": "s"},  # Google Red
    "a100":   {"color": "#FBBC05", "marker": "^"},  # Google Yellow
    "h100":   {"color": "#34A853", "marker": "o"},  # Google Green
    "gh200":  {"color": "#4285F4", "marker": "D"},  # Google Blue (unused in 8B-only runs)
}
PRECISION_STYLE = {"bf16": "-", "fp8": "--"}
DEVICE_LABEL = {"gaudi2": "Gaudi2", "a100": "A100", "h100": "H100", "gh200": "GH200"}


def load(path):
    df = pd.read_csv(path)
    df["device"] = df["device"].astype(str).str.strip().str.lower()
    if "precision" not in df.columns:
        df["precision"] = "bf16"
    df["precision"] = df["precision"].astype(str).str.strip().str.lower()
    for c in ["seq_len", "batch_size", "throughput", "throughput_std",
              "energy_per_token", "first_token_ms", "rest_token_ms",
              "mem_max_gb", "cpu_offload_gb"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # short model label from the HF id
    mid = df.get("model", pd.Series([""] * len(df))).astype(str).str.lower()
    df["model_label"] = "model"
    df.loc[mid.str.contains("70b"), "model_label"] = "70B"
    df.loc[mid.str.contains("8b"), "model_label"] = "8B"
    return df


def agg(df, metric):
    """mean of metric per (device, precision, seq_len, batch_size), plus an
    error column (throughput_std if available, else std across rows)."""
    g = (df.dropna(subset=[metric])
           .groupby(["device", "precision", "seq_len", "batch_size"])
           .agg(mean=(metric, "mean"),
                std=(metric, "std"),
                tstd=("throughput_std", "mean") if "throughput_std" in df else (metric, "std"))
           .reset_index())
    g["err"] = g["tstd"].fillna(g["std"]).fillna(0.0) if metric == "throughput" \
        else g["std"].fillna(0.0)
    return g


def offload_pairs(sub):
    """(device, precision) combos that ran with CPU offload — labelled distinctly
    so GH200's offloaded BF16 isn't read as an in-HBM result (plan.txt)."""
    if "cpu_offload_gb" not in sub.columns:
        return set()
    s = sub[sub["cpu_offload_gb"].fillna(0) > 0]
    return set(zip(s.device, s.precision))


def _label(dev, prec, offp):
    base = f"{DEVICE_LABEL[dev]} {prec}"
    return base + (" (offload)" if (dev, prec) in offp else "")


def _dev_prec_pairs(g):
    seen, out = set(), []
    for dev in DEVICE_STYLE:
        for prec in ("bf16", "fp8"):
            if ((g.device == dev) & (g.precision == prec)).any() and (dev, prec) not in seen:
                seen.add((dev, prec))
                out.append((dev, prec))
    return out


def _legend(fig, axes, ncol):
    h, l = axes[0].get_legend_handles_labels()
    nrows = -(-len(l) // ncol)                      # ceil
    y = 1.04 + 0.045 * nrows                        # clear the legend rows
    fig.legend(h, l, loc="upper center", ncol=ncol, frameon=False,
               bbox_to_anchor=(0.5, y))
    return y


# ── per-metric small multiples (one panel per seq_len, shared y) ─────────────
def metric_panels(df, model, metric, ylabel, fname, outdir, logy=False):
    sub = df[df.model_label == model]
    g = agg(sub, metric)
    if g.empty:
        return
    seqs = sorted(g.seq_len.dropna().unique())
    pairs = _dev_prec_pairs(g)
    offp = offload_pairs(sub)

    fig, axes = plt.subplots(1, len(seqs),
                             figsize=(max(5.5, 4.8 * len(seqs)), 4.3),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for i, seq in enumerate(seqs):
        ax = axes[i]
        for dev, prec in pairs:
            s = g[(g.device == dev) & (g.precision == prec) &
                  (g.seq_len == seq)].sort_values("batch_size")
            if s.empty:
                continue
            ax.errorbar(s.batch_size, s["mean"], yerr=s["err"],
                        color=DEVICE_STYLE[dev]["color"],
                        marker=DEVICE_STYLE[dev]["marker"],
                        linestyle=PRECISION_STYLE.get(prec, "-"),
                        linewidth=2, markersize=6, capsize=3,
                        label=_label(dev, prec, offp))
        ax.set_title(f"output len = {int(seq)}", fontweight="bold")
        ax.set_xlabel("Batch size (concurrency)")
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        if logy:
            ax.set_yscale("log")
        ax.set_ylabel(ylabel if i == 0 else "")
        ax.grid(True, which="both", alpha=0.3)

    _legend(fig, axes, ncol=min(len(pairs), 4))
    fig.tight_layout()
    out = os.path.join(outdir, fname)
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"✓ {out}")


# ── speedup at a FIXED precision (apples-to-apples) ─────────────────────────
def speedup_panels(df, model, precision, outdir, baseline="gaudi2"):
    sub = df[(df.model_label == model) & (df.precision == precision)]
    g = agg(sub, "throughput")
    if g.empty or not (g.device == baseline).any():
        return
    others = [d for d in DEVICE_STYLE if d != baseline and (g.device == d).any()]
    if not others:
        return
    offp = offload_pairs(sub)
    seqs = sorted(g.seq_len.dropna().unique())

    fig, axes = plt.subplots(1, len(seqs),
                             figsize=(max(5.5, 4.8 * len(seqs)), 4.3),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for i, seq in enumerate(seqs):
        ax = axes[i]
        base = g[(g.device == baseline) & (g.seq_len == seq)].set_index("batch_size")["mean"]
        for dev in others:
            comp = g[(g.device == dev) & (g.seq_len == seq)].set_index("batch_size")["mean"]
            common = base.index.intersection(comp.index)
            if common.empty:
                continue
            ratio = base.loc[common] / comp.loc[common]
            tag = " (offload)" if (dev, precision) in offp else ""
            ax.plot(common, ratio, color=DEVICE_STYLE[dev]["color"],
                    marker=DEVICE_STYLE[dev]["marker"], linewidth=2, markersize=6,
                    label=f"{DEVICE_LABEL[baseline]} / {DEVICE_LABEL[dev]}{tag}")
            for x, y in zip(common, ratio):
                ax.annotate(f"{y:.1f}x", (x, y), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=8,
                            fontweight="bold", color=DEVICE_STYLE[dev]["color"])
        ax.axhline(1.0, color="grey", ls="--", lw=0.8, alpha=0.6)
        ax.set_title(f"output len = {int(seq)}", fontweight="bold")
        ax.set_xlabel("Batch size (concurrency)")
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.set_ylabel(f"Speedup ({precision})" if i == 0 else "")
        ax.grid(True, alpha=0.3)

    _legend(fig, axes, ncol=min(len(others), 4))
    fig.tight_layout()
    out = os.path.join(outdir, f"speedup_{model}_{precision}.pdf")
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"✓ {out}")


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "bench_results/merged.csv"
    df = load(csv_path)
    outdir = os.path.dirname(csv_path) or "."

    for model in [m for m in ["8B", "70B"] if (df.model_label == m).any()]:
        metric_panels(df, model, "throughput", "Throughput (tokens/s)",
                      f"throughput_{model}.pdf", outdir, logy=True)
        metric_panels(df, model, "first_token_ms", "Time to first token (ms)",
                      f"ttft_{model}.pdf", outdir, logy=True)
        metric_panels(df, model, "rest_token_ms", "Inter-token latency / TPOT (ms)",
                      f"itl_{model}.pdf", outdir, logy=True)
        metric_panels(df, model, "mem_max_gb", "Peak memory (GB)",
                      f"memory_{model}.pdf", outdir, logy=False)
        metric_panels(df, model, "energy_per_token", "Energy per token (J)",
                      f"energy_{model}.pdf", outdir, logy=True)
        for prec in ("bf16", "fp8"):
            speedup_panels(df, model, prec, outdir)

    print("All plots saved.")


if __name__ == "__main__":
    main()
