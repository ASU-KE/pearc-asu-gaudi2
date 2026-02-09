#!/usr/bin/env python3
"""
plot_results.py  —  Generate comparison plots from the merged benchmark CSV.

Usage:
    python plot_results.py [path/to/merged.csv]

Produces three PDF figures in the same directory as the CSV:
    1. throughput_vs_batchsize.pdf
    2. energy_per_token.pdf
    3. speedup_hpu_over_cuda.pdf
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patheffects as pe

# ── config ───────────────────────────────────────────────────────────────────
DEVICE_STYLE = {
    "gaudi2": {"color": "#2563EB", "marker": "s", "label": "Gaudi2 (HPU)"},
    "a100":   {"color": "#EAB308", "marker": "^", "label": "A100"},
    "h100":   {"color": "#16A34A", "marker": "o", "label": "H100"},
}

SPEEDUP_STYLE = {
    "a100": {"color": "#EAB308", "marker": "^", "label": "HPU / A100"},
    "h100": {"color": "#16A34A", "marker": "o", "label": "HPU / H100"},
}


# ── load CSV ────────────────────────────────────────────────────────────────
def load(path):
    df = pd.read_csv(path)
    df["device"] = df["device"].str.strip().str.lower()
    for c in ["seq_len", "batch_size", "throughput",
              "energy_per_token", "wall_time_s"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ── helper: mean ± std over repeats ─────────────────────────────────────────
def agg(df, metric):
    return (
        df.groupby(["device", "seq_len", "batch_size"])[metric]
        .agg(["mean", "std"])
        .reset_index()
    )


# ── Plot 1: Throughput vs batch size ────────────────────────────────────────
def plot_throughput(df, outdir):
    seq_lens = sorted(df["seq_len"].dropna().unique())
    g = agg(df.dropna(subset=["throughput"]), "throughput")

    fig, axes = plt.subplots(
        1, len(seq_lens), figsize=(5.5 * len(seq_lens), 4.5),
        squeeze=False
    )
    axes = axes[0]

    for i, seq in enumerate(seq_lens):
        ax = axes[i]
        for dev, sty in DEVICE_STYLE.items():
            sub = g[(g.device == dev) & (g.seq_len == seq)].sort_values("batch_size")
            if sub.empty:
                continue
            ax.errorbar(
                sub.batch_size, sub["mean"], yerr=sub["std"],
                color=sty["color"], marker=sty["marker"],
                linewidth=2, markersize=7, capsize=3, label=sty["label"]
            )

        ax.set_title(f"Seq length = {int(seq)}", fontweight="bold")
        ax.set_xlabel("Batch size")
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.set_ylabel("Throughput (tokens/s)" if i == 0 else "")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Throughput vs Batch Size", fontweight="bold", y=1.07)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "throughput_vs_batchsize.pdf"),
                bbox_inches="tight", dpi=150)
    plt.close(fig)


# ── Plot 2: Energy per token ───────────────────────────────────────────────
def plot_energy(df, outdir):
    if "energy_per_token" not in df or df["energy_per_token"].dropna().empty:
        print("⚠  No energy_per_token data – skipping energy plot.")
        return

    seq_lens = sorted(df["seq_len"].dropna().unique())
    g = agg(df.dropna(subset=["energy_per_token"]), "energy_per_token")

    fig, axes = plt.subplots(
        1, len(seq_lens), figsize=(5.5 * len(seq_lens), 4.5),
        squeeze=False
    )
    axes = axes[0]

    for i, seq in enumerate(seq_lens):
        ax = axes[i]
        for dev, sty in DEVICE_STYLE.items():
            sub = g[(g.device == dev) & (g.seq_len == seq)].sort_values("batch_size")
            if sub.empty:
                continue
            ax.errorbar(
                sub.batch_size, sub["mean"], yerr=sub["std"],
                color=sty["color"], marker=sty["marker"],
                linewidth=2, markersize=7, capsize=3, label=sty["label"]
            )

        ax.set_title(f"Seq length = {int(seq)}", fontweight="bold")
        ax.set_xlabel("Batch size")
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.set_ylabel("Energy per token (J)" if i == 0 else "")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Energy per Token", fontweight="bold", y=1.07)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "energy_per_token.pdf"),
                bbox_inches="tight", dpi=150)
    plt.close(fig)


# ── Plot 3: Speedup (HPU / CUDA) ────────────────────────────────────────────
def plot_speedup(df, outdir):
    thr = agg(df.dropna(subset=["throughput"]), "throughput")
    seq_lens = sorted(df["seq_len"].dropna().unique())

    fig, axes = plt.subplots(
        1, len(seq_lens), figsize=(5.5 * len(seq_lens), 4.5),
        squeeze=False
    )
    axes = axes[0]

    for i, seq in enumerate(seq_lens):
        ax = axes[i]
        hpu = thr[(thr.device == "gaudi2") & (thr.seq_len == seq)].set_index("batch_size")

        for dev, sty in SPEEDUP_STYLE.items():
            cuda = thr[(thr.device == dev) & (thr.seq_len == seq)].set_index("batch_size")
            common = hpu.index.intersection(cuda.index)
            if common.empty:
                continue

            speedup = hpu.loc[common, "mean"] / cuda.loc[common, "mean"]
            ax.plot(common, speedup, marker=sty["marker"],
                    color=sty["color"], linewidth=2, label=sty["label"])

            ymin, ymax = ax.get_ylim()
            for x, y in zip(common, speedup):
                if y > ymax * 0.9:
                    offset, va = (0, -12), "top"
                else:
                    offset, va = (0, 10), "bottom"

                ax.annotate(
                    f"{y:.1f}x",
                    xy=(x, y),
                    xytext=offset,
                    textcoords="offset points",
                    ha="center", va=va,
                    fontsize=8, fontweight="bold",
                    color=sty["color"],
                    clip_on=True,
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")]
                )

        ax.axhline(1.0, linestyle="--", color="grey", alpha=0.6)
        ax.set_title(f"Seq length = {int(seq)}", fontweight="bold")
        ax.set_xlabel("Batch size")
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.set_ylabel("HPU Speedup" if i == 0 else "")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Gaudi2 (HPU) Speedup over CUDA", fontweight="bold", y=1.07)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "speedup_hpu_over_cuda.pdf"),
                bbox_inches="tight", dpi=150)
    plt.close(fig)


# ── main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "bench_results/merged.csv"
    df = load(csv_path)
    outdir = os.path.dirname(csv_path) or "."

    plot_throughput(df, outdir)
    plot_energy(df, outdir)
    plot_speedup(df, outdir)
    print("✓ All plots saved.")

