#!/usr/bin/env python3
"""
plot_results.py  —  Generate comparison plots from the merged benchmark CSV.

Usage:
    python plot_results.py [path/to/merged.csv]

Produces three PDF figures in the same directory as the CSV:
    1. throughput_vs_batchsize.pdf
    2. energy_per_token.pdf
    3. speedup_vs_hpu.pdf

Colors:  Blue = Gaudi2 (HPU)  |  Yellow = A100  |  Green = H100
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

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


def load(path):
    df = pd.read_csv(path)
    # normalise device names to lowercase
    df["device"] = df["device"].str.strip().str.lower()
    # ensure numeric
    for c in ["seq_len", "batch_size", "throughput", "energy_per_token", "wall_time_s"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ── helper: mean ± std over repeats ─────────────────────────────────────────
def agg(df, metric):
    """Group by (device, seq_len, batch_size) and return mean + std."""
    grouped = (
        df.groupby(["device", "seq_len", "batch_size"])[metric]
        .agg(["mean", "std"])
        .reset_index()
    )
    return grouped


# ── Plot 1: Throughput vs batch size ────────────────────────────────────────
def plot_throughput(df, outdir):
    seq_lens = sorted(df["seq_len"].dropna().unique())
    g = agg(df.dropna(subset=["throughput"]), "throughput")

    fig, axes = plt.subplots(1, len(seq_lens), figsize=(5.5 * len(seq_lens), 4.5),
                             sharey=False, squeeze=False)
    axes = axes[0]

    for i, seq in enumerate(seq_lens):
        ax = axes[i]
        for dev, sty in DEVICE_STYLE.items():
            sub = g[(g["device"] == dev) & (g["seq_len"] == seq)].sort_values("batch_size")
            if sub.empty:
                continue
            ax.errorbar(
                sub["batch_size"], sub["mean"], yerr=sub["std"],
                color=sty["color"], marker=sty["marker"], label=sty["label"],
                linewidth=2, markersize=7, capsize=3,
            )
        ax.set_title(f"Seq length = {int(seq)}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Batch size")
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.set_ylabel("Throughput (tokens/s)" if i == 0 else "")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Throughput vs Batch Size", fontsize=14, fontweight="bold", y=1.07)
    fig.tight_layout()
    path = os.path.join(outdir, "throughput_vs_batchsize.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"✓ {path}")
    plt.close(fig)


# ── Plot 2: Energy per token ───────────────────────────────────────────────
def plot_energy(df, outdir):
    col = "energy_per_token"
    if col not in df.columns or df[col].dropna().empty:
        print("⚠  No energy_per_token data – skipping energy plot.")
        return

    seq_lens = sorted(df["seq_len"].dropna().unique())
    g = agg(df.dropna(subset=[col]), col)

    fig, axes = plt.subplots(1, len(seq_lens), figsize=(5.5 * len(seq_lens), 4.5),
                             sharey=False, squeeze=False)
    axes = axes[0]

    for i, seq in enumerate(seq_lens):
        ax = axes[i]
        for dev, sty in DEVICE_STYLE.items():
            sub = g[(g["device"] == dev) & (g["seq_len"] == seq)].sort_values("batch_size")
            if sub.empty:
                continue
            ax.errorbar(
                sub["batch_size"], sub["mean"], yerr=sub["std"],
                color=sty["color"], marker=sty["marker"], label=sty["label"],
                linewidth=2, markersize=7, capsize=3,
            )
        ax.set_title(f"Seq length = {int(seq)}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Batch size")
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.set_ylabel("Energy per token (J)" if i == 0 else "")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Energy per Token", fontsize=14, fontweight="bold", y=1.07)
    fig.tight_layout()
    path = os.path.join(outdir, "energy_per_token.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"✓ {path}")
    plt.close(fig)


# ── Plot 3: Speedup  (HPU throughput / CUDA throughput) ─────────────────────
def plot_speedup(df, outdir):
    thr = agg(df.dropna(subset=["throughput"]), "throughput")
    seq_lens = sorted(df["seq_len"].dropna().unique())

    fig, axes = plt.subplots(1, len(seq_lens), figsize=(5.5 * len(seq_lens), 4.5),
                             sharey=False, squeeze=False)
    axes = axes[0]

    for i, seq in enumerate(seq_lens):
        ax = axes[i]
        hpu = thr[(thr["device"] == "gaudi2") & (thr["seq_len"] == seq)].set_index("batch_size")

        for dev, sty in SPEEDUP_STYLE.items():
            cuda = thr[(thr["device"] == dev) & (thr["seq_len"] == seq)].set_index("batch_size")
            # align on common batch sizes
            common = hpu.index.intersection(cuda.index)
            if common.empty:
                continue
            speedup = hpu.loc[common, "mean"].values / cuda.loc[common, "mean"].values
            ax.plot(
                common, speedup,
                color=sty["color"], marker=sty["marker"], label=sty["label"],
                linewidth=2, markersize=7,
            )
            # label each point with "1.5x", "2.0x", etc.
            for x, y in zip(common, speedup):
                ax.annotate(
                    f"{y:.1f}x",
                    xy=(x, y), xytext=(0, 10),
                    textcoords="offset points", ha="center", fontsize=8,
                    fontweight="bold", color=sty["color"],
                )

        ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_title(f"Seq length = {int(seq)}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Batch size")
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.set_ylabel("HPU Speedup" if i == 0 else "")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Gaudi2 (HPU) Speedup over CUDA", fontsize=14,
                 fontweight="bold", y=1.07)
    fig.tight_layout()
    path = os.path.join(outdir, "speedup_hpu_over_cuda.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"✓ {path}")
    plt.close(fig)


# ── main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "/scratch/tianche5/gaudi_bench/bench_results/merged.csv"
    df = load(csv_path)
    outdir = os.path.dirname(csv_path) or "."

    plot_throughput(df, outdir)
    plot_energy(df, outdir)
    plot_speedup(df, outdir)
    print("All plots saved.")
