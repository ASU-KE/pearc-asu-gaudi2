#!/usr/bin/env python3
"""
plot_results_ohf.py — Comparison plots for the Optimum-Habana lazy-mode +
HPU-graphs Gaudi2 path (device label `gaudi2-ohf`), which is now the *only*
Gaudi2 path in the paper (the vLLM Habana extension never came up reliably).

It reuses the aggregation/labelling primitives from plot_results.py but draws
its own figures so the encoding can differ:

  * Colour = device, fixed map:   A100 orange · H100 green · GH200 blue ·
    Gaudi2 red.
  * Precision = colour SHADE of the same hue (bf16 = full, fp8 = light), all
    lines solid. Dashed "dot" lines were dropped — they read poorly in a small
    legend.

Two PNGs per model:
  * combined_<model>.png — throughput | peak memory | energy, side by side
    (1×3), one shared legend on top, no output-len title.
  * speedup_<model>.png  — bf16 | fp8 speedup vs the Gaudi2 baseline, side by
    side (1×2), one shared legend on top.

TTFT/ITL are omitted because run_generation.py reports aggregate throughput,
not a prefill/decode split (see run_optimum_gaudi.py).

Usage:
    python plot_results_ohf.py [path/to/merged.csv] [out_subdir]
"""

import os
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_results as base   # noqa: E402

# Fixed device colour map (overrides plot_results defaults).
base.DEVICE_STYLE["a100"]["color"] = "#FF7F0E"   # orange
base.DEVICE_STYLE["h100"]["color"] = "#2CA02C"   # green
base.DEVICE_STYLE["gh200"]["color"] = "#1F77B4"  # blue
# Optimum-Habana lazy run is the sole Gaudi2 series → plain "Gaudi2", red.
base.DEVICE_STYLE["gaudi2-ohf"] = {"color": "#D62728", "marker": "s"}  # red
base.DEVICE_LABEL["gaudi2-ohf"] = "Gaudi2"

# Precision → how far to lighten the device hue (0 = full colour, 1 = white).
PREC_SHADE = {"bf16": 0.0, "fp8": 0.45}

# Panels of the combined figure: (metric, title, y-axis label, log-y?).
COMBINED_METRICS = [
    ("throughput",       "Throughput",      "Tokens/s", True),
    ("mem_max_gb",       "Peak memory",     "GB",       False),
    ("energy_per_token", "Energy per token", "J",       True),
]


def _shade(color, amount):
    """Lighten `color` toward white by `amount` in [0,1] (0 = unchanged)."""
    r, g, b = mcolors.to_rgb(color)
    return tuple(1 - (1 - ch) * (1 - amount) for ch in (r, g, b))


def _dev_color(dev, prec):
    return _shade(base.DEVICE_STYLE[dev]["color"], PREC_SHADE.get(prec, 0.0))


def _top_legend(fig, handles):
    """One deduped legend, centred above the grid. `handles` is [(label, artist)]."""
    labels = [l for l, _ in handles]
    artists = [a for _, a in handles]
    fig.legend(artists, labels, loc="upper center", ncol=min(len(labels), 4),
               frameon=False, bbox_to_anchor=(0.5, 1.0))


# ── throughput | memory | energy, side by side (1×3) ────────────────────────
def combined_panels(df, model, fname, outdir):
    sub = df[df.model_label == model]
    if base.agg(sub, "throughput").empty:
        return
    offp = base.offload_pairs(sub)
    ncols = len(COMBINED_METRICS)

    fig, axes = plt.subplots(1, ncols, figsize=(4.8 * ncols, 4.3), squeeze=False)
    axes = axes[0]

    handles, seen = [], set()
    for c, (metric, title, ylabel, logy) in enumerate(COMBINED_METRICS):
        ax = axes[c]
        g = base.agg(sub, metric)
        for dev, prec in base._dev_prec_pairs(g):
            s = g[(g.device == dev) & (g.precision == prec)].sort_values("batch_size")
            if s.empty:
                continue
            lbl = base._label(dev, prec, offp)
            line = ax.errorbar(s.batch_size, s["mean"], yerr=s["err"],
                               color=_dev_color(dev, prec),
                               marker=base.DEVICE_STYLE[dev]["marker"],
                               linestyle="-", linewidth=2, markersize=6,
                               capsize=3, label=lbl)
            if lbl not in seen:
                seen.add(lbl)
                handles.append((lbl, line))
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Batch size (concurrency)")
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        if logy:
            ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)

    _top_legend(fig, handles)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = os.path.join(outdir, fname)
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"✓ {out}")


# ── bf16 | fp8 speedup vs Gaudi2 baseline, side by side (1×2) ────────────────
def combined_speedup(df, model, fname, outdir, baseline):
    panels = []
    for prec in ("bf16", "fp8"):
        sub = df[(df.model_label == model) & (df.precision == prec)]
        g = base.agg(sub, "throughput")
        if g.empty or not (g.device == baseline).any():
            continue
        others = [d for d in base.DEVICE_STYLE
                  if d != baseline and (g.device == d).any()]
        if others:
            panels.append((prec, sub, g, others))
    if not panels:
        return

    fig, axes = plt.subplots(1, len(panels),
                             figsize=(max(5.5, 4.8 * len(panels)), 4.3),
                             squeeze=False)
    axes = axes[0]

    handles, seen = [], set()
    for i, (prec, sub, g, others) in enumerate(panels):
        ax = axes[i]
        offp = base.offload_pairs(sub)
        for seq in sorted(g.seq_len.dropna().unique()):
            bse = g[(g.device == baseline) & (g.seq_len == seq)] \
                .set_index("batch_size")["mean"]
            for dev in others:
                comp = g[(g.device == dev) & (g.seq_len == seq)] \
                    .set_index("batch_size")["mean"]
                common = bse.index.intersection(comp.index)
                if common.empty:
                    continue
                ratio = bse.loc[common] / comp.loc[common]
                tag = " (offload)" if (dev, prec) in offp else ""
                lbl = f"{base.DEVICE_LABEL[baseline]} / {base.DEVICE_LABEL[dev]}{tag}"
                ln, = ax.plot(common, ratio, color=base.DEVICE_STYLE[dev]["color"],
                              marker=base.DEVICE_STYLE[dev]["marker"],
                              linewidth=2, markersize=6, label=lbl)
                if lbl not in seen:
                    seen.add(lbl)
                    handles.append((lbl, ln))
                for x, y in zip(common, ratio):
                    ax.annotate(f"{y:.1f}x", (x, y), textcoords="offset points",
                                xytext=(0, 8), ha="center", fontsize=8,
                                fontweight="bold",
                                color=base.DEVICE_STYLE[dev]["color"])
        ax.axhline(1.0, color="grey", ls="--", lw=0.8, alpha=0.6)
        ax.set_title(prec.upper(), fontweight="bold")
        ax.set_xlabel("Batch size (concurrency)")
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.set_ylabel(f"Speedup ({base.DEVICE_LABEL[baseline]} baseline)"
                      if i == 0 else "")
        ax.grid(True, alpha=0.3)

    _top_legend(fig, handles)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
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
        combined_speedup(df, model, f"speedup_{model}.png", outdir, baseline)

    print(f"All OHF-overlay plots saved → {outdir}")


if __name__ == "__main__":
    main()
