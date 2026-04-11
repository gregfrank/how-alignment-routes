#!/usr/bin/env python3
"""
Generate Figure 5: Cross-architecture routing profiles.

The left panel shows how quickly routing signal concentrates into the top-K
heads. The right panel is deliberately more cautious: it shows confirmed gate
depths where available and top-routing-head depths elsewhere.
"""

from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODELS = {
    "Qwen3-8B\n(Alibaba)": {
        "path": str(REPO_ROOT / "results/panel/qwen3_8b/headDLA/head_summary.csv"),
        "color": "#d62728", "marker": "o", "domain": "political",
        "knockout": True,
    },
    "Phi-4-mini\n(Microsoft)": {
        "path": str(REPO_ROOT / "results/panel/phi4_mini/headDLA/head_summary.csv"),
        "color": "#1f77b4", "marker": "s", "domain": "safety",
        "knockout": True,
    },
    "Llama-3.2\n(Meta)": {
        "path": str(REPO_ROOT / "results/panel/llama32_3b/headDLA/head_summary.csv"),
        "color": "#ff7f0e", "marker": "^", "domain": "safety",
        "knockout": True,
    },
    "Gemma-2-2B\n(Google)": {
        "path": str(REPO_ROOT / "results/panel/gemma2_2b/headDLA/head_summary.csv"),
        "color": "#2ca02c", "marker": "D", "domain": "safety",
        "knockout": False,
    },
    "GLM-Z1\n(Zhipu)": {
        "path": str(REPO_ROOT / "results/panel/glmz1_9b/headDLA/head_summary.csv"),
        "color": "#9467bd", "marker": "v", "domain": "safety",
        "knockout": False,
    },
    "Mistral-7B\n(Mistral AI)": {
        "path": str(REPO_ROOT / "results/panel/mistral_7b/headDLA/head_summary.csv"),
        "color": "#8c564b", "marker": "p", "domain": "safety",
        "knockout": False,
    },
}


def load_cumulative_profile(path, max_k=30):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    head_cols = [c for c in rows[0].keys() if c.startswith("mean_delta_head_")]
    n_heads = len(head_cols)

    all_deltas = []
    for row in rows:
        for h in range(n_heads):
            all_deltas.append(abs(float(row[f"mean_delta_head_{h}"])))
    all_deltas.sort(reverse=True)
    total = sum(all_deltas)
    if total == 0:
        return list(range(1, max_k + 1)), [0.0] * max_k

    ks = list(range(1, min(max_k + 1, len(all_deltas) + 1)))
    cumulative = []
    running = 0.0
    for k in ks:
        running += all_deltas[k - 1]
        cumulative.append(100.0 * running / total)
    return ks, cumulative


def load_gate_depths(models):
    """Return confirmed gate depth or top-routing-head depth as % of layers."""
    # Confirmed gates where knockout exists; otherwise top routing head from DLA/ablation
    gate_info = {
        "Qwen3-8B\n(Alibaba)": (17, 36),        # L17.H17, 36 layers
        "Phi-4-mini\n(Microsoft)": (13, 32),      # L13.H7, 32 layers
        "Llama-3.2\n(Meta)": (13, 28),            # L13.H18, 28 layers
        "Gemma-2\n(Google)": (13, 26),            # L13.H2, 26 layers
        "GLM-Z1\n(Zhipu)": (19, 40),             # L19.H23, 40 layers
        "Mistral-7B\n(Mistral AI)": (14, 32),    # L14.H21, 32 layers
    }
    depths = {}
    for name in models:
        clean = name.split("\n")[0]
        for gname, (layer, total) in gate_info.items():
            if clean in gname:
                depths[name] = layer / total
    return depths


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [3, 1.2]})

    # === Panel A: Cumulative signal curves ===
    for name, info in MODELS.items():
        path = Path(info["path"])
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping {name}")
            continue
        ks, cum = load_cumulative_profile(str(path), max_k=25)
        linestyle = "-" if info["knockout"] else "--"
        linewidth = 2.5 if info["knockout"] else 1.5
        ax1.plot(ks, cum, linestyle=linestyle, color=info["color"],
                 marker=info["marker"], markersize=5, linewidth=linewidth,
                 label=name.replace("\n", " "), markevery=2)

    ax1.set_xlabel("Number of top heads (K)", fontsize=11)
    ax1.set_ylabel("Cumulative % of total routing signal", fontsize=11)
    ax1.set_title("Routing Signal Concentration Across Architectures", fontsize=12, fontweight="bold")
    ax1.set_xlim(0.5, 25.5)
    ax1.set_ylim(0, 80)
    ax1.axhline(y=30, color="#dddddd", linestyle=":", linewidth=1)
    ax1.axhline(y=50, color="#dddddd", linestyle=":", linewidth=1)
    ax1.legend(fontsize=8, loc="lower right", ncol=2)
    ax1.grid(True, alpha=0.2)

    # Annotation: solid = knockout confirmed, dashed = DLA only
    ax1.text(0.02, 0.98, "Solid: knockout confirmed\nDashed: DLA only",
             transform=ax1.transAxes, fontsize=8, va="top", color="#777777",
             fontstyle="italic")

    # === Panel B: Gate head depth ===
    depths = load_gate_depths(MODELS)
    names = []
    depth_vals = []
    colors = []
    confirmed = []
    for name, info in MODELS.items():
        if name in depths and "distributed" not in name:
            names.append(name.split("\n")[0])
            depth_vals.append(depths[name] * 100)
            colors.append(info["color"])
            confirmed.append(info["knockout"])

    y_pos = np.arange(len(names))
    for i, (dv, c, conf) in enumerate(zip(depth_vals, colors, confirmed)):
        hatch = None if conf else "///"
        ec = "white" if conf else "#333333"
        ax2.barh(y_pos[i], dv, color=c, alpha=0.8, height=0.6,
                 edgecolor=ec, hatch=hatch, linewidth=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names, fontsize=9)
    ax2.set_xlabel("Confirmed gate / top routing head depth (% of layers)", fontsize=10)
    ax2.set_title("Confirmed Gates and Top Routing Heads\ncluster at 40-50% depth", fontsize=11, fontweight="bold")
    ax2.set_xlim(0, 70)
    ax2.axvline(x=40, color="#999999", linestyle="--", linewidth=1, alpha=0.5)
    ax2.axvline(x=50, color="#999999", linestyle="--", linewidth=1, alpha=0.5)
    ax2.invert_yaxis()

    # Add percentage labels on bars
    for i, v in enumerate(depth_vals):
        ax2.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=8, color="#555555")

    # Annotation distinguishing confirmed vs DLA-only
    ax2.text(0.02, 0.02, "Solid fill: knockout confirmed\nHatched: DLA top head only",
             transform=ax2.transAxes, fontsize=7, va="bottom", color="#777777",
             fontstyle="italic")

    fig.suptitle("Sparse Routing Recurs Across Refusal-Based Architectures",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(f"figures/output/fig_cross_architecture.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_cross_architecture.png/pdf")


if __name__ == "__main__":
    main()
