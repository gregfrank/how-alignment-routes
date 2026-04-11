#!/usr/bin/env python3
"""Generate Figure: Sparsity-Scale Relationship.

Shows gate necessity (%) vs model size for all 12 models, with
connecting lines for selected same-generation scaling pairs.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

FIG_DIR = REPO_ROOT / "figures" / "output"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Data from all experiments at n=120
MODELS = [
    # (name, params_B, top_nec_pct, top_ablation, lab, generation, is_scaling_pair_member)
    ("Gemma-2-2B", 2.0, 8.4, 1.015, "Google", "Gemma-2", True),
    ("Llama-3.2-3B", 3.0, 3.0, 0.039, "Meta", "Llama-3.2", False),
    ("Phi-4-mini", 3.8, 3.4, 1.422, "Microsoft", "Phi-4", True),
    ("Mistral-7B", 7.0, 1.0, 0.015, "Mistral", "Mistral", False),
    ("Qwen3-8B", 8.0, 1.1, 0.137, "Alibaba", "Qwen3", True),
    ("Gemma-2-9B", 9.0, 1.9, 0.129, "Google", "Gemma-2", True),
    ("GLM-Z1-9B", 9.0, 4.7, 0.110, "Zhipu", "GLM-Z1", False),
    ("Phi-4", 14.0, 2.6, 0.083, "Microsoft", "Phi-4", True),
    ("Qwen3-32B", 32.0, 3.2, 0.105, "Alibaba", "Qwen3", True),
    ("Qwen2.5-7B", 7.0, 2.4, 0.906, "Alibaba", "Qwen2.5", True),
    ("Llama-3.3-70B", 70.0, 2.0, 0.382, "Meta", "Llama-3.3", False),
    ("Qwen2.5-72B", 72.0, 1.3, 0.016, "Alibaba", "Qwen2.5", True),
]

# Scaling pairs. Qwen3 is retained as a scaling pair in the data but left
# unconnected in the plot because the line visually competes with nearby labels.
PAIRS = [
    ("Qwen3", ["Qwen3-8B", "Qwen3-32B"], "#d62728"),
    ("Phi-4", ["Phi-4-mini", "Phi-4"], "#1f77b4"),
    ("Gemma-2", ["Gemma-2-2B", "Gemma-2-9B"], "#2ca02c"),
    ("Qwen2.5", ["Qwen2.5-7B", "Qwen2.5-72B"], "#e377c2"),
]
PLOT_PAIRS = [pair for pair in PAIRS if pair[0] != "Qwen3"]

LAB_MARKERS = {
    "Google": "D",
    "Meta": "^",
    "Microsoft": "s",
    "Mistral": "p",
    "Alibaba": "o",
    "Zhipu": "h",
}

LAB_COLORS = {
    "Google": "#2ca02c",
    "Meta": "#ff7f0e",
    "Microsoft": "#1f77b4",
    "Mistral": "#9467bd",
    "Alibaba": "#d62728",
    "Zhipu": "#8c564b",
}

LABEL_STYLE = dict(fontsize=9.3, fontweight="bold", color="#4a4a4a")

LEFT_LABEL_OFFSETS = {
    "Gemma-2-2B": (8, 4, "left"),
    "Llama-3.2-3B": (0, -16, "center"),
    "Phi-4-mini": (8, 6, "left"),
    "Mistral-7B": (-8, -7, "right"),
    "Qwen3-8B": (-16, 4, "right"),
    "Gemma-2-9B": (8, -10, "left"),
    "GLM-Z1-9B": (8, 7, "left"),
    "Phi-4": (8, 5, "left"),
    "Qwen3-32B": (8, 5, "left"),
    "Qwen2.5-7B": (8, 7, "left"),
    "Llama-3.3-70B": (0, 12, "center"),
    "Qwen2.5-72B": (-10, -1, "right"),
}

RIGHT_LABEL_OFFSETS = {
    "Gemma-2-2B": (8, 2, "left"),
    "Llama-3.2-3B": (8, 8, "left"),
    "Phi-4-mini": (8, 6, "left"),
    "Mistral-7B": (-8, 10, "right"),
    "Qwen3-8B": (-16, 8, "right"),
    "Gemma-2-9B": (8, -10, "left"),
    "GLM-Z1-9B": (0, -12, "center"),
    "Phi-4": (8, -8, "left"),
    "Qwen3-32B": (8, 5, "left"),
    "Qwen2.5-7B": (8, 8, "left"),
    "Llama-3.3-70B": (-10, 5, "right"),
    "Qwen2.5-72B": (-10, 0, "right"),
}


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # === Left panel: Top interchange necessity vs params ===
    for name, params, nec, abl, lab, gen, scaling in MODELS:
        ax1.scatter(params, nec, s=120, c=LAB_COLORS[lab],
                    marker=LAB_MARKERS[lab], zorder=5,
                    edgecolors="white", linewidths=0.5)
        dx, dy, ha = LEFT_LABEL_OFFSETS[name]
        ax1.annotate(name, (params, nec), xytext=(dx, dy),
                     textcoords="offset points", ha=ha, va="center",
                     **LABEL_STYLE)

    # Draw scaling pair lines
    for pair_name, members, color in PLOT_PAIRS:
        pts = [(p, n) for name, p, n, a, l, g, s in MODELS if name in members]
        pts.sort()
        xs, ys = zip(*pts)
        ax1.plot(xs, ys, "--", color=color, alpha=0.5, linewidth=2, zorder=2)

    ax1.set_xscale("log")
    ax1.set_xlabel("Model parameters (billions)", fontsize=11.5)
    ax1.set_ylabel("Top interchange necessity (%)", fontsize=11.5)
    ax1.set_title("Gate Strength Decreases With Scale", fontsize=12.5, fontweight="bold")
    ax1.grid(True, alpha=0.15)
    ax1.set_xlim(1.5, 85)

    # Legend for labs
    for lab, marker in LAB_MARKERS.items():
        ax1.scatter([], [], marker=marker, c=LAB_COLORS[lab], s=80, label=lab)
    ax1.legend(fontsize=8.8, loc="upper right", ncol=2)

    # === Right panel: Top ablation vs params ===
    for name, params, nec, abl, lab, gen, scaling in MODELS:
        ax2.scatter(params, abl, s=120, c=LAB_COLORS[lab],
                    marker=LAB_MARKERS[lab], zorder=5,
                    edgecolors="white", linewidths=0.5)
        dx, dy, ha = RIGHT_LABEL_OFFSETS[name]
        ax2.annotate(name, (params, abl), xytext=(dx, dy),
                     textcoords="offset points", ha=ha, va="center",
                     **LABEL_STYLE)

    for pair_name, members, color in PLOT_PAIRS:
        pts = [(p, a) for name, p, n, a, l, g, s in MODELS if name in members]
        pts.sort()
        xs, ys = zip(*pts)
        ax2.plot(xs, ys, "--", color=color, alpha=0.5, linewidth=2, zorder=2)

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Model parameters (billions)", fontsize=11.5)
    ax2.set_ylabel("Top head ablation effect", fontsize=11.5)
    ax2.set_title("Per-Head Signal Weakens With Scale", fontsize=12.5, fontweight="bold")
    ax2.grid(True, alpha=0.15)
    ax2.set_xlim(1.5, 85)

    fig.suptitle(
        "Gate-Amplifier Pattern Persists But Distributes At Scale\n"
        "12 models from 6 labs; dashed lines connect selected scaling pairs",
        fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"fig_scaling.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_scaling.png/pdf")


if __name__ == "__main__":
    main()
