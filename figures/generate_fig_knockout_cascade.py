#!/usr/bin/env python3
"""
Generate Figure: Gate-Amplifier knockout cascade across 3 models.

Shows paired bars (normal vs after gate knockout) for each amplifier head,
with percentage change labels. Demonstrates that knocking out the gate head
silences or reverses downstream amplifiers — consistent across labs.
"""

from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = REPO_ROOT / "figures" / "output"

MODELS = [
    {
        "name": "Qwen3-8B (n=120)",
        "gate": "L17.H17",
        "path": REPO_ROOT / "results/knockout/qwen3_8b/cascade_summary.csv",
        "heads": [(22, 7), (23, 3), (22, 4), (22, 5), (22, 6), (23, 2)],
        "normalized": False,
    },
    {
        "name": "Phi-4-mini (n=120)",
        "gate": "L13.H7",
        "path": REPO_ROOT / "results/knockout/phi4_mini/cascade_summary.csv",
        "heads": [(16, 13), (26, 9), (29, 18), (26, 11), (27, 19)],
        "normalized": False,
    },
    {
        "name": "Gemma-2-2B (n=120)",
        "gate": "L13.H2",
        "path": REPO_ROOT / "results/knockout/gemma2_2b/cascade_summary.csv",
        "heads": [(16, 1), (15, 2), (12, 3), (8, 3), (15, 7)],
        "normalized": False,
    },
]


def load_cascade(path):
    """Load cascade summary into a dict keyed by (layer, head)."""
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            key = (int(row["layer"]), int(row["head"]))
            data[key] = {
                "normal": float(row["mean_normal_delta"]),
                "knockout": float(row["mean_knockout_delta"]),
                "change_pct": float(row["change_pct"]),
            }
    return data


def main():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5),
                              gridspec_kw={"width_ratios": [1.1, 1, 1]})

    for ax, model in zip(axes, MODELS):
        cascade = load_cascade(model["path"])
        heads = model["heads"]
        labels = [f"L{l}.H{h}" for l, h in heads]
        normal_vals = []
        knockout_vals = []
        change_pcts = []

        for lh in heads:
            d = cascade[lh]
            normal_vals.append(d["normal"])
            knockout_vals.append(d["knockout"])
            change_pcts.append(d["change_pct"])

        normal_vals = np.array(normal_vals)
        knockout_vals = np.array(knockout_vals)

        # Normalize Llama to make it visually comparable
        if model["normalized"]:
            scale = np.max(np.abs(normal_vals))
            normal_vals = normal_vals / scale
            knockout_vals = knockout_vals / scale

        x = np.arange(len(heads))
        width = 0.35

        ax.bar(x - width / 2, normal_vals, width, color="#1f77b4", alpha=0.85,
               label="Normal", edgecolor="white", linewidth=0.5)
        ax.bar(x + width / 2, knockout_vals, width, color="#d62728", alpha=0.7,
               label="After gate knockout", edgecolor="white", linewidth=0.5)

        # Add percentage change labels above each pair
        for i, pct in enumerate(change_pcts):
            bar_top = max(normal_vals[i], knockout_vals[i])
            bar_bot = min(normal_vals[i], knockout_vals[i])
            # Place label above the taller bar
            if bar_top >= 0:
                y_label = bar_top + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]) if ax.get_ylim()[1] > 0 else bar_top + 0.05
            else:
                y_label = bar_bot - 0.05

            sign = "+" if pct > 0 else ""
            color = "#d62728" if pct < 0 else "#2ca02c"
            ax.text(x[i], bar_top, f"{sign}{pct:.0f}%",
                    ha="center", va="bottom", fontsize=12, color=color,
                    fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10.5, rotation=0)
        ax.axhline(0, color="#cccccc", linewidth=0.8)

        subtitle = f"(Gate: {model['gate']})"
        if model["normalized"]:
            subtitle += " — normalized scale"
        ax.set_title(f"{model['name']}\n{subtitle}", fontsize=12, fontweight="bold")

        if ax == axes[0]:
            ax.set_ylabel("Amplifier head contribution", fontsize=12)
            ax.legend(fontsize=12, loc="upper right")
        else:
            ax.legend(fontsize=12, loc="best")

        ax.grid(True, alpha=0.15, axis="y")

        # Expand y-limits to give headroom for percentage labels
        ymin, ymax = ax.get_ylim()
        margin = (ymax - ymin) * 0.18
        ax.set_ylim(ymin - margin * 0.3, ymax + margin)

    fig.suptitle(
        "Gate-Amplifier Architecture: Knockout Cascade Across Three Architectures\n"
        "Knocking out the gate head silences or reverses downstream amplifiers",
        fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"fig_knockout_cascade.{ext}", dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_knockout_cascade.png/pdf")


if __name__ == "__main__":
    main()
