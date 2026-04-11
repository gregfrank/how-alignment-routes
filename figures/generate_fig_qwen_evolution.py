#!/usr/bin/env python3
"""
Generate Figure 3: Qwen family evolution across generations.

Two-panel version for the main text:
- behavioral shift from refusal to steering
- mechanistic signature in total signal and top-1 head amplitude

The attention/MLP balance panel is omitted from the main figure to keep the
story focused on the mystery that motivates the mechanistic investigation.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Canonical data from m77_qwen_evolution/evolution_summary.json ────────
MODELS = ["Qwen2.5-7B", "Qwen3-8B", "Qwen3.5-4B", "Qwen3.5-9B"]

# Panel 1: Behavioral evolution
REFUSAL_PCT = [4.2, 33.3, 0.0, 0.0]
STEERING    = [2.62, 3.25, 5.0, 5.0]

# Panel 2: Mechanistic evolution
TOTAL_DLA   = [12.54, 83.20, 2.08, 6.53]
TOP1_DELTA  = [8.23, 37.96, 4.87, 14.74]  # ×100

def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8))

    x = np.arange(len(MODELS))
    width = 0.35

    # ── Panel 1: Behavioral evolution (dual y-axis) ────────────────────
    color_refusal = "#d62728"
    color_steer = "#1f77b4"

    bars_r = ax1.bar(x - width / 2, REFUSAL_PCT, width, color=color_refusal,
                     alpha=0.8, label="Refusal rate (%)")
    ax1.set_ylabel("Refusal rate (%)", color=color_refusal, fontsize=12)
    ax1.tick_params(axis="y", labelcolor=color_refusal, labelsize=10)
    ax1.set_ylim(0, 40)

    ax1b = ax1.twinx()
    bars_s = ax1b.bar(x + width / 2, STEERING, width, color=color_steer,
                      alpha=0.8, label="Steering score")
    ax1b.set_ylabel("Steering score (1-5)", color=color_steer, fontsize=12)
    ax1b.tick_params(axis="y", labelcolor=color_steer, labelsize=10)
    ax1b.set_ylim(0, 5.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(MODELS, fontsize=10)
    ax1.set_title("Behavioral Evolution", fontsize=13, fontweight="bold")

    # Combined legend
    ax1.legend([bars_r, bars_s], ["Refusal rate (%)", "Steering score"],
               loc="upper left", fontsize=9.5)

    # ── Panel 2: Mechanistic evolution ─────────────────────────────────
    color_dla = "#2ca02c"
    color_top1 = "#9467bd"

    ax2.bar(x - width / 2, TOTAL_DLA, width, color=color_dla, alpha=0.8,
            label="Total DLA signal")
    ax2.bar(x + width / 2, TOP1_DELTA, width, color=color_top1, alpha=0.8,
            label="Top-1 head |delta| (×100)")
    ax2.set_ylabel("Signal magnitude", fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(MODELS, fontsize=10)
    ax2.set_title("Mechanistic Evolution", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9.5, loc="upper right")
    ax2.tick_params(labelsize=10)

    ax2.text(
        0.48, 0.72, "Circuit relocation discussed in §4.3",
        transform=ax2.transAxes, ha="left", va="top",
        fontsize=9.5, color="#666666", fontstyle="italic",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                  edgecolor="#cccccc", alpha=0.9)
    )

    # ── Title: softened to match revised thesis ────────────────────────
    fig.suptitle(
        "Qwen Family: Behavioral Shift Has a Mechanistic Signature\n"
        "Refusal vanished while steering rose to maximum; the routing signal became quieter",
        fontsize=12, fontweight="bold", y=1.03,
    )
    fig.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(f"figures/output/fig_qwen_evolution.{ext}", dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_qwen_evolution.png/pdf")


if __name__ == "__main__":
    main()
