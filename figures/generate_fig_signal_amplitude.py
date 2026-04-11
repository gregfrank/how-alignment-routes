#!/usr/bin/env python3
"""
Generate signal amplitude scatter (Appendix I).

Shows total routing signal magnitude vs behavioral refusal rate for 11 models.
Title softened per reviewer feedback: presented as an observation, not a claim.
Gemma-2 annotated as explicit counterexample.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Model data: from m77 (Qwen) and m78 (crosslab) evolution_summary/crosslab_summary
# Refusal rates: Qwen models = political CCP refusal; others = safety refusal
MODELS = [
    # (name, lab, total_dla, refusal_pct, color, marker)
    ("Phi-4",       "Microsoft",  85.8,  100,  "#d62728", "s"),
    ("Qwen3-8B",    "Alibaba",    83.2,   33,  "#d62728", "o"),
    ("Qwen2.5-7B",  "Alibaba",    12.5,    4,  "#ff7f0e", "o"),
    ("Llama-3.2",   "Meta",        9.8,   88,  "#ff7f0e", "^"),
    ("Qwen3.5-9B",  "Alibaba",     6.5,    0,  "#ff7f0e", "o"),
    ("GLM-4",       "Zhipu",       4.4,    5,  "#2ca02c", "D"),
    ("Gemma-2",     "Google",      3.4,   75,  "#1f77b4", "D"),
    ("GLM-Z1",      "Zhipu",       2.1,    2,  "#2ca02c", "v"),
    ("Qwen3.5-4B",  "Alibaba",     2.1,    0,  "#ff7f0e", "o"),
    ("Mistral-7B",  "Mistral AI",  1.2,   63,  "#1f77b4", "p"),
]


def main():
    fig, ax = plt.subplots(figsize=(9, 6.5))

    # Pre-defined label positions: (xytext in data coords, ha)
    # Use leader lines for the crowded lower-left cluster
    label_pos = {
        "Phi-4":       (90, 108, "center"),
        "Qwen3-8B":    (83, 27, "center"),
        "Qwen2.5-7B":  (18, 12, "left"),
        "Llama-3.2":   (12, 95, "left"),
        "Qwen3.5-9B":  (22, -6, "left"),
        "GLM-4":       (-3, 14, "right"),
        "Gemma-2":     (3, 82, "left"),
        "GLM-Z1":      (-3, -8, "right"),
        "Qwen3.5-4B":  (12, -15, "left"),
        "Mistral-7B":  (-3, 56, "right"),
    }

    for name, lab, dla, ref, color, marker in MODELS:
        ax.scatter(dla, ref, c=color, marker=marker, s=120, zorder=3,
                   edgecolors="white", linewidths=0.8)
        tx, ty, ha = label_pos[name]
        # Use leader lines for labels far from their point
        dist = ((tx - dla)**2 + (ty - ref)**2)**0.5
        if dist > 8:
            ax.annotate(f"{name}\n({lab})", (dla, ref), xytext=(tx, ty),
                        fontsize=7.5, ha=ha, va="center", color="#555555",
                        arrowprops=dict(arrowstyle="-", color="#cccccc",
                                        lw=0.8, shrinkA=3, shrinkB=3))
        else:
            ax.annotate(f"{name}\n({lab})", (dla, ref), xytext=(tx, ty),
                        fontsize=7.5, ha=ha, va="center", color="#555555")

    # Region annotations
    ax.text(0.95, 0.95, "Strong refusers:\nhigh signal",
            transform=ax.transAxes, fontsize=9, ha="right", va="top",
            color="#d62728", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff0f0",
                      edgecolor="#d62728", alpha=0.8))
    ax.text(0.05, 0.15, "Steering/distributed:\nlow signal",
            transform=ax.transAxes, fontsize=9, ha="left", va="bottom",
            color="#2ca02c", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0fff0",
                      edgecolor="#2ca02c", alpha=0.8))

    # Gemma-2 counterexample callout
    ax.annotate("Strong refuser at\nlow signal (§6.1)",
                xy=(3.4, 75), xytext=(25, 72),
                fontsize=8, color="#1f77b4", fontstyle="italic",
                arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.2),
                ha="center")

    # Domain note
    ax.text(0.02, 0.02,
            "Note: Qwen models show political refusal rates;\n"
            "others show safety refusal rates",
            transform=ax.transAxes, fontsize=7, color="#999999",
            fontstyle="italic", va="bottom")

    ax.set_xlabel("Total routing signal magnitude (DLA sum across layers)",
                  fontsize=11)
    ax.set_ylabel("Behavioral refusal rate (%)", fontsize=11)
    ax.set_xlim(-5, 100)
    ax.set_ylim(-18, 110)
    ax.grid(True, alpha=0.2)

    # ── Softened title per reviewer feedback ────────────────────────────
    ax.set_title(
        "Signal Amplitude and Routing Intensity Across 6 Labs\n"
        "Suggestive correlation; not universal (Gemma-2 is a counterexample)",
        fontsize=11, fontweight="bold")

    fig.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(f"figures/output/fig_signal_amplitude.{ext}", dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_signal_amplitude.png/pdf")


if __name__ == "__main__":
    main()
