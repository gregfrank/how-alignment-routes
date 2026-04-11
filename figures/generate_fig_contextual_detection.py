#!/usr/bin/env python3
"""
Generate Figure 2: Routing is prompt-time and contextual.

Panel A shows that the Qwen routing signal is effectively unchanged between
the last prompt token and the first generated token, supporting the claim
that the routing decision is made during prompt processing.

Panel B shows that the same keyword ("Tiananmen") produces different probe
scores depending on framing, with two annotated edge cases showing that
routing is not determined by a simple scalar threshold.

Data:
- M62 prompt-time localization (position comparison)
- M75 contextual detection
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = REPO_ROOT / "results" / "m75_contextual_detection" / "contextual_gradient_results.csv"
POSITION_PATH = REPO_ROOT / "results" / "m62_routing_localization" / "qwen_position_comparison.json"
FIG_DIR = REPO_ROOT / "figures" / "output"

# Map CSV categories to display names and x-positions
CATEGORY_ORDER = [
    ("control_tourism",       "Non-Tiananmen\ncontrols"),
    ("control_cultural",      "Non-Tiananmen\ncontrols"),
    ("control_ambiguous",     "Non-Tiananmen\ncontrols"),
    ("control_political",     "Non-Tiananmen\ncontrols"),
    ("tourism",               "Tiananmen\ntourism"),
    ("cultural",              "Tiananmen\ncultural"),
    ("ambiguous",             "Tiananmen\nambiguous"),
    ("implicit_political",    "Tiananmen\nimplicit political"),
    ("explicit_political",    "Tiananmen\nexplicit political"),
]

DISPLAY_NAMES = [
    "Non-Tiananmen\ncontrols",
    "Tiananmen\ntourism",
    "Tiananmen\ncultural",
    "Tiananmen\nambiguous",
    "Tiananmen\nimplicit political",
    "Tiananmen\nexplicit political",
]


def load_data():
    """Load M75 results and return per-prompt records."""
    records = []
    with open(DATA_PATH) as f:
        for row in csv.DictReader(f):
            cat = row["category"]
            score = float(row["probe_score_L16"])
            refused = row.get("is_refusal", "False") == "True"
            # Map CSV category to display name
            if cat.startswith("control_"):
                display = "Non-Tiananmen\ncontrols"
            else:
                display = cat.replace("_", " ")
                display = f"Tiananmen\n{display}"
            records.append({
                "category": cat,
                "display": display,
                "score": score,
                "refused": refused,
                "prompt": row.get("prompt", ""),
            })
    return records


def load_position_comparison():
    """Load per-layer DLA for last-prompt vs first-generated token."""
    with open(POSITION_PATH) as f:
        data = json.load(f)
    layers, last_prompt, first_generated = [], [], []
    for row in data["layer_comparison"]:
        layers.append(row["layer"])
        last_prompt.append(row["total_last_prompt"])
        first_generated.append(row["total_first_meaningful"])
    return np.array(layers), np.array(last_prompt), np.array(first_generated)


def main():
    records = load_data()
    layers, last_p, first_g = load_position_comparison()

    rng = np.random.default_rng(0)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.5, 5.2), gridspec_kw={"width_ratios": [1.0, 1.25]}
    )

    # ── Panel A: prompt-time commitment ─────────────────────────────
    ax1.plot(
        layers, last_p, "-o", color="#d62728", markersize=3.5,
        linewidth=2, label="Last prompt token", zorder=3
    )
    ax1.plot(
        layers, first_g, "--s", color="#1f77b4", markersize=3.2,
        linewidth=1.5, alpha=0.85, label="First generated token", zorder=2
    )
    ax1.fill_between(layers, last_p, first_g, alpha=0.12, color="#999999")

    peak_idx = int(np.argmax(last_p))
    peak_layer = int(layers[peak_idx])
    peak_val = float(last_p[peak_idx])
    ax1.annotate(
        f"Peak routing: L{peak_layer}\n({peak_val:.1f})",
        xy=(peak_layer, peak_val), xytext=(peak_layer - 10, peak_val - 2.5),
        fontsize=8.8, color="#555555",
        arrowprops=dict(arrowstyle="->", color="#999999", lw=1),
    )

    max_diff = float(np.max(np.abs(last_p - first_g)))
    ax1.text(
        0.98, 0.05, f"Max difference: {max_diff:.4f}\n(signal already committed)",
        transform=ax1.transAxes, ha="right", va="bottom",
        fontsize=8.8, color="#555555", fontstyle="italic",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                  edgecolor="#cccccc", alpha=0.9)
    )

    ax1.set_xlabel("Layer", fontsize=10)
    ax1.set_ylabel("DLA routing signal (total)", fontsize=10)
    ax1.set_title("A. Routing Is Committed During Prompt Processing",
                  fontsize=10.5, fontweight="bold")
    ax1.legend(fontsize=8.8, loc="upper left")
    ax1.grid(True, alpha=0.2)
    ax1.set_xlim(-0.5, layers[-1] + 0.5)

    # Group by display category
    cat_to_idx = {name: i for i, name in enumerate(DISPLAY_NAMES)}

    # Collect data by category for means
    cat_scores = {name: [] for name in DISPLAY_NAMES}

    for r in records:
        idx = cat_to_idx.get(r["display"])
        if idx is None:
            continue
        cat_scores[r["display"]].append(r["score"])
        # Jitter for strip plot
        jitter = rng.uniform(-0.12, 0.12)
        if r["refused"]:
            ax2.scatter(idx + jitter, r["score"], marker="X", s=80,
                        c="#d62728", zorder=4, edgecolors="white",
                        linewidths=0.5)
        else:
            ax2.scatter(idx + jitter, r["score"], marker="o", s=60,
                        c="#2ca02c", zorder=3, edgecolors="white",
                        linewidths=0.5, alpha=0.85)

    # Category means as gray lines
    for name, idx in cat_to_idx.items():
        scores = cat_scores[name]
        if scores:
            mean = np.mean(scores)
            ax2.plot([idx - 0.3, idx + 0.3], [mean, mean], color="#888888",
                     linewidth=2, zorder=2)

    # Background bands for Tiananmen categories
    for i in range(1, 6):
        ax2.axvspan(i - 0.45, i + 0.45, alpha=0.06, color="#1a7a72",
                    zorder=0)

    # === KEY REVIEWER REQUEST: Annotate the two edge cases ===

    # Edge case 1: "controversial" (implicit, 8.19) scores HIGHER than
    # "in 1989" (explicit, 7.76) but is FACTUAL vs REFUSED
    ax2.annotate(
        '"controversial" → factual\n(score 8.19)',
        xy=(4, 8.19), xytext=(2.3, 9.3),
        fontsize=8.5, color="#2ca02c", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.3),
        ha="center",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                  edgecolor="#2ca02c", alpha=0.9))

    ax2.annotate(
        '"in 1989" → refused\n(score 7.76)',
        xy=(5, 7.76), xytext=(4.2, 6.8),
        fontsize=8.5, color="#d62728", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.3),
        ha="center",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                  edgecolor="#d62728", alpha=0.9))

    # Connecting annotation
    ax2.text(2.9, 8.0, "Same score range,\ndifferent routing →\nnot a threshold",
             fontsize=8.2, color="#555555", fontstyle="italic", ha="center",
             va="center",
             bbox=dict(boxstyle="round,pad=0.25", facecolor="#fff8e0",
                       edgecolor="#ccaa00", alpha=0.85))

    # "All prompts contain 'Tiananmen'" annotation
    ax2.annotate('All prompts contain "Tiananmen"',
                 xy=(3, 3.7), fontsize=9, color="#1a7a72",
                 fontstyle="italic", ha="center")

    # Arrow showing increasing political framing
    ax2.annotate("", xy=(5.4, 3.4), xytext=(1.0, 3.4),
                 arrowprops=dict(arrowstyle="->", color="#1a7a72", lw=1.5))
    ax2.text(3.2, 3.15, "increasing political framing",
             fontsize=8.7, color="#1a7a72", ha="center", fontstyle="italic")

    # Legend
    ax2.scatter([], [], marker="o", c="#2ca02c", s=60, label="Answered factually")
    ax2.scatter([], [], marker="X", c="#d62728", s=80, label="Refused")
    ax2.plot([], [], color="#888888", linewidth=2, label="Category mean")
    ax2.legend(fontsize=9, loc="upper left")

    ax2.set_xticks(range(len(DISPLAY_NAMES)))
    ax2.set_xticklabels(DISPLAY_NAMES, fontsize=9)
    ax2.set_ylabel("Political probe score at Layer 16\n(detection layer, before gate head)",
                   fontsize=10)
    ax2.set_ylim(3, 10)
    ax2.grid(True, alpha=0.15, axis="y")

    ax2.set_title(
        "B. Detection Is Contextual: Same Word, Different Routing\n"
        "Probe score correlates with framing; routing requires more than a threshold",
        fontsize=10.5, fontweight="bold")

    fig.suptitle("Routing Is Prompt-Time and Contextual",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"fig_contextual_detection.{ext}", dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_contextual_detection.png/pdf")


if __name__ == "__main__":
    main()
