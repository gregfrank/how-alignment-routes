#!/usr/bin/env python3
# NOTE: This script produces a supplementary/appendix figure from early
# discovery experiments (pre-redraft, small n). The required data is not
# included in the public release. The main paper uses later, larger-n
# versions of these analyses. Retained for methodological transparency.
"""
Generate prompt-time routing figure for §2.1.

Two panels showing routing is committed during prompt processing:
  A: Per-layer DLA at last-prompt-token vs first-generated-token (Qwen3-8B).
     Lines overlap almost perfectly — the decision is made before generation.
  B: KL divergence trajectory for 3 models (Qwen, GLM-4, DeepSeek) at
     last-prompt-token. GLM-4 peaks at 2.8 nats despite 0% refusal,
     proving non-refusing models still route.
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

DATA_DIR = REPO_ROOT / "results/m62_routing_localization"
FIG_DIR = REPO_ROOT / "figures" / "output"


def load_position_comparison(path):
    """Load per-layer DLA for last-prompt vs first-generated token."""
    with open(path) as f:
        data = json.load(f)
    layers = []
    last_prompt = []
    first_gen = []
    for entry in data["layer_comparison"]:
        layers.append(entry["layer"])
        last_prompt.append(entry["total_last_prompt"])
        first_gen.append(entry["total_first_meaningful"])
    return np.array(layers), np.array(last_prompt), np.array(first_gen)


def load_kl_trajectory(path):
    """Load per-layer symmetric KL divergence."""
    layers = []
    kl_means = []
    kl_lo = []
    kl_hi = []
    with open(path) as f:
        for row in csv.DictReader(f):
            layers.append(int(row["layer"]))
            kl_means.append(float(row["mean_kl_symmetric"]))
            kl_lo.append(float(row["kl_sym_ci_low"]))
            kl_hi.append(float(row["kl_sym_ci_high"]))
    return np.array(layers), np.array(kl_means), np.array(kl_lo), np.array(kl_hi)


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8),
                                    gridspec_kw={"width_ratios": [1, 1.2]})

    # ── Panel A: Position invariance (Qwen3-8B) ───────────────────────
    pos_path = DATA_DIR / "qwen_position_comparison.json"
    layers, last_p, first_g = load_position_comparison(pos_path)

    ax1.plot(layers, last_p, "-o", color="#d62728", markersize=3,
             linewidth=2, label="Last prompt token", zorder=3)
    ax1.plot(layers, first_g, "--s", color="#1f77b4", markersize=3,
             linewidth=1.5, label="First generated token", alpha=0.8,
             zorder=2)

    # Highlight the overlap
    ax1.fill_between(layers, last_p, first_g, alpha=0.15, color="#888888")

    # Annotate the peak
    peak_layer = int(layers[np.argmax(last_p)])
    peak_val = np.max(last_p)
    ax1.annotate(f"Peak: L{peak_layer} ({peak_val:.1f})",
                 xy=(peak_layer, peak_val),
                 xytext=(peak_layer - 15, peak_val - 2.0),
                 fontsize=8, color="#555555",
                 arrowprops=dict(arrowstyle="->", color="#999999", lw=1))

    # Annotate the maximum difference — bottom-right where signal is flat
    max_diff = np.max(np.abs(last_p - first_g))
    ax1.text(0.98, 0.05,
             f"Max difference: {max_diff:.4f} (lines overlap)",
             transform=ax1.transAxes, fontsize=8, ha="right", va="bottom",
             color="#555555", fontstyle="italic",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="#cccccc", alpha=0.9))

    ax1.set_xlabel("Layer", fontsize=10)
    ax1.set_ylabel("DLA routing signal (total)", fontsize=10)
    ax1.set_title("Routing Signal Is Identical Before and After Generation",
                  fontsize=10, fontweight="bold")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(True, alpha=0.2)
    ax1.set_xlim(-0.5, layers[-1] + 0.5)

    # ── Panel B: KL divergence for 3 models ────────────────────────────
    kl_models = [
        ("qwen_lastprompt_kl/kl_trajectory_summary.csv",
         "Qwen3-8B (33% refusal)", "#d62728", "-o"),
        ("glm4_lastprompt_kl/kl_trajectory_summary.csv",
         "GLM-4-9B (0% refusal)", "#2ca02c", "-^"),
        ("deepseek_lastprompt_kl/kl_trajectory_summary.csv",
         "DeepSeek-R1 (0% refusal)", "#9467bd", "-D"),
    ]

    for fname, label, color, fmt in kl_models:
        path = DATA_DIR / fname
        layers_kl, kl_mean, kl_lo, kl_hi = load_kl_trajectory(path)
        # Normalize layer index to fraction of total depth
        frac = layers_kl / layers_kl[-1]
        ax2.plot(frac, kl_mean, fmt, color=color, markersize=3,
                 linewidth=1.8, label=label, markevery=2)
        ax2.fill_between(frac, kl_lo, kl_hi, alpha=0.12, color=color)

    # Annotate GLM-4 peak
    glm_path = DATA_DIR / "glm4_lastprompt_kl/kl_trajectory_summary.csv"
    layers_glm, kl_glm, _, _ = load_kl_trajectory(glm_path)
    frac_glm = layers_glm / layers_glm[-1]
    peak_idx = np.argmax(kl_glm)
    ax2.annotate(
        f"GLM-4 peak: {kl_glm[peak_idx]:.1f} nats\n(routes without refusing)",
        xy=(frac_glm[peak_idx], kl_glm[peak_idx]),
        xytext=(0.25, kl_glm[peak_idx] + 3.5),
        fontsize=7.5, color="#2ca02c",
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.2),
        ha="center")

    ax2.set_xlabel("Relative depth (fraction of layers)", fontsize=10)
    ax2.set_ylabel("Symmetric KL divergence (nats)", fontsize=10)
    ax2.set_title("Non-Refusing Models Still Route:\nKL Divergence at Last Prompt Token",
                  fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(True, alpha=0.2)
    ax2.set_xlim(-0.02, 1.02)

    # ── Suptitle ───────────────────────────────────────────────────────
    fig.suptitle(
        "Routing Is Committed During Prompt Processing",
        fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"fig_prompt_time.{ext}", dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_prompt_time.png/pdf")


if __name__ == "__main__":
    main()
