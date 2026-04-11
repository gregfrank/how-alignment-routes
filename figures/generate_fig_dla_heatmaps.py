#!/usr/bin/env python3
# NOTE: This script produces a supplementary/appendix figure from early
# discovery experiments (pre-redraft, small n). The required data is not
# included in the public release. The main paper uses later, larger-n
# versions of these analyses. Retained for methodological transparency.
"""
Generate appendix figure: Small-multiples DLA heatmaps across all models.

Shows per-head DLA contribution (layer × head) for 10 models from 7 labs.
Makes the breadth of the study visually tangible: "we screened every head
in every model."
"""

from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA_ROOT = REPO_ROOT / "results/pre_redraft/gpu/mechanistic"
FIG_DIR = REPO_ROOT / "figures" / "output"

# One file per distinct model, ordered by lab diversity
MODELS = [
    ("Qwen3-8B\n(Alibaba)", "m63_head_localization/qwen_headDLA_logitdiff/head_summary.csv", "political"),
    ("Phi-4-mini\n(Microsoft)", "m66_safety_domain/phi4_safety_headDLA/head_summary.csv", "safety"),
    ("Llama-3.2\n(Meta)", "m66_safety_domain/llama32_safety_headDLA/head_summary.csv", "safety"),
    ("Gemma-2\n(Google)", "m68_expanded_models/gemma2_safety_headDLA/head_summary.csv", "safety"),
    ("GLM-Z1\n(Zhipu)", "m68_expanded_models/glmz1_safety_headDLA/head_summary.csv", "safety"),
    ("Mistral-7B\n(Mistral AI)", "m68_expanded_models/mistral_safety_headDLA/head_summary.csv", "safety"),
    ("GLM-4\n(Zhipu)", "m63_head_localization/glm4_headDLA_logitdiff/head_summary.csv", "political"),
    ("DeepSeek-R1\n(DeepSeek)", "m66_safety_domain/deepseek_safety_headDLA/head_summary.csv", "safety"),
    ("Qwen2.5-7B\n(Alibaba)", "m64_circuit_evolution/qwen25_7b_headDLA/head_summary.csv", "political"),
    ("Qwen3.5-9B\n(Alibaba)", "m64_circuit_evolution/qwen35_9b_headDLA/head_summary.csv", "political"),
]


def load_heatmap(path):
    """Load per-head DLA into a (layers, heads) numpy array."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    head_cols = sorted(
        [c for c in rows[0].keys() if c.startswith("mean_delta_head_")],
        key=lambda c: int(c.split("_")[-1])
    )
    n_layers = len(rows)
    n_heads = len(head_cols)
    mat = np.zeros((n_layers, n_heads))
    for i, row in enumerate(rows):
        for j, col in enumerate(head_cols):
            mat[i, j] = abs(float(row[col]))
    return mat


def main():
    # 2 rows × 5 columns
    n_cols = 5
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 7))

    # Find global max for consistent color scale
    all_mats = []
    labels = []
    for name, rel_path, domain in MODELS:
        path = DATA_ROOT / rel_path
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping {name}")
            all_mats.append(None)
            labels.append(name)
            continue
        mat = load_heatmap(path)
        all_mats.append(mat)
        labels.append(name)

    # Use 95th percentile as vmax to avoid outlier domination
    all_vals = np.concatenate([m.ravel() for m in all_mats if m is not None])
    vmax = np.percentile(all_vals, 97)

    for idx, (mat, label) in enumerate(zip(all_mats, labels)):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        if mat is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="#999999")
            ax.set_title(label, fontsize=9, fontweight="bold")
            continue

        n_layers, n_heads = mat.shape
        # Normalize y-axis to % of depth
        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax,
                       extent=[0, n_heads, n_layers, 0],
                       interpolation="nearest")

        # Mark the top-1 head
        top_idx = np.unravel_index(np.argmax(mat), mat.shape)
        ax.plot(top_idx[1] + 0.5, top_idx[0] + 0.5, "w*", markersize=8,
                markeredgecolor="black", markeredgewidth=0.5, zorder=5)

        domain_tag = "pol" if "political" in MODELS[idx][2] else "safe"
        ax.set_title(f"{label}\n({n_layers}L×{n_heads}H, {domain_tag})",
                     fontsize=8, fontweight="bold")

        if col == 0:
            ax.set_ylabel("Layer", fontsize=9)
        if row == 1:
            ax.set_xlabel("Head", fontsize=9)

        ax.tick_params(labelsize=7)

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label("|DLA contribution| per head", fontsize=9)
    cb.ax.tick_params(labelsize=7)

    fig.suptitle(
        "Per-Head DLA Routing Signal: 10 Models from 7 Labs\n"
        "★ = top-1 head; sparse concentration visible in strong refusers",
        fontsize=12, fontweight="bold", y=1.02)
    fig.subplots_adjust(left=0.04, right=0.90, top=0.85, bottom=0.08,
                        wspace=0.25, hspace=0.50)

    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"fig_dla_heatmaps.{ext}", dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_dla_heatmaps.png/pdf")


if __name__ == "__main__":
    main()
