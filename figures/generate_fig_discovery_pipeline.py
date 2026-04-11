#!/usr/bin/env python3
"""
Generate Figure: Three-Step Discovery Pipeline for finding the gate head.

Three panels:
  1. Per-head DLA heatmap (layer × head) — screening step
  2. Head-level ablation bar chart — necessity ranking
  3. Interchange test scatter — necessity vs sufficiency identifies gate
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

DLA_PATH = REPO_ROOT / "results" / "discovery_pipeline" / "headDLA" / "head_summary.csv"
ABLATION_PATH = REPO_ROOT / "results" / "discovery_pipeline" / "headablate" / "head_ablate_summary.csv"
INTERCHANGE_PATH = REPO_ROOT / "results" / "discovery_pipeline" / "interchange" / "head_interchange_summary.csv"


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


def load_ablation(path, top_n=10):
    """Load per-head ablation results, return top N by absolute effect."""
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            baseline = float(row["mean_baseline_delta"])
            reduction = float(row["mean_nll_reduction"])
            pct = (reduction / baseline) * 100
            rows.append({
                "label": f"L{row['layer']}.H{row['head']}",
                "layer": int(row["layer"]),
                "head": int(row["head"]),
                "pct": pct,
                "reduction": reduction,
            })
    # Sort by absolute pct descending
    rows.sort(key=lambda r: abs(r["pct"]), reverse=True)
    return rows[:top_n]


def load_interchange(path):
    """Load interchange test results."""
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            baseline = float(row["mean_baseline_delta"])
            necessity = (float(row["mean_ctrl_to_ccp_reduction"]) / baseline) * 100
            sufficiency = (float(row["mean_ccp_to_ctrl_increase"]) / baseline) * 100
            rows.append({
                "label": f"L{row['layer']}.H{row['head']}",
                "layer": int(row["layer"]),
                "head": int(row["head"]),
                "necessity": necessity,
                "sufficiency": sufficiency,
            })
    return rows


def main():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5),
                                         gridspec_kw={"width_ratios": [1, 1.1, 1.2]})

    # ── Panel 1: DLA Heatmap ──────────────────────────────────────────
    mat = load_heatmap(DLA_PATH)
    n_layers, n_heads = mat.shape
    vmax = np.percentile(mat.ravel(), 97)

    im = ax1.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax,
                    extent=[0, n_heads, n_layers, 0], interpolation="nearest")

    # Mark top-1 DLA head (NOT the gate — gate is only found in step 3)
    top_idx = np.unravel_index(np.argmax(mat), mat.shape)
    ax1.plot(top_idx[1] + 0.5, top_idx[0] + 0.5, "k*", markersize=12,
             markeredgecolor="white", markeredgewidth=0.8, zorder=5)
    ax1.annotate(f"L{top_idx[0]}.H{top_idx[1]}\n(top DLA)",
                 xy=(top_idx[1] + 0.5, top_idx[0] + 0.5),
                 xytext=(top_idx[1] - 8, top_idx[0] - 18),
                 fontsize=12.8, fontweight="bold", color="black",
                 arrowprops=dict(arrowstyle="->", color="black", lw=1),
                 ha="center")

    # Mark second-highest cluster (L23 area)
    # Find second-highest absolute DLA head
    mat_copy = mat.copy()
    mat_copy[top_idx] = 0
    second_idx = np.unravel_index(np.argmax(mat_copy), mat_copy.shape)
    ax1.plot(second_idx[1] + 0.5, second_idx[0] + 0.5, "k*", markersize=8,
             markeredgecolor="white", markeredgewidth=0.5, zorder=5)
    ax1.annotate(f"L{second_idx[0]}.H{second_idx[1]}",
                 xy=(second_idx[1] + 0.5, second_idx[0] + 0.5),
                 xytext=(second_idx[1] + 5, second_idx[0] + 1),
                 fontsize=10.5, color="black",
                 arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
                 ha="left")

    cb = fig.colorbar(im, ax=ax1, shrink=0.85, pad=0.02)
    cb.set_label("|delta contribution|", fontsize=12.8)
    cb.ax.tick_params(labelsize=8.5)

    ax1.set_xlabel("Head index", fontsize=12.8)
    ax1.set_ylabel("Layer", fontsize=12.8)
    ax1.set_title("Step 1: Per-Head DLA Screening\n(bright = strong routing contribution)",
                  fontsize=12.2, fontweight="bold")

    # ── Panel 2: Ablation bar chart ───────────────────────────────────
    ablation_rows = load_ablation(ABLATION_PATH, top_n=10)
    labels = [r["label"] for r in ablation_rows]
    pcts = [r["pct"] for r in ablation_rows]

    colors = []
    for r in ablation_rows:
        if r["layer"] == 17 and r["head"] == 17:
            colors.append("#d62728")  # Gate head highlighted
        else:
            colors.append("#1f77b4")

    y_pos = np.arange(len(labels))
    ax2.barh(y_pos, pcts, color=colors, alpha=0.85, edgecolor="white",
             linewidth=0.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=10.5)
    ax2.invert_yaxis()
    ax2.set_xlabel("Ablation effect (% of baseline)", fontsize=12.8)
    ax2.set_title("Step 2: Head-Level Ablation\n(which heads are necessary?)",
                  fontsize=12.2, fontweight="bold")

    # Annotate the gate candidate
    gate_idx = next(i for i, r in enumerate(ablation_rows)
                    if r["layer"] == 17 and r["head"] == 17)
    gate_pct = pcts[gate_idx]
    ax2.annotate("Gate candidate\n(modest ablation\nbut unique sufficiency)",
                 xy=(gate_pct + 0.15, gate_idx),
                 xytext=(gate_pct + 2.5, gate_idx + 1.5),
                 fontsize=10.5, color="#d62728",
                 arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
                 ha="left", va="center")

    # ── Panel 3: Interchange scatter ──────────────────────────────────
    interchange_rows = load_interchange(INTERCHANGE_PATH)

    for r in interchange_rows:
        if r["layer"] == 17 and r["head"] == 17:
            ax3.scatter(r["necessity"], r["sufficiency"], s=120, c="#d62728",
                        marker="*", zorder=5, edgecolors="black", linewidths=0.5)
        else:
            ax3.scatter(r["necessity"], r["sufficiency"], s=60, c="#1f77b4",
                        zorder=3, edgecolors="white", linewidths=0.5)

    # Label all points — with adjusted positions to avoid axis clashes
    # Pre-compute custom offsets for specific heads to avoid overlap
    label_offsets = {
        "L17.H19": (0.4, -0.03),   # far left — push label well right of point
        "L23.H1":  (0.25, 0.06),
        "L22.H6":  (0.25, -0.04),
        "L22.H5":  (0.30, 0.08),
        "L22.H7":  (0.34, -0.08),
        "L23.H30": (0.22, 0.04),
        "L23.H2":  (-0.55, -0.06),
        "L22.H4":  (0.24, 0.10),
        "L20.H14": (-0.62, 0.02),
    }

    for r in interchange_rows:
        label = r["label"]
        x, y = r["necessity"], r["sufficiency"]

        # Special positioning for gate head
        if r["layer"] == 17 and r["head"] == 17:
            continue  # Labeled separately below

        dx, dy = label_offsets.get(label, (0.2, 0.02))

        ax3.annotate(label, xy=(x, y), xytext=(x + dx, y + dy),
                     fontsize=11.2, color="#333333", ha="left")

    # Gate head label — positioned to the right
    gate = next(r for r in interchange_rows if r["layer"] == 17 and r["head"] == 17)
    ax3.annotate("GATE\n(necessary +\nsufficient)",
                 xy=(gate["necessity"], gate["sufficiency"]),
                 xytext=(-1.0, 0.45),
                 fontsize=11.2, fontweight="bold", color="#d62728",
                 arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
                 ha="center",
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="#fff0f0",
                           edgecolor="#d62728", alpha=0.9))

    # Amplifiers label for the cluster in positive necessity, low sufficiency
    ax3.text(0.97, 0.08, "AMPLIFIERS\n(necessary only)",
             transform=ax3.transAxes, fontsize=12.2, color="#d62728",
             fontstyle="italic", ha="right", va="bottom",
             bbox=dict(boxstyle="round,pad=0.2", facecolor="#fff8f8",
                       edgecolor="#ffcccc", alpha=0.9))

    ax3.set_xlabel("Necessity (% reduction when swapped out)", fontsize=12.8)
    ax3.set_ylabel("Sufficiency (% increase when swapped in)", fontsize=12.8)
    ax3.set_title("Step 3: Interchange Test\n(L17.H17 has strongest combined necessity + sufficiency)",
                  fontsize=11.5, fontweight="bold")
    ax3.axhline(0, color="#cccccc", linewidth=0.8, zorder=0)
    ax3.axvline(0, color="#cccccc", linewidth=0.8, zorder=0)
    ax3.grid(True, alpha=0.15)

    # ── Suptitle ──────────────────────────────────────────────────────
    fig.suptitle("How We Found the Gate Head: Three-Step Discovery Pipeline",
                 fontsize=14.5, fontweight="bold", y=1.02)
    fig.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"fig_discovery_pipeline.{ext}", dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_discovery_pipeline.png/pdf")


if __name__ == "__main__":
    main()
