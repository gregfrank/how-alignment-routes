#!/usr/bin/env python3
"""
Generate Figure 1: high-level overview of alignment routing.

This is a schematic, not a computational graph. It emphasizes the paper's
main causal story with large text that remains readable after LaTeX scaling.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def rbox(ax, x, y, w, h, *, fc, ec, lw=1.6, rad=0.12, z=2, alpha=1.0):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.055,rounding_size={rad}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z, alpha=alpha))


def arrow(ax, x0, y0, x1, y1, *, color="#444", lw=2.0, ls="-", z=4):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, linestyle=ls,
                        shrinkA=2, shrinkB=2),
        zorder=z,
    )


def label(ax, x, y, text, *, size=14, color="#222", weight="normal",
          ha="center", va="center", style="normal", linespacing=1.15,
          bbox=None):
    ax.text(
        x, y, text, fontsize=size, color=color, fontweight=weight,
        ha=ha, va=va, fontstyle=style, linespacing=linespacing,
        bbox=bbox,
    )


def main():
    fig, ax = plt.subplots(figsize=(12.0, 5.15))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.15)
    ax.axis("off")

    c_detect = "#2aaea5"
    c_gate = "#df4f4f"
    c_amp = "#b99000"
    c_mlp = "#2b78a0"
    c_resid = "#4c4c4c"
    c_out = "#5f4b8b"

    # Background stage cards.
    rbox(ax, 0.15, 0.35, 3.05, 4.35, fc="#edf9f7", ec="#cbe8e4", lw=1.0, z=0)
    rbox(ax, 3.45, 0.35, 4.30, 4.35, fc="#fff7f7", ec="#efd3d3", lw=1.0, z=0)
    rbox(ax, 7.86, 0.35, 3.99, 4.35, fc="#f8f6fb", ec="#ddd4ec", lw=1.0, z=0)

    label(ax, 6.0, 4.98, "Alignment routing: detection becomes policy behavior",
          size=18, weight="bold")

    # Stage headers.
    label(ax, 1.68, 4.48, "DETECT", size=20, weight="bold", color=c_detect)
    label(ax, 5.62, 4.48, "ROUTE", size=20, weight="bold", color=c_gate)
    label(ax, 9.95, 4.48, "OUTPUT", size=20, weight="bold", color="#333")

    label(ax, 1.68, 4.08, "contextual signal forms\nbefore generation",
          size=11.5, color="#28766f", linespacing=1.15)
    label(ax, 5.62, 4.08, "attention entry point plus\nparallel residual pathways",
          size=11.5, color="#7a3333", linespacing=1.15)
    label(ax, 9.95, 4.08, "routing amplitude selects\na policy regime",
          size=11.5, color="#555", linespacing=1.15)

    # DETECT panel.
    rbox(ax, 0.42, 3.18, 2.52, 0.55, fc="white", ec="#a9c9c5", lw=1.3)
    label(ax, 1.68, 3.45, "Prompt content", size=14, weight="bold")

    rbox(ax, 0.42, 0.86, 2.52, 2.05, fc="#dff5f2", ec=c_detect, lw=2.2)
    label(ax, 1.68, 2.62, "Contextual\nrepresentation",
          size=14, color="#187d75", weight="bold", linespacing=1.05)
    label(ax, 1.68, 2.22, "Layer 15-16", size=11.2, color="#187d75")

    # Two examples with larger labels and simple signal bars.
    label(ax, 0.70, 1.87, "tourism prompt", size=11.4, color="#555", ha="left")
    ax.barh(1.87, 0.48, height=0.14, left=2.10, color=c_detect, alpha=0.35, zorder=3)
    label(ax, 2.72, 1.87, "weak", size=10.5, color="#28766f", ha="right")

    label(ax, 0.70, 1.39, "1989 prompt", size=11.4, color="#555", ha="left")
    ax.barh(1.39, 0.86, height=0.14, left=2.10, color="#168b82", alpha=0.95, zorder=3)
    label(ax, 2.72, 1.39, "strong", size=10.5, color="#187d75", weight="bold",
          ha="right")

    label(ax, 1.68, 1.04, "not keyword matching", size=12.5, color="#187d75",
          weight="bold")

    # ROUTE panel.
    rbox(ax, 3.68, 2.54, 1.68, 1.05, fc="#fff0f0", ec=c_gate, lw=2.2)
    label(ax, 4.52, 3.23, "Gate head", size=15.4, color=c_gate, weight="bold")
    label(ax, 4.52, 2.86, "reads detection\ntriggers routing", size=11.6,
          color="#333", linespacing=1.08)

    rbox(ax, 6.08, 2.54, 1.50, 1.05, fc="#fff9db", ec=c_amp, lw=2.2)
    label(ax, 6.83, 3.23, "Amplifiers", size=14.6, color=c_amp, weight="bold")
    label(ax, 6.83, 2.86, "boost policy\nsignal", size=11.6, color="#333",
          linespacing=1.08)

    arrow(ax, 5.36, 3.06, 6.08, 3.06, color=c_gate, lw=2.0)
    label(ax, 5.72, 3.35, "causal\nentry", size=8.8, color=c_gate,
          style="italic", linespacing=1.0,
          bbox=dict(facecolor="#fff7f7", edgecolor="none", pad=0.4))

    rbox(ax, 3.75, 0.99, 3.75, 1.05, fc="#e4f1f7", ec=c_mlp, lw=2.0)
    label(ax, 5.62, 1.66, "Parallel MLP pathways", size=15.0,
          color=c_mlp, weight="bold")
    label(ax, 5.62, 1.31, "carry additional topic-specific signal", size=11.6,
          color="#2b6a85")

    # Residual stream summary.
    rbox(ax, 7.92, 1.04, 1.25, 2.55, fc="white", ec=c_resid, lw=1.8)
    label(ax, 8.54, 2.82, r"$\Sigma$", size=28, color="#333", weight="bold")
    label(ax, 8.54, 2.26, "Residual\nstream", size=13.2, color="#333",
          weight="bold", linespacing=1.0)
    label(ax, 8.54, 1.60, "signals\naccumulate", size=10.5, color="#666",
          linespacing=1.08)

    # Cross-panel arrows.
    arrow(ax, 2.94, 2.14, 3.68, 3.06, color=c_gate, lw=2.0)
    arrow(ax, 2.94, 1.44, 3.75, 1.52, color=c_mlp, lw=2.0)
    arrow(ax, 7.58, 3.06, 7.92, 2.82, color=c_gate, lw=2.0)
    arrow(ax, 7.50, 1.52, 7.92, 1.56, color=c_mlp, lw=2.0)
    arrow(ax, 9.17, 2.32, 9.75, 2.32, color="#444", lw=2.3)

    # OUTPUT panel.
    output_rows = [
        ("REFUSAL", "#d62728", 3.19),
        ("EVASION", "#9467bd", 2.56),
        ("STEERED", "#caa82a", 1.93),
        ("FACTUAL", "#2ca02c", 1.30),
    ]
    for text, color, y in output_rows:
        rbox(ax, 9.75, y - 0.22, 1.55, 0.44, fc="white", ec=color, lw=1.8,
             rad=0.08)
        label(ax, 10.52, y, text, size=14.2, color=color, weight="bold")

    ax.annotate(
        "", xy=(11.42, 3.42), xytext=(11.42, 1.06),
        arrowprops=dict(arrowstyle="<->", color=c_out, lw=1.8),
        zorder=4,
    )
    label(ax, 11.50, 3.42, "more", size=9.2, color=c_out, style="italic",
          ha="left")
    label(ax, 11.50, 1.06, "less", size=9.2, color=c_out, style="italic",
          ha="left")
    label(ax, 10.47, 0.72, "same mechanism, different regimes", size=10.7,
          color="#555", style="italic")

    fig.tight_layout(pad=0.25)
    for ext in ["png", "pdf"]:
        fig.savefig(
            f"figures/output/fig_overview.{ext}",
            dpi=250,
            bbox_inches="tight",
        )
    plt.close(fig)
    print("Saved fig_overview.png/pdf")


if __name__ == "__main__":
    main()
