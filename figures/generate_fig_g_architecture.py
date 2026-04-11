#!/usr/bin/env python3
"""Generate Figure G: Staged Alignment Architecture.

Produces a publication-quality diagram showing the three-stage model:
  Stage 1 (Detection) -> Stage 2 (Routing) -> Stage 3 (Output)
with per-model routing differences and annotations.

Design: vertical flowchart with three horizontal bands, orthogonal arrows,
uniform box sizing, and compact side annotations.  Arrows run through
dedicated inter-band gutters so they never cross title text.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 9,
    "axes.linewidth": 0,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
BLUE_LIGHT  = "#d0e2f2"
BLUE_MED    = "#5b9bd5"
BLUE_DARK   = "#2e75b6"
BLUE_BAND   = "#f0f5fb"

AMBER_LIGHT = "#fce4b0"
AMBER_MED   = "#f4b942"
AMBER_DARK  = "#d48b0a"
AMBER_BAND  = "#fdf6e8"

GREEN_LIGHT = "#c8e6c8"
GREEN_MED   = "#70ad70"
GREEN_DARK  = "#3a7d3a"
GREEN_BAND  = "#f0f8f0"

GRAY_TEXT   = "#333333"
GRAY_ANNOT  = "#666666"
GRAY_FADED  = "#bbbbbb"

# ---------------------------------------------------------------------------
# Coordinate system  (x: 0-6,  y: 0-12)
#
# Layout from top to bottom:
#   DETECT band    : 9.6 - 11.8   (title at top, boxes in middle)
#   gutter 1       : 8.6 -  9.6   (detection->routing arrows jog here)
#   ROUTE band     : 5.0 -  8.6   (title at top, 4 routing boxes below)
#   gutter 2       : 3.4 -  5.0   (routing->output arrows jog here)
#   OUTPUT band    : 0.8 -  3.4   (title at top, boxes below, summary at bottom)
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 6, 7.5

DETECT_BAND = (9.6, 11.8)
GUTTER_1    = (8.6, 9.6)
ROUTE_BAND  = (5.0, 8.6)
GUTTER_2    = (3.4, 5.0)
OUTPUT_BAND = (0.8, 3.4)

X_MID = 3.0

# Uniform box sizes
BOX_W = 1.0
BOX_H = 0.42
ROUTE_BOX_W = 1.6
ROUTE_BOX_H = 0.42

# Detection boxes
DETECT_Y  = 10.2
DETECT_XS = [0.75, 2.15, 3.55, 4.95]

# Routing boxes (stacked vertically in the routing band, below title)
ROUTE_X = 2.95
ROUTE_YS = [7.35, 6.75, 6.15, 5.55]  # Refusal, Steering, No action, Safety

# Output boxes (same x positions as detection)
OUTPUT_Y  = 1.6
OUTPUT_XS = DETECT_XS[:]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def draw_box(ax, cx, cy, w, h, label, facecolor, edgecolor,
             fontsize=8, fontweight="normal", text_color="black",
             alpha=1.0, linestyle="-"):
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.03",
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=1.0, alpha=alpha, linestyle=linestyle, zorder=3,
    )
    ax.add_patch(box)
    ax.text(cx, cy, label, ha="center", va="center", fontsize=fontsize,
            fontweight=fontweight, color=text_color, zorder=4)
    return box


def ortho_arrow(ax, x0, y0, x1, y1, mid_y, color="#555", lw=1.0,
                linestyle="-", alpha=1.0):
    """Orthogonal arrow: vertical down, horizontal jog, vertical down."""
    # Skip zero-length segments gracefully
    if abs(x0 - x1) < 0.001:
        # Straight vertical — just one arrow
        ax.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle="-|>", color=color,
            linewidth=lw, mutation_scale=10, zorder=2,
            linestyle=linestyle, alpha=alpha))
        return

    ax.plot([x0, x0], [y0, mid_y], color=color, lw=lw,
            linestyle=linestyle, alpha=alpha, zorder=2, solid_capstyle="butt")
    ax.plot([x0, x1], [mid_y, mid_y], color=color, lw=lw,
            linestyle=linestyle, alpha=alpha, zorder=2, solid_capstyle="butt")
    ax.add_patch(FancyArrowPatch(
        (x1, mid_y), (x1, y1), arrowstyle="-|>", color=color,
        linewidth=lw, mutation_scale=10, zorder=2,
        linestyle=linestyle, alpha=alpha))


def straight_arrow(ax, x0, y0, x1, y1, **kw):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>",
        color=kw.get("color", "#555"), linewidth=kw.get("lw", 1.0),
        mutation_scale=10, zorder=2,
        linestyle=kw.get("linestyle", "-"), alpha=kw.get("alpha", 1.0)))


def badge(ax, x, y, text, color=AMBER_DARK, bg="#fff7e0", ec=AMBER_MED):
    ax.text(x, y, text, ha="center", va="center", fontsize=6.5,
            fontweight="bold", color=color,
            bbox=dict(boxstyle="round,pad=0.15", fc=bg, ec=ec, lw=0.6),
            zorder=5)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, 6)
ax.set_ylim(0.7, 12)
ax.axis("off")

# === Background bands =====================================================
for (y_lo, y_hi), bc in [
    (DETECT_BAND, BLUE_BAND),
    (ROUTE_BAND, AMBER_BAND),
    (OUTPUT_BAND, GREEN_BAND),
]:
    ax.add_patch(plt.Rectangle(
        (0.05, y_lo), 5.9, y_hi - y_lo,
        facecolor=bc, edgecolor="none", zorder=0, alpha=0.55))

# Subtle dividers at band boundaries
for y in [DETECT_BAND[0], ROUTE_BAND[1], ROUTE_BAND[0], OUTPUT_BAND[1]]:
    ax.axhline(y, xmin=0.01, xmax=0.99, color="#d0d0d0", lw=0.6, zorder=1)

# === STAGE 1: DETECTION ===================================================
ax.text(X_MID, 11.35, "STAGE 1: DETECTION", fontsize=11,
        fontweight="bold", color=BLUE_DARK, ha="center")
ax.text(X_MID, 11.0, "Layers 0\u20138  \u00b7  Topic classification",
        fontsize=7.5, color=GRAY_ANNOT, ha="center")

for x, lab in zip(DETECT_XS, ["Political", "Safety", "Food / Tech", "Sci / History"]):
    draw_box(ax, x, DETECT_Y, BOX_W, BOX_H, lab,
             facecolor=BLUE_LIGHT, edgecolor=BLUE_MED, fontsize=8, fontweight="bold")

ax.text(X_MID, DETECT_Y - 0.38, "All 100% separable",
        ha="center", fontsize=7.5, fontstyle="italic", color=GRAY_ANNOT)

# === STAGE 2: ROUTING =====================================================
route_title_bg = dict(boxstyle="round,pad=0.15", fc=AMBER_BAND, ec="none", alpha=0.9)
ax.text(X_MID, 8.25, "STAGE 2: ROUTING", fontsize=11,
        fontweight="bold", color=AMBER_DARK, ha="center", zorder=6,
        bbox=route_title_bg)
ax.text(X_MID, 7.92, "Layers 12\u201320  \u00b7  Policy selection",
        fontsize=7.5, color=GRAY_ANNOT, ha="center", zorder=6,
        bbox=dict(boxstyle="round,pad=0.1", fc=AMBER_BAND, ec="none", alpha=0.9))

route_labels = [
    "Political \u2192 Refusal",
    "Political \u2192 Steering",
    "Political \u2192 No action",
    "Safety \u2192 Refusal",
]
route_models = ["Qwen3-8B", "Qwen3.5", "GLM-4", "all models"]
route_mcolors = [
    (AMBER_DARK, "#fff7e0", AMBER_MED),
    (AMBER_DARK, "#fff7e0", AMBER_MED),
    ("#777777", "#f0f0f0", "#aaaaaa"),
    (AMBER_DARK, "#fff7e0", AMBER_MED),
]

for lab, ry, mdl, (mc, mbg, mec) in zip(
    route_labels, ROUTE_YS, route_models, route_mcolors
):
    fc = AMBER_LIGHT if "No action" not in lab else "#eeeeee"
    ec = AMBER_MED if "No action" not in lab else "#aaaaaa"
    draw_box(ax, ROUTE_X, ry, ROUTE_BOX_W, ROUTE_BOX_H, lab,
             facecolor=fc, edgecolor=ec, fontsize=7.5)
    badge(ax, ROUTE_X + ROUTE_BOX_W / 2 + 0.55, ry, mdl,
          color=mc, bg=mbg, ec=mec)

# Annotations — compact list on the left margin
for i, line in enumerate([
    "\u2022 Language-conditioned",
    "\u2022 Lab-specific",
    "\u2022 Learned & ablatable",
]):
    ax.text(0.2, 6.8 - i * 0.28, line,
            fontsize=6.5, color=GRAY_ANNOT, ha="left")

# === STAGE 3: OUTPUT ======================================================
# Title has a subtle background so arrows pass behind it cleanly.
title_bg = dict(boxstyle="round,pad=0.15", fc=GREEN_BAND, ec="none", alpha=0.9)
ax.text(X_MID, 3.10, "STAGE 3: OUTPUT", fontsize=11,
        fontweight="bold", color=GREEN_DARK, ha="center", zorder=6,
        bbox=title_bg)
ax.text(X_MID, 2.77, "Layers 20+  \u00b7  Response generation",
        fontsize=7.5, color=GRAY_ANNOT, ha="center", zorder=6,
        bbox=dict(boxstyle="round,pad=0.1", fc=GREEN_BAND, ec="none", alpha=0.9))

for x, lab in zip(OUTPUT_XS, [
    "Refusal", "Controlled\nCompliance", "Factual\nComply", "Safety\nRefusal"
]):
    draw_box(ax, x, OUTPUT_Y, BOX_W, BOX_H, lab,
             facecolor=GREEN_LIGHT, edgecolor=GREEN_MED,
             fontsize=7.5, fontweight="bold")

ax.text(X_MID, 1.05,
        "Same detection, different routing \u2192 different output",
        ha="center", fontsize=8, fontstyle="italic",
        fontweight="bold", color=GRAY_TEXT)

# === ARROWS: Detection -> Routing =========================================
# Arrows run through gutter 1 (y 8.6-9.6).
# Political (x=1.0) fans to three routing boxes; Safety (x=2.3) to one.
# Each arrow: drop from detect box bottom into gutter, jog horizontally
# to the routing-box x, drop into routing box top.

det_bottom = DETECT_Y - BOX_H / 2
rte_top = lambda i: ROUTE_YS[i] + ROUTE_BOX_H / 2

# Stagger horizontal runs inside gutter 1 (9.6 top -> 8.6 bottom)
gutter1_ys = [9.35, 9.15, 8.95, 8.75]

# Political -> Refusal, Steering, No action (indices 0,1,2)
# Slight x offsets on entry so lines don't merge at ROUTE_X
pol_entry_offsets = [-0.20, 0.0, 0.20]
for i in range(3):
    ortho_arrow(ax, DETECT_XS[0], det_bottom,
                ROUTE_X + pol_entry_offsets[i], rte_top(i),
                mid_y=gutter1_ys[i], color=BLUE_MED, lw=0.9)

# Safety -> Safety Refusal (index 3)
ortho_arrow(ax, DETECT_XS[1], det_bottom,
            ROUTE_X, rte_top(3),
            mid_y=gutter1_ys[3], color=BLUE_MED, lw=0.9)

# Food/Tech and Sci/History -> dashed stubs
for dx in DETECT_XS[2:]:
    straight_arrow(ax, dx, det_bottom, dx, det_bottom - 0.55,
                   color=GRAY_FADED, lw=0.8, linestyle="--", alpha=0.5)
    ax.text(dx, det_bottom - 0.70, "no routing",
            ha="center", va="top", fontsize=6, color=GRAY_FADED,
            fontstyle="italic")

# === ARROWS: Routing -> Output ============================================
# Arrows run through gutter 2 (y 3.8-5.0).
rte_bottom = lambda i: ROUTE_YS[i] - ROUTE_BOX_H / 2
out_top = OUTPUT_Y + BOX_H / 2

# Stagger horizontal runs inside gutter 2 (5.0 top -> 3.4 bottom)
gutter2_ys = [4.75, 4.45, 4.15, 3.85]

# Slight x offsets on departure from routing box bottom
rte_dep_offsets = [-0.25, -0.08, 0.08, 0.25]

# 0: Political->Refusal  -> Refusal output
ortho_arrow(ax, ROUTE_X + rte_dep_offsets[0], rte_bottom(0),
            OUTPUT_XS[0], out_top,
            mid_y=gutter2_ys[0], color=AMBER_MED, lw=0.9)

# 1: Political->Steering -> Controlled Compliance
ortho_arrow(ax, ROUTE_X + rte_dep_offsets[1], rte_bottom(1),
            OUTPUT_XS[1], out_top,
            mid_y=gutter2_ys[1], color=AMBER_MED, lw=0.9)

# 2: Political->Nothing  -> Factual Comply (dashed)
ortho_arrow(ax, ROUTE_X + rte_dep_offsets[2], rte_bottom(2),
            OUTPUT_XS[2], out_top,
            mid_y=gutter2_ys[2], color="#aaaaaa", lw=0.9, linestyle="--")

# 3: Safety->Refusal     -> Safety Refusal
ortho_arrow(ax, ROUTE_X + rte_dep_offsets[3], rte_bottom(3),
            OUTPUT_XS[3], out_top,
            mid_y=gutter2_ys[3], color=AMBER_MED, lw=0.9)

# === Save =================================================================
out_path = REPO_ROOT / "figures" / "output" / "fig_g_staged_architecture.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.2)
plt.close(fig)
print(f"Saved: {out_path}")
