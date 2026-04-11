#!/usr/bin/env python3
# NOTE: This script produces a supplementary/appendix figure that requires
# probe_data.pt files from political_probe.py runs. These are not included
# in the public release. Retained for methodological transparency.
"""
Generate Figure H: Routing Geometry — direction cosine between political
and safety (generic refusal) directions by layer across multiple models.

Saves to: figures/fig_h_routing_geometry.png

Usage:
    python generate_fig_h_routing_geometry.py
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# ── Model configs ────────────────────────────────────────────────────────
# (display_name, probe_dir, color, linestyle, linewidth, marker_size, alpha,
#  annotation, is_primary)
# Primary models (paradoxical pair): solid, thick, full opacity
# Secondary models: thinner, semi-transparent, smaller markers
MODELS = [
    (
        "Qwen3-8B (33% refusal)",
        str(REPO_ROOT / "results/political_probe"),
        "#1f77b4",   # blue
        "-",         # solid
        2.5,         # thick primary line
        4.5,         # marker size
        1.0,         # full opacity
        None,
        True,        # PRIMARY — paradoxical: high refusal despite moderate overlap
    ),
    (
        "Yi-1.5-9B (0% refusal, highest overlap)",
        str(REPO_ROOT / "results/political_probe_01_ai_yi_1.5_9b_chat"),
        "#d62728",   # red
        "-",         # solid (was dashed — now primary)
        2.5,         # thick primary line
        4.5,         # marker size
        1.0,         # full opacity
        "peak",      # mark the peak
        True,        # PRIMARY — paradoxical: highest overlap yet zero refusal
    ),
    (
        "Phi-4-mini (safety-only refusal)",
        str(REPO_ROOT / "results/political_probe_microsoft_phi_4_mini_instruct"),
        "#2ca02c",   # green
        "--",        # dashed
        1.2,         # thin secondary line
        2.5,         # small markers
        0.4,         # semi-transparent
        None,
        False,
    ),
    (
        "DeepSeek-R1",
        str(REPO_ROOT / "results/political_probe_deepseek_ai_deepseek_r1_distill_qwen_7b"),
        "#ff7f0e",   # orange
        "--",        # dashed
        1.2,
        2.5,
        0.4,
        None,
        False,
    ),
    (
        "Qwen3.5-9B (0% refusal, pure steering)",
        str(REPO_ROOT / "results/political_probe_qwen_qwen3.5_9b"),
        "#9467bd",   # purple
        ":",         # dotted
        1.2,
        2.5,
        0.4,
        None,
        False,
    ),
    (
        "Llama-3.2-3B (base model, no RLHF)",
        str(REPO_ROOT / "results/political_probe_meta_llama_llama_3.2_3b"),
        "#7f7f7f",   # gray
        "--",        # dashed
        1.2,
        2.5,
        0.4,
        "baseline",  # annotate as baseline
        False,
    ),
]

OUTPUT_PATH = REPO_ROOT / "figures" / "output" / "fig_h_routing_geometry.png"


def load_probe_data(probe_dir: str):
    """Load probe_data.pt, returning None on failure."""
    pt_file = Path(probe_dir) / "probe_data.pt"
    if not pt_file.exists():
        return None
    try:
        return torch.load(pt_file, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  WARNING: Failed to load {pt_file}: {e}")
        return None


def compute_layer_cosines(data: dict):
    """Return (normalized_layers, cosines) for a single model's probe data.

    normalized_layers are in [0, 1] (fraction of total layers).
    """
    pol_probe = data.get("political_probe", {})
    gen_probe = data.get("generic_probe", {})

    pol_layers = set(k for k in pol_probe if isinstance(k, int))
    gen_layers = set(k for k in gen_probe if isinstance(k, int))
    layers = sorted(pol_layers & gen_layers)

    if not layers:
        return None, None

    cosines = []
    valid_layers = []
    for l in layers:
        d_pol = pol_probe[l].get("caa_direction")
        d_gen = gen_probe[l].get("caa_direction")
        if d_pol is not None and d_gen is not None:
            cos = torch.nn.functional.cosine_similarity(
                d_pol.float().unsqueeze(0), d_gen.float().unsqueeze(0)
            ).item()
            cosines.append(cos)
            valid_layers.append(l)
        else:
            cosines.append(np.nan)
            valid_layers.append(l)

    max_layer = max(valid_layers)
    if max_layer == 0:
        norm_layers = np.array(valid_layers, dtype=float)
    else:
        norm_layers = np.array(valid_layers, dtype=float) / max_layer

    return norm_layers, np.array(cosines)


def main():
    print("Generating Figure H: Routing Geometry...")

    # ── Figure setup ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)

    # Sans-serif fonts
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

    # Track Yi peak for annotation
    yi_peak_x, yi_peak_y = None, None
    llama_final_x, llama_final_y = None, None

    # ── Plot each model ──────────────────────────────────────────────────
    for name, probe_dir, color, ls, lw, ms, alpha, annot_type, is_primary in MODELS:
        data = load_probe_data(probe_dir)
        if data is None:
            print(f"  WARNING: Skipping {name} — probe_data.pt not found at {probe_dir}")
            continue

        norm_layers, cosines = compute_layer_cosines(data)
        if norm_layers is None:
            print(f"  WARNING: Skipping {name} — no overlapping layers with caa_direction")
            continue

        # Primary models drawn on top (higher zorder)
        ax.plot(
            norm_layers, cosines,
            color=color, linestyle=ls, linewidth=lw,
            marker="o", markersize=ms, alpha=alpha,
            label=name,
            zorder=10 if is_primary else 5,
        )

        # Track annotation targets
        if annot_type == "peak":
            # Find the peak cosine value
            valid_mask = ~np.isnan(cosines)
            if valid_mask.any():
                peak_idx = np.nanargmax(cosines)
                yi_peak_x = norm_layers[peak_idx]
                yi_peak_y = cosines[peak_idx]

        if annot_type == "baseline":
            # Track final point for Llama annotation
            valid_mask = ~np.isnan(cosines)
            if valid_mask.any():
                last_valid = np.where(valid_mask)[0][-1]
                llama_final_x = norm_layers[last_valid]
                llama_final_y = cosines[last_valid]

    # ── Background shading: late layers region (0.7–1.0) ────────────────
    ax.axvspan(0.7, 1.02, alpha=0.06, color="#888888", zorder=0)
    ax.text(
        0.85, 0.90,
        "Late layers",
        fontsize=7.5, color="#999999", style="italic",
        ha="center", va="top",
        transform=ax.transData,
    )

    # ── Horizontal reference: y = 0 ─────────────────────────────────────
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)

    # ── Shaded band: "Early layers: orthogonal" ─────────────────────────
    ax.axhspan(-0.08, 0.08, alpha=0.06, color="green", zorder=0)
    ax.text(
        0.08, -0.10,
        "Early layers: orthogonal (independent detection)",
        fontsize=7.5, color="#2e7d32", style="italic",
        ha="left", va="top",
        transform=ax.transData,
    )

    # ── Annotation: convergence zone bracket ─────────────────────────────
    # Mid-layer convergence region (roughly 0.3–0.7 of normalized depth)
    bracket_y = 0.72
    ax.annotate(
        "",
        xy=(0.30, bracket_y), xytext=(0.70, bracket_y),
        arrowprops=dict(
            arrowstyle="<->", color="#cc6600", lw=1.5,
            connectionstyle="arc3,rad=0",
        ),
    )
    ax.text(
        0.50, bracket_y + 0.04,
        "Mid layers: routing geometry emerges",
        fontsize=8.5, color="#cc6600", fontweight="bold",
        ha="center", va="bottom",
    )

    # ── Annotation: Yi peak ──────────────────────────────────────────────
    if yi_peak_x is not None and yi_peak_y is not None:
        ax.annotate(
            "Highest overlap\nbut zero refusal",
            xy=(yi_peak_x, yi_peak_y),
            xytext=(min(yi_peak_x + 0.22, 0.92), yi_peak_y - 0.18),
            fontsize=7.5, color="#d62728", fontweight="bold",
            ha="left", va="top",
            arrowprops=dict(
                arrowstyle="->", color="#d62728", lw=1.2,
                connectionstyle="arc3,rad=0.3",
            ),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#d62728", alpha=0.92),
            zorder=15,
        )

    # ── Annotation: Llama baseline ───────────────────────────────────────
    if llama_final_x is not None and llama_final_y is not None:
        ax.annotate(
            "No instruction tuning\n=> no convergence",
            xy=(llama_final_x, llama_final_y),
            xytext=(0.08, -0.25),
            fontsize=7.5, color="#7f7f7f", fontweight="bold",
            ha="left", va="top",
            arrowprops=dict(
                arrowstyle="->", color="#7f7f7f", lw=1.2,
                connectionstyle="arc3,rad=-0.3",
            ),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#7f7f7f", alpha=0.92),
            zorder=15,
        )

    # ── Axes labels and title ────────────────────────────────────────────
    ax.set_xlabel("Normalized Layer Depth (fraction of total layers)", fontsize=11)
    ax.set_ylabel("Cosine Similarity (Political vs Safety Direction)", fontsize=11)
    ax.set_title(
        "Routing geometry emerges after detection and is not sufficient for refusal",
        fontsize=12, fontweight="bold", pad=12,
    )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.35, 0.95)

    ax.grid(True, alpha=0.15, linewidth=0.5)
    ax.tick_params(labelsize=9)

    # ── Legend ────────────────────────────────────────────────────────────
    ax.legend(
        fontsize=6, loc="lower right",
        framealpha=0.92, edgecolor="#ccc",
        ncol=1, handlelength=1.8, labelspacing=0.35,
        borderpad=0.4, handletextpad=0.5,
    )

    # ── Save ─────────────────────────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
