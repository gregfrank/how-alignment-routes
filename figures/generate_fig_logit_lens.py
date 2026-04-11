#!/usr/bin/env python3
"""Generate logit lens figure: refusal token probability by layer, plaintext vs cipher."""

from __future__ import annotations

import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

FIG_DIR = REPO_ROOT / "figures" / "output"

def main():
    # Load data
    path = REPO_ROOT / "results/intent_separation/qwen3_8b/logit_lens_refusal.csv"
    layers, plain, cipher = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            layers.append(int(row["layer"]))
            plain.append(float(row["plain_refusal_pct"]) * 100)
            cipher.append(float(row["cipher_refusal_pct"]) * 100)

    fig, ax = plt.subplots(figsize=(8, 3.5))

    ax.plot(layers, plain, "-o", color="#d62728", linewidth=2.2, markersize=4,
            label="Plaintext harmful", alpha=0.9)
    ax.plot(layers, cipher, "-s", color="#ff7f0e", linewidth=2.2, markersize=4,
            label="Cipher harmful", alpha=0.9)

    # Mark gate layer
    gate_layer = 17
    ax.axvspan(gate_layer - 0.5, gate_layer + 0.5, alpha=0.12, color="#d62728")
    ax.annotate("Gate\n(L17)", xy=(gate_layer, 1.5), fontsize=9, fontweight="bold",
                color="#d62728", ha="center", va="bottom")

    # Mark where refusal appears
    ax.annotate("Refusal\nfirst appears", xy=(24, plain[24]),
                xytext=(27, plain[24] + 5), fontsize=9, fontweight="bold",
                color="#d62728",
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
                ha="center")

    ax.annotate("Refusal\nconsolidates", xy=(34.5, max(plain[34], plain[35])),
                xytext=(31, max(plain[34], plain[35]) + 4), fontsize=9,
                fontweight="bold", color="#d62728",
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
                ha="center")

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Refusal token probability (%)", fontsize=11)
    ax.set_title("Qwen3-8B Logit Lens ($n{=}120$): Refusal Never Materializes Under Cipher",
                fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.15)
    ax.set_ylim(-1, max(plain) + 8)

    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"fig_logit_lens.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_logit_lens.png/pdf")


if __name__ == "__main__":
    main()
