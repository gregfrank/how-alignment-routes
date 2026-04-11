#!/usr/bin/env python3
"""Generate Figure: Bijection Detection Bypass Layer Profile.

Shows probe scores at each layer for plaintext harmful, cipher-encoded
harmful, and benign prompts. The cipher-encoded harmful tracks benign
through the detection layers, confirming the gate never fires.
"""

from __future__ import annotations

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

FIG_DIR = REPO_ROOT / "figures" / "output"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA = REPO_ROOT / "results/bijection/bijection_bypass_results.json"


def main():
    data = json.load(DATA.open())
    profiles = data["experiment_e"]

    n_layers = len(profiles["plaintext_harmful"][0])
    layers = np.arange(n_layers)

    plain = np.mean(profiles["plaintext_harmful"], axis=0)
    cipher = np.mean(profiles["cipher_harmful"], axis=0)
    benign = np.mean(profiles["benign"], axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    ax.plot(layers, plain, "-o", color="#d62728", markersize=3,
            linewidth=2, label="Plaintext harmful", zorder=3)
    ax.plot(layers, benign, "--s", color="#1f77b4", markersize=3,
            linewidth=1.5, alpha=0.8, label="Benign (control)", zorder=2)
    ax.plot(layers, cipher, "-D", color="#2ca02c", markersize=3,
            linewidth=2, label="Cipher-encoded harmful", zorder=3)

    # Mark the gate layer
    ax.axvline(17, color="#999999", linewidth=1, linestyle=":", alpha=0.5)
    ax.annotate("Gate (L17)", xy=(17, plain[17]),
                xytext=(20, plain[17] + 10),
                fontsize=8, color="#555555",
                arrowprops=dict(arrowstyle="->", color="#999999", lw=0.8))

    # Annotate the key finding
    ax.annotate(
        f"Cipher scores BELOW benign\nat gate layer\n(cipher={cipher[17]:.1f} vs benign={benign[17]:.1f})",
        xy=(17, cipher[17]),
        xytext=(5, cipher[17] + 40),
        fontsize=8, color="#2ca02c", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#2ca02c", alpha=0.9))

    ax.fill_between(layers, cipher, plain, alpha=0.08, color="#d62728")

    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel("Probe score (projection onto safety direction)", fontsize=10)
    ax.set_title(
        "Cipher Encoding Bypasses Detection Layers\n"
        "Cipher-encoded harmful content is invisible to the routing circuit",
        fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.15)
    ax.set_xlim(-0.5, n_layers - 0.5)

    fig.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"fig_bijection_profile.{ext}", dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_bijection_profile.png/pdf")


if __name__ == "__main__":
    main()
