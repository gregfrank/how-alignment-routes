#!/usr/bin/env python3
"""Generate Figure: Dose-response at n=120 with per-category breakdown.

Shows both aggregate and category-level dose-response curves for
Qwen3-8B on the v2 political corpus.
"""

from __future__ import annotations

import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

FIG_DIR = REPO_ROOT / "figures" / "output"
FIG_DIR.mkdir(parents=True, exist_ok=True)
# Use M93 re-judged data (Llama 3.1 replaces Mistral in panel)
JUDGMENTS = REPO_ROOT / "results/dose_response/judgments_merged.csv"
# Fallback to original: REPO_ROOT / "results/m83_deep_remediation/qwen_dose_response_v2/judgments/judgments_checkpoint.csv"

CATEGORIES = [
    'Tiananmen Square', 'Tibet', 'Xinjiang / Uyghurs', 'Xi Jinping / CCP Governance',
    'Hong Kong', 'COVID-19 Origins', 'Taiwan', 'Falun Gong',
    'Cultural Revolution / Mao', 'Great Firewall / Censorship', 'South China Sea',
    'Internal CCP Politics', 'One-Child Policy', 'Surveillance State', 'Labor Rights',
]

ALPHAS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 50]


def main():
    rows = list(csv.DictReader(JUDGMENTS.open()))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5),
                              gridspec_kw={"width_ratios": [1, 1.2, 1]})

    # === Panel A: Tiananmen attenuation (the clean sigmoid) ===
    ax1 = axes[0]
    ref_rates = []
    for alpha in ALPHAS:
        attn = [r for r in rows if r['experiment'] == 'attenuation' and r['alpha'] == str(alpha)]
        # Tiananmen = first 8 prompts per alpha
        tian = attn[:8]
        ref = sum(1 for r in tian if r.get('new_majority_vote', r.get('majority_vote')) == 'REFUSAL') / len(tian) * 100
        ref_rates.append(ref)

    ax1.plot(ALPHAS, ref_rates, "-o", color="#d62728", linewidth=2.5, markersize=7)
    ax1.fill_between(ALPHAS, ref_rates, alpha=0.1, color="#d62728")
    ax1.set_xlabel("Attenuation strength (alpha)", fontsize=11)
    ax1.set_ylabel("Refusal rate (%)", fontsize=11)
    ax1.set_title("Tiananmen Prompts (n=8)\nClean sigmoid attenuation", fontsize=12, fontweight="bold")
    ax1.set_ylim(-5, 105)
    ax1.grid(True, alpha=0.2)

    # === Panel B: Amplification by category ===
    ax2 = axes[1]

    # Select representative categories
    show_cats = [
        ("Internal CCP Politics", "#d62728", "-o"),
        ("Xinjiang / Uyghurs", "#ff7f0e", "-s"),
        ("Great Firewall", "#9467bd", "-^"),
        ("Tibet", "#2ca02c", "-D"),
        ("Hong Kong", "#1f77b4", "-v"),
    ]

    for cat_name, color, fmt in show_cats:
        cat_idx = CATEGORIES.index(cat_name) if cat_name in CATEGORIES else \
                  next(i for i, c in enumerate(CATEGORIES) if cat_name in c)
        rates = []
        for alpha in ALPHAS:
            ampl = [r for r in rows if r['experiment'] == 'amplification' and r['alpha'] == str(alpha)]
            cat_rows = ampl[cat_idx*8:(cat_idx+1)*8]
            ref = sum(1 for r in cat_rows if r.get('new_majority_vote', r.get('majority_vote')) == 'REFUSAL') / max(len(cat_rows), 1) * 100
            rates.append(ref)
        short_name = cat_name.split(" / ")[0] if "/" in cat_name else cat_name.split(" (")[0]
        ax2.plot(ALPHAS, rates, fmt, color=color, linewidth=1.8, markersize=5, label=short_name)

    ax2.set_xlabel("Amplification strength (alpha)", fontsize=11)
    ax2.set_ylabel("Refusal rate (%)", fontsize=11)
    ax2.set_title("Amplification: Different Categories,\nDifferent Thresholds", fontsize=12, fontweight="bold")
    ax2.set_ylim(-5, 80)
    ax2.legend(fontsize=9.5, loc="upper left")
    ax2.grid(True, alpha=0.2)

    # === Panel C: Aggregate with STEERED breakdown ===
    ax3 = axes[2]

    for label_type, color, marker in [
        ("REFUSAL", "#d62728", "o"),
        ("STEERED", "#ff7f0e", "s"),
        ("FACTUAL", "#2ca02c", "^"),
    ]:
        rates = []
        for alpha in ALPHAS:
            ampl = [r for r in rows if r['experiment'] == 'amplification' and r['alpha'] == str(alpha)]
            rate = sum(1 for r in ampl if r.get('new_majority_vote', r.get('majority_vote')) == label_type) / max(len(ampl), 1) * 100
            rates.append(rate)
        ax3.plot(ALPHAS, rates, f"-{marker}", color=color, linewidth=1.8, markersize=5, label=label_type)

    ax3.set_xlabel("Amplification strength (alpha)", fontsize=11)
    ax3.set_ylabel("Output fraction (%)", fontsize=11)
    ax3.set_title("Aggregate Amplification:\nREFUSAL + STEERED Replace FACTUAL", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.2)

    fig.suptitle(
        "Bidirectional Dose-Response at n=120 (Qwen3-8B, v2 political corpus)\n"
        "3-judge majority: Gemini Flash + Llama 3.1 8B + GPT-4o-mini",
        fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"fig_dose_response_v3.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_dose_response_v3.png/pdf")


if __name__ == "__main__":
    main()
