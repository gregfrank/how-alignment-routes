#!/usr/bin/env python3
"""Generate Figure: Cipher contrast analysis scatter plot.

For each model, plots plaintext DLA (x) vs cipher DLA (y) for every attention head.
Heads on the diagonal are unaffected by cipher; heads pulled toward y=0 are
the content-dependent routing circuit.
"""

from __future__ import annotations

import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

FIG_DIR = REPO_ROOT / "figures" / "output"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = REPO_ROOT / "results/cipher_contrast"

models = [
    ('Gemma-2-2B', 'gemma2_2b',
     {(13,2): 'gate', (15,2): 'amp', (16,1): 'amp', (15,7): 'amp', (8,3): 'amp'}),
    ('Phi-4-mini', 'phi4_mini',
     {(13,7): 'gate', (16,13): 'amp', (26,9): 'amp', (29,18): 'amp'}),
    ('Qwen3-8B', 'qwen3_8b',
     {(17,17): 'gate', (22,7): 'amp', (23,2): 'amp', (22,4): 'amp', (22,5): 'amp', (22,6): 'amp'}),
]


def main():
    plt.rcParams.update({'font.size': 12})
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.0))

    for ax, (name, subdir, known) in zip(axes, models):
        path = DATA_DIR / subdir / "cipher_diagnostic_all_heads.csv"
        rows = list(csv.DictReader(open(path)))

        plain = np.array([float(r['mean_plain_dla']) for r in rows])
        cipher = np.array([float(r['mean_cipher_dla']) for r in rows])

        # Plot all heads as small gray dots
        ax.scatter(plain, cipher, s=4, alpha=0.25, color='#aaaaaa', zorder=1,
                   linewidths=0)

        # Diagonal
        lim = max(abs(plain).max(), abs(cipher).max()) * 1.15
        ax.plot([-lim, lim], [-lim, lim], '--', color='#cccccc', linewidth=0.8, zorder=0)
        ax.axhline(0, color='#e8e8e8', linewidth=0.5)
        ax.axvline(0, color='#e8e8e8', linewidth=0.5)

        # Collect labeled points to avoid overlaps
        labeled_points = []

        # Highlight known circuit heads with arrows from diagonal
        for r in rows:
            key = (int(r['layer']), int(r['head']))
            if key in known:
                role = known[key]
                color = '#d62728' if role == 'gate' else '#1f77b4'
                marker = '*' if role == 'gate' else 'o'
                size = 150 if role == 'gate' else 70
                p = float(r['mean_plain_dla'])
                c = float(r['mean_cipher_dla'])
                # Arrow from diagonal position (p, p) to actual position (p, c)
                if abs(p - c) > 0.05:
                    ax.annotate('', xy=(p, c), xytext=(p, p),
                               arrowprops=dict(arrowstyle='->', color=color,
                                              lw=1.5, alpha=0.5))
                ax.scatter([p], [c], s=size, color=color, marker=marker, zorder=4,
                          edgecolors='black', linewidths=0.6)
                labeled_points.append((p, c, f'L{key[0]}.H{key[1]}', color))

        # Highlight top 3 NEW heads with arrows
        rank = 0
        for r in rows:
            key = (int(r['layer']), int(r['head']))
            if key not in known:
                rank += 1
                if rank <= 3:
                    p = float(r['mean_plain_dla'])
                    c = float(r['mean_cipher_dla'])
                    if abs(p - c) > 0.05:
                        ax.annotate('', xy=(p, c), xytext=(p, p),
                                   arrowprops=dict(arrowstyle='->', color='#ff7f0e',
                                                  lw=1.5, alpha=0.5))
                    ax.scatter([p], [c], s=50, color='#ff7f0e', marker='D', zorder=3,
                              edgecolors='black', linewidths=0.5)
                    labeled_points.append((p, c, f'L{key[0]}.H{key[1]}', '#ff7f0e'))

        # Manual label offsets per model to avoid overlaps
        # Some crowded labels are omitted (set to None)
        LABEL_OFFSETS = {
            'Gemma-2-2B': {
                'L13.H2':  (7, -14),
                'L15.H2':  (-55, 10),
                'L16.H1':  (-55, -4),
                'L15.H7':  (7, -16),
                'L8.H3':   None,        # too close to L16.H1, skip
                'L14.H5':  (7, -14),
                'L12.H2':  (7, 8),
                'L14.H4':  (7, 8),
            },
            'Phi-4-mini': {
                'L13.H7':  (7, 10),
                'L16.H13': (7, -14),
                'L26.H9':  (7, 10),
                'L29.H18': (7, 10),
                'L16.H9':  (7, 8),
                'L16.H12': (7, -16),
                'L23.H9':  (-55, -8),
                'L16.H10': (7, -14),
            },
            'Qwen3-8B': {
                'L17.H17': (-55, 8),
                'L22.H7':  (7, -16),
                'L23.H2':  (-55, 4),
                'L22.H4':  None,        # overlaps L22.H7, skip
                'L22.H5':  (7, 12),
                'L22.H6':  (-55, -8),
                'L31.H3':  (7, -8),
                'L34.H15': (7, 10),
                'L35.H5':  None,        # too close to L22.H7, skip
            },
        }
        offsets = LABEL_OFFSETS.get(name, {})
        for px, py, label, color in labeled_points:
            offset = offsets.get(label, (7, 6))
            if offset is None:
                continue  # skip crowded labels
            dx, dy = offset
            ax.annotate(label, (px, py), fontsize=9, fontweight='bold',
                       color=color, xytext=(dx, dy), textcoords='offset points',
                       zorder=5)

        ax.set_xlabel('Plaintext DLA', fontsize=13)
        ax.set_ylabel('Cipher DLA', fontsize=13)
        ax.set_title(name, fontsize=14, fontweight='bold')
        # Don't force equal aspect — let y-axis expand to show movement
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim * 0.6, lim * 0.6)  # compress y range to zoom into the action
        ax.tick_params(labelsize=11)

    # Legend below plots
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#d62728',
               markeredgecolor='black', markersize=12, label='Known gate'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4',
               markeredgecolor='black', markersize=8, label='Known amplifier'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#ff7f0e',
               markeredgecolor='black', markersize=7, label='New (top 3 by contrast)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#aaaaaa',
               markersize=5, label='All other heads'),
        Line2D([0], [0], linestyle='--', color='#cccccc', label='y = x (unaffected)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=11,
              bbox_to_anchor=(0.5, -0.01), frameon=False)

    fig.tight_layout(rect=[0, 0.04, 1, 1])

    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"fig_cipher_diagnostic_scatter.{ext}", dpi=250,
                    bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_cipher_diagnostic_scatter.png/pdf")


if __name__ == "__main__":
    main()
