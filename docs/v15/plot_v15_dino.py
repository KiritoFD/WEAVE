"""V15 Group A — DINO-only style/content landscape.

Cross600 means per arm x noise level (t0 = 0.0 / 0.4 / 0.7), computed
from runs/v15/eval/W_*.json rows. X = dino_c (content preservation),
Y = dino_s (target-style similarity). Both are DINO-readout metrics —
this figure deliberately avoids CLIP/CSD/LPIPS which V15's probe eval
did not compute.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

LEVELS = [0.0, 0.4, 0.7]
LEVEL_COLORS = {0.0: '#4C78A8', 0.4: '#F58518', 0.7: '#54A24B'}
LEVEL_LABELS = {0.0: 't0=0.0 pure style', 0.4: 't0=0.4 heavy edit',
                0.7: 't0=0.7 light edit'}

# external anchors (cross600, same DINO readout family)
ANCHORS = [
    ('LD4 / V10 E4', 0.7718, 0.4401, '#182A42'),
    ('real paintings\n(self-sim)', 0.75, 0.6113, '#8A8F98'),
]


def load_cross600():
    rows = []
    for f in sorted((REPO / 'runs/v15/eval').glob('W_*.json')):
        arm = f.stem
        for r in json.loads(f.read_text(encoding='utf-8'))['rows']:
            if r['cross']:
                rows.append({'arm': arm, 'level': r['level'],
                             'dino_s': r['dino_s'], 'dino_c': r['dino_c'],
                             'top1': r['style_top1']})
    df = pd.DataFrame(rows)
    return df.groupby(['arm', 'level'], as_index=False).mean(numeric_only=True)


def arm_parts(arm: str):
    # W_dim96_conv_base -> (dim, arch)
    p = arm.split('_')
    return p[1], p[2]


def main():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 11,
        'svg.fonttype': 'none', 'pdf.fonttype': 42,
        'axes.spines.top': False, 'axes.spines.right': False,
    })
    ink, muted = '#182A42', '#62758B'
    bg = '#FCFDFE'

    df = load_cross600()
    arms = sorted(df.arm.unique())

    fig, ax = plt.subplots(figsize=(9.2, 6.4), dpi=160)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    # per-arm trajectories across levels (faint)
    for arm in arms:
        sub = df[df.arm == arm].sort_values('level')
        ax.plot(sub.dino_c, sub.dino_s, color='#C9D3DD', lw=1.0,
                zorder=2, solid_capstyle='round')

    # points: color = level, marker = arch
    for _, r in df.iterrows():
        dim, arch = arm_parts(r.arm)
        marker = 'o' if arch == 'conv' else 's'
        ax.scatter(r.dino_c, r.dino_s, s=46,
                   c=LEVEL_COLORS[r.level], marker=marker,
                   edgecolors='white', linewidths=.7,
                   alpha=.92, zorder=3)

    # highlight dim96 arms with labels at t0=0.7 endpoint
    for arch in ('conv_base', 'att_base', 'conv_where', 'att_where'):
        arm = f'W_dim96_{arch}'
        sub = df[(df.arm == arm)].sort_values('level')
        if len(sub):
            ax.annotate(arm.replace('W_dim96_', 'dim96·'),
                        xy=(sub.iloc[-1].dino_c, sub.iloc[-1].dino_s),
                        xytext=(8, 4), textcoords='offset points',
                        fontsize=8.5, color=ink)

    # anchors
    for name, x, y, c in ANCHORS:
        ax.scatter(x, y, s=120, marker='*', c=c, edgecolors='white',
                   linewidths=1.0, zorder=5)
        ax.annotate(name, xy=(x, y), xytext=(7, 6),
                    textcoords='offset points', fontsize=9.5,
                    color=c, weight='semibold')

    ax.axhline(0.6113, color='#8A8F98', ls=(0, (4, 3)), lw=1.0, alpha=.7)
    ax.text(0.245, 0.6175, 'real-painting self-sim dino_s = 0.611',
            fontsize=8.5, color=muted)

    ax.set_xlabel('DINO-C  (content preservation, ↑)')
    ax.set_ylabel('DINO-S  (target-style similarity, ↑)')
    ax.set_title('V15 Group A — style × content landscape (Cross600, DINO readout)\n'
                 '16 arms × 3 noise levels · style_top1 ≈ random (0.17–0.20) at every cell',
                 fontsize=13, weight='bold', color=ink, loc='left')
    ax.set_xlim(0.22, 0.80)
    ax.set_ylim(0.42, 0.60)

    handles = [
        Line2D([], [], marker='o', ls='none', ms=8,
               mfc=LEVEL_COLORS[l], mec='white',
               label=LEVEL_LABELS[l]) for l in LEVELS
    ] + [
        Line2D([], [], marker='o', ls='none', ms=8, mfc='#888', mec='white',
               label='conv'),
        Line2D([], [], marker='s', ls='none', ms=8, mfc='#888', mec='white',
               label='attention'),
        Line2D([], [], color='#C9D3DD', lw=1.2, label='per-arm t0 trajectory'),
    ]
    ax.legend(handles=handles, loc='lower right', frameon=False,
              fontsize=9.5, title='noise level / arch', title_fontsize=9.5)

    fig.text(.085, .045,
             'Source: runs/v15/eval/W_*.json rows (cross600 means). '
             'V15 probe eval computes DINO-S/DINO-C/style_top1 only — '
             'no CLIP/CSD/LPIPS, so this figure uses the DINO pair only. '
             't0 = fraction of source latent kept (0 = pure style gen).',
             fontsize=8.8, color=muted)

    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(HERE / f'style_content_dino_v15.{ext}',
                    bbox_inches='tight', pad_inches=.22, dpi=200)
    plt.close(fig)
    print('saved docs/v15/style_content_dino_v15.*')


if __name__ == '__main__':
    main()
