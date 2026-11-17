#!/usr/bin/env python3
"""Cross-metric visualisations for docs/metrics_750.csv.

All figures are rendered on a FIXED canvas: no bbox_inches='tight', no text
outside the figure, no footnotes hanging off the axes. Every caption/legend
lives inside the figure box, so the saved PNG is exactly the intended size.
(The v15 scripts this is modelled on used tight bbox + off-canvas fig.text,
which stretched zoom figures to >10k px with large white margins.)

Reads only docs/metrics_750.csv — zero recompute, zero GPU.

Outputs (default PNG only; --vector adds SVG+PDF):
  docs/figures_750/correlation_heatmap.png
  docs/figures_750/pairs_grid.png
  docs/figures_750/artfid_landscape.png
  docs/figures_750/artfid_ranking.png

Sign convention: metrics where LOWER is better are negated before ranking,
so a positive Spearman rho always means "ranks arms the same way about
quality". Raw axis values are never transformed.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import patheffects as pe
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[0]
SRC = REPO / 'docs' / 'performance' / 'metrics.csv'
OUT = REPO / 'docs' / 'performance' / 'figures'

INK, MUTED = '#182A42', '#62758B'
BLUE, GREY = '#356C8F', '#B9C3CE'
ORANGE, ORANGE_DARK = '#ED6547', '#AE3A29'
PURPLE, PURPLE_DARK = '#7A4FA3', '#4E2F6E'
BG, GRID = '#FCFDFE', '#E8EEF4'

# (column, axis label). Arrow = direction of "better".
M = [
    ('dino_s',         'DINO-S ↑'),
    ('clip_s',         'CLIP-S ↑'),
    ('style_affinity', 'Style Affinity ↑'),
    ('target_top1',    'CSD Top-1 ↑'),
    ('target_margin',  'CSD Margin ↑'),
    ('G',              'G specificity ↑'),
    ('dT',             'ΔT ↑'),
    ('dino_c',         'DINO-C ↑'),
    ('lpips',          'LPIPS ↓'),
    ('artfid',         'ArtFID ↓'),
    ('kid',            'KID ↓'),
    ('ms_per_img',     'ms/img ↓'),
]
LOWER = {'lpips', 'artfid', 'kid', 'ms_per_img'}   # negated for rho only
LABEL = dict(M)
COLS = [c for c, _ in M]

# 4 visual families, same palette language as docs/v15 plots.
CATS = [
    ('External',        BLUE),
    ('Historical WEAVE', '#8497AB'),
    ('V8 / V10 / V11B / WEAVE', ORANGE),
    ('V15',             PURPLE),
]
CAT_COLOR = dict(CATS)
CAT_OF = {
    'Baseline': 'External', 'Baseline (reproduced)': 'External',
    'Ours': 'Historical WEAVE',
    'V8': 'V8 / V10 / V11B / WEAVE', 'V10': 'V8 / V10 / V11B / WEAVE',
    'V11B': 'V8 / V10 / V11B / WEAVE', 'WEAVE': 'V8 / V10 / V11B / WEAVE',
    'WEAVE (D5 outputs)': 'V8 / V10 / V11B / WEAVE',
    'V15': 'V15',
}

PAIRS = [
    ('dino_s',   'dino_c',         'style–content trade-off'),
    ('lpips',    'dino_c',         'two content readouts'),
    ('artfid',   'kid',            'two distribution metrics'),
    ('artfid',   'dino_c',         'ArtFID follows content'),
    ('artfid',   'dino_s',         'ArtFID vs style: decoupled'),
    ('artfid',   'style_affinity', 'ArtFID vs affinity: decoupled'),
]


def setup():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10.5,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#B9C5D0', 'text.color': INK,
        'xtick.color': MUTED, 'ytick.color': MUTED,
        'svg.fonttype': 'none', 'pdf.fonttype': 42,
    })


def load():
    df = pd.read_csv(SRC)
    df['category'] = df.group.map(CAT_OF).fillna('External')
    return df


def quality(col, series):
    return -series if col in LOWER else series


def rho_n(df, a, b):
    """Quality-aligned Spearman rho + pair count (never fabricates values)."""
    m = df[a].notna() & df[b].notna()
    x, y = quality(a, df.loc[m, a]), quality(b, df.loc[m, b])
    n = int(m.sum())
    if n < 4:
        return np.nan, n
    return float(x.rank().corr(y.rank())), n


def save(fig, stem, vector):
    OUT.mkdir(parents=True, exist_ok=True)
    exts = ('png', 'svg', 'pdf') if vector else ('png',)
    for e in exts:
        fig.savefig(OUT / f'{stem}.{e}', dpi=180,
                    facecolor=fig.get_facecolor())
    plt.close(fig)
    print('saved', stem, exts)


def cat_handles():
    return [Line2D([], [], marker='o', ls='none', ms=8, mfc=c, mec='white',
                   mew=.8, label=k) for k, c in CATS]


# ---------------------------------------------------------------- 1. heatmap
def fig_heatmap(df, vector):
    k = len(COLS)
    R = np.full((k, k), np.nan)
    N = np.zeros((k, k), dtype=int)
    for i, a in enumerate(COLS):
        for j, b in enumerate(COLS):
            if a == b:
                continue
            r, n = rho_n(df, a, b)
            R[i, j] = r
            N[i, j] = n

    fig = plt.figure(figsize=(11.2, 9.4), facecolor=BG)
    ax = fig.add_axes([.135, .155, .70, .72], facecolor=BG)
    cm = plt.get_cmap('RdBu')
    norm = Normalize(vmin=-1, vmax=1)
    for i in range(k):
        for j in range(k):
            if i == j:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fc='#F1F4F8',
                                           ec='white', lw=1.4))
                ax.text(j + .5, i + .5, '—', ha='center', va='center',
                        color='#9AA8B8', fontsize=11)
                continue
            r = R[i, j]
            fc = cm(norm(r)) if np.isfinite(r) else '#F4F6F9'
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fc=fc,
                                       ec='white', lw=1.4))
            if j > i:                     # upper triangle: rho
                txt = f'{r:+.2f}' if np.isfinite(r) else 'n/a'
                col = 'white' if np.isfinite(r) and abs(r) > .62 else INK
                ax.text(j + .5, i + .5, txt, ha='center', va='center',
                        fontsize=9.2, color=col, weight='bold')
            else:                         # lower triangle: sample count
                ax.text(j + .5, i + .5, f'n={N[i, j]}', ha='center',
                        va='center', fontsize=8.6, color=MUTED)

    ax.set_xlim(0, k)
    ax.set_ylim(k, 0)
    ax.set_xticks(np.arange(k) + .5)
    ax.set_yticks(np.arange(k) + .5)
    ax.set_xticklabels([LABEL[c] for c in COLS], rotation=42, ha='right',
                       fontsize=9.2)
    ax.set_yticklabels([LABEL[c] for c in COLS], fontsize=9.2)
    ax.tick_params(length=0, pad=5)
    for s in ax.spines.values():
        s.set_visible(False)

    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cm),
                      ax=ax, fraction=.035, pad=.022)
    cb.set_label('Spearman ρ  (quality-aligned)', color=INK, fontsize=10,
                 labelpad=8)
    cb.ax.tick_params(colors=MUTED, length=0)
    cb.outline.set_visible(False)

    fig.text(.135, .945, 'Cross-metric agreement · 750-task arms',
             fontsize=20, weight='bold', color=INK, va='top')
    fig.text(.135, .905,
             'upper triangle = rank correlation ρ   ·   lower triangle = '
             'pairwise complete cases n   ·   ↓ metrics negated so ρ>0 = '
             'both metrics prefer the same arm',
             fontsize=10, color=MUTED, va='top')
    fig.text(.135, .055,
             f'{len(df)} arms · missing values left blank per pair '
             '(ArtFID 44, Style Affinity 63, LPIPS 95, DINO 143) · '
             'source: docs/metrics_750.csv',
             fontsize=9, color=MUTED, va='bottom')
    return save(fig, 'correlation_heatmap', vector)


# ------------------------------------------------------------- 2. pairs grid
def fig_pairs(df, vector):
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.6), facecolor=BG)
    fig.subplots_adjust(left=.070, right=.985, bottom=.105, top=.855,
                        wspace=.265, hspace=.34)
    for ax, (a, b, note) in zip(axes.ravel(), PAIRS):
        m = df[a].notna() & df[b].notna()
        d = df.loc[m]
        for cat, col in CATS:
            s = d[d['category'] == cat]
            ax.scatter(s[a], s[b], s=30, c=col, ec='white', lw=.55,
                       alpha=.88, zorder=3)
        r, n = rho_n(df, a, b)
        if n >= 4:
            xx = np.linspace(d[a].min(), d[a].max(), 50)
            p = np.polyfit(d[a].values, d[b].values, 1)
            ax.plot(xx, np.polyval(p, xx), color='#8FA1B4', lw=1.3,
                    ls=(0, (4, 3)), alpha=.8, zorder=2)
        ax.set_title(note, fontsize=11, color=INK, weight='bold', pad=7)
        ax.set_xlabel(LABEL[a], fontsize=10.5, color=INK, labelpad=5)
        ax.set_ylabel(LABEL[b], fontsize=10.5, color=INK, labelpad=5)
        ax.grid(color=GRID, lw=1)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, labelsize=9, pad=4)
        ax.text(.025, .965, f'ρ = {r:+.2f}   n = {n}',
                transform=ax.transAxes, ha='left', va='top', fontsize=10,
                color=INK, weight='bold', zorder=6,
                bbox=dict(boxstyle='round,pad=.28', fc='white',
                          ec='#DCE3EB', alpha=.92))

    fig.text(.070, .945, 'Six cross-metric views',
             fontsize=20, weight='bold', color=INK, va='top')
    fig.text(.070, .907,
             'Every dot is one arm. ρ is Spearman on quality-aligned ranks '
             '(↓ metrics negated). Dashed line = least-squares fit, '
             'shown only as a visual guide.',
             fontsize=10, color=MUTED, va='top')
    fig.legend(handles=cat_handles(), loc='upper right',
               bbox_to_anchor=(.985, .935), ncol=4, frameon=False,
               fontsize=9.4, columnspacing=1.15, handletextpad=.5)
    fig.text(.070, .035,
             'n differs per panel because each metric was computed on a '
             'different subset of arms — no imputation, no recompute.',
             fontsize=9, color=MUTED, va='bottom')
    return save(fig, 'pairs_grid', vector)


# -------------------------------------------------------- 3. ArtFID landscape
KEY = {
    'Identity': (10, 6), 'SD-Turbo': (8, -14), 'StyleID': (8, 7),
    'Seedream 4.5': (-12, -16), 'SaMAM': (8, 7), 'CUT': (8, -14),
    'WCT': (8, 7), 'AdaIN': (8, -14), 'StyleShot': (8, 7),
    'StyTr2': (8, -14), 'AesPA': (8, 7), 'SaMST': (8, -14),
    'Hist2Style': (8, 7), 'StyleGallery': (8, -14),
    'WEAVE.LD4': (-2, -18), 'V10.E4': (10, 10),
    'V15.W_dim96_conv_base.t00': (8, -14),
    'V15.W_dim96_conv_base.t04': (8, 7),
    'V15.W_dim96_conv_base.t07': (8, -15),
    'WEAVE.D5.weave': (-6, -16), 'WEAVE.D5.weave_yfix': (-6, 9),
    'WEAVE.D5.weave_oriented_e4': (-6, 20),
}


def pareto(points):
    """points = (name, x_good, y_good); keep non-dominated (both larger better)."""
    return sorted([p for p in points if not any(
        q[1] >= p[1] and q[2] >= p[2] and (q[1] > p[1] or q[2] > p[2])
        for q in points if q is not p)], key=lambda p: p[1])


def fig_landscape(df, vector):
    fig = plt.figure(figsize=(13.4, 8.8), facecolor=BG)
    ax = fig.add_axes([.088, .135, .735, .72], facecolor=BG)

    # background: all arms with DINO readout (no ArtFID column)
    back = df[df.dino_s.notna() & df.dino_c.notna() & df.artfid.isna()]
    ax.scatter(back.dino_c, back.dino_s, s=22, c='#D6DEE7', ec='none',
               alpha=.8, zorder=2)

    front = df[df.artfid.notna() & df.dino_s.notna() & df.dino_c.notna()]
    cm = plt.get_cmap('RdYlGn_r')
    norm = Normalize(vmin=front.artfid.min(), vmax=front.artfid.max())
    ax.scatter(front.dino_c, front.dino_s, s=92, c=front.artfid,
               cmap=cm, norm=norm, ec=INK, lw=.75, alpha=.96, zorder=4)

    # external-only Pareto on the displayed style–content plane
    ext = df[(df['group'] == 'Baseline') |
             (df['group'] == 'Baseline (reproduced)')]
    pts = [(r.arm, r.dino_c, r.dino_s) for _, r in ext.iterrows()
           if pd.notna(r.dino_c) and pd.notna(r.dino_s)]
    pf = pareto(pts)
    ax.plot([p[1] for p in pf], [p[2] for p in pf],
            color=BLUE, lw=1.8, ls=(0, (5, 3)), alpha=.85, zorder=3)
    for p in pf:
        ax.scatter(p[1], p[2], s=190, fc='none', ec=BLUE, lw=1.5,
                   zorder=3)

    # short, in-canvas labels for key arms only
    for name, (dx, dy) in KEY.items():
        r = df[df.arm == name]
        if r.empty or pd.isna(r.iloc[0].dino_c) or pd.isna(r.iloc[0].dino_s):
            continue
        r = r.iloc[0]
        ann = ax.annotate(name.replace('V15.W_dim96_conv_base.', 'V15.'),
                          xy=(r.dino_c, r.dino_s),
                          xytext=(dx, dy), textcoords='offset points',
                          fontsize=8.6, color=INK, weight='bold' if
                          r['category'] == 'External' else 'normal', zorder=7,
                          arrowprops=dict(arrowstyle='-', lw=.65,
                                          color='#AAB9C8', shrinkA=2,
                                          shrinkB=4))
        ann.set_path_effects([pe.withStroke(linewidth=2.6, foreground=BG)])

    ax.set_xlabel('Content preservation / DINO-C   → higher is better',
                  fontsize=12, color=INK, weight='bold', labelpad=12)
    ax.set_ylabel('Style purity / DINO-S   ↑ higher is better',
                  fontsize=12, color=INK, weight='bold', labelpad=12)
    ax.grid(color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=9.5, pad=5)

    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cm),
                      ax=ax, fraction=.032, pad=.022)
    cb.set_label('ArtFID ↓  (green = better)', color=INK, fontsize=10,
                 labelpad=8)
    cb.ax.tick_params(colors=MUTED, length=0)
    cb.outline.set_visible(False)

    grey_proxy = Line2D([], [], marker='o', ls='none', ms=6.5,
                        mfc='#D6DEE7', mec='none', label='DINO only (no ArtFID)')
    pareto_proxy = Line2D([], [], color=BLUE, lw=1.8, ls=(0, (5, 3)),
                          label='external Pareto (content, DINO-S)')
    ring_proxy = Line2D([], [], marker='o', ls='none', ms=10,
                        mfc='none', mec=BLUE, mew=1.5,
                        label='Pareto members')
    leg = ax.legend(handles=[grey_proxy, pareto_proxy, ring_proxy],
                    loc='upper right', bbox_to_anchor=(1.0, .985),
                    frameon=False, fontsize=9.2)
    for t in leg.get_texts():
        t.set_color(MUTED)

    fig.text(.088, .945, 'ArtFID on the style–content plane',
             fontsize=21, weight='bold', color=INK, va='top')
    fig.text(.088, .905,
             f'{len(front)} arms with both DINO and ArtFID · '
             'colour = distribution quality, position = perceptual quality · '
             'dashed rings = external Pareto set (content, DINO-S)',
             fontsize=10, color=MUTED, va='top')
    fig.text(.088, .045,
             'ArtFID tracks content preservation (ρ=+0.77 with DINO-C) but is '
             'nearly orthogonal to style metrics (ρ=−0.16 with DINO-S, '
             '+0.10 with Affinity). Grey dots have DINO only.',
             fontsize=9, color=MUTED, va='bottom')
    return save(fig, 'artfid_landscape', vector)


# ----------------------------------------------------------- 4. ArtFID ranking
def fig_ranking(df, vector):
    d = df[df.artfid.notna()].sort_values('artfid').reset_index(drop=True)
    fig = plt.figure(figsize=(10.4, 13.2), facecolor=BG)
    ax = fig.add_axes([.315, .075, .60, .855], facecolor=BG)
    colors = [CAT_COLOR[c] for c in d['category']]
    ax.barh(np.arange(len(d)), d.artfid, color=colors, ec='white',
            lw=.55, height=.76, zorder=3)
    for i, (_, r) in enumerate(d.iterrows()):
        star = '*' if r.artfid_n_missing and r.artfid_n_missing > 0 else ''
        ax.text(r.artfid + 5, i, f'{r.artfid:.0f}{star}',
                va='center', ha='left', fontsize=8.4, color=MUTED, zorder=4)
    ax.set_yticks(np.arange(len(d)))
    ax.set_yticklabels(d.arm, fontsize=8.2)
    ax.tick_params(length=0, pad=4)
    ax.set_xlim(0, d.artfid.max() * 1.12)
    ax.set_xlabel('ArtFID ↓  lower is better', fontsize=11, color=INK,
                  weight='bold', labelpad=9)
    ax.grid(axis='x', color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.invert_yaxis()

    fig.text(.315, .975, 'ArtFID · 44 arms ranked',
             fontsize=20, weight='bold', color=INK, va='top')
    fig.text(.315, .945,
             'ArtFID = (1+FID)(1+LPIPS), mean over 5 target styles · '
             '* = incomplete image set',
             fontsize=9.5, color=MUTED, va='top')
    fig.legend(handles=cat_handles(), loc='lower left',
               bbox_to_anchor=(.315, .012), ncol=4, frameon=False,
               fontsize=9.2, columnspacing=1.1, handletextpad=.5)
    fig.text(.985, .012, 'source: docs/metrics_750.csv',
             ha='right', va='bottom', fontsize=8.6, color=MUTED)
    fig.text(.315, .045, 'Seedream 30 · StyleShot 5 missing images',
             ha='left', va='bottom', fontsize=8.2, color=MUTED)
    return save(fig, 'artfid_ranking', vector)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vector', action='store_true',
                    help='also write SVG and PDF')
    a = ap.parse_args()
    setup()
    df = load()
    print(f'{len(df)} arms · {df.artfid.notna().sum()} with ArtFID')
    fig_heatmap(df, a.vector)
    fig_pairs(df, a.vector)
    fig_landscape(df, a.vector)
    fig_ranking(df, a.vector)
    print('all figures →', OUT)


if __name__ == '__main__':
    main()
