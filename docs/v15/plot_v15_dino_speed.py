#!/usr/bin/env python3
"""DINO variant of docs/v15/plot_v15_style_content_speed.py (itself a copy of
docs/v8/plot_v8_style_content_speed.py — same visual language).

x = DINO-C (content preservation), y = DINO-S (style purity),
bubble area ∝ throughput where a measured ms/img exists.
Baselines: published numbers from docs/baseline/UNIFIED_FULL_RANKING.md
section 一 (DINO-S column); Hist2Style/StyleGallery = our 750-task repro.
V8/V11B arms: docs/v8/results.csv + docs/v15/v11b_arms.csv (Cross600)
+ runs/v8_speed.json + docs/v15/v11b_speed.json.
V15 Group A: runs/v15/eval/W_*.json, cross600 means per t0 level
(16 arms x 3 levels = 48 points, fixed-size diamonds — V15 inference
throughput was not instrumented, so no bubble).

Outputs: docs/v15/dino_speed_v15{suffix}.png/.svg/.pdf
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import patheffects as pe
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MIN_DC = 0.20          # lower than speed plot: keep all V15 t0 levels visible
HIDE = set()
X0 = 0.20

# Published external baselines: (name, DINO-C, DINO-S, ms/img)
# DINO-S values from UNIFIED_FULL_RANKING.md section 一; speeds as in the
# style_content_speed plot (paper.tex for published, our measured runner
# for the two reproduced methods).
EXTERNAL = [
    ('StyleShot',   0.3569, 0.5750, 24480.0),
    ('StyleID',     0.3885, 0.5143,  5040.0),
    ('AesPA',       0.6075, 0.4661,  1840.0),
    ('StyTr2',      0.7023, 0.4074,  3040.0),
    ('Seedream 4.5',0.7343, 0.3885,  2000.0),
    ('SaMAM',       0.8239, 0.3832,  1408.0),
    ('CUT',         0.8435, 0.3755,   400.0),
    ('SD-Turbo',    0.9998, 0.3633,   400.0),
    ('SaMST',       0.1628, 0.2738,   800.0),   # hidden: dino_c < MIN_DC
    ('WCT',         0.0232, 0.1542,   293.6),   # hidden: dino_c < MIN_DC
    # reproduced on WEAVE 750 (Cross600), same protocol as src.eval.measure.
    # H2S dino_s .3643 / dino_c .9083; SG dino_s .5604 / dino_c .6113.
    ('Hist2Style',  0.9083, 0.3643,   200.0),
    ('StyleGallery',0.6113, 0.5604, 76000.0),
]
WEAVE = ('WEAVE (published)', 0.8079, 0.4003, 97.60)

# Reference bounds: Identity style floor and target-reference content floor.
IDT_STYLE = 0.3633   # Identity row, UNIFIED_FULL_RANKING (DINO-S)
TGT_CONTENT = 0.2190 # tgt reference image as output, dino_c mean
                     # (docs/reproduction/tgt_reference_sensitivity.json)

EXT_LABELS = {
    'StyleShot':    (0.42, 0.60, 'left'),
    'StyleID':      (0.45, 0.535, 'left'),
    'AesPA':        (0.56, 0.445, 'right'),
    'StyTr2':       (0.66, 0.385, 'right'),
    'Seedream 4.5': (0.775, 0.368, 'left'),
    'SaMAM':        (0.885, 0.405, 'left'),
    'CUT':          (0.90, 0.395, 'left'),
    'SD-Turbo':     (1.015, 0.345, 'right'),
    'Hist2Style':   (0.945, 0.385, 'left'),
    'StyleGallery': (0.60, 0.585, 'right'),
}
V8_LABELS = {
    'L1.b4.r15': (0.80, 0.50, 'left'),
    'L1.b4.r25': (0.71, 0.49, 'right'),
    'L1.topk':   (0.70, 0.52, 'left'),
    'L2.e':      (0.84, 0.50, 'left'),
    'I2':        (0.875, 0.49, 'right'),
    'A2b':       (0.665, 0.505, 'left'),
    'V11B.H2.t035': (0.72, 0.525, 'left'),
    'V11B.BOTH.t035': (0.70, 0.545, 'right'),
    'V11B.H2.t06': (0.66, 0.50, 'left'),
}
# V15 labels: only a few representative cells (48 points would be noise).
V15_LABELS = {
    'W_dim96_conv_base.t07': (0.79, 0.485, 'left'),
    'W_dim96_conv_base.t00': (0.36, 0.60, 'left'),
}


def pareto_front(points):
    return sorted([
        a for a in points
        if not any(
            b[1] >= a[1] and b[2] >= a[2]
            and (b[1] > a[1] or b[2] > a[2])
            for b in points if b is not a
        )
    ], key=lambda p: p[1])


def load_v8():
    """V8 + V11B arms with measured speed -> (name, dino_c, dino_s, ms)."""
    df = pd.read_csv(HERE.parent / 'v8' / 'results.csv')
    extra = pd.read_csv(HERE / 'v11b_arms.csv')
    df = pd.concat([df, extra], ignore_index=True)
    df = df[df.split == 'cross'][['arm', 'dino_c', 'dino_s']]
    df = df[df.dino_c >= MIN_DC]
    speed = json.loads((REPO / 'runs/v8_speed.json').read_text())
    speed.update(json.loads((HERE / 'v11b_speed.json').read_text()))
    rows = []
    for _, r in df.iterrows():
        sp = speed.get(r.arm)
        if not sp or not sp.get('ms_per_img'):
            continue
        rows.append((r.arm, float(r.dino_c), float(r.dino_s),
                     float(sp['ms_per_img'])))
    return rows


def load_v15():
    """V15 Group A: cross600 means per (arm, t0 level).
    Returns (name, dino_c, dino_s) — no speed, fixed-size markers."""
    import statistics as st
    rows = []
    for f in sorted((REPO / 'runs/v15/eval').glob('W_*.json')):
        arm = f.stem
        data = json.loads(f.read_text())
        for lev in sorted({r['level'] for r in data['rows']}):
            cr = [r for r in data['rows']
                  if r['level'] == lev and r['cross']]
            dc = st.mean(r['dino_c'] for r in cr)
            if dc < MIN_DC:
                continue
            ds = st.mean(r['dino_s'] for r in cr)
            tag = {0.0: 't00', 0.4: 't04', 0.7: 't07'}.get(lev, f't{lev}')
            rows.append((f'{arm}.{tag}', dc, ds, lev))
    return rows


def main():
    global MIN_DC, HIDE, X0
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-dc", type=float, default=MIN_DC)
    ap.add_argument("--x0", type=float, default=X0)
    ap.add_argument("--hide", nargs="*", default=[])
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    MIN_DC, HIDE, X0 = a.min_dc, set(a.hide), a.x0

    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 11,
        'svg.fonttype': 'none', 'pdf.fonttype': 42,
        'axes.spines.top': False, 'axes.spines.right': False,
    })
    ink, muted = '#182A42', '#62758B'
    blue, grey = '#356C8F', '#B9C3CE'
    orange, orange_dark = '#ED6547', '#AE3A29'
    purple, purple_dark = '#7A4FA3', '#4E2F6E'
    bg = '#FCFDFE'

    v8 = load_v8()
    v15 = load_v15()
    ext = [p for p in EXTERNAL if p[1] >= MIN_DC and p[0] not in HIDE]
    speed = lambda p: 1000.0 / p[3]
    ref = min(v8 + ext, key=lambda p: p[3])
    K = 2450.0 / speed(ref)
    area = lambda p: K * speed(p)

    # Pareto front = external baselines only; ours (WEAVE/V8/V11B/V15)
    # are isolated points, never frontier members.
    front = pareto_front(ext)
    front_names = {p[0] for p in front}
    all_pts = ext + [(n, c, s, 1.0) for n, c, s, _ in v15] + v8
    front_all = pareto_front(all_pts)
    front_all_names = {p[0] for p in front_all}

    fig = plt.figure(figsize=(15.6, 9.2), facecolor=bg)
    ax = fig.add_axes([.085, .20, .855, .63], facecolor=bg)
    # adaptive x-range: right edge hugs the rightmost point AND the
    # estimated right extent of left-anchored labels (≈0.0075/char).
    def label_end(name, tx, ha):
        if ha == 'left':
            return tx + 0.0075 * len(name)
        if ha == 'right':
            return tx
        return tx + 0.0038 * len(name)
    lab = ([(n, *EXT_LABELS[n]) for n in EXT_LABELS]
           + [(n, *V8_LABELS[n]) for n in V8_LABELS]
           + [(n, *V15_LABELS[n]) for n in V15_LABELS])
    xs = ([p[1] for p in ext] + [WEAVE[1]] + [p[1] for p in v8]
          + [p[1] for p in v15]
          + [label_end(n, tx, ha) for n, tx, ty, ha in lab])
    xmax = max(xs)
    ax.set_xlim(X0, xmax + 0.015)
    ax.set_xticks([i / 10 for i in
                   range(int(X0 * 10), int((xmax + 0.015) * 10) + 1)])
    ys = ([p[2] for p in ext] + [WEAVE[2]] + [p[2] for p in v8]
          + [p[2] for p in v15] + [IDT_STYLE])
    ymin, ymax = min(ys), max(ys)
    pad = (ymax - ymin) * 0.06 + 1e-4
    ax.set_ylim(max(0.0, ymin - pad), ymax + pad)
    yticks = [round(ymin + (ymax - ymin) * i / 8, 3) for i in range(9)]
    ax.set_yticks(yticks)
    ax.set_axisbelow(True)
    ax.grid(color='#E8EEF4', lw=1)
    for edge in ('bottom', 'left'):
        ax.spines[edge].set_color('#B9C5D0')
    ax.tick_params(colors=muted, length=0, pad=8)
    ax.set_xlabel('Content preservation / DINO-C   → Higher is better',
                  fontsize=13, weight='semibold', color=ink, labelpad=18)
    ax.set_ylabel('Style purity / DINO-S   ↑ Higher is better',
                  fontsize=13, weight='semibold', color=ink, labelpad=16)

    ax.axhline(IDT_STYLE, color='#C0392B', lw=1.4, ls=(0, (6, 4)),
               alpha=.85, zorder=2)
    ax.text(xmax + 0.012, IDT_STYLE, '  idt style floor', fontsize=9.5,
            color='#C0392B', va='center', ha='right', style='italic')
    ax.axvline(TGT_CONTENT, color='#C0392B', lw=1.4, ls=(0, (6, 4)),
               alpha=.85, zorder=2)
    ax.text(TGT_CONTENT, ymax + pad * .55, 'tgt content floor',
            fontsize=9.5, color='#C0392B', ha='center', style='italic')

    ax.plot([p[1] for p in front], [p[2] for p in front],
            color=blue, lw=2.1, ls=(0, (4, 3)), zorder=2)
    ax.fill_between([p[1] for p in front], [p[2] for p in front],
                    max(0.0, ymin - pad), color=blue, alpha=.035, zorder=1)

    for p in ext:
        hit = p[0] in front_names
        ax.scatter(p[1], p[2], s=area(p), marker='o',
                   fc=blue if hit else grey,
                   ec=blue if area(p) < 25 else 'white',
                   lw=.6 if area(p) < 25 else 1.3,
                   alpha=.94 if hit else .72,
                   zorder=6 if hit else 3)
    ax.scatter(WEAVE[1], WEAVE[2], s=area(WEAVE), marker='o',
               fc='#E2E8EF', ec='#5A6F88', lw=2, alpha=.88, zorder=4)
    ax.annotate('WEAVE', xy=(WEAVE[1], WEAVE[2]),
                xytext=(0.855, 0.43), ha='left', va='center',
                color='#5A6F88', fontsize=10, weight='bold',
                arrowprops=dict(arrowstyle='-', color='#5A6F88',
                                lw=1.0, shrinkA=4, shrinkB=8), zorder=12)

    for p in v8:
        hit = p[0] in front_all_names
        ax.scatter(p[1], p[2], s=area(p), marker='o',
                   fc=orange if hit else '#F4B8AC',
                   ec=orange_dark if hit else '#D98F80',
                   lw=1.4 if hit else .7,
                   alpha=.95 if hit else .55,
                   zorder=8 if hit else 5)

    # V15 Group A — fixed-size diamonds colored by t0 level
    t0_colors = {0.0: '#C9B8E8', 0.4: '#9A7CC4', 0.7: purple}
    for name, dc, ds, lev in v15:
        hit = name in front_all_names
        ax.scatter(dc, ds, s=42 if hit else 26, marker='D',
                   fc=t0_colors.get(lev, purple),
                   ec=purple_dark, lw=1.1 if hit else .6,
                   alpha=.95 if hit else .6,
                   zorder=9 if hit else 6)

    for p in ext:
        if p[0] not in EXT_LABELS:
            continue
        tx, ty, ha = EXT_LABELS[p[0]]
        ann = ax.annotate(p[0], xy=(p[1], p[2]), xytext=(tx, ty),
                          ha=ha, va='center', fontsize=10,
                          color=ink if p[0] in front_names else muted,
                          weight='semibold' if p[0] in front_names else 'normal',
                          arrowprops=dict(arrowstyle='-', lw=.8,
                                          color='#AAB9C8', shrinkA=3, shrinkB=5),
                          zorder=12)
        ann.set_path_effects([pe.withStroke(linewidth=3, foreground=bg)])

    for p in v8:
        if p[0] not in V8_LABELS:
            continue
        tx, ty, ha = V8_LABELS[p[0]]
        ann = ax.annotate(p[0], xy=(p[1], p[2]), xytext=(tx, ty),
                          ha=ha, va='center', fontsize=10,
                          color=orange_dark, weight='semibold',
                          arrowprops=dict(arrowstyle='-', lw=.8,
                                          color='#C98A7B', shrinkA=3, shrinkB=5),
                          zorder=12)
        ann.set_path_effects([pe.withStroke(linewidth=3, foreground=bg)])

    for name, dc, ds, lev in v15:
        if name not in V15_LABELS:
            continue
        tx, ty, ha = V15_LABELS[name]
        ann = ax.annotate(name, xy=(dc, ds), xytext=(tx, ty),
                          ha=ha, va='center', fontsize=10,
                          color=purple_dark, weight='semibold',
                          arrowprops=dict(arrowstyle='-', lw=.8,
                                          color='#B49ACB', shrinkA=3, shrinkB=5),
                          zorder=12)
        ann.set_path_effects([pe.withStroke(linewidth=3, foreground=bg)])

    fig.text(.085, .932, 'V15  ×  STYLE  ×  CONTENT  ×  SPEED  (DINO)',
             fontsize=25, weight='heavy', color=ink, va='top')
    fig.text(.085, .878,
             f'Cross600 · y = DINO-S (style purity) · x = DINO-C · DINO-C ≥ {MIN_DC}',
             fontsize=12.5, color=muted, va='top')
    fig.text(.928, .934, f'{len(v8)} arms',
             ha='right', va='top', fontsize=30, color=orange, weight='heavy')
    fig.text(.928, .878, 'V8+V11B trained variants',
             ha='right', va='top', fontsize=10.5, color=muted)

    cat_handles = [
        Line2D([], [], marker='o', ls='none', ms=9, mfc=orange, mec=orange_dark,
               label='V8/V11B on combined Pareto front'),
        Line2D([], [], marker='o', ls='none', ms=9, mfc='#F4B8AC', mec='#D98F80',
               label='V8/V11B dominated'),
        Line2D([], [], marker='D', ls='none', ms=8, mfc=purple, mec=purple_dark,
               label='V15 on combined Pareto front'),
        Line2D([], [], marker='D', ls='none', ms=8, mfc='#C9B8E8', mec=purple_dark,
               label='V15 dominated (light=t0.0 mid=t0.4 dark=t0.7)'),
        Line2D([], [], marker='o', ls='none', ms=9, mfc=blue, mec='white',
               label='External frontier'),
        Line2D([], [], marker='o', ls='none', ms=9, mfc=grey, mec='white',
               label='Other external'),
        Line2D([], [], color=blue, ls=(0, (4, 3)), lw=2,
               label='External-only Pareto'),
    ]
    fig.legend(handles=cat_handles, bbox_to_anchor=(.085, .135),
               loc='lower left', ncol=4, frameon=False, fontsize=10)

    speeds = [1, 3, 6, 10]
    h = [ax.scatter([], [], s=K * v, marker='o',
                    fc='#D9E1E9', ec='#93A5B8', lw=1.1) for v in speeds]
    leg = ax.legend(h, [f'{v:g} img/s' for v in speeds],
                    title='Bubble area ∝ throughput (V8/V11B/ext only)',
                    loc='upper right', bbox_to_anchor=(0.995, 0.985),
                    ncol=4, columnspacing=1.2,
                    frameon=False, fontsize=9.7, title_fontsize=10,
                    scatterpoints=1, labelspacing=0.4, handletextpad=2.5,
                    borderaxespad=0.2)
    leg.get_title().set_color(ink)
    for t in leg.get_texts():
        t.set_color(muted)

    fig.text(.085, .10,
             'y-axis = DINO-S (style purity), NOT Style Affinity — V15 was '
             'evaluated with DINO metrics only (no CLIP-S), so this is the '
             'largest common metric set across all series. '
             'V8/V11B ms/img = gen wall time incl. model+edit (750 tasks, '
             'batch 8, RTX 3060); V11B on RTX 4070. External times from '
             'paper.tex; H2S/SG reproduced on same 750 tasks (H2S ~5 img/s '
             'resident runner; SG official per-image subprocess 76.0 s). '
             'V15 Group A markers are fixed-size (throughput not '
             'instrumented): 16 arms x t0 {0.0,0.4,0.7}, cross600 means '
             'from runs/v15/eval/W_*.json. '
             'idt floor = Identity DINO-S 0.3633; tgt floor = 0.2190. '
             'WCT/SaMST dropped (DINO-C < 0.20).',
             fontsize=9.3, color=muted)

    out = HERE / f'dino_speed_v15{a.suffix}'
    for ext_ in ('png', 'svg', 'pdf'):
        fig.savefig(out.with_suffix('.' + ext_), bbox_inches='tight',
                    pad_inches=.22, dpi=230 if ext_ == 'png' else None)
    plt.close(fig)
    print('Saved:', out.with_suffix('.png'))
    print('External front:', [p[0] for p in front])
    print('V8 on front:', sorted(p[0] for p in v8 if p[0] in front_all_names))
    print('V15 on front:', sorted(p[0] for p in v15 if p[0] in front_all_names))


if __name__ == '__main__':
    main()
