#!/usr/bin/env python3
"""V8 style/content/speed bubble plot, same visual language as
docs/v8/plot_style_content_speed_standalone.py.

x = DINO-C (content preservation), y = Style Affinity, bubble area ∝ throughput.
Baselines: published numbers hard-coded (same as the standalone script).
V8 arms: docs/v8/results.csv (Cross600) + runs/v8_speed.json (gen ms/img).
Points with DINO-C < 0.34 are dropped (content collapse, off-topic region).

Outputs: docs/v8/style_content_speed_v8.png/.svg/.pdf
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
MIN_DC = 0.34
HIDE = set()          # baseline names to drop entirely (zoom mode)
X0 = 0.20             # x-axis lower bound (zoom mode: 0.60)

# Published external baselines: (name, DINO-C, Style Affinity, ms/img)
EXTERNAL = [
    ('StyleShot',   0.3569, 0.6949, 24480.0),
    ('StyleID',     0.3885, 0.6469,  5040.0),
    ('AesPA',       0.6075, 0.5784,  1840.0),
    ('StyTr2',      0.7023, 0.5317,  3040.0),
    ('Seedream 4.5',0.7343, 0.5314,  2000.0),
    ('SaMAM',       0.8239, 0.5199,  1408.0),
    ('CUT',         0.8435, 0.5093,   400.0),
    ('SD-Turbo',    0.9998, 0.4894,   400.0),
    ('SaMST',       0.1628, 0.4288,   800.0),   # hidden: dino_c < 0.34
    ('WCT',         0.0232, 0.3658,   293.6),   # hidden: dino_c < 0.34
    # reproduced on WEAVE 750 (Cross600), eval identical to src.eval.measure.
    # Speed = OUR measured resident-runner throughput on RTX 3060:
    # H2S batch_h2s_750.py infer_wall_s median 0.23 s -> ~5 img/s
    # (mean 0.36 s -> 2.8 img/s; we plot the 5 img/s operating point).
    # SG official per-image subprocess runner, 19-task mean 76.0 s.
    ('Hist2Style',  0.9083, 0.5069,   200.0),
    ('StyleGallery',0.6113, 0.6432, 76000.0),
]
WEAVE = ('WEAVE (published)', 0.8079, 0.5367, 97.60)

# Reference bounds: Identity style floor and target-reference content floor.
IDT_STYLE = 0.4902   # Identity row, UNIFIED_FULL_RANKING (style affinity)
TGT_CONTENT = 0.2190 # tgt reference image as output, dino_c mean
                     # (docs/reproduction/tgt_reference_sensitivity.json)

# Label offsets in data coords (tx, ty, ha) — same convention as the
# standalone script; hand-tuned to avoid overlap in the cropped view.
EXT_LABELS = {
    'StyleShot':    (0.42, 0.712, 'left'),
    'StyleID':      (0.45, 0.664, 'left'),
    'AesPA':        (0.55, 0.560, 'right'),
    'StyTr2':       (0.64, 0.508, 'right'),
    'Seedream 4.5': (0.775, 0.512, 'left'),
    'SaMAM':        (0.885, 0.558, 'left'),
    'CUT':          (0.90, 0.528, 'left'),
    'SD-Turbo':     (1.015, 0.468, 'right'),
    'Hist2Style':   (0.945, 0.545, 'left'),
    'StyleGallery': (0.60, 0.665, 'right'),
}
V8_LABELS = {
    'L1.b4.r15': (0.80, 0.566, 'left'),
    'L1.b4.r25': (0.71, 0.560, 'right'),
    'L1.topk':   (0.70, 0.592, 'left'),
    'L2.e':      (0.84, 0.560, 'left'),
    'I2':        (0.875, 0.556, 'right'),
    'A2b':       (0.665, 0.545, 'left'),
    'V11B.H2.t035': (0.72, 0.585, 'left'),
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
    df = pd.read_csv(HERE / 'results.csv')
    df = df[df.split == 'cross'][['arm', 'dino_c', 'style_affinity', 'dino_s']]
    df = df[df.dino_c >= MIN_DC]
    speed = json.loads((REPO / 'runs/v8_speed.json').read_text())
    rows = []
    for _, r in df.iterrows():
        sp = speed.get(r.arm)
        if not sp or not sp.get('ms_per_img'):
            continue
        rows.append((r.arm, float(r.dino_c), float(r.style_affinity),
                     float(sp['ms_per_img'])))
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
    bg = '#FCFDFE'

    v8 = load_v8()
    ext = [p for p in EXTERNAL if p[1] >= MIN_DC and p[0] not in HIDE]
    speed = lambda p: 1000.0 / p[3]
    # scale: fastest V8 arm gets the reference bubble size (same K idea)
    ref = min(v8 + ext, key=lambda p: p[3])
    K = 2450.0 / speed(ref)
    area = lambda p: K * speed(p)

    # Pareto fronts use EXTERNAL published baselines only — WEAVE (published)
    # is ours, not a baseline, so it is plotted as reference but excluded.
    front = pareto_front(ext)
    front_names = {p[0] for p in front}
    all_pts = ext + v8
    front_all = pareto_front(all_pts)
    front_all_names = {p[0] for p in front_all}

    fig = plt.figure(figsize=(15.6, 9.2), facecolor=bg)
    ax = fig.add_axes([.085, .20, .855, .63], facecolor=bg)
    ax.set_xlim(X0, 1.06)
    ax.set_xticks([i / 10 for i in range(int(X0 * 10), 11)])
    # adaptive y-range: stretch so the highest/lowest points touch the edges
    ys = [p[2] for p in ext + [WEAVE] + v8] + [IDT_STYLE]
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
    ax.set_ylabel('Style fidelity / Style Affinity   ↑ Higher is better',
                  fontsize=13, weight='semibold', color=ink, labelpad=16)

    # reference bounds: identity style floor (h) and target-ref content floor (v)
    ax.axhline(IDT_STYLE, color='#C0392B', lw=1.4, ls=(0, (6, 4)),
               alpha=.85, zorder=2)
    ax.text(1.055, IDT_STYLE, '  idt style floor', fontsize=9.5,
            color='#C0392B', va='center', ha='right', style='italic')
    ax.axvline(TGT_CONTENT, color='#C0392B', lw=1.4, ls=(0, (6, 4)),
               alpha=.85, zorder=2)
    ax.text(TGT_CONTENT, ymax + pad * .55, 'tgt content floor',
            fontsize=9.5, color='#C0392B', ha='center', style='italic')

    # external-only Pareto (baselines define the frontier we compare against)
    ax.plot([p[1] for p in front], [p[2] for p in front],
            color=blue, lw=2.1, ls=(0, (4, 3)), zorder=2)
    ax.fill_between([p[1] for p in front], [p[2] for p in front],
                    max(0.0, ymin - pad), color=blue, alpha=.035, zorder=1)

    # external baselines — every point labeled
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
                xytext=(0.855, 0.575), ha='left', va='center',
                color='#5A6F88', fontsize=10, weight='bold',
                arrowprops=dict(arrowstyle='-', color='#5A6F88',
                                lw=1.0, shrinkA=4, shrinkB=8), zorder=12)

    # V8 arms
    for p in v8:
        hit = p[0] in front_all_names
        ax.scatter(p[1], p[2], s=area(p), marker='o',
                   fc=orange if hit else '#F4B8AC',
                   ec=orange_dark if hit else '#D98F80',
                   lw=1.4 if hit else .7,
                   alpha=.95 if hit else .55,
                   zorder=8 if hit else 5)

    # external labels
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

    # V8 labels (key arms only)
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

    fig.text(.085, .932, 'V8  ×  STYLE  ×  CONTENT  ×  SPEED',
             fontsize=25, weight='heavy', color=ink, va='top')
    fig.text(.085, .878, f'Cross600 · paper_rgb_v1 · DINO-C ≥ {MIN_DC} · bubble ∝ throughput (RTX 3060)',
             fontsize=12.5, color=muted, va='top')
    fig.text(.928, .934, f'{len(v8)} arms',
             ha='right', va='top', fontsize=30, color=orange, weight='heavy')
    fig.text(.928, .878, 'trained+inference variants',
             ha='right', va='top', fontsize=10.5, color=muted)

    cat_handles = [
        Line2D([], [], marker='o', ls='none', ms=9, mfc=orange, mec=orange_dark,
               label='V8 on combined Pareto front'),
        Line2D([], [], marker='o', ls='none', ms=9, mfc='#F4B8AC', mec='#D98F80',
               label='V8 dominated'),
        Line2D([], [], marker='o', ls='none', ms=9, mfc=blue, mec='white',
               label='External frontier'),
        Line2D([], [], marker='o', ls='none', ms=9, mfc=grey, mec='white',
               label='Other external'),
        Line2D([], [], color=blue, ls=(0, (4, 3)), lw=2,
               label='External-only Pareto'),
    ]
    fig.legend(handles=cat_handles, bbox_to_anchor=(.085, .155),
               loc='lower left', ncol=5, frameon=False, fontsize=10)

    speeds = [1, 3, 6, 10]
    h = [ax.scatter([], [], s=K * v, marker='o',
                    fc='#D9E1E9', ec='#93A5B8', lw=1.1) for v in speeds]
    leg = ax.legend(h, [f'{v:g} img/s' for v in speeds],
                    title='Bubble area ∝ throughput',
                    loc='upper right', bbox_to_anchor=(0.995, 0.985),
                    ncol=4, columnspacing=1.2,
                    frameon=False, fontsize=9.7, title_fontsize=10,
                    scatterpoints=1, labelspacing=0.4, handletextpad=2.5,
                    borderaxespad=0.2)
    leg.get_title().set_color(ink)
    for t in leg.get_texts():
        t.set_color(muted)

    fig.text(.085, .10,
             'V8 ms/img = gen wall time incl. model+edit (750 tasks, batch 8, RTX 3060). '
             'External times from paper.tex; H2S/SG reproduced on the same 750 tasks, '
             'Speed = our measured runner on idle RTX 3060: H2S resident batch '
             '~5 img/s (median 0.23 s/img, mean 0.36 s/img); SG official '
             'per-image subprocess mean 76.0 s over 19 tasks (SG fast-runner '
             '~33 s/img not counted). '
             'V11B.H2.t035 = wavelet stroke flow post-edit on LD4 outputs '
             '(RTX 4070, 121 ms/img end-to-end incl. model load, 750 tasks); '
             'Cross600 dino_c .7770 / affinity .5666 — new combined-front point. '
             'WCT/SaMST dropped (DINO-C < 0.34).',
             fontsize=9.3, color=muted)

    out = HERE / f'style_content_speed_v8{a.suffix}'
    for ext_ in ('png', 'svg', 'pdf'):
        fig.savefig(out.with_suffix('.' + ext_), bbox_inches='tight',
                    pad_inches=.22, dpi=230 if ext_ == 'png' else None)
    plt.close(fig)
    print('Saved:', out.with_suffix('.png'))
    print('External front:', [p[0] for p in front])
    print('V8 on front:', sorted(p[0] for p in v8 if p[0] in front_all_names))


if __name__ == '__main__':
    main()
