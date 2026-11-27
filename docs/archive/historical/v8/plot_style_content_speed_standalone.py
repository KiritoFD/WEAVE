#!/usr/bin/env python3
"""Self-contained WEAVE style/content/speed bubble plot.

Usage: pip install matplotlib; python plot_style_content_speed_standalone.py
Outputs: style_content_speed.png/.svg/.pdf

Quality: paper_rgb_v1 cross600.
Timing: RTX 3060 750-image inference, except Seedream (ASSUMED)
and S10+TRT (STITCHED estimate, NOT an end-to-end measurement).
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import patheffects as pe

# (model, DINO-C content, Style Affinity, milliseconds / image)
# All values hard-coded; no Markdown / CSV / JSON input required.
EXTERNAL = [
    ('StyleShot',   0.3569, 0.6949, 24480.0),
    ('StyleID',     0.3885, 0.6469,  5040.0),
    ('AesPA',       0.6075, 0.5784,  1840.0),
    ('StyTr2',      0.7023, 0.5317,  3040.0),
    ('Seedream 4.5',0.7343, 0.5314,  2000.0),  # user-assumed 0.5 img/s
    ('SaMAM',       0.8239, 0.5199,  1408.0),
    ('CUT',         0.8435, 0.5093,   400.0),
    ('SD-Turbo',    0.9998, 0.4894,   400.0),
    ('SaMST',       0.1628, 0.4288,   800.0),
    ('WCT',         0.0232, 0.3658,   293.6),
]
WEAVE = ('WEAVE (published)', 0.8079, 0.5367, 97.60)
OURS  = ('OURS / S10 + TRT',  0.7324, 0.5575, 45.78)  # stitched estimate

# Positions of text labels in data coordinates (not altered measurements).
LABELS = {
    'StyleShot':    (0.414, 0.712, 'left'),
    'StyleID':      (0.448, 0.662, 'left'),
    'AesPA':        (0.548, 0.565, 'right'),
    'StyTr2':       (0.636, 0.504, 'right'),
    'Seedream 4.5': (0.760, 0.485, 'left'),
    'SaMAM':        (0.882, 0.561, 'left'),
    'CUT':          (0.899, 0.503, 'left'),
    'SD-Turbo':     (1.019, 0.464, 'right'),
    'SaMST':        (0.219, 0.446, 'left'),
    'WCT':          (0.085, 0.387, 'left'),
}


def pareto_front(points):
    """Maximize both DINO-C and Style Affinity; only EXTERNAL points."""
    return sorted([
        a for a in points
        if not any(
            b[1] >= a[1] and b[2] >= a[2]
            and (b[1] > a[1] or b[2] > a[2])
            for b in points if b is not a
        )
    ], key=lambda p: p[1])


def main():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 11,
        'svg.fonttype': 'none', 'pdf.fonttype': 42,
        'axes.spines.top': False, 'axes.spines.right': False,
    })
    ink, muted = '#182A42', '#62758B'
    blue, grey = '#356C8F', '#B9C3CE'
    orange, orange_dark = '#ED6547', '#AE3A29'
    bg = '#FCFDFE'

    # IMPORTANT: matplotlib.scatter(s=...) uses AREA (points squared).
    # All plotted bubbles (including WEAVE, Ours, Seedream, legend)
    # use exactly the same scale: area = K * throughput_img_per_second.
    speed = lambda p: 1000.0 / p[3]
    K = 2450.0 / speed(OURS)
    area = lambda p: K * speed(p)

    front = pareto_front(EXTERNAL)
    front_names = {p[0] for p in front}
    fig = plt.figure(figsize=(15.6, 9.2), facecolor=bg)
    ax = fig.add_axes([.09, .245, .846, .555], facecolor=bg)
    ax.set_xlim(0, 1.06)
    ax.set_ylim(.336, .746)
    ax.set_xticks([i / 10 for i in range(11)])
    ax.set_yticks([.35 + .05 * i for i in range(9)])
    ax.set_axisbelow(True)
    ax.grid(color='#E8EEF4', lw=1)
    for edge in ('bottom', 'left'):
        ax.spines[edge].set_color('#B9C5D0')
    ax.tick_params(colors=muted, length=0, pad=8)
    ax.set_xlabel('Content preservation / DINO-C   → Higher is better',
                  fontsize=13, weight='semibold', color=ink, labelpad=18)
    ax.set_ylabel('Style fidelity / Style Affinity   ↑ Higher is better',
                  fontsize=13, weight='semibold', color=ink, labelpad=16)

    # 2-D external-only Pareto frontier; throughput is NOT part of this line.
    ax.plot([p[1] for p in front], [p[2] for p in front],
            color=blue, lw=2.1, ls=(0, (4, 3)), zorder=2)
    ax.fill_between([p[1] for p in front], [p[2] for p in front],
                    .336, color=blue, alpha=.035, zorder=1)

    # Every real data marker is a CIRCLE. Only the colors distinguish groups.
    # WEAVE first so adjacent small points are not buried.
    ax.scatter(WEAVE[1], WEAVE[2], s=area(WEAVE), marker='o',
               fc='#E2E8EF', ec='#5A6F88', lw=2, alpha=.88, zorder=4)
    for p in EXTERNAL:
        hit = p[0] in front_names
        ax.scatter(p[1], p[2], s=area(p), marker='o',
                   fc=blue if hit else grey,
                   ec=blue if area(p) < 25 else 'white',
                   lw=.6 if area(p) < 25 else 1.3,
                   alpha=.94 if hit else .72, zorder=6 if hit else 3)
    ax.scatter(OURS[1], OURS[2], s=area(OURS), marker='o',
               fc=orange, ec='white', lw=2.4, alpha=.98, zorder=9)
    # Seedream is extremely close to Ours; draw its correctly sized blue dot
    # last so that it remains visible rather than hidden by the orange circle.
    sd = next(p for p in EXTERNAL if p[0] == 'Seedream 4.5')
    ax.scatter(sd[1], sd[2], s=area(sd), marker='o',
               fc=blue, ec='white', lw=.9, alpha=1.0, zorder=10)

    # Annotate every model without changing point positions/sizes.
    for p in EXTERNAL:
        tx, ty, ha = LABELS[p[0]]
        ann = ax.annotate(p[0], xy=(p[1], p[2]), xytext=(tx, ty),
                          ha=ha, va='center', fontsize=10,
                          color=ink if p[0] in front_names else muted,
                          weight='semibold' if p[0] in front_names else 'normal',
                          arrowprops=dict(arrowstyle='-', lw=.8,
                                          color='#AAB9C8', shrinkA=3, shrinkB=5),
                          zorder=12)
        ann.set_path_effects([pe.withStroke(linewidth=3, foreground=bg)])

    ax.annotate('WEAVE (published)', xy=(WEAVE[1], WEAVE[2]),
                xytext=(.856, .600), ha='left', va='center',
                color='#5A6F88', fontsize=11, weight='bold',
                arrowprops=dict(arrowstyle='-', color='#5A6F88',
                                lw=1.0, shrinkA=4, shrinkB=8), zorder=12)
    ax.annotate('OURS / S10 + TRT', xy=(OURS[1], OURS[2]),
                xytext=(.699, .672), ha='left', va='center',
                color=orange_dark, fontsize=15, weight='bold',
                arrowprops=dict(arrowstyle='-', color=orange,
                                lw=1.4, shrinkA=5, shrinkB=18), zorder=14)
    ax.text(.699, .643, 'Style 0.5575  ·  Content 0.7324',
            color=ink, weight='semibold', fontsize=10.5)
    ax.text(.699, .621, '45.78 ms/img  ·  21.84 img/s*',
            color=orange_dark, weight='bold', fontsize=10.5)

    fig.text(.09, .932, 'STYLE  ×  CONTENT  ×  SPEED',
             fontsize=25, weight='heavy', color=ink, va='top')
    fig.text(.09, .867, 'Higher content and style is better; larger circles mean faster inference',
             fontsize=13, color=muted, va='top')
    fig.text(.928, .934, f'{3040.0 / OURS[3]:.0f}×',
             ha='right', va='top', fontsize=30, color=orange, weight='heavy')
    fig.text(.928, .867, 'faster than StyTr2*',
             ha='right', va='top', fontsize=10.5, color=muted)

    cat_handles = [
        Line2D([], [], marker='o', ls='none', ms=9, mfc=orange, mec='white', label='Ours / S10'),
        Line2D([], [], marker='o', ls='none', ms=9, mfc='#E2E8EF', mec='#5A6F88', label='WEAVE'),
        Line2D([], [], marker='o', ls='none', ms=9, mfc=blue, mec='white', label='External frontier'),
        Line2D([], [], marker='o', ls='none', ms=9, mfc=grey, mec='white', label='Other external'),
        Line2D([], [], color=blue, ls=(0, (4, 3)), lw=2, label='External-only 2-D Pareto'),
    ]
    fig.legend(handles=cat_handles, bbox_to_anchor=(.09, .136),
               loc='lower left', ncol=5, frameon=False, fontsize=10)

    # Speed legend: exact same area formula, separately from model category.
    speeds = [0.5, 2, 10, 20]
    h = [ax.scatter([], [], s=K * v, marker='o',
                    fc='#D9E1E9', ec='#93A5B8', lw=1.1) for v in speeds]
    leg = ax.legend(h, [f'{v:g} img/s' for v in speeds],
                    title='Bubble area ∝ throughput',
                    loc='lower right', bbox_to_anchor=(.76, .015),
                    frameon=False, fontsize=9.7, title_fontsize=10,
                    scatterpoints=1, labelspacing=1.55, handletextpad=3.5)
    leg.get_title().set_color(ink)
    for t in leg.get_texts():
        t.set_color(muted)
    ax.add_artist(leg)

    fig.text(.09, .085,
             '* Ours 45.78 ms/img is a stitched estimate (18.7 + 27.08 ms), NOT end-to-end remeasurement. '
             'Seedream 4.5 speed is user-assumed 0.5 img/s.',
             fontsize=9.3, color=muted)
    fig.text(.09, .056,
             'Quality: paper_rgb_v1 / cross600. Baseline times: paper.tex / RTX 3060 / 750 images. '
             'External-only Pareto computed from DINO-C and Style Affinity.',
             fontsize=9.3, color=muted)

    out = Path(__file__).resolve().parent / 'style_content_speed'
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(out.with_suffix('.' + ext), bbox_inches='tight',
                    pad_inches=.22, dpi=230 if ext == 'png' else None)
    plt.close(fig)
    print('Saved:', ', '.join(str(out.with_suffix('.' + e)) for e in ('png', 'svg', 'pdf')))
    print('External Pareto:', ', '.join(p[0] for p in front))


if __name__ == '__main__':
    main()
