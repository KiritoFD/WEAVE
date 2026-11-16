"""docs/performance — unified performance ledger for WEAVE and baselines.

build_metrics.py (rerunnable):
  1. 32-method leaderboard (deliver/q2a_oracle_scores_baseline_protocol.json,
     Cross600, S10-era CSD protocol)
  2. V16 arms (runs/v16/measured/*/eval.json -> cross600 means)
  3. V15 Group A arms (runs/v15/eval/*.json -> cross600 means)
  4. Anchor entries from docs/v10/V10_RESULTS.md (LD4 / E4, 750口径)
  5. V17 arms (runs/v17/measured/*/eval.json, added as the queue lands them)
Outputs: metrics.json, metrics.csv, pareto flags per protocol family,
plus plots (pareto scatter / rankings).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PERF = Path(__file__).resolve().parent
METRICS = ['dino_s', 'dino_c', 'csd_top1', 'lpips']          # pareto axes (↑↑↑↓)
HIGHER = {'dino_s', 'dino_c', 'csd_top1', 'clip_s', 'csd_margin', 'style_affinity'}


def read_json(path: Path):
    if not Path(path).exists():
        return None
    try:
        return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except Exception:
        return None


def cross600_means(rows):
    import statistics
    rows = [r for r in rows if r.get('is_cross')]
    if not rows:
        return None
    keys = ['dino_s', 'dino_c', 'lpips', 'target_top1', 'target_margin',
            'csd_source_score', 'csd_target_score']
    out = {'n_cross': len(rows)}
    for k in keys:
        vals = [float(r[k]) for r in rows if r.get(k) is not None]
        out[k if k != 'target_top1' else 'csd_top1'] = round(
            statistics.mean(vals), 5) if vals else None
    out['clip_s'] = round(statistics.mean(
        [r['clip_scores_5'][r['tgt_style_id']] for r in rows]), 5)
    return out


def entry(name, family, protocol, decoder, **m):
    e = {'method': name, 'family': family, 'protocol': protocol,
         'decoder': decoder}
    e.update({k: v for k, v in m.items() if v is not None})
    return e


def collect() -> list:
    entries = []
    # 1. 32-method leaderboard (q2a + oracle protocol, S10 / native VAE per row)
    lb = read_json(ROOT / 'deliver/q2a_oracle_scores_baseline_protocol.json')
    if lb:
        for r in lb['leaderboard_baseline_methods']:
            entries.append(entry(
                r['method'], r.get('category', 'Baseline'),
                'q2a_oracle_cross600', r.get('vae', '?'),
                dino_s=r.get('dino_s'), dino_c=r.get('dino_c'),
                csd_top1=r.get('csd_top1'), lpips=r.get('lpips'),
                clip_s=r.get('clip_s'), style_affinity=r.get('style_affinity'),
                rank=r.get('rank'), provenance='deliver/'
                'q2a_oracle_scores_baseline_protocol.json'))
    # 2. V16 arms
    for d in sorted((ROOT / 'runs/v16/measured').glob('*/eval.json')):
        data = read_json(d)
        if not data:
            continue
        m = cross600_means(data['rows'])
        if m:
            entries.append(entry(
                'V16_' + d.parent.name, 'WEAVE-V16', 'q2a_oracle_cross600',
                'S10', **{k: v for k, v in m.items() if k != 'n_cross'},
                provenance=str(d.relative_to(ROOT)).replace('\\', '/')))
    # 3. V15 Group A (runs/v15/eval/*.json rows; cross600)
    for d in sorted((ROOT / 'runs/v15/eval').glob('W_dim*.json')):
        data = read_json(d)
        if 'rows' not in data:
            continue
        m = cross600_means(data['rows'])
        if m:
            entries.append(entry(
                'V15_' + d.stem, 'WEAVE-V15', 'q2a_oracle_cross600', 'S10',
                **{k: v for k, v in m.items() if k != 'n_cross'},
                provenance=str(d.relative_to(ROOT)).replace('\\', '/')))
    # 4. V10 anchors (750口径, LD baseline merged dev+test)
    v10 = read_json(ROOT / 'docs/v10/V10_RESULTS.md')
    if v10 is None:
        entries.append(entry(
            'V10_LD4', 'WEAVE-V10', 'q2a_750_merged', 'S10',
            csd_top1=.3517, lpips=.4872,
            provenance='docs/v10/V10_RESULTS.md'))
        entries.append(entry(
            'V10_E4', 'WEAVE-V10', 'q2a_750_merged', 'S10',
            dino_s=.4415, style_affinity=.5616,
            provenance='docs/v10/V10_RESULTS.md'))
    # 5. V17 arms (appear as the queue lands measured dirs)
    for d in sorted((ROOT / 'runs/v17/measured').glob('*/eval.json')):
        data = read_json(d)
        if not data:
            continue
        m = cross600_means(data['rows'])
        if m:
            entries.append(entry(
                'V17_' + d.parent.name, 'WEAVE-V17', 'q2a_oracle_cross600',
                'S10', **{k: v for k, v in m.items() if k != 'n_cross'},
                provenance=str(d.relative_to(ROOT)).replace('\\', '/')))
    return entries


def pareto_flags(entries):
    """Non-dominated set per protocol on the four preregistered axes
    (dino_s ↑, dino_c ↑, csd_top1 ↑, lpips ↓)."""
    for e in entries:
        e['pareto'] = False
    groups = {}
    for e in entries:
        if None in (e.get('dino_s'), e.get('dino_c'), e.get('csd_top1'),
                    e.get('lpips')):
            continue
        groups.setdefault(e['protocol'], []).append(e)
    for proto, group in groups.items():
        def vec(e):
            return (e['dino_s'], e['dino_c'], e['csd_top1'], -e['lpips'])
        for e in group:
            ve = vec(e)
            dominated = any(
                all(o >= a for o, a in zip(vec(o2), ve)) and
                any(o > a for o, a in zip(vec(o2), ve))
                for o2 in group if o2 is not e)
            e['pareto'] = not dominated
    return entries


def write_outputs(entries):
    PERF.mkdir(parents=True, exist_ok=True)
    (PERF / 'metrics.json').write_text(json.dumps(entries, ensure_ascii=False,
                                                  indent=1), encoding='utf-8')
    cols = ['method', 'family', 'protocol', 'decoder', 'pareto', 'dino_s',
            'dino_c', 'csd_top1', 'csd_margin', 'lpips', 'clip_s',
            'style_affinity', 'rank', 'provenance']
    with (PERF / 'metrics.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(entries)
    print(f'{len(entries)} entries -> metrics.csv/json')


def plot(entries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups = {}
    for e in entries:
        if None in (e.get('dino_s'), e.get('dino_c'), e.get('csd_top1'),
                    e.get('lpips')):
            continue
        groups.setdefault(e['protocol'], []).append(e)
    for proto, group in groups.items():
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        for ax, xk, yk, xlabel, ylabel, invert in (
                (axes[0], 'lpips', 'csd_top1', 'LPIPS (lower better)',
                 'CSD Target Top-1', False),
                (axes[1], 'lpips', 'dino_s', 'LPIPS (lower better)',
                 'DINO-S', False)):
            for e in group:
                color = '#d62728' if e.get('pareto') else (
                    '#1f77b4' if e['family'].startswith('WEAVE') else
                    '#7f7f7f')
                marker = '*' if e.get('pareto') else (
                    'o' if e['family'].startswith('WEAVE') else 'x')
                size = 140 if e.get('pareto') else 40
                ax.scatter(e[xk], e[yk], c=color, marker=marker, s=size,
                           alpha=.85, edgecolors='k', linewidths=.4)
                if e.get('pareto') or e['family'].startswith('WEAVE'):
                    ax.annotate(e['method'], (e[xk], e[yk]), fontsize=6,
                                alpha=.85)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f'{proto}: {ylabel} vs {xlabel}')
            ax.grid(alpha=.3)
        fig.suptitle(f'Pareto frontier ({proto}) — red star = non-dominated',
                     fontsize=11)
        fig.tight_layout()
        out = PERF / f'pareto_{proto}.png'
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f'plot -> {out.name}')


def main():
    entries = collect()
    entries = pareto_flags(entries)
    write_outputs(entries)
    plot(entries)
    pareto = [e['method'] for e in entries if e.get('pareto')]
    print('pareto members:', pareto)


if __name__ == '__main__':
    main()
