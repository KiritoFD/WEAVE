"""Build the V17 36-hour manifest: preregistered main arms + repeat pool.

python -m exp.v17.build_manifest --out exp/v17/manifest_36h.json
Main arms run unconditionally in preregistered order (P0-P5).  The repeat
pool (S5/S6/H4/F1 seeds 1..32, round-robin) is consumed by the queue until
the 36h target; 24h is the hard floor before any early stop would be legal.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

POOL = 'I:/Latent_Style/pseudo_gt_24h/orig_mean_fp16'


def arm(arm_id, family, seed=0, extra=None):
    return {'arm_id': arm_id, 'family': family, 'seed': seed,
            'config': {**(extra or {}), 'arm_id': arm_id, 'family': family,
                       'pool': POOL}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='exp/v17/manifest_36h.json')
    a = ap.parse_args(argv)
    arms = []
    # ---- P0: condition paths (~1.5h)
    arms += [arm('P0_tcurve', 'probe_tcurve')]
    arms += [arm(f'P0_cfg{w}', 'cfg', extra={'w': w}) for w in (2, 4)]
    arms += [arm('P0_cfg_early', 'cfg', extra={'w': 3, 'cfg_schedule': 'early'}),
             arm('P0_cfg_late', 'cfg', extra={'w': 3, 'cfg_schedule': 'late'})]
    arms += [arm('P0_swap_t0', 'labelswap', extra={'tau0': 0.0}),
             arm('P0_swap_t07', 'labelswap', extra={'tau0': 0.7})]
    # ---- P1: teachers T0-T4 (~30 min)
    arms += [arm(f'P1_{v}', 'teacher', extra={'variant': v,
                                              'calibrate_r1': v == 'T1'})
             for v in ('T0', 'T1', 'T2', 'T3', 'T4')]
    # ---- P2: students S0-S8 (~6.5h) + evals
    for s in ['S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S8']:
        arms.append(arm(f'P2_{s}_train', 'student', extra={'kind': s}))
    for s in ['S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S8']:
        arms.append(arm(f'P2_{s}_eval', 'student_eval',
                        extra={'kind': s, 'wrong_label': s in ('S0', 'S6')}))
    arms += [arm('P2_S7_train', 'student', extra={'kind': 'S7'}),
             arm('P2_S7_eval', 'student_eval', extra={'kind': 'S7'})]
    # ---- P3: signed RGB-H2 (~2h)
    arms += [arm('P3_H1_stats', 'h2', extra={'kind': 'H1'}),
             arm('P3_H2arm_eval', 'h2', extra={'kind': 'H2arm'}),
             arm('P3_H3_train', 'h2', extra={'kind': 'H3'}),
             arm('P3_H4_train', 'h2', extra={'kind': 'H4'})]
    for kind in ('H3', 'H4'):
        for variant in ('full', 'nomask', 'wronglabel'):
            arms.append(arm(f'P3_{kind}_{variant}_eval', 'h2_eval',
                            extra={'kind': kind, 'variant': variant,
                                   f'{kind}_dir': f'runs/v17/P3_{kind}_train'}))
    arms += [arm('P3_H4_srctgt_eval', 'h2_eval',
                 extra={'kind': 'H4', 'variant': 'src_tgt',
                        'H4_dir': 'runs/v17/P3_H4_train'})]
    # ---- P4: flow residual (~3h)
    for mode in ('F0', 'F1', 'F2', 'F3'):
        arms.append(arm(f'P4_{mode}_train', 'fm', extra={'mode': mode}))
    for mode in ('F0', 'F1', 'F2', 'F3'):
        arms.append(arm(f'P4_{mode}_eval', 'fm_eval',
                        extra={'mode': mode,
                               'residual_dir': f'runs/v17/P4_{mode}_train'}))
    # ---- P5: preregistered combos (~1h)
    for combo, need in [('S6+H4', ('S6', 'H4')), ('S5+H4', ('S5', 'H4')),
                        ('F1+H4', ('F1', 'H4'))]:
        extra = {'parts': [f'student:{need[0]}' if need[0].startswith('S')
                           else f'fm:{need[0]}', f'h2:{need[1]}'],
                 f'{need[0]}_dir': f'runs/v17/P2_{need[0]}_train'
                 if need[0].startswith('S') else f'runs/v17/P4_{need[0]}_train',
                 f'{need[1]}_dir': f'runs/v17/P3_{need[1]}_train'}
        arms.append(arm(f'P5_{combo.replace("+", "_")}', 'combo',
                        extra={k: (f'runs/v17/P2_{v}' if k.endswith('_dir')
                                   and 'S' in k else v)
                               for k, v in extra.items()}))
    manifest = {
        'version': '2026-11-16',
        'hours': {'min_valid': 24, 'target': 36},
        'main_arms': arms,
        'repeat_pool': {'arms': ['S5', 'S6', 'H4', 'F1'],
                        'seeds': list(range(1, 33)),
                        'order': 'round-robin arms then seeds',
                        'train_updates': {'S5': 12000, 'S6': 12000,
                                          'F1': 8000},
                        'note': 'S5/S6 are students (T0 pairs); H4 is the '
                                'H2 head; F1 the flow residual.  Each '
                                'repeat trains + evals full 750.'},
        'long_run': {'arms': ['S5', 'S6', 'H4', 'F1'],
                     'updates': [24000, 48000], 'seeds': [0, 1, 2]},
        'stop_rules': 'no new repeat launches at 36h; in-flight arm finishes; '
                      'never stop before 24h valid task time unless '
                      'infrastructure failure',
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(manifest, indent=1), encoding='utf-8')
    print(f'manifest: {len(arms)} main arms -> {a.out}')


if __name__ == '__main__':
    main()
