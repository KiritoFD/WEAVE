"""V17 universal full-750 evaluation with the DONE.json contract.

One operator per arm produces latents for every Test750 task at fixed paired
noise (seed 20261185, same convention as V16/V15), S10 decode, RawWriter
payloads, src.eval.measure scoring, per-target/confusion summary, and a
DONE.json written ONLY after every assertion passes (750/600/150, finite,
hashes).  The queue admits arms by DONE.json presence + hash match.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch

from .common import EVAL_SEED, OUT, STYLES, TAU_EVAL, autocast, save, sha
from exp.v16 import RawWriter, load_base, tasks_and_sources


def _cfg_sample(model, z, sid, tid, eps, tau0, steps=20, w=1.0,
                w_schedule=None):
    """V15 sampling; w>1 uses the TRAINED p_drop unconditional branch.
    w_schedule(tau) may override w at specific tau (early/late CFG arms)."""
    zt = tau0 * z + (1 - tau0) * eps
    dt = (1 - tau0) / steps
    for k in range(steps):
        tau_s = tau0 + k * dt
        tau = torch.full((len(zt),), tau_s, device=zt.device)
        weight = w if w_schedule is None else w_schedule(tau_s)
        with autocast(zt.device):
            vc = model(zt, tid, tau).float()
            if weight != 1.0:
                vu = model(zt, tid, tau, unconditional=True).float()
                vc = vu + weight * (vc - vu)
        zt = zt + dt * vc
    return zt


@torch.no_grad()
def make_latents(op, tasks, source, device='cuda', batch=32):
    """op(group, z, sid, tid, eps_slice) -> latents (B,4,64,64)."""
    g = torch.Generator().manual_seed(EVAL_SEED)
    eps = torch.randn(source.shape, generator=g)
    out = torch.empty_like(source)
    for st in range(0, 750, batch):
        sl = slice(st, st + batch)
        group = tasks[sl]
        sid = torch.tensor([t['source_style_id'] for t in group], device=device)
        tid = torch.tensor([t['target_style_id'] for t in group], device=device)
        out[sl] = op(group, source[sl].to(device), sid, tid, eps[sl].to(device))
    return out


def evaluate_arm(name: str, op, *, tau0=TAU_EVAL, steps=20, device='cuda',
                 batch=32, measured_root: Path | None = None,
                 extra_meta: dict | None = None, out_root: Path | None = None,
                 output_rgb=False, skip_measure=True):
    """Generate + save raw payloads.  By default does NOT run the referee
    inline: generation models must fully exit before CSD/DINO/CLIP/LPIPS
    load, or the 12GB card OOMs.  Call measure_only.py as a separate process.
    Set skip_measure=False only for tiny debug runs on an idle GPU."""
    from src.utils import load_s10_decoder, decode_s10
    from exp.v16 import load_base
    tasks, source = tasks_and_sources()
    out_root = out_root or (OUT / 'raw')
    raw = out_root / name
    model, _ = load_base() if not output_rgb else (None, None)
    decoder = load_s10_decoder(device) if not output_rgb else None
    writer = RawWriter.__new__(RawWriter)
    writer.path = raw
    writer.path.mkdir(parents=True, exist_ok=True)
    writer.batches, writer.n = [], 0
    g = torch.Generator().manual_seed(EVAL_SEED)
    eps = torch.randn(source.shape, generator=g)
    t0 = time.time()
    for st in range(0, 750, batch):
        sl = slice(st, st + batch)
        group = tasks[sl]
        sid = torch.tensor([t['source_style_id'] for t in group], device=device)
        tid = torch.tensor([t['target_style_id'] for t in group], device=device)
        with torch.no_grad():
            out = op(group, source[sl].to(device), sid, tid, eps[sl].to(device))
            if output_rgb:
                rgb = out
                z_lat = torch.zeros(len(sl), 4, 64, 64)
            else:
                z_lat = out
                rgb = decode_s10(decoder, z_lat, batch=1)
        writer.append([t['task_id'] for t in group], rgb, z_lat)
    meta = {'kind': 'v17', 'arm': name, 'tau0': tau0, 'steps': steps,
            'noise_seed': EVAL_SEED, 'n_tasks': writer.n, 'batches': writer.batches,
            'wall_sec': time.time() - t0, **(extra_meta or {})}
    writer.finish(**meta)
    # Generation done.  Release everything; the referee runs in a SEPARATE
    # process (measure_only.py) so its CSD/DINO/CLIP/LPIPS weights never
    # coexist with the generator on the 12GB card.
    del op
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    if skip_measure:
        return None, raw
    measured = (measured_root or (OUT / 'measured')) / name
    import subprocess, sys
    r = subprocess.run([sys.executable, '-u', '-m', 'src.eval.measure',
                        '--raw', str(raw), '--out', str(measured),
                        '--device', device, '--no-png'],
                       cwd=Path(__file__).resolve().parents[2])
    if r.returncode != 0:
        raise RuntimeError(f'measure failed for {name}')
    return finish_eval(name, measured, raw, meta, out_root)


def finish_eval(name, measured: Path, raw: Path, meta: dict,
                out_root: Path | None = None):
    """Post-measure: summary, confusion, assertions, DONE.json."""
    import numpy as np
    from .common import read, save
    d = read(measured / 'eval.json')
    rows = d['rows']
    assert d['n_total'] == 750 and d['n_cross'] == 600 and d['n_same'] == 150

    def block(rs):
        return {'n': len(rs),
                'dino_s': float(np.mean([r['dino_s'] for r in rs])),
                'dino_c': float(np.mean([r['dino_c'] for r in rs])),
                'lpips': float(np.mean([r['lpips'] for r in rs])),
                'csd_top1': float(np.mean([r['target_top1'] for r in rs])),
                'csd_margin': float(np.mean([r['target_margin'] for r in rs])),
                'dT': float(np.mean([r['dT'] for r in rs])),
                'csd_source_score': float(np.mean(
                    [r['csd_scores_5'][r['src_style_id']] for r in rs])),
                'csd_target_score': float(np.mean(
                    [r['csd_scores_5'][r['tgt_style_id']] for r in rs]))}

    summary = {'cross600': block([r for r in rows if r['is_cross']]),
               'same150': block([r for r in rows if not r['is_cross']]),
               'per_target': {s: block([r for r in rows if r['tgt_style'] == s
                                        and r['is_cross']]) for s in STYLES},
               'confusion_5x5': [[sum(1 for r in rows if r['is_cross'] and
                                      r['tgt_style'] == STYLES[i] and
                                      int(np.argmax(r['csd_scores_5'])) == j)
                                  for j in range(5)] for i in range(5)]}
    save(measured / 'summary.json', summary)
    done = {'arm': name, 'utc_done': True, 'n_tasks': d['n_total'],
            'n_cross': d['n_cross'], 'n_same': d['n_same'],
            'tasks_sha256': sha(Path(__file__).resolve().parents[2] /
                                'eval_protocol/tasks_750.json'),
            'raw_meta_sha256': sha(raw / 'meta.json'),
            'summary': summary, **meta}
    save(Path(out_root or OUT) / name / 'DONE.json', done)
    return summary, Path(out_root or OUT) / name / 'DONE.json'
