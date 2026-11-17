"""V17 unified arm entry (implementation contract, plan §4).

  python -m exp.v17.run --config exp/v17/configs/<ARM>.json --seed N \
      --resume auto --out runs/v17/<ARM>_sN [--dry-run] [--eval full750]

Families: probe_tcurve / cfg / labelswap / teacher / student / h2 / fm /
combo.  --dry-run runs one small forward/backward and exits WITHOUT touching
Test750.  --eval full750 is the independent scoring entry (generation ->
S10 -> measure -> DONE.json).  DONE.json is written only by the verifier.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

import torch
import torch.nn.functional as F

from .common import (EVAL_SEED, OUT, STYLES, TAU_EVAL, autocast,
                     heartbeat, save, seed_all, sha, utc)
from exp.v16 import load_base


def _pool(cfg):
    return cfg.get('pool', 'I:/Latent_Style/pseudo_gt_24h/orig_mean_fp16')


def _split(pool, cfg):
    from .data import balanced_pool
    return balanced_pool(pool, train_each=cfg.get('train_each', 100),
                         hold_each=cfg.get('hold_each', 20),
                         seed=cfg.get('split_seed', 20261117))


# ---------------------------------------------------------------- P0
def arm_probe_tcurve(cfg, out: Path, seed: int, dry: bool):
    from .probe import t_curve
    sp, _manifest = _split(_pool(cfg), cfg)
    if dry:
        t_curve(sp['train']['z'][:40], sp['train']['sid'][:40],
                n_per_class=8, seed=seed)
        return {'ok': True}
    rows = t_curve(sp['train']['z'], sp['train']['sid'], seed=seed)
    save(out / 't_curve.json', {'rows': rows, 'seed': seed,
                                'note': 'network-estimated conditional '
                                        'response, not theoretical delta-v'})
    return {'ok': True}


def arm_cfg(cfg, out: Path, seed: int, dry: bool):
    from .probe import run_cfg_arms
    from exp.v16 import tasks_and_sources
    if dry:
        return {'ok': True}
    w, tau0 = cfg.get('w', 1.0), cfg.get('tau0', .7)
    sched = None
    if cfg.get('cfg_schedule') == 'early':
        sched = lambda t: 3. if t <= .35 else 1.        # noqa: E731
    if cfg.get('cfg_schedule') == 'late':
        sched = lambda t: 3. if t > .35 else 1.         # noqa: E731
    name = cfg['arm_id']
    tasks, source = tasks_and_sources()
    from .eval750 import evaluate_arm, _cfg_sample
    from exp.v16 import load_base

    def op(group, z, sid, tid, eps):
        model, _ = load_base()
        return _cfg_sample(model, z, sid, tid, eps, tau0, w=w,
                           w_schedule=sched)

    summary, _ = evaluate_arm(name, op, tau0=tau0,
                              extra_meta={'cfg_w': w,
                                          'schedule': cfg.get('cfg_schedule')})
    return {'ok': True, 'summary': summary}


def arm_labelswap(cfg, out: Path, seed: int, dry: bool):
    from .probe import label_swap
    from exp.v16 import tasks_and_sources
    from .eval750 import evaluate_arm
    from exp.v16 import load_base
    tasks, source = tasks_and_sources()
    tau0 = cfg.get('tau0', 0.0)
    if dry:
        return {'ok': True}
    summaries = {}
    for label in range(5):
        name = f"{cfg['arm_id']}_label{label}"

        def op(group, z, sid, tid, eps, _label=label):
            model, _ = load_base()
            zt = tau0 * z + (1 - tau0) * eps
            dt = (1 - tau0) / 20
            lab = torch.full((len(zt),), _label, device=zt.device,
                             dtype=torch.long)
            for k in range(20):
                tt = torch.full((len(zt),), tau0 + k * dt, device=zt.device)
                with autocast(zt.device):
                    v = model(zt, lab, tt).float()
                zt = zt + dt * v
            return zt
        summary, _ = evaluate_arm(name, op, tau0=tau0,
                                  extra_meta={'swap_label': label})
        summaries[f'label{label}'] = summary
    # post-hoc label-following confusion from csd_scores_5
    from .common import read
    import numpy as np
    conf = []
    for label in range(5):
        d = read(OUT / 'measured' / f"{cfg['arm_id']}_label{label}" /
                 'eval.json')
        follow = float(np.mean([int(np.argmax(r['csd_scores_5'])) == label
                                for r in d['rows']]))
        conf.append({'label': label, 'followed_fraction': follow})
    save(out / 'label_swap.json', {'confusion': conf, 'tau0': tau0})
    return {'ok': True}


# ---------------------------------------------------------------- P1
def arm_teacher(cfg, out: Path, seed: int, dry: bool):
    from .teacher import prepare_pairs
    if dry:
        return {'ok': True}
    bias = None
    variant = cfg.get('variant', 'T0')
    if cfg.get('calibrate_r1'):
        from .teacher import r1_logit_bias
        bias = r1_logit_bias()
    path = prepare_pairs(variant, _pool(cfg), batch=cfg.get('batch', 8),
                         seed=seed, calib_bias=bias)
    return {'ok': True, 'pairs': str(path)}


# ---------------------------------------------------------------- P2
def arm_student(cfg, out: Path, seed: int, dry: bool):
    from .train_student import train_student
    from .dino_cache import cache_tokens
    from .adapter import build_student, call_student
    kind = cfg['kind']
    if dry:
        net = build_student(kind).cuda()
        b = torch.randn(2, 4, 64, 64).cuda()
        c = torch.randn(2, 4, 64, 64).cuda()
        tid = torch.tensor([0, 1]).cuda()
        tok = torch.randn(2, 384, 16, 16).cuda() if kind != 'S0' else None
        delta, scale, mask = call_student(net, kind, b, c, tid, tok,
                                          torch.tensor([1, 2]).cuda())
        assert torch.isfinite(delta).all()
        if kind == 'S0':
            assert delta.abs().max() == 0, 'S0 must be identity at init'
        if kind in ('S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8'):
            assert delta.abs().max() == 0, 'custom students must be identity'
        return {'ok': True}
    sp, _manifest = _split(_pool(cfg), cfg)
    if kind in ('S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8'):
        cache_tokens(sp)
    pairs_variant = cfg.get('pairs', 'T0' if kind != 'S7' else 'T4')
    pairs = OUT / 'pairs' / f'teacher_pairs_{pairs_variant}.pt'
    train_student(kind, pairs, out, updates=cfg.get('updates', 12000),
                  micro=cfg.get('micro', 8), accum=cfg.get('accum', 4),
                  seed=seed)
    return {'ok': True}


def arm_student_eval(cfg, out: Path, seed: int, dry: bool):
    """Full-750 evaluation of a trained student (+ optional wrong-label)."""
    from .eval750 import evaluate_arm
    from .adapter import build_student, call_student
    from .dino_cache import cache_tokens
    from exp.v16 import tasks_and_sources
    from exp.v16.distill import generate as v16_generate
    from src.model import canonicalizer
    kind = cfg['kind']
    student_dir = Path(cfg.get('student_dir') or
                       (OUT / f'P2_{kind}_train'))
    if dry:
        return {'ok': True}
    ck = torch.load(student_dir / 'student.pt', map_location='cpu',
                    weights_only=False)
    net = build_student(kind).cuda()
    net.load_state_dict(ck['model'])
    net.eval()
    base_model, _ = load_base()          # frozen V15-96 generator
    canon = canonicalizer('cuda')
    need_dino = kind in ('S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8')
    if need_dino:
        cache_tokens(_split(_pool(cfg), cfg))   # train/holdout only
    tasks, source = tasks_and_sources()
    name = cfg['arm_id']

    def op(group, z, sid, tid, epsl):
        # CRITICAL: students are residual-over-V15-base, NOT over raw source
        # (matching V16 B_adapter eval).  Without this, output ≈ source image.
        with torch.no_grad(), autocast():
            base = v16_generate(base_model, z, tid, epsl)
            content = canon(z, mode='c1_gentle', augment=False)
            dtok = student_tokens(group, base) if need_dino else None
            delta, scale, mask_logits = call_student(net, kind, base,
                                                     content, tid, dtok, sid)
            if scale is not None:
                delta = delta * scale.view(-1, 1, 1, 1).clamp(0, 2)
        return base + delta

    def student_tokens(group, base):
        from .dino_cache import encode_tokens_for_images
        from src.utils import load_s10_decoder, decode_s10
        from PIL import Image
        dec = load_s10_decoder('cuda')
        rgb = decode_s10(dec, base, batch=len(base))
        imgs = [Image.fromarray(x) for x in rgb.cpu().permute(0, 2, 3, 1)
                .mul(255).round().clamp(0, 255).byte().numpy()]
        return encode_tokens_for_images(imgs)

    summary, _ = evaluate_arm(name, op, tau0=.7,
                              extra_meta={'student_kind': kind,
                                          'load_dir': str(student_dir)})
    wrong = None
    if cfg.get('wrong_label'):
        def opw(group, z, sid, tid, epsl):
            with torch.no_grad(), autocast():
                base = v16_generate(base_model, z, tid, epsl)
                content = canon(z, mode='c1_gentle', augment=False)
                dtok = student_tokens(group, base) if need_dino else None
                delta, scale, _ = call_student(net, kind, base, content,
                                               (tid + 2) % 5, dtok, sid)
                if scale is not None:
                    delta = delta * scale.view(-1, 1, 1, 1).clamp(0, 2)
            return base + delta
        summary_w, _ = evaluate_arm(name + '_wrong', opw, tau0=.7,
                                    extra_meta={'student_kind': kind,
                                                'wrong_label': True})
        wrong = summary_w
    return {'ok': True, 'summary': summary, 'wrong': wrong}


# ---------------------------------------------------------------- P3
def arm_h2(cfg, out: Path, seed: int, dry: bool):
    kind = cfg['kind']
    if dry:
        from .h2 import H2Head, signed_gain
        net = H2Head(bias=(kind == 'H4')).cuda()
        rgb = torch.rand(2, 3, 512, 512).cuda()
        tid = torch.tensor([0, 1]).cuda()
        lat = signed_gain(rgb, tid, torch.rand(5, 3).cuda().abs())
        assert torch.isfinite(lat).all()
        return {'ok': True}
    from .h2 import H2Head, h1_stats, corruption_batch, signed_gain
    sp, _manifest = _split(_pool(cfg), cfg)
    stats = h1_stats(pool_split=sp)
    torch.save(sp, OUT / 'h2' / 'pool_split.pt') if not \
        (OUT / 'h2' / 'pool_split.pt').exists() else None
    if kind == 'H1':
        return {'ok': True, 'stats': stats}
    if kind == 'H2arm':
        from .eval750 import evaluate_arm
        from exp.v16 import tasks_and_sources
        tasks, source = tasks_and_sources()
        target_rms = torch.tensor([[stats[STYLES[s]]['h2_rms_per_direction'][d]
                                    for d in range(9)] for s in range(5)])

        def op(group, z, sid, tid, eps):
            from src.utils import load_s10_decoder, decode_s10
            dec = load_s10_decoder('cuda')
            rgb = decode_s10(dec, z, batch=len(z)).clamp(0, 1)
            return signed_gain(rgb, tid, target_rms)
        summary, _ = evaluate_arm(cfg['arm_id'], op, tau0=.7,
                                  output_rgb=True,
                                  extra_meta={'h2_kind': 'analytic_signed'})
        return {'ok': True, 'summary': summary}
    if kind in ('H3', 'H4'):
        train_h2_head(kind, sp, stats, out, seed=seed,
                      updates=cfg.get('updates', 4000))
        return {'ok': True}
    raise ValueError(kind)


def train_h2_head(kind, sp, stats, out: Path, seed=20261117, updates=4000):
    """Real-target restoration of synthetic five-style spectral corruption.
    Loss: MSE(H2_correct, H2_real) + region energy matching.  LL2/H1 fixed."""
    from .h2 import H2Head, corruption_batch
    from .common import cosine_lr
    from .checkpoint import find_resume, load_ckpt, save_ckpt
    torch.manual_seed(seed)
    net = H2Head(bias=(kind == 'H4')).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=.01)
    gen = torch.Generator().manual_seed(seed)
    dec_path = None
    z = sp['train']['z'].cuda()
    sid = sp['train']['sid'].cuda()
    t0 = time.time()
    start = 0
    ck = find_resume(out)
    if ck is not None and (out / 'ckpt_latest.pt').exists():
        st = load_ckpt(out / 'ckpt_latest.pt', model=net, optimizer=opt)
        start = st['step']
    for step in range(start + 1, updates + 1):
        idx = torch.randint(0, len(z), (4,), generator=gen).to(z.device)
        rgb = s10_decode_latents(z[idx])
        tid = sid[idx]
        h2_real, h2_corrupt, bands = corruption_batch(rgb, tid, stats, gen=gen)
        edge = sobel_edge(rgb)
        h2_out, gamma = net(h2_corrupt, edge, tid)
        loss = F.mse_loss(h2_out, h2_real)
        loss = loss + .1 * (h2_out.square().mean((2, 3)) -
                            h2_real.square().mean((2, 3))).square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
        for pg in opt.param_groups:
            pg['lr'] = cosine_lr(step, updates, 2e-4)
        opt.step()
        if step == 1 or step % 200 == 0:
            print(f'{kind} {step}/{updates} loss={float(loss):.6f} '
                  f'{time.time()-t0:.0f}s', flush=True)
        if step % 1000 == 0 or step == updates:
            save_ckpt(out / f'ckpt_step_{step}.pt', model=net, optimizer=opt,
                      step=step)
            save_ckpt(out / 'ckpt_latest.pt', model=net, optimizer=opt,
                      step=step)
            heartbeat(out, 'training', {'step': step})
    torch.save({'model': net.cpu().state_dict(), 'kind': kind},
               out / 'h2_head.pt')


def s10_decode_latents(z):
    from src.utils import load_s10_decoder, decode_s10
    dec = load_s10_decoder('cuda')
    return decode_s10(dec, z, batch=len(z)).to(z.device).clamp(0, 1)


def sobel_edge(rgb):
    from exp.v11b.bands import dwt2, sobel_mag
    edge = sobel_mag(dwt2(rgb)['LL2'])
    return (edge / edge.amax((1, 2, 3), keepdim=True).clamp_min(1e-8))


def arm_h2_eval(cfg, out: Path, seed: int, dry: bool):
    """Evaluate a trained H3/H4 head (or its controls) on the full 750."""
    from .eval750 import evaluate_arm
    from .h2 import H2Head
    from exp.v11b.bands import dwt2, idwt2, sobel_mag
    from exp.v16 import tasks_and_sources
    from src.utils import load_s10_decoder, decode_s10
    kind = cfg['kind']
    variant = cfg.get('variant', 'full')   # full | nomask | wronglabel | src_tgt
    if dry:
        return {'ok': True}
    tasks, source = tasks_and_sources()
    head = None
    if kind in ('H3', 'H4'):
        head = H2Head(bias=(kind == 'H4')).cuda()
        sd = torch.load(Path(cfg['head_dir']) / 'h2_head.pt',
                        map_location='cpu', weights_only=False)
        head.load_state_dict(sd['model'])
        head.eval()
    dec = load_s10_decoder('cuda')

    def op(group, z, sid, tid, epsl):
        rgb = decode_s10(dec, z, batch=len(z)).clamp(0, 1)
        bands = dwt2(rgb)
        edge = sobel_mag(bands['LL2'])
        edge = edge / edge.amax((1, 2, 3), keepdim=True).clamp_min(1e-8)
        label = (tid + 2) % 5 if variant == 'wronglabel' else tid
        if head is None:
            return rgb
        beta_trust = bands['H2'].square().mean(1, keepdim=True).sqrt() \
            if kind == 'H4' else None
        h2_out, _ = head(bands['H2'], edge, label, beta_trust)
        if variant == 'nomask':
            delta = h2_out - bands['H2']
        else:
            em = (1 - .3 * edge)
            delta = (h2_out - bands['H2']) * em
        if variant == 'src_tgt':
            delta = delta * 0.
        return idwt2(bands['LL2'], bands['H2'] + delta, bands['H1']) \
            .clamp(0, 1)
    summary, _ = evaluate_arm(cfg['arm_id'], op, tau0=.7,
                              output_rgb=True,
                              extra_meta={'h2_kind': kind,
                                          'variant': variant})
    return {'ok': True, 'summary': summary}


# ---------------------------------------------------------------- P4
def arm_fm(cfg, out: Path, seed: int, dry: bool):
    from .fm_residual import VelocityResidual, train_fm
    mode = cfg['mode']
    if dry:
        net = VelocityResidual().cuda()
        z = torch.randn(2, 4, 64, 64).cuda()
        tau = torch.rand(2).cuda()
        u = net(z, torch.tensor([0, 1]).cuda(), tau)
        assert torch.isfinite(u).all() and u.abs().max() == 0
        return {'ok': True}
    sp, _manifest = _split(_pool(cfg), cfg)
    out_dir = train_fm(mode, sp, out, seed=seed,
                       updates=cfg.get('updates', 8000))
    return {'ok': True, 'dir': str(out_dir)}


def arm_fm_eval(cfg, out: Path, seed: int, dry: bool):
    from .eval750 import evaluate_arm
    from .fm_residual import VelocityResidual, centered_output
    from exp.v16 import load_base, tasks_and_sources
    mode = cfg['mode']
    if dry:
        return {'ok': True}
    tasks, source = tasks_and_sources()
    resid = VelocityResidual().cuda()
    sd = torch.load(Path(cfg['residual_dir']) / 'residual.pt',
                    map_location='cpu', weights_only=False)
    resid.load_state_dict(sd['model'])
    resid.eval()
    base, _ = load_base()

    def op(group, z, sid, tid, epsl):
        zt = .7 * z + .3 * epsl
        dt = .3 / 20
        for k in range(20):
            tau = torch.full((len(zt),), .7 + k * dt, device=zt.device)
            with autocast(zt.device):
                v = base(zt, tid, tau).float()
                if mode == 'F2':
                    u = resid(zt, tid, tau)
                else:
                    u = centered_output(resid, zt, tid, tau, mode)
            zt = zt + dt * (v + u)
        return zt
    summary, _ = evaluate_arm(cfg['arm_id'], op, tau0=.7,
                              extra_meta={'fm_mode': mode})
    return {'ok': True, 'summary': summary}


# ---------------------------------------------------------------- P5
def arm_combo(cfg, out: Path, seed: int, dry: bool):
    """S-adapter then decode then H4 head on RGB (P5 preregistered combos)."""
    from .eval750 import evaluate_arm
    from .adapter import build_student, call_student
    from .h2 import H2Head
    from exp.v11b.bands import dwt2, idwt2, sobel_mag
    from exp.v16 import tasks_and_sources
    from src.utils import load_s10_decoder, decode_s10
    if dry:
        return {'ok': True}
    tasks, source = tasks_and_sources()
    parts = cfg['parts']                       # e.g. ["student:S6", "h2:H4"]
    student, h2head = None, None
    for p in parts:
        if p.startswith('student:'):
            kind = p.split(':')[1]
            sdir = Path(cfg.get(f'{kind}_dir') or (OUT / kind /
                                                   f'{kind}_s{cfg.get("load_seed", 0)}'))
            ck = torch.load(sdir / 'student.pt', map_location='cpu',
                            weights_only=False)
            student = (build_student(kind).cuda())
            student.load_state_dict(ck['model'])
            student.eval()
        if p.startswith('h2:'):
            kind = p.split(':')[1]
            hdir = Path(cfg.get(f'{kind}_dir'))
            h2head = H2Head(bias=(kind == 'H4')).cuda()
            sd = torch.load(hdir / 'h2_head.pt', map_location='cpu',
                            weights_only=False)
            h2head.load_state_dict(sd['model'])
            h2head.eval()
    dec = load_s10_decoder('cuda')

    def op(group, z, sid, tid, epsl):
        zt = .7 * z + .3 * epsl
        dt = .3 / 20
        if student is None:
            for k in range(20):
                tau = torch.full((len(zt),), .7 + k * dt, device=zt.device)
                with autocast(zt.device):
                    v = _base(zt, tid, tau).float()
                zt = zt + dt * v
        rgb = decode_s10(dec, zt, batch=len(zt)).clamp(0, 1)
        if h2head is not None:
            bands = dwt2(rgb)
            edge = sobel_mag(bands['LL2'])
            edge = edge / edge.amax((1, 2, 3), keepdim=True).clamp_min(1e-8)
            beta_trust = bands['H2'].square().mean(1, keepdim=True).sqrt()
            h2_out, _ = h2head(bands['H2'], edge, tid, beta_trust)
            delta = (h2_out - bands['H2']) * (1 - .3 * edge)
            rgb = idwt2(bands['LL2'], bands['H2'] + delta,
                        bands['H1']).clamp(0, 1)
        return rgb

    def _base(z, tid, tau):
        m, _ = load_base()
        return m(z, tid, tau).float()

    summary, _ = evaluate_arm(cfg['arm_id'], op, tau0=.7,
                              output_rgb=True,
                              extra_meta={'parts': parts})
    return {'ok': True, 'summary': summary}


# ---------------------------------------------------------------- dispatch
def load_base_cached():
    from exp.v16 import load_base
    return load_base()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--resume', default='auto')
    ap.add_argument('--out', required=True)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--eval', default='full750', choices=['full750', 'none'])
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text(encoding='utf-8-sig'))
    cfg.setdefault('arm_id', Path(a.config).stem)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seed_all(a.seed)
    torch.backends.cudnn.benchmark = False
    save(out / 'resolved_config.json', {
        'config': cfg, 'seed': a.seed, 'utc': utc(),
        'tasks_sha256': sha(Path(__file__).resolve().parents[2] /
                            'eval_protocol/tasks_750.json'),
        'config_sha256': sha(a.config)})
    dispatch = {'probe_tcurve': arm_probe_tcurve, 'cfg': arm_cfg,
                'labelswap': arm_labelswap, 'teacher': arm_teacher,
                'student': arm_student, 'student_eval': arm_student_eval,
                'h2': arm_h2, 'h2_eval': arm_h2_eval, 'fm': arm_fm,
                'fm_eval': arm_fm_eval, 'combo': arm_combo}
    family = cfg['family']
    heartbeat(out, 'running', {'family': family})
    t0 = time.time()
    result = dispatch[family](cfg, out, a.seed, a.dry_run)
    # Referee runs in a SEPARATE process: generation models are fully gone
    # before CSD/DINO/CLIP/LPIPS load, so the 12GB card never OOMs.  Measure
    # every raw payload produced by this arm that has no summary yet.
    raw_root = OUT / 'raw'
    measured_root = OUT / 'measured'
    if not a.dry_run and raw_root.exists():
        import subprocess as _sp
        for raw_dir in sorted(raw_root.glob(f"{cfg['arm_id']}*")):
            if not raw_dir.is_dir() or not (raw_dir / 'meta.json').exists():
                continue
            mdir = measured_root / raw_dir.name
            if (mdir / 'summary.json').exists():
                continue
            print(f'MEASURE_STAGE {raw_dir.name}', flush=True)
            r = _sp.run([sys.executable, '-u', '-m', 'exp.v17.measure_only',
                         '--arm', raw_dir.name], cwd=REPO)
            if r.returncode != 0:
                print(f'MEASURE_FAIL {raw_dir.name}', flush=True)
    save(out / 'result.json', {'ok': True, 'family': family,
                               'wall_sec': time.time() - t0, **result})
    heartbeat(out, 'complete')
    print(f"ARM_DONE {cfg['arm_id']} {family} {time.time()-t0:.0f}s",
          flush=True)


if __name__ == '__main__':
    main()
