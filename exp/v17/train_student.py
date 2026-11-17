"""P2 student training S0-S8 (12k updates, eff. batch 32, Seed-fixed).

Phase II (last 50% steps, S5/S6/S8 only): bounded soft norm with
a_target = stopgrad(clamp(cos,0,.8)) * RMS(teacher); S8 uses the hard
stopgrad(rho)>0.6 gate as an ABLATION — no test750-derived thresholds.
Every 1000 updates a full-state ckpt is written; 2000/4000/8000/final keep
the preregistered trajectory.  holdout.json gets direction/where/RMS fits.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import torch

from .adapter import build_student, call_student
from .checkpoint import save_ckpt
from .common import OUT, autocast, cosine_lr, heartbeat, save, sha
from .losses import student_loss


def load_pairs(variant: str):
    return torch.load(OUT / 'pairs' / f'teacher_pairs_{variant}.pt',
                      map_location='cpu', weights_only=False)


def _canon_pid(pid):
    import unicodedata as _u
    return _u.normalize('NFC', str(pid).replace('\\', '/')
                        .split('/')[-1].split('__')[-1]).casefold()


def train_student(kind: str, pairs_path: Path, out_dir: Path, *,
                  updates=12000, micro=8, accum=4, seed=20261117,
                  dino_tokens_path=None, lr=3e-4):
    from .checkpoint import find_resume, load_ckpt
    out_dir.mkdir(parents=True, exist_ok=True)
    tokens = tok_map = ho_tokens = ho_map = None
    pairs = torch.load(pairs_path, map_location='cpu', weights_only=False)
    torch.manual_seed(seed)
    net = build_student(kind).cuda()
    n_params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=.01)
    gen = torch.Generator().manual_seed(seed)

    ho_map = None
    if kind in ('S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8'):
        cpath = dino_tokens_path or (OUT / 'dino_cache' / 'pool_tokens.pt')
        cache = torch.load(cpath, map_location='cpu', weights_only=False)
        tokens = cache['tokens']['train']['tokens'].float()
        ho_tokens = cache['tokens']['holdout']['tokens'].float()
        # Always key by canonical painting id: pair sampling seeds differ
        # across teacher variants, so literal ids may not match the cache.
        def _canon(pid):
            import unicodedata as _u
            return _u.normalize('NFC', str(pid).replace('\\', '/')
                                .split('/')[-1].split('__')[-1]).casefold()
        tok_map = {_canon(pid): i for i, pid in
                   enumerate(cache['tokens']['train']['ids'])}
        ho_map = {_canon(pid): i for i, pid in
                  enumerate(cache['tokens']['holdout']['ids'])}

    tr = pairs['train']
    ho = pairs.get('holdout') or pairs.get('calib')
    src_label = tr['sid'].clone()
    phase2_from = updates // 2
    start_step = 0
    curves = []
    ck = find_resume(out_dir)
    if ck is not None and (out_dir / 'ckpt_latest.pt').exists():
        state = load_ckpt(out_dir / 'ckpt_latest.pt', model=net, optimizer=opt)
        start_step = state['step']
        print(f'{kind}: resume at {start_step}', flush=True)

    net.train()
    t0 = time.time()
    for step in range(start_step + 1, updates + 1):
        phase2 = kind in ('S5', 'S6', 'S7', 'S8') and step > phase2_from
        hard_gate = kind == 'S8'
        opt.zero_grad(set_to_none=True)
        for _ in range(accum):
            idx = torch.randint(0, len(tr['tid']), (micro,), generator=gen)
            base = tr['base'][idx].cuda().float()
            content = tr['content'][idx].cuda().float()
            delta_t = tr['delta'][idx].cuda().float()
            mask = tr['mask'][idx].cuda().float() if 'mask' in tr else None
            tid = tr['tid'][idx].cuda()
            slabel = tr['sid'][idx].cuda()
            dtok = None
            if tokens is not None:
                # pairs were built with repeat_interleave(5): row // 5 = source
                row_src = torch.tensor([
                    tok_map[_canon_pid(tr['source_ids'][int(i)])]
                    for i in (idx // 5).tolist()])
                dtok = tokens[row_src].cuda()
            with autocast():
                delta, norm_scale, mask_logits = call_student(
                    net, kind, base, content, tid, dtok, slabel)
            delta = delta.float()
            rho = None
            a_target = None
            if phase2:
                with torch.no_grad():
                    rho = torch.nn.functional.cosine_similarity(
                        delta.flatten(1), delta_t.flatten(1), dim=1)
                    t_rms = delta_t.flatten(1).norm(dim=1)
                if hard_gate:
                    rho = torch.where(rho > .6, rho,
                                      torch.zeros_like(rho))
                    a_target = rho * t_rms
            loss, parts = student_loss(delta, delta_t,
                                       mask_logits=mask_logits
                                       if mask_logits is not None else None,
                                       teacher_mask=mask, phase2=phase2,
                                       rho=rho, a_target=a_target,
                                       lam_norm=.1)
            (loss / accum).backward()
        grad = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
        if not torch.isfinite(grad):
            raise FloatingPointError(f'nonfinite grad at {step}')
        for pg in opt.param_groups:
            pg['lr'] = cosine_lr(step, updates, lr)
        opt.step()
        if step == 1 or step % 100 == 0:
            row = {'step': step, 'lr': cosine_lr(step, updates, lr),
                   'phase2': phase2, 'grad_norm': float(grad),
                   'wall_sec': time.time() - t0, **parts}
            curves.append(row)
            print(f"{kind} {step}/{updates} loss={parts['loss']:.5f} "
                  f"cos={parts.get('cos', float('nan')):.4f} "
                  f"{row['wall_sec']:.0f}s", flush=True)
        if step % 1000 == 0 or step == updates:
            save_ckpt(out_dir / f'ckpt_step_{step}.pt', model=net,
                      optimizer=opt, step=step, gen_state=gen.get_state())
            save_ckpt(out_dir / 'ckpt_latest.pt', model=net, optimizer=opt,
                      step=step, gen_state=gen.get_state())
            heartbeat(out_dir, 'training', {'step': step, 'updates': updates})
    save(out_dir / 'train.jsonl', curves)
    net.eval()
    ho_fit = holdout_fit(net, ho, kind, ho_tokens, ho_map)
    save(out_dir / 'holdout.json', ho_fit)
    torch.save({'model': net.cpu().state_dict(), 'kind': kind,
                'updates': updates, 'seed': seed, 'n_params': n_params,
                'pairs_sha256': sha(pairs_path)},
               out_dir / 'student.pt')
    return n_params


@torch.no_grad()
def holdout_fit(net, ho, kind, ho_tokens=None, ho_map=None, batch=16):
    out = {}
    for name, sel in [('cross', ho['sid'] != ho['tid']), ('all', None)]:
        m = torch.ones(len(ho['tid']), dtype=torch.bool) if sel is None else sel
        sqerr = energy = dot = pn = tn = 0.
        n = 0
        for st in range(0, int(m.sum()), batch):
            rows = m.nonzero(as_tuple=True)[0][st:st + batch]
            base = ho['base'][rows].cuda().float()
            content = ho['content'][rows].cuda().float()
            dt = ho['delta'][rows].cuda().float()
            tid = ho['tid'][rows].cuda()
            slabel = ho['sid'][rows].cuda()
            dtok = None
            if ho_tokens is not None:
                row_src = torch.tensor([
                    ho_map[_canon_pid(ho['source_ids'][int(r)])] for r in
                    (rows // 5).tolist()])
                dtok = ho_tokens[row_src].cuda()
            with autocast():
                delta, _, _ = call_student(net, kind, base, content, tid,
                                           dtok, slabel)
            p, t = delta.flatten(1), dt.flatten(1)
            sqerr += float((p - t).square().sum())
            energy += float(t.square().sum())
            dot += float((p * t).sum())
            pn += float(p.square().sum())
            tn += float(t.square().sum())
            n += p.shape[0]
        out[name] = {'relative_mse': sqerr / max(energy, 1e-20),
                     'direction_cosine': dot / max((pn * tn) ** .5, 1e-20),
                     'prediction_rms': (pn / max(n * 16384, 1)) ** .5,
                     'teacher_rms': (tn / max(n * 16384, 1)) ** .5,
                     'n': int(m.sum())}
    return out
