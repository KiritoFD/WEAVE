"""V15 Style-Bridge post-training v2: P1-P4 unified entry.

Modes:
  p1  balanced FM continuation (control, whole model, lr 2e-5)
  p2  Style-Bridge FM, whole model low-lr (bridge + FM + retain-teacher)
  p3  Style-Bridge + frozen base + zero-init ContentAdapter
  p4  p3 + style-contrast on real-FM pairs + RMS-limited residual;
      adapter lr 1e-4, last decoder block + out head lr 1e-5

Bridge pairs: z_tilde = T_r(z) via V6 subband moments (strength {0,.5,1}),
tau0 in {0.4,0.7}, endpoint = real z, condition = real style s.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as Fn

from exp.v15.model import V15UNet
from exp.v15.train import load_training_latents, fm_loss, STYLES
from exp.v15.style_bridge import (load_style_targets, sample_bridge,
                                  bridge_loss)
from exp.v15.content_adapter import V15AdapterUNet
from src.model import canonicalizer

REPO = Path(__file__).resolve().parents[2]


def fm_item(model, z_real, sid, tau, teacher=None, canon=None,
            p_drop=0.15):
    """Normal FM forward; returns (v_new, v_star, v_teacher_or_None)."""
    dev = z_real.device
    B = z_real.shape[0]
    eps = torch.randn_like(z_real)
    ta = tau.view(-1, 1, 1, 1)
    z_tau = (1 - ta) * eps + ta * z_real
    v_star = z_real - eps
    drop = torch.rand(B, device=dev) < p_drop
    kw = {}
    if canon is not None:
        kw["content"] = canon(z_real.float(), mode="c1_gentle",
                              augment=False)
    with torch.autocast(dev.type, dtype=torch.bfloat16,
                        enabled=dev.type == "cuda"):
        v = model(z_tau, sid, tau, unconditional=False, **kw).float()
    if drop.any():
        idx = drop.nonzero(as_tuple=True)[0]
        kw2 = {"content": kw["content"][idx]} if kw else {}
        with torch.autocast(dev.type, dtype=torch.bfloat16,
                            enabled=dev.type == "cuda"):
            v[idx] = model(z_tau[idx], sid[idx], tau[idx],
                           unconditional=True, **kw2).float()
    vt = None
    if teacher is not None:
        with torch.no_grad(), torch.autocast(
                dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            vt = teacher(z_tau, sid, tau, unconditional=False).float()
    return v, v_star, vt, z_tau


def contrast_loss(model, z_real, sid, tau, canon=None, T=0.5):
    """P4: correct style-id should predict FM velocity better than wrong ids.
    Returns (loss, e_correct_mean, e_wrong_mean)."""
    dev = z_real.device
    B = z_real.shape[0]
    eps = torch.randn_like(z_real)
    ta = tau.view(-1, 1, 1, 1)
    z_tau = (1 - ta) * eps + ta * z_real
    v_star = z_real - eps
    kw = {"content": canon(z_real.float(), mode="c1_gentle",
                           augment=False)} if canon is not None else {}
    errs = []
    for k in range(5):
        ks = torch.full((B,), k, device=dev, dtype=torch.long)
        with torch.autocast(dev.type, dtype=torch.bfloat16,
                            enabled=dev.type == "cuda"):
            v = model(z_tau, ks, tau, unconditional=False, **kw).float()
        e = ((v - v_star) ** 2).mean(dim=(1, 2, 3)) / \
            (v_star ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-8)
        errs.append(e)
    E = torch.stack(errs, dim=1)                     # (B,5)
    # contrast: -log softmax(-E/T)[sid]
    logits = -E / T
    lse = torch.logsumexp(logits, dim=1)
    loss = (lse - logits.gather(1, sid.view(-1, 1))).mean()
    ec = E.gather(1, sid.view(-1, 1)).mean()
    mask = torch.ones(5, device=dev, dtype=torch.bool)
    ew = []
    for i in range(B):
        m = mask.clone(); m[sid[i]] = False
        ew.append(E[i][m].mean())
    return loss, ec.item(), torch.stack(ew).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["p1", "p2", "p3", "p4"])
    ap.add_argument("--source", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--style-targets",
                    default="runs/v6/cache/indomain_style_targets_v2_k64.pt")
    ap.add_argument("--updates", type=int, default=4000)
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lr-adapter", type=float, default=1e-4)
    ap.add_argument("--lr-decoder", type=float, default=1e-5)
    ap.add_argument("--lambda-fm", type=float, default=1.0)
    ap.add_argument("--lambda-retain", type=float, default=1.0)
    ap.add_argument("--lambda-contrast", type=float, default=0.5)
    ap.add_argument("--rms-frac", type=float, default=0.10,
                    help="P4 residual-RMS cap as fraction of base v RMS")
    ap.add_argument("--contrast-T", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device
    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(a.source, map_location="cpu", weights_only=False)
    cfg = ck["config"]

    def build_base():
        return V15UNet(base_dim=int(cfg["base_dim"]),
                       use_attention=bool(cfg["use_attention"]),
                       style_mode=str(cfg.get("style_mode", "adaln")),
                       n_experts=int(cfg.get("n_experts", 4))).to(dev)

    base = build_base()
    base.load_state_dict(ck["model"])
    if a.mode in ("p3", "p4"):
        model = V15AdapterUNet(base).to(dev)
        model.adapter.apply(lambda m: m.reset_parameters()
                            if hasattr(m, "reset_parameters") else None)
        # zero-init heads are already zero; ensure base frozen
        model.set_base_requires_grad(False)
        teacher = build_base()
        teacher.load_state_dict(ck["model"])
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
    else:
        model = base
        model.train()
        teacher = build_base()
        teacher.load_state_dict(ck["model"])
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    canon = canonicalizer(dev) if a.mode in ("p3", "p4") else None
    style_targets = load_style_targets(a.style_targets, dev) \
        if a.mode in ("p2", "p3", "p4") else None

    z_all, y_all, counts = load_training_latents(a.pool, "cpu")
    print(f"pool {z_all.shape[0]} {counts} mode={a.mode}", flush=True)

    # optimizer param groups
    if a.mode == "p1":
        groups = [{"params": model.parameters(), "lr": a.lr}]
    elif a.mode == "p2":
        groups = [{"params": model.parameters(), "lr": a.lr}]
    elif a.mode == "p3":
        groups = [{"params": model.adapter.parameters(), "lr": a.lr_adapter}]
    else:  # p4: adapter + last decoder block + out head
        dec_params = list(model.base.dec[-1].parameters()) + \
            list(model.base.res_dec[-1].parameters()) + \
            list(model.base.out.parameters())
        groups = [{"params": model.adapter.parameters(), "lr": a.lr_adapter},
                  {"params": dec_params, "lr": a.lr_decoder}]
        for p in dec_params:
            p.requires_grad_(True)
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.updates)

    # balanced index sampler (reuse simple approach)
    from exp.v15.posttrain_4070lp import balanced_batches
    rng = torch.Generator().manual_seed(a.seed)
    batch_iter = balanced_batches(z_all, y_all, a.micro, rng)

    def next_batch():
        idx = next(batch_iter)
        return z_all[idx].to(dev), y_all[idx].to(dev)

    curve = out / "train_curves.jsonl"
    opt_step = micro_in = 0
    run = {}
    t0 = time.time()
    peak = 0
    while opt_step < a.updates:
        z_real, sid = next_batch()
        tau = torch.rand(len(idx := range(len(sid))), device=dev)

        parts = {}
        if a.mode == "p1":
            loss = fm_loss(model, z_real, sid, tau) / a.accum
            parts["fm"] = loss.item() * a.accum
        else:
            loss = 0.0
            # --- FM path (also serves retain + contrast) ---
            v, v_star, vt, _ = fm_item(model, z_real, sid, tau,
                                       teacher=teacher, canon=canon)
            l_fm = Fn.mse_loss(v, v_star)
            parts["fm"] = l_fm.item()
            loss = loss + a.lambda_fm * l_fm
            if a.lambda_retain > 0 and vt is not None:
                l_ret = Fn.mse_loss(v, vt)
                parts["retain"] = l_ret.item()
                loss = loss + a.lambda_retain * l_ret
                if a.mode == "p4":
                    # RMS cap: ||v - v_base|| / ||v_base|| <= rms_frac
                    resid = v - vt
                    ratio = resid.flatten(1).pow(2).mean(1).sqrt() / \
                        vt.flatten(1).pow(2).mean(1).sqrt().clamp_min(1e-8)
                    l_rms = Fn.relu(ratio - a.rms_frac).mean()
                    parts["rms_ratio"] = ratio.mean().item()
                    parts["rms_pen"] = l_rms.item()
                    loss = loss + 10.0 * l_rms
            if a.mode == "p4":
                l_con, ec, ew = contrast_loss(model, z_real, sid, tau,
                                              canon=canon, T=a.contrast_T)
                parts["contrast"] = l_con.item()
                parts["e_corr"] = ec
                parts["e_wrong"] = ew
                loss = loss + a.lambda_contrast * l_con
            # --- Bridge path ---
            bb = sample_bridge(z_real, sid, style_targets,
                               device=dev)
            if canon is not None:
                bb["content"] = canon(bb["z_tilde"].float(),
                                      mode="c1_gentle", augment=False)
            l_br = bridge_loss(model, bb)
            parts["bridge"] = l_br.item()
            loss = loss + l_br
            loss = loss / a.accum
        loss.backward()
        micro_in += 1
        for k, v_ in parts.items():
            run[k] = run.get(k, 0.0) + v_
        if micro_in % a.accum == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for g in groups for p in g["params"]], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            opt_step += 1
            peak = max(peak, torch.cuda.memory_allocated() // 2**20)
            if opt_step % 100 == 0:
                rec = {"opt_step": opt_step,
                       "sec": round(time.time() - t0, 1),
                       "peak_mb": peak}
                n = a.accum * 100
                for k, v in run.items():
                    rec[k] = round(v / n, 5)
                with curve.open("a") as fp:
                    fp.write(json.dumps(rec) + "\n")
                print(f"[{a.mode}] {opt_step}/{a.updates} "
                      + " ".join(f"{k}={v:.4f}" for k, v in rec.items()
                                 if k not in ("opt_step", "sec", "peak_mb"))
                      + f" {rec['sec']}s peak={peak}MB", flush=True)
                run = {}
    sd = {"model": model.state_dict() if a.mode in ("p3", "p4")
          else model.state_dict(),
          "opt_step": opt_step,
          "config": {**cfg, "posttrain_v2": a.mode, "seed": a.seed,
                     "updates": a.updates, "lr": a.lr,
                     "lambda_fm": a.lambda_fm,
                     "lambda_retain": a.lambda_retain,
                     "lambda_contrast": a.lambda_contrast,
                     "rms_frac": a.rms_frac}}
    torch.save(sd, out / "final.pt")
    print(f"DONE {a.mode} {opt_step} {time.time()-t0:.0f}s peak={peak}MB",
          flush=True)


if __name__ == "__main__":
    main()
