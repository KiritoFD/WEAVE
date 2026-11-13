"""V7-R P2: new Predictor design — keep B4 frozen, learn a small correction.

  mode corrector : z_out = z0 + H(c, z0, [source], target), z0 = B4(c, target)
                   N1 = use_source=False, N2 = use_source=True (ONLY difference)
  mode rand      : P2 alternative route — full C3UNet from B4 init trained with
                   the V6 objective on style-randomized c1_gentle inputs
                   (constrained LL colour + HF amplitude perturbation)

Loss (corrector): L = w_s*L_target_style + w_c*L_content_inv + w_a*||z_out-z0||²
                   + w_r*weak_R1_margin
  L_target_style = V6 distribution supervision by default; E3/E4 patch energy
  when the P1 gate passed (runner passes --style-objective).

Budgets (predefined): 3000 steps, effective batch 240 (40x6), bf16, AdamW lr
2e-4 with 100-step warmup + cosine decay, clip 1.0. Checkpoints at
0/500/1500/3000 — each evaluated on the full 750 by the runner (no subset
filtering). Weighted per-term gradient norms audited at step 0 and every 100
steps; if any weighted term exceeds 100x the smallest at the step-100 audit,
its weight is rescaled ONCE toward the median (capped 100x) and recorded.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tools.v6.canonicalize import ContentCanonicalizer
from tools.v6.models_v6 import C3UNet
from tools.v6.train_eval_v6 import (
    STYLES, build_indomain_style_targets, compute_style_distribution_loss,
    make_coherent_real_target, normalized_band_mse,
)
from tools.v7.v7_common import (
    B4_CKPT, ROOT, load_c3unet, load_canon, load_critic, tasks_sha256,
    write_json,
)
from tools.v7r.v7r_common import (
    CORRECTOR_BASE_DIM, IMBALANCE_RATIO, N_CKPT_STEPS, N_EFFECTIVE_BATCH,
    N_LR, N_STEPS, N_WARMUP, P2_DIR, SD_POOL, W_ANCHOR, W_CONTENT, W_R1,
    W_STYLE, canon_inv_loss, load_pool, randomized_canon, save_corrector,
)
from tools.legacy_root.wavelet import dwt2_haar


def build_style_objective(name: str, ctx):
    if name == "v6dist":
        return lambda z, tid: compute_style_distribution_loss(
            z, tid, ctx["style_targets"])
    if name in ("E3", "E4"):
        banks = ctx["patch_banks"]
        from tools.v7r.p1_patchbank import bank_softnn_loss
        return lambda z, tid: bank_softnn_loss(z, tid, banks,
                                               use_ll=(name == "E4"))
    raise ValueError(name)


def grad_audit_cons(args, H, ctx, device) -> dict:
    """Weighted style vs consistency gradient norms on a throwaway batch."""
    was = H.training
    H.eval()
    per_style_mb = max(4, args.microbatch // len(STYLES))
    z_chunks, y_chunks = [], []
    for sid in range(len(STYLES)):
        pool = ctx["gpu_train"][STYLES[sid]]
        idx = torch.randint(0, pool.shape[0], (per_style_mb,), device=device)
        z_chunks.append(pool[idx])
        y_chunks.append(torch.full((per_style_mb,), sid, device=device,
                                   dtype=torch.long))
    z_real = torch.cat(z_chunks, 0)
    y_src = torch.cat(y_chunks, 0)
    shift = torch.randint(1, len(STYLES), (z_real.shape[0],), device=device)
    y_tgt = (y_src + shift) % len(STYLES)
    params = [p for p in H.parameters() if p.requires_grad]
    with torch.no_grad():
        c_base = ctx["canon"](z_real, mode=args.canon_mode, augment=True)
        z0 = ctx["b4"](c_base, 0.0, c_base, y_tgt)
        c1 = randomized_canon(z_real, ctx["canon"])
        c2 = randomized_canon(z_real, ctx["canon"])
    h1, h2 = H(c1, z0, y_src, y_tgt), H(c2, z0, y_src, y_tgt)
    z_out = z0 + h1
    terms = {
        "style": args.w_style * ctx["style_fn"](z_out, y_tgt),
        "cons": args.w_cons * (h1.float() - h2.float()).square()
        .flatten(1).mean(),
    }
    norms = {}
    for name, tl in terms.items():
        gs = torch.autograd.grad(tl, params, retain_graph=True, allow_unused=True)
        ssum = torch.zeros((), device=z_real.device)
        for g in gs:
            if g is not None:
                ssum = ssum + g.detach().float().square().sum()
        norms[f"grad_{name}"] = float(ssum.sqrt())
    if was:
        H.train()
    return norms


def grad_audit_corrector(args, H, ctx, device) -> dict:
    """Weighted per-term gradient norms on a throwaway batch."""
    was = H.training
    H.eval()
    per_style_mb = max(4, args.microbatch // len(STYLES))
    z_chunks, y_chunks = [], []
    for sid in range(len(STYLES)):
        pool = ctx["gpu_train"][STYLES[sid]]
        idx = torch.randint(0, pool.shape[0], (per_style_mb,), device=device)
        z_chunks.append(pool[idx])
        y_chunks.append(torch.full((per_style_mb,), sid, device=device,
                                   dtype=torch.long))
    z_real = torch.cat(z_chunks, 0)
    y_src = torch.cat(y_chunks, 0)
    shift = torch.randint(1, len(STYLES), (z_real.shape[0],), device=device)
    y_tgt = (y_src + shift) % len(STYLES)
    params = [p for p in H.parameters() if p.requires_grad]
    with torch.no_grad():
        c_in = ctx["canon"](z_real, mode=args.canon_mode, augment=True)
        z0 = ctx["b4"](c_in, 0.0, c_in, y_tgt)
    z_out = H(c_in, z0, y_src, y_tgt)
    terms = {
        "style": args.w_style * ctx["style_fn"](z_out, y_tgt),
        "content": args.w_content * canon_inv_loss(z_out, z_real, ctx["canon"]),
        "anchor": args.w_anchor * (z_out.float() - z0.float()).square()
        .flatten(1).mean(),
    }
    q = ctx["critic"](z_out.float()).float()
    oh = F.one_hot(y_tgt, len(STYLES)).bool()
    m = (q.gather(1, y_tgt[:, None]).squeeze(1)
         - torch.logsumexp(q.masked_fill(oh, -1e9), dim=1))
    terms["r1"] = args.w_r1 * F.softplus(-m / 0.5).mean()
    norms = {}
    for name, tl in terms.items():
        gs = torch.autograd.grad(tl, params, retain_graph=True, allow_unused=True)
        s = torch.zeros((), device=device)
        for g in gs:
            if g is not None:
                s = s + g.detach().float().square().sum()
        norms[f"grad_{name}"] = float(s.sqrt())
    norms["loss_total"] = float(sum(t for t in terms.values()))
    if was:
        H.train()
    return norms


def train_corrector(args, device: str) -> int:
    t0 = time.time()
    out_dir = P2_DIR / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(out_dir / "train.log", "a", encoding="utf-8")

    def log(msg):
        line = f"[{args.name} {time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    if args.consistency and args.augment != "rand":
        raise SystemExit("--consistency 1 requires --augment rand (E2-C)")
    torch.manual_seed(args.seed)
    train_by_style, val_by_style, painters = load_pool(device="cpu")
    gpu_train = {s: train_by_style[s].to(device) for s in STYLES}
    canon = load_canon(device)
    b4 = load_c3unet(B4_CKPT, device)
    for p in b4.parameters():
        p.requires_grad_(False)
    critic = load_critic(device=device)
    style_targets = build_indomain_style_targets(
        {s: train_by_style[s] for s in STYLES}, ROOT.parent, device=device,
        top_k=64)
    ctx = {"canon": canon, "b4": b4, "critic": critic, "gpu_train": gpu_train,
           "style_targets": style_targets}
    if args.style_objective in ("E3", "E4"):
        ctx["patch_banks"] = torch.load(
            ROOT / "runs/v7r/p1_energy/patch_banks.pt", map_location="cpu",
            weights_only=False)
    ctx["style_fn"] = build_style_objective(args.style_objective, ctx)

    use_source = (args.name.upper() != "N1")
    H = CorrectorWrapper(use_source, args.base_dim).to(device)
    start_step = 0
    if args.init_corrector:
        ck = torch.load(ROOT / args.init_corrector, map_location="cpu",
                        weights_only=False)
        H.load_state_dict(ck["corrector"])
        start_step = int(ck.get("step", 0))
        log(f"resumed corrector from {args.init_corrector} (step {start_step})")
    n_params = sum(p.numel() for p in H.parameters())
    optimizer = torch.optim.AdamW(H.parameters(), lr=args.lr, weight_decay=1e-4)

    def lr_at(step):
        total = args.steps
        s = start_step + step
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        # cosine from warmup end to eta_min over the remaining schedule
        t = (s - args.warmup) / max(1, total - args.warmup)
        return 1e-5 + 0.5 * (args.lr - 1e-5) * (1 + np.cos(np.pi * min(t, 1.0)))

    per_style_mb = max(4, args.microbatch // len(STYLES))
    log(f"start: steps={args.steps} microbatch={args.microbatch} accum={args.accum} "
        f"effective_batch={args.microbatch * args.accum} params={n_params} "
        f"use_source={use_source} style_objective={args.style_objective} "
        f"ckpt_steps={args.ckpt_steps}")

    ckpt_steps = {int(x) for x in args.ckpt_steps.split(",") if x.strip()}
    if 0 in ckpt_steps and start_step == 0:
        save_corrector(out_dir / "step_0000.pt", H.net, vars(args), 0)
        log("saved step_0000.pt")

    audited_rescale = False
    cons_aligned = False
    loss_hist = []

    # E2 perturbation safety pre-check: content-proxy change under augmentation
    if args.augment == "rand":
        with torch.no_grad():
            zb = torch.cat([gpu_train[s][:4] for s in STYLES], 0).to(device)
            c_orig = canon(zb, mode=args.canon_mode, augment=False)
            c_rand = randomized_canon(zb, canon)
            pre = {
                "content_proxy_inv_orig": float(canon_inv_loss(c_orig, zb, canon)),
                "content_proxy_inv_rand": float(canon_inv_loss(c_rand, zb, canon)),
                "ll_drift_rms": float((c_rand - c_orig).flatten(1).pow(2)
                                      .mean(dim=1).sqrt().mean()),
            }
        log(f"augment pre-check: {json.dumps(pre)}")
        with open(out_dir / "augment_precheck.json", "w", encoding="utf-8") as f:
            json.dump(pre, f, indent=2)
    for step in range(start_step + 1, args.steps + 1):
        for g in optimizer.param_groups:
            g["lr"] = lr_at(step - start_step)
        H.train()
        optimizer.zero_grad(set_to_none=True)
        step_val = 0.0
        for _ in range(args.accum):
            z_chunks, y_chunks = [], []
            for sid in range(len(STYLES)):
                pool_c = style_targets["styles"][STYLES[sid]]["core_latents"].float().to(device)
                pool_a = gpu_train[STYLES[sid]]
                n_core = int(round(per_style_mb * args.core_frac))
                parts = []
                if n_core > 0:
                    parts.append(pool_c[torch.randint(0, pool_c.shape[0], (n_core,), device=device)])
                n_all = per_style_mb - n_core
                if n_all > 0:
                    parts.append(pool_a[torch.randint(0, pool_a.shape[0],
                                                      (n_all,), device=device)])
                z_chunks.append(torch.cat(parts, 0))
                y_chunks.append(torch.full((per_style_mb,), sid, device=device,
                                           dtype=torch.long))
            z_real = torch.cat(z_chunks, 0)
            y_src = torch.cat(y_chunks, 0)
            with torch.no_grad():
                shift = torch.randint(1, len(STYLES), (z_real.shape[0],),
                                      device=device)
                y_tgt = (y_src + shift) % len(STYLES)
                c_base = canon(z_real, mode=args.canon_mode, augment=True)
                if args.augment == "rand":
                    c1v = randomized_canon(z_real, canon)
                    c2v = randomized_canon(z_real, canon)
                    z0 = b4(c_base, 0.0, c_base, y_tgt)  # shared ORIGINAL B4 result (E2-C)
                else:
                    c1v = c_base
                    z0 = b4(c_base, 0.0, c_base, y_tgt)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=args.amp_dtype == "bf16"):
                def _terms(z_out, z0_ref):
                    l_st = ctx["style_fn"](z_out, y_tgt)
                    l_ct = canon_inv_loss(z_out, z_real, canon)
                    l_an = (z_out.float() - z0_ref.float()).square() \
                        .flatten(1).mean()
                    q = critic(z_out.float()).float()
                    oh = F.one_hot(y_tgt, len(STYLES)).bool()
                    m = (q.gather(1, y_tgt[:, None]).squeeze(1)
                         - torch.logsumexp(q.masked_fill(oh, -1e9), dim=1))
                    return l_st, l_ct, l_an, F.softplus(-m / 0.5).mean()
                if args.consistency:
                    h1 = H(c1v, z0, y_src, y_tgt)
                    h2 = H(c2v, z0, y_src, y_tgt)
                    z_out1, z_out2 = z0 + h1, z0 + h2
                    l_st1, l_ct1, l_an1, l_r11 = _terms(z_out1, z0)
                    l_st2, l_ct2, l_an2, l_r12 = _terms(z_out2, z0)
                    l_style = 0.5 * (l_st1 + l_st2)
                    l_content = 0.5 * (l_ct1 + l_ct2)
                    l_anchor = 0.5 * (l_an1 + l_an2)
                    l_r1 = 0.5 * (l_r11 + l_r12)
                    l_cons = (h1.float() - h2.float()).square().flatten(1).mean()
                    loss = (args.w_style * l_style + args.w_content * l_content
                            + args.w_anchor * l_anchor + args.w_r1 * l_r1
                            + args.w_cons * l_cons)
                else:
                    z_out = H(c1v, z0, y_src, y_tgt)
                    l_style, l_content, l_anchor, l_r1 = _terms(z_out, z0)
                    loss = (args.w_style * l_style + args.w_content * l_content
                            + args.w_anchor * l_anchor + args.w_r1 * l_r1)
            (loss / args.accum).backward()
            step_val += float(loss.detach()) / args.accum
        torch.nn.utils.clip_grad_norm_(H.parameters(), 1.0)
        optimizer.step()
        loss_hist.append(step_val)

        if step % 100 == 0 or step == 1:
            norms = grad_audit_corrector(args, H, ctx, device)
            with open(out_dir / "grad_norm.log", "a", encoding="utf-8") as gf:
                gf.write(json.dumps({"step": step, **norms}) + "\n")
            # predefined one-time consistency-weight alignment (E2-C)
            if args.consistency and not cons_aligned and step >= 100:
                gn = grad_audit_cons(args, H, ctx, device)
                if gn.get("grad_cons", 0) > 0:
                    args.w_cons = float(np.clip(
                        args.w_style * gn["grad_style"] / gn["grad_cons"],
                        1e-3, 1e3))
                cons_aligned = True
                log(f"consistency weight aligned (one-time): w_cons={args.w_cons:.4f}")
            # predefined one-time rescale policy
            if (not audited_rescale and step >= 100):
                wn = {k: v for k, v in norms.items() if k.startswith("grad_")
                      and v > 0}
                mx, mn = max(wn.values()), min(wn.values())
                if mx > IMBALANCE_RATIO * max(mn, 1e-12):
                    med = float(np.median(list(wn.values())))
                    new_w = dict(w_style=args.w_style, w_content=args.w_content,
                                 w_anchor=args.w_anchor, w_r1=args.w_r1)
                    for term, wkey in (("style", "w_style"),
                                       ("content", "w_content"),
                                       ("anchor", "w_anchor"), ("r1", "w_r1")):
                        g = norms.get(f"grad_{term}", 0)
                        if g > IMBALANCE_RATIO * max(mn, 1e-12):
                            w = getattr(args, wkey)
                            setattr(args, wkey,
                                    float(np.clip(w * med / g,
                                                  w / IMBALANCE_RATIO,
                                                  w * IMBALANCE_RATIO)))
                    audited_rescale = True
                    log(f"IMBALANCE RESCALE (one-time, predefined policy): "
                        f"{json.dumps(vars(args))}")
        if step % 100 == 0 or step == 1 or step == args.steps:
            log(f"step {step}/{args.steps} loss={np.mean(loss_hist[-100:]):.4f} "
                f"lr={lr_at(step - start_step):.2e} "
                f"vram={torch.cuda.max_memory_allocated(0)/(1024**2):.0f}MiB "
                f"elapsed={time.time()-t0:.0f}s")
        if step in ckpt_steps:
            save_corrector(out_dir / f"step_{step:05d}.pt", H.net, vars(args), step)
            log(f"saved step_{step:05d}.pt")

    write_json({**vars(args), "effective_batch": args.microbatch * args.accum,
                "n_params": n_params, "use_source": use_source,
                "train_wall_sec": round(time.time() - t0, 1),
                "tasks_sha256": tasks_sha256()}, out_dir / "config.json")

    # quick mechanism probe (no decode): residual stats + five-ID spread
    H.eval()
    probe = {"residual_rms": [], "pred_is_source": [], "five_id_spread": []}
    with torch.inference_mode():
        for sid in range(5):
            z = val_by_style[STYLES[sid]][:8].to(device)
            c = canon(z, mode=args.canon_mode, augment=False)
            y_src = torch.full((z.shape[0],), sid, device=device,
                               dtype=torch.long)
            shift = torch.randint(1, 5, (z.shape[0],), device=device)
            y_tgt = (y_src + shift) % 5
            z0 = b4(c, 0.0, c, y_tgt)
            z_out = H(c, z0, y_src, y_tgt)
            probe["residual_rms"].extend(
                (z_out - z0).flatten(1).pow(2).mean(dim=1).sqrt().tolist())
            q = critic(z_out.float()).argmax(dim=1)
            probe["pred_is_source"].extend((q == y_src).float().tolist())
            # same content, five target IDs: residual spread (E2 question)
            outs = H(c[:1].expand(5, -1, -1, -1).contiguous(),
                     z0[:1].expand(5, -1, -1, -1).contiguous(),
                     torch.full((5,), sid, device=device, dtype=torch.long),
                     torch.arange(5, device=device))
            pair = [float((outs[a] - outs[b]).flatten().pow(2).mean().sqrt())
                    for a in range(5) for b in range(a + 1, 5)]
            probe["five_id_spread"].append(float(np.mean(pair)))
    write_json({"residual_rms_mean": float(np.mean(probe["residual_rms"])),
                "pred_is_source": float(np.mean(probe["pred_is_source"])),
                "five_id_spread_mean": float(np.mean(probe["five_id_spread"]))},
               out_dir / "corrector_probe.json")
    log_f.close()
    return 0


class CorrectorWrapper(torch.nn.Module):
    """Thin adapter so grad_audit can treat N1/N2 uniformly."""

    def __init__(self, use_source: bool, base_dim: int):
        super().__init__()
        from tools.v7r.v7r_common import ResidualCorrector
        self.net = ResidualCorrector(base_dim=base_dim, use_source=use_source)

    def forward(self, c, z0, y_src, y_tgt):
        return self.net(c, z0, y_src, y_tgt)

    @property
    def use_source(self):
        return self.net.use_source


def train_rand(args, device: str) -> int:
    """P2 alternative route: style-randomized canonicalization + V6 objective."""
    t0 = time.time()
    out_dir = P2_DIR / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(out_dir / "train.log", "a", encoding="utf-8")

    def log(msg):
        line = f"[{args.name} {time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    torch.manual_seed(args.seed)
    train_by_style, _, _ = load_pool(device="cpu")
    gpu_train = {s: train_by_style[s].to(device) for s in STYLES}
    canon = load_canon(device)
    style_targets = build_indomain_style_targets(
        {s: train_by_style[s] for s in STYLES}, ROOT.parent, device=device,
        top_k=64)
    model = C3UNet(base_dim=16, num_styles=5, mode="direct").to(device)
    ck = torch.load(B4_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-4)
    per_style_mb = max(4, args.microbatch // len(STYLES))
    log(f"start RAND route: steps={args.steps} microbatch={args.microbatch} "
        f"accum={args.accum}")

    for step in range(1, args.steps + 1):
        s = step
        lr = args.lr * min(1.0, s / args.warmup) if s < args.warmup else (
            1e-5 + 0.5 * (args.lr - 1e-5)
            * (1 + np.cos(np.pi * min((s - args.warmup)
                                      / max(1, args.steps - args.warmup), 1.0))))
        for g in optimizer.param_groups:
            g["lr"] = lr
        model.train()
        optimizer.zero_grad(set_to_none=True)
        step_val = 0.0
        for _ in range(args.accum):
            z_chunks, y_chunks = [], []
            for sid in range(len(STYLES)):
                pool_c = style_targets["styles"][STYLES[sid]]["core_latents"].float().to(device)
                pool_a = gpu_train[STYLES[sid]]
                n_core = int(round(per_style_mb * args.core_frac))
                parts = []
                if n_core > 0:
                    parts.append(pool_c[torch.randint(0, pool_c.shape[0], (n_core,), device=device)])
                n_all = per_style_mb - n_core
                if n_all > 0:
                    parts.append(pool_a[torch.randint(0, pool_a.shape[0],
                                                      (n_all,), device=device)])
                z_chunks.append(torch.cat(parts, 0))
                y_chunks.append(torch.full((per_style_mb,), sid, device=device,
                                           dtype=torch.long))
            z_real = torch.cat(z_chunks, 0)
            y_real = torch.cat(y_chunks, 0)
            with torch.no_grad():
                c_rand = randomized_canon(z_real, canon)
                z_target = make_coherent_real_target(
                    z_real, y_real, style_targets, coherence_beta=0.65)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=args.amp_dtype == "bf16"):
                z_pred = model(c_rand, 0.0, c_rand, y_real)
                l_rec = normalized_band_mse(z_pred, z_target, canon)
                l_same = compute_style_distribution_loss(z_pred, y_real,
                                                         style_targets)
                shift = torch.randint(1, len(STYLES), (z_real.shape[0],),
                                      device=device)
                y_cross = (y_real + shift) % len(STYLES)
                z_cross = model(c_rand, 0.0, c_rand, y_cross)
                l_cross = compute_style_distribution_loss(z_cross, y_cross,
                                                          style_targets)
                c_clean = canon(z_real, mode="c1", augment=False)
                c_pred = canon(z_cross.float(), mode="c1", augment=False)
                r_ll, r_lh, r_hl, _ = dwt2_haar(c_clean)
                g_ll, g_lh, g_hl, _ = dwt2_haar(c_pred)
                l_inv = F.mse_loss(g_ll, r_ll) + 0.35 * (
                    F.mse_loss(g_lh, r_lh) + F.mse_loss(g_hl, r_hl))
                loss = 1.0 * l_rec + 0.25 * l_same + 0.35 * l_cross + 0.525 * l_inv
            (loss / args.accum).backward()
            step_val += float(loss.detach()) / args.accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0 or step == 1 or step == args.steps:
            log(f"step {step}/{args.steps} loss={step_val:.4f} "
                f"vram={torch.cuda.max_memory_allocated(0)/(1024**2):.0f}MiB "
                f"elapsed={time.time()-t0:.0f}s")
        if step in {int(x) for x in args.ckpt_steps.split(",") if x.strip()}:
            torch.save({"model": {k: v.detach().cpu()
                                  for k, v in model.state_dict().items()},
                        "args": {"base_dim": 16, "mode": "direct",
                                 "canon_mode": "c1_gentle", "name": args.name},
                        "step": step}, out_dir / f"step_{step:05d}.pt")
            log(f"saved step_{step:05d}.pt")
    write_json({**vars(args), "route": "randomization"}, out_dir / "config.json")
    log_f.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="V7-R P2 predictor training")
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", choices=["corrector", "rand"], default="corrector")
    ap.add_argument("--style-objective", default="v6dist",
                    choices=["v6dist", "E3", "E4"])
    ap.add_argument("--steps", type=int, default=N_STEPS)
    ap.add_argument("--microbatch", type=int, default=40)
    ap.add_argument("--accum", type=int, default=6)
    ap.add_argument("--lr", type=float, default=N_LR)
    ap.add_argument("--warmup", type=int, default=N_WARMUP)
    ap.add_argument("--base-dim", type=int, default=CORRECTOR_BASE_DIM)
    ap.add_argument("--canon-mode", default="c1_gentle")
    ap.add_argument("--w-style", type=float, default=W_STYLE)
    ap.add_argument("--w-content", type=float, default=W_CONTENT)
    ap.add_argument("--w-anchor", type=float, default=W_ANCHOR)
    ap.add_argument("--w-r1", type=float, default=W_R1)
    ap.add_argument("--amp-dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt-steps", default="0,500,1500,3000")
    ap.add_argument("--init-corrector", default="",
                    help="resume H from a saved corrector ckpt (P4 extension)")
    # ---- V7-R+ additions ----
    ap.add_argument("--augment", choices=["none", "rand"], default="none",
                    help="rand = constrained c1_gentle appearance perturbation (E2)")
    ap.add_argument("--consistency", type=int, default=0,
                    help="1 = E2-C dual-view residual consistency (requires --augment rand)")
    ap.add_argument("--w-cons", type=float, default=1.0,
                    help="initial consistency weight; gradient-scale aligned once at the step-100 audit")
    ap.add_argument("--core-frac", type=float, default=0.5,
                    help="fraction of per-style samples drawn from the Core64 pool (E5: 0.0)")
    args = ap.parse_args()
    device = "cuda"
    if args.mode == "corrector":
        return train_corrector(args, device)
    return train_rand(args, device)


if __name__ == "__main__":
    raise SystemExit(main())
