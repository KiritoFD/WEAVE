"""V7 P3: train explicit source->target style Predictors M1/M2 with the relative
residual parameterization (V7_IMPLEMENTATION_PLAN.md P3).

    z_pred = zs + A_theta(c(zs), target) - A_theta(c(zs), source)

- init from frozen B4; same-style identity is structural (source==target -> z_pred==zs);
- M1 loss = original V6 distribution loss on the cross-style pass
  (band-normalized reconstruction toward the coherent real target + style
  distribution + canonical invariance);
- M2 loss = L_style (P1 best real-distribution energy) + content proxy + residual
  RMS hinge + weak R1 margin;
- effective batch 240 = microbatch x accumulation (never changed to dodge VRAM);
- grad-norm audit at step 1 and every 100 steps on a THROWAWAY batch (the main
  training graph is never probed — see train_eval_v6.py note on autograd probes);
- five_style_probe.json + source_leak_probe.json + config.json + V6-compatible ckpt.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from wavelet import dwt2_haar

from tools.v6.canonicalize import ContentCanonicalizer
from tools.v6.models_v6 import C3UNet
from tools.v6.train_eval_v6 import (
    STYLES, build_indomain_style_targets, compute_style_distribution_loss,
    make_coherent_real_target, normalized_band_mse,
)
from tools.v7.v7_common import (
    B4_CKPT, ROOT, load_canon, load_critic, rms_normalize, s1_loss, s2_loss,
    tasks_sha256, write_json,
)

V6_ROOT = ROOT / "runs" / "v6"
CANON_STATS = ROOT / "runs" / "v6" / "cache" / "canonicalizer_stats.pt"
P3_MODELS = ROOT / "runs" / "v7" / "p3_models"
SD_POOL = Path("/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16")
CLASS_RMS = {"Early_Renaissance": 0.32, "Impressionism": 0.15,
             "Minimalism": 0.15, "Rococo": 0.32, "Ukiyo_e": 0.32}


def build_args_model(args, device):
    model = C3UNet(base_dim=args.base_dim, num_styles=5, mode=args.mode,
                   no_style=False, content_mode="full",
                   inject_64=not args.no_inject_64, inject_32=not args.no_inject_32,
                   inject_16=not args.no_inject_16,
                   style_keys=False).to(device)
    if args.init_from:
        ck = torch.load(ROOT / args.init_from, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        print(f"[{args.name}] init_from={args.init_from} missing={len(missing)} "
              f"unexpected={len(unexpected)}", flush=True)
    return model


def relative_forward(model, z_real, c_in, y_real, y_cross):
    """z_pred = zs + A(c(zs), t) - A(c(zs), s) (source==target -> exactly zs)."""
    z_t = model(c_in, 0.0, c_in, y_cross)
    z_s = model(c_in, 0.0, c_in, y_real)
    return z_real + z_t - z_s


def canon_inv_loss(z_pred, z_real, canon):
    """V6 content-canonical invariance: c1(z_pred) vs c1(z_real), LL + 0.35*HF."""
    c_clean = canon(z_real, mode="c1", augment=False)
    c_cross = canon(z_pred.float(), mode="c1", augment=False)
    r_ll, r_lh, r_hl, _ = dwt2_haar(c_clean)
    g_ll, g_lh, g_hl, _ = dwt2_haar(c_cross)
    return F.mse_loss(g_ll, r_ll) + 0.35 * (F.mse_loss(g_lh, r_lh)
                                            + F.mse_loss(g_hl, r_hl))


def compute_terms(args, model, batch, ctx):
    """One training microbatch -> (total loss, dict of term losses, grads info)."""
    z_real, y_real = batch
    canon, device = ctx["canon"], z_real.device
    with torch.no_grad():
        c_in = canon(z_real, mode=args.canon_mode, augment=True)
        shift = torch.randint(1, len(STYLES), (z_real.shape[0],), device=device)
        y_cross = (y_real + shift) % len(STYLES)
        z_target = make_coherent_real_target(z_real, y_real, ctx["style_targets"],
                                             coherence_beta=args.coherence_beta)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                        enabled=args.amp_dtype == "bf16"):
        z_pred = relative_forward(model, z_real, c_in, y_real, y_cross)
        terms = {}
        if args.loss == "v6":
            terms["rec"] = normalized_band_mse(z_pred, z_target, canon)
            terms["style_cross"] = compute_style_distribution_loss(
                z_pred, y_cross, ctx["style_targets"])
            terms["canon_inv"] = canon_inv_loss(z_pred, z_real, canon)
            loss = (args.w_rec * terms["rec"] + args.lambda_cross * terms["style_cross"]
                    + args.lambda_canon * terms["canon_inv"])
        else:  # v7 (M2): P1 best energy + content + residual + weak R1
            kind, stats = ctx["energy"]
            if kind.startswith("S1"):
                terms["style"] = s1_loss(z_pred, y_cross, stats)
            else:
                terms["style"] = s2_loss(z_pred, y_cross, stats, ctx["feat_ext"])
            terms["content"] = canon_inv_loss(z_pred, z_real, canon)
            resid = (z_pred.float() - z_real.float()).flatten(1)
            rms = resid.pow(2).mean(dim=1).sqrt()
            budget = torch.tensor([CLASS_RMS[STYLES[int(t)]] for t in y_cross.tolist()],
                                  device=device)
            terms["residual"] = F.relu(rms - budget).square().mean()
            q = ctx["critic"](z_pred.float()).float()
            oh = F.one_hot(y_cross, len(STYLES)).bool()
            m_t = (q.gather(1, y_cross[:, None]).squeeze(1)
                   - torch.logsumexp(q.masked_fill(oh, torch.finfo(q.dtype).min), dim=1))
            terms["r1_margin"] = F.softplus(-m_t / 0.5).mean()
            loss = (terms["style"] + args.w_content * terms["content"]
                    + args.w_residual * terms["residual"]
                    + args.w_r1 * terms["r1_margin"])
        terms["total"] = loss
    return loss, terms, y_cross


def grad_audit(args, model, ctx, device) -> dict:
    """Per-term gradient norms on a throwaway batch (main graph untouched)."""
    was_training = model.training
    model.eval()
    per_style_mb = max(4, args.microbatch // len(STYLES))
    z_chunks, y_chunks = [], []
    for sid in range(len(STYLES)):
        pool = ctx["gpu_train"][STYLES[sid]]
        idx = torch.randint(0, pool.shape[0], (per_style_mb,), device=device)
        z_chunks.append(pool[idx])
        y_chunks.append(torch.full((per_style_mb,), sid, device=device,
                                   dtype=torch.long))
    batch = (torch.cat(z_chunks, 0), torch.cat(y_chunks, 0))
    loss, terms, _ = compute_terms(args, model, batch, ctx)
    params = [p for p in model.parameters() if p.requires_grad]
    norms = {}
    for name, tl in terms.items():
        gs = torch.autograd.grad(tl, params, retain_graph=True, allow_unused=True)
        s = torch.zeros((), device=device)
        for g in gs:
            if g is not None:
                s = s + g.detach().float().square().sum()
        norms[f"grad_{name}"] = float(s.sqrt())
    if was_training:
        model.train()
    return norms


def probe_five_styles(args, model, ctx, device) -> dict:
    """Same content, five target Style-IDs: output spread + source residual."""
    from tools.v7.v7_common import load_critic
    critic = ctx["critic"]
    val = {}
    for sid, s in enumerate(STYLES):
        val[s] = ctx["pool_val"][s][:8].to(device)
    out = {"per_style": {}, "overall": {}}
    res_r1, spread, resid_rms = [], [], []
    with torch.inference_mode():
        for sid, s in enumerate(STYLES):
            z = val[s]
            y_real = torch.full((z.shape[0],), sid, device=device, dtype=torch.long)
            c_in = ctx["canon"](z, mode=args.canon_mode, augment=False)
            preds = {}
            for tid in range(5):
                y_t = torch.full_like(y_real, tid)
                zp = relative_forward(model, z, c_in, y_real, y_t)
                preds[tid] = zp
                if tid != sid:
                    resid_rms.extend(
                        (zp - z).flatten(1).pow(2).mean(dim=1).sqrt().tolist())
            q = critic(torch.stack([preds[t] for t in range(5)], 0).float()
                       .reshape(-1, 4, 64, 64)).float().argmax(dim=1)
            q = q.view(5, -1).tolist()  # [target][sample]
            pair_rms = [float((preds[a] - preds[b]).flatten().pow(2).mean().sqrt())
                        for a in range(5) for b in range(a + 1, 5)]
            leak = float(np.mean([1.0 if q[t][i] == sid else 0.0
                                  for t in range(5) for i in range(len(q[t]))
                                  if t != sid]))
            res_r1.append(leak)
            spread.append(float(np.mean(pair_rms)))
            out["per_style"][s] = {
                "pairwise_latent_rms_mean": float(np.mean(pair_rms)),
                "source_residual_rate": leak,
                "r1_pred_matrix": q,
            }
    out["overall"] = {
        "source_residual_rate_r1": float(np.mean(res_r1)),
        "mean_pairwise_latent_rms": float(np.mean(spread)),
        "residual_rms_mean": float(np.mean(resid_rms)),
        "residual_rms_p95": float(np.percentile(resid_rms, 95)),
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="V7 P3 predictor training (M1/M2)")
    ap.add_argument("--name", required=True, choices=["M1", "M2"])
    ap.add_argument("--loss", choices=["v6", "v7"], default=None,
                    help="v6 => M1 (original V6 distribution loss), v7 => M2")
    ap.add_argument("--init-from", default=str(B4_CKPT))
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--microbatch", type=int, default=40)
    ap.add_argument("--accum", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--base-dim", type=int, default=16)
    ap.add_argument("--mode", default="direct")
    ap.add_argument("--canon-mode", default="c1_gentle")
    ap.add_argument("--no-inject-64", action="store_true")
    ap.add_argument("--no-inject-32", action="store_true")
    ap.add_argument("--no-inject-16", action="store_true")
    ap.add_argument("--coherence-beta", type=float, default=0.65)
    ap.add_argument("--w-rec", type=float, default=1.0)
    ap.add_argument("--lambda-cross", type=float, default=0.35)
    ap.add_argument("--lambda-canon", type=float, default=0.525)
    ap.add_argument("--w-content", type=float, default=0.525)
    ap.add_argument("--w-residual", type=float, default=1.0)
    ap.add_argument("--w-r1", type=float, default=0.003)
    ap.add_argument("--amp-dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--energies-dir", default=str(ROOT / "runs" / "v7" / "p1_energy"))
    # V7-R P0 additions: save intermediate checkpoints / override output dir so
    # the training trajectory can be audited without touching V7 results.
    ap.add_argument("--ckpt-steps", default="",
                    help="comma-separated steps to snapshot, e.g. 0,100,400,800")
    ap.add_argument("--out-dir", default="",
                    help="override output directory (default runs/v7/p3_models/<name>)")
    args = ap.parse_args()
    if args.loss is None:
        args.loss = "v6" if args.name == "M1" else "v7"
    device = "cuda"
    torch.manual_seed(args.seed)
    t0 = time.time()
    out_dir = Path(args.out_dir) if args.out_dir else P3_MODELS / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(out_dir / "train.log", "a", encoding="utf-8")

    def log(msg):
        line = f"[{args.name} {time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    # data
    pool_train, pool_val = {}, {}
    for sid, s in enumerate(STYLES):
        d = torch.load(SD_POOL / f"{sid:02d}_{s}.pt", map_location="cpu",
                       weights_only=False)
        x = d["latents"].float()
        pool_val[s] = x[:100]
        pool_train[s] = x[100:].to(device)
    canon = load_canon(device)
    critic = load_critic(device=device)
    style_targets = build_indomain_style_targets(
        {s: pool_train[s].cpu() for s in STYLES}, ROOT.parent, device=device, top_k=64)

    ctx = {"canon": canon, "critic": critic, "style_targets": style_targets,
           "gpu_train": pool_train, "pool_val": pool_val, "energy": (None, None)}
    if args.loss == "v7":
        from tools.v6_next.w_distill import R1FeatureExtractor
        feat_ext = R1FeatureExtractor(critic)
        ev = json.loads((Path(args.energies_dir) / "energy_validity.json")
                        .read_text(encoding="utf-8"))
        best = ev.get("best_energy")
        if not best:
            raise SystemExit("[M2] P1 returned NO_ENERGY; M2 training is not "
                             "allowed to proceed without its loss defined")
        stats = torch.load(Path(args.energies_dir) / f"{best}.pt",
                           map_location="cpu", weights_only=False)
        ctx["energy"] = (best, stats)
        ctx["feat_ext"] = feat_ext
        log(f"energy={best}")

    model = build_args_model(args, device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps,
                                                           eta_min=1e-5)
    per_style_mb = max(4, args.microbatch // len(STYLES))
    log(f"start: steps={args.steps} microbatch={args.microbatch} accum={args.accum} "
        f"effective_batch={args.microbatch * args.accum} per_style_mb={per_style_mb} "
        f"params={n_params} loss={args.loss}")

    norms0 = grad_audit(args, model, ctx, device)
    log(f"grad_audit@0: {json.dumps(norms0)}")
    with open(out_dir / "grad_norm.log", "a", encoding="utf-8") as gf:
        gf.write(json.dumps({"step": 0, **norms0}) + "\n")

    ckpt_steps = set()
    if args.ckpt_steps:
        ckpt_steps = {int(x) for x in args.ckpt_steps.split(",") if x.strip()}
        if 0 in ckpt_steps:
            torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "args": vars(args), "step": 0},
                       out_dir / "step_0000.pt")
            log("saved step_0000.pt")

    loss_hist, t_start = [], time.time()
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        step_val = 0.0
        for _ in range(args.accum):
            z_chunks, y_chunks = [], []
            for sid in range(len(STYLES)):
                pool_c = ctx["style_targets"]["styles"][STYLES[sid]]["core_latents"].float().to(device)
                pool_a = pool_train[STYLES[sid]]
                n_core = per_style_mb // 2
                parts = [pool_c[torch.randint(0, pool_c.shape[0], (n_core,), device=device)],
                         pool_a[torch.randint(0, pool_a.shape[0],
                                              (per_style_mb - n_core,), device=device)]]
                z_chunks.append(torch.cat(parts, 0))
                y_chunks.append(torch.full((per_style_mb,), sid, device=device,
                                           dtype=torch.long))
            batch = (torch.cat(z_chunks, 0), torch.cat(y_chunks, 0))
            loss, terms, _ = compute_terms(args, model, batch, ctx)
            (loss / args.accum).backward()
            step_val += float(loss.detach()) / args.accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        loss_hist.append(step_val)

        if step in ckpt_steps:
            torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "args": vars(args), "step": step},
                       out_dir / f"step_{step:04d}.pt")
            log(f"saved step_{step:04d}.pt")

        if step % 100 == 0 or step == 1 or step == args.steps:
            norms = grad_audit(args, model, ctx, device)
            with open(out_dir / "grad_norm.log", "a", encoding="utf-8") as gf:
                gf.write(json.dumps({"step": step, **norms}) + "\n")
        if step % 50 == 0 or step == 1 or step == args.steps:
            peak = torch.cuda.max_memory_allocated(0) / (1024 ** 2)
            log(f"step {step}/{args.steps} loss={np.mean(loss_hist[-50:]):.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e} vram={peak:.0f}MiB "
                f"elapsed={time.time()-t_start:.0f}s")

    train_wall = round(time.time() - t0, 1)
    peak_vram = round(torch.cuda.max_memory_allocated(0) / (1024 ** 2), 1)
    ckpt = {
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "args": vars(args),
        "n_params": int(n_params),
        "train_wall_sec": train_wall,
        "peak_vram_mib": peak_vram,
        "final_loss": float(np.mean(loss_hist[-50:])),
        "parameterization": "relative_residual",
        "tasks_sha256": tasks_sha256(),
    }
    torch.save(ckpt, out_dir / f"{args.name}.pt")
    log(f"saved {out_dir / f'{args.name}.pt'} wall={train_wall}s "
        f"peak_vram={peak_vram}MiB final_loss={ckpt['final_loss']:.4f}")

    model.eval()
    probes = probe_five_styles(args, model, ctx, device)
    write_json(probes, out_dir / "five_style_probe.json")
    write_json({
        "source_leak_rate_r1": probes["overall"]["source_residual_rate_r1"],
        "note": "fraction of cross-target outputs still classified as source style",
    }, out_dir / "source_leak_probe.json")
    write_json({**vars(args), "effective_batch": args.microbatch * args.accum,
                "train_wall_sec": train_wall, "peak_vram_mib": peak_vram,
                "final_loss": ckpt["final_loss"],
                "sd_pool": str(SD_POOL)}, out_dir / "config.json")
    log_f.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
