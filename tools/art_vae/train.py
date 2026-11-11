"""Art-VAE encoder training with Gradient Cache.

    python -m tools.art_vae.train --stage A
    python -m tools.art_vae.train --stage B --init-from runs/art_vae/A_best.pt
    python -m tools.art_vae.train --stage C --init-from runs/art_vae/B_best.pt

Stage A : encoder tail only, S10 frozen, 500 updates (does the objective shape z?)
Stage B : full style training, S10 frozen, 3000 updates
Stage C : also unfreeze S10's LOW-RESOLUTION input modules, 1500 updates
          (must be compared against B -- see eval.py)

One update = 64 logical images through SupCon (Gradient Cache), with L1 +
LPIPS@512 + weak KL reconstruction accumulated in the same second pass.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.art_vae.data import (ArtVAEDataset, LOGICAL_BATCH, OUT, StyleContrastBatchSampler,
                                TrainPool, collate)
from tools.art_vae.gradcache import cache_style_gradient
from tools.art_vae.losses import (ReconstructionLoss, TAU, gamma_for_dim,
                                  supcon, supcon_relation_weights, vicreg_var)
from tools.art_vae.style_readout import (D_OUT, readout_scale_diagnostics,
                                         style_readout)

A0_DIR = str(ROOT / "runs_vae" / "A0" / "final_diffusers_vae")
S10_CK = ROOT / "runs" / "vae_decoder_prune" / "s10_p5_lowlr" / "decoder.pt"

TARGET_SHARE = 0.15   # style grad as a fraction of recon grad (plan: 10-20%)

STAGES = {
    "A": dict(updates=500, lr_enc=1e-5, lr_dec=0.0, mb=6, batches=1),
    "B": dict(updates=3000, lr_enc=1e-5, lr_dec=0.0, mb=6, batches=1),
    "C": dict(updates=1500, lr_enc=5e-6, lr_dec=5e-6, mb=6, batches=1),
}


class ArtVAE(torch.nn.Module):
    """A0 encoder + S10 decoder, native 4x64x64 latent contract."""

    def __init__(self, dev):
        super().__init__()
        from tools.vae_suite.decoder_prune import build_decoder, StudentDecoder
        a0 = AutoencoderKL.from_pretrained(A0_DIR, torch_dtype=torch.float32)
        self.encoder = a0.encoder.to(dev)
        self.quant_conv = a0.quant_conv.to(dev)
        self.post_quant_conv = a0.post_quant_conv.to(dev)
        self.scaling_factor = float(a0.config.scaling_factor)
        ck = torch.load(S10_CK, map_location="cpu", weights_only=False)
        d = build_decoder(ck["meta"]["widths_forward"])
        d.load_state_dict(ck["state_dict"])
        self.decoder = StudentDecoder(self.post_quant_conv, d.to(dev)).to(dev)
        for p in self.post_quant_conv.parameters():
            p.requires_grad = False

    def encode(self, x):
        return DiagonalGaussianDistribution(self.quant_conv(self.encoder(x)))

    def decode(self, z):
        return self.decoder(z)


def set_trainable(model, stage):
    """Stage A/B: encoder tail + quant_conv only. Stage C: also S10 low-res."""
    for p in model.parameters():
        p.requires_grad = False
    names = []
    tail = [model.encoder.mid_block, model.encoder.conv_norm_out,
            model.encoder.conv_out, model.quant_conv]
    for m in tail:
        for p in m.parameters():
            p.requires_grad = True
        names.append(type(m).__name__)
    if stage == "C":
        # S10 decoder: mid_block is closest to the latent, up_blocks[0] is the
        # first (lowest-resolution) upsampling stage. High-res stages stay frozen.
        for m in (model.decoder.decoder.mid_block,
                  model.decoder.decoder.up_blocks[0]):
            for p in m.parameters():
                p.requires_grad = True
            names.append(f"S10.{type(m).__name__}")
    return names


@torch.no_grad()
def calibrate_gamma(model, ds, dev, n=64, target=0.7):
    """VICReg variance floor calibrated on the ACTUAL readout statistics.

    gamma_for_dim() returns a theoretical 1/sqrt(d) scale, but real latent
    readouts have a much smaller per-dim std (measured ~0.011 vs 0.027 for
    random unit vectors). Importing the theoretical threshold would keep the
    variance loss permanently active. Here the floor is `target` times the
    empirical per-dim std of the INITIAL (A0) encoder, so the constraint only
    fires when the representation collapses below its starting diversity.
    """
    model.eval()
    qs = []
    for i in range(0, n, 4):
        x = torch.stack([ds.get(j, 0)["x"] for j in range(i, i + 4)]).to(dev)
        qs.append(style_readout(model.encode(x).mean.float()))
        del x
        torch.cuda.empty_cache()
    Q = torch.cat(qs)
    empirical = float(Q.std(dim=0).mean())
    model.train()
    return empirical * target, empirical


@torch.no_grad()
def fixed_ref_style(model, x_ref, W_ref, gamma, lam_v, mb=6):
    """Style loss on ONE FIXED batch -> noise-free curve for the stage-A gate."""
    qs = []
    for i in range(0, x_ref.shape[0], mb):
        qs.append(style_readout(model.encode(x_ref[i:i + mb]).mean))
    q = torch.cat(qs, dim=0)
    loss = supcon(q, W_ref, TAU) + lam_v * vicreg_var(q, gamma)
    del qs, q
    torch.cuda.empty_cache()
    return float(loss)


@torch.no_grad()
def monitor(model, ds, pool, dev, n=64):
    """Latent health diagnostics on the frozen holdout (no training grads)."""
    model.eval()
    zs, qs = [], []
    for i in range(0, n, 4):
        recs = [ds.get((i + k) % ds.__len__(), 0) for k in range(4)]
        x = torch.stack([r["x"] for r in recs]).to(dev)
        z = model.encode(x).mean.float()
        zs.append(z)
        qs.append(style_readout(z))
        torch.cuda.empty_cache()
    Z = torch.cat(zs); Q = torch.cat(qs)
    flat = Z.flatten(1)
    out = {
        "z_mean": round(float(Z.mean()), 5),
        "z_std": round(float(Z.std()), 5),
        "z_q01": round(float(flat.quantile(0.01)), 4),
        "z_q99": round(float(flat.quantile(0.99)), 4),
        "ch_mean": [round(float(v), 4) for v in Z.mean(dim=(0, 2, 3))],
        "ch_std": [round(float(v), 4) for v in Z.std(dim=(0, 2, 3))],
        "q_per_dim_std_mean": round(float(Q.std(dim=0).mean()), 6),
        "q_per_dim_std_min": round(float(Q.std(dim=0).min()), 6),
        "group_rms": readout_scale_diagnostics(Z)["group_rms"],
    }
    model.train()
    return out


@torch.no_grad()
def validate(model, ds, pool, dev, n=128, bs=4):
    """Lightweight fixed-validation: reconstruction L1/LPIPS + style AUROC."""
    import lpips
    model.eval()
    lp_fns = getattr(validate, "_lp", None)
    if lp_fns is None:
        lp_fns = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
        for p in lp_fns.parameters():
            p.requires_grad = False
        validate._lp = lp_fns
    l1s, lps = [], []
    Z, S = [], []
    rng = np.random.default_rng(1234)
    picks = rng.choice(len(ds), min(n, len(ds)), replace=False)
    for i in range(0, len(picks), bs):
        recs = [ds.get(int(p), 0) for p in picks[i:i + bs]]
        x = torch.stack([r["x"] for r in recs]).to(dev)
        z = model.encode(x).mean.float()
        y = model.decode(z).float()
        l1s.append(float((y - x).abs().mean()))
        lps.append(float(lp_fns(y.clamp(-1, 1), x.clamp(-1, 1)).mean()))
        Z.append(style_readout(z))
        S += [r["style_id"] for r in recs]
        del x, y, z
        torch.cuda.empty_cache()
    Z = torch.cat(Z)
    S = np.array(S)
    sim = (Z @ Z.t()).cpu().numpy()
    same = S[:, None] == S[None, :]
    np.fill_diagonal(same, False)
    pos = sim[same]; neg = sim[~same & ~np.eye(len(S), dtype=bool)]
    auroc = float((pos[:, None] > neg[None, :]).mean()) if len(pos) and len(neg) else float("nan")
    model.train()
    return {"l1": round(float(np.mean(l1s)), 5),
            "lpips": round(float(np.mean(lps)), 5),
            "style_auroc": round(auroc, 4)}


def calibrate_lam_s(model, x, meta, recon, lam_v, gamma, dev, n=4):
    """One-shot gradient-ratio calibration.

    Returns the UNSCALED ratio ||grad_z L_style|| / ||grad_z L_recon||.
    The caller then sets lam_s = TARGET_SHARE / ratio, so that the injected
    style gradient is exactly TARGET_SHARE of the reconstruction gradient.

    NOTE: g_s is the gradient of the UNSCALED style loss. The earlier version
    reused the old code's `lam_s * 0.15 / ratio` update, which assumed g_s had
    already been multiplied by lam_s -- that made the style gradient 10x too
    small (1.5% instead of 15%) and suppressed the contrastive objective.
    """
    xc = x[:n]
    # ONE encoder forward; its posterior serves both z and the KL term.
    post = model.encode(xc)
    z = post.mean
    q = style_readout(z)
    W = supcon_relation_weights(meta["style"][:n], meta["artist"][:n],
                                meta["view"][:n],
                                supervised=meta["supervised"][:n])
    l_s = supcon(q, W, TAU) + lam_v * vicreg_var(q, gamma)
    g_s, = torch.autograd.grad(l_s, z, retain_graph=True)
    y = model.decode(z)
    l_r, _ = recon(xc, y, post)
    g_r, = torch.autograd.grad(l_r, z)
    ratio = float(g_s.norm() / (g_r.norm() + 1e-12))
    del z, q, W, l_s, g_s, post, y, l_r, g_r
    torch.cuda.empty_cache()
    return ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["A", "B", "C"])
    ap.add_argument("--init-from", default="")
    ap.add_argument("--updates", type=int, default=0)
    ap.add_argument("--mb", type=int, default=0)
    ap.add_argument("--lam-s", type=float, default=0.1)
    ap.add_argument("--lam-v", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-calib", action="store_true")
    args = ap.parse_args()
    dev = args.device
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    cfg = dict(STAGES[args.stage])
    if args.updates:
        cfg["updates"] = args.updates
    if args.mb:
        cfg["mb"] = args.mb

    pool = TrainPool().describe()
    ds = ArtVAEDataset(pool)
    model = ArtVAE(dev).to(dev).train()
    trainable = set_trainable(model, args.stage)
    print(f"[Trainable] stage {args.stage}: {trainable}", flush=True)

    if args.init_from and Path(args.init_from).exists():
        ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
        model.encoder.load_state_dict(ck["encoder"])
        model.quant_conv.load_state_dict(ck["quant_conv"])
        model.decoder.decoder.load_state_dict(ck["decoder"])
        model.post_quant_conv.load_state_dict(ck["post_quant_conv"])
        print(f"[Init] {args.init_from} {ck.get('meta', {})}", flush=True)
        set_trainable(model, args.stage)

    import lpips
    lp_fn = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    for p in lp_fn.parameters():
        p.requires_grad = False
    recon = ReconstructionLoss(lpips_fn=lp_fn, lam_p=0.5, beta=1e-4,
                               kl_reduction="elem", lpips_size=512)
    gamma, gamma_emp = calibrate_gamma(model, ds, dev)
    print(f"[VICReg] empirical per-dim std={gamma_emp:.6f} -> gamma={gamma:.6f} "
          f"(target 0.7x, NOT the theoretical {gamma_for_dim(D_OUT):.6f})", flush=True)
    lam_v = args.lam_v
    enc_ps = [p for p in model.encoder.parameters() if p.requires_grad]
    dec_ps = [p for p in model.decoder.parameters() if p.requires_grad]
    qc_ps = [p for p in model.quant_conv.parameters() if p.requires_grad]
    groups = [{"params": enc_ps + qc_ps, "lr": cfg["lr_enc"]}]
    if dec_ps:
        groups.append({"params": dec_ps, "lr": cfg["lr_dec"]})
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 20) *
        (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / cfg["updates"])))))

    sampler = StyleContrastBatchSampler(pool, seed=args.seed,
                                        batches_per_epoch=cfg["updates"])
    ref_sampler = StyleContrastBatchSampler(pool, seed=args.seed + 7919,
                                            batches_per_epoch=1)
    ref_batch = collate(next(iter(ref_sampler)), ds)
    x_ref = ref_batch["x"].to(dev)
    W_ref = supcon_relation_weights(ref_batch["style"].to(dev),
                                    ref_batch["artist"].to(dev),
                                    ref_batch["view"].to(dev),
                                    supervised=ref_batch["supervised"].to(dev))
    print("[Ref] frozen reference batch built (64 images, seed+7919)", flush=True)

    lam_s = args.lam_s
    calib_done = False
    best = None
    t0 = time.time()
    log_path = OUT / f"train_{args.stage}.log"
    metrics_path = OUT / f"{args.stage}_metrics.jsonl"

    print(f"\n=== STAGE {args.stage}  updates={cfg['updates']}  mb={cfg['mb']}  "
          f"logical={LOGICAL_BATCH}  lam_s={lam_s} ===", flush=True)

    def encode_fn(xb):
        return model.encode(xb).mean

    for upd, bitems in enumerate(sampler, start=1):
        batch = collate(bitems, ds)
        x = batch["x"].to(dev)
        style = batch["style"].to(dev)
        artist = batch["artist"].to(dev)
        view = batch["view"].to(dev)
        sup = batch["supervised"].to(dev)
        W = supcon_relation_weights(style, artist, view, supervised=sup)
        meta = {"style": style, "artist": artist, "view": view,
                "supervised": sup}

        def style_loss_fn(q):
            return supcon(q, W, TAU) + lam_v * vicreg_var(q, gamma)

        # ---- one-shot lam_s calibration ----
        if upd == 20 and not calib_done and not args.no_calib:
            torch.cuda.empty_cache()
            ratio = calibrate_lam_s(model, x, meta, recon, lam_v, gamma, dev)
            if ratio > 0:
                lam_s = float(min(max(TARGET_SHARE / ratio, 1e-4), 10.0))
            calib_done = True
            print(f"  [CALIB] unscaled style/recon grad ratio={ratio:.4f} -> "
                  f"lam_s={lam_s:.5f} (target share {TARGET_SHARE:.0%})", flush=True)
            opt.zero_grad(set_to_none=True)

        # ---- pass 1: no-grad encoder, full-batch style grad on tiny q ----
        q_all, g_q, l_style = cache_style_gradient(
            encode_fn, style_readout, style_loss_fn, x, cfg["mb"])

        # ---- pass 2: inject style grad + accumulate reconstruction ----
        opt.zero_grad(set_to_none=True)
        rec_parts_sum = {}
        n_chunks = math.ceil(LOGICAL_BATCH / cfg["mb"])
        for i in range(0, LOGICAL_BATCH, cfg["mb"]):
            # ONE encoder forward per chunk (with grad): its posterior supplies
            # both z (for the injected style gradient) and the KL term.
            xc = x[i:i + cfg["mb"]]
            post = model.encode(xc)
            z = post.mean
            q = style_readout(z)
            (q * lam_s * g_q[i:i + cfg["mb"]]).sum().backward(retain_graph=True)
            y = model.decode(z)
            l_rec, parts = recon(xc, y, post)
            (l_rec / n_chunks).backward()
            for k, v in parts.items():
                rec_parts_sum[k] = rec_parts_sum.get(k, 0.0) + v / n_chunks
            del z, y, post
            torch.cuda.empty_cache()

        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
        opt.step(); sched.step()

        if upd % 20 == 0 or upd == 1:
            el = time.time() - t0
            ref_style = fixed_ref_style(model, x_ref, W_ref, gamma, lam_v, cfg["mb"])
            parts = " ".join(f"{k}={v:.4f}" for k, v in rec_parts_sum.items())
            print(f"  [{args.stage}] {upd:5d}/{cfg['updates']} | {parts} "
                  f"| style={l_style:.4f} ref_style={ref_style:.4f} | {el/upd:.2f}s/upd "
                  f"{LOGICAL_BATCH*upd/el:.1f} img/s | "
                  f"{(cfg['updates']-upd)*el/upd/60:.1f}min left", flush=True)
            log_path.open("a").write(
                f"{upd} style={l_style:.5f} ref_style={ref_style:.5f} " +
                " ".join(f"{k}={v:.5f}" for k, v in rec_parts_sum.items()) + "\n")

        if upd % 250 == 0 or upd == cfg["updates"]:
            mon = monitor(model, ds, pool, dev)
            val = validate(model, ds, pool, dev)
            rec = {"stage": args.stage, "upd": upd, "lam_s": round(lam_s, 5),
                   **mon, **val}
            print(f"  [{args.stage}] MONITOR {json.dumps(mon)}", flush=True)
            print(f"  [{args.stage}] VALID   {json.dumps(val)}", flush=True)
            metrics_path.open("a").write(json.dumps(rec) + "\n")
            if best is None or val["lpips"] < best["lpips"]:
                best = val
                torch.save({
                    "encoder": {k: v.detach().cpu() for k, v in model.encoder.state_dict().items()},
                    "quant_conv": {k: v.detach().cpu() for k, v in model.quant_conv.state_dict().items()},
                    "decoder": {k: v.detach().cpu() for k, v in model.decoder.decoder.state_dict().items()},
                    "post_quant_conv": {k: v.detach().cpu() for k, v in model.post_quant_conv.state_dict().items()},
                    "meta": {"stage": args.stage, "upd": upd, "lam_s": lam_s,
                             "best": best, "seed": args.seed},
                }, OUT / f"{args.stage}_best.pt")
                print(f"  [{args.stage}] saved best (lpips={val['lpips']})", flush=True)

    el = time.time() - t0
    torch.save({
        "encoder": {k: v.detach().cpu() for k, v in model.encoder.state_dict().items()},
        "quant_conv": {k: v.detach().cpu() for k, v in model.quant_conv.state_dict().items()},
        "decoder": {k: v.detach().cpu() for k, v in model.decoder.decoder.state_dict().items()},
        "post_quant_conv": {k: v.detach().cpu() for k, v in model.post_quant_conv.state_dict().items()},
        "meta": {"stage": args.stage, "upd": cfg["updates"], "lam_s": lam_s,
                 "best": best, "wall_min": round(el / 60, 1)},
    }, OUT / f"{args.stage}_last.pt")
    (OUT / f"train_{args.stage}_summary.json").write_text(json.dumps({
        "stage": args.stage, "updates": cfg["updates"], "best": best,
        "wall_min": round(el / 60, 1),
        "img_per_sec": round(LOGICAL_BATCH * cfg["updates"] / el, 2),
        "final_lam_s": lam_s}, indent=2), encoding="utf-8")
    print(f"  [{args.stage}] DONE {json.dumps(best)} wall={el/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
