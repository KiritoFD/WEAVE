"""WEAVE Style-Rank 20-class training (full-coverage epoch sampler).

L = L_rec + beta_KL * L_KL + lambda_r * L_rank + lambda_c * L_CE + lambda_a * L_anchor

Key components (Plan C):
  * 768-d Style center EMA:
      h_i = P z_i
      m_t = 0.95 * m_{t-1} + 0.05 * mean(h_i)  (accumulated over 48 images per update)
      u_i = normalize(h_i - m_t)
  * Negative queue coordinate consistency:
      Stored vectors are normalized with m_t subtracted at insertion time.
      Anchor uses current m_t.
      Negatives: K=4 content-matched (|c_ap - c_an| < 0.05), distinct styles, recent priority.
  * Weak A0 latent anchor:
      L_anchor = mean( (E_theta(x) - E_A0(x))^2 ) / 16384  (posterior mean, frozen A0)
      Calibrated to ~15% of reconstruction gradient.
  * Diagnostic tracking every 10 updates:
      |h|, c_ratio = ||m_t|| / E[||h||], z_std, sim_p, sim_n, margin, rank_acc, ce_acc, anchor.
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/Workspace/WEAVE_VAE")

os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/root/Workspace/.inductor_cache"
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader

from tools.vae_suite.dataset import RawMemmapWikiArtDataset
from tools.vae_suite.losses import compute_mixed_reconstruction_loss, compute_kl_divergence
from tools.vae_suite.style_rank_data import (RamStyleDataset, EpochCoverageSampler,
                                            pick_negatives, NegQueue)
import lpips

OUT = Path("/root/Workspace/WEAVE_VAE/runs/vae_style_rank20")   # frozen resources
RUN = OUT                                                          # run outputs (overridable)
MANIFEST = Path("/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")
A0_DIR = "/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae"
BETA_KL = 1.126e-4
MARGIN = 0.05
TAU_R = 0.10
TAU_C = 0.15


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        for p in self.shadow.parameters():
            p.requires_grad = False

    def update(self, model):
        with torch.no_grad():
            for s, m in zip(self.shadow.parameters(), model.parameters()):
                s.data.mul_(self.decay).add_(m.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()


def rank_loss(s_ap, s_an, valid=None, margin=MARGIN, temperature=TAU_R):
    """s_ap:(B,) s_an:(B,K) valid:(B,K) or None. Averaged over valid (anchor, negative) pairs."""
    if s_an.dim() == 1:
        s_an = s_an.unsqueeze(1)
    if valid is None:
        valid = torch.ones_like(s_an, dtype=torch.bool)
    per = F.softplus((s_an - s_ap.unsqueeze(1) + margin) / temperature)
    v = valid.float()
    n_ok = v.sum()
    if n_ok == 0:
        return s_ap.sum() * 0.0, {
            "sim_pos": float(s_ap.detach().mean()),
            "sim_neg": 0.0,
            "margin": 0.0,
            "rank_acc": 0.0,
            "n_neg": 0.0,
        }
    loss = (per * v).sum() / n_ok
    s_an_m = (s_an.detach() * v).sum() / n_ok
    s_ap_m = (s_ap.detach() * v.sum(1) / s_an.shape[1]).sum() / v.sum(1).clamp(min=1).sum()
    return loss, {
        "sim_pos": float(s_ap.detach().mean()),
        "sim_neg": float(s_an_m),
        "margin": float(s_ap_m - s_an_m),
        "rank_acc": float(((s_ap.unsqueeze(1) > s_an.detach()) & valid).float().sum() / n_ok),
        "n_neg": float(v.sum(1).mean()),
    }


# ---------------------------------------------------------------- validation
class EvalReader:
    """Reads the cached 20-class eval set straight from the memmap."""

    def __init__(self, manifest_path, raw_indices):
        man = json.loads(Path(manifest_path).read_text())
        self.mmap = np.memmap(man["bin_path"], dtype=np.uint8, mode="r",
                              shape=tuple(man["shape"]))
        self.raw = list(raw_indices)

    def get(self, idx_list, device):
        arr = np.stack([self.mmap[self.raw[i]] for i in idx_list])
        x = torch.from_numpy(arr).permute(0, 3, 1, 2).float().to(device)
        return x.div_(127.5).sub_(1.0)


def ssim_simple(x, y):
    x01 = (x.clamp(-1, 1) + 1) / 2; y01 = (y.clamp(-1, 1) + 1) / 2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mx = F.avg_pool2d(x01, 11, 1, 5); my = F.avg_pool2d(y01, 11, 1, 5)
    mxx, myy, mxy = mx * mx, my * my, mx * my
    sx = F.avg_pool2d(x01 * x01, 11, 1, 5) - mxx
    sy = F.avg_pool2d(y01 * y01, 11, 1, 5) - myy
    sxy = F.avg_pool2d(x01 * y01, 11, 1, 5) - mxy
    m = ((2 * mxy + C1) * (2 * sxy + C2)) / ((mxx + myy + C1) * (sx + sy + C2))
    return float(m.mean())


def _auroc(feats, pos, neg, cap=800):
    from sklearn.metrics import roc_auc_score
    n = min(len(pos), len(neg), cap)
    if n < 10:
        return None
    sp = np.array(pos[:n]); sn = np.array(neg[:n])
    a1 = (feats[sp[:, 0]] * feats[sp[:, 1]]).sum(1)
    a0 = (feats[sn[:, 0]] * feats[sn[:, 1]]).sum(1)
    lbs = np.concatenate([np.ones(n), np.zeros(n)])
    try:
        return round(float(roc_auc_score(lbs, np.concatenate([a1, a0])) * 100), 2)
    except ValueError:
        return None


@torch.no_grad()
def quick_validate(vae, ev, reader, P, m_center, lpips_fn, device,
                   n_img=800, n_rec=100, micro=2):
    """CM / Cross-Artist / Hard AUROC using m_center from training + recon metrics."""
    n_img = min(n_img, len(ev["raw"]))
    idx = np.arange(n_img)
    us = []
    for i in range(0, n_img, micro):
        chunk = idx[i:i + micro]
        x = reader.get(chunk, device)
        mu = vae.encode(x).latent_dist.mean.float().flatten(1)
        h = F.linear(mu, P)
        us.append(F.normalize(h - m_center.unsqueeze(0), dim=-1, eps=1e-6).cpu())
    u = torch.cat(us).numpy()

    def sub(pairs):
        return [(a, b) for a, b in pairs if a < n_img and b < n_img]

    res = {
        "n": int(n_img),
        "content_matched": _auroc(u, sub(ev["cm_pos"]), sub(ev["cm_neg"])),
        "cross_artist": _auroc(u, sub(ev["cross"]), sub(ev["diff"])),
        "same_genre_hard": _auroc(u, sub(ev["cross"]), sub(ev["hard"])),
    }

    lps, sss = [], []
    n_rec = min(n_rec, n_img)
    for i in range(0, n_rec, micro):
        chunk = idx[i:i + micro]
        x = reader.get(chunk, device)
        xh = vae.decode(vae.encode(x).latent_dist.mean).sample
        lps.append(float(lpips_fn(xh.float(), x.float()).mean()))
        sss.append(ssim_simple(xh, x))
    res["lpips"] = round(float(np.mean(lps)), 4) if lps else None
    res["ssim"] = round(float(np.mean(sss)), 4) if sss else None
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--hours", type=float, default=11.0)
    ap.add_argument("--accum", type=int, default=0,
                    help="0 = auto (effective batch 48 / images per microbatch)")
    ap.add_argument("--n-styles", type=int, default=1,
                    help="styles per microbatch; images = n_styles*2")
    ap.add_argument("--queue-size", type=int, default=4096,
                    help="negatives queue capacity")
    ap.add_argument("--neg-k", type=int, default=4,
                    help="content-matched negatives per anchor")
    ap.add_argument("--stale-every", type=int, default=100,
                    help="staleness monitoring interval (0 disables)")
    ap.add_argument("--drift-every", type=int, default=100,
                    help="drift monitoring interval (0 disables)")
    ap.add_argument("--lr-enc", type=float, default=3e-5)
    ap.add_argument("--lr-dec", type=float, default=2e-5)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--val-every", type=int, default=0,
                    help="0 disables in-training validation (recommended: run eval script after)")
    ap.add_argument("--val-images", type=int, default=800)
    ap.add_argument("--val-recon", type=int, default=100)
    ap.add_argument("--calib-batches", type=int, default=8)
    ap.add_argument("--calib-only", action="store_true")
    ap.add_argument("--speed-test", type=int, default=0)
    ap.add_argument("--resume", default="",
                    help="path to checkpoint to resume from (e.g. ckpt_u00200.pt)")
    ap.add_argument("--anchor-scale", type=float, default=1.0,
                    help="P1 ablation: multiply lambda_anchor by this (0 = no anchor)")
    ap.add_argument("--calib-from", default="",
                    help="load lambda_rank/ce/anchor from this calibration.json")
    ap.add_argument("--run-dir", default=str(OUT),
                    help="output directory for logs/checkpoints of this run")
    ap.add_argument("--preload", action="store_true",
                    help="load images into RAM (57.5 GB); default uses page-cached memmap")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    global RUN
    RUN = Path(args.run_dir)
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "checkpoints").mkdir(parents=True, exist_ok=True)
    print(f"[RunDir] {RUN} | anchor_scale={args.anchor_scale} "
          f"| calib_from={args.calib_from or 'recompute'}", flush=True)

    device = "cuda"
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # ---------- frozen coordinate interface ----------
    proj = torch.load(OUT / "projection.pt", weights_only=False)
    P = proj["P"].to(device).float()       # (768, 16384)
    mu_z = proj["mu_z"].to(device).float() # (1, 16384)
    proto = torch.load(OUT / "prototypes.pt", weights_only=False).to(device).float()
    idx = torch.load(OUT / "index.pt", weights_only=False)
    pos_list = idx["pos_list"]; yn = idx["y"].numpy(); an = idx["a"].numpy()
    classes = json.loads((OUT / "classes.json").read_text())
    print(f"[Load] P{tuple(P.shape)} mu_z{tuple(mu_z.shape)} proto{tuple(proto.shape)} "
          f"pool={len(pos_list)}", flush=True)

    # DINO content features, aligned with pos_list (no gradient)
    DINO = torch.load(OUT / "train_dino.pt", weights_only=False).to(device).float()
    DINO = F.normalize(DINO, dim=-1)
    print(f"[DINO] {tuple(DINO.shape)} on GPU ({DINO.numel()*4/1024**2:.0f} MB)", flush=True)

    # ---------- data ----------
    style_to_items = collections.defaultdict(list)
    for i in range(len(yn)):
        style_to_items[int(yn[i])].append((i, int(an[i])))

    ds = RamStyleDataset(MANIFEST, pos_list, preload=args.preload, style_ids=yn)
    sampler = EpochCoverageSampler(style_to_items, seed=args.seed, n_styles=args.n_styles)
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=4, pin_memory=True,
                        persistent_workers=False)
    micro_bs = args.n_styles * 2
    if args.accum <= 0:
        args.accum = max(1, 48 // micro_bs)
    print(f"[Data] microbatch={micro_bs} images ({args.n_styles} styles x 2) | "
          f"accum={args.accum} | effective={micro_bs*args.accum}", flush=True)

    # ---------- models ----------
    # 1. Main VAE (trained)
    vae = AutoencoderKL.from_pretrained(A0_DIR).to(device)
    vae.train()
    if args.no_compile:
        print("[Compile] disabled", flush=True)
    else:
        try:
            t_c = time.time()
            vae.encoder.compile(dynamic=False)
            vae.decoder.compile(dynamic=False)
            print(f"[Compile] encoder+decoder compiled in-place "
                  f"(dynamic=False) in {time.time()-t_c:.1f}s", flush=True)
        except AttributeError:
            vae.encoder = torch.compile(vae.encoder, dynamic=False)
            vae.decoder = torch.compile(vae.decoder, dynamic=False)
            print("[Compile] fell back to torch.compile wrapper", flush=True)

    # 2. Frozen A0 Encoder for Weak Latent Anchor
    vae_a0 = AutoencoderKL.from_pretrained(A0_DIR)
    del vae_a0.decoder
    del vae_a0.post_quant_conv
    vae_a0.to(device).eval()
    for p in vae_a0.parameters():
        p.requires_grad = False
    print("[A0_Anchor] frozen encoder loaded on GPU for trust-region anchor", flush=True)

    lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    for p in lpips_fn.parameters():
        p.requires_grad = False

    enc_params = [p for p in vae.encoder.parameters() if p.requires_grad] + \
                 [p for p in vae.quant_conv.parameters() if p.requires_grad]

    opt = torch.optim.AdamW([
        {"params": list(vae.encoder.parameters()), "lr": args.lr_enc},
        {"params": list(vae.decoder.parameters()), "lr": args.lr_dec},
        {"params": list(vae.quant_conv.parameters()) + list(vae.post_quant_conv.parameters()),
         "lr": args.lr_enc},
    ], betas=(0.9, 0.99), weight_decay=args.wd)

    def lr_lambda(s):
        if s < args.warmup:
            return float(s) / max(1, args.warmup)
        return 0.2 + 0.8 * 0.5 * (1 + math.cos(math.pi * min(1.0, (s - args.warmup) / 4000)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    ema = EMA(vae, 0.999)

    it = iter(loader)

    def next_batch():
        nonlocal it
        try:
            return next(it)
        except StopIteration:
            it = iter(loader)
            return next(it)

    negq = NegQueue(args.queue_size, 768, device) if args.n_styles == 1 else None
    if negq is not None:
        print(f"[NegQueue] size={args.queue_size} K={args.neg_k} "
              f"tol={negq.tol} recent={negq.recent}", flush=True)

    # Initial style center m_0 from A0 training statistics
    m_t = F.linear(mu_z, P).squeeze(0).detach() # shape (768,)
    print(f"[Center] initialized m_0 from A0 train mean: ||m_0|| = {m_t.norm().item():.2f}", flush=True)
    h_accum = torch.zeros(768, device=device)
    h_count = 0

    start_upd = 0
    if args.resume and Path(args.resume).exists():
        print(f"[Resume] loading {args.resume}...", flush=True)
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        vae.load_state_dict(ck["model"])
        ema.shadow.load_state_dict(ck.get("ema", ck["model"]))
        opt.load_state_dict(ck["optimizer"])
        if "m_center" in ck:
            m_t = ck["m_center"].to(device).float()
            print(f"[Resume] loaded m_center: norm={m_t.norm().item():.2f}", flush=True)
        start_upd = ck.get("upd", 0)
        for _ in range(start_upd):
            sched.step()
        print(f"[Resume] successfully resumed from update {start_upd}", flush=True)

    ev = None
    reader = None
    if args.val_every > 0:
        fv = OUT / "eval_pairs.pt"
        if fv.exists():
            ev = torch.load(fv, weights_only=False)
            reader = EvalReader(MANIFEST, ev["raw"])
            print(f"[Val] {len(ev['raw'])} eval images loaded", flush=True)

    def forward_terms(x, pos, y):
        """x:(B,3,512,512) pos:(B,) local indices y:(B,) new style index 0..19."""
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            l_rec, _, _, post, _ = compute_mixed_reconstruction_loss(vae, x, lpips_fn)
            kl_e, kl_v, _, _ = compute_kl_divergence(post)

        # Frozen A0 target for latent anchor
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                h_a0 = vae_a0.encoder(x)
                h_a0 = vae_a0.quant_conv(h_a0)
                mu_a0 = h_a0[:, :4].float().flatten(1)

        mu = post.mean.float().flatten(1)
        l_anchor = F.mse_loss(mu, mu_a0)

        # 768-d Style projection centered by EMA center m_t
        h = F.linear(mu, P)
        u = F.normalize(h - m_t.unsqueeze(0), dim=-1, eps=1e-6)

        dino_b = DINO[pos]
        if negq is not None:
            s_ap, s_an, neg_valid, match_ok = negq.pick(u, dino_b, y, K=args.neg_k)
            l_rank, rst = rank_loss(s_ap, s_an, neg_valid)
            st = {"match_lt005": match_ok, **rst}
            negq.push(u, dino_b, y, pos)
        else:
            partner, neg, c_ap, c_an, match_ok = pick_negatives(dino_b)
            ar = torch.arange(u.shape[0], device=u.device)
            S = u @ u.t()
            s_ap, s_an = S[ar, partner], S[ar, neg]
            l_rank, rst = rank_loss(s_ap, s_an)
            st = {"match_lt005": match_ok, **rst}

        logits = u @ proto.t() / TAU_C
        l_ce = F.cross_entropy(logits, y)
        ce_acc = float((logits.argmax(-1) == y).float().mean())
        return l_rec, kl_e, kl_v, l_rank, l_ce, l_anchor, st, ce_acc, h, u, mu

    def flat_grad(loss, params, retain=False):
        gs = torch.autograd.grad(loss, params, retain_graph=retain, allow_unused=True)
        return torch.cat([g.reshape(-1) if g is not None else torch.zeros_like(p).reshape(-1)
                          for g, p in zip(gs, params)])

    # ================= negative queue pre-warm (all branches) =================
    if negq is not None and negq.n == 0 and not args.resume:
        print("[NegQueue] pre-warming with 32 microbatches...", flush=True)
        with torch.no_grad():
            for _ in range(32):
                bw = next_batch()
                xw = bw["image"].to(device, non_blocking=True)
                pw = bw["pos"].to(device, non_blocking=True)
                yw = bw["style_id"].to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    post_w = vae.encode(xw).latent_dist
                mu_w = post_w.mean.float().flatten(1)
                hw = F.linear(mu_w, P)
                uw = F.normalize(hw - m_t.unsqueeze(0), dim=-1, eps=1e-6)
                negq.push(uw, DINO[pw], yw, pw)
        print(f"[NegQueue] warm: {negq.n} items ready", flush=True)

    # ================= gradient calibration =================
    calib_file = RUN / "calibration.json"
    if args.calib_from and Path(args.calib_from).exists():
        base = json.loads(Path(args.calib_from).read_text())
        calib = dict(base)
        lam_r = base["lambda_rank"]
        lam_c = base["lambda_ce"]
        lam_a = base["lambda_anchor"] * args.anchor_scale
        calib["lambda_anchor"] = round(lam_a, 6)
        calib["anchor_scale"] = args.anchor_scale
        calib["calib_from"] = args.calib_from
        calib_file.write_text(json.dumps(calib, indent=2), encoding="utf-8")
        print(f"[Calibration] loaded from {args.calib_from}; "
              f"lambda_rank={lam_r} lambda_ce={lam_c} "
              f"lambda_anchor={lam_a:.8f} (scale={args.anchor_scale})", flush=True)
    elif args.resume and calib_file.exists():
        calib = json.loads(calib_file.read_text())
        lam_r = calib["lambda_rank"]
        lam_c = calib["lambda_ce"]
        lam_a = calib["lambda_anchor"]
        print(f"[Calibration] resumed: lambda_rank={lam_r} lambda_ce={lam_c} lambda_anchor={lam_a}", flush=True)
    else:
        print(f"\n=== Gradient calibration ({args.calib_batches} microbatches) ===", flush=True)
        gb, gr, gc, ga = [], [], [], []
        cos_rb, cos_rc, cos_ra = [], [], []

        for bi in range(args.calib_batches):
            b = next_batch()
            x = b["image"].to(device, non_blocking=True)
            pos = b["pos"].to(device, non_blocking=True)
            y = b["style_id"].to(device, non_blocking=True)
            l_rec, kl_e, kl_v, l_rank, l_ce, l_anchor, st, ce_acc, h, u, _ = forward_terms(x, pos, y)
            base = l_rec + BETA_KL * kl_e
            g_base = flat_grad(base, enc_params, retain=True)
            g_r = flat_grad(l_rank, enc_params, retain=True)
            g_c = flat_grad(l_ce, enc_params, retain=True)
            g_a = flat_grad(l_anchor, enc_params, retain=False)
            gb.append(g_base.norm().item()); gr.append(g_r.norm().item())
            gc.append(g_c.norm().item()); ga.append(g_a.norm().item())
            cos_rb.append(float(F.cosine_similarity(g_r, g_base, dim=0)))
            cos_rc.append(float(F.cosine_similarity(g_r, g_c, dim=0)))
            cos_ra.append(float(F.cosine_similarity(g_r, g_a, dim=0)))
            if bi % 4 == 0:
                print(f"  calib {bi}: g_base={gb[-1]:.4f} g_rank={gr[-1]:.4f} g_ce={gc[-1]:.4f} "
                      f"g_anch={ga[-1]:.4f} cos(r,base)={cos_rb[-1]:+.3f} | racc={st['rank_acc']:.2f}",
                      flush=True)
            del g_base, g_r, g_c, g_a
            torch.cuda.empty_cache()

        gb_m = float(np.median(gb)); gr_m = float(np.median(gr))
        gc_m = float(np.median(gc)); ga_m = float(np.median(ga))
        lam_r = 0.5 * gb_m / max(gr_m, 1e-12)
        lam_c = 0.1 * gb_m / max(gc_m, 1e-12)
        lam_a = 0.15 * gb_m / max(ga_m, 1e-12) # ~15% of base
        calib = {"g_base_median": round(gb_m, 6), "g_rank_median": round(gr_m, 6),
                 "g_ce_median": round(gc_m, 6), "g_anchor_median": round(ga_m, 6),
                 "lambda_rank": round(lam_r, 6), "lambda_ce": round(lam_c, 6),
                 "lambda_anchor": round(lam_a, 6),
                 "cos_rank_base_mean": round(float(np.mean(cos_rb)), 4),
                 "cos_rank_ce_mean": round(float(np.mean(cos_rc)), 4),
                 "cos_rank_anchor_mean": round(float(np.mean(cos_ra)), 4),
                 "target_ratios": {"rank": 0.5, "ce": 0.1, "anchor": 0.15}}
        calib_file.write_text(json.dumps(calib, indent=2), encoding="utf-8")
        print(f"\n[Calibration] g_base={gb_m:.4f} g_rank={gr_m:.4f} g_ce={gc_m:.4f} g_anchor={ga_m:.4f}", flush=True)
        print(f"  -> lambda_rank={lam_r:.5f} (target 0.5x)  lambda_ce={lam_c:.5f} (target 0.1x)  lambda_anchor={lam_a:.5f} (target 0.15x)",
              flush=True)
        print(f"  cos(rank, base)={calib['cos_rank_base_mean']:+.4f} | cos(rank, anchor)={calib['cos_rank_anchor_mean']:+.4f}", flush=True)

    if args.calib_only:
        print("[calib-only] exit", flush=True)
        return

    # ================= training =================
    t0 = time.time()
    total_sec = args.hours * 3600
    upd = start_upd
    hist = []
    best_cm = -1.0
    mf = (RUN / "train_metrics.jsonl").open("a" if start_upd > 0 else "w", encoding="utf-8")

    while True:
        el = time.time() - t0
        if args.steps > 0 and upd >= args.steps:
            break
        if args.steps == 0 and el >= total_sec:
            break
        if args.speed_test and upd >= args.speed_test:
            break

        t_up = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        acc = collections.defaultdict(float)

        for _ in range(args.accum):
            b = next_batch()
            x = b["image"].to(device, non_blocking=True)
            pos = b["pos"].to(device, non_blocking=True)
            y = b["style_id"].to(device, non_blocking=True)
            l_rec, kl_e, kl_v, l_rank, l_ce, l_anchor, st, ce_acc, h, u, mu = forward_terms(x, pos, y)
            loss = (l_rec + BETA_KL * kl_e + lam_r * l_rank + lam_c * l_ce + lam_a * l_anchor) / args.accum
            loss.backward()

            acc["rec"] += float(l_rec.detach()) / args.accum
            acc["kl"] += float(kl_v) / args.accum
            acc["rank"] += float(l_rank.detach()) / args.accum
            acc["ce"] += float(l_ce.detach()) / args.accum
            acc["anchor"] += float(l_anchor.detach()) / args.accum
            acc["sim_pos"] += st["sim_pos"] / args.accum
            acc["sim_neg"] += st["sim_neg"] / args.accum
            acc["rank_acc"] += st["rank_acc"] / args.accum
            acc["ce_acc"] += ce_acc / args.accum
            acc["match_lt005"] += st["match_lt005"] / args.accum
            acc["n_neg"] += st.get("n_neg", 1.0) / args.accum
            acc["h_norm"] += float(h.norm(dim=-1).mean()) / args.accum
            acc["z_std"] += float(mu.view(-1, 4, 4096).std(dim=-1).mean()) / args.accum

            h_accum += h.detach().sum(dim=0)
            h_count += x.shape[0]

        # Optimizer step
        gn = float(torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0))
        opt.step(); sched.step(); ema.update(vae)
        upd += 1
        dt = time.perf_counter() - t_up

        # Update EMA style center m_t once per optimizer step (over 48 images)
        h_bar = h_accum / max(h_count, 1)
        m_t = 0.95 * m_t + 0.05 * h_bar
        m_t = m_t.detach()
        h_accum.zero_()
        h_count = 0

        # Periodic logging every 10 updates
        if upd % 10 == 0 or upd == 1:
            el = time.time() - t0
            ep = upd / max(sampler.n_micro // args.accum, 1)
            c_norm = float(m_t.norm().item())
            h_norm = float(acc["h_norm"])
            c_ratio = round(c_norm / max(h_norm, 1e-12), 4)
            z_std = round(float(acc["z_std"]), 4)

            print(f"[SR] upd {upd:5d} ep{ep:5.2f} | rec={acc['rec']:.4f} kl={acc['kl']:.2f} "
                  f"rank={acc['rank']:.4f} ce={acc['ce']:.3f} anch={acc['anchor']:.4f} | "
                  f"sim_p={acc['sim_pos']:+.3f} sim_n={acc['sim_neg']:+.3f} "
                  f"marg={acc['sim_pos']-acc['sim_neg']:+.3f} racc={acc['rank_acc']:.2f} "
                  f"ceacc={acc['ce_acc']:.2f} mt={acc['match_lt005']:.2f} | "
                  f"|h|={h_norm:.1f} c_rat={c_ratio:.3f} z_std={z_std:.3f} g={gn:.2f} | {dt:.2f}s | {el/3600:.2f}h",
                  flush=True)
            rec = {"upd": upd, "epoch": round(ep, 3),
                   **{k: round(v, 5) for k, v in acc.items()},
                   "c_norm": round(c_norm, 4), "c_ratio": c_ratio, "z_std": z_std,
                   "grad_norm": round(gn, 3), "sec": round(dt, 2),
                   "elapsed_h": round(el / 3600, 3), "lr_enc": sched.get_last_lr()[0]}
            hist.append(rec)
            mf.write(json.dumps(rec) + "\n"); mf.flush()

        # Drift check (staggered at odd 50s, e.g. 50, 150)
        if args.drift_every > 0 and (upd % args.drift_every == 50):
            dr = []
            for _ in range(4):
                bb = next_batch()
                xx = bb["image"].to(device, non_blocking=True)
                pp = bb["pos"].to(device, non_blocking=True)
                yy = bb["style_id"].to(device, non_blocking=True)
                l_r, kl_ee, _, l_rk, l_c, l_a, _, _, _, _, _ = forward_terms(xx, pp, yy)
                gb_ = flat_grad(l_r + BETA_KL * kl_ee, enc_params, retain=True)
                gr_ = flat_grad(l_rk, enc_params, retain=True)
                gc_ = flat_grad(l_c, enc_params, retain=True)
                ga_ = flat_grad(l_a, enc_params, retain=False)
                dr.append({"g_base": gb_.norm().item(), "g_rank": gr_.norm().item(),
                           "g_ce": gc_.norm().item(), "g_anchor": ga_.norm().item(),
                           "cos_rank_base": float(F.cosine_similarity(gr_, gb_, dim=0))})
                del gb_, gr_, gc_, ga_
                torch.cuda.empty_cache()
            gbm = float(np.median([d["g_base"] for d in dr]))
            grm = float(np.median([d["g_rank"] for d in dr]))
            gcm = float(np.median([d["g_ce"] for d in dr]))
            gam = float(np.median([d["g_anchor"] for d in dr]))
            drift = {"upd": upd,
                     "ratio_rank_base": round(lam_r * grm / max(gbm, 1e-12), 4),
                     "ratio_ce_base": round(lam_c * gcm / max(gbm, 1e-12), 4),
                     "ratio_anchor_base": round(lam_a * gam / max(gbm, 1e-12), 4),
                     "cos_rank_base": round(float(np.mean([d["cos_rank_base"] for d in dr])), 4)}
            mf.write(json.dumps({"drift": drift}) + "\n"); mf.flush()
            print(f"[DRIFT] upd {upd}: |lr*gr|/g_base={drift['ratio_rank_base']:.3f} "
                  f"|la*ga|/g_base={drift['ratio_anchor_base']:.3f} "
                  f"cos(rank,base)={drift['cos_rank_base']:+.3f}", flush=True)

        # Staleness check (staggered at even 100s, e.g. 100, 200)
        if negq is not None and args.stale_every > 0 and (upd % args.stale_every == 0):
            slots, raws = negq.sample_slots(32)
            if raws is not None:
                u_old = negq.qu[slots].clone()
                raw_list = [int(r) for r in raws.cpu().tolist()]
                nows = []
                was_training = vae.training
                vae.eval()
                try:
                    with torch.no_grad():
                        for i in range(0, len(raw_list), micro_bs):
                            chunk = raw_list[i:i + micro_bs]
                            xr = torch.stack([ds[r]["image"] for r in chunk]).to(device)
                            mu = vae.encode(xr).latent_dist.mean.float().flatten(1)
                            h_now = F.linear(mu, P)
                            nows.append(F.normalize(h_now - m_t.unsqueeze(0), dim=-1, eps=1e-6))
                finally:
                    vae.train(was_training)
                u_now = torch.cat(nows)
                d_stale = float((1.0 - (u_old * u_now).sum(-1)).mean())
                a = u_now[0]
                d_sim = float(((u_old * a).sum(-1) - (u_now * a).sum(-1)).abs().mean())
                sm = {"upd": upd, "d_stale": round(d_stale, 5),
                      "d_sim_to_anchor": round(d_sim, 5),
                      "margin": MARGIN,
                      "ratio_vs_margin": round(d_sim / MARGIN, 4)}
                mf.write(json.dumps({"stale": sm}) + "\n"); mf.flush()
                print(f"[STALE] upd {upd}: d_stale(1-cos)={d_stale:.5f} | "
                      f"|Δs(anchor)|={d_sim:.5f} vs margin {MARGIN} "
                      f"-> {d_sim/MARGIN*100:.1f}% of margin", flush=True)
                del u_old, u_now
                torch.cuda.empty_cache()

        # Checkpoints
        if upd % args.save_every == 0:
            ck = RUN / "checkpoints" / f"ckpt_u{upd:05d}.pt"
            ck.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": vae.state_dict(), "ema": ema.state_dict(),
                        "optimizer": opt.state_dict(), "upd": upd,
                        "m_center": m_t.cpu(), "args": vars(args)}, ck)
            torch.save(m_t.cpu(), RUN / "m_center.pt")
            print(f"[Saved] {ck} + m_center.pt (norm={m_t.norm().item():.2f})", flush=True)

        # In-training validation (if explicitly requested, else runs via clean subprocess after)
        if ev is not None and args.val_every > 0 and (upd % args.val_every == 0):
            was_training = vae.training
            vae.eval()
            try:
                vm = quick_validate(vae, ev, reader, P, m_t, lpips_fn, device,
                                    n_img=args.val_images, n_rec=args.val_recon,
                                    micro=micro_bs)
            finally:
                vae.train(was_training)
            vm["upd"] = upd
            vm["epoch"] = round(upd / max(sampler.n_micro // args.accum, 1), 3)
            mf.write(json.dumps({"val": vm}) + "\n"); mf.flush()
            print(f"[VAL] upd {upd} ep{vm['epoch']:.2f} | CM={vm['content_matched']} "
                  f"Cross={vm['cross_artist']} Hard={vm['same_genre_hard']} | "
                  f"LPIPS={vm['lpips']} SSIM={vm['ssim']}", flush=True)
            if vm["content_matched"] is not None and vm["content_matched"] > best_cm:
                best_cm = vm["content_matched"]
                bdir = RUN / "best_diffusers_vae"
                vae.save_pretrained(str(bdir))
                torch.save(m_t.cpu(), bdir / "m_center.pt")
                (RUN / "best_meta.json").write_text(json.dumps(
                    {"upd": upd, "content_matched": best_cm, **vm}, indent=2), encoding="utf-8")
                print(f"[BEST] upd {upd} CM={best_cm} -> {bdir}", flush=True)
            torch.cuda.empty_cache()

    mf.close()
    final = RUN / "final_diffusers_vae"
    vae.load_state_dict(ema.state_dict())
    vae.save_pretrained(str(final))
    torch.save(m_t.cpu(), final / "m_center.pt")
    torch.save(m_t.cpu(), RUN / "m_center.pt")
    (RUN / "train_log.json").write_text(json.dumps({
        "updates": upd, "hours": round((time.time() - t0) / 3600, 3),
        "microbatches_per_epoch": sampler.n_micro,
        "updates_per_epoch": sampler.n_micro // args.accum,
        "epochs_completed": round(upd / max(sampler.n_micro // args.accum, 1), 3),
        "m_center_norm": round(float(m_t.norm().item()), 4),
        "anchor_scale": args.anchor_scale,
        "calibration": calib, "history": hist}, indent=2), encoding="utf-8")
    print(f"\n[Done] {upd} updates in {(time.time()-t0)/3600:.2f}h -> {final}", flush=True)


if __name__ == "__main__":
    main()
