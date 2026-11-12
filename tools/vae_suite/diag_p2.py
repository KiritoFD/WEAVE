"""WEAVE-VAE V2 Diagnosis: P2 Loss & Gradient Attribution.

P2.1 Scale-shortcut counterfactual:
  z_s = mu_ref + s*(z - mu_ref), s in {0.25..2.0}
  Measure: style loss, d_pos, d_neg, rel margin, AUROC, latent RMS.

P2.2 Gradient conflict under real AdamW:
  Compute g_rec, g_style, g_kl on full Encoder param subspace.
  Measure cos(g_s, g_r), cos(g_s, g_k), cos(g_s, g_total),
  and g_s^T delta_theta_AdamW (one-step diagnostic with fresh optimizer).

P2.3 Counterfactual single-step updates (in-memory copies):
  recon-only / style-only / joint / deconflicted / scale-normalized style.

P2.4 Positive sample strategy analysis:
  d_near2, d_random_crossartist, d_far2, ratio curves.

P2.5 Negative mining balance.
"""
from __future__ import annotations

import copy
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "/root/Workspace/WEAVE_VAE")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from tools.vae_suite.dataset import RawMemmapWikiArtDataset
from tools.vae_suite.swd_metric import LatentSWDDescriptor
from tools.vae_suite.losses import compute_mixed_reconstruction_loss, compute_kl_divergence
from tools.vae_suite.reference_bank import StyleReferenceBank, SupervisedStyleRankingLoss
import lpips

OUT_DIR = Path("/root/Workspace/WEAVE_VAE/runs_vae/v2_diagnosis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST = Path("/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")
CALIB = Path("/root/Workspace/WEAVE_VAE/swd_calibration.json")
A0_DIR = "/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae"
V2_CKPT_DIR = Path("/root/Workspace/WEAVE_VAE/runs_vae/V2")

BETA_KL = 2.252e-5
LAMBDA_STYLE = 0.01


def build_desc():
    c = json.loads(CALIB.read_text())
    d = LatentSWDDescriptor()
    d.set_calibration(
        torch.tensor(c["chan_mean"]), torch.tensor(c["chan_std"]),
        c["scale_swd1"], c["scale_swd2"], c["scale_moment"], c["median_inter_dist"],
    )
    return d, c


def load_model(ckpt_name=None):
    vae = AutoencoderKL.from_pretrained(A0_DIR)
    if ckpt_name:
        ck = torch.load(V2_CKPT_DIR / ckpt_name, map_location="cpu", weights_only=False)
        vae.load_state_dict(ck.get("ema", ck["model"]))
    return vae


def get_fixed_latents(vae, dataset, n_batches=8, device="cuda"):
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    zs, ys, ars, ges = [], [], [], []
    with torch.no_grad():
        for i, b in enumerate(loader):
            if i >= n_batches:
                break
            post = vae.encode(b["image"].to(device)).latent_dist
            zs.append(post.mean.float())
            ys.append(b["style_id"].to(device))
            ars.append(b["artist_id"].to(device))
            ges.append(b["genre_id"].to(device))
    return torch.cat(zs), torch.cat(ys), torch.cat(ars), torch.cat(ges)


# ---------------------------------------------------------------- P2.1
def p2_1_scale_shortcut(desc, z, styles, artists, genres):
    print("\n===== P2.1 Scale Shortcut Counterfactual =====", flush=True)
    mu_ref = z.mean(dim=(0, 2, 3), keepdim=True)
    zc = z - mu_ref

    # fixed pair sets
    rng = np.random.default_rng(42)
    N = len(styles)
    same, diff = [], []
    for i in range(N):
        sc = np.where(styles == styles[i])[0]
        sc = sc[sc != i]
        if len(sc):
            same.append((i, int(rng.choice(sc))))
        dc = np.where(styles != styles[i])[0]
        if len(dc):
            diff.append((i, int(rng.choice(dc))))
    n = min(len(same), len(diff))
    sp, dp = np.array(same[:n]), np.array(diff[:n])

    rows = []
    for s in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        zs = mu_ref + s * zc
        with torch.no_grad():
            q1, q2, mom = desc(zs)
            dpos = torch.diag(desc.compute_pairwise_distance(
                q1[sp[:, 0]], q2[sp[:, 0]], mom[sp[:, 0]],
                q1[sp[:, 1]], q2[sp[:, 1]], mom[sp[:, 1]])).cpu().numpy()
            dneg = torch.diag(desc.compute_pairwise_distance(
                q1[dp[:, 0]], q2[dp[:, 0]], mom[dp[:, 0]],
                q1[dp[:, 1]], q2[dp[:, 1]], mom[dp[:, 1]])).cpu().numpy()

        # style loss proxy: softplus((dpos - dneg + m)/tau)
        m, tau = 0.15, 0.20
        l_style = float(F.softplus((torch.tensor(dpos - dneg) + m) / tau).mean().item())

        ia = np.concatenate([sp[:, 0], dp[:, 0]])
        ib = np.concatenate([sp[:, 1], dp[:, 1]])
        lbs = np.concatenate([np.ones(n), np.zeros(n)])
        with torch.no_grad():
            dall = torch.diag(desc.compute_pairwise_distance(
                q1[ia], q2[ia], mom[ia], q1[ib], q2[ib], mom[ib])).cpu().numpy()
        auc = float(roc_auc_score(lbs, -dall) * 100.0)

        rows.append({
            "scale": s,
            "latent_rms": round(float(zs.flatten(1).pow(2).mean().sqrt().item()), 4),
            "d_pos": round(float(dpos.mean()), 4),
            "d_neg": round(float(dneg.mean()), 4),
            "ratio_dpos_dneg": round(float(dpos.mean() / dneg.mean()), 4),
            "rel_margin": round(float((dneg.mean() - dpos.mean()) / (dneg.mean() + dpos.mean())), 4),
            "style_loss_proxy": round(l_style, 4),
            "auroc": round(auc, 2),
        })
        print(f"  s={s:.2f}: RMS={rows[-1]['latent_rms']:.3f} d_pos={rows[-1]['d_pos']:.4f} "
              f"d_neg={rows[-1]['d_neg']:.4f} Loss={l_style:.4f} AUROC={auc:.2f}%", flush=True)

    # radial gradient component
    z_req = z[:32].clone().requires_grad_(True)
    mu_r = mu_ref
    ref_bank = StyleReferenceBank(num_styles=136, refs_per_style=4, descriptor=desc)
    ref_bank.bank_q1.normal_(); ref_bank.bank_q2.normal_(); ref_bank.bank_mom.normal_()
    ref_bank.bank_styles = torch.tensor([s for s in range(136) for _ in range(4)])
    ref_bank.bank_artists.fill_(1); ref_bank.bank_genres.fill_(1)
    loss_fn = SupervisedStyleRankingLoss()
    q1, q2, mom = desc(z_req)
    l_s, _ = loss_fn(q1, q2, mom, torch.tensor(styles[:32]), torch.tensor(artists[:32]),
                     torch.tensor(genres[:32]), ref_bank)
    gz = torch.autograd.grad(l_s, z_req)[0]
    radial = (z_req - mu_r).flatten(1)
    gz_flat = gz.flatten(1)
    cos_radial = float(F.cosine_similarity(gz_flat, radial, dim=1).mean().item())
    print(f"  cos(grad_style, z-mu_ref) = {cos_radial:+.4f}", flush=True)

    return {"scale_sweep": rows, "cos_grad_style_radial": round(cos_radial, 4)}


# ---------------------------------------------------------------- P2.2
def p2_2_gradient_conflict(vae, desc, dataset, device="cuda"):
    print("\n===== P2.2 Gradient Conflict (Full Encoder Subspace) =====", flush=True)
    vae.train()
    vae.enable_gradient_checkpointing()
    lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    ref_bank = StyleReferenceBank(num_styles=136, refs_per_style=4, descriptor=desc).to(device)
    ref_bank.bank_q1.normal_(); ref_bank.bank_q2.normal_(); ref_bank.bank_mom.normal_()
    ref_bank.bank_styles = torch.tensor([s for s in range(136) for _ in range(4)], device=device)
    ref_bank.bank_artists.fill_(1); ref_bank.bank_genres.fill_(1)
    loss_fn = SupervisedStyleRankingLoss()

    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    enc_params = [p for p in vae.encoder.parameters() if p.requires_grad]
    qc_params = [p for p in vae.quant_conv.parameters() if p.requires_grad]
    target_params = enc_params + qc_params

    def flatten_params(params):
        return torch.cat([p.reshape(-1) for p in params])

    def collect(loss_name, x, st, ar, ge, params):
        """Run ONE forward+backward for a single loss term; return flattened grad."""
        vae.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            l_rec, _, _, post, _ = compute_mixed_reconstruction_loss(vae, x, lpips_fn)
            if loss_name == "rec":
                loss = l_rec
            else:
                kl_e, _, _, _ = compute_kl_divergence(post)
                if loss_name == "kl":
                    loss = kl_e
                else:
                    q1, q2, mom = desc(post.mean.float())
                    l_st, stats = loss_fn(q1, q2, mom, st, ar, ge, ref_bank)
                    loss = l_st
        loss.backward()
        g = torch.cat([(p.grad.reshape(-1) if p.grad is not None else torch.zeros_like(p).reshape(-1))
                       for p in params]).clone()
        val = float(loss.item())
        vae.zero_grad(set_to_none=True)
        return g, val

    metrics = []
    for bi, b in enumerate(loader):
        if bi >= 4:
            break
        x = b["image"].to(device)
        st = b["style_id"].to(device)
        ar = b["artist_id"].to(device)
        ge = b["genre_id"].to(device)

        g_r, l_rec_v = collect("rec", x, st, ar, ge, target_params)
        g_k, kl_v = collect("kl", x, st, ar, ge, target_params)
        g_s, l_st_v = collect("style", x, st, ar, ge, target_params)

        ws = LAMBDA_STYLE
        wk = BETA_KL
        g_s_w = g_s * ws
        g_k_w = g_k * wk
        g_total = g_r + g_s_w + g_k_w

        cos_sr = float(F.cosine_similarity(g_s_w, g_r, dim=0).item())
        cos_sk = float(F.cosine_similarity(g_s_w, g_k_w, dim=0).item())
        cos_rk = float(F.cosine_similarity(g_r, g_k_w, dim=0).item())
        cos_s_tot = float(F.cosine_similarity(g_s_w, g_total, dim=0).item())
        ratio_sr = float((g_s_w.norm() / g_r.norm().clamp(min=1e-12)).item())
        ratio_kr = float((g_k_w.norm() / g_r.norm().clamp(min=1e-12)).item())

        metrics.append({
            "batch": bi,
            "l_rec": round(l_rec_v, 4),
            "l_style": round(l_st_v, 4),
            "kl": round(kl_v, 2),
            "norm_g_style_w": round(float(g_s_w.norm().item()), 6),
            "norm_g_rec": round(float(g_r.norm().item()), 6),
            "norm_g_kl_w": round(float(g_k_w.norm().item()), 8),
            "ratio_style_rec": round(ratio_sr, 4),
            "ratio_kl_rec": round(ratio_kr, 6),
            "cos_style_rec": round(cos_sr, 4),
            "cos_style_kl": round(cos_sk, 4),
            "cos_rec_kl": round(cos_rk, 4),
            "cos_style_total": round(cos_s_tot, 4),
        })
        print(f"  b{bi}: cos(s,r)={cos_sr:+.3f} cos(s,k)={cos_sk:+.3f} "
              f"cos(s,tot)={cos_s_tot:+.3f} |g_s|/|g_r|={ratio_sr:.3f}", flush=True)
        del g_r, g_k, g_s, g_s_w, g_k_w, g_total
        torch.cuda.empty_cache()

    agg = {
        "cos_style_rec_mean": round(float(np.mean([m["cos_style_rec"] for m in metrics])), 4),
        "cos_style_kl_mean": round(float(np.mean([m["cos_style_kl"] for m in metrics])), 4),
        "cos_rec_kl_mean": round(float(np.mean([m["cos_rec_kl"] for m in metrics])), 4),
        "cos_style_total_mean": round(float(np.mean([m["cos_style_total"] for m in metrics])), 4),
        "ratio_style_rec_mean": round(float(np.mean([m["ratio_style_rec"] for m in metrics])), 4),
        "ratio_kl_rec_mean": round(float(np.mean([m["ratio_kl_rec"] for m in metrics])), 8),
        "per_batch": metrics,
    }
    print(f"\n  MEAN: cos(s,r)={agg['cos_style_rec_mean']:+.4f} | cos(s,tot)={agg['cos_style_total_mean']:+.4f} "
          f"| |g_s|/|g_r|={agg['ratio_style_rec_mean']:.4f}", flush=True)
    return agg


# ---------------------------------------------------------------- P2.3
def p2_3_counterfactual_updates(desc, dataset, device="cuda"):
    print("\n===== P2.3 Counterfactual Single-Step Updates =====", flush=True)
    lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    batch = next(iter(loader))
    x = batch["image"].to(device)
    st = batch["style_id"].to(device)
    ar = batch["artist_id"].to(device)
    ge = batch["genre_id"].to(device)

    ref_bank = StyleReferenceBank(num_styles=136, refs_per_style=4, descriptor=desc).to(device)
    ref_bank.bank_q1.normal_(); ref_bank.bank_q2.normal_(); ref_bank.bank_mom.normal_()
    ref_bank.bank_styles = torch.tensor([s for s in range(136) for _ in range(4)], device=device)
    ref_bank.bank_artists.fill_(1); ref_bank.bank_genres.fill_(1)
    loss_fn = SupervisedStyleRankingLoss()

    # fixed eval set: use same batch for style loss measurement
    def eval_metrics(vae):
        vae.eval()
        with torch.no_grad():
            post = vae.encode(x).latent_dist
            q1, q2, mom = desc(post.mean.float())
            l_s, stats = loss_fn(q1, q2, mom, st, ar, ge, ref_bank)
            l_rec, _, _, _, _ = compute_mixed_reconstruction_loss(vae, x, lpips_fn)
        vae.train()
        return {
            "l_style": round(float(l_s.item()), 4),
            "l_rec": round(float(l_rec.item()), 4),
            "d_pos": round(float(stats["d_pos"]), 4),
            "d_neg": round(float(stats["d_neg"]), 4),
            "rel_margin": round(float((stats["d_neg"] - stats["d_pos"]) /
                                     (stats["d_neg"] + stats["d_pos"] + 1e-8)), 4),
        }

    variants = {}
    for name in ["recon_only", "style_only", "joint", "deconflicted", "scale_norm_style"]:
        vae = load_model("checkpoint_step_001800.pt").to(device)
        vae.train()
        vae.enable_gradient_checkpointing()
        before = eval_metrics(vae)

        vae.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            l_rec, _, _, post, _ = compute_mixed_reconstruction_loss(vae, x, lpips_fn)
            kl_e, _, _, _ = compute_kl_divergence(post)
            q1, q2, mom = desc(post.mean.float())
            l_st, _ = loss_fn(q1, q2, mom, st, ar, ge, ref_bank)

        enc_params = [p for p in vae.encoder.parameters() if p.requires_grad] + \
                     [p for p in vae.quant_conv.parameters() if p.requires_grad]

        def flat(glist, plist):
            return torch.cat([
                g.reshape(-1) if g is not None else torch.zeros_like(p).reshape(-1)
                for g, p in zip(glist, plist)
            ])

        if name == "recon_only":
            (l_rec + BETA_KL * kl_e).backward()
        elif name == "style_only":
            (LAMBDA_STYLE * l_st).backward()
        elif name == "joint":
            (l_rec + BETA_KL * kl_e + LAMBDA_STYLE * l_st).backward()
        elif name == "scale_norm_style":
            gr = torch.autograd.grad(l_rec, enc_params, retain_graph=True, allow_unused=True)
            gs = torch.autograd.grad(l_st * LAMBDA_STYLE, enc_params, retain_graph=True, allow_unused=True)
            ngr = flat(gr, enc_params).norm()
            ngs = flat(gs, enc_params).norm()
            scale = float((ngr / ngs.clamp(min=1e-12)).item())
            (l_rec + BETA_KL * kl_e + (LAMBDA_STYLE * scale) * l_st).backward()
        else:  # deconflicted: remove anti-aligned style component on encoder
            gr = torch.autograd.grad(l_rec, enc_params, retain_graph=True, allow_unused=True)
            gs = torch.autograd.grad(l_st * LAMBDA_STYLE, enc_params, retain_graph=True, allow_unused=True)
            grv, gsv = flat(gr, enc_params), flat(gs, enc_params)
            proj = (gsv @ grv) / (grv @ grv + 1e-12) * grv
            gs_dec = gsv - proj
            # encoder: recon + deconflicted style ; decoder: recon+KL only
            dec_params = [p for p in vae.decoder.parameters() if p.requires_grad] + \
                         [p for p in vae.post_quant_conv.parameters() if p.requires_grad]
            gr_dec = torch.autograd.grad(l_rec + BETA_KL * kl_e, dec_params,
                                         retain_graph=True, allow_unused=True)
            gr_enc = torch.autograd.grad(l_rec + BETA_KL * kl_e, enc_params,
                                         retain_graph=True, allow_unused=True)
            for p in vae.parameters():
                p.grad = None
            off = 0
            for g_r, p in zip(gr_enc, enc_params):
                n = p.numel()
                gd = gs_dec[off:off + n].reshape(p.shape)
                off += n
                p.grad = gd + (g_r if g_r is not None else 0)
            for g_r, p in zip(gr_dec, dec_params):
                p.grad = g_r if g_r is not None else torch.zeros_like(p)

        torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
        opt = torch.optim.AdamW(vae.parameters(), lr=1e-5)
        opt.step()
        after = eval_metrics(vae)
        variants[name] = {"before": before, "after": after,
                          "delta_l_style": round(after["l_style"] - before["l_style"], 4),
                          "delta_l_rec": round(after["l_rec"] - before["l_rec"], 4),
                          "delta_rel_margin": round(after["rel_margin"] - before["rel_margin"], 4)}
        print(f"  {name:18s}: ΔL_style={variants[name]['delta_l_style']:+.4f} "
              f"ΔL_rec={variants[name]['delta_l_rec']:+.4f} "
              f"ΔrelMargin={variants[name]['delta_rel_margin']:+.4f}", flush=True)
        del vae
        torch.cuda.empty_cache()

    return variants


# ---------------------------------------------------------------- P2.4
def p2_4_positive_strategy(desc, z, styles, artists):
    print("\n===== P2.4 Positive Sample Strategy =====", flush=True)
    with torch.no_grad():
        q1, q2, mom = desc(z)
    rng = np.random.default_rng(42)
    N = len(styles)
    d_near2, d_rand_cross, d_far2 = [], [], []
    near_artist_same = 0
    near_artist_total = 0

    idx_pool = rng.choice(N, size=min(200, N), replace=False)
    for i in idx_pool:
        sc = np.where(styles == styles[i])[0]
        sc = sc[sc != i]
        if len(sc) < 4:
            continue
        d = torch.diag(desc.compute_pairwise_distance(
            q1[[i]*len(sc)], q2[[i]*len(sc)], mom[[i]*len(sc)],
            q1[sc], q2[sc], mom[sc])).cpu().numpy()
        order = np.argsort(d)
        d_near2.append(float(d[order[:2]].mean()))
        d_far2.append(float(d[order[-2:]].mean()))
        # nearest 2 artist identity
        for oi in order[:2]:
            near_artist_total += 1
            if artists[sc[oi]] == artists[i]:
                near_artist_same += 1
        # random cross-artist
        ca = sc[artists[sc] != artists[i]] if artists[i] >= 0 else np.array([], dtype=int)
        if len(ca):
            pick = rng.choice(ca, size=min(2, len(ca)), replace=False)
            dd = torch.diag(desc.compute_pairwise_distance(
                q1[[i]*len(pick)], q2[[i]*len(pick)], mom[[i]*len(pick)],
                q1[pick], q2[pick], mom[pick])).cpu().numpy()
            d_rand_cross.append(float(dd.mean()))

    res = {
        "d_near2_mean": round(float(np.mean(d_near2)), 4),
        "d_random_crossartist_mean": round(float(np.mean(d_rand_cross)), 4),
        "d_far2_mean": round(float(np.mean(d_far2)), 4),
        "ratio_random_over_near2": round(float(np.mean(d_rand_cross) / np.mean(d_near2)), 4),
        "nearest2_same_artist_rate": round(near_artist_same / max(near_artist_total, 1), 4),
        "n_anchors_measured": len(d_near2),
    }
    print(f"  d_near2={res['d_near2_mean']:.4f} d_random_cross={res['d_random_crossartist_mean']:.4f} "
          f"d_far2={res['d_far2_mean']:.4f}", flush=True)
    print(f"  ratio random/near2 = {res['ratio_random_over_near2']:.4f}", flush=True)
    print(f"  nearest-2 same-artist rate = {res['nearest2_same_artist_rate']*100:.1f}%", flush=True)
    return res


# ---------------------------------------------------------------- P2.5
def p2_5_negative_balance(desc, z, styles, genres):
    print("\n===== P2.5 Negative Mining Balance =====", flush=True)
    with torch.no_grad():
        q1, q2, mom = desc(z)
    rng = np.random.default_rng(42)
    N = len(styles)
    style_neg_count = Counter()
    hard_style_count = Counter()
    idx_pool = rng.choice(N, size=min(200, N), replace=False)

    for i in idx_pool:
        dc = np.where(styles != styles[i])[0]
        d = torch.diag(desc.compute_pairwise_distance(
            q1[[i]*len(dc)], q2[[i]*len(dc)], mom[[i]*len(dc)],
            q1[dc], q2[dc], mom[dc])).cpu().numpy()
        order = np.argsort(d)[:8]
        for oi in order:
            style_neg_count[int(styles[dc[oi]])] += 1
        if genres[i] >= 0:
            hc = dc[genres[dc] == genres[i]]
            if len(hc):
                hard_style_count[int(styles[hc[0]])] += 1

    top = style_neg_count.most_common(10)
    total = sum(style_neg_count.values())
    top_share = sum(c for _, c in top) / max(total, 1)
    res = {
        "n_negatives_sampled": total,
        "distinct_styles_as_negative": len(style_neg_count),
        "top10_style_share": round(top_share, 4),
        "top10": [{"style_id": s, "count": c} for s, c in top],
    }
    print(f"  Distinct styles appearing as hard negative: {len(style_neg_count)}/136", flush=True)
    print(f"  Top-10 share of all negatives: {top_share*100:.1f}%", flush=True)
    return res


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    desc_cpu, calib = build_desc()          # stays on CPU for analysis
    desc_gpu = build_desc()[0].to(device)   # on GPU for gradient work
    dataset = RawMemmapWikiArtDataset(MANIFEST, is_training=True)

    results = {}

    # P2.1 on A0 latents (CPU-side descriptor work)
    print("\n[P2] Loading A0 for latent extraction...", flush=True)
    a0 = load_model(None).to(device).eval()
    z0, ys, ars, ges = get_fixed_latents(a0, dataset, n_batches=8, device=device)
    z0 = z0.cpu()
    styles0, artists0, genres0 = ys.cpu().numpy(), ars.cpu().numpy(), ges.cpu().numpy()
    results["p2_1_scale_shortcut_A0"] = p2_1_scale_shortcut(desc_cpu, z0, styles0, artists0, genres0)
    (OUT_DIR / "03a_scale_A0.json").write_text(
        json.dumps(results["p2_1_scale_shortcut_A0"], indent=2), encoding="utf-8")
    print("[Saved] 03a_scale_A0.json", flush=True)
    del a0
    torch.cuda.empty_cache()

    # P2.1 / P2.4 / P2.5 on V2-1800 latents
    print("\n[P2] Loading V2-1800 for latent extraction...", flush=True)
    v2 = load_model("checkpoint_step_001800.pt").to(device).eval()
    z1, ys1, ars1, ges1 = get_fixed_latents(v2, dataset, n_batches=8, device=device)
    z1 = z1.cpu()
    s1, a1, g1 = ys1.cpu().numpy(), ars1.cpu().numpy(), ges1.cpu().numpy()
    results["p2_1_scale_shortcut_V2_1800"] = p2_1_scale_shortcut(desc_cpu, z1, s1, a1, g1)
    results["p2_4_positive_strategy_V2"] = p2_4_positive_strategy(desc_cpu, z1, s1, a1)
    results["p2_5_negative_balance_V2"] = p2_5_negative_balance(desc_cpu, z1, s1, g1)
    (OUT_DIR / "03b_scale_V2.json").write_text(
        json.dumps(results["p2_1_scale_shortcut_V2_1800"], indent=2), encoding="utf-8")
    (OUT_DIR / "06_positive_negative_sampling.json").write_text(
        json.dumps({k: v for k, v in results.items() if "p2_4" in k or "p2_5" in k}, indent=2),
        encoding="utf-8")
    print("[Saved] 03b + 06", flush=True)
    del v2
    torch.cuda.empty_cache()

    # P2.2 gradient conflict on V2-1800 (GPU descriptor)
    vae = load_model("checkpoint_step_001800.pt").to(device)
    results["p2_2_gradient_conflict_V2_1800"] = p2_2_gradient_conflict(vae, desc_gpu, dataset, device)
    (OUT_DIR / "04_gradient_conflict.json").write_text(
        json.dumps(results["p2_2_gradient_conflict_V2_1800"], indent=2), encoding="utf-8")
    print("[Saved] 04_gradient_conflict.json", flush=True)
    del vae
    torch.cuda.empty_cache()

    # P2.3 counterfactual updates (GPU descriptor)
    results["p2_3_counterfactual"] = p2_3_counterfactual_updates(desc_gpu, dataset, device)
    (OUT_DIR / "05_counterfactual_updates.json").write_text(
        json.dumps(results["p2_3_counterfactual"], indent=2), encoding="utf-8")
    print("[Saved] 05_counterfactual_updates.json", flush=True)

    (OUT_DIR / "03_distance_scale_audit.json").write_text(
        json.dumps({k: v for k, v in results.items() if "p2_1" in k}, indent=2), encoding="utf-8")
    (OUT_DIR / "04_gradient_conflict.json").write_text(
        json.dumps(results["p2_2_gradient_conflict_V2_1800"], indent=2), encoding="utf-8")
    (OUT_DIR / "05_counterfactual_updates.json").write_text(
        json.dumps(results["p2_3_counterfactual"], indent=2), encoding="utf-8")
    (OUT_DIR / "06_positive_negative_sampling.json").write_text(
        json.dumps({k: v for k, v in results.items() if "p2_4" in k or "p2_5" in k}, indent=2),
        encoding="utf-8")

    print(f"\n[Saved] P2 outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
