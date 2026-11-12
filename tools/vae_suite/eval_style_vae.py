"""Style-VAE final acceptance evaluation.

Checks (per protocol):
  1. S1 criterion : DIRECT Content-Matched AUROC using u = P(z - mu_z), no readout head.
                    target >= 56.5%, baselines: A0 raw 51.65 / SmallCNN 53.39 / CSD 59.62
  2. Cross-Artist AUROC : must not regress (A0 raw 54.67, SmallCNN 61.40, CSD 74.27)
  3. Reconstruction : LPIPS <= 0.085, SSIM >= 0.765, PSNR reported
  4. Latent RMS / KL : no abnormal shrink/explosion
  5. Style swap effectiveness : z_swap = z_a + a*P^T(u_b - u_a); must beat random direction
  6. Visual panels saved
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/root/Workspace/WEAVE_VAE")
sys.path.insert(0, "/root/Workspace/WEAVE_VAE/tools/12h")
sys.path.insert(0, "/tmp/aladin_repo")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL
from PIL import Image
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from transformers import AutoModel

from tools.vae_suite.dataset import RawMemmapWikiArtDataset
import csd_model
import lpips

OUT = Path("/root/Workspace/WEAVE_VAE/runs_vae/style_vae")
IMG = OUT / "swap_images"
IMG.mkdir(parents=True, exist_ok=True)
MANIFEST = Path("/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")
A0_DIR = "/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae"
ALADIN_CKPT = "/tmp/aladin_repo/aladinVIT_bamfg_64.47.pt"
NORM_MEAN = [0.48145466, 0.4578275, 0.40821073]
NORM_STD = [0.26862954, 0.26130258, 0.27577711]
DINO_MEAN = [0.485, 0.456, 0.406]
DINO_STD = [0.229, 0.224, 0.225]


def csd_feat(csd, img, nm, ns):
    x01 = (img.clamp(-1, 1) + 1.0) / 2.0
    x224 = F.interpolate(x01, (224, 224), mode="bicubic", align_corners=False)
    _, _, s = csd((x224 - nm) / ns)
    return F.normalize(s.float(), dim=-1)


def load_aladin(device):
    from vit_networks import VisionTransformer

    class AViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc_style = VisionTransformer()

        def encode(self, image):
            p, _ = self.enc_style(image, just_style_code=True)
            return p

    ck = torch.load(ALADIN_CKPT, map_location="cpu", weights_only=False)
    m = AViT(); m.load_state_dict(ck); m.eval().to(device)
    for p in m.parameters():
        p.requires_grad = False
    return m


@torch.no_grad()
def aladin_feat(m, img, device):
    x = F.interpolate(img.clamp(-1, 1), (224, 224), mode="bicubic", align_corners=False)
    return F.normalize(m.encode(x.float()), dim=-1)


def to_pil(t):
    x = (t.detach().clamp(-1, 1) + 1) / 2
    return Image.fromarray((x.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))


def psnr(x, y):
    mse = F.mse_loss((x.clamp(-1, 1) + 1) / 2, (y.clamp(-1, 1) + 1) / 2).item()
    return 50.0 if mse <= 1e-10 else float(10 * np.log10(1.0 / mse))


def ssim_simple(x, y):
    x01 = (x.clamp(-1, 1) + 1) / 2; y01 = (y.clamp(-1, 1) + 1) / 2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mx = F.avg_pool2d(x01, 11, 1, 5); my = F.avg_pool2d(y01, 11, 1, 5)
    mxx = mx * mx; myy = my * my; mxy = mx * my
    sx = F.avg_pool2d(x01 * x01, 11, 1, 5) - mxx
    sy = F.avg_pool2d(y01 * y01, 11, 1, 5) - myy
    sxy = F.avg_pool2d(x01 * y01, 11, 1, 5) - mxy
    m = ((2 * mxy + C1) * (2 * sxy + C2)) / ((mxx + myy + C1) * (sx + sy + C2))
    return float(m.mean())


def build_pairs(y, a, g, seed=42):
    rng = np.random.default_rng(seed)
    N = len(y)
    same, cross, diff, hard = [], [], [], []
    for i in range(N):
        sc = np.where(y == y[i])[0]; sc = sc[sc != i]
        if len(sc):
            same.append((i, int(rng.choice(sc))))
            if a[i] >= 0:
                ca = sc[a[sc] != a[i]]
                if len(ca):
                    cross.append((i, int(rng.choice(ca))))
        dc = np.where(y != y[i])[0]
        if len(dc):
            diff.append((i, int(rng.choice(dc))))
            if g[i] >= 0:
                hc = dc[g[dc] == g[i]]
                if len(hc):
                    hard.append((i, int(rng.choice(hc))))
    return same, cross, diff, hard


def build_cm(dino, y, a, g, cap=1000, seed=42):
    rng = np.random.default_rng(seed)
    pos, neg = [], []
    for i in rng.permutation(len(y)):
        if len(pos) >= cap:
            break
        pc = np.where(y == y[i])[0]; pc = pc[pc != i]
        if a[i] >= 0:
            pc = pc[a[pc] != a[i]]
        if len(pc) == 0:
            continue
        nc = np.where((y != y[i]) & (g == g[i]))[0]
        if len(nc) < 2:
            continue
        if len(pc) > 12: pc = rng.choice(pc, 12, replace=False)
        if len(nc) > 24: nc = rng.choice(nc, 24, replace=False)
        sp = dino[pc] @ dino[i]; sn = dino[nc] @ dino[i]
        gap = np.abs(sp[:, None] - sn[None, :])
        pi, ni = np.unravel_index(np.argmin(gap), gap.shape)
        pos.append((int(i), int(pc[pi]))); neg.append((int(i), int(nc[ni])))
    return pos, neg


def auroc(feats, pos, neg, n_boot=200, seed=42, cap=800):
    n = min(len(pos), len(neg))
    if n < 10:
        return None
    n = min(n, cap)
    sp = np.array(pos[:n]); sn = np.array(neg[:n])
    a1 = (feats[sp[:, 0]] * feats[sp[:, 1]]).sum(1)
    a0 = (feats[sn[:, 0]] * feats[sn[:, 1]]).sum(1)
    lbs = np.concatenate([np.ones(n), np.zeros(n)])
    try:
        auc = float(roc_auc_score(lbs, np.concatenate([a1, a0])) * 100)
    except ValueError:
        return None
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        pi = rng.integers(0, n, n); ni = rng.integers(0, n, n)
        try:
            boot.append(roc_auc_score(lbs, np.concatenate([a1[pi], a0[ni]])) * 100)
        except ValueError:
            pass
    lo, hi = np.percentile(boot, [2.5, 97.5]) if boot else (auc, auc)
    return {"auroc": round(auc, 2), "lo": round(float(lo), 2), "hi": round(float(hi), 2), "n": int(n)}


def knn_map(feats, styles, ks=(1, 5, 10), n_q=200, seed=42):
    N = len(feats)
    sim = feats @ feats.T
    np.fill_diagonal(sim, -np.inf)
    order = np.argsort(-sim, axis=1)
    st = np.asarray(styles)
    knn = {f"knn{k}": round(float((st[order[:, :k]] == st[:, None]).mean()) * 100, 2) for k in ks}
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N); q, g = idx[:n_q], idx[n_q:]
    gs = st[g]; aps = []
    for qi in q:
        sims = feats[g] @ feats[qi]
        top = np.argsort(-sims)[:10]
        rel = (gs[top] == st[qi]).astype(float)
        if rel.sum() == 0:
            aps.append(0.0); continue
        prec = np.cumsum(rel) / (np.arange(len(rel)) + 1)
        aps.append(float((prec * rel).sum() / rel.sum()))
    return knn, round(float(np.mean(aps) * 100), 2)


def main():
    device = "cuda"
    t0 = time.time()
    print("=== Style-VAE final acceptance evaluation ===", flush=True)

    sub = torch.load(OUT / "style_subspace.pt", weights_only=False)
    P = sub["P"].to(device).float()
    mu_z = sub["mu_z"].to(device).float()
    t_bar = sub["t_bar"].to(device).float()
    K = sub["k"]

    ds = RawMemmapWikiArtDataset(MANIFEST, is_training=False)
    y, a, g = [], [], []
    for i in range(len(ds)):
        rec = ds.records_by_idx[ds.indices[i]]
        y.append(rec["style_id"]); a.append(rec.get("artist_id", -1)); g.append(rec.get("genre_id", -1))
    y = np.array(y); a = np.array(a); g = np.array(g)

    csd, _, _ = csd_model.load_csd(device)
    csd = csd.float().eval()
    for p in csd.parameters():
        p.requires_grad = False
    nm = torch.tensor(NORM_MEAN, device=device).view(1, 3, 1, 1)
    ns = torch.tensor(NORM_STD, device=device).view(1, 3, 1, 1)
    aladin = load_aladin(device)
    lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    for p in lpips_fn.parameters():
        p.requires_grad = False

    # ---- DINO for content matching ----
    fdin = OUT / "val_dino.pt"
    if fdin.exists():
        dino = torch.load(fdin, weights_only=False).numpy()
    else:
        dm = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()
        nmd = torch.tensor(DINO_MEAN, device=device).view(1, 3, 1, 1)
        nsd = torch.tensor(DINO_STD, device=device).view(1, 3, 1, 1)
        dl = DataLoader(ds, batch_size=8, shuffle=False)
        df = []
        with torch.no_grad():
            for i, b in enumerate(dl):
                x = b["image"].to(device)
                x01 = (x.clamp(-1, 1) + 1) / 2
                x224 = F.interpolate(x01, (224, 224), mode="bicubic", align_corners=False)
                o = dm(pixel_values=(x224 - nmd) / nsd)
                df.append(F.normalize(o.last_hidden_state[:, 0].float(), dim=-1).cpu())
        dino = torch.cat(df).numpy()
        del dm; torch.cuda.empty_cache()
        torch.save(torch.from_numpy(dino), fdin)
    dino = dino / (np.linalg.norm(dino, axis=1, keepdims=True) + 1e-8)

    same, cross, diff, hard = build_pairs(y, a, g)
    cm_pos, cm_neg = build_cm(dino, y, a, g)
    print(f"[Pairs] cross={len(cross)} hard={len(hard)} cm={len(cm_pos)}", flush=True)

    results = {"pairs": {"cross": len(cross), "hard": len(hard), "cm": len(cm_pos)}}

    # ================= evaluate a model =================
    def eval_model(tag, vae_path, use_subspace=True):
        vae = AutoencoderKL.from_pretrained(vae_path).to(device).eval()
        for p in vae.parameters():
            p.requires_grad = False

        # ---- latents ----
        Z = []
        dl = DataLoader(ds, batch_size=16, shuffle=False)
        with torch.no_grad():
            for i, b in enumerate(dl):
                post = vae.encode(b["image"].to(device)).latent_dist
                Z.append(post.mean.float().flatten(1).cpu())
                if (i + 1) % 40 == 0:
                    print(f"  [{tag}] latents {i+1}/{len(dl)}", flush=True)
        Z = torch.cat(Z)
        rms = float(Z.pow(2).mean().sqrt())
        ch_std = [round(float(Z.view(-1, 4, 4096)[:, c].std()), 4) for c in range(4)]

        if use_subspace:
            u = (Z.to(device) - mu_z) @ P.t()
        else:
            u = Z.to(device)[:, :K]
        un = F.normalize(u, dim=-1).cpu().numpy()

        r_cm = auroc(un, cm_pos, cm_neg)
        r_cross = auroc(un, cross, diff)
        r_hard = auroc(un, cross, hard)
        knn, mp = knn_map(un, y)
        print(f"  [{tag}] CM={r_cm['auroc']} Cross={r_cross['auroc']} Hard={r_hard['auroc']} "
              f"kNN1={knn['knn1']} mAP={mp} RMS={rms:.3f}", flush=True)

        # ---- reconstruction ----
        lp, ss, ps, kls = [], [], [], []
        with torch.no_grad():
            for i, b in enumerate(dl):
                x = b["image"].to(device)
                post = vae.encode(x).latent_dist
                xh = vae.decode(post.mean).sample
                lp.append(lpips_fn(xh.float(), x.float()).mean().item())
                ss.append(ssim_simple(xh, x))
                ps.append(psnr(xh, x))
                kls.append(post.kl().mean().item())
        recon = {"LPIPS": round(float(np.mean(lp)), 4), "SSIM": round(float(np.mean(ss)), 4),
                 "PSNR": round(float(np.mean(ps)), 2), "KL_per_img": round(float(np.mean(kls)), 2)}
        print(f"  [{tag}] recon: {recon}", flush=True)

        # ---- style swap ----
        by_style = defaultdict(list)
        for i in range(len(y)):
            by_style[int(y[i])].append(i)
        usable = [s for s, v in by_style.items() if len(v) >= 10]
        rng = np.random.default_rng(11)
        alphas = [0.25, 0.5, 1.0]
        swap_rows = {al: [] for al in alphas}
        rand_rows = {al: [] for al in alphas}
        n_tasks = 24
        tasks = []
        while len(tasks) < n_tasks:
            s = int(rng.choice(usable)); t = int(rng.choice([x for x in usable if x != s]))
            src = int(rng.choice(by_style[s])); tref = int(rng.choice(by_style[t]))
            tasks.append((s, t, src, tref))

        with torch.no_grad():
            for ti, (s, t, src, tref) in enumerate(tasks):
                x = ds[src]["image"].unsqueeze(0).to(device)
                xt = ds[tref]["image"].unsqueeze(0).to(device)
                z0 = vae.encode(x).latent_dist.mean.flatten(1)
                zt = vae.encode(xt).latent_dist.mean.flatten(1)
                rec0 = vae.decode(z0.view(1, 4, 64, 64)).sample
                f_tgt = csd_feat(csd, xt, nm, ns)
                al_tgt = aladin_feat(aladin, xt, device)
                f0 = csd_feat(csd, rec0, nm, ns)
                a0 = aladin_feat(aladin, rec0, device)
                base_c = float((f0 * f_tgt).sum()); base_a = float((a0 * al_tgt).sum())

                if use_subspace:
                    u0 = (z0 - mu_z) @ P.t(); ut = (zt - mu_z) @ P.t()
                    delta = (ut - u0) @ P
                else:
                    delta = zt - z0

                for al in alphas:
                    zs = (z0 + al * delta).view(1, 4, 64, 64)
                    xs = vae.decode(zs).sample
                    fs = csd_feat(csd, xs, nm, ns); as_ = aladin_feat(aladin, xs, device)
                    swap_rows[al].append({
                        "dt_csd": float((fs * f_tgt).sum()) - base_c,
                        "dt_aladin": float((as_ * al_tgt).sum()) - base_a,
                        "lpips": float(lpips_fn(xs.float(), rec0.float()).mean()),
                    })
                    rd = torch.randn_like(delta); rd = rd / (rd.norm() + 1e-12) * delta.norm()
                    zr = (z0 + al * rd).view(1, 4, 64, 64)
                    xr = vae.decode(zr).sample
                    fr = csd_feat(csd, xr, nm, ns); ar = aladin_feat(aladin, xr, device)
                    rand_rows[al].append({
                        "dt_csd": float((fr * f_tgt).sum()) - base_c,
                        "dt_aladin": float((ar * al_tgt).sum()) - base_a,
                        "lpips": float(lpips_fn(xr.float(), rec0.float()).mean()),
                    })

                if ti < 8:
                    panels = [x[0], rec0[0]]
                    for al in (0.5, 1.0):
                        zs = (z0 + al * delta).view(1, 4, 64, 64)
                        panels.append(vae.decode(zs).sample[0])
                    panels.append(xt[0])
                    W = H = 320
                    cv = Image.new("RGB", (W * len(panels), H))
                    for pi, pp in enumerate(panels):
                        cv.paste(to_pil(pp).resize((W, H)), (pi * W, 0))
                    cv.save(IMG / f"{tag}_task{ti:02d}_s{s}_t{t}.jpg", quality=92)

        swap = {}
        for al in alphas:
            rr = swap_rows[al]; rn = rand_rows[al]
            swap[str(al)] = {
                "swap_dt_csd": round(float(np.mean([r["dt_csd"] for r in rr])), 5),
                "swap_pct_pos": round(float(np.mean([r["dt_csd"] > 0 for r in rr]) * 100), 1),
                "rand_dt_csd": round(float(np.mean([r["dt_csd"] for r in rn])), 5),
                "rand_pct_pos": round(float(np.mean([r["dt_csd"] > 0 for r in rn]) * 100), 1),
                "swap_dt_aladin": round(float(np.mean([r["dt_aladin"] for r in rr])), 5),
                "rand_dt_aladin": round(float(np.mean([r["dt_aladin"] for r in rn])), 5),
                "swap_lpips": round(float(np.mean([r["lpips"] for r in rr])), 4),
                "rand_lpips": round(float(np.mean([r["lpips"] for r in rn])), 4),
            }
            print(f"  [{tag}] alpha={al}: swap dT={swap[str(al)]['swap_dt_csd']:+.4f} "
                  f"({swap[str(al)]['swap_pct_pos']}%pos) vs rand "
                  f"{swap[str(al)]['rand_dt_csd']:+.4f} | LPIPS={swap[str(al)]['swap_lpips']:.4f}",
                  flush=True)

        del vae
        torch.cuda.empty_cache()
        return {
            "latent": {"rms": round(rms, 4), "channel_std": ch_std},
            "content_matched": r_cm, "cross_artist": r_cross, "same_genre_hard": r_hard,
            "knn": knn, "map10": mp, "reconstruction": recon, "swap": swap,
        }

    print("\n--- A0 baseline (subspace projection) ---", flush=True)
    results["A0_subspace"] = eval_model("A0", A0_DIR, use_subspace=True)
    print("\n--- A0 baseline (raw first 768) ---", flush=True)
    results["A0_raw768"] = eval_model("A0raw", A0_DIR, use_subspace=False)

    final = OUT / "final_diffusers_vae"
    if final.exists():
        print("\n--- Style-VAE final ---", flush=True)
        results["StyleVAE"] = eval_model("StyleVAE", str(final), use_subspace=True)
    for ck in sorted(OUT.glob("ckpt_*.pt")):
        print(f"\n--- {ck.stem} ---", flush=True)
        tmp = OUT / f"_tmp_{ck.stem}"
        tmp.mkdir(exist_ok=True)
        d = torch.load(ck, map_location="cpu", weights_only=False)
        v = AutoencoderKL.from_pretrained(A0_DIR)
        v.load_state_dict(d["ema"] if "ema" in d else d["model"])
        v.save_pretrained(str(tmp))
        results[ck.stem] = eval_model(ck.stem, str(tmp), use_subspace=True)

    results["elapsed_sec"] = round(time.time() - t0, 1)
    (OUT / "eval_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n" + "=" * 104)
    print("ACCEPTANCE TABLE")
    print("=" * 104)
    print(f"{'model':<20}{'CM AUROC':>10}{'CrossArt':>10}{'Hard':>8}{'kNN@1':>8}{'LPIPS':>8}{'SSIM':>8}"
          f"{'swap@0.5':>10}{'rand@0.5':>10}")
    print("-" * 104)
    for k, v in results.items():
        if not isinstance(v, dict) or "content_matched" not in v:
            continue
        print(f"{k:<20}{v['content_matched']['auroc']:>10.2f}{v['cross_artist']['auroc']:>10.2f}"
              f"{v['same_genre_hard']['auroc']:>8.2f}{v['knn']['knn1']:>8.2f}"
              f"{v['reconstruction']['LPIPS']:>8.4f}{v['reconstruction']['SSIM']:>8.4f}"
              f"{v['swap']['0.5']['swap_dt_csd']:>10.4f}{v['swap']['0.5']['rand_dt_csd']:>10.4f}")
    print("=" * 104)
    print("targets: CM >= 56.5 | CrossArt >= 61.40 | LPIPS <= 0.085 | SSIM >= 0.765 | swap >> rand")
    print(f"\n[Saved] {OUT/'eval_metrics.json'}  ({results['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
