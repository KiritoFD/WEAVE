"""Style-Rank 20-class geometry evaluation.

Builds a 20-class held-out eval set from the test split (excluding the frozen
2048-image val set), then evaluates any VAE checkpoint with

    u = normalize(P (vec(posterior.mean) - mu_z))

directly (no readout head), reporting:
  Content-Matched AUROC (primary), Cross-Artist AUROC, Same-Genre Hard AUROC,
  kNN@1/5/10, mAP@10, reconstruction (LPIPS/SSIM/PSNR/KL/latent RMS),
  and style-coordinate swap edit test.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/root/Workspace/WEAVE_VAE")

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
from tools.vae_suite.losses import compute_kl_divergence
import lpips

OUT = Path("/root/Workspace/WEAVE_VAE/runs/vae_style_rank20")
MANIFEST = Path("/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")
A0_DIR = "/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae"
DEV = "cuda"
DINO_MEAN = [0.485, 0.456, 0.406]
DINO_STD = [0.229, 0.224, 0.225]


class MemmapEvalSet:
    """Reads arbitrary raw_index entries directly from the memmap binary."""

    def __init__(self, manifest_path, raw_indices, image_size=512):
        man = json.loads(Path(manifest_path).read_text())
        self.mmap = np.memmap(man["bin_path"], dtype=np.uint8, mode="r",
                              shape=tuple(man["shape"]))
        self.raw = list(raw_indices)
        self.records = {r["raw_index"]: r for r in man["records"]}

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, i):
        ri = self.raw[i]
        arr = self.mmap[ri]
        t = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).float()
        t = t.div_(127.5).sub_(1.0)
        return {"image": t, "raw_index": ri}


def build_eval_set(max_per_style=120):
    """20-class TEST-split images, excluding the frozen 2048 val set."""
    man = json.loads(MANIFEST.read_text())
    val_idx = set(man["val_indices"])
    keep = set(c["id"] for c in json.loads((OUT / "classes.json").read_text()))

    by_style = defaultdict(list)
    for r in man["records"]:
        if r["in_train"]:
            continue
        if r["raw_index"] in val_idx:
            continue
        if r["style_id"] not in keep:
            continue
        by_style[r["style_id"]].append(r["raw_index"])

    rng = np.random.default_rng(7)
    picks = []
    for s, lst in by_style.items():
        n = min(max_per_style, len(lst))
        picks += [int(x) for x in rng.choice(lst, n, replace=False)]
    picks = sorted(set(picks))
    print(f"[EvalSet] {len(picks)} images over {len(by_style)} styles "
          f"(per-style cap {max_per_style}, from test split excl. frozen 2048)", flush=True)
    ds = MemmapEvalSet(MANIFEST, picks)
    return ds, list(range(len(picks)))


def psnr(x, y):
    mse = F.mse_loss((x.clamp(-1, 1) + 1) / 2, (y.clamp(-1, 1) + 1) / 2).item()
    return 50.0 if mse <= 1e-10 else float(10 * np.log10(1.0 / mse))


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


def build_cm(dino, y, a, g, cap=1200, seed=42, tol=0.05):
    rng = np.random.default_rng(seed)
    pos, neg, gaps = [], [], []
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
            nc = np.where(y != y[i])[0]
        if len(nc) < 2:
            continue
        if len(pc) > 16: pc = rng.choice(pc, 16, replace=False)
        if len(nc) > 64: nc = rng.choice(nc, 64, replace=False)
        sp = dino[pc] @ dino[i]; sn = dino[nc] @ dino[i]
        gap = np.abs(sp[:, None] - sn[None, :])
        pi, ni = np.unravel_index(np.argmin(gap), gap.shape)
        pos.append((int(i), int(pc[pi]))); neg.append((int(i), int(nc[ni])))
        gaps.append(float(gap[pi, ni]))
    return pos, neg, np.array(gaps)


def auroc(feats, pos, neg, n_boot=300, seed=42, cap=1000):
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
    idx = rng.permutation(N); q, gg = idx[:n_q], idx[n_q:]
    gs = st[gg]; aps = []
    for qi in q:
        sims = feats[gg] @ feats[qi]
        top = np.argsort(-sims)[:10]
        rel = (gs[top] == st[qi]).astype(float)
        if rel.sum() == 0:
            aps.append(0.0); continue
        prec = np.cumsum(rel) / (np.arange(len(rel)) + 1)
        aps.append(float((prec * rel).sum() / rel.sum()))
    return knn, round(float(np.mean(aps) * 100), 2)


def to_pil(t):
    x = (t.detach().clamp(-1, 1) + 1) / 2
    return Image.fromarray((x.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="")   # comma sep: name=path
    ap.add_argument("--max-per-style", type=int, default=120)
    args = ap.parse_args()

    t0 = time.time()
    proj = torch.load(OUT / "projection.pt", weights_only=False)
    P = proj["P"].to(DEV).float(); mu_z = proj["mu_z"].to(DEV).float()
    classes = json.loads((OUT / "classes.json").read_text())
    name2new = {c["name"]: i for i, c in enumerate(classes)}
    id2new = {c["id"]: i for i, c in enumerate(classes)}

    ds, picks = build_eval_set(args.max_per_style)
    y, a, g = [], [], []
    for p in picks:
        r = ds.records[ds.raw[p]]
        y.append(id2new[r["style_id"]]); a.append(r.get("artist_id", -1)); g.append(r.get("genre_id", -1))
    y = np.array(y); a = np.array(a); g = np.array(g)
    print(f"[Meta] {len(y)} samples | styles present={len(np.unique(y))} | "
          f"genres={len(np.unique(g[g>=0]))}", flush=True)

    # DINO for content matching on eval set
    fdin = OUT / "eval_dino.pt"
    if fdin.exists():
        dino = torch.load(fdin, weights_only=False).numpy()
        print("[DINO] eval cache found", flush=True)
    else:
        dm = AutoModel.from_pretrained("facebook/dinov2-base").to(DEV).eval()
        nm = torch.tensor(DINO_MEAN, device=DEV).view(1, 3, 1, 1)
        ns = torch.tensor(DINO_STD, device=DEV).view(1, 3, 1, 1)
        df = []
        with torch.no_grad():
            for i in range(0, len(picks), 16):
                x = torch.stack([ds[p]["image"] for p in picks[i:i + 16]]).to(DEV)
                x01 = (x.clamp(-1, 1) + 1) / 2
                x224 = F.interpolate(x01, (224, 224), mode="bicubic", align_corners=False)
                o = dm(pixel_values=(x224 - nm) / ns)
                df.append(F.normalize(o.last_hidden_state[:, 0].float(), dim=-1).cpu())
        dino = torch.cat(df).numpy()
        del dm; torch.cuda.empty_cache()
        torch.save(torch.from_numpy(dino), fdin)
    dino = dino / (np.linalg.norm(dino, axis=1, keepdims=True) + 1e-8)

    same, cross, diff, hard = build_pairs(y, a, g)
    cm_pos, cm_neg, cm_gap = build_cm(dino, y, a, g)
    print(f"[Pairs] same={len(same)} cross={len(cross)} hard={len(hard)} "
          f"cm={len(cm_pos)} gap_mean={cm_gap.mean():.4f} "
          f"frac<0.05={float((cm_gap<0.05).mean()):.2f}", flush=True)

    lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(DEV).eval()
    for p in lpips_fn.parameters():
        p.requires_grad = False

    def eval_one(tag, path):
        pth = Path(path)
        m_c = None
        if pth.is_file() and pth.suffix == ".pt":
            ck = torch.load(pth, map_location="cpu", weights_only=False)
            vae = AutoencoderKL.from_pretrained(A0_DIR).to(DEV).eval()
            state = ck.get("ema", ck.get("model", ck))
            vae.load_state_dict(state)
            if "m_center" in ck:
                m_c = ck["m_center"].to(DEV).float()
                print(f"[{tag}] loaded m_center from checkpoint: norm={m_c.norm().item():.2f}", flush=True)
        else:
            vae = AutoencoderKL.from_pretrained(path).to(DEV).eval()
            cf = pth / "m_center.pt"
            if cf.exists():
                m_c = torch.load(cf, weights_only=False).to(DEV).float()
                print(f"[{tag}] loaded m_center from {cf}: norm={m_c.norm().item():.2f}", flush=True)

        if m_c is None:
            m_c = F.linear(mu_z, P).squeeze(0).to(DEV).float()
            print(f"[{tag}] using default A0 center: norm={m_c.norm().item():.2f}", flush=True)

        for p in vae.parameters():
            p.requires_grad = False
        Z = []
        lp, ss, ps, kls = [], [], [], []
        recs = {}
        with torch.no_grad():
            for i in range(0, len(picks), 16):
                idx = picks[i:i + 16]
                x = torch.stack([ds[p]["image"] for p in idx]).to(DEV)
                post = vae.encode(x).latent_dist
                mu = post.mean.float()
                Z.append(mu.flatten(1).cpu())
                xh = vae.decode(post.mean).sample
                lp.append(lpips_fn(xh.float(), x.float()).mean().item())
                ss.append(ssim_simple(xh, x))
                ps.append(psnr(xh, x))
                _, kl_v, _, _ = compute_kl_divergence(post)
                kls.append(kl_v)
                if tag == "A0" and i == 0:
                    recs["sample"] = (x[0], xh[0])
        Zt = torch.cat(Z)
        rms = float(Zt.pow(2).mean().sqrt())
        h = F.linear(Zt.to(DEV), P)
        c_norm = float(m_c.norm().item())
        h_norm = float(h.norm(dim=-1).mean().item())
        c_ratio = round(c_norm / max(h_norm, 1e-12), 4)

        u = F.normalize(h - m_c.unsqueeze(0), dim=-1).cpu().numpy()

        r_cm = auroc(u, cm_pos, cm_neg)
        r_cross = auroc(u, cross, diff)
        r_hard = auroc(u, cross, hard)
        knn, mp = knn_map(u, y)

        # ---- edit test: h swap ----
        rng = np.random.default_rng(11)
        by_style = defaultdict(list)
        for i in range(len(y)):
            by_style[int(y[i])].append(i)
        usable = [s for s, v in by_style.items() if len(v) >= 4]
        alphas = [0.25, 0.5, 1.0]
        swap = {al: [] for al in alphas}; rand = {al: [] for al in alphas}
        with torch.no_grad():
            for _ in range(20):
                if len(usable) < 2:
                    break
                s = int(rng.choice(usable)); t = int(rng.choice([x for x in usable if x != s]))
                si = int(rng.choice(by_style[s])); ti = int(rng.choice(by_style[t]))
                xa = ds[picks[si]]["image"].unsqueeze(0).to(DEV)
                xb = ds[picks[ti]]["image"].unsqueeze(0).to(DEV)
                za = vae.encode(xa).latent_dist.mean.flatten(1)
                zb = vae.encode(xb).latent_dist.mean.flatten(1)
                rec0 = vae.decode(za.view(1, 4, 64, 64)).sample
                ha = F.linear(za, P) - m_c
                hb = F.linear(zb, P) - m_c
                d = (hb - ha) @ P
                # proxy target: use target image itself as style reference via LPIPS-to-target reduction
                for al in alphas:
                    zs = (za + al * d).view(1, 4, 64, 64)
                    xs = vae.decode(zs).sample
                    swap[al].append({
                        "lpips_vs_noop": float(lpips_fn(xs.float(), rec0.float()).mean()),
                        "lpips_to_target": float(lpips_fn(xs.float(), xb.float()).mean()),
                        "base_to_target": float(lpips_fn(rec0.float(), xb.float()).mean()),
                    })
                    rd = torch.randn_like(d); rd = rd / (rd.norm() + 1e-12) * d.norm()
                    zr = (za + al * rd).view(1, 4, 64, 64)
                    xr = vae.decode(zr).sample
                    rand[al].append({
                        "lpips_vs_noop": float(lpips_fn(xr.float(), rec0.float()).mean()),
                        "lpips_to_target": float(lpips_fn(xr.float(), xb.float()).mean()),
                        "base_to_target": float(lpips_fn(rec0.float(), xb.float()).mean()),
                    })
        swap_res = {}
        for al in alphas:
            sw = swap[al]; rn = rand[al]
            if not sw:
                continue
            swap_res[str(al)] = {
                "swap_delta_to_target": round(float(np.mean([r["base_to_target"] - r["lpips_to_target"] for r in sw])), 4),
                "rand_delta_to_target": round(float(np.mean([r["base_to_target"] - r["lpips_to_target"] for r in rn])), 4),
                "swap_lpips_vs_noop": round(float(np.mean([r["lpips_vs_noop"] for r in sw])), 4),
                "rand_lpips_vs_noop": round(float(np.mean([r["lpips_vs_noop"] for r in rn])), 4),
            }

        res = {
            "n": int(len(y)),
            "latent_rms": round(rms, 4),
            "content_matched": r_cm, "cross_artist": r_cross, "same_genre_hard": r_hard,
            "knn": knn, "map10": mp,
            "reconstruction": {"LPIPS": round(float(np.mean(lp)), 4),
                               "SSIM": round(float(np.mean(ss)), 4),
                               "PSNR": round(float(np.mean(ps)), 2),
                               "KL": round(float(np.mean(kls)), 2)},
            "edit": swap_res,
        }
        print(f"[{tag}] CM={r_cm['auroc'] if r_cm else None} "
              f"[{r_cm['lo'] if r_cm else ''},{r_cm['hi'] if r_cm else ''}] "
              f"Cross={r_cross['auroc'] if r_cross else None} "
              f"Hard={r_hard['auroc'] if r_hard else None} kNN1={knn['knn1']} mAP={mp} "
              f"LPIPS={res['reconstruction']['LPIPS']} SSIM={res['reconstruction']['SSIM']} "
              f"RMS={rms:.3f}", flush=True)
        if swap_res:
            k0 = "0.5"
            print(f"      edit@0.5: swap dTarget={swap_res[k0]['swap_delta_to_target']:+.4f} "
                  f"rand={swap_res[k0]['rand_delta_to_target']:+.4f} "
                  f"| LPIPS swap={swap_res[k0]['swap_lpips_vs_noop']:.4f} "
                  f"rand={swap_res[k0]['rand_lpips_vs_noop']:.4f}", flush=True)
        del vae
        torch.cuda.empty_cache()
        return res

    models = [("A0", A0_DIR)]
    if args.models:
        for kv in args.models.split(","):
            if "=" in kv:
                nm, pth = kv.split("=", 1)
                models.append((nm, pth))
    fin = OUT / "final_diffusers_vae"
    if fin.exists() and not any(n == "FINAL" for n, _ in models):
        models.append(("FINAL", str(fin)))

    results = {}
    for nm, pth in models:
        print(f"\n--- {nm} ---", flush=True)
        results[nm] = eval_one(nm, pth)

    results["_meta"] = {"pairs": {"cross": len(cross), "hard": len(hard), "cm": len(cm_pos)},
                        "cm_gap_mean": round(float(cm_gap.mean()), 4),
                        "cm_frac_lt_005": round(float((cm_gap < 0.05).mean()), 4),
                        "n_classes": len(classes), "elapsed_sec": round(time.time() - t0, 1)}
    (OUT / "validation_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n" + "=" * 96)
    print("20-CLASS GEOMETRY ACCEPTANCE")
    print("=" * 96)
    print(f"{'model':<14}{'CM AUROC':>10}{'95% CI':>16}{'CrossArt':>10}{'Hard':>8}{'kNN@1':>8}"
          f"{'mAP':>7}{'LPIPS':>8}{'SSIM':>8}")
    print("-" * 96)
    for k, v in results.items():
        if k.startswith("_"):
            continue
        cm = v["content_matched"]
        ci = "[{},{}]".format(cm["lo"], cm["hi"]) if cm else ""
        print(f"{k:<14}{cm['auroc']:>10.2f}{ci:>16}{v['cross_artist']['auroc']:>10.2f}"
              f"{v['same_genre_hard']['auroc']:>8.2f}{v['knn']['knn1']:>8.2f}{v['map10']:>7.2f}"
              f"{v['reconstruction']['LPIPS']:>8.4f}{v['reconstruction']['SSIM']:>8.4f}")
    print("=" * 96)
    print(f"\n[Saved] {OUT/'validation_metrics.json'}  ({results['_meta']['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
