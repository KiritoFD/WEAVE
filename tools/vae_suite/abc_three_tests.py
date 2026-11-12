"""Three decisive tests: A) style readability probes, B) style direction consistency,
C) real editing ability with directions estimated from data (NOT CSD image gradient).

No VAE retraining. Frozen B0 / A0 encoders + decoders.

A  Train Linear Probe (16384->136) and SmallCNN Probe on full 4x64x64 latent.
   Report accuracy + AUROC on probe features (cross-artist, content-matched).
B  Estimate style delta vectors from TWO independent, artist-disjoint reference
   sets and compare cos(delta1, delta2). Also test universality of target direction.
C  Edit latents along the B-derived direction, decode, measure CSD/ALADIN/LPIPS,
   and dump blind comparison images.

Outputs: probeA.json, dirB.json, editC.json, blind_images/
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
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from transformers import AutoModel

from tools.vae_suite.dataset import RawMemmapWikiArtDataset
import csd_model
import lpips

OUT = Path("/root/Workspace/WEAVE_VAE/runs_vae/v2_diagnosis")
BLIND = OUT / "blind_images"
BLIND.mkdir(parents=True, exist_ok=True)

MANIFEST = Path("/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")
B0_DIR = "/root/Workspace/xy/DiT/data/pretrained/sd-vae-ft-ema"
A0_DIR = "/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae"
ALADIN_CKPT = "/tmp/aladin_repo/aladinVIT_bamfg_64.47.pt"

N_TRAIN_PER_STYLE = 90        # ~12k images total
N_VAL = 2048
NORM_MEAN = [0.48145466, 0.4578275, 0.40821073]
NORM_STD = [0.26862954, 0.26130258, 0.27577711]
DINO_MEAN = [0.485, 0.456, 0.406]
DINO_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------- data
def build_splits():
    ds_tr = RawMemmapWikiArtDataset(MANIFEST, is_training=True)
    ds_va = RawMemmapWikiArtDataset(MANIFEST, is_training=False)

    by_style = defaultdict(list)
    for pos, raw in enumerate(ds_tr.indices):
        rec = ds_tr.records_by_idx[raw]
        by_style[rec["style_id"]].append(pos)

    rng = np.random.default_rng(11)
    tr_pos, tr_y = [], []
    for s, lst in by_style.items():
        pick = lst if len(lst) <= N_TRAIN_PER_STYLE else list(rng.choice(lst, N_TRAIN_PER_STYLE, replace=False))
        tr_pos += [int(p) for p in pick]
        tr_y += [s] * len(pick)

    va_y, va_ar, va_ge = [], [], []
    for i in range(len(ds_va)):
        rec = ds_va.records_by_idx[ds_va.indices[i]]
        va_y.append(rec["style_id"]); va_ar.append(rec.get("artist_id", -1))
        va_ge.append(rec.get("genre_id", -1))
    return ds_tr, ds_va, tr_pos, np.array(tr_y), np.array(va_y), np.array(va_ar), np.array(va_ge)


def extract_latents(vae, ds, positions, device, bs=16, tag=""):
    out = []
    with torch.no_grad():
        for i in range(0, len(positions), bs):
            chunk = positions[i:i + bs]
            x = torch.stack([ds[p]["image"] for p in chunk]).to(device)
            post = vae.encode(x).latent_dist
            out.append(post.mean.float().cpu())
            if (i // bs + 1) % 100 == 0:
                print(f"    [{tag}] {i+len(chunk)}/{len(positions)}", flush=True)
    return torch.cat(out)


def load_vae(path, device):
    v = AutoencoderKL.from_pretrained(path).to(device).eval()
    for p in v.parameters():
        p.requires_grad = False
    return v


# ---------------------------------------------------------------- probes
class LinearProbe(nn.Module):
    def __init__(self, d_in=16384, n_cls=136):
        super().__init__()
        self.fc = nn.Linear(d_in, n_cls)

    def forward(self, z):            # z: (B,4,64,64)
        return self.fc(z.flatten(1))


class SmallCNN(nn.Module):
    def __init__(self, n_cls=136):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
        )
        self.fc = nn.Linear(128, n_cls)

    def forward(self, z):
        return self.fc(self.body(z))


def train_probe(model, z_tr, y_tr, z_va, y_va, device, epochs=25, lr=1e-3, bs=256, tag=""):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    ds = TensorDataset(z_tr, y_tr)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True)
    for ep in range(epochs):
        model.train()
        tot, cnt = 0.0, 0
        for zb, yb in dl:
            zb, yb = zb.to(device), yb.to(device)
            opt.zero_grad()
            loss = lossf(model(zb), yb)
            loss.backward()
            opt.step()
            tot += float(loss); cnt += 1
        if (ep + 1) % 5 == 0:
            print(f"      {tag} ep{ep+1} loss={tot/max(cnt,1):.3f}", flush=True)
    model.eval()
    return model


@torch.no_grad()
def probe_eval(model, z, y, device, bs=256):
    model.eval()
    preds, top5 = [], 0
    for i in range(0, len(z), bs):
        zb = z[i:i + bs].to(device)
        lg = model(zb)
        p = lg.argmax(1).cpu()
        preds.append(p)
        t5 = torch.topk(lg, k=min(5, lg.shape[1]), dim=1).indices.cpu()
        for r in range(len(p)):
            if y[i + r] in t5[r]:
                top5 += 1
    preds = torch.cat(preds).numpy()
    acc = float((preds == y).mean() * 100)
    return {"top1_acc": round(acc, 2), "top5_acc": round(top5 / len(y) * 100, 2)}


@torch.no_grad()
def probe_features(model, z, device, bs=256):
    """Use probe logits as a learned style representation."""
    model.eval()
    out = []
    for i in range(0, len(z), bs):
        out.append(model(z[i:i + bs].to(device)).float().cpu())
    f = torch.cat(out).numpy()
    return f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)


# ---------------------------------------------------------------- pairs / metrics
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
            continue
        if len(pc) > 12: pc = rng.choice(pc, 12, replace=False)
        if len(nc) > 24: nc = rng.choice(nc, 24, replace=False)
        sp = dino[pc] @ dino[i]; sn = dino[nc] @ dino[i]
        gap = np.abs(sp[:, None] - sn[None, :])
        pi, ni = np.unravel_index(np.argmin(gap), gap.shape)
        pos.append((int(i), int(pc[pi]))); neg.append((int(i), int(nc[ni])))
        gaps.append(float(gap[pi, ni]))
    return pos, neg, np.array(gaps)


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


# ---------------------------------------------------------------- CSD / ALADIN
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


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    device = "cuda"
    torch.manual_seed(0)
    np.random.seed(0)

    ds_tr, ds_va, tr_pos, tr_y, va_y, va_ar, va_ge = build_splits()
    print(f"[Split] train={len(tr_pos)} ({len(set(tr_y.tolist()))} styles) | val={len(va_y)}", flush=True)

    # ---- DINO for content matching ----
    print("\n[DINO] val content features...", flush=True)
    dino_m = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()
    nmd = torch.tensor(DINO_MEAN, device=device).view(1, 3, 1, 1)
    nsd = torch.tensor(DINO_STD, device=device).view(1, 3, 1, 1)
    dl = DataLoader(ds_va, batch_size=8, shuffle=False)
    dfeat = []
    with torch.no_grad():
        for i, b in enumerate(dl):
            x = b["image"].to(device)
            x01 = (x.clamp(-1, 1) + 1) / 2
            x224 = F.interpolate(x01, (224, 224), mode="bicubic", align_corners=False)
            o = dino_m(pixel_values=(x224 - nmd) / nsd)
            dfeat.append(F.normalize(o.last_hidden_state[:, 0].float(), dim=-1).cpu())
    del dino_m
    torch.cuda.empty_cache()
    dino = torch.cat(dfeat).numpy()

    same, cross, diff, hard = build_pairs(va_y, va_ar, va_ge)
    cm_pos, cm_neg, cm_gap = build_cm(dino, va_y, va_ar, va_ge)
    print(f"[Pairs] same={len(same)} cross={len(cross)} hard={len(hard)} cm={len(cm_pos)} "
          f"(gap={cm_gap.mean():.4f})", flush=True)

    results = {"pair_stats": {"cross": len(cross), "hard": len(hard),
                              "content_matched": len(cm_pos),
                              "cm_gap_mean": round(float(cm_gap.mean()), 5)}}

    # ---- CSD + ALADIN refs ----
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

    latents = {}
    for tag, path in (("B0", B0_DIR), ("A0", A0_DIR)):
        print(f"\n=== {tag}: extract latents ===", flush=True)
        vae = load_vae(path, device)
        z_tr = extract_latents(vae, ds_tr, tr_pos, device, tag=f"{tag}-train")
        z_va = extract_latents(vae, ds_va, list(range(len(ds_va))), device, tag=f"{tag}-val")
        latents[tag] = {"z_tr": z_tr, "z_va": z_va}
        del vae
        torch.cuda.empty_cache()
        print(f"  {tag}: train {tuple(z_tr.shape)} val {tuple(z_va.shape)}", flush=True)

    # ================= A: readability probes =================
    print("\n" + "=" * 70)
    print("A: STYLE READABILITY PROBES (frozen encoder, full 4x64x64)")
    print("=" * 70, flush=True)
    results["A_readability"] = {}
    for tag in ("B0", "A0"):
        z_tr = latents[tag]["z_tr"]; z_va = latents[tag]["z_va"]
        y_tr = torch.from_numpy(tr_y).long()
        results["A_readability"][tag] = {}

        for pname, pcls, lr, ep in (
            ("LinearProbe_16384", LinearProbe, 2e-3, 25),
            ("SmallCNN", SmallCNN, 1e-3, 25),
        ):
            print(f"\n  [{tag}/{pname}] training...", flush=True)
            model = pcls(n_cls=136)
            model = train_probe(model, z_tr, y_tr, z_va, torch.from_numpy(va_y).long(),
                                device, epochs=ep, lr=lr, tag=f"{tag}/{pname}")
            ev = probe_eval(model, z_va, va_y, device)
            feats = probe_features(model, z_va, device)
            r_cross = auroc(feats, cross, diff)
            r_hard = auroc(feats, cross, hard)
            r_cm = auroc(feats, cm_pos, cm_neg)
            results["A_readability"][tag][pname] = {
                **ev, "cross": r_cross, "hard": r_hard, "content_matched": r_cm,
                "feat_dim": int(feats.shape[1]),
            }
            print(f"    {pname}: top1={ev['top1_acc']}% top5={ev['top5_acc']}% | "
                  f"Cross={r_cross['auroc'] if r_cross else None} "
                  f"Hard={r_hard['auroc'] if r_hard else None} "
                  f"CM={r_cm['auroc'] if r_cm else None}", flush=True)
            del model
            torch.cuda.empty_cache()

    # ================= B: direction consistency =================
    print("\n" + "=" * 70)
    print("B: STYLE DIRECTION CONSISTENCY (artist-disjoint reference sets)")
    print("=" * 70, flush=True)
    results["B_direction"] = {}
    for tag in ("B0", "A0"):
        z_va = latents[tag]["z_va"]
        # index style -> list of (val position, artist)
        by_style = defaultdict(list)
        for i in range(len(va_y)):
            by_style[int(va_y[i])].append((i, int(va_ar[i])))
        usable = [s for s, v in by_style.items() if len(v) >= 8]
        rng = np.random.default_rng(5)

        # per-style centroid of a given half
        def half_centroid(style, half, n_min=3):
            items = by_style[style]
            arts = sorted({a for _, a in items})
            if len(arts) < 2:
                return None
            rng.shuffle(arts)
            k = max(1, len(arts) // 2)
            keep = set(arts[:k] if half == 0 else arts[k:])
            idxs = [i for i, a in items if a in keep]
            if len(idxs) < n_min:
                return None
            return z_va[idxs].mean(0)

        # 1) same transition, two independent reference halves
        cos_same_transition = []
        # 2) same target, different sources  (universal target direction?)
        cos_same_target = []
        # 3) same source, different targets (should be low)
        cos_same_source = []

        pair_list = []
        for s in usable:
            for t in usable:
                if s >= t:
                    continue
                c1s = half_centroid(s, 0); c1t = half_centroid(t, 0)
                c2s = half_centroid(s, 1); c2t = half_centroid(t, 1)
                if any(v is None for v in (c1s, c1t, c2s, c2t)):
                    continue
                d1 = (c1t - c1s).flatten()
                d2 = (c2t - c2s).flatten()
                if d1.norm() < 1e-8 or d2.norm() < 1e-8:
                    continue
                cs = float(F.cosine_similarity(d1.unsqueeze(0), d2.unsqueeze(0)))
                cos_same_transition.append(cs)
                pair_list.append((s, t, d1, d2))

        # universal target direction: delta_{s1->t} vs delta_{s2->t}
        by_target = defaultdict(list)
        for s, t, d1, d2 in pair_list:
            by_target[t].append(((d1 + d2) / 2).flatten())
        for t, ds in by_target.items():
            if len(ds) < 3:
                continue
            for i in range(len(ds)):
                for j in range(i + 1, min(len(ds), i + 6)):
                    cos_same_target.append(float(F.cosine_similarity(ds[i].unsqueeze(0), ds[j].unsqueeze(0))))

        by_source = defaultdict(list)
        for s, t, d1, d2 in pair_list:
            by_source[s].append(((d1 + d2) / 2).flatten())
        for s, ds in by_source.items():
            if len(ds) < 3:
                continue
            for i in range(len(ds)):
                for j in range(i + 1, min(len(ds), i + 6)):
                    cos_same_source.append(float(F.cosine_similarity(ds[i].unsqueeze(0), ds[j].unsqueeze(0))))

        res = {
            "n_transitions": len(cos_same_transition),
            "cos_same_transition_mean": round(float(np.mean(cos_same_transition)), 4),
            "cos_same_transition_median": round(float(np.median(cos_same_transition)), 4),
            "cos_same_transition_p25": round(float(np.percentile(cos_same_transition, 25)), 4),
            "cos_same_transition_p75": round(float(np.percentile(cos_same_transition, 75)), 4),
            "cos_same_target_mean": round(float(np.mean(cos_same_target)), 4) if cos_same_target else None,
            "cos_same_source_mean": round(float(np.mean(cos_same_source)), 4) if cos_same_source else None,
            "n_target_pairs": len(cos_same_target),
            "n_source_pairs": len(cos_same_source),
        }
        results["B_direction"][tag] = res
        print(f"  {tag}: same-transition cos = {res['cos_same_transition_mean']:.4f} "
              f"(median {res['cos_same_transition_median']:.4f}, "
              f"IQR [{res['cos_same_transition_p25']:.3f}, {res['cos_same_transition_p75']:.3f}])",
              flush=True)
        print(f"        same-target cos = {res['cos_same_target_mean']} | "
              f"same-source cos = {res['cos_same_source_mean']}", flush=True)

    # ================= C: real editing =================
    print("\n" + "=" * 70)
    print("C: REAL EDITING with data-derived direction")
    print("=" * 70, flush=True)
    results["C_editing"] = {}
    for tag, path in (("A0", A0_DIR),):
        vae = load_vae(path, device)
        z_va = latents[tag]["z_va"]

        by_style = defaultdict(list)
        for i in range(len(va_y)):
            by_style[int(va_y[i])].append((i, int(va_ar[i])))
        usable = [s for s, v in by_style.items() if len(v) >= 10]
        rng = np.random.default_rng(17)

        # build averaged transition directions from FULL val centroids
        centroid = {s: z_va[[i for i, _ in by_style[s]]].mean(0) for s in usable}

        rows = {"data_dir": [], "random_dir": [], "csd_grad": []}
        norm_fracs = [0.02, 0.05, 0.10]
        for nf in norm_fracs:
            rows["data_dir"].append([])
            rows["random_dir"].append([])
            rows["csd_grad"].append([])

        n_tasks = 24
        tasks = []
        tries = 0
        while len(tasks) < n_tasks and tries < 500:
            tries += 1
            s = int(rng.choice(usable))
            t = int(rng.choice([x for x in usable if x != s]))
            src = int(rng.choice([i for i, _ in by_style[s]]))
            tgt_ref = int(rng.choice([i for i, _ in by_style[t]]))
            tasks.append((s, t, src, tgt_ref))

        for ti, (s, t, src, tgt_ref) in enumerate(tasks):
            x = ds_va[src]["image"].unsqueeze(0).to(device)
            x_t = ds_va[tgt_ref]["image"].unsqueeze(0).to(device)
            with torch.no_grad():
                z0 = vae.encode(x).latent_dist.mean
                rec0 = vae.decode(z0).sample
                f_tgt = csd_feat(csd, x_t, nm, ns)
                al_tgt = aladin_feat(aladin, x_t, device)
                f_rec0 = csd_feat(csd, rec0, nm, ns)
                al_rec0 = aladin_feat(aladin, rec0, device)
            base_c = float((f_rec0 * f_tgt).sum())
            base_a = float((al_rec0 * al_tgt).sum())

            # (1) data-derived direction: target centroid - source centroid
            d_data = (centroid[t] - centroid[s]).flatten()
            d_data = d_data / (d_data.norm() + 1e-12)

            # (2) random
            d_rand = torch.randn_like(d_data)
            d_rand = d_rand / (d_rand.norm() + 1e-12)

            # (3) CSD image gradient (reference upper bound)
            zr = z0.detach().clone().requires_grad_(True)
            fr = csd_feat(csd, vae.decode(zr).sample, nm, ns)
            g = torch.autograd.grad((fr * f_tgt).sum(), zr)[0].flatten()
            d_grad = g / (g.norm() + 1e-12)

            z_norm = float(z0.flatten().norm())
            for ni, nf in enumerate(norm_fracs):
                eps = nf * z_norm
                for key, d in (("data_dir", d_data), ("random_dir", d_rand), ("csd_grad", d_grad)):
                    zn = (z0.flatten() + eps * d).view(1, 4, 64, 64)
                    with torch.no_grad():
                        rec = vae.decode(zn).sample
                        fc = csd_feat(csd, rec, nm, ns)
                        ac = aladin_feat(aladin, rec, device)
                        rows[key][ni].append({
                            "dt_csd": float((fc * f_tgt).sum()) - base_c,
                            "dt_aladin": float((ac * al_tgt).sum()) - base_a,
                            "lpips": float(lpips_fn(rec.float(), rec0.float()).mean()),
                        })

            # blind image: 4 panels in random order
            if ti < 16:
                panels = {"noop": rec0[0]}
                eps = norm_fracs[1] * z_norm
                for key, d in (("data", d_data), ("grad", d_grad), ("rand", d_rand)):
                    zn = (z0.flatten() + eps * d).view(1, 4, 64, 64)
                    with torch.no_grad():
                        panels[key] = vae.decode(zn).sample[0]
                order = ["noop", "data", "grad", "rand"]
                rng.shuffle(order)
                W = H = 384
                canvas = Image.new("RGB", (W * 4, H))
                for pi, k in enumerate(order):
                    canvas.paste(to_pil(panels[k]).resize((W, H)), (pi * W, 0))
                canvas.save(BLIND / f"blind_{ti:02d}_key_{'-'.join(order)}.jpg", quality=92)

        res = {}
        for key in ("data_dir", "random_dir", "csd_grad"):
            res[key] = {}
            for ni, nf in enumerate(norm_fracs):
                rr = rows[key][ni]
                res[key][str(nf)] = {
                    "mean_dt_csd": round(float(np.mean([r["dt_csd"] for r in rr])), 5),
                    "pct_pos_csd": round(float(np.mean([r["dt_csd"] > 0 for r in rr]) * 100), 1),
                    "mean_dt_aladin": round(float(np.mean([r["dt_aladin"] for r in rr])), 5),
                    "pct_pos_aladin": round(float(np.mean([r["dt_aladin"] > 0 for r in rr]) * 100), 1),
                    "mean_lpips": round(float(np.mean([r["lpips"] for r in rr])), 4),
                }
        results["C_editing"][tag] = res
        del vae
        torch.cuda.empty_cache()

    results["elapsed_sec"] = round(time.time() - t0, 1)
    (OUT / "ABC_three_tests.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # ---- summary ----
    print("\n" + "=" * 92)
    print("A SUMMARY: style readability of frozen latent")
    print("=" * 92)
    print(f"{'model':<6}{'probe':<20}{'top1':>8}{'top5':>8}{'Cross':>8}{'Hard':>8}{'CM':>8}")
    for tag, d in results["A_readability"].items():
        for pn, v in d.items():
            print(f"{tag:<6}{pn:<20}{v['top1_acc']:>8.2f}{v['top5_acc']:>8.2f}"
                  f"{(v['cross']['auroc'] if v['cross'] else 0):>8.2f}"
                  f"{(v['hard']['auroc'] if v['hard'] else 0):>8.2f}"
                  f"{(v['content_matched']['auroc'] if v['content_matched'] else 0):>8.2f}")

    print("\n" + "=" * 92)
    print("C SUMMARY: editing response")
    print("=" * 92)
    print(f"{'dir':<12}{'norm':>6}{'dT_CSD':>10}{'%pos':>7}{'dT_ALADIN':>11}{'%pos':>7}{'LPIPS':>8}")
    for key, d in results["C_editing"]["A0"].items():
        for nf, v in d.items():
            print(f"{key:<12}{nf:>6}{v['mean_dt_csd']:>10.4f}{v['pct_pos_csd']:>7.1f}"
                  f"{v['mean_dt_aladin']:>11.4f}{v['pct_pos_aladin']:>7.1f}{v['mean_lpips']:>8.4f}")

    print(f"\n[Saved] {OUT/'ABC_three_tests.json'}  ({results['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
