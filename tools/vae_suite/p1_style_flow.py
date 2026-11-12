"""VAE-P1: reversible style subspace pre-training.

Frozen A0 Encoder/Decoder. Only the invertible transform T is trained.
  z [B,4,64,64] --PU(4)--> [B,64,16,16] --4x AdditiveCoupling--> [B,16384]
  split -> u [B,768], v [B,15616] with  u = u0 + f_theta(v0), v = v0

Two arms (identical init & budget):
  A  TeacherOnly   : CSD cosine alignment + light displacement reg
  B  TeacherLabels : A + cross-artist supervised contrastive on u

Success bar: Content-Matched AUROC >= 56.5% with paired bootstrap vs SmallCNN 53.39%.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/Workspace/WEAVE_VAE")
sys.path.insert(0, "/root/Workspace/WEAVE_VAE/tools/12h")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from tools.vae_suite.dataset import RawMemmapWikiArtDataset
import csd_model

OUT = Path("/root/Workspace/WEAVE_VAE/runs_vae/v2_diagnosis")
CACHE = OUT / "p1_cache"
CACHE.mkdir(parents=True, exist_ok=True)

MANIFEST = Path("/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")
A0_DIR = "/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae"
NORM_MEAN = [0.48145466, 0.4578275, 0.40821073]
NORM_STD = [0.26862954, 0.26130258, 0.27577711]

U_DIM = 768
TOTAL = 16384
V_DIM = TOTAL - U_DIM


# ================================================================ caching
def build_cache(n_train=20000, device="cuda"):
    fz = CACHE / f"train_z_{n_train}.pt"
    ft = CACHE / f"train_t_{n_train}.pt"
    fmeta = CACHE / f"train_meta_{n_train}.pt"
    if fz.exists() and ft.exists() and fmeta.exists():
        print("[Cache] train cache found, loading.", flush=True)
        return (torch.load(fz, weights_only=False), torch.load(ft, weights_only=False),
                torch.load(fmeta, weights_only=False))

    ds = RawMemmapWikiArtDataset(MANIFEST, is_training=True)
    by_style = {}
    for pos, raw in enumerate(ds.indices):
        rec = ds.records_by_idx[raw]
        by_style.setdefault(rec["style_id"], []).append(pos)

    rng = np.random.default_rng(21)
    per = max(1, n_train // max(len(by_style), 1))
    sel = []
    for s, lst in by_style.items():
        sel += [int(p) for p in (lst if len(lst) <= per else rng.choice(lst, per, replace=False))]
    sel = sel[:n_train]
    print(f"[Cache] selected {len(sel)} train images over {len(by_style)} styles", flush=True)

    vae = AutoencoderKL.from_pretrained(A0_DIR).to(device).eval()
    for p in vae.parameters():
        p.requires_grad = False
    csd, _, _ = csd_model.load_csd(device)
    csd = csd.float().eval()
    for p in csd.parameters():
        p.requires_grad = False
    nm = torch.tensor(NORM_MEAN, device=device).view(1, 3, 1, 1)
    ns = torch.tensor(NORM_STD, device=device).view(1, 3, 1, 1)

    Z, T, S, A, G = [], [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(sel), 16):
            idx = sel[i:i + 16]
            x = torch.stack([ds[p]["image"] for p in idx]).to(device)
            Z.append(vae.encode(x).latent_dist.mean.float().cpu())
            x01 = (x.clamp(-1, 1) + 1.0) / 2.0
            x224 = F.interpolate(x01, (224, 224), mode="bicubic", align_corners=False)
            _, _, sty = csd((x224 - nm) / ns)
            T.append(sty.float().cpu())
            for p in idx:
                r = ds.records_by_idx[ds.indices[p]]
                S.append(r["style_id"]); A.append(r.get("artist_id", -1)); G.append(r.get("genre_id", -1))
            if (i // 16 + 1) % 100 == 0:
                print(f"  [{i+len(idx)}/{len(sel)}] {time.time()-t0:.0f}s", flush=True)

    z = torch.cat(Z); t = torch.cat(T)
    meta = {"style": torch.tensor(S), "artist": torch.tensor(A), "genre": torch.tensor(G)}
    torch.save(z, fz); torch.save(t, ft); torch.save(meta, fmeta)
    print(f"[Cache] saved {tuple(z.shape)} latents, {tuple(t.shape)} teacher feats", flush=True)
    return z, t, meta


# ================================================================ flow
class Coupling(nn.Module):
    """Additive coupling on [B,64,H,W] with 32/32 channel split."""

    def __init__(self, ch=64, hidden=64):
        super().__init__()
        self.c = nn.Sequential(
            nn.Conv2d(ch // 2, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, ch // 2, 3, padding=1),
        )
        self.d = nn.Sequential(
            nn.Conv2d(ch // 2, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, ch // 2, 3, padding=1),
        )
        for m in (self.c, self.d):
            nn.init.zeros_(m[-1].weight); nn.init.zeros_(m[-1].bias)

    def forward(self, x):
        x1, x2 = x[:, :32], x[:, 32:]
        y1 = x1 + self.c(x2)
        y2 = x2 + self.d(y1)
        return torch.cat([y1, y2], 1)

    def inverse(self, y):
        y1, y2 = y[:, :32], y[:, 32:]
        x2 = y2 - self.d(y1)
        x1 = y1 - self.c(x2)
        return torch.cat([x1, x2], 1)


class StyleFlow(nn.Module):
    def __init__(self, n_coupling=4, u_dim=U_DIM, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.u_dim = u_dim
        self.couplings = nn.ModuleList([Coupling(64, 64) for _ in range(n_coupling)])
        # fixed invertible channel permutations
        perms = []
        for _ in range(n_coupling):
            p = torch.randperm(64, generator=g)
            perms.append(p)
        self.register_buffer("perms", torch.stack(perms))   # (K, 64)
        # global additive coupling: u = u0 + f(v0)
        self.f = nn.Sequential(
            nn.Linear(TOTAL - u_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, u_dim),
        )
        nn.init.zeros_(self.f[-1].weight); nn.init.zeros_(self.f[-1].bias)

    def forward(self, z):
        """z: [B,4,64,64] -> (u [B,768], v [B,15616])"""
        x = F.pixel_unshuffle(z, 4)                     # [B,64,16,16]
        for k, cp in enumerate(self.couplings):
            x = cp(x)
            x = x[:, self.perms[k]]
        flat = x.flatten(1)                             # [B,16384]
        u0, v0 = flat[:, :self.u_dim], flat[:, self.u_dim:]
        u = u0 + self.f(v0)
        return u, v0

    def inverse(self, u, v):
        u0 = u - self.f(v)
        flat = torch.cat([u0, v], 1)
        x = flat.view(-1, 64, 16, 16)
        for k in reversed(range(len(self.couplings))):
            inv = torch.argsort(self.perms[k])
            x = x[:, inv]
            x = self.couplings[k].inverse(x)
        return F.pixel_shuffle(x, 4)


# ================================================================ losses
def supcon(feats, labels, temp=0.10):
    """feats already L2-normalized. Positive = same label, excluding self."""
    sim = feats @ feats.t() / temp
    B = feats.shape[0]
    eye = torch.eye(B, dtype=torch.bool, device=feats.device)
    pos = (labels[:, None] == labels[None, :]) & ~eye
    logits = sim - sim.max(1, keepdim=True).values.detach()
    exp = torch.exp(logits) * (~eye)
    denom = exp.sum(1, keepdim=True).clamp(min=1e-12)
    logp = logits - torch.log(denom)
    npos = pos.sum(1)
    valid = npos > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=feats.device)
    per = -(logp * pos).sum(1) / npos.clamp(min=1)
    return per[valid].mean()


# ================================================================ eval
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


def auroc_pairs(feats, pos, neg, n_boot=200, seed=42, cap=800):
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
    return {"auroc": round(auc, 2), "lo": round(float(lo), 2), "hi": round(float(hi), 2), "n": int(n),
            "_pos": a1, "_neg": a0}


def knn_purity(feats, styles, ks=(1, 5, 10), block=256):
    N = len(styles)
    hits = {k: 0 for k in ks}
    for st in range(0, N, block):
        en = min(st + block, N)
        sim = feats[st:en] @ feats.T
        for r in range(en - st):
            sim[r, st + r] = -np.inf
        order = np.argsort(-sim, axis=1)
        for r in range(en - st):
            for k in ks:
                if styles[st + r] in styles[order[r, :k]]:
                    hits[k] += 1
    return {f"knn{k}": round(hits[k] / N * 100, 2) for k in ks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--temp", type=float, default=0.10)
    ap.add_argument("--w-supcon", type=float, default=0.1)
    ap.add_argument("--w-reg", type=float, default=1e-3)
    ap.add_argument("--w-rms", type=float, default=0.1)
    ap.add_argument("--speed-test", type=int, default=50)
    ap.add_argument("--arms", default="A,B")
    args = ap.parse_args()

    device = "cuda"
    torch.manual_seed(0); np.random.seed(0)

    # ---------- cache ----------
    z_tr, t_tr, meta_tr = build_cache(args.n_train, device)
    y_tr = meta_tr["style"]; a_tr = meta_tr["artist"]; g_tr = meta_tr["genre"]

    # ---------- val cache ----------
    fvz = CACHE / "val_z.pt"; fvt = CACHE / "val_t.pt"; fvm = CACHE / "val_meta.pt"
    if fvz.exists():
        z_va = torch.load(fvz, weights_only=False); t_va = torch.load(fvt, weights_only=False)
        mv = torch.load(fvm, weights_only=False)
    else:
        ds = RawMemmapWikiArtDataset(MANIFEST, is_training=False)
        vae = AutoencoderKL.from_pretrained(A0_DIR).to(device).eval()
        csd, _, _ = csd_model.load_csd(device)
        csd = csd.float().eval()
        nm = torch.tensor(NORM_MEAN, device=device).view(1, 3, 1, 1)
        ns = torch.tensor(NORM_STD, device=device).view(1, 3, 1, 1)
        Z, T = [], []
        S, A, G = [], [], []
        print("[Cache] val ...", flush=True)
        with torch.no_grad():
            for i in range(0, len(ds), 16):
                idx = list(range(i, min(i + 16, len(ds))))
                x = torch.stack([ds[p]["image"] for p in idx]).to(device)
                Z.append(vae.encode(x).latent_dist.mean.float().cpu())
                x01 = (x.clamp(-1, 1) + 1) / 2
                x224 = F.interpolate(x01, (224, 224), mode="bicubic", align_corners=False)
                _, _, sty = csd((x224 - nm) / ns)
                T.append(sty.float().cpu())
                for p in idx:
                    r = ds.records_by_idx[ds.indices[p]]
                    S.append(r["style_id"]); A.append(r.get("artist_id", -1)); G.append(r.get("genre_id", -1))
                if (i // 16 + 1) % 40 == 0:
                    print(f"  {i+len(idx)}/{len(ds)}", flush=True)
        z_va = torch.cat(Z); t_va = torch.cat(T)
        mv = {"style": torch.tensor(S), "artist": torch.tensor(A), "genre": torch.tensor(G)}
        torch.save(z_va, fvz); torch.save(t_va, fvt); torch.save(mv, fvm)
        del vae, csd
        torch.cuda.empty_cache()

    y_va = mv["style"].numpy(); a_va = mv["artist"].numpy(); g_va = mv["genre"].numpy()
    print(f"[Data] train {tuple(z_tr.shape)} | val {tuple(z_va.shape)}", flush=True)

    # ---------- content-matched pairs (DINO on val) ----------
    from transformers import AutoModel
    DINO_MEAN = [0.485, 0.456, 0.406]; DINO_STD = [0.229, 0.224, 0.225]
    ds_va = RawMemmapWikiArtDataset(MANIFEST, is_training=False)
    dm = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()
    nmd = torch.tensor(DINO_MEAN, device=device).view(1, 3, 1, 1)
    nsd = torch.tensor(DINO_STD, device=device).view(1, 3, 1, 1)
    dl = DataLoader(ds_va, batch_size=8, shuffle=False)
    df = []
    with torch.no_grad():
        for i, b in enumerate(dl):
            x = b["image"].to(device)
            x01 = (x.clamp(-1, 1) + 1) / 2
            x224 = F.interpolate(x01, (224, 224), mode="bicubic", align_corners=False)
            o = dm(pixel_values=(x224 - nmd) / nsd)
            df.append(F.normalize(o.last_hidden_state[:, 0].float(), dim=-1).cpu())
    del dm
    torch.cuda.empty_cache()
    dino = torch.cat(df).numpy()
    dino = dino / (np.linalg.norm(dino, axis=1, keepdims=True) + 1e-8)
    same, cross, diff, hard = build_pairs(y_va, a_va, g_va)
    cm_pos, cm_neg = build_cm(dino, y_va, a_va, g_va)
    print(f"[Pairs] cross={len(cross)} hard={len(hard)} cm={len(cm_pos)}", flush=True)

    # ---------- reference: A0 raw latent u (first 768 of standardized) ----------
    def eval_u(u_np, tag, results):
        un = u_np / (np.linalg.norm(u_np, axis=1, keepdims=True) + 1e-8)
        r_cm = auroc_pairs(un, cm_pos, cm_neg)
        r_cross = auroc_pairs(un, cross, diff)
        r_hard = auroc_pairs(un, cross, hard)
        r_knn = knn_purity(un, y_va)
        # teacher alignment on val
        tva = t_va.numpy(); tva = tva / (np.linalg.norm(tva, axis=1, keepdims=True) + 1e-8)
        cos = (un * tva).sum(1)
        results[tag] = {
            "content_matched": {k: v for k, v in (r_cm or {}).items() if not k.startswith("_")},
            "cross_artist": {k: v for k, v in (r_cross or {}).items() if not k.startswith("_")},
            "same_genre_hard": {k: v for k, v in (r_hard or {}).items() if not k.startswith("_")},
            "knn": r_knn,
            "teacher_cos_mean": round(float(cos.mean()), 4),
        }
        print(f"  [{tag}] CM={r_cm['auroc'] if r_cm else None} Cross={r_cross['auroc'] if r_cross else None} "
              f"Hard={r_hard['auroc'] if r_hard else None} kNN1={r_knn['knn1']} "
              f"teacherCos={cos.mean():.4f}", flush=True)
        return (r_cm or {}), (r_cross or {})

    results = {}

    # baseline: raw A0 latent first 768 dims (after PU4 flatten, standardized)
    def raw_u(z):
        flat = F.pixel_unshuffle(z, 4).flatten(1)
        return flat[:, :U_DIM]

    print("\n=== BASELINE (raw A0 latent u0) ===", flush=True)
    with torch.no_grad():
        u_base = raw_u(z_va).numpy()
    cm_base, cross_base = eval_u(u_base, "A0_raw_u0", results)

    # ---------- train ----------
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    trained = {}

    for arm in arms:
        print(f"\n{'='*70}\nARM {arm}: {'TeacherOnly' if arm=='A' else 'Teacher+Labels'}\n{'='*70}", flush=True)
        torch.manual_seed(1234)      # identical init for both arms
        flow = StyleFlow(n_coupling=4, u_dim=U_DIM, seed=7).to(device).float()
        opt = torch.optim.AdamW(flow.parameters(), lr=args.lr, weight_decay=args.wd)

        N = z_tr.shape[0]
        steps_done = 0
        t0 = time.time()
        hist = []
        n_steps = args.steps
        bs = args.batch

        for step in range(1, n_steps + 1):
            flow.train()
            idx = torch.from_numpy(np.random.choice(N, bs, replace=False))
            zb = z_tr[idx].to(device)
            tb = t_tr[idx].to(device)
            yb = y_tr[idx].to(device)

            u, v = flow(zb)
            un = F.normalize(u, dim=-1)
            tn = F.normalize(tb, dim=-1)

            l_teacher = (1.0 - (un * tn).sum(1)).mean()
            rms_u = u.pow(2).mean(1).sqrt().mean()
            l_rms = (rms_u - 1.0) ** 2
            # displacement reg toward standardized PU4 coords
            p = F.pixel_unshuffle(zb, 4).flatten(1)
            l_reg = (torch.cat([u, v], 1) - p).pow(2).mean() / TOTAL

            loss = l_teacher + args.w_rms * l_rms + args.w_reg * l_reg
            l_sup = torch.tensor(0.0, device=device)
            if arm == "B":
                l_sup = supcon(un, yb, temp=args.temp)
                loss = loss + args.w_supcon * l_sup

            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
            opt.step()
            steps_done += 1

            if step % 10 == 0 or step == 1:
                vram = torch.cuda.max_memory_allocated(0) / 1024**2
                print(f"  [ARM {arm}] {step:4d}/{n_steps} | L_t={l_teacher.item():.4f} "
                      f"L_rms={l_rms.item():.4f} L_reg={l_reg.item():.2e} "
                      f"L_sup={l_sup.item():.4f} | rms_u={rms_u.item():.3f} "
                      f"g={gn:.2f} | {time.time()-t0:.0f}s VRAM={vram:.0f}M", flush=True)
                hist.append({"step": step, "l_teacher": round(l_teacher.item(), 5),
                             "l_rms": round(l_rms.item(), 5), "l_sup": round(l_sup.item(), 5),
                             "rms_u": round(rms_u.item(), 4), "sec": round(time.time()-t0, 1)})

            if step == args.speed_test:
                el = time.time() - t0
                print(f"  [SPEED] {args.speed_test} steps in {el:.1f}s -> {el/args.speed_test:.3f}s/step; "
                      f"500 steps ~ {el/args.speed_test*500/60:.1f} min", flush=True)

            if step in (100, 250, 500):
                torch.save({"flow": flow.state_dict(), "step": step, "arm": arm},
                           OUT / f"t_{'teacher_only' if arm=='A' else 'teacher_labels'}_s{step}.pt")
                # eval on val
                flow.eval()
                with torch.no_grad():
                    u_val, _ = flow(z_va.to(device))
                eval_u(u_val.float().cpu().numpy(), f"ARM{arm}_step{step}", results)

        # invertibility check
        flow.eval()
        with torch.no_grad():
            zs = z_va[:64].to(device)
            u, v = flow(zs)
            zr = flow.inverse(u, v)
            rel = float(((zr - zs).norm() / zs.norm()).item())
        print(f"  [Invert] rel error ||T^-1(T(z))-z||/||z|| = {rel:.3e}", flush=True)
        results[f"ARM{arm}_invertibility"] = {"rel_err": rel}
        results[f"ARM{arm}_history"] = hist

        torch.save({"flow": flow.state_dict(), "arm": arm}, OUT / f"t_{'teacher_only' if arm=='A' else 'teacher_labels'}.pt")
        trained[arm] = flow
        del flow
        torch.cuda.empty_cache()

    (OUT / "p1_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # ---------- summary ----------
    print("\n" + "=" * 96)
    print("P1 SUMMARY (Content-Matched AUROC is the primary metric)")
    print("=" * 96)
    print(f"{'variant':<24}{'CM AUROC':>10}{'95% CI':>18}{'CrossArtist':>13}{'Hard':>8}{'kNN@1':>8}{'tCos':>8}")
    print("-" * 96)
    for k, v in results.items():
        if not isinstance(v, dict) or "content_matched" not in v:
            continue
        cm = v["content_matched"]; cr = v["cross_artist"]
        ci = "[{}, {}]".format(cm["lo"], cm["hi"])
        print(f"{k:<24}{cm['auroc']:>10.2f}{ci:>18}"
              f"{cr['auroc']:>13.2f}{v['same_genre_hard']['auroc']:>8.2f}"
              f"{v['knn']['knn1']:>8.2f}{v['teacher_cos_mean']:>8.4f}")
    print("=" * 96)
    print(f"\n[Saved] {OUT/'p1_metrics.json'}")


if __name__ == "__main__":
    main()
