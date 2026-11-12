"""Style-Rank 20-class prep: dataset filter, DINO cache, triplet index, fixed P, prototypes.

Produces (all under runs/vae_style_rank20/):
  projection.pt   : P (768x16384) random orthonormal, mu_z (16384), seed
  prototypes.pt   : 20 x 768 regular-simplex prototypes (a_i . a_j = -1/19)
  triplets.pt     : anchor/positive/negative position triples + matching stats
  classes.json    : the 20 selected styles
  prep_stats.json : coverage / matching quality / mu_z provenance
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/root/Workspace/WEAVE_VAE")

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader

from tools.vae_suite.dataset import RawMemmapWikiArtDataset

OUT = Path("/root/Workspace/WEAVE_VAE/runs/vae_style_rank20")
OUT.mkdir(parents=True, exist_ok=True)
MANIFEST = Path("/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")
A0_DIR = "/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae"
DEV = "cuda"
K = 768
DIM = 16384
MIN_IMGS = 1000
SEED = 20260109
DINO_MEAN = [0.485, 0.456, 0.406]
DINO_STD = [0.229, 0.224, 0.225]


def main():
    print("=== Style-Rank 20-class prep ===", flush=True)
    ds = RawMemmapWikiArtDataset(MANIFEST, is_training=True)
    man = json.loads(MANIFEST.read_text())
    id2name = {v: k for k, v in man["style_to_id"].items()}

    # ---- style counts ----
    style_pos = defaultdict(list)
    for pos, raw in enumerate(ds.indices):
        rec = ds.records_by_idx[raw]
        style_pos[rec["style_id"]].append(pos)

    counts = {s: len(v) for s, v in style_pos.items()}
    keep = sorted([s for s, n in counts.items() if n >= MIN_IMGS])
    print(f"[Styles] total={len(counts)} | kept(>={MIN_IMGS})={len(keep)}", flush=True)
    classes = [{"id": s, "name": id2name.get(s, "?"), "n": counts[s]} for s in keep]
    for c in classes:
        print(f"   {c['id']:>4} {c['name']:<32} {c['n']:>6}", flush=True)
    (OUT / "classes.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")

    # ---- positions in kept styles ----
    pos_list = []
    for s in keep:
        pos_list += style_pos[s]
    pos_list = sorted(set(pos_list))
    print(f"[Pool] {len(pos_list)} images in {len(keep)} classes", flush=True)

    # meta per position
    y, a, g = [], [], []
    for p in pos_list:
        r = ds.records_by_idx[ds.indices[p]]
        y.append(r["style_id"]); a.append(r.get("artist_id", -1)); g.append(r.get("genre_id", -1))
    y = np.array(y); a = np.array(a); g = np.array(g)
    style2new = {s: i for i, s in enumerate(keep)}
    yn = np.array([style2new[s] for s in y])

    # ---- DINO cache ----
    fdin = OUT / "train_dino.pt"
    if fdin.exists():
        print("[DINO] cache found", flush=True)
        D = torch.load(fdin, weights_only=False)
    else:
        print("[DINO] extracting (this is the long step)...", flush=True)
        from transformers import AutoModel
        dm = AutoModel.from_pretrained("facebook/dinov2-base").to(DEV).eval()
        nm = torch.tensor(DINO_MEAN, device=DEV).view(1, 3, 1, 1)
        ns = torch.tensor(DINO_STD, device=DEV).view(1, 3, 1, 1)
        out = []
        import time
        t0 = time.time()
        with torch.no_grad():
            for i in range(0, len(pos_list), 32):
                chunk = pos_list[i:i + 32]
                x = torch.stack([ds[p]["image"] for p in chunk]).to(DEV)
                x01 = (x.clamp(-1, 1) + 1) / 2
                x224 = F.interpolate(x01, (224, 224), mode="bicubic", align_corners=False)
                o = dm(pixel_values=(x224 - nm) / ns)
                out.append(F.normalize(o.last_hidden_state[:, 0].float(), dim=-1).cpu())
                if (i // 32 + 1) % 100 == 0:
                    print(f"   {i+len(chunk)}/{len(pos_list)}  {time.time()-t0:.0f}s", flush=True)
        D = torch.cat(out)
        del dm
        torch.cuda.empty_cache()
        torch.save(D, fdin)
        print(f"[DINO] done {tuple(D.shape)}", flush=True)
    D = D.to(DEV)

    # ---- mu_z from A0 latents (subsample for speed) ----
    fmu = OUT / "mu_z.pt"
    if fmu.exists():
        mu_z = torch.load(fmu, weights_only=False).to(DEV)
        print(f"[mu_z] cache found {tuple(mu_z.shape)}", flush=True)
    else:
        print("[mu_z] computing from A0 on 15000 images...", flush=True)
        rng = np.random.default_rng(SEED)
        sub = list(rng.choice(len(pos_list), min(15000, len(pos_list)), replace=False))
        vae = AutoencoderKL.from_pretrained(A0_DIR).to(DEV).eval()
        acc = []
        with torch.no_grad():
            for i in range(0, len(sub), 16):
                chunk = [pos_list[j] for j in sub[i:i + 16]]
                x = torch.stack([ds[p]["image"] for p in chunk]).to(DEV)
                mu = vae.encode(x).latent_dist.mean.float().flatten(1)
                acc.append(mu.cpu())
                if (i // 16 + 1) % 100 == 0:
                    print(f"   {i+16}/{len(sub)}", flush=True)
        Z = torch.cat(acc)
        mu_z = Z.mean(0, keepdim=True).to(DEV)
        torch.save(mu_z.cpu(), fmu)
        del vae
        torch.cuda.empty_cache()
        print(f"[mu_z] {tuple(mu_z.shape)} rms={mu_z.pow(2).mean().sqrt().item():.4f}", flush=True)

    # ---- P: random orthonormal (NO CSD-PLS) ----
    fp = OUT / "projection.pt"
    if fp.exists():
        proj = torch.load(fp, weights_only=False)
        P = proj["P"].to(DEV)
        print(f"[P] cache found orth_err="
              f"{(P @ P.t() - torch.eye(K, device=DEV)).abs().max().item():.2e}", flush=True)
    else:
        g0 = torch.Generator().manual_seed(SEED)
        R = torch.randn(DIM, K, generator=g0)
        Q, _ = torch.linalg.qr(R)
        P = Q.t().contiguous().to(DEV)
        print(f"[P] random orthonormal, orth_err="
              f"{(P @ P.t() - torch.eye(K, device=DEV)).abs().max().item():.2e}", flush=True)

    # ---- prototypes: 20-class regular simplex embedded in 768-d ----
    fproto = OUT / "prototypes.pt"
    if fproto.exists():
        proto = torch.load(fproto, weights_only=False).to(DEV)
    else:
        n = len(keep)
        G = (n / (n - 1)) * (torch.eye(n) - torch.ones(n, n) / n)
        w, V = torch.linalg.eigh(G)
        w = w.flip(0); V = V.flip(1)
        r = int((w > 1e-8).sum())
        A = V[:, :r] @ torch.diag(torch.sqrt(w[:r]))        # (n, r) rows unit norm
        gg = torch.Generator().manual_seed(SEED + 1)
        R2 = torch.randn(r, K, generator=gg)
        Q2, _ = torch.linalg.qr(R2.t())                      # (K, r) orthonormal cols
        proto = (A @ Q2.t()).float()
        proto = F.normalize(proto, dim=-1)
        torch.save(proto.cpu(), fproto)
    proto = proto.to(DEV)
    Gp = proto @ proto.t()
    off = Gp[~torch.eye(len(keep), dtype=torch.bool, device=DEV)]
    print(f"[Proto] {tuple(proto.shape)} | off-diag mean={off.mean().item():+.4f} "
          f"(target {-1/(len(keep)-1):+.4f})", flush=True)

    torch.save({"P": P.cpu(), "mu_z": mu_z.cpu(), "seed": SEED, "k": K, "dim": DIM},
               fp)

    # ---- triplet index ----
    ftri = OUT / "triplets.pt"
    if ftri.exists():
        print("[Triplets] cache found", flush=True)
        tri = torch.load(ftri, weights_only=False)
    else:
        print("[Triplets] building content-matched triplets...", flush=True)
        rng = np.random.default_rng(SEED + 7)
        by_style = defaultdict(list)
        for i, s in enumerate(yn):
            by_style[int(s)].append(i)
        # style -> genre -> [local idx]
        style_genre = defaultdict(lambda: defaultdict(list))
        for i, s in enumerate(yn):
            style_genre[int(s)][int(g[i])].append(i)

        N_TARGET = 24000
        anchors, poss, negs, stats = [], [], [], []
        max_gap_05 = max_gap_10 = 0
        fail = 0
        for _ in range(N_TARGET * 3):
            if len(anchors) >= N_TARGET:
                break
            si = int(rng.integers(len(keep)))
            cand_a = by_style[si]
            ai = int(rng.choice(cand_a))
            # positive: same style, different artist
            same = np.array(cand_a)
            diff_art = same[a[same] != a[ai]]
            if len(diff_art) == 0:
                fail += 1
                continue
            pi = int(rng.choice(diff_art))
            # negative: different style, prefer same genre as positive
            gp = int(g[pi])
            neg_pool = []
            for sj in range(len(keep)):
                if sj == si:
                    continue
                lst = style_genre[sj].get(gp, [])
                if lst:
                    neg_pool += lst
            if len(neg_pool) < 8:
                neg_pool = []
                for sj in range(len(keep)):
                    if sj == si:
                        continue
                    neg_pool += by_style[sj]
            if len(neg_pool) < 8:
                fail += 1
                continue
            neg_pool = np.array(neg_pool)
            cand = rng.choice(neg_pool, min(96, len(neg_pool)), replace=False)
            c_ap = float(D[ai] @ D[pi])
            c_an = (D[cand] @ D[ai]).cpu().numpy()
            gap = np.abs(c_an - c_ap)
            best = int(np.argmin(gap))
            gbest = float(gap[best])
            if gbest < 0.05:
                max_gap_05 += 1
            elif gbest < 0.10:
                max_gap_10 += 1
            else:
                fail += 1
                continue
            anchors.append(ai); poss.append(pi); negs.append(int(cand[best]))
            stats.append(gbest)
            if len(anchors) % 4000 == 0:
                print(f"   built {len(anchors)}", flush=True)

        tri = {
            "anchor": torch.tensor(anchors, dtype=torch.long),
            "positive": torch.tensor(poss, dtype=torch.long),
            "negative": torch.tensor(negs, dtype=torch.long),
            "gap": torch.tensor(stats, dtype=torch.float32),
            "n": len(anchors),
            "frac_gap_lt_005": max_gap_05 / max(len(anchors), 1),
            "frac_gap_lt_010": max_gap_10 / max(len(anchors), 1),
            "n_rejected": fail,
        }
        torch.save(tri, ftri)
        print(f"[Triplets] {len(anchors)} built | gap<0.05: {max_gap_05} "
              f"| 0.05-0.10: {max_gap_10} | rejected: {fail}", flush=True)

    # per-style triplet coverage
    tri_cov = defaultdict(int)
    for ai in tri["anchor"].tolist():
        tri_cov[int(yn[ai])] += 1

    prep = {
        "n_classes": len(keep),
        "pool_images": len(pos_list),
        "n_triplets": int(tri["n"]),
        "frac_gap_lt_005": round(float(tri["frac_gap_lt_005"]), 4),
        "frac_gap_lt_010": round(float(tri["frac_gap_lt_010"]), 4),
        "n_rejected": int(tri["n_rejected"]),
        "gap_mean": round(float(tri["gap"].mean()), 4),
        "gap_p95": round(float(torch.quantile(tri["gap"].float(), 0.95)), 4),
        "mu_z_rms": round(float(mu_z.pow(2).mean().sqrt()), 4),
        "P_orth_err": float((P @ P.t() - torch.eye(K, device=DEV)).abs().max()),
        "proto_offdiag_mean": float(off.mean()),
        "triplets_per_style": {keep[i]: tri_cov.get(i, 0) for i in range(len(keep))},
        "classes": classes,
        "seed": SEED,
    }
    (OUT / "prep_stats.json").write_text(json.dumps(prep, indent=2), encoding="utf-8")

    # save index arrays for training/eval
    torch.save({"y": torch.from_numpy(yn), "a": torch.from_numpy(a),
                "g": torch.from_numpy(g), "pos_list": pos_list}, OUT / "index.pt")
    print(f"\n[Saved] {OUT}", flush=True)
    print(json.dumps({k: v for k, v in prep.items()
                      if k not in ("classes", "triplets_per_style")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
