"""Style-Subspace Probe: can the full 4x64x64 latent be style-edited via a fixed 768-d subspace?

20-minute measurement on frozen A0 + frozen CSD. No training.

P0  Build orthonormal Q_s (768 x 16384), verify Q_s Q_s^T = I.
P1  32 source->target style tasks: g = grad_z S_CSD,target(D(z)),
    report r = ||P_s g||^2 / ||g||^2  (random expectation 768/16384 = 4.69%).
P2  Decode test at matched L2 perturbation: full grad / projected grad / random dir,
    3 norms each. Measure target-CSD gain vs no-op, ALADIN gain, LPIPS.
P3  Summary + 32 comparison image grids.

Outputs: style_subspace_probe.json, style_subspace_images/
"""
from __future__ import annotations

import json
import sys
import time
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

from tools.vae_suite.dataset import RawMemmapWikiArtDataset
import csd_model
import lpips

OUT_DIR = Path("/root/Workspace/WEAVE_VAE/runs_vae/v2_diagnosis")
IMG_DIR = OUT_DIR / "style_subspace_images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST = Path("/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")
A0_DIR = "/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae"
ALADIN_CKPT = "/tmp/aladin_repo/aladinVIT_bamfg_64.47.pt"

LAT_DIM = 4 * 64 * 64
SUB_DIM = 768
N_TASKS = 32
NORMS = [0.01, 0.03, 0.06]     # fraction of ||z0||

NORM_MEAN = [0.48145466, 0.4578275, 0.40821073]
NORM_STD = [0.26862954, 0.26130258, 0.27577711]


def load_aladin(device):
    from vit_networks import VisionTransformer

    class AViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc_style = VisionTransformer()

        def encode(self, image):
            proj, _ = self.enc_style(image, just_style_code=True)
            return proj

    ck = torch.load(ALADIN_CKPT, map_location="cpu", weights_only=False)
    m = AViT(); m.load_state_dict(ck); m.eval().to(device)
    for p in m.parameters():
        p.requires_grad = False
    return m


@torch.no_grad()
def aladin_feat(model, img, device):
    x224 = F.interpolate(img.clamp(-1, 1), (224, 224), mode="bicubic", align_corners=False)
    f = model.encode(x224.to(device).float())
    return F.normalize(f.float(), dim=-1)


def csd_feat(csd, img, nm, ns):
    x01 = (img.clamp(-1, 1) + 1.0) / 2.0
    x224 = F.interpolate(x01, (224, 224), mode="bicubic", align_corners=False)
    xin = (x224 - nm) / ns
    _, _, sty = csd(xin)
    return F.normalize(sty.float(), dim=-1)


def to_pil(img):
    x = (img.detach().clamp(-1, 1) + 1.0) / 2.0
    a = (x.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(a)


def main():
    t_start = time.time()
    device = "cuda"

    # ---------------- P0: subspace ----------------
    print("=== P0: build fixed 768-d orthonormal subspace ===", flush=True)
    g0 = torch.Generator().manual_seed(20260109)
    R = torch.randn(LAT_DIM, SUB_DIM, generator=g0)
    Q, _ = torch.linalg.qr(R)                 # (LAT_DIM, SUB_DIM) orthonormal columns
    Qs = Q.t().contiguous().to(device)        # (SUB_DIM, LAT_DIM) orthonormal rows
    I_check = Qs @ Qs.t()
    proj_err = float((I_check - torch.eye(SUB_DIM, device=device)).abs().max().item())
    print(f"  Qs shape={tuple(Qs.shape)} | max|Qs Qs^T - I| = {proj_err:.3e}", flush=True)
    print(f"  random-subspace expected energy fraction = {SUB_DIM/LAT_DIM*100:.2f}%", flush=True)

    # ---------------- models ----------------
    vae = AutoencoderKL.from_pretrained(A0_DIR).to(device).eval()
    for p in vae.parameters():
        p.requires_grad = False

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

    # ---------------- data: 32 cross-style tasks ----------------
    ds = RawMemmapWikiArtDataset(MANIFEST, is_training=False)
    style_to_idx = {}
    for i, idx in enumerate(ds.indices):
        rec = ds.records_by_idx[idx]
        style_to_idx.setdefault(rec["style_id"], []).append(i)
    usable = sorted([s for s, v in style_to_idx.items() if len(v) >= 20])
    rng = np.random.default_rng(7)

    tasks = []
    tries = 0
    while len(tasks) < N_TASKS and tries < 2000:
        tries += 1
        s_idx = int(rng.choice(len(ds)))
        rec = ds.records_by_idx[ds.indices[s_idx]]
        src_style = rec["style_id"]
        cands = [s for s in usable if s != src_style]
        if not cands:
            continue
        tgt_style = int(rng.choice(cands))
        tgt_pos = int(rng.choice(style_to_idx[tgt_style]))
        tasks.append({"src_pos": s_idx, "tgt_pos": tgt_pos,
                      "src_style": int(src_style), "tgt_style": tgt_style})
    print(f"\n[Tasks] {len(tasks)} cross-style tasks", flush=True)

    # ---------------- P1 + P2 ----------------
    p1_rows = []
    p2_rows = {k: {n: [] for n in NORMS} for k in ("full", "proj", "rand")}
    gallery = []

    for ti, tk in enumerate(tasks):
        x = ds[tk["src_pos"]]["image"].unsqueeze(0).to(device)
        x_tgt = ds[tk["tgt_pos"]]["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            z0 = vae.encode(x).latent_dist.mean
            rec0 = vae.decode(z0).sample
            f_rec0 = csd_feat(csd, rec0, nm, ns)
            f_tgt = csd_feat(csd, x_tgt, nm, ns)
            f_src = csd_feat(csd, x, nm, ns)
            al_rec0 = aladin_feat(aladin, rec0, device)
            al_tgt = aladin_feat(aladin, x_tgt, device)

        base_csd = float((f_rec0 * f_tgt).sum())
        base_aladin = float((al_rec0 * al_tgt).sum())

        # ---- P1: gradient of target CSD score wrt full latent ----
        z = z0.detach().clone().requires_grad_(True)
        f = csd_feat(csd, vae.decode(z).sample, nm, ns)
        score = (f * f_tgt).sum()
        g = torch.autograd.grad(score, z)[0]                  # (1,4,64,64)
        g_flat = g.flatten()

        Pg = (Qs.t() @ (Qs @ g_flat))                         # projected
        r = float((Pg.norm() ** 2 / (g_flat.norm() ** 2 + 1e-12)).item())

        p1_rows.append({"task": ti, "r": round(r, 6),
                        "g_norm": round(float(g_flat.norm()), 4),
                        "proj_norm": round(float(Pg.norm()), 4)})

        # ---- P2: decode with matched L2 perturbations ----
        g_dir = g_flat / (g_flat.norm() + 1e-12)
        p_dir = Pg / (Pg.norm() + 1e-12)
        r_dir = torch.randn_like(g_flat)
        r_dir = r_dir / (r_dir.norm() + 1e-12)

        z_norm = float(z0.flatten().norm())
        for nm_frac in NORMS:
            eps = nm_frac * z_norm
            for tag, d in (("full", g_dir), ("proj", p_dir), ("rand", r_dir)):
                z_new = (z0.flatten() + eps * d).view(1, 4, 64, 64)
                with torch.no_grad():
                    rec = vae.decode(z_new).sample
                    fr = csd_feat(csd, rec, nm, ns)
                    ar = aladin_feat(aladin, rec, device)
                    dt_csd = float((fr * f_tgt).sum()) - base_csd
                    dt_al = float((ar * al_tgt).sum()) - base_aladin
                    dt_src_drop = float((fr * f_src).sum()) - float((f_rec0 * f_src).sum())
                    lp = float(lpips_fn(rec.float(), rec0.float()).mean())
                p2_rows[tag][nm_frac].append({
                    "dt_csd": dt_csd, "dt_aladin": dt_al,
                    "src_style_drop": dt_src_drop, "lpips_vs_noop": lp,
                })

                if tag == "full" and nm_frac == NORMS[1]:
                    gallery.append({"task": ti, "src": x[0], "noop": rec0[0],
                                    "full": rec[0]})

        if (ti + 1) % 8 == 0:
            print(f"  [{ti+1}/{len(tasks)}] r={r*100:.2f}% | "
                  f"elapsed={time.time()-t_start:.0f}s", flush=True)

    # ---- extra: proj/rand panels for gallery (mid norm) ----
    print("\n[Gallery] rendering projected + random panels...", flush=True)
    for gi, gitem in enumerate(gallery):
        tk = tasks[gitem["task"]]
        x = ds[tk["src_pos"]]["image"].unsqueeze(0).to(device)
        x_tgt = ds[tk["tgt_pos"]]["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            z0 = vae.encode(x).latent_dist.mean
            f_tgt = csd_feat(csd, x_tgt, nm, ns)
        z = z0.detach().clone().requires_grad_(True)
        f = csd_feat(csd, vae.decode(z).sample, nm, ns)
        g = torch.autograd.grad((f * f_tgt).sum(), z)[0].flatten()
        Pg = Qs.t() @ (Qs @ g)
        eps = NORMS[1] * float(z0.flatten().norm())
        panels = [x[0], gitem["noop"], gitem["full"]]
        for d in (Pg / (Pg.norm() + 1e-12),):
            z_new = (z0.flatten() + eps * d).view(1, 4, 64, 64)
            with torch.no_grad():
                panels.append(vae.decode(z_new).sample[0])
        rd = torch.randn_like(g); rd = rd / (rd.norm() + 1e-12)
        z_new = (z0.flatten() + eps * rd).view(1, 4, 64, 64)
        with torch.no_grad():
            panels.append(vae.decode(z_new).sample[0])

        W, H = 512, 512
        canvas = Image.new("RGB", (W * len(panels), H))
        for pi, p in enumerate(panels):
            canvas.paste(to_pil(p).resize((W, H)), (pi * W, 0))
        canvas.save(IMG_DIR / f"task_{gi:02d}_src{tk['src_style']}_tgt{tk['tgt_style']}.jpg",
                    quality=92)

    # ---------------- P3: summary ----------------
    rs = np.array([row["r"] for row in p1_rows])
    summary = {
        "n_tasks": len(p1_rows),
        "subspace_dim": SUB_DIM,
        "latent_dim": LAT_DIM,
        "random_expectation_r": round(SUB_DIM / LAT_DIM, 6),
        "p1_gradient_energy_fraction": {
            "mean_r": round(float(rs.mean()), 6),
            "median_r": round(float(np.median(rs)), 6),
            "std_r": round(float(rs.std()), 6),
            "ratio_vs_random": round(float(rs.mean() / (SUB_DIM / LAT_DIM)), 3),
            "per_task": p1_rows,
        },
        "p2_decode": {},
        "elapsed_sec": round(time.time() - t_start, 1),
    }

    for tag in ("full", "proj", "rand"):
        summary["p2_decode"][tag] = {}
        for n in NORMS:
            rows = p2_rows[tag][n]
            summary["p2_decode"][tag][str(n)] = {
                "mean_dt_csd": round(float(np.mean([r["dt_csd"] for r in rows])), 5),
                "pct_pos_csd": round(float(np.mean([r["dt_csd"] > 0 for r in rows]) * 100), 1),
                "mean_dt_aladin": round(float(np.mean([r["dt_aladin"] for r in rows])), 5),
                "mean_src_style_drop": round(float(np.mean([r["src_style_drop"] for r in rows])), 5),
                "mean_lpips_vs_noop": round(float(np.mean([r["lpips_vs_noop"] for r in rows])), 4),
            }

    OUT_DIR.joinpath("style_subspace_probe.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    # ---------------- print table ----------------
    print("\n" + "=" * 92)
    print("P1: FRACTION OF CSD STYLE GRADIENT ENERGY INSIDE FIXED 768-d SUBSPACE")
    print("=" * 92)
    print(f"  mean r = {rs.mean()*100:.2f}%  (median {np.median(rs)*100:.2f}%, std {rs.std()*100:.2f}%)")
    print(f"  random-subspace expectation = {SUB_DIM/LAT_DIM*100:.2f}%")
    print(f"  ratio vs random = {rs.mean()/(SUB_DIM/LAT_DIM):.2f}x")

    print("\n" + "=" * 92)
    print("P2: DECODE RESPONSE (perturbation = frac * ||z0||)")
    print("=" * 92)
    print(f"{'dir':<8}{'norm':>7}{'dT_CSD':>11}{'%pos':>8}{'dT_ALADIN':>12}{'srcDrop':>10}{'LPIPS':>9}")
    print("-" * 92)
    for tag in ("full", "proj", "rand"):
        for n in NORMS:
            v = summary["p2_decode"][tag][str(n)]
            print(f"{tag:<8}{n:>7.2f}{v['mean_dt_csd']:>11.4f}{v['pct_pos_csd']:>8.1f}"
                  f"{v['mean_dt_aladin']:>12.4f}{v['mean_src_style_drop']:>10.4f}"
                  f"{v['mean_lpips_vs_noop']:>9.4f}")
        print("-" * 92)

    print(f"\n[Saved] {OUT_DIR/'style_subspace_probe.json'}")
    print(f"[Saved] {len(gallery)} gallery grids -> {IMG_DIR}")
    print(f"[Time] {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
