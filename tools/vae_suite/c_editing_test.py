"""C-stage only: real editing with data-derived style direction (A0, frozen).

Direction = (target style centroid - source style centroid), estimated from the
validation set with the source image itself excluded (no leakage).

Compares against random direction and CSD image-gradient direction at matched L2 norm.
Also dumps blind 4-panel images with shuffled panel order.
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

from tools.vae_suite.dataset import RawMemmapWikiArtDataset
import csd_model
import lpips

OUT = Path("/root/Workspace/WEAVE_VAE/runs_vae/v2_diagnosis")
BLIND = OUT / "blind_images"
BLIND.mkdir(parents=True, exist_ok=True)

MANIFEST = Path("/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")
A0_DIR = "/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae"
ALADIN_CKPT = "/tmp/aladin_repo/aladinVIT_bamfg_64.47.pt"
NORM_MEAN = [0.48145466, 0.4578275, 0.40821073]
NORM_STD = [0.26862954, 0.26130258, 0.27577711]


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


def main():
    t0 = time.time()
    device = "cuda"
    torch.manual_seed(0)
    rng = np.random.default_rng(17)

    ds = RawMemmapWikiArtDataset(MANIFEST, is_training=False)
    y, ar = [], []
    for i in range(len(ds)):
        rec = ds.records_by_idx[ds.indices[i]]
        y.append(rec["style_id"]); ar.append(rec.get("artist_id", -1))
    y = np.array(y); ar = np.array(ar)

    vae = AutoencoderKL.from_pretrained(A0_DIR).to(device).eval()
    for p in vae.parameters():
        p.requires_grad = False

    print("[Latents] A0 val ...", flush=True)
    zs = []
    with torch.no_grad():
        for i in range(0, len(ds), 16):
            idx = list(range(i, min(i + 16, len(ds))))
            x = torch.stack([ds[p]["image"] for p in idx]).to(device)
            zs.append(vae.encode(x).latent_dist.mean.float().cpu())
            if (i // 16 + 1) % 40 == 0:
                print(f"  {i+len(idx)}/{len(ds)}", flush=True)
    z_va = torch.cat(zs)                      # CPU

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

    by_style = defaultdict(list)
    for i in range(len(y)):
        by_style[int(y[i])].append(i)
    usable = [s for s, v in by_style.items() if len(v) >= 10]
    print(f"[Styles] usable={len(usable)}", flush=True)

    norm_fracs = [0.02, 0.05, 0.10]
    rows = {k: {nf: [] for nf in norm_fracs} for k in ("data_dir", "random_dir", "csd_grad")}

    n_tasks = 24
    tasks = []
    while len(tasks) < n_tasks:
        s = int(rng.choice(usable))
        t = int(rng.choice([u for u in usable if u != s]))
        src = int(rng.choice(by_style[s]))
        tgt_ref = int(rng.choice(by_style[t]))
        tasks.append((s, t, src, tgt_ref))

    for ti, (s, t, src, tgt_ref) in enumerate(tasks):
        x = ds[src]["image"].unsqueeze(0).to(device)
        x_t = ds[tgt_ref]["image"].unsqueeze(0).to(device)

        # leak-free centroids: exclude the source image from its own style centroid
        src_pool = [p for p in by_style[s] if p != src]
        tgt_pool = [p for p in by_style[t] if p != src]
        c_s = z_va[src_pool].mean(0).to(device)
        c_t = z_va[tgt_pool].mean(0).to(device)
        d_data = (c_t - c_s).flatten()
        d_data = d_data / (d_data.norm() + 1e-12)

        d_rand = torch.randn_like(d_data)
        d_rand = d_rand / (d_rand.norm() + 1e-12)

        with torch.no_grad():
            z0 = vae.encode(x).latent_dist.mean
            rec0 = vae.decode(z0).sample
            f_tgt = csd_feat(csd, x_t, nm, ns)
            al_tgt = aladin_feat(aladin, x_t, device)
            f_rec0 = csd_feat(csd, rec0, nm, ns)
            al_rec0 = aladin_feat(aladin, rec0, device)
        base_c = float((f_rec0 * f_tgt).sum())
        base_a = float((al_rec0 * al_tgt).sum())

        zr = z0.detach().clone().requires_grad_(True)
        fr = csd_feat(csd, vae.decode(zr).sample, nm, ns)
        g = torch.autograd.grad((fr * f_tgt).sum(), zr)[0].flatten()
        d_grad = g / (g.norm() + 1e-12)

        z_norm = float(z0.flatten().norm())
        for nf in norm_fracs:
            eps = nf * z_norm
            for key, d in (("data_dir", d_data), ("random_dir", d_rand), ("csd_grad", d_grad)):
                zn = (z0.flatten() + eps * d).view(1, 4, 64, 64)
                with torch.no_grad():
                    rec = vae.decode(zn).sample
                    fc = csd_feat(csd, rec, nm, ns)
                    ac = aladin_feat(aladin, rec, device)
                    rows[key][nf].append({
                        "dt_csd": float((fc * f_tgt).sum()) - base_c,
                        "dt_aladin": float((ac * al_tgt).sum()) - base_a,
                        "lpips": float(lpips_fn(rec.float(), rec0.float()).mean()),
                    })

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

        if (ti + 1) % 6 == 0:
            print(f"  [{ti+1}/{n_tasks}] elapsed={time.time()-t0:.0f}s", flush=True)

    res = {}
    for key in ("data_dir", "random_dir", "csd_grad"):
        res[key] = {}
        for nf in norm_fracs:
            rr = rows[key][nf]
            res[key][str(nf)] = {
                "mean_dt_csd": round(float(np.mean([r["dt_csd"] for r in rr])), 5),
                "pct_pos_csd": round(float(np.mean([r["dt_csd"] > 0 for r in rr]) * 100), 1),
                "mean_dt_aladin": round(float(np.mean([r["dt_aladin"] for r in rr])), 5),
                "pct_pos_aladin": round(float(np.mean([r["dt_aladin"] > 0 for r in rr]) * 100), 1),
                "mean_lpips": round(float(np.mean([r["lpips"] for r in rr])), 4),
            }

    out = {"n_tasks": n_tasks, "norm_fracs": norm_fracs, "results": res,
           "elapsed_sec": round(time.time() - t0, 1)}
    (OUT / "C_editing.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n" + "=" * 88)
    print("C: REAL EDITING (A0, direction from data centroids, no leakage)")
    print("=" * 88)
    print(f"{'dir':<12}{'norm':>6}{'dT_CSD':>10}{'%pos':>7}{'dT_ALADIN':>11}{'%pos':>7}{'LPIPS':>8}")
    print("-" * 88)
    for key in ("data_dir", "random_dir", "csd_grad"):
        for nf in norm_fracs:
            v = res[key][str(nf)]
            print(f"{key:<12}{nf:>6.2f}{v['mean_dt_csd']:>10.4f}{v['pct_pos_csd']:>7.1f}"
                  f"{v['mean_dt_aladin']:>11.4f}{v['pct_pos_aladin']:>7.1f}{v['mean_lpips']:>8.4f}")
        print("-" * 88)
    print(f"\n[Saved] {OUT/'C_editing.json'}  blind imgs -> {BLIND}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
