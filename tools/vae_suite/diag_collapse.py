"""Diagnose the Style-Rank collapse: is it a stale coordinate centre?

For ckpt_u01000 compare the style geometry under
  (a) frozen A0 centre mu_z
  (b) batch-mean centre (recomputed on the eval set)
and report ||mean(h)|| / mean||h||, sim_pos / sim_neg, CM AUROC.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/Workspace/WEAVE_VAE")

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from sklearn.metrics import roc_auc_score

OUT = Path("/root/Workspace/WEAVE_VAE/runs/vae_style_rank20")
MANIFEST = Path("/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")
A0_DIR = "/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae"
DEV = "cuda"


class Reader:
    def __init__(self, raw):
        man = json.loads(MANIFEST.read_text())
        self.mmap = np.memmap(man["bin_path"], dtype=np.uint8, mode="r",
                              shape=tuple(man["shape"]))
        self.raw = list(raw)

    def get(self, idx, device):
        arr = np.stack([self.mmap[self.raw[i]] for i in idx])
        x = torch.from_numpy(arr).permute(0, 3, 1, 2).float().to(device)
        return x.div_(127.5).sub_(1.0)


def auroc(feats, pos, neg, cap=800):
    n = min(len(pos), len(neg), cap)
    sp = np.array(pos[:n]); sn = np.array(neg[:n])
    a1 = (feats[sp[:, 0]] * feats[sp[:, 1]]).sum(1)
    a0 = (feats[sn[:, 0]] * feats[sn[:, 1]]).sum(1)
    lbs = np.concatenate([np.ones(n), np.zeros(n)])
    return round(float(roc_auc_score(lbs, np.concatenate([a1, a0])) * 100), 2)


def pair_stats(u, y, a):
    rng = np.random.default_rng(42)
    sp, sn = [], []
    for i in range(len(y)):
        sc = np.where(y == y[i])[0]; sc = sc[sc != i]
        if len(sc) == 0:
            continue
        if a[i] >= 0:
            ca = sc[a[sc] != a[i]]
            if len(ca):
                sp.append((u[i] * u[int(rng.choice(ca))]).sum())
        dc = np.where(y != y[i])[0]
        if len(dc):
            sn.append((u[i] * u[int(rng.choice(dc))]).sum())
    return round(float(np.mean(sp)), 4), round(float(np.mean(sn)), 4)


def main():
    proj = torch.load(OUT / "projection.pt", weights_only=False)
    P = proj["P"].to(DEV).float()
    mu_z = proj["mu_z"].to(DEV).float()
    ev = torch.load(OUT / "eval_pairs.pt", weights_only=False)
    reader = Reader(ev["raw"])
    y, a, g = ev["y"], ev["a"], ev["g"]
    cm_pos, cm_neg = ev["cm_pos"], ev["cm_neg"]
    cross, diff = ev["cross"], ev["diff"]

    ck = torch.load(OUT / "checkpoints" / "ckpt_u01000.pt", map_location="cpu",
                    weights_only=False)
    vae = AutoencoderKL.from_pretrained(A0_DIR)
    vae.load_state_dict(ck["model"])
    vae.to(DEV).eval()

    Z = []
    with torch.no_grad():
        for i in range(0, len(ev["raw"]), 8):
            x = reader.get(list(range(i, min(i + 8, len(ev["raw"])))), DEV)
            Z.append(vae.encode(x).latent_dist.mean.float().flatten(1).cpu())
    Z = torch.cat(Z).to(DEV)

    h_frozen = F.linear(Z - mu_z, P)
    batch_mean = Z.mean(0, keepdim=True)
    h_centered = F.linear(Z - batch_mean, P)

    print("=== centre diagnostics (ckpt_u01000) ===")
    for tag, h in (("frozen mu_z", h_frozen), ("batch mean", h_centered)):
        m = h.mean(0)
        print(f"[{tag}] ||mean(h)||={m.norm().item():.2f} | mean||h||={h.norm(dim=1).mean().item():.2f} "
              f"| ratio={m.norm().item()/h.norm(dim=1).mean().item():.3f}")
        u = F.normalize(h, dim=-1).cpu().numpy()
        sp, sn = pair_stats(u, y, a)
        print(f"           sim_pos={sp:.4f} sim_neg={sn:.4f} margin={sp-sn:+.4f} "
              f"| CM AUROC={auroc(u, cm_pos, cm_neg)} Cross={auroc(u, cross, diff)}")

    # A0 reference with its own centre
    vae0 = AutoencoderKL.from_pretrained(A0_DIR).to(DEV).eval()
    Z0 = []
    with torch.no_grad():
        for i in range(0, len(ev["raw"]), 8):
            x = reader.get(list(range(i, min(i + 8, len(ev["raw"])))), DEV)
            Z0.append(vae0.encode(x).latent_dist.mean.float().flatten(1).cpu())
    Z0 = torch.cat(Z0).to(DEV)
    print("\n=== A0 reference ===")
    for tag, h in (("frozen mu_z", F.linear(Z0 - mu_z, P)),
                   ("own batch mean", F.linear(Z0 - Z0.mean(0, keepdim=True), P))):
        u = F.normalize(h, dim=-1).cpu().numpy()
        sp, sn = pair_stats(u, y, a)
        print(f"[{tag}] sim_pos={sp:.4f} sim_neg={sn:.4f} margin={sp-sn:+.4f} "
              f"| CM={auroc(u, cm_pos, cm_neg)} Cross={auroc(u, cross, diff)}")

    # how far did the latent distribution move?
    print("\n=== latent drift A0 -> ckpt1000 ===")
    print(f"  mean ||z||: A0={Z0.norm(dim=1).mean().item():.2f} "
          f"ckpt={Z.norm(dim=1).mean().item():.2f}")
    d = (Z.mean(0) - Z0.mean(0))
    print(f"  ||mean(z)_ckpt - mean(z)_A0|| = {d.norm().item():.2f}")
    print(f"  mean per-dim std: A0={Z0.std(0).mean().item():.4f} ckpt={Z.std(0).mean().item():.4f}")


if __name__ == "__main__":
    main()
