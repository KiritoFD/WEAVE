"""V7-R P1-B: local real-painting patch banks + the E3/E4 patch energy.

Per style, from the SD latent pool (train split, style-balanced):
  bank_ll44 : 4x4 patches of the latent itself (d=64, stride 4)   [E4]
  bank_hf44 : 4x4 patches of (LH,HL,HH) (d=192, stride 4)         [E3/E4]
  bank_raw88: 8x8 raw-latent patches (d=256, stride 4, projected) [E4]
  ll stats  : per-style LL channel mean / covariance              [E4]

The E3/E4 energy is a smooth nearest-neighbour distance (P1-B "平滑最近邻"):
patches are z-scored against bank per-dim statistics, then
    loss = mean_i -tau*log sum_j exp(-||p_i-b_j||^2 / tau)
with tau=1.0 (predefined). Differentiable w.r.t. z; no per-position
registration, so matching never forces copying a target composition.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from tools.v7.v7_common import STYLES, write_json
from tools.v7r.v7r_common import P1_DIR, SD_POOL

N_LATENTS = 1500          # per style
PATCHES_PER_LATENT = 32   # random subsample per type
TAU = 1.0
PROJ_RAW88 = 64
PROJ_HF = 64
BANK_EVAL_MAX = 16384     # bank subsample used at optimization time


def _rand_proj(d_in: int, d_out: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(d_out, d_in, generator=g)
    return F.normalize(r, dim=1)


def extract_patches(z: torch.Tensor) -> dict:
    """z: (B,4,64,64) -> dict of patch tensors (n_total, d)."""
    out = {}
    ll, lh, hl, hh = None, None, None, None
    from wavelet import dwt2_haar
    ll, lh, hl, hh = dwt2_haar(z.float())
    out["ll44"] = F.unfold(ll, kernel_size=4, stride=4).permute(0, 2, 1).reshape(-1, 64)
    out["hf44"] = F.unfold(torch.cat([lh, hl, hh], dim=1), kernel_size=4,
                           stride=4).permute(0, 2, 1).reshape(-1, 192)
    out["raw88"] = F.unfold(z.float(), kernel_size=8, stride=4).permute(0, 2, 1).reshape(-1, 256)
    return out


def build_banks(device: str = "cuda") -> dict:
    proj_raw88 = _rand_proj(256, PROJ_RAW88, seed=11)
    proj_hf = _rand_proj(192, PROJ_HF, seed=12)
    banks = {"proj_raw88": proj_raw88, "proj_hf": proj_hf, "styles": {}}
    for sid, s in enumerate(STYLES):
        d = torch.load(SD_POOL / f"{sid:02d}_{s}.pt", map_location="cpu",
                       weights_only=False)
        x = d["latents"].float()[100:100 + N_LATENTS]
        acc = {"ll44": [], "hf44": [], "raw88": []}
        n_per_map = {"ll44": 64, "hf44": 64, "raw88": 225}
        for i in range(0, x.shape[0], 16):
            with torch.no_grad():
                p = extract_patches(x[i:i + 16].to(device))
            nb = x[i:i + 16].shape[0]
            for k in acc:
                n_per = p[k].shape[0] // nb
                assert n_per == n_per_map[k], (k, n_per)
                idx = torch.randint(0, n_per, (PATCHES_PER_LATENT * nb,),
                                    device=device)
                v = p[k][idx]
                if k == "raw88":
                    v = v @ proj_raw88.t().to(device)
                elif k == "hf44":
                    v = v @ proj_hf.t().to(device)
                acc[k].append(v.cpu())
        ll = x.to(device)
        mu = ll.mean(dim=(0, 2, 3))
        flat = ll.mean(dim=(2, 3))
        fc = flat - flat.mean(dim=0, keepdim=True)
        cov = (fc.t() @ fc) / max(fc.shape[0] - 1, 1)
        banks["styles"][s] = {
            "ll44": torch.cat(acc["ll44"], 0),
            "hf44": torch.cat(acc["hf44"], 0),
            "raw88": torch.cat(acc["raw88"], 0),
            "ll_mu": mu.cpu(), "ll_cov": cov.cpu(),
            "n_latents": int(x.shape[0]),
        }
    return banks


def _subsample(bank: torch.Tensor, cap: int) -> torch.Tensor:
    if bank.shape[0] <= cap:
        return bank
    idx = torch.randint(0, bank.shape[0], (cap,),
                        generator=torch.Generator().manual_seed(7))
    return bank[idx]


def bank_softnn_loss(z: torch.Tensor, tid: torch.Tensor, banks: dict,
                     use_ll: bool) -> torch.Tensor:
    """Smooth-NN distance from generated patches to the target style bank.

    Per style slice (batch usually single-style; multi-style batches are
    handled by splitting). Patches are standardized by bank per-dim stats.
    """
    from wavelet import dwt2_haar
    total = z.new_tensor(0.0)
    B = z.shape[0]
    for sid in torch.unique(tid):
        mask = tid == sid
        if not mask.any():
            continue
        st = banks["styles"][STYLES[int(sid)]]
        dev = z.device
        parts = []
        ll, lh, hl, hh = dwt2_haar(z[mask].float())
        hf = torch.cat([lh, hl, hh], dim=1)
        p_hf = F.unfold(hf, kernel_size=4, stride=4).permute(0, 2, 1).reshape(-1, 192)
        p_hf = p_hf @ banks["proj_hf"].t().to(dev)
        parts.append((p_hf, st["hf44"]))
        if use_ll:
            p_ll = F.unfold(ll, kernel_size=4, stride=4).permute(0, 2, 1).reshape(-1, 64)
            parts.append((p_ll, st["ll44"]))
            p_raw = F.unfold(z[mask].float(), kernel_size=8, stride=4).permute(0, 2, 1).reshape(-1, 256)
            p_raw = p_raw @ banks["proj_raw88"].t().to(dev)
            parts.append((p_raw, st["raw88"]))
        loss_s = z.new_tensor(0.0)
        for p, bank in parts:
            bank = _subsample(bank, BANK_EVAL_MAX).to(dev)
            mu = bank.mean(dim=0, keepdim=True)
            sd = bank.std(dim=0, keepdim=True).clamp(min=1e-4)
            pz = (p - mu) / sd
            bz = (bank - mu) / sd
            soft_blocks = []
            CH = 512
            for i in range(0, pz.shape[0], CH):
                d2 = torch.cdist(pz[i:i + CH], bz).square()   # (chunk, n_b)
                soft_blocks.append(-TAU * torch.logsumexp(-d2 / TAU, dim=1))
            loss_s = loss_s + torch.cat(soft_blocks).mean()
        total = total + loss_s * (mask.sum().float() / B)
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description="V7-R P1-B patch banks")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    t0 = time.time()
    P1_DIR.mkdir(parents=True, exist_ok=True)
    banks = build_banks(args.device)
    # audit: per-style bank separability (mean soft-NN of held-out real patches
    # to own vs other banks) using the val split
    audit = {"tau": TAU, "n_latents": N_LATENTS,
             "patches_per_latent": PATCHES_PER_LATENT, "per_style": {}}
    for sid, s in enumerate(STYLES):
        d = torch.load(SD_POOL / f"{sid:02d}_{s}.pt", map_location="cpu",
                       weights_only=False)
        zv = d["latents"].float()[:40].to(args.device)
        row = {}
        for t_name, t_sid in [(STYLES[k], k) for k in range(len(STYLES))]:
            with torch.no_grad():
                row[t_name] = float(bank_softnn_loss(
                    zv, torch.full((zv.shape[0],), t_sid, device=args.device),
                    banks, use_ll=True))
        own = row[s]
        others = [v for k, v in row.items() if k != s]
        auroc_like = sum(1 for v in others if own < v) / len(others)
        audit["per_style"][s] = {"own": own, "others": row,
                                 "own_closest_ratio": auroc_like}
    audit["separable"] = all(v["own_closest_ratio"] >= 0.75
                             for v in audit["per_style"].values())
    torch.save(banks, P1_DIR / "patch_banks.pt")
    write_json(audit, P1_DIR / "patch_bank_audit.json")
    print(f"[p1bank] done in {time.time()-t0:.0f}s separable={audit['separable']}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
