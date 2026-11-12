"""V7 P1: build real-painting style energies S1 (latent Haar/patch stats) and
S2 (frozen R1 intermediate feature stats) from the SD-VAE and A0 latent pools.

Outputs (docs/v7/V7_IMPLEMENTATION_PLAN.md P1.2):
  runs/v7/p1_energy/S1_sd.pt S1_a0.pt S2_sd.pt S2_a0.pt
  runs/v7/p1_energy/energy_audit.json

S1 reuses the V6 distribution-statistics implementation
(tools.v6.train_eval_v6.build_indomain_style_targets) as the required对照;
S2 is new: spatially-agnostic frozen-R1 feature statistics (NOT W1's Oracle
per-position fitting).
"""
from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from tools.v7.v7_common import P1_DIR, ROOT, STYLES, build_s2_style_stats, write_json

ASSET = ROOT.parent if ROOT.name != "WEAVE" else ROOT.parent
SD_POOL = Path("/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16")
A0_POOL = ROOT / "pseudo_gt_24h" / "orig_mean_fp16_a0"
VAL_N = 100          # matches V6: first 100 per style are validation
S2_PER_STYLE = 1500  # subsample bound for S2 statistics


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank-based AUROC for DISTANCES: positive class should have SMALLER values.

    Returns P(pos < neg) (same-style distance smaller than other-style distance).
    """
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    all_v = np.concatenate([pos, neg])
    ranks = np.argsort(np.argsort(all_v)).astype(np.float64) + 1.0
    r_pos = ranks[: len(pos)].sum()
    n_pos, n_neg = len(pos), len(neg)
    return float(1.0 - (r_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def load_pool(pool_dir: Path) -> tuple[dict, dict]:
    train_by_style, val_by_style = {}, {}
    for sid, s in enumerate(STYLES):
        d = torch.load(pool_dir / f"{sid:02d}_{s}.pt", map_location="cpu",
                       weights_only=False)
        x = d["latents"].float()
        val_by_style[s] = x[:VAL_N]
        train_by_style[s] = x[VAL_N:]
    return train_by_style, val_by_style


def build_s1(pool_dir: Path, tag: str, device: str) -> dict:
    """S1 via the V6 implementation; A0 builds into a temp cache dir so the
    frozen V6 cache (runs/v6/cache/indomain_style_targets_v2_k*.pt) is untouched."""
    import tools.v6.train_eval_v6 as tev

    train_by_style, _ = load_pool(pool_dir)
    if tag == "sd":
        targets = tev.build_indomain_style_targets(train_by_style, ROOT.parent,
                                                   device=device, top_k=64)
    else:
        orig_root = tev.V6_ROOT
        with tempfile.TemporaryDirectory() as td:
            tev.V6_ROOT = Path(td)
            try:
                targets = tev.build_indomain_style_targets(
                    train_by_style, ROOT.parent, device=device, top_k=64)
            finally:
                tev.V6_ROOT = orig_root
    return targets


def energy_scores_per_sample(kind: str, stats: dict, z: torch.Tensor,
                             tgt_ids: torch.Tensor, feat_ext, device: str,
                             batch: int = 100) -> np.ndarray:
    """(N,) per-sample energy distance to the style given in tgt_ids."""
    from tools.v7.v7_common import s1_loss_per_sample, s2_loss_per_sample
    out = []
    for i in range(0, z.shape[0], batch):
        zb = z[i:i + batch].to(device)
        ids = tgt_ids[i:i + batch].to(device)
        with torch.no_grad():
            if kind == "S1":
                v = s1_loss_per_sample(zb, ids, stats)
            else:
                v = s2_loss_per_sample(zb, ids, stats, feat_ext)
        out.append(v.detach().cpu().numpy())
    return np.concatenate(out, axis=0)


def audit_energy(kind: str, tag: str, stats: dict, val_by_style: dict,
                 feat_ext, device: str) -> dict:
    """Separability audit on validation samples: own-style distance vs other-style."""
    z_val, own = [], []
    for sid, s in enumerate(STYLES):
        z_val.append(val_by_style[s])
        own.extend([sid] * val_by_style[s].shape[0])
    z_val = torch.cat(z_val, dim=0)
    own_t = torch.tensor(own, dtype=torch.long)
    n = z_val.shape[0]
    dist_mat = np.zeros((n, 5), dtype=np.float64)
    for t in range(5):
        ids = torch.full((n,), t, dtype=torch.long)
        dist_mat[:, t] = energy_scores_per_sample(kind, stats, z_val, ids,
                                                  feat_ext, device)
    own_idx = np.arange(n)
    own_d = dist_mat[own_idx, own_t.numpy()]
    other_min = np.array([min(dist_mat[i, t] for t in range(5) if t != own_t[i])
                          for i in range(n)])
    aurocs = {}
    for sid, s in enumerate(STYLES):
        m = own_t.numpy() == sid
        pos = dist_mat[m, sid]
        neg = np.concatenate([dist_mat[m, t] for t in range(5) if t != sid])
        aurocs[s] = auroc(pos, neg)
    return {
        "kind": kind, "pool": tag,
        "auroc_mean": float(np.nanmean(list(aurocs.values()))),
        "auroc_per_style": aurocs,
        "intra_mean": float(own_d.mean()),
        "inter_mean_min": float(other_min.mean()),
        "margin_mean": float(other_min.mean() - own_d.mean()),
        "margin_median": float(np.median(other_min - own_d)),
        "n_val": int(n),
        "separable": bool(np.nanmean(list(aurocs.values())) > 0.5),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="V7 P1 energy construction + audit")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-s1", action="store_true")
    ap.add_argument("--skip-s2", action="store_true")
    args = ap.parse_args()
    device = args.device
    t0 = time.time()
    P1_DIR.mkdir(parents=True, exist_ok=True)

    from tools.v6_next.w_distill import R1FeatureExtractor
    from tools.v7.v7_common import load_critic
    critic = load_critic(device=device)
    feat_ext = R1FeatureExtractor(critic)

    pools = {"sd": SD_POOL, "a0": A0_POOL}
    audits = []
    for tag, pool_dir in pools.items():
        if not pool_dir.exists():
            print(f"[p1] pool missing: {pool_dir}; skipping {tag}", flush=True)
            continue
        train_by_style, val_by_style = load_pool(pool_dir)
        if not args.skip_s1:
            print(f"[p1] building S1_{tag} from {pool_dir}", flush=True)
            s1 = build_s1(pool_dir, tag, device)
            torch.save(s1, P1_DIR / f"S1_{tag}.pt")
            audits.append(audit_energy("S1", tag, s1, val_by_style, feat_ext, device))
        if not args.skip_s2:
            print(f"[p1] building S2_{tag} from {pool_dir}", flush=True)
            s2 = build_s2_style_stats(feat_ext, train_by_style,
                                      per_style=S2_PER_STYLE, device=device)
            torch.save(s2, P1_DIR / f"S2_{tag}.pt")
            audits.append(audit_energy("S2", tag, s2, val_by_style, feat_ext, device))

    write_json({
        "generated_by": "tools/v7/p1_energy.py",
        "wall_sec": round(time.time() - t0, 1),
        "val_per_style": VAL_N,
        "s2_per_style": S2_PER_STYLE,
        "pools": {"sd": str(SD_POOL), "a0": str(A0_POOL)},
        "audits": audits,
        "note": "A0 pools are对照 statistics only; optimization on generated SD "
                "latents may only use *_sd energies.",
    }, P1_DIR / "energy_audit.json")
    print(f"[p1] done in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
