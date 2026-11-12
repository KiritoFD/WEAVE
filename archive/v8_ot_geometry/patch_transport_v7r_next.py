"""P1: content-conditioned style Patch Transport (WEAVE-CPT).

Core hypothesis: style transfer should match the target school's PAINTING
EXPRESSION that has a similar STRUCTURAL FUNCTION to the source patch, not
match a real painting at the same spatial position.

Geometry-conditioned matching, NOT semantic understanding. Do not use
"semantic-aware" wording until semantic correspondence is validated.

Spatial alignment (IMPORTANT):
  phi_geom  : 8x8 patches on 64x64 latent  -> N = 64
  phi_style : 4x4 patches on 32x32 Haar HF -> N = 64
  Index i of both corresponds to the SAME 8x8 region of the source latent
  (Haar halves resolution, so a 4x4 HF patch == an 8x8 latent patch).
  Both must have identical N or the transport plan pi is undefined.

  phi_geom(z)  : fixed, training-free, from SOURCE latent, DETACHED.
                 multi-scale gradient orientation / structure tensor /
                 relative edge-strength rank / local self-similarity /
                 flat-edge-texture continuous memberships.
                 Does NOT preserve absolute edge magnitude.
  phi_style(z) : E3-validated Haar HF local patches (NOT E3-A/B Style Encoder).

  Bank  : Style -> Geometry Group -> Style Prototypes
          5 styles x 8 groups x 96 prototypes; painter IDs + original indices
          retained; TRAIN pool only; chunked (no full pairwise matrix).
  pi*   : fixed plan on geometry; source descriptors never receive gradient.
  L_CPT : sum_ij pi*_ij || phi_style(z)_i - a_j^t ||^2

Ablation modes (C0..C6) are selected here via `mode`:
  geom_soft  C2  geometry-gated soft-NN (default)
  geom_ot    C3  entropic OT group weights (chunked Sinkhorn, small)
  uniform    C1  no geometry (uniform pi), compressed bank
  geom_shuf  C5  geometry shuffled (negative control)
  wrong_tgt  C6  wrong target style (negative control)
C0 (original E3) and C4 (C3 + E3 global) live in corrector.py.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.v7r_next import common as C  # noqa: E402
from tools.v6.train_eval_v6 import STYLES  # noqa: E402

N_GROUPS = 8
PROTO_PER_GROUP = 96            # within 64-128
GEOM_PATCH = 8                  # 8x8 patch on 64x64
HF_PATCH = 4                    # 4x4 patch on 32x32 Haar HF
N_PATCH = 64                    # (64/8)^2 == (32/4)^2
GEOM_DIM = 16
STYLE_DIM = 192                 # 12 HF channels * 4 * 4
BANK = C.RUNS / "p1_bank.pt"
BANK_META = C.RUNS / "p1_bank_meta.json"
TAU_GEOM = 0.5                  # geometry group softmax temperature (fixed)
EPS_OT = 1.0                    # entropic OT reg (fixed)


# ---------------------------------------------------------------- descriptors
def _haar_hf(x: torch.Tensor) -> torch.Tensor:
    """2-D Haar HF subbands (LH/HL/HH) at half resolution -> (B,12,32,32)."""
    a = x[:, :, 0::2, 0::2]
    b = x[:, :, 1::2, 0::2]
    c = x[:, :, 0::2, 1::2]
    d = x[:, :, 1::2, 1::2]
    lh = (a - b + c - d) * .5
    hl = (a + b - c - d) * .5
    hh = (a - b - c + d) * .5
    return torch.cat([lh, hl, hh], dim=1)


def phi_style(z: torch.Tensor) -> torch.Tensor:
    """Style expression patches (B,N_PATCH,192) from Haar HF 4x4 patches."""
    hf = _haar_hf(z.float())                            # (B,12,32,32)
    p = F.unfold(hf, kernel_size=HF_PATCH, stride=HF_PATCH)   # (B,192,64)
    return p.permute(0, 2, 1)                           # (B,64,192)


def phi_geom(z: torch.Tensor) -> torch.Tensor:
    """Training-free geometry descriptor (B,N_PATCH,16) from SOURCE latent.

    Uses orientation / relative statistics, NOT absolute edge magnitude, so
    a target style may change stroke width or texture frequency. 8x8 patches.
    """
    x = z.float()
    B = x.shape[0]
    gx = F.pad(x[:, :, :, 1:] - x[:, :, :, :-1], (0, 1))
    gy = F.pad(x[:, :, 1:, :] - x[:, :, :-1, :], (0, 0, 0, 1))
    jxx = (gx * gx).mean(1, keepdim=True)
    jyy = (gy * gy).mean(1, keepdim=True)
    jxy = (gx * gy).mean(1, keepdim=True)
    tr = (jxx + jyy).clamp(min=1e-8)
    coh = ((jxx - jyy).square() + 4 * jxy.square()).sqrt() / tr       # 0..1
    ori = 0.5 * torch.atan2(2 * jxy, jxx - jyy) / np.pi              # -0.5..0.5
    edge = tr.sqrt()
    # relative edge-strength rank within image (scale invariant)
    e_flat = edge.flatten(1)
    rank = e_flat.argsort(dim=1).argsort(dim=1).float() / (e_flat.shape[1] - 1)
    rank = rank.view_as(edge)
    loc = F.avg_pool2d(ori, 5, stride=1, padding=2)
    lss = (ori - loc).abs()                                            # local self-sim
    mu = edge.mean(dim=(1, 2, 3), keepdim=True) + 1e-6
    r = edge / mu
    flat = torch.sigmoid((0.4 - r) * 4)
    text = torch.sigmoid((r - 1.6) * 3)
    edge_m = torch.clamp(1 - flat - text, 0, 1)
    feat = torch.cat([coh, ori, rank, lss, r, flat, edge_m, text], dim=1)  # (B,8,64,64)
    # 8x8 patches: mean + std per feature -> 16 dims
    p = F.unfold(feat, kernel_size=GEOM_PATCH, stride=GEOM_PATCH) \
         .reshape(B, 8, GEOM_PATCH * GEOM_PATCH, -1)                  # (B,8,64,64)
    m, sd = p.mean(2), p.std(2)                                        # (B,8,64)
    g = torch.cat([m, sd], dim=1).permute(0, 2, 1)                     # (B,64,16)
    g = (g - g.mean(1, keepdim=True)) / (g.std(1, keepdim=True) + 1e-6)
    return g


# ---------------------------------------------------------------- bank
def geometry_groups(q: torch.Tensor) -> torch.Tensor:
    """Fixed k-means on TRAIN geometry descriptors -> (N_GROUPS, GEOM_DIM)."""
    C.set_seed()
    idx = torch.randperm(q.shape[0])[:N_GROUPS]
    cent = q[idx].clone()
    for _ in range(12):
        a = torch.cdist(q, cent).square().argmin(1)
        for k in range(N_GROUPS):
            m = a == k
            if m.any():
                cent[k] = q[m].mean(0)
    return cent


def build_bank(device="cuda") -> dict:
    """Build Style -> Geometry Group -> Prototypes from TRAIN pool only.

    Chunked over latents; each latent contributes N_PATCH geometry+style
    patches. Never constructs a full all-patch pairwise distance matrix.
    """
    t0 = time.time()
    C.set_seed()
    all_q, all_a, all_sid = [], [], []
    per_style = {}
    for sid, s in enumerate(STYLES):
        d = torch.load(C.SD_POOL / f"{sid:02d}_{s}.pt", map_location="cpu",
                       weights_only=False)
        x = d["latents"].float()[100:]           # drop first 100 = TRAIN split
        ids = d.get("ids", list(range(d["latents"].shape[0])))[100:]
        per_style[s] = {"n_latents": int(x.shape[0]),
                        "n_painters": len(set(ids))}
        for i in range(0, x.shape[0], 16):
            xb = x[i:i + 16].to(device)
            with torch.no_grad():
                qb = phi_geom(xb).reshape(-1, GEOM_DIM).cpu()   # (16*64,16)
                ab = phi_style(xb).reshape(-1, STYLE_DIM).cpu()  # (16*64,192)
            all_q.append(qb)
            all_a.append(ab)
            all_sid.append(torch.full((qb.shape[0],), sid, dtype=torch.long))
    q = torch.cat(all_q, 0)
    a = torch.cat(all_a, 0)
    sid_all = torch.cat(all_sid, 0)
    cent = geometry_groups(q)
    assign = torch.cdist(q, cent).square().argmin(1)     # (M,)

    C.set_seed()
    bank = {"centroids": cent, "styles": {}, "meta": {
        "n_patches": int(q.shape[0]), "n_groups": N_GROUPS,
        "proto_per_group": PROTO_PER_GROUP, "n_patch_per_latent": N_PATCH,
        "geom_dim": GEOM_DIM, "style_dim": STYLE_DIM,
        "geom_patch": GEOM_PATCH, "hf_patch": HF_PATCH,
        "per_style": per_style, "tasks_sha256": C.tasks_sha(),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "build_sec": round(time.time() - t0, 1),
    }}
    for sid, s in enumerate(STYLES):
        bank["styles"][s] = []
        for g in range(N_GROUPS):
            pool = ((assign == g) & (sid_all == sid)).nonzero().flatten()
            if pool.numel() == 0:                      # fall back to style pool
                pool = (sid_all == sid).nonzero().flatten()
            take = pool[torch.randperm(pool.numel())[:PROTO_PER_GROUP]]
            if take.numel() < PROTO_PER_GROUP:
                take = torch.cat([take, take[:PROTO_PER_GROUP - take.numel()]])
            bank["styles"][s].append({"a": a[take].clone(),
                                      "idx": take.clone()})
    # overlap audit
    tasks = C.read_json(C.TASKS)
    refs = C.read_json(C.ROOT / "eval_protocol/references_v1.json")
    bank["meta"]["overlap_audit"] = {
        "n_eval_source_stems": len({t["source_stem"] for t in tasks["tasks"]}),
        "n_ref_stems": len({r["stem"] for v in refs["references"].values()
                            for r in v}),
        "train_pool_rule": "packed style latents [100:] only",
        "note": "bank built from TRAIN latents; no tasks_750 RGB; prototype "
                "indices retained for exact overlap check",
    }
    torch.save(bank, BANK)
    C.write_json(BANK_META, bank["meta"])
    return bank


# ---------------------------------------------------------------- matching + loss
def _group_weights(q: torch.Tensor, cent: torch.Tensor,
                   mode: str) -> torch.Tensor:
    """(B,N,G) geometry group weights, fixed plan (q is detached upstream)."""
    B, N, _ = q.shape
    d2 = torch.cdist(q.reshape(-1, GEOM_DIM), cent).square().reshape(B, N, N_GROUPS)
    if mode in ("geom_soft", "geom_shuf"):
        return torch.softmax(-d2 / TAU_GEOM, dim=-1)
    if mode == "geom_ot":
        # chunked entropic OT over GROUPS only (tiny B*N x 8 matrix, no giant
        # all-patch distance). Sinkhorn with uniform marginals.
        c = d2.reshape(-1, N_GROUPS)
        K = torch.exp(-c / EPS_OT)
        for _ in range(8):
            K = K / (K.sum(1, keepdim=True) + 1e-9)
            K = K / (K.sum(0, keepdim=True) + 1e-9)
        w = K / (K.sum(1, keepdim=True) + 1e-9)
        return w.reshape(B, N, N_GROUPS)
    if mode in ("uniform", "wrong_tgt"):
        return torch.full((B, N, N_GROUPS), 1.0 / N_GROUPS,
                          device=q.device, dtype=q.dtype)
    raise ValueError(mode)


def cpt_loss(z: torch.Tensor, q_src: torch.Tensor, bank: dict, style_id,
             mode: str = "geom_soft") -> tuple[torch.Tensor, dict]:
    """L_CPT = sum_ij pi*_ij || phi_style(z)_i - a_j^t ||^2.

    pi* is fixed on SOURCE geometry (q_src detached here); the loss gradient
    flows ONLY to z through phi_style. Negative controls:
      geom_shuf  C5  shuffle patch order of q
      wrong_tgt  C6  use (style_id+2)%5 bank
      uniform    C1  ignore geometry
    """
    B = z.shape[0]
    style_id = int(style_id) if not torch.is_tensor(style_id) else int(style_id.item())
    q = q_src.detach()
    if mode == "geom_shuf":
        q = q[:, torch.randperm(q.shape[1], device=q.device)]
    sid = (style_id + 2) % 5 if mode == "wrong_tgt" else style_id
    style = STYLES[sid]
    cent = bank["centroids"].to(z.device)
    A = torch.stack([bank["styles"][style][g]["a"].to(z.device)
                     for g in range(N_GROUPS)])            # (G,K,D)
    G, K, D = A.shape
    wg = _group_weights(q, cent, mode)                      # (B,N,G)
    pi = wg[..., None] / K                                  # (B,N,G,1)
    s = phi_style(z).reshape(B, N_PATCH, 1, 1, STYLE_DIM)
    a = A.view(1, 1, G, K, D)
    d2 = (s - a).square().sum(-1)                           # (B,N,G,K)
    loss = (pi * d2).sum(dim=(2, 3)).mean()
    return loss, {"mode": mode, "style": style, "n_groups": G, "n_proto": K,
                  "geom_mean_max_weight": round(float(wg.max(-1).values.mean()), 4),
                  "mean_d2": round(float(d2.mean()), 5)}


def cpt_global_loss(z: torch.Tensor, bank: dict, style_id) -> torch.Tensor:
    """C4: weak E3-style global HF distribution constraint (mean of style bank)."""
    sid = int(style_id) if not torch.is_tensor(style_id) else int(style_id.item())
    style = STYLES[sid]
    mu = torch.cat([bank["styles"][style][g]["a"] for g in range(N_GROUPS)]).mean(0)
    # per-image mean style vector over patches: (B, STYLE_DIM), NOT a scalar
    return (phi_style(z.float()).mean(1) - mu.to(z.device)).square().mean()


# ---------------------------------------------------------------- selftest
def selftest() -> None:
    C.set_seed()
    z = torch.randn(2, 4, 64, 64)
    q = phi_geom(z)
    s = phi_style(z)
    assert q.shape == (2, N_PATCH, GEOM_DIM), q.shape
    assert s.shape == (2, N_PATCH, STYLE_DIM), s.shape
    fake = {"centroids": torch.randn(N_GROUPS, GEOM_DIM),
            "styles": {st: [{"a": torch.randn(PROTO_PER_GROUP, STYLE_DIM),
                             "idx": torch.arange(PROTO_PER_GROUP)}
                            for _ in range(N_GROUPS)] for st in STYLES}}
    for mode in ("geom_soft", "geom_ot", "uniform", "geom_shuf", "wrong_tgt"):
        zz = z.clone().requires_grad_(True)
        l, info = cpt_loss(zz, q, fake, 0, mode)
        l.backward()
        assert torch.isfinite(l), (mode, l)
        assert zz.grad is not None and float(zz.grad.abs().sum()) > 0, mode
        print(f"  {mode:<10} loss={float(l):.4f} {info}")
    q2 = q.clone().requires_grad_(True)
    l2, _ = cpt_loss(z, q2, fake, 0)
    l2.backward()
    assert q2.grad is None or float(q2.grad.abs().sum()) < 1e-9, "geometry leaked grad"
    print("  geometry detached OK")
    print("SELFTEST PASSED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job", choices=["build", "selftest"])
    a = ap.parse_args()
    if a.job == "build":
        b = build_bank("cuda")
        print("bank built:", json.dumps(b["meta"], indent=1))
    else:
        selftest()


if __name__ == "__main__":
    import json
    main()
