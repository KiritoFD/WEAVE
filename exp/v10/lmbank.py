"""V10.1 P0: multi-scale (content k, stroke v) banks from the real-painting pool.

Scales: 8x8 (64 patches/painting, fine) and 16x16 (16 patches/painting, coarse).
Banks come from the TRAIN split only; holdout is reserved for gate G1.
Run: python -m exp.v10.lmbank
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from .features import patchify as patchify8, q_feats
from .featv2 import v_feats
from .splits import POOL_DIR, STYLES, build_splits, check_integrity

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs/v10_local_4070"
CAP = {8: 200_000, 16: 50_000}
LATENT_CHUNK = 512
SEED = 0


def patchify16(z: torch.Tensor) -> torch.Tensor:
    """(B,4,64,64) -> (B*16,4,16,16)."""
    b = z.shape[0]
    return z.reshape(b, 4, 4, 16, 4, 16).permute(0, 2, 4, 1, 3, 5) \
            .reshape(b * 16, 4, 16, 16)


def extract_scale_feats(si: int, style: str, idx: torch.Tensor, device: str,
                        scale: int):
    d = torch.load(POOL_DIR / f"{si:02d}_{style}.pt",
                   map_location="cpu", weights_only=False)
    sel = d["latents"][idx]
    pf = patchify8 if scale == 8 else patchify16
    qs, vs = [], []
    for i in range(0, sel.shape[0], LATENT_CHUNK):
        z = sel[i:i + LATENT_CHUNK].float().to(device)
        p = pf(z)
        qs.append(q_feats(p).half().cpu())
        vs.append(v_feats(p).half().cpu())
    return torch.cat(qs), torch.cat(vs)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    (OUT / "bank").mkdir(exist_ok=True)
    t0 = time.time()
    splits = build_splits()
    rep = check_integrity(splits)
    print("split integrity:", json.dumps(rep), flush=True)

    raw = {8: {}, 16: {}}
    for si, s in enumerate(STYLES):
        for scale in (8, 16):
            q, v = extract_scale_feats(si, s, splits[s]["train_idx"], device, scale)
            # locate bad rows immediately (before any mixing)
            qbad = torch.nonzero(torch.isnan(q.float()).any(1)
                                 | torch.isinf(q.float()).any(1)).flatten()
            vbad = torch.nonzero(torch.isnan(v.float()).any(1)
                                 | torch.isinf(v.float()).any(1)).flatten()
            if qbad.numel() or vbad.numel():
                qcols = (torch.unique(torch.nonzero(
                    torch.isnan(q[qbad].float()) | torch.isinf(q[qbad].float()))[:, 1])
                    .tolist() if qbad.numel() else [])
                vcols = (torch.unique(torch.nonzero(
                    torch.isnan(v[vbad].float()) | torch.isinf(v[vbad].float()))[:, 1])
                    .tolist() if vbad.numel() else [])
                print(f"!! BAD {s} s{scale}: q rows {qbad.numel()} cols {qcols} | "
                      f"v rows {vbad.numel()} cols {vcols} -> sanitized to 0 "
                      f"(rare TF32 matmul edge on near-constant channels; logged)",
                      flush=True)
                q = torch.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
                v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            raw[scale][s] = (q, v)
            print(f"[feat] {s} s{scale}: {q.shape[0]} patches "
                  f"(q{q.shape[1]}/v{v.shape[1]}) {time.time()-t0:.0f}s", flush=True)

    bank = {"styles": list(STYLES), "scales": [8, 16], "Q": {}, "V": {},
            "stats": {}, "mu": {}, "meta": {"caps": CAP, "seed": SEED}}
    g = torch.Generator().manual_seed(SEED)
    for scale in (8, 16):
        Q_all = torch.cat([raw[scale][s][0] for s in STYLES]).float()
        V_all = torch.cat([raw[scale][s][1] for s in STYLES]).float()
        qm, qs_ = Q_all.mean(0), Q_all.std(0).clamp_min(1e-6)
        vm, vs_ = V_all.mean(0), V_all.std(0).clamp_min(1e-6)
        bank["stats"][scale] = {"q_mean": qm, "q_std": qs_,
                                "v_mean": vm, "v_std": vs_}
        del Q_all, V_all
        for s in STYLES:
            q = ((raw[scale][s][0].float() - qm) / qs_).half()
            v = ((raw[scale][s][1].float() - vm) / vs_).half()
            n = q.shape[0]
            take = torch.randperm(n, generator=g)[:min(n, CAP[scale])]
            bank["Q"][(s, scale)] = q[take]
            bank["V"][(s, scale)] = v[take]
            bank["mu"][(s, scale)] = v[take].float().mean(0).half()  # M0 target
            print(f"[bank] {s} s{scale}: {int(take.shape[0])} kept", flush=True)
    # serialize tuple keys as nested dicts
    ser = {"styles": bank["styles"], "scales": bank["scales"],
           "Q": {f"{s}|{sc}": t for (s, sc), t in bank["Q"].items()},
           "V": {f"{s}|{sc}": t for (s, sc), t in bank["V"].items()},
           "mu": {f"{s}|{sc}": t for (s, sc), t in bank["mu"].items()},
           "stats": {sc: bank["stats"][sc] for sc in bank["stats"]},
           "meta": bank["meta"]}
    torch.save(ser, OUT / "bank/lm_bank.pt")
    # save+reload verification: the bank must be NaN/inf-free on disk
    chk = torch.load(OUT / "bank/lm_bank.pt", map_location="cpu", weights_only=False)
    bad = []
    for k, t in list(chk["Q"].items()) + list(chk["V"].items()):
        tf = t.float()
        if torch.isnan(tf).any() or torch.isinf(tf).any():
            bad.append(k)
    for sc, st in chk["stats"].items():
        for kk, vv in st.items():
            if torch.isnan(vv).any() or torch.isinf(vv).any():
                bad.append(f"stats{sc}.{kk}")
    assert not bad, f"bank saved with bad tensors: {bad}"
    print("save+reload verification: CLEAN")
    print("saved", OUT / "bank/lm_bank.pt",
          (OUT / "bank/lm_bank.pt").stat().st_size, f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
