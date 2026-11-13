"""V10.1 arm runner: retrieval-target constrained optimization on the 750 cache.

Arms M0/M1/M2/M3 (see retrieval.py). Same budget/gate/backtrack contract as
V10-E4: eps=0.10, rho=0.15, content gate 1.2x base LL, eta backtrack
{1.0,0.5,0.25}, k=4 steps, identity short-circuit. Acceptance = the target
distance decreases. Mask = top-25% patches by current fine-scale distance.

Run: python -m exp.v10.lm_run --arm M2 --split all
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .optim import cost_tot, c_ll
from .retrieval import MODES, LocalBank, lm_direction, lm_style_loss
from exp.v8.guidance import project_rms

REPO = Path(__file__).resolve().parents[2]
TASKS = json.loads((REPO / "eval_protocol/tasks_750.json").read_text(encoding="utf-8"))["tasks"]
CACHE = REPO / "runs/v9_local_4070/cache/n2c_z.pt"
OUT = REPO / "runs/v10_local_4070"
K_STEPS = 4
ETAS = (1.0, 0.5, 0.25)
EPS = 0.10
GATE = 1.2


def _stem_hash(stem: str) -> int:
    import hashlib
    return int(hashlib.md5(stem.encode()).hexdigest()[:8], 16)


def split_indices(name: str):
    if name == "all":
        return list(range(len(TASKS)))
    out = []
    for i, t in enumerate(TASKS):
        is_dev = _stem_hash(t["source_stem"]) % 5 == 0
        if is_dev == (name == "dev"):
            out.append(i)
    return out


@torch.no_grad()
def distance_mask(lb: LocalBank, z: torch.Tensor, tid: torch.Tensor, mode: str,
                  frac: float = 0.25) -> torch.Tensor:
    """Top-frac patches by current fine-scale target distance -> soft mask."""
    _, per = lm_style_loss(lb, z, tid, mode)
    dp = per[8]                                        # (B,64)
    b = z.shape[0]
    k = max(1, int(round(frac * 64)))
    thr = dp.topk(k, dim=1).values[:, -1:]
    m = (dp >= thr).float().view(b, 1, 8, 8)
    m = torch.nn.functional.interpolate(m, size=(64, 64), mode="bilinear",
                                        align_corners=False)
    return (m > 0.5).float() * m


def optimize_batch(lb, z_n2, z0, z_src, sid, tid, mode, mask,
                   k_steps=K_STEPS, eps=EPS, gate=GATE):
    b = z_n2.shape[0]
    z = z_n2.clone()
    gate_ll = (GATE * c_ll(z_n2, z0)).clamp_min(1e-8)
    infos = [{"steps": 0, "accepted": 0, "over": False} for _ in range(b)]
    for _ in range(k_steps):
        g = lm_direction(lb, z, tid, mode)
        if mask is not None:
            g = g * mask
        # content-safe projection (same first-order rule as V10)
        from .optim import safe_direction
        g_safe, _ = safe_direction(g, z, z0)
        with torch.no_grad():
            cur, _ = lm_style_loss(lb, z, tid, mode)
        best_z, best_d = z.clone(), cur.clone()
        acc = torch.zeros(b, dtype=torch.bool, device=z.device)
        for eta in ETAS:
            cand = project_rms(z + eta * g_safe, z0, 0.15)
            feas = (cost_tot(cand, z0, z_src) <= eps) & (c_ll(cand, z0) <= gate_ll)
            if not feas.any():
                continue
            with torch.no_grad():
                d, _ = lm_style_loss(lb, cand[feas], tid[feas], mode)
            bi = feas.nonzero(as_tuple=True)[0]
            better = d < best_d[bi]
            if better.any():
                sel = bi[better]
                best_z[sel] = cand[sel]
                best_d[sel] = d[better]
                acc[sel] = True
        for i in range(b):
            if acc[i]:
                z[i] = best_z[i]
                infos[i]["steps"] += 1
                infos[i]["accepted"] += 1
            else:
                infos[i]["over"] = infos[i]["steps"] == 0
        if not acc.any():
            break
    bad = cost_tot(z, z0, z_src) > eps
    if bad.any():
        z[bad] = z_n2[bad]
        for i in bad.nonzero(as_tuple=True)[0].tolist():
            infos[i]["over"] = True
    return z, infos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(MODES))
    ap.add_argument("--split", default="all", choices=["all", "dev", "test"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--mask-frac", type=float, default=0.25)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    device = a.device

    idx_all = split_indices(a.split)
    if a.limit:
        idx_all = idx_all[:a.limit]
    out_dir = OUT / a.split / a.arm
    out_dir.mkdir(parents=True, exist_ok=True)
    done_path = out_dir / "done.json"
    done = set(json.loads(done_path.read_text())) if done_path.exists() and a.resume else set()
    todo = [i for i in idx_all if TASKS[i]["task_id"] not in done]
    print(f"[{a.arm}/{a.split}] {len(todo)}/{len(idx_all)} tasks batch={a.batch}")
    if not todo:
        return

    cache = torch.load(CACHE, map_location="cpu", weights_only=False)
    lb = LocalBank(OUT / "bank/lm_bank.pt", device)
    use_mask = a.arm != "M0"   # unconditional control runs without the mask

    t0 = time.time()
    bi = 0
    while bi < len(todo):
        rows = todo[bi:bi + a.batch]
        z_n2 = cache["z_n2"][rows].to(device)
        z0 = cache["z0"][rows].to(device)
        z_src = cache["z_src"][rows].to(device)
        sid = torch.tensor([TASKS[i]["source_style_id"] for i in rows], device=device)
        tid = torch.tensor([TASKS[i]["target_style_id"] for i in rows], device=device)
        mask = distance_mask(lb, z_n2, tid, a.arm, a.mask_frac) if use_mask else None
        z_out, infos = optimize_batch(lb, z_n2, z0, z_src, sid, tid, a.arm, mask)
        same = sid == tid
        if same.any():
            z_out[same] = z_n2[same]
            for j in same.nonzero(as_tuple=True)[0].tolist():
                infos[j]["short"] = True
        payload = {"ids": [TASKS[i]["task_id"] for i in rows],
                   "z_src": z_src.cpu(), "z0": z0.cpu(),
                   "z_final": z_out.cpu(), "infos": infos}
        bp = out_dir / f"batch_{bi:04d}.pt"
        torch.save(payload, bp.with_suffix(".tmp"))
        bp.with_suffix(".tmp").replace(bp)
        done.update(payload["ids"])
        done_path.write_text(json.dumps(sorted(done)))
        el = time.time() - t0
        acc = sum(x["accepted"] for x in infos)
        print(f"  {len(done)}/{len(idx_all)} accepted={acc} {el:.0f}s "
              f"({el / max(len(done), 1):.2f}s/img)", flush=True)
        bi += a.batch
    print(f"[{a.arm}/{a.split}] DONE {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
