"""V10-E arm runner: KDE-score-driven constrained optimization on the N2C cache.

Arms (pre-registered plan §6, §10-P3):
  E1       k=1 step, V10-B direction, v10 score-gain mask 25%
  E2       k=2 steps, same
  E4       k=4 steps, same
  E2_nomask  k=2 steps without the region mask (mask ablation)

POST-HOC (labelled exploratory, added after the pre-registered dev gate verdict;
NOT gate-passing — see docs/v10/V10_RESULTS.md):
  E2_mix   v9-style shortlist {r1, v10_B, mix} with the v9 proxy acceptance
           (margin - 0.3*L_loc); tests complementarity (r1 drives Top-1,
           v10 drives DINO-S).

Latents out only (no VAE/metrics here). Resume-safe, same protocol as v9.
Run: python -m exp.v10.run_v10 --arm E2 --split dev
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .featfn import load_bank, make_feat_fn
from .optim import (CONTENT_GATE, EPS, optimize_batch, region_mask_v10,
                    sample_scores)
from .splits import STYLES

REPO = Path(__file__).resolve().parents[2]
TASKS = json.loads((REPO / "eval_protocol/tasks_750.json").read_text(encoding="utf-8"))["tasks"]
CACHE = REPO / "runs/v9_local_4070/cache/n2c_z.pt"
OUT = REPO / "runs/v10_local_4070"
ARMS = {"E1": 1, "E2": 2, "E4": 4, "E2_nomask": 2, "E2_mix": 2, "E2_ld4plus": 2}
POST_HOC_ARMS = {"E2_mix", "E2_ld4plus"}


def _stem_hash(stem: str) -> int:
    import hashlib
    return int(hashlib.md5(stem.encode()).hexdigest()[:8], 16)


def split_indices(name: str):
    """name: "all" (all 750 tasks — the decision-caliber per the 2026-11-14
    protocol correction) or the legacy v9 dev/test preview split."""
    if name == "all":
        return list(range(len(TASKS)))
    out = []
    for i, t in enumerate(TASKS):
        is_dev = _stem_hash(t["source_stem"]) % 5 == 0
        if is_dev == (name == "dev"):
            out.append(i)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--variant", default="B", choices=["A", "A_f", "B", "C"])
    ap.add_argument("--split", default="all", choices=["all", "dev", "test"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--mask-frac", type=float, default=0.25)
    ap.add_argument("--eps", type=float, default=EPS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    device = a.device
    k_steps = ARMS[a.arm]

    idx_all = split_indices(a.split)
    if a.limit:
        idx_all = idx_all[:a.limit]
    arm_name = f"{a.arm}_{a.variant}" if a.variant != "B" else a.arm
    out_dir = OUT / a.split / arm_name
    out_dir.mkdir(parents=True, exist_ok=True)
    done_path = out_dir / "done.json"
    done = set(json.loads(done_path.read_text())) if done_path.exists() and a.resume else set()
    todo = [i for i in idx_all if TASKS[i]["task_id"] not in done]
    print(f"[{arm_name}/{a.split}] {len(todo)}/{len(idx_all)} tasks batch={a.batch}")
    if not todo:
        return

    cache = torch.load(CACHE, map_location="cpu", weights_only=False)
    bank, hpsi = load_bank(device)
    stats = {k: v.to(device) for k, v in bank.stats.items()}
    feat_fn = make_feat_fn(stats, hpsi, a.variant)
    use_mask = a.arm != "E2_nomask"

    critic = None
    if a.arm == "E2_ld4plus":
        # minimal-delta complementarity test: v9 LD4 machinery EXACTLY
        # (SPDHead-preconditioned r1 + v9 local energy + v9 proxy acceptance),
        # with the V10 KDE direction added as one extra shortlist candidate.
        from src.infer import load_critic
        from exp.v8.guidance import SPDHead
        from exp.v9_local.local_style import LocalStyleEnergy
        from exp.v9_local.direction import DirectionComputer
        from exp.v9_local.solver import solve_batch as v9_solve_batch
        critic = load_critic(device)
        ck = torch.load(REPO / "runs/v8/train/N2C/step_04000.pt",
                        map_location="cpu", weights_only=False)
        head = SPDHead(rank=int(ck.get("rank", 0))).to(device).eval()
        head.load_state_dict(ck["head"])
        for p in head.parameters():
            p.requires_grad_(False)
        lse9 = LocalStyleEnergy(
            str(REPO / "runs/v9_local_4070/bank/style_bank.pt"), device)
        dcomp = DirectionComputer(critic, head, lse9, device)
    elif a.arm == "E2_mix":
        from src.infer import load_critic
        from exp.v8.guidance import normalize, project_rms as _pr, target_margin
        from .baselines import r1_direction
        from .optim import c_ll, cost_tot
        critic = load_critic(device)

        def proxy_mix(zb, tvec):
            m = target_margin(critic, zb.float(), tvec)
            s = sample_scores(bank, stats, hpsi, zb, tvec, a.variant)
            return m + 0.3 * (s / 64.0)

    t0 = time.time()
    bi = 0
    while bi < len(todo):
        rows = todo[bi:bi + a.batch]
        z_n2 = cache["z_n2"][rows].to(device)
        z0 = cache["z0"][rows].to(device)
        z_src = cache["z_src"][rows].to(device)
        sid = torch.tensor([TASKS[i]["source_style_id"] for i in rows], device=device)
        tid = torch.tensor([TASKS[i]["target_style_id"] for i in rows], device=device)
        mask = None
        if a.arm == "E2_ld4plus":
            mask = dcomp.region_mask(z_n2, z0, tid, a.mask_frac)
        elif use_mask:
            mask = region_mask_v10(bank, stats, hpsi, z_n2, tid, a.variant, a.mask_frac)
        if a.arm == "E2_ld4plus":
            with torch.enable_grad():
                dirs9 = dcomp.directions(z_n2, z0, tid, "mix")
            g_v10 = torch.zeros_like(z_n2)
            with torch.enable_grad():
                for i in range(z_n2.shape[0]):
                    g_v10[i:i + 1] = bank.direction(
                        z_n2[i:i + 1], int(tid[i]), a.variant, feat_fn,
                        samples_per_call=1)
            dirs9 = dict(dirs9)
            dirs9["v10"] = g_v10
            z_out, infos = v9_solve_batch(
                z_n2, z0, z_src, sid, tid, [dirs9], mask, critic, lse9,
                eps=a.eps, k_steps=k_steps)
        elif a.arm == "E2_mix":
            # post-hoc arm: shortlist {r1, v10, mix}, mixed acceptance proxy
            # (R1 margin + 0.3 * per-patch v10 score); budget + content gate
            # identical to the pre-registered arms.
            with torch.enable_grad():
                g_r1 = r1_direction(critic, z_n2, tid)
            g_v10 = torch.zeros_like(z_n2)
            with torch.enable_grad():
                for i in range(z_n2.shape[0]):
                    g_v10[i:i + 1] = bank.direction(
                        z_n2[i:i + 1], int(tid[i]), a.variant, feat_fn,
                        samples_per_call=1)
            g_mix = normalize(0.5 * g_r1 + 0.5 * g_v10)
            z = z_n2.clone()
            base_ll = c_ll(z_n2, z0)
            gate_ll = (1.2 * base_ll).clamp_min(1e-8)
            infos = [{"steps": 0, "accepted": 0, "over": False} for _ in rows]
            for _ in range(k_steps):
                p_cur = proxy_mix(z, tid)
                best_p = p_cur.clone()
                best_z = z.clone()
                acc = torch.zeros(len(rows), dtype=torch.bool, device=device)
                for name, d in (("r1", g_r1), ("v10", g_v10), ("mix", g_mix)):
                    dm = d if mask is None else d * mask
                    for eta in (0.5, 1.0):
                        cand = _pr(z + eta * dm, z0, 0.15)
                        feas = (cost_tot(cand, z0, z_src) <= a.eps) & \
                               (c_ll(cand, z0) <= gate_ll)
                        if not feas.any():
                            continue
                        p = proxy_mix(cand[feas], tid[feas])
                        bi2 = feas.nonzero(as_tuple=True)[0]
                        better = p > best_p[bi2]
                        if better.any():
                            sel = bi2[better]
                            best_z[sel] = cand[sel]
                            best_p[sel] = p[better]
                            acc[sel] = True
                for i in range(len(rows)):
                    if acc[i]:
                        z[i] = best_z[i]
                        infos[i]["steps"] += 1
                        infos[i]["accepted"] += 1
                    else:
                        infos[i]["over"] = infos[i]["steps"] == 0
                if not acc.any():
                    break
            z_out, infos_final = z, infos
            bad = cost_tot(z_out, z0, z_src) > a.eps
            if bad.any():
                z_out[bad] = z_n2[bad]
            infos = infos_final
        else:
            z_out, infos = optimize_batch(
                bank, stats, hpsi, z_n2, z0, z_src, sid, tid,
                variant=a.variant, feat_fn=feat_fn, mask=mask, k_steps=k_steps,
                eps=a.eps)
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
    print(f"[{arm_name}/{a.split}] DONE {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
