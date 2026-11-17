"""V12-GCL arm runner: LD4 machinery + Query-GCL content gates (plan §4-5).

Arms (pre-registered):
  G0S  LD4-strong  (eps .14, rho .20)          — expanded style budget control
  G1   LD4-base + Haar Query gate   (B .0274)
  G2O  LD4-base + old selfsim gate  (B .1667)  — frozen V5 control
  G2   LD4-base + relation gate     (B .00209, tau=2.0)
  G2S  LD4-strong + relation gate                — expanded budget under GCL

All arms keep the frozen LD4 machinery: v9 DirectionComputer shortlist
{r1, local, mix}, proxy acceptance (margin - 0.3*L_loc), v9 risk mask,
eta backtrack, k=2, existing eps/rho/C_LL gates; the Query gate is ADDITIONAL and ANCHORED TO z_0 (2026-11-14 amendment:
the source-anchored budget sits 2.2x BELOW the B4/N2C stylization distance and
rejects every candidate; z_0 anchoring preserves the no-new-damage semantics).
G0 = LD4 base (reused from the v9 merged-750 results, no rerun).

Run: python -m exp.v12_gcl.run_gcl --arm G2 --split all
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
TASKS = json.loads((REPO / "eval_protocol/tasks_750.json").read_text(encoding="utf-8"))["tasks"]
CACHE = REPO / "runs/v9_local_4070/cache/n2c_z.pt"
OUT = REPO / "runs/v12_gcl"
N2C_CKPT = REPO / "runs/v8/train/N2C/step_04000.pt"
BUDGETS = json.loads((REPO / "runs/v12_gcl/budgets.json").read_text())["budgets"]

ARMS = {
    "G0R": dict(eps=0.10, rho=0.15, gate=None),   # LD4 base (remote rebuild)
    "G0S": dict(eps=0.14, rho=0.20, gate=None),
    "G1": dict(eps=0.10, rho=0.15, gate="haar", B=BUDGETS["haar"]),
    "G2O": dict(eps=0.10, rho=0.15, gate="g2o", B=BUDGETS["g2o"]),
    "G2": dict(eps=0.10, rho=0.15, gate="rel", B=BUDGETS["rel"]),
    "G2S": dict(eps=0.14, rho=0.20, gate="rel", B=BUDGETS["rel"]),
}


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


def q_cost(arm, z, z_src):
    from .query_haar import haar_gcl_loss
    from .query_relation import g2o_selfsim_loss, relation_gcl_loss
    g = ARMS[arm]["gate"]
    tau = float(BUDGETS["chosen_tau"])
    if g == "haar":
        return haar_gcl_loss(z, z_src)
    if g == "g2o":
        return g2o_selfsim_loss(z, z_src)
    return relation_gcl_loss(z, z_src, tau)


def q_grad(z, z0, arm):
    """grad_z of the arm's Query loss at z vs the z_0 anchor (trace telemetry)."""
    from .query_haar import haar_query
    from .query_relation import _tokens16, _relations
    g = ARMS[arm]["gate"]
    zl = z.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        if g == "haar":
            q = haar_query(zl)
            qs = haar_query(z0.detach())
            L = torch.nn.functional.smooth_l1_loss(q, qs, reduction="none").mean(1).sum()
        elif g == "g2o":
            L = q_cost(arm, zl, z0).sum()
        else:
            tau = float(BUDGETS["chosen_tau"])
            t16z, t16s = _tokens16(zl), _tokens16(z0.detach())
            a_z = _relations(t16z, tau)
            a_s = _relations(t16s, tau)
            L = (a_s * (a_s.add(1e-10).log() - a_z.add(1e-10).log())).sum(-1).mean(1).sum()
        gg = torch.autograd.grad(L, zl)[0]
    return gg.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--split", default="all", choices=["all", "dev", "test"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    device = a.device
    cfg = ARMS[a.arm]

    idx_all = split_indices(a.split)
    if a.limit:
        idx_all = idx_all[:a.limit]
    out_dir = OUT / a.split / a.arm
    out_dir.mkdir(parents=True, exist_ok=True)
    done_path = out_dir / "done.json"
    done = set(json.loads(done_path.read_text())) if done_path.exists() and a.resume else set()
    todo = [i for i in idx_all if TASKS[i]["task_id"] not in done]
    print(f"[{a.arm}/{a.split}] {len(todo)}/{len(idx_all)} tasks batch={a.batch} "
          f"cfg={cfg}")
    if not todo:
        return

    cache = torch.load(CACHE, map_location="cpu", weights_only=False)

    from exp.v8.guidance import SPDHead, normalize, target_margin
    from exp.v9_local.local_style import LocalStyleEnergy
    from exp.v9_local.direction import DirectionComputer
    from src.infer import load_critic
    from exp.v8.guidance import project_rms

    critic = load_critic(device)
    ck = torch.load(N2C_CKPT, map_location="cpu", weights_only=False)
    head = SPDHead(rank=int(ck.get("rank", 0))).to(device).eval()
    head.load_state_dict(ck["head"])
    for p in head.parameters():
        p.requires_grad_(False)
    lse = LocalStyleEnergy(str(REPO / "runs/v9_local_4070/bank/style_bank.pt"),
                           device)
    dcomp = DirectionComputer(critic, head, lse, device)

    def proxy(zb, tvec):
        return target_margin(critic, zb.float(), tvec) - 0.3 * lse.loss(zb, tvec)

    def c_ll(zb, z0b):
        return (Fn.avg_pool2d(zb, 4) - Fn.avg_pool2d(z0b, 4)) \
            .square().flatten(1).mean(1)

    trace = []
    t0 = time.time()
    bi = 0
    while bi < len(todo):
        rows = todo[bi:bi + a.batch]
        z_n2 = cache["z_n2"][rows].to(device)
        z0 = cache["z0"][rows].to(device)
        z_src = cache["z_src"][rows].to(device)
        sid = torch.tensor([TASKS[i]["source_style_id"] for i in rows], device=device)
        tid = torch.tensor([TASKS[i]["target_style_id"] for i in rows], device=device)
        with torch.enable_grad():
            dirs9 = dcomp.directions(z_n2, z0, tid, "mix")
        mask = dcomp.region_mask(z_n2, z0, tid, 0.25)
        if cfg["gate"] is not None:
            gq = q_grad(z_n2, z0, a.arm)
            gmix = dirs9["mix"]
            cosv = torch.nn.functional.cosine_similarity(
                gmix.flatten(1), gq.flatten(1), dim=1)
            with torch.no_grad():
                d0 = q_cost(a.arm, z_n2, z0)
                ds = q_cost(a.arm, z_n2, z_src)
            trace.append({"cos_style_gq": [round(v, 4) for v in cosv.tolist()],
                          "q_cost_vs_z0_mean": round(d0.mean().item(), 6),
                          "q_cost_vs_src_mean": round(ds.mean().item(), 6)})

        # solve loop (v9 machinery) + Query gate
        z = z_n2.clone()
        base_ll = c_ll(z_n2, z0)
        gate_ll = (1.2 * base_ll).clamp_min(1e-8)
        infos = [{"steps": 0, "accepted": 0, "over": False} for _ in rows]
        for _ in range(2):
            p_cur = proxy(z, tid)
            best_p, best_z = p_cur.clone(), z.clone()
            acc = torch.zeros(len(rows), dtype=torch.bool, device=device)
            for name, d in dirs9.items():
                dm = d * mask
                for eta in (0.5, 1.0):
                    cand = project_rms(z + eta * dm, z0, cfg["rho"])
                    feas = (v9_cost(cand, z0, z_src) <= cfg["eps"]) & \
                        (c_ll(cand, z0) <= gate_ll)
                    if cfg["gate"] is not None:
                        qc = q_cost(a.arm, cand, z0)
                        feas = feas & (qc <= cfg["B"])
                    if not feas.any():
                        continue
                    p = proxy(cand[feas], tid[feas])
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
        bad = v9_cost(z, z0, z_src) > cfg["eps"]
        if bad.any():
            z[bad] = z_n2[bad]
        z_out = z
        same = sid == tid
        if same.any():
            z_out[same] = z_n2[same]
            for j in same.nonzero(as_tuple=True)[0].tolist():
                infos[j]["short"] = True
        payload = {"ids": [TASKS[i]["task_id"] for i in rows],
                   "z_src": z_src.cpu(), "z0": z0.cpu(),
                   "z_final": z_out.cpu(), "infos": infos}
        bp = out_dir / f"batch_{rows[0]:04d}.pt"   # task-index keyed: resume-safe
        torch.save(payload, bp.with_suffix(".tmp"))
        bp.with_suffix(".tmp").replace(bp)
        done.update(payload["ids"])
        done_path.write_text(json.dumps(sorted(done)))
        accn = sum(x["accepted"] for x in infos)
        print(f"  {len(done)}/{len(idx_all)} accepted={accn} {time.time()-t0:.0f}s",
              flush=True)
        bi += a.batch
    if trace:
        (out_dir / "trace_gq_cos.json").write_text(json.dumps(trace))
    print(f"[{a.arm}/{a.split}] DONE {time.time()-t0:.0f}s")


import torch.nn.functional as Fn  # noqa: E402


def v9_cost(z, z0, z_src):
    num = (z - z0).square().flatten(1).mean(1)
    den = (z0 - z_src).square().flatten(1).mean(1).clamp_min(1e-8)
    return num / den


if __name__ == "__main__":
    main()
