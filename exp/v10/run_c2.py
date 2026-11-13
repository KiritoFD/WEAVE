"""V10.2 Experiment C: swap LD4's style direction for the v10.2 score variants.

LD4 machinery EXACTLY (SPDHead-preconditioned r1, v9 proxy acceptance
margin - 0.3*L_loc, v9 risk mask, backtracking) with the 'local' direction
REPLACED by:

  C-T    : target KDE direction            (= v10 E4's direction)
  C-TM   : target - mixture KDE (beta=1)   + absolute target-density constraint
  C-TS   : target - source KDE (beta=1)    + absolute target-density constraint

Run: python -m exp.v10.run_c2 --arm C-T --split all
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
OUT = REPO / "runs/v10_local_4070"
N2C_CKPT = REPO / "runs/v8/train/N2C/step_04000.pt"
ARMS = ("C-T", "C-TM", "C-TS")


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


def score_direction(vs, stats, z_leaf, t_idx, s_idx, arm, feat_fn):
    """Per-sample normalized direction for the arm's score (detached banks)."""
    from .features import patchify, q_feats, f_feats
    from .kde import SIGMA, TAU
    zi = z_leaf.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        p = patchify(zi)
        feats = {"q": (q_feats(p) - stats["q_mean"]) / stats["q_std"],
                 "f": (f_feats(p) - stats["f_mean"]) / stats["f_std"]}
        q, f = feats["q"], feats["f"]
        s_t = vs.styles[t_idx]
        Lt = vs._logp_rows(q, f, vs.Q[s_t], vs.V[s_t])
        if arm == "C-T":
            L = Lt
        else:
            s_s = vs.styles[s_idx if arm == "C-TS" else t_idx]
            Qo = vs.Q_mix if arm == "C-TM" else vs.Q[s_s]
            Vo = vs.V_mix if arm == "C-TM" else vs.V[s_s]
            Lo = vs._logp_rows(q, f, Qo, Vo)
            L = Lt - 1.0 * Lo
        g = torch.autograd.grad(L.sum(), zi)[0]
    flat = g.flatten(1)
    nrm = flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return (flat / nrm).view_as(z_leaf)


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

    from exp.v8.guidance import SPDHead, normalize, target_margin
    from exp.v9_local.local_style import LocalStyleEnergy
    from exp.v9_local.direction import DirectionComputer
    from exp.v9_local.solver import cost_tot as v9_cost
    from src.infer import load_critic
    from .featfn import load_bank
    from .scorevar import VariantScores
    from .optim import c_ll

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
    critic = load_critic(device)
    ck = torch.load(N2C_CKPT, map_location="cpu", weights_only=False)
    head = SPDHead(rank=int(ck.get("rank", 0))).to(device).eval()
    head.load_state_dict(ck["head"])
    for p in head.parameters():
        p.requires_grad_(False)
    lse = LocalStyleEnergy(str(REPO / "runs/v9_local_4070/bank/style_bank.pt"), device)
    dcomp = DirectionComputer(critic, head, lse, device)
    kde, hpsi = load_bank(device)
    stats = {k: v.to(device) for k, v in kde.stats.items()}
    vs = VariantScores(OUT, device)

    def proxy(zb, tvec):
        m = target_margin(critic, zb.float(), tvec)
        e = lse.loss(zb, tvec)
        return m - 0.3 * e

    def target_density(zb, tvec):
        """Absolute target log-density per sample (B,)."""
        from .features import patchify as pf, q_feats as qf, f_feats as ff
        p = pf(zb)
        q = (qf(p) - stats["q_mean"]) / stats["q_std"]
        f = (ff(p) - stats["f_mean"]) / stats["f_std"]
        d = []
        for i in range(zb.shape[0]):
            s_t = vs.styles[int(tvec[i])]
            d.append(vs._logp_rows(q[i * 64:(i + 1) * 64], f[i * 64:(i + 1) * 64],
                                   vs.Q[s_t], vs.V[s_t]).sum())
        return torch.stack(d)

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
        g_score = torch.zeros_like(z_n2)
        for i in range(z_n2.shape[0]):
            g_score[i:i + 1] = score_direction(vs, stats, z_n2[i:i + 1],
                                               int(tid[i]), int(sid[i]), a.arm, None)
        dirs = {"r1": dirs9["r1"], "score": g_score,
                "mix": normalize(0.5 * dirs9["r1"] + 0.5 * g_score)}
        mask = dcomp.region_mask(z_n2, z0, tid, 0.25)

        # v9 solve loop with optional absolute-density constraint for diff arms
        from exp.v8.guidance import project_rms
        z = z_n2.clone()
        infos = [{"steps": 0, "accepted": 0, "over": False} for _ in rows]
        for _ in range(2):
            p_cur = proxy(z, tid)
            d_cur = target_density(z, tid) if a.arm != "C-T" else None
            best_p, best_z = p_cur.clone(), z.clone()
            acc = torch.zeros(len(rows), dtype=torch.bool, device=device)
            for name, d in dirs.items():
                dm = d * mask
                for eta in (0.5, 1.0):
                    cand = project_rms(z + eta * dm, z0, 0.15)
                    feas = v9_cost(cand, z0, z_src) <= 0.10
                    if not feas.any():
                        continue
                    if a.arm != "C-T":
                        dok = target_density(cand[feas], tid[feas]) > d_cur[feas]
                        feas = feas.clone()
                        idx_f = feas.nonzero(as_tuple=True)[0]
                        feas[idx_f[~dok]] = False
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
        bad = v9_cost(z, z0, z_src) > 0.10
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
        bp = out_dir / f"batch_{bi:04d}.pt"
        torch.save(payload, bp.with_suffix(".tmp"))
        bp.with_suffix(".tmp").replace(bp)
        done.update(payload["ids"])
        done_path.write_text(json.dumps(sorted(done)))
        el = time.time() - t0
        accn = sum(x["accepted"] for x in infos)
        print(f"  {len(done)}/{len(idx_all)} accepted={accn} {el:.0f}s "
              f"({el / max(len(done), 1):.2f}s/img)", flush=True)
        bi += a.batch
    print(f"[{a.arm}/{a.split}] DONE {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
