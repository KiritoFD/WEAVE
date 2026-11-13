"""Experiment C: cross-painting generalization, confusion, density monotonicity.

1) 5-way confusion on holdout patches: argmax_t log p_t(v|q) for V10-B (f space)
   and V10-C (residual space), plus the R1 critic as a discriminative reference.
2) Density monotonicity on the attenuation ladder att03<att05<att07:
   per-painting Spearman(logp, alpha), mean over paintings (gate >= 0.9).
3) Conditional probe: for holdout patches of style t' whose content q is nearest
   to style t's bank, does conditioning keep them assigned to t'?
   (evidence that q-conditioning prevents content-driven style capture)

Run: python -m exp.v10.exp_c_generalize
Appends runs/v10_local_4070/p2_results.json (key "exp_c").
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .destyl import degrade
from .featfn import holdout_latents, load_bank, make_feat_fn
from .kde import QUERY_CHUNK, SIGMA, TAU, TOPK, _d2
from .splits import STYLES

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs/v10_local_4070"
CONF_PER_STYLE = 5000     # holdout patches per style for the confusion matrix
MON_N = 20                # paintings per style for the monotonicity ladder
PROBE_PAIRS_PER_STYLE = 200


def spearman3(y: torch.Tensor) -> float:
    """Spearman of a 3-point ladder (alpha increasing -> logp should DECREASE)."""
    y = y.detach().cpu().float()
    rx = torch.tensor([0.0, 1.0, 2.0])
    ry = torch.argsort(torch.argsort(y)).float()
    rx = rx - rx.mean(); ry = ry - ry.mean()
    d = (rx * ry).sum() / (rx.norm() * ry.norm()).clamp_min(1e-12)
    return d.item()


@torch.no_grad()
def logp_matrix(bank, q, v, space, cond):
    """(n,Dq),(n,Dv) -> (n,5) truncated log-density under each style bank."""
    out = []
    for ti in range(5):
        s = STYLES[ti]
        Vb = bank.V[space][s].float()
        Qb = bank.Q[s].float() if cond else None
        rows = []
        k = min(TOPK, Vb.shape[0])
        for i in range(0, v.shape[0], QUERY_CHUNK):
            dv = _d2(v[i:i + QUERY_CHUNK].float(), Vb)
            logits = -dv / (2 * SIGMA ** 2)
            if cond:
                dq = _d2(q[i:i + QUERY_CHUNK].float(), Qb)
                logits = logits - dq / (2 * TAU ** 2)
            rows.append(torch.logsumexp(logits, dim=1))
        out.append(torch.cat(rows))
    return torch.stack(out, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--conf-per-style", type=int, default=CONF_PER_STYLE)
    a = ap.parse_args()
    device = a.device
    bank, hpsi = load_bank(device)
    stats = {k: v.to(device) for k, v in bank.stats.items()}
    fn_b = make_feat_fn(stats, hpsi, "B")
    fn_c = make_feat_fn(stats, hpsi, "C")
    from src.infer import load_critic
    critic = load_critic(device)

    res = {"created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    t0 = time.time()

    # ---------------- 1) confusion matrices ----------------
    def confusion(space, cond, feat_fn):
        cm = torch.zeros(5, 5, dtype=torch.long)
        with torch.no_grad():
            for si, s in enumerate(STYLES):
                lat, _, _ = holdout_latents(si, max(200, a.conf_per_style // 64), seed=1)
                from .features import patchify, q_feats, f_feats
                p = patchify(lat.float().to(device))
                q_raw = q_feats(p)
                f_raw = f_feats(p)
                q = ((q_raw - stats["q_mean"]) / stats["q_std"])
                if space == "res":
                    v = ((f_raw - hpsi(q)) - stats["r_mean"]) / stats["r_std"]
                else:
                    v = (f_raw - stats["f_mean"]) / stats["f_std"]
                n = v.shape[0]
                take = torch.randperm(n)[:a.conf_per_style]
                L = logp_matrix(bank, q[take], v[take], space, cond)
                pred = L.argmax(1).cpu()
                true = torch.full_like(pred, si)
                for tt in range(5):
                    cm[si, tt] += (pred == tt).sum()
        return cm

    cm_b = confusion("f", True, fn_b)
    cm_c = confusion("res", True, fn_c)
    cm_r1 = torch.zeros(5, 5, dtype=torch.long)
    with torch.no_grad():
        for si, s in enumerate(STYLES):
            lat, _, _ = holdout_latents(si, max(64, a.conf_per_style // 64), seed=1)
            logits = critic(lat.float().to(device))
            pred = logits.argmax(1).cpu()
            true = torch.full_like(pred, si)
            for tt in range(5):
                cm_r1[si, tt] += (pred == tt).sum()
    acc = lambda cm: round((cm.diag().sum() / cm.sum()).item(), 4)
    res["confusion"] = {
        "v10_b": {"matrix": cm_b.tolist(), "acc": acc(cm_b)},
        "v10_c": {"matrix": cm_c.tolist(), "acc": acc(cm_c)},
        "r1_critic": {"matrix": cm_r1.tolist(), "acc": acc(cm_r1)},
    }
    print(f"[expC] confusion acc: B={acc(cm_b)} C={acc(cm_c)} r1={acc(cm_r1)} "
          f"{time.time()-t0:.0f}s", flush=True)

    # ---------------- 2) density monotonicity on att ladder ----------------
    mono = {"B": [], "C": []}
    for si, s in enumerate(STYLES):
        lat, _, _ = holdout_latents(si, MON_N, seed=2)
        lps = {"B": [], "C": []}
        for op in ("att03", "att05", "att07"):
            zm = degrade(lat.float().to(device), op)
            from .features import patchify, q_feats, f_feats
            p = patchify(zm)
            q_raw = q_feats(p)
            f_raw = f_feats(p)
            q = (q_raw - stats["q_mean"]) / stats["q_std"]
            v_f = (f_raw - stats["f_mean"]) / stats["f_std"]
            v_r = ((f_raw - hpsi(q)) - stats["r_mean"]) / stats["r_std"]
            lps["B"].append(bank.logp_per_patch(q, v_f, si, True, "f", "joint"))
            lps["C"].append(bank.logp_per_patch(q, v_r, si, True, "res", "joint"))
        for key in ("B", "C"):
            per = torch.stack(lps[key], 1)          # (N*64, 3)
            per = per.reshape(lat.shape[0], 64, 3).mean(1)   # per-painting mean logp
            mono[key].extend([spearman3(per[i]) for i in range(per.shape[0])])
    res["monotonicity_spearman"] = {k: round(sum(v) / len(v), 4) for k, v in mono.items()}
    print(f"[expC] monotonicity spearman: {res['monotonicity_spearman']} "
          f"{time.time()-t0:.0f}s", flush=True)

    # ---------------- 3) conditional probe (cross-style nearest-q) ----------------
    probe_hits_cond, probe_hits_uncond, n_probe = 0, 0, 0
    with torch.no_grad():
        for si_t in range(5):                    # bank style t (decoy)
            from .features import patchify, q_feats, f_feats
            bank_qs = bank.Q[STYLES[si_t]].float()
            for si_tp in range(5):
                if si_tp == si_t:
                    continue
                lat, _, _ = holdout_latents(si_tp, 4, seed=3)
                p = patchify(lat.float().to(device))
                q_raw = q_feats(p)
                q = ((q_raw - stats["q_mean"]) / stats["q_std"])
                f_raw = f_feats(p)
                v_f = (f_raw - stats["f_mean"]) / stats["f_std"]
                take = torch.randperm(q.shape[0])[:PROBE_PAIRS_PER_STYLE]
                q, v_f = q[take], v_f[take]
                # nearest bank-q of style si_t for each probe patch
                nn_idx = []
                for i in range(0, q.shape[0], QUERY_CHUNK):
                    d = _d2(q[i:i + QUERY_CHUNK], bank_qs)
                    nn_idx.append(d.argmin(1))
                nn_idx = torch.cat(nn_idx)
                # restricted probe: is the nearest bank patch style-consistent in logp?
                L = logp_matrix(bank, q, v_f, "f", True)         # (n,5) conditional
                Lu = logp_matrix(bank, q, v_f, "f", False)       # (n,5) unconditional
                probe_hits_cond += (L.argmax(1) == si_tp).sum().item()
                probe_hits_uncond += (Lu.argmax(1) == si_tp).sum().item()
                n_probe += q.shape[0]
    res["conditional_probe"] = {
        "n": n_probe,
        "acc_cond": round(probe_hits_cond / n_probe, 4),
        "acc_uncond": round(probe_hits_uncond / n_probe, 4),
        "note": "content-matched cross-style patches; higher=conditioning prevents capture",
    }
    print(f"[expC] conditional probe: {res['conditional_probe']} "
          f"{time.time()-t0:.0f}s", flush=True)

    p = OUT / "p2_results.json"
    cur = json.loads(p.read_text()) if p.exists() else {}
    cur["exp_c"] = res
    p.write_text(json.dumps(cur, indent=1))
    print("saved ->", p)


if __name__ == "__main__":
    main()
