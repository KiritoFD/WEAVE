"""L2/L3 SPD direction diagnostic.

For latent-space SPD unrolling (UnsupervisedUnroll space='latent'):
  g   = grad_z margin_R1(z,t)
  Pg  = P_theta g          (SPDHead, identity-init + trained)
  g'Pg must be >= 0 for ANY g if P is truly SPD — a negative value is a
  hard implementation bug (sign flip / non-SPD factorisation).

Also reports, per step:
  - directional derivative d margin/deta at eta=0 via finite difference
  - margin before/after the ACTUAL normalized+projected update (overshoot)

Usage: python3 -m exp.v8.check_spd_dir L2 --n 16
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def latest_ckpt(arm_id: str) -> Path:
    return sorted((REPO / "runs/v8/train" / arm_id).glob("step_*.pt"))[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", help="trained SPD arm id, e.g. L2 or L3")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    from src.infer import canonicalizer, load_c3unet, load_critic
    from exp.v8.arms import GROUPS
    from exp.v8.guidance import SPDHead, normalize, project_rms, target_margin

    arm = next(x for g in GROUPS.values() for x in g if x["id"] == a.arm)
    device = a.device
    ck = torch.load(latest_ckpt(a.arm), map_location="cpu", weights_only=False)
    rank = int(ck.get("rank", 0))
    head = SPDHead(rank=rank).to(device).eval()
    head.load_state_dict(ck["head"])
    for p in head.parameters():
        p.requires_grad_(False)

    base_path = REPO / "runs/v6" / (
        "P1_B4_distill50/P1_B4_distill50.pt" if arm.get("start") == "B4"
        else "Q2a_embed_style/Q2a_embed_style.pt")
    model, _ = load_c3unet(base_path, device)
    canon = canonicalizer(device)
    critic = load_critic(device)

    tasks = json.loads((REPO / "eval_protocol/tasks_750.json").read_text())["tasks"][:a.n]
    src = torch.load(REPO / "runs/v6_r2/cache/tasks750_source_features.pt",
                     map_location="cpu", weights_only=False)
    z = torch.cat([src["z_src"][t["source_stem"]] for t in tasks]).to(device)
    tid = torch.tensor([t["target_style_id"] for t in tasks], device=device)

    with torch.no_grad():
        c = canon(z.float(), mode="c1_gentle", augment=False)
        z0 = model(c, 0.0, c, tid).detach()

    def margin_of(zz):
        return target_margin(critic, zz.float(), tid)

    def spd_apply(diag, u, g):
        dg = diag * g
        ug = (u * g.unsqueeze(1)).sum(2)
        return dg + (u * ug.unsqueeze(2)).sum(1)

    # --- check 1: g'Pg >= 0 on the real gradient field ---
    zr = z0.detach().clone().requires_grad_(True)
    g = torch.autograd.grad(margin_of(zr).sum(), zr)[0]
    diag, u = head(c, z0, tid, g)
    pg = spd_apply(diag, u, g)
    gtp = (g * pg).sum((1, 2, 3))
    gn = g.flatten(1).norm(dim=1)
    print(f"[g'Pg] min={gtp.min().item():.6e} mean={gtp.mean().item():.6e} "
          f"rel_mean={(gtp / gn.clamp_min(1e-12)).mean().item():.6e}")

    # --- check 2: finite-diff directional derivative at eta=0 ---
    eps = 1e-3
    d = normalize(pg)
    with torch.no_grad():
        m_plus = margin_of(z0 + eps * d)
        m_minus = margin_of(z0 - eps * d)
    fd = ((m_plus - m_minus) / (2 * eps)).mean().item()
    print(f"[finite-diff d margin/deta along Pg dir] {fd:+.6f}  (expect >=0)")

    # --- check 3: actual K-step update, margin trajectory ---
    cur = z0.detach()
    margins = [float(margin_of(cur).mean())]
    for _ in range(int(arm.get("steps_k", 2))):
        zr = cur.detach().clone().requires_grad_(True)
        gk = torch.autograd.grad(margin_of(zr).sum(), zr)[0].detach()
        diag, u = head(c, cur, tid, gk)
        delta = spd_apply(diag, u, gk)
        cur = project_rms(cur + float(arm.get("step_rms", 0.08)) * normalize(delta),
                          z0, float(arm.get("rho", 0.25))).detach()
        margins.append(float(margin_of(cur).mean()))
    print(f"[margin trajectory] {['%.4f' % m for m in margins]}")

    verdict = "OK" if fd >= -1e-4 and margins[-1] >= margins[0] - 1e-3 else "SUSPECT"
    print(f"VERDICT {a.arm}: {verdict}")


if __name__ == "__main__":
    main()
