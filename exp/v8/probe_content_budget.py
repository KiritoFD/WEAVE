"""Calibrate the N2 content budget: measure edit-vs-generation displacement.

For a fixed task slice, reports per-sample
  d_edit  = mean((z' - z0)^2)          (the EDIT's own content cost)
  d_gen   = mean((z0 - z_s)^2)         (generation's shift from source)
  ratio   = d_edit / d_gen
under three editors: R1 oracle rho=0.15 (the r15 anchor), R1 rho=0.25,
and the trained L2 SPD head (L2.e protocol). Pick eps for the hinge so the
budget sits between r15 and r25 content cost.

Usage: python3 -m exp.v8.probe_content_budget --n 16
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def sq_mean(a, b):
    return (a - b).square().flatten(1).mean(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    from src.infer import canonicalizer, load_c3unet, load_critic
    from exp.v8.guidance import SPDHead, normalize, project_rms, target_margin

    device = a.device
    tasks = json.loads((REPO / "eval_protocol/tasks_750.json").read_text())["tasks"][:a.n]
    src = torch.load(REPO / "runs/v6_r2/cache/tasks750_source_features.pt",
                     map_location="cpu", weights_only=False)
    z = torch.cat([src["z_src"][t["source_stem"]] for t in tasks]).to(device)
    tid = torch.tensor([t["target_style_id"] for t in tasks], device=device)

    model, _ = load_c3unet(REPO / "runs/v6/P1_B4_distill50/P1_B4_distill50.pt", device)
    canon = canonicalizer(device)
    critic = load_critic(device)
    ck = torch.load(sorted((REPO / "runs/v8/train/L2").glob("step_*.pt"))[-1],
                    map_location="cpu", weights_only=False)
    head = SPDHead(rank=int(ck.get("rank", 0))).to(device).eval()
    head.load_state_dict(ck["head"])
    for p in head.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        c = canon(z.float(), mode="c1_gentle", augment=False)
        z0 = model(c, 0.0, c, tid).detach()
    d_gen = sq_mean(z0, z)

    def margin_of(zz):
        return target_margin(critic, zz.float(), tid)

    def r1_edit(rho, steps=1):
        cur = z0.clone()
        for _ in range(steps):
            with torch.enable_grad():
                zr = cur.detach().clone().requires_grad_(True)
                g = torch.autograd.grad(margin_of(zr).sum(), zr)[0]
            cur = project_rms(cur + (rho / steps) * normalize(g), z0, rho).detach()
        return cur

    def spd_edit():
        cur = z0.clone()
        for _ in range(2):
            with torch.enable_grad():
                zr = cur.detach().clone().requires_grad_(True)
                g = torch.autograd.grad(margin_of(zr).sum(), zr)[0].detach()
                diag, u = head(c, cur, tid, g)
                dg = diag * g
                ug = (u * g.unsqueeze(1)).sum(2)
                delta = dg + (u * ug.unsqueeze(2)).sum(1)
            cur = project_rms(cur + 0.08 * normalize(delta), z0, 0.25).detach()
        return cur

    editors = {"r1_r15": r1_edit(0.15), "r1_r25": r1_edit(0.25), "L2e": spd_edit()}
    print(f"{'editor':<8} {'d_edit':>9} {'d_gen':>9} {'ratio':>8} {'margin':>8}")
    print(f"{'(z0)':<8} {'0':>9} {d_gen.mean().item():>9.4f}")
    for name, zf in editors.items():
        de = sq_mean(zf, z0)
        m = margin_of(zf).mean().item()
        print(f"{name:<8} {de.mean().item():>9.4f} {d_gen.mean().item():>9.4f} "
              f"{(de / d_gen.clamp_min(1e-8)).mean().item():>8.4f} {m:>8.3f}")
    print(f"margin(z0)={margin_of(z0).mean().item():.3f}")


if __name__ == "__main__":
    main()
