"""P2: constrained TEST-TIME style optimization (WEAVE-CPT).

Fixed start z0 = B4(c(z_s), t). B4 is FROZEN, never retrained.

    min_z  L_CPT(z,t) + lambda_r L_R1(z,t) + lambda_c L_structure(z,z_s)
    s.t.   RMS(z - z0) <= rho

rho is NOT guessed: derived from the measured cumulative displacement of
R3c / R4c / R3c_disp (two fixed tiers). A "only lower RMS" fair control (T6)
is kept. Two levels:
    level 1  CPT only, 4 steps
    level 2  CPT + weak R1 (only if level 1 is effective)
Every step records CPT grad, R1 grad, their cosine, weighted-grad norm,
cumulative displacement, whether the objective decreased, constraint triggers.
This avoids V7's failure mode of unbalanced losses that never act.

Why NOT reuse tools/v7r/v7r_common.BudgetedEditor: its _step() updates from
`zr` (the requires_grad initial tensor) instead of the running `z_cur`, so
edits do not accumulate. We implement a small explicit loop here (low coupling).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.v7r_next import common as C  # noqa: E402
from tools.v7r_next import patch_transport as PT  # noqa: E402
from tools.v6.train_eval_v6 import STYLES  # noqa: E402

OUT = C.RUNS / "p2"
R1_WEAK = 0.25                  # fixed, not tuned
RHO_TIERS = None                 # set from audit at runtime


# ---------------------------------------------------------------- objectives
def r1_loss(critic, z, tid):
    """Ascent target: negative R1 target margin (we DESCEND it)."""
    q = critic(z.float()).float()
    oh = torch.nn.functional.one_hot(tid, 5).bool()
    margin = (q.gather(1, tid[:, None]).squeeze(1)
              - torch.logsumexp(q.masked_fill(oh, torch.finfo(q.dtype).min), 1))
    return -margin.mean()


def structure_loss(z, q_src):
    """Content proxy: geometry descriptor should not drift from source (detached)."""
    q = PT.phi_geom(z.float())
    return (q - q_src.detach()).square().mean()


def measure_rho_tiers() -> list[float]:
    """P2.2: rho from REAL cumulative displacement of R3c/R4c/R3c_disp.

    Reads their optim_trace.json; two fixed tiers = median and max-ish (p90).
    Never guesses rho.
    """
    vals = []
    for arm in ("R3c", "R4c", "R3c_disp", "R3c_noC"):
        p = C.V7R / f"p3_corrector/{arm}/optim_trace.json"
        if not p.exists():
            continue
        d = C.read_json(p)
        v = d.get("cum_displacement_rms") or d.get("final_displacement_rms")
        if isinstance(v, (int, float)):
            vals.append(float(v))
        elif isinstance(v, list):
            vals.extend([float(x) for x in v])
    if not vals:
        # fall back to the recorded per-step audit's last displacement
        for arm in ("R3c", "R4c", "R3c_disp"):
            p = C.V7R / f"p3_corrector/{arm}/optim_trace.json"
            if p.exists():
                d = C.read_json(p)
                for row in d if isinstance(d, list) else d.get("steps", []):
                    if "cum_displacement_rms" in row:
                        vals.append(float(row["cum_displacement_rms"]))
    vals = sorted(v for v in vals if np.isfinite(v) and v > 0)
    if not vals:
        raise RuntimeError("no real displacement data; cannot set rho")
    return [round(float(np.median(vals)), 5), round(float(np.percentile(vals, 90)), 5)]


class ConstrainedEditor:
    def __init__(self, z0, z_src, tid, canon, bank, q_src, *, mode="geom_soft",
                 steps=4, per_step_rms=0.08, rho=None, w_cpt=1.0, w_r1=0.0,
                 w_struct=0.0, critic=None, use_global=False):
        self.z0 = z0.detach()
        self.z_src = z_src
        self.tid = tid if torch.is_tensor(tid) else torch.tensor([tid], device=z0.device)
        self.canon, self.bank, self.q_src = canon, bank, q_src.detach()
        self.mode, self.steps, self.per_step_rms = mode, steps, per_step_rms
        self.rho, self.w_cpt, self.w_r1, self.w_struct = rho, w_cpt, w_r1, w_struct
        self.critic, self.use_global = critic, use_global
        self.c_src = canon(z_src.float(), mode="c1_gentle", augment=False).detach()
        self.audit = []

    def _losses(self, z):
        out = {}
        out["cpt"], _ = PT.cpt_loss(z, self.q_src, self.bank, self.tid[0].item(),
                                    self.mode)
        if self.use_global:
            out["global"] = PT.cpt_global_loss(z, self.bank, self.tid[0].item())
        if self.w_r1 > 0 and self.critic is not None:
            out["r1"] = r1_loss(self.critic, z, self.tid)
        if self.w_struct > 0:
            out["struct"] = structure_loss(z, self.q_src)
        return out

    def run(self):
        z = self.z0.clone()
        for step in range(self.steps):
            zr = z.clone().requires_grad_(True)
            losses = self._losses(zr)
            grads = {}
            for k, v in losses.items():
                g = torch.autograd.grad(v, zr, retain_graph=True)[0]
                grads[k] = g
            # step-0 fixed scale calibration (predefined, once)
            if step == 0:
                self.scales = {k: max(float(g.flatten(1).norm(dim=1).mean()), 1e-12)
                               for k, g in grads.items()}
            w = {"cpt": self.w_cpt, "global": 0.1 * self.w_cpt,
                 "r1": self.w_r1, "struct": self.w_struct}
            direction = zr.new_zeros(zr.shape)
            for k, g in grads.items():
                direction = direction + w.get(k, 0.0) * (-g / self.scales[k])
            d = direction / (direction.flatten(1).pow(2).mean(dim=1).sqrt()
                             .view(-1, 1, 1, 1) + 1e-12)
            f0 = sum(w.get(k, 0.0) * float(v.detach()) for k, v in losses.items())
            alpha_used, accepted, z_try = 0.0, False, z
            for a in (1.0, 0.5, 0.25, 0.125):
                z_try = (zr + self.per_step_rms * a * d).detach()
                ft = sum(w.get(k, 0.0) * float(v.detach())
                         for k, v in self._losses(z_try).items())
                if ft < f0:
                    alpha_used, accepted, z = a, True, z_try
                    break
            if not accepted:
                z = (zr + self.per_step_rms * 0.03125 * d).detach()
            # hard RMS constraint in the forward path
            disp = (z - self.z0).flatten(1).pow(2).mean(dim=1).sqrt()
            clipped = False
            if self.rho is not None:
                over = disp > self.rho
                if over.any():
                    scale = (self.rho / disp.clamp(min=1e-9)).clamp(max=1.0)
                    z = self.z0 + (z - self.z0) * scale.view(-1, 1, 1, 1)
                    clipped = True
                    disp = (z - self.z0).flatten(1).pow(2).mean(dim=1).sqrt()
            cos = None
            if "r1" in grads and "cpt" in grads:
                cos = round(float(torch.nn.functional.cosine_similarity(
                    grads["cpt"].flatten(1), grads["r1"].flatten(1), dim=1).mean()), 4)
            self.audit.append({
                "step": step,
                "cpt_grad_norm": round(float(grads["cpt"].flatten(1).norm(dim=1).mean()), 6),
                "r1_grad_norm": (round(float(grads["r1"].flatten(1).norm(dim=1).mean()), 6)
                                 if "r1" in grads else None),
                "grad_cos_cpt_r1": cos,
                "weighted_grad_norm": round(float(direction.flatten(1).norm(dim=1).mean()), 6),
                "alpha_used": alpha_used, "objective_decreased": accepted,
                "cpt_before": round(f0, 6),
                "cum_displacement_rms": round(float(disp.mean()), 6),
                "constraint_triggered": clipped,
                "losses": {k: round(float(v.detach()), 6) for k, v in losses.items()},
            })
        return z.detach(), self.audit


# ---------------------------------------------------------------- arm configs
def arm_config(name: str, rho: list[float]) -> dict:
    """T0..T6 + C-ablations. T6 = same cumulative displacement as T5, R1 only."""
    lo, hi = rho[0], rho[-1]
    cfg = {
        "T0": {"kind": "history", "arm": "R3c"},
        "T1": {"kind": "history", "arm": "R4c"},
        "T2": {"kind": "cpt", "mode": "geom_ot", "rho": lo, "w_r1": 0.0,
               "w_struct": 0.0, "steps": 4},
        "T3": {"kind": "cpt", "mode": "geom_ot", "rho": lo, "w_r1": 0.0,
               "w_struct": 0.5, "steps": 4},
        "T4": {"kind": "cpt", "mode": "geom_ot", "rho": lo, "w_r1": R1_WEAK,
               "w_struct": 0.0, "steps": 4},
        "T5": {"kind": "cpt", "mode": "geom_ot", "rho": hi, "w_r1": R1_WEAK,
               "w_struct": 0.5, "steps": 4},
        "T6": {"kind": "r1_only", "rho": hi, "steps": 4},   # fair RMS control
        # P1 ablations (4-step edits, same budget)
        "C1": {"kind": "cpt", "mode": "uniform", "rho": lo, "steps": 4},
        "C2": {"kind": "cpt", "mode": "geom_soft", "rho": lo, "steps": 4},
        "C3": {"kind": "cpt", "mode": "geom_ot", "rho": lo, "steps": 4},
        "C4": {"kind": "cpt", "mode": "geom_ot", "rho": lo, "steps": 4,
               "use_global": True},
        "C5": {"kind": "cpt", "mode": "geom_shuf", "rho": lo, "steps": 4},
        "C6": {"kind": "cpt", "mode": "wrong_tgt", "rho": lo, "steps": 4},
    }
    return cfg[name]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job", choices=["rho", "selftest"])
    a = ap.parse_args()
    if a.job == "rho":
        r = measure_rho_tiers()
        C.write_json(OUT / "rho_tiers.json", {"rho_tiers": r,
                     "source": "R3c/R4c/R3c_disp optim_trace.json"})
        print("rho tiers:", r)
    else:
        C.set_seed()
        bank = {"centroids": torch.randn(PT.N_GROUPS, PT.GEOM_DIM),
                "styles": {s: [{"a": torch.randn(8, PT.STYLE_DIM),
                                "idx": torch.arange(8)} for _ in range(PT.N_GROUPS)]
                           for s in STYLES}}
        z0 = torch.randn(2, 4, 64, 64)
        zs = torch.randn(2, 4, 64, 64)
        q = PT.phi_geom(zs)
        class _Canon:
            def __call__(self, z, mode=None, augment=False): return z
        for mode in ("geom_ot", "uniform", "geom_shuf", "wrong_tgt"):
            ed = ConstrainedEditor(z0, zs, 0, _Canon(), bank, q,
                                   mode=mode, steps=4, rho=0.1)
            z, audit = ed.run()
            assert z.shape == z0.shape
            assert (z - z0).flatten(1).pow(2).mean(1).sqrt().max() <= 0.1 + 1e-4
            assert len(audit) == 4
            print(f"  {mode:<10} final disp={audit[-1]['cum_displacement_rms']} OK")
        print("SELFTEST PASSED")


if __name__ == "__main__":
    main()
