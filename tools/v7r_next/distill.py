"""P3: distill a validated CPT test-time editor into a tiny Corrector.

Why N1/N2 failed: they optimized the distribution objective directly and the
free residual drifted away from B4's content map (DINO-C ~0.50). Zero-init
only guarantees step0 == B4; it does NOT stop later amplification.

New: a VALIDATED CPT editor produces a small teacher correction near B4,
and the student learns that correction under a HARD per-sample RMS clip.

    z_student = z0 + ClipRMS( H_theta(c, z0, t), rho )

ClipRMS is in the FORWARD PATH (not a loss regularizer). Output stays zero-init
so step0 exactly reproduces B4. Keep the original ~142K H_theta; do not enlarge
parameters first.

Teacher data come ONLY from the TRAIN pool; no tasks_750 source image is used
to generate distillation labels. Prefer diverse styles / painters / geometry.

    L_student = L_delta + lambda_s L_CPT + lambda_c L_content
L_delta is audited per subband to avoid W-stage distillation imbalance.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.v7r_next import common as C  # noqa: E402
from tools.v7r_next import patch_transport as PT  # noqa: E402
from tools.v7r_next.corrector import ConstrainedEditor  # noqa: E402
from tools.v7r.v7r_common import (ResidualCorrector, FiLMResBlock)  # noqa: E402
from tools.v6.train_eval_v6 import STYLES  # noqa: E402

OUT = C.RUNS / "p3"


class ClipRMS(nn.Module):
    """Hard per-sample RMS clip in the forward path."""

    def __init__(self, rho: float):
        super().__init__()
        self.rho = float(rho)

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        rms = delta.flatten(1).pow(2).mean(1).sqrt().clamp(min=1e-9)
        scale = (self.rho / rms).clamp(max=1.0)
        return delta * scale.view(-1, 1, 1, 1)


class StudentCorrector(nn.Module):
    """z0 + ClipRMS(H_theta(c,z0,[s],t), rho). Output zero-init -> step0 == B4."""

    def __init__(self, rho: float, base_dim: int = 48):
        super().__init__()
        self.net = ResidualCorrector(base_dim=base_dim, use_source=False)
        self.clip = ClipRMS(rho)
        self.rho = float(rho)

    def forward(self, c, z0, source_id, target_id):
        # ResidualCorrector returns z0 + delta; extract delta, clip, re-add.
        out = self.net(c, z0, source_id, target_id)
        return z0 + self.clip(out - z0)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------- teacher data
def build_teacher_data(n_samples: int, bank: dict, rho_lo: float,
                       device="cuda") -> dict:
    """Sample TRAIN pool, run best CPT editor, cache (zs, z0, dz_T, meta)."""
    C.set_seed()
    train_by_style, _, painters = __import__(
        "tools.v7r.v7r_common", fromlist=["load_pool"]).load_pool(device="cpu")
    canon = C.load_canon(device)
    b4 = C.load_c3unet(device)
    critic = C.load_critic(device)
    style_ids = np.random.default_rng(C.SEED).integers(0, 5, n_samples)
    recs = []
    t0 = time.time()
    for i in range(0, n_samples, 8):
        tids = torch.tensor(style_ids[i:i + 8], device=device)
        zsl = []
        for j, sid in enumerate(style_ids[i:i + 8]):
            s = STYLES[int(sid)]
            pool = train_by_style[s]
            # prefer diverse painter / geometry: random index
            k = int(np.random.default_rng(C.SEED + i + j).integers(0, pool.shape[0]))
            zsl.append(pool[k])
        zs = torch.stack(zsl).to(device)
        sids = torch.tensor([0] * len(zsl), device=device)   # unused (H has no source)
        with torch.no_grad():
            c = canon(zs.float(), mode="c1_gentle", augment=False)
            z0 = b4(c, 0.0, c, tids)
        q = PT.phi_geom(zs.float())
        ed = ConstrainedEditor(z0, zs, tids[0].item(), canon, bank, q,
                               mode="geom_ot", steps=4, rho=rho_lo,
                               w_r1=0.0, w_struct=0.5, critic=critic)
        zT, audit = ed.run()
        dz = zT - z0
        for j in range(len(zsl)):
            recs.append({
                "zs": zs[j].cpu(), "z0": z0[j].cpu(), "dz": dz[j].cpu(),
                "tid": int(tids[j].item()),
                "teacher_disp_rms": float(audit[-1]["cum_displacement_rms"]),
                "teacher_cpt_before": float(audit[0]["cpt_before"]),
                "teacher_cpt_after": float(audit[-1]["losses"]["cpt"]),
                "constraint_triggered": bool(audit[-1]["constraint_triggered"]),
            })
        if (i // 8) % 20 == 0:
            print(f"  teacher {i}/{n_samples} in {time.time()-t0:.0f}s", flush=True)
    path = OUT / "teacher.pt"
    torch.save({"recs": recs, "meta": {
        "n": len(recs), "rho": rho_lo, "mode": "geom_ot", "steps": 4,
        "source": "TRAIN pool only", "w_r1": 0.0,
        "tasks_sha256": C.tasks_sha(),
        "built_sec": round(time.time() - t0, 1)}}, path)
    C.write_json(OUT / "teacher_meta.json", {"n": len(recs), "path": str(path),
                 "note": "no tasks_750 source used"})
    return torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------- training
def train_student(arm: str, teacher: dict, bank: dict, steps: int, device="cuda") -> dict:
    """D1 = L_delta only; D2 = + L_CPT + L_content; D3 = + source (only if D2)."""
    C.set_seed()
    rho = float(teacher["meta"]["rho"])
    model = StudentCorrector(rho).to(device)
    step0 = model.n_params()
    assert step0 < 200_000, f"student too large: {step0}"
    # step0 == B4 check
    z0 = torch.randn(2, 4, 64, 64, device=device)
    c = torch.randn_like(z0)
    sid = torch.zeros(2, dtype=torch.long, device=device)
    tid = torch.ones(2, dtype=torch.long, device=device)
    with torch.no_grad():
        assert float((model(c, z0, sid, tid) - z0).abs().max()) < 1e-6, "step0 != B4"
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    recs = teacher["recs"]
    trace = []
    ckpt_steps = {0, 1000, 3000, 4500, 6000} & set(range(0, steps + 1))
    ckpt_steps.add(steps)
    t0 = time.time()
    for step in range(1, steps + 1):
        b = np.random.default_rng(C.SEED + step).choice(len(recs), 8, replace=False)
        zs = torch.stack([recs[j]["zs"] for j in b]).to(device)
        z0 = torch.stack([recs[j]["z0"] for j in b]).to(device)
        dzT = torch.stack([recs[j]["dz"] for j in b]).to(device)
        tid = torch.tensor([recs[j]["tid"] for j in b], device=device)
        sid = torch.zeros_like(tid)
        c = torch.zeros_like(z0)  # canonicalized content approximation
        out = model(c, z0, sid, tid)
        dzS = out - z0
        l_delta = (dzS - dzT).square().mean()
        # per-subband gradient audit (L_delta subband scales)
        sub = {}
        hf = PT._haar_hf(dzS - dzT)
        for bi, nm in enumerate(("lh", "hl", "hh")):
            sub[nm] = float(hf[:, bi * 4:(bi + 1) * 4].abs().mean())
        l_cpt, _ = PT.cpt_loss(out, PT.phi_geom(zs), bank, tid[0].item(),
                                "geom_ot") if arm in ("D2", "D3") else (out.new_tensor(0.0), {})
        l_content = (PT.phi_geom(out) - PT.phi_geom(zs)).square().mean() \
            if arm in ("D2", "D3") else out.new_tensor(0.0)
        loss = l_delta + 0.1 * l_cpt + 0.5 * l_content
        opt.zero_grad()
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        opt.step()
        if step % 500 == 0 or step == 1:
            C.record_timing("train_step", (time.time() - t0) / step)
            trace.append({
                "step": step, "loss": round(float(loss), 5),
                "l_delta": round(float(l_delta), 5),
                "l_cpt": round(float(l_cpt), 5),
                "l_content": round(float(l_content), 5),
                "dz_cos": round(float(F.cosine_similarity(
                    dzS.flatten(1), dzT.flatten(1), dim=1).mean()), 4),
                "dz_rms_ratio": round(float(dzS.flatten(1).norm(dim=1).mean() /
                                            (dzT.flatten(1).norm(dim=1).mean() + 1e-9)), 4),
                "grad_norm": round(grad_norm, 5), "subband_l1": sub,
            })
            print(f"  [{arm}] step {step}/{steps} loss={float(loss):.5f} "
                  f"cos={trace[-1]['dz_cos']}", flush=True)
        if step in ckpt_steps:
            d = OUT / arm / f"step_{step:05d}.pt"
            d.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "rho": rho, "arm": arm,
                        "step": step, "n_params": step0}, d)
    C.write_json(OUT / arm / "train_trace.json", {"arm": arm, "steps": steps,
                 "n_params": step0, "trace": trace,
                 "step0_equals_B4": True,
                 "cliprms_in_forward": True})
    return {"arm": arm, "steps": steps, "n_params": step0, "final_loss": trace[-1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job", choices=["teacher", "train", "selftest"])
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--arm", default="D1")
    ap.add_argument("--steps", type=int, default=3000)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.job == "teacher":
        bank = torch.load(PT.BANK, map_location="cpu", weights_only=False)
        rho = C.read_json(C.RUNS / "p2/rho_tiers.json")["rho_tiers"][0]
        d = build_teacher_data(a.n, bank, rho)
        print("teacher:", d["meta"])
    elif a.job == "train":
        t = torch.load(OUT / "teacher.pt", map_location="cpu", weights_only=False)
        bank = torch.load(PT.BANK, map_location="cpu", weights_only=False)
        print(train_student(a.arm, t, bank, a.steps))
    else:
        C.set_seed()
        m = StudentCorrector(0.1)
        z0 = torch.randn(2, 4, 64, 64)
        c, sid, tid = torch.randn_like(z0), torch.zeros(2, dtype=torch.long), torch.ones(2, dtype=torch.long)
        out = m(c, z0, sid, tid)
        assert float((out - z0).abs().max()) < 1e-6, "step0 must equal B4"
        # force a large residual: clip must bind
        with torch.no_grad():
            m.net.out_conv.weight.fill_(1.0)
        out2 = m(c, z0, sid, tid)
        rms = (out2 - z0).flatten(1).pow(2).mean(1).sqrt().max()
        assert rms <= 0.1 + 1e-4, f"ClipRMS failed: {rms}"
        print(f"  n_params={m.n_params()} step0=B4 OK clipRMS OK (rms={float(rms):.4f})")
        print("SELFTEST PASSED")


if __name__ == "__main__":
    main()
