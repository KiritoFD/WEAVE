"""r2_eval adapter for V7R_NEXT arms (--gen v7r_next).

Keeps tools/v7r/arms.py and its ARM_CONFIGS READ-ONLY (low coupling).
build_v7r_pack() is reused for B4 / canon / critic; the edit function is
swapped to our constrained CPT editor or the distilled student.

Usage (from r2_eval):
    --gen v7r_next --v7r-next-arm T2|C1..C6|D1@step ...
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.v7r_next import common as C  # noqa: E402
from tools.v7r_next import patch_transport as PT  # noqa: E402
from tools.v7r_next import corrector as CO  # noqa: E402
from tools.v7r.arms import build_v7r_pack  # noqa: E402


def build_pack(args, device="cuda") -> dict:
    """Reuse V7-R's frozen model pack, add CPT bank / student / rho."""
    pack = build_v7r_pack(args, device)
    pack["kind"] = "v7r_next"
    pack["next_arm"] = args.v7r_next_arm
    bank = torch.load(PT.BANK, map_location="cpu", weights_only=False) \
        if PT.BANK.exists() else None
    pack["cpt_bank"] = bank
    rho_file = C.RUNS / "p2/rho_tiers.json"
    pack["rho_tiers"] = C.read_json(rho_file)["rho_tiers"] if rho_file.exists() else [0.08, 0.12]
    # student checkpoint for D* arms: "D1@1000" -> runs/v7r_next/p3/D1/step_01000.pt
    pack["student"] = None
    if "@" in args.v7r_next_arm:
        name, step = args.v7r_next_arm.split("@")
        p = C.RUNS / f"p3/{name}/step_{int(step):05d}.pt"
        if p.exists():
            from tools.v7r_next.distill import StudentCorrector
            ck = torch.load(p, map_location="cpu", weights_only=False)
            m = StudentCorrector(ck["rho"]).to(device).eval()
            m.load_state_dict(ck["model"])
            pack["student"] = m
    return pack


def run_edit(pack, z_batch, sid_batch, tid_batch, task_batch=None):
    """z_batch = SOURCE latents; produce edited latents from B4 start."""
    arm = pack["next_arm"]
    canon, b4, critic = pack["canon"], pack["b4"], pack["critic"]
    bank = pack["cpt_bank"]
    dev = z_batch.device
    with torch.no_grad():
        c = canon(z_batch.float(), mode=pack["canon_mode"], augment=False)
        z0 = b4(c, 0.0, c, tid_batch).detach()
    if arm.startswith("D") and pack["student"] is not None:
        # distilled student (inference-only, no CPT steps)
        with torch.no_grad():
            sid = torch.zeros_like(tid_batch)
            return pack["student"](c, z0, sid, tid_batch).detach()
    if arm in ("T0", "T1"):
        # history baselines are evaluated from their own eval.json; do not
        # regenerate here (keeps P5 index simple)
        return z0.detach()
    cfg = CO.arm_config(arm, pack["rho_tiers"])
    if cfg.get("kind") == "history":
        return z0.detach()
    if cfg.get("kind") == "r1_only":
        # fair control: same cumulative displacement as T5, original R1 only
        from tools.v7r.v7r_common import rms_normalize
        obj = __import__("tools.v7r.arms", fromlist=["make_style_objective"])\
              .make_style_objective("R1", pack, tid_batch)
        zr = z0.clone().requires_grad_(True)
        g = torch.autograd.grad(obj(zr), zr)[0]
        g = rms_normalize(g)
        return (z0 - cfg["rho"] * g).detach()
    # CPT editor (C1..C6 / T2..T5)
    if bank is None:
        raise RuntimeError("CPT bank missing; run patch_transport build first")
    q = PT.phi_geom(z_batch.float())
    ed = CO.ConstrainedEditor(
        z0, z_batch, tid_batch, canon, bank, q,
        mode=cfg["mode"], steps=cfg.get("steps", 4),
        rho=cfg.get("rho"), w_r1=cfg.get("w_r1", 0.0),
        w_struct=cfg.get("w_struct", 0.0), critic=critic,
        use_global=cfg.get("use_global", False))
    z, audit = ed.run()
    # per-batch trace (P2 requires per-step audit)
    if task_batch is not None:
        pack.setdefault("next_audit", []).append({
            "tasks": [t.get("task_id") for t in task_batch],
            "arm": arm, "audit": audit,
        })
    return z.detach()
