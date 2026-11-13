"""Checkpoint-effectiveness gate.

Runs the SAME task slice through inference specs built from arm ids and
reports pairwise RMS differences of z_final. A trained-checkpoint arm whose
z_final is identical to its untrained counterpart means the checkpoint is not
being consumed — that arm must NOT enter formal evaluation.

Usage: python3 -m exp.v8.check_ckpt_effect A3.eval A2.s2 A4.eval --n 8
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def zfinal_of(arm_id: str, n: int, device: str, tmp: Path) -> torch.Tensor:
    from exp.v8.arms import GROUPS
    from exp.v8.specs import infer_spec_for
    from src.infer import InferSpec
    from src.infer.run import run as infer_run

    arm = next(a for g in GROUPS.values() for a in g if a["id"] == arm_id)
    spec = infer_spec_for(arm)
    out = tmp / arm_id
    infer_run(spec, out, tasks_file=REPO / "eval_protocol/tasks_750.json",
              src_cache=REPO / "runs/v6_r2/cache/tasks750_source_features.pt",
              batch=8, limit=n, decode=False, device=device)
    payloads = [torch.load(p, map_location="cpu", weights_only=False)
                for p in sorted(out.glob("batch_*.pt"))]
    return torch.cat([p["z_final"] for p in payloads])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    with tempfile.TemporaryDirectory(dir=REPO / "runs/v8") as td:
        tmp = Path(td)
        zs = {aid: zfinal_of(aid, a.n, a.device, tmp) for aid in a.arms}

    base = zs[a.arms[0]]
    print(f"{'arm':<12} {'rms_vs_first':>14}  verdict")
    ok = True
    for aid in a.arms:
        z = zs[aid]
        assert z.shape == base.shape, (aid, z.shape, base.shape)
        rms = float((z - base).flatten(1).pow(2).mean(1).sqrt().mean())
        same = rms < 1e-6
        if aid != a.arms[0] and same:
            ok = False
        print(f"{aid:<12} {rms:>14.6f}  {'IDENTICAL' if same else 'differs'}")
    print("GATE:", "PASS" if ok else "FAIL — checkpoint not consumed")


if __name__ == "__main__":
    main()
