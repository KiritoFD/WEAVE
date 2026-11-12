"""Run one V8 arm end to end: generate (GPU) then measure (CPU), at full 750.

Usage:
  python3 -m exp.v8.run_arm --arm A2.s2
  python3 -m exp.v8.run_arm --arm L0.r25 --limit 32

Writes to runs/v8/<arm>/{gen/, eval/, state.json}. The arm -> inference mapping
lives in exp/v8/specs.py, so this file never grows a second model registry.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from exp.v8.arms import GROUPS
from exp.v8.specs import infer_spec_for

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "runs/v8"


def find_arm(arm_id: str) -> dict:
    for group, arms in GROUPS.items():
        for a in arms:
            if a["id"] == arm_id:
                return {**a, "group": group}
    raise KeyError(f"unknown arm {arm_id!r}")


def state_path(arm_id: str) -> Path:
    return RUNS / arm_id / "state.json"


def write_state(arm_id: str, **kw):
    p = state_path(arm_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    cur = json.loads(p.read_text()) if p.exists() else {}
    cur.update(kw, updated_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    p.write_text(json.dumps(cur, indent=2, ensure_ascii=False), encoding="utf-8")


def run_arm(arm_id: str, *, limit: int = 0, device: str = "cuda",
            batch: int = 8, skip_gen: bool = False) -> dict:
    arm = find_arm(arm_id)
    out = RUNS / arm_id
    gen_dir, eval_dir = out / "gen", out / "eval"
    write_state(arm_id, state="RUNNING", arm=arm, started_at=time.time())
    t0 = time.time()

    if not skip_gen:
        from src.infer.run import run as infer_run
        from src.infer import InferSpec
        spec = infer_spec_for(arm)
        meta = infer_run(spec, gen_dir, tasks_file=REPO / "eval_protocol/tasks_750.json",
                         src_cache=REPO / "runs/v6_r2/cache/tasks750_source_features.pt",
                         batch=batch, limit=limit, decode=True, device=device)
        write_state(arm_id, gen_sec=meta["wall_sec"], n_tasks=meta["n_tasks"])

    from src.eval.measure import measure
    rows = measure(gen_dir, eval_dir, device=device, write_png=True, limit=limit)
    write_state(arm_id, eval_sec=round(time.time() - t0, 1), n_rows=len(rows),
                state="DONE")
    return {"arm": arm_id, "rows": len(rows), "sec": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-gen", action="store_true")
    a = ap.parse_args()
    try:
        r = run_arm(a.arm, limit=a.limit, device=a.device, batch=a.batch,
                    skip_gen=a.skip_gen)
        print("ARM_DONE", json.dumps(r), flush=True)
    except Exception as exc:
        write_state(a.arm, state="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
