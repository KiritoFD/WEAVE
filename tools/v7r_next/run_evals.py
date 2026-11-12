"""Run all formal V7R_NEXT arm evaluations (full 750), in Gate order.

Called by run_v7r_next.sh after each stage, or standalone. Uses
`r2_eval --gen v7r_next` (our adapter) for new arms and `--gen v7r` for the
history anchors. Every arm writes runs/v7r_next/evals/<arm>/eval.json so P5's
eval_index picks them up.

Gate order:
  P1  C1..C6 ablations (C0 = original E3 history, read from runs/v7r)
  P2  T2..T6 (T0/T1 = R3c/R4c history)
  P3  D1@steps / D2@steps / D3@steps (only if teacher + ckpt exist)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.v7r_next import common as C  # noqa: E402

PY = "/usr/bin/python3"
EVALS = C.RUNS / "evals"


def run(arm: str, *, gen: str = "v7r_next", extra: list[str] | None = None) -> bool:
    out = EVALS / arm
    f = out / "eval.json"
    if f.exists():
        print(f"  [skip] {arm} already evaluated", flush=True)
        return True
    out.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "-m", "tools.v6_next.r2_eval", "--gen", gen,
           "--v7r-next-arm" if gen == "v7r_next" else "--v7r-arm", arm,
           "--ckpt", str(C.B4_CKPT.relative_to(C.ROOT)),
           "--out-dir", str(out.relative_to(C.ROOT)), "--batch-size", "8"]
    if extra:
        cmd += extra
    log = C.LOGS / f"eval_{arm}.log"
    t0 = time.time()
    print(f"  [eval] {arm} ...", flush=True)
    with open(log, "w") as lf:
        rc = subprocess.call(cmd, cwd=str(C.ROOT), stdout=lf, stderr=lf)
    if rc != 0:
        print(f"  [FAILED] {arm} rc={rc} ({log})", flush=True)
        C.write_json(out / "FAILED.json", {"arm": arm, "rc": rc})
        return False
    C.record_timing("eval750", time.time() - t0)
    d = C.read_json(f)
    c = d["cross600"]
    print(f"  [done] {arm} DINO-S {c['dino_s']:.4f} / DINO-C {c['dino_c']:.4f} "
          f"/ LPIPS {c['lpips']:.4f} in {time.time()-t0:.0f}s", flush=True)
    return True


def p1_gate() -> dict:
    """C2/C3 must beat C1 (uniform, no geometry) on target specificity."""
    def get(a, key):
        p = EVALS / a / "eval.json"
        if not p.exists():
            return None
        d = C.read_json(p)
        rows = [r for r in d.get("rows", []) if r.get("is_cross")]
        vals = [r.get("dino_s_specificity") for r in rows
                if "dino_s_specificity" in r]
        if key == "dino_s":
            return d["cross600"]["dino_s"]
        if key == "spec":
            return float(sum(vals) / len(vals)) if vals else None
        return d["cross600"].get(key)
    c1, c2, c3 = get("C1", "spec"), get("C2", "spec"), get("C3", "spec")
    ok = all(v is not None for v in (c1, c2, c3)) and (c2 > c1 or c3 > c1)
    r = {"C1_spec": c1, "C2_spec": c2, "C3_spec": c3,
         "geometry_valuable": bool(ok),
         "C1_dino_s": get("C1", "dino_s"),
         "C2_dino_s": get("C2", "dino_s"),
         "C3_dino_s": get("C3", "dino_s"),
         "note": "P1 Gate: if C2/C3 not > C1 on specificity, geometry "
                 "conditioning is NOT demonstrated; do not claim it."}
    C.write_json(C.RUNS / "p1/p1_gate.json", r)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["p1", "p2", "p3", "all", "gate"])
    a = ap.parse_args()
    EVALS.mkdir(parents=True, exist_ok=True)
    C.set_seed()
    if a.stage in ("p1", "all"):
        print("=== P1 C-ablations ===", flush=True)
        for arm in ("C1", "C2", "C3", "C4", "C5", "C6"):
            run(arm)
        print("P1 Gate:", p1_gate(), flush=True)
    if a.stage in ("p2", "all"):
        print("=== P2 T-arms ===", flush=True)
        for arm in ("T2", "T3", "T4", "T5", "T6"):
            run(arm)
    if a.stage in ("p3", "all"):
        print("=== P3 student checkpoints ===", flush=True)
        for name in ("D1", "D2", "D3"):
            for step in (0, 1000, 3000, 4500, 6000):
                ck = C.RUNS / f"p3/{name}/step_{step:05d}.pt"
                if ck.exists():
                    run(f"{name}@{step}")
    if a.stage == "gate":
        print("P1 Gate:", p1_gate(), flush=True)


if __name__ == "__main__":
    main()
