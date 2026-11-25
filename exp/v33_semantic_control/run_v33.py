"""Single Entrypoint for CLEAVE V33 Quality Frontier Breakthrough Suite.

Supports running smoke test or any stage (p0, p1, p2, p3, all) with error isolation
so failure of one branch is logged without blocking independent experiments.
"""
from __future__ import annotations

import argparse
import traceback

from exp.v33_semantic_control import OUT
from exp.v33_semantic_control.p0_diagnose import run_p0
from exp.v33_semantic_control.p1_q64_adapter import run_p1
from exp.v33_semantic_control.p2_semantic_control import run_p2
from exp.v33_semantic_control.p3_joint_verify import run_p3
from src.utils.runtime import sanitize_cuda_env, save_json, utc


def main() -> None:
    ap = argparse.ArgumentParser(description="CLEAVE V33 Unified Runner")
    ap.add_argument("--stage", choices=("smoke", "p0", "p1", "p2", "p3", "all"), default="all")
    ap.add_argument("--use-q64", action="store_true")
    ap.add_argument("--force-p3", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    sanitize_cuda_env()
    OUT.mkdir(parents=True, exist_ok=True)
    status: dict = {"updated_utc": utc(), "stage": args.stage, "results": {}, "errors": {}}

    if args.stage == "smoke":
        status["results"]["p1_smoke"] = run_p1(smoke=True, device=args.device)
        status["results"]["p2_smoke"] = run_p2(use_q64=args.use_q64, smoke=True, device=args.device)
        save_json(OUT / "v33_smoke_report.json", status)
        print("[V33 SMOKE PASSED]", flush=True)
        return

    stages_to_run = ["p0", "p1", "p2", "p3"] if args.stage == "all" else [args.stage]
    for st in stages_to_run:
        try:
            if st == "p0":
                status["results"]["p0"] = run_p0(device=args.device)
            elif st == "p1":
                status["results"]["p1"] = run_p1(smoke=False, device=args.device)
            elif st == "p2":
                use_q64 = args.use_q64
                if st == "p2" and "p1" in status["results"]:
                    g = status["results"]["p1"].get("gate", {})
                    if g.get("Q1_q64_plus", {}).get("pass_gate", False):
                        use_q64 = True
                status["results"]["p2"] = run_p2(use_q64=use_q64, smoke=False, device=args.device)
            elif st == "p3":
                status["results"]["p3"] = run_p3(force=args.force_p3, device=args.device)
        except Exception as exc:
            err_msg = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            status["errors"][st] = err_msg
            print(f"[V33 ERROR in {st}] {type(exc).__name__}: {exc}", flush=True)
            if st == "p2":
                print("[V33] Skipping P3 because P2 failed.", flush=True)
                break

    status["updated_utc"] = utc()
    save_json(OUT / "v33_run_status.json", status)
    if status["errors"]:
        print(f"[V33 FINISHED WITH ERRORS] {list(status['errors'].keys())}", flush=True)
    else:
        print("[V33 ALL REQUESTED STAGES COMPLETED CLEANLY]", flush=True)


if __name__ == "__main__":
    main()
