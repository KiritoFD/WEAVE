"""Master CLI & 18h resumable runner for V28 PRISM-1-Unified v2."""
from __future__ import annotations

import argparse
import time

from exp.v28_unified import OUT, ROOT, assert_vram_budget
from exp.v28_unified.phase0_audit import run_phase0
from exp.v28_unified.phase1_race import run_phase1
from exp.v28_unified.phase2_pools import run_phase2
from exp.v28_unified.phase3_4_5 import run_knob2_frontier, run_phase3, run_phase4, run_phase4_5, run_phase5
from exp.v28_unified.train_student import train_student_arm
from exp.v28_unified.train_duel import run_duel_smoke, run_training_duel
from exp.v28_unified.tto_ablation import run_tto_ablation, run_tto_probe
from src.utils.runtime import read_json, sanitize_cuda_env, save_json, utc


def run_smoke(device: str = "cuda") -> dict:
    """20-step throughput & VRAM smoke test (< 18.5 GB) for new U-Net + S10-BF16 bridge."""
    pool_path = OUT / "pools/pool_A.pt"
    if not pool_path.exists():
        # Fall back to existing pairs.pt for architecture/VRAM/throughput smoke before pools exist
        pool_path = ROOT / "runs/v27_mainline/prism1_unified/pairs.pt"
    rep_unet = train_student_arm(
        "SMOKE_unet_20", pool_path, arch="unet", interp_mode="2tier" if pool_path.name == "pairs.pt" else "3tier",
        loss_mode="calibrated", smoke_steps=20, device=device,
    )
    assert_vram_budget(device=device, tag="smoke_unet")
    return rep_unet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--phase", default="all", choices=("all", "0", "1", "2", "3", "4", "4_5", "5", "knob2", "tto_probe", "tto", "duel_smoke", "duel"))
    args = ap.parse_args()
    sanitize_cuda_env()
    OUT.mkdir(parents=True, exist_ok=True)
    status_p = OUT / "pipeline_status.json"
    status = read_json(status_p) if status_p.exists() else {"started_utc": utc(), "phases": {}}

    if args.smoke:
        rep = run_smoke(device=args.device)
        save_json(OUT / "smoke_report.json", rep)
        return

    phases = ("0", "1", "2", "3", "4", "4_5", "5") if args.phase == "all" else (args.phase,)
    for ph in phases:
        t0 = time.monotonic()
        status["current_phase"] = ph
        status["updated_utc"] = utc()
        save_json(status_p, status)
        print(f"\n==================== PHASE {ph} START ====================", flush=True)
        if ph == "0":
            run_phase0(device=args.device)
        elif ph == "1":
            rep = run_phase1(args.pool, device=args.device)
            # Smoke test new U-Net before Phase 2/3 long run
            run_smoke(device=args.device)
            if not (rep["gates"]["T1"]["passed"] or rep["gates"]["T1_T4"]["passed"] or rep["gates"]["T2"]["passed"]):
                print("[WARN] Phase 1 gate warning: check phase1_race.json metrics!", flush=True)
        elif ph == "2":
            run_phase2(args.pool, device=args.device)
        elif ph == "3":
            run_phase3(device=args.device)
        elif ph == "4":
            p3 = read_json(OUT / "phase3_summary.json") if (OUT / "phase3_summary.json").exists() else {}
            winner = p3.get("winner", "M3_unet_3tier_cal")
            arch = "flat" if winner.startswith(("M0", "M1")) else "unet"
            interp = "2tier" if "2tier" in winner else "3tier"
            loss = "base" if ("M0" in winner or "M1" in winner) else "calibrated"
            run_phase4(stu_star_arch=arch, stu_star_interp=interp, stu_star_loss=loss, device=args.device)
        elif ph == "4_5":
            run_phase4_5(args.pool, device=args.device)
        elif ph == "5":
            p3 = read_json(OUT / "phase3_summary.json") if (OUT / "phase3_summary.json").exists() else {}
            winner = p3.get("winner", "M3_unet_3tier_cal")
            arch = "flat" if winner.startswith(("M0", "M1")) else "unet"
            run_phase5(arch=arch, device=args.device)
        elif ph == "knob2":
            run_knob2_frontier(device=args.device)
        elif ph == "tto_probe":
            run_tto_probe(device=args.device)
        elif ph == "tto":
            run_tto_ablation(device=args.device)
        elif ph == "duel_smoke":
            run_duel_smoke(device=args.device)
        elif ph == "duel":
            run_training_duel(device=args.device)

        status["phases"][ph] = {"done": True, "elapsed_s": round(time.monotonic() - t0, 1), "utc": utc()}
        status["current_phase"] = "done" if ph == phases[-1] else ph
        save_json(status_p, status)
        print(f"==================== PHASE {ph} DONE ({status['phases'][ph]['elapsed_s']}s) ====================", flush=True)


if __name__ == "__main__":
    main()
