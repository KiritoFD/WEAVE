"""Full V36 CLEAVE-One Pipeline Orchestrator (36h RTX 3060 Schedule):
  - P0: Whole-WikiArt dedup + Core-5 holdout isolation + resumable memmap cache
  - Profile: 30-step/cand micro-batch 4/8/16 throughput & <=9.5GB VRAM check + 200-step P1 sanity
  - P1: Full 131-movement art pretraining (~21.5h budget, 20k-40k updates adaptive)
  - P2: Transfer strengthening (60% proxy content, ~6.0h budget)
  - P3: Reference-A3 (I0/I1/I2) & CFG calibration on Seen-Dev ONLY
  - P4: Full test750 (Core-5 strictly unseen movements) + I0/I1/I2 ablation + fixed visual grids
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

from exp.v36_art_foundation import CKPT_DIR, LOG_DIR, MEAS_DIR, ROOT
from exp.v36_art_foundation.eval_v36 import calibrate_stage_p3, eval_v36_full
from exp.v36_art_foundation.prepare_data import build_v36_caches
from exp.v36_art_foundation.train_foundation import _MemmapDataset, profile_micro_batch, train_foundation
from src.utils.runtime import read_json, sanitize_cuda_env, save_json, utc


def run_all(
    *,
    holdout_preset: str = "core5",
    p1_hours: float = 21.5,
    p2_hours: float = 6.0,
    tag: str = "v36_main",
    train_subset: int = 0,
    run_sanity_200: bool = True,
    device: str = "cuda",
) -> None:
    os.chdir(ROOT)
    sanitize_cuda_env()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    MEAS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Stage P0: Curate + Precompute Memmap Caches (resumable)
    build_v36_caches(holdout_preset=holdout_preset, device=device)

    # 2. Pre-flight Profiling (32/48 micro-batch under ~10GB VRAM, AsyncPrefetcher)
    prof_p = LOG_DIR / "profile_micro.json"
    if not prof_p.exists():
        ds = _MemmapDataset(train_subset=train_subset)
        prof = profile_micro_batch(ds, eff_batch=96, device=device)
    else:
        prof = read_json(prof_p)
    best_cfg = prof["best"]
    micro = int(best_cfg["micro"])
    eff_batch = int(best_cfg.get("eff_batch", 96))
    use_ckpt = bool(best_cfg["use_checkpoint"])
    sec_u = float(best_cfg["sec_per_update"])
    p1_steps = max(4000, min(60000, int(p1_hours * 3600.0 / max(0.05, sec_u))))
    p2_steps = max(1500, min(20000, int(p2_hours * 3600.0 / max(0.05, sec_u))))
    print(f"[V36 Plan] best={best_cfg} -> eff_batch={eff_batch}, p1_steps={p1_steps} ({p1_hours}h), p2_steps={p2_steps} ({p2_hours}h)", flush=True)

    # 3. 200-step verification before multi-hour training
    if run_sanity_200 and not (LOG_DIR / f"{tag}_sanity200.ok").exists():
        train_foundation(
            stage="p1", steps=200, micro=micro, eff_batch=eff_batch, use_checkpoint=use_ckpt,
            tag=f"{tag}_sanity200", train_subset=train_subset, save_every=200, diag_every=200, device=device,
        )
        save_json(LOG_DIR / f"{tag}_sanity200.ok", {"ok": True, "utc": utc(), "micro": micro, "eff_batch": eff_batch, "use_checkpoint": use_ckpt})

    # 4. Stage P1: Full 131-movement generation pretraining
    p1_ckpt = train_foundation(
        stage="p1", steps=p1_steps, micro=micro, eff_batch=eff_batch, use_checkpoint=use_ckpt,
        tag=tag, train_subset=train_subset, save_every=2500, diag_every=5000, max_hours=p1_hours, device=device,
    )

    # 5. Stage P2: Transfer strengthening (60% proxy content, lower LR)
    p2_ckpt = train_foundation(
        stage="p2", steps=p2_steps, micro=micro, eff_batch=eff_batch, use_checkpoint=use_ckpt,
        resume_from=str(p1_ckpt), tag=tag, train_subset=train_subset,
        save_every=2500, diag_every=2500, max_hours=p2_hours, device=device,
    )

    # 6. Stage P3: Seen-Dev calibration (I0/I1/I2 + CFG)
    p3_rep = calibrate_stage_p3(p2_ckpt, device=device)
    best_p3 = p3_rep["best"]
    best_mode = str(best_p3["mode"])
    best_wj = float(best_p3["wj"])

    # 7. Stage P4: Full test750 sweep (Core-5 unseen) + Reference-A3 ablation (I0/I1/I2)
    eval_v36_full(p2_ckpt, tag=f"{tag}_p2", start_mode=best_mode, wj=best_wj, wc=1.6, steps=6, device=device)
    for m_ab in ("I0", "I1", "I2"):
        if m_ab != best_mode:
            eval_v36_full(p2_ckpt, tag=f"{tag}_ablate", start_mode=m_ab, taus=(0.50, 0.65), wj=best_wj, wc=1.6, steps=6, device=device)

    save_json(LOG_DIR / f"{tag}_pipeline_complete.json", {
        "utc": utc(), "p1_ckpt": str(p1_ckpt), "p2_ckpt": str(p2_ckpt),
        "p3_best": best_p3, "p1_steps": p1_steps, "p2_steps": p2_steps,
    })
    print(f"[V36 Pipeline] ALL STAGES COMPLETE -> {tag}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout-preset", default="core5", choices=["core5", "v35"])
    ap.add_argument("--p1-hours", type=float, default=21.5)
    ap.add_argument("--p2-hours", type=float, default=6.0)
    ap.add_argument("--tag", default="v36_main")
    ap.add_argument("--train-subset", type=int, default=0)
    ap.add_argument("--skip-sanity", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    run_all(
        holdout_preset=a.holdout_preset,
        p1_hours=a.p1_hours,
        p2_hours=a.p2_hours,
        tag=a.tag,
        train_subset=a.train_subset,
        run_sanity_200=not a.skip_sanity,
        device=a.device,
    )


if __name__ == "__main__":
    main()
