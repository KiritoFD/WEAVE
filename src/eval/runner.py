"""Process-isolated referee runner and standardized arm summary builder."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from src.utils.protocol import STYLES
from src.utils.runtime import read_json, run_isolated_python, save_json, utc


def summarize_measured_arm(arm_name: str, meas_dir: Path | str) -> dict:
    meas_p = Path(meas_dir)
    d = read_json(meas_p / "eval.json")
    rows = d["rows"]
    if d["n_total"] != 750 or d["n_cross"] != 600 or d["n_same"] != 150:
        raise ValueError(f"unexpected eval counts in {meas_p}: {d['n_total']}/{d['n_cross']}/{d['n_same']}")

    def _stats(rs):
        pred_cls = [int(np.argmax(r["csd_scores_5"])) for r in rs]
        clip_s = float(np.mean([r["clip_scores_5"][r["tgt_style_id"]] for r in rs]))
        dino_s = float(np.mean([r["dino_s"] for r in rs]))
        return {
            "n": len(rs),
            "csd_top1": float(np.mean([r["target_top1"] for r in rs])),
            "csd_margin": float(np.mean([r["target_margin"] for r in rs])),
            "dT": float(np.mean([r["dT"] for r in rs])),
            "dino_s": dino_s,
            "clip_s": clip_s,
            "style_affinity": (dino_s + clip_s) / 2.0,
            "dino_c": float(np.mean([r["dino_c"] for r in rs])),
            "lpips": float(np.mean([r["lpips"] for r in rs])),
            "pred_eq_src": float(np.mean([p == r["src_style_id"] for p, r in zip(pred_cls, rs)])),
            "pred_eq_imp": float(np.mean([p == 1 for p in pred_cls])),
        }

    cross_rows = [r for r in rows if r["is_cross"]]
    same_rows = [r for r in rows if not r["is_cross"]]
    return {
        "arm": arm_name,
        "utc": utc(),
        "cross600": _stats(cross_rows),
        "same150": _stats(same_rows),
        "per_target_cross120": {
            st: _stats([r for r in cross_rows if r["tgt_style_id"] == sid])
            for sid, st in enumerate(STYLES)
        },
    }


def run_measure_subprocess(
    raw_dir: Path | str,
    meas_dir: Path | str,
    *,
    arm_name: str | None = None,
    device: str = "cuda",
    write_png: bool = False,
    micro_batch: int = 8,
    force: bool = False,
) -> dict:
    raw_p, meas_p = Path(raw_dir), Path(meas_dir)
    name = arm_name or raw_p.name
    if force or not (meas_p / "eval.json").exists():
        args = [
            "--raw", str(raw_p),
            "--out", str(meas_p),
            "--device", device,
            "--micro-batch", str(micro_batch),
        ]
        if not write_png:
            args.append("--no-png")
        rc = run_isolated_python("src.eval.measure", args)
        if rc != 0:
            raise RuntimeError(f"src.eval.measure failed for {name} (rc={rc})")
    summary = summarize_measured_arm(name, meas_p)
    save_json(meas_p / "summary_arm.json", summary)
    return summary
