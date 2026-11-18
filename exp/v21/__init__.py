"""V21: True Arbitrary-Reference 1-Shot Transfer & Deep Pareto Breakthrough over StyleGallery."""
from __future__ import annotations

from pathlib import Path
import torch
from src.eval.runner import run_measure_subprocess
from src.utils import decode_s10, load_s10_decoder
from src.utils.raw_io import RawWriter, export_raw_pngs
from src.utils.protocol import EVAL_SEED, STYLES, pick_wrong_cross_tid
from src.utils.runtime import (
    REPO as ROOT,
    bf16_autocast as autocast,
    read_json,
    sanitize_cuda_env,
    save_json,
    seed_all,
    tier_micro_accum,
    utc,
)

OUT = ROOT / "runs/v21"


class RawWriterV21(RawWriter):
    def __init__(self, arm_name: str, dec=None, device: str = "cuda"):
        super().__init__(arm_name, raw_root=OUT / "raw", kind="v21")
        self._dec = dec
        self._device = device

    def write_batch(self, z_final: torch.Tensor, tasks_slice: list[dict]) -> None:
        if self._dec is None:
            self._dec = load_s10_decoder(self._device)
        ids = [t["task_id"] for t in tasks_slice]
        rgb01 = decode_s10(self._dec, z_final, batch=16).clamp(0.0, 1.0)
        self.append(ids, rgb01, z_final=z_final)

    def finalize(self, **extra) -> dict:
        self.finish(**extra)
        return read_json(self.path / "meta.json")


def run_measure_v21(arm_name: str, device: str = "cuda") -> dict:
    s = run_measure_subprocess(OUT / "raw" / arm_name, OUT / "measured" / arm_name, arm_name=arm_name, device=device)
    save_json(OUT / "measured" / arm_name / "summary_v21.json", s)
    c = s["cross600"]
    pt = {k[:3]: round(v["csd_top1"] * 100, 1) for k, v in s["per_target_cross120"].items()}
    pt_ds = {k[:3]: round(v["dino_s"], 3) for k, v in s["per_target_cross120"].items()}
    print(
        f"[V21 EVAL] {arm_name:30s} | Top1={c['csd_top1']*100:5.2f}% "
        f"DINO-S={c['dino_s']:.4f} CLIP-S={c['clip_s']:.4f} Aff={c['style_affinity']:.4f} "
        f"DINO-C={c['dino_c']:.4f} LPIPS={c['lpips']:.4f} Same={s['same150']['csd_top1']*100:4.1f}% "
        f"| Top1:{pt} | D-S:{pt_ds}",
        flush=True,
    )
    return s
