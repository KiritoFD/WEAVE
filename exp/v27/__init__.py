"""V27-Zero: 纯全局 Stiefel 噪声流 — V21 减法重构 (删加性 RefAttn) + 无压缩全统计 E_ref + 无监督 kNN 异图配对.

用户 2026-11-22 立项 (门控 1: 4.4k, ~1.5h):
  1. 架构减法 (V26 Exp 1 落实): 删 mid_ref_attn / dec32_ref_attn (16×16 商流形零加性 V);
  2. 保留 V21 四铁律: Noise-to-Data Flow + Q32 (20% Gram 地板) + y16_lcn + 4-Branch CFG;
  3. E_ref 升级: RGB-DINO 全局 μ/σ (384) + Oklab 27 + 2 级 Haar 能量/偏度 12 → MLP 128, 零 5 类压缩;
  4. 无监督 DINO-kNN 异图近邻配对 (r≠i, 零类别标签, 可平移 78k).
门控: SG-5 1-Shot Top-1 ≥ 45% 且 Holdout 单图 ≥ 38% 且 DINO-C ≥ 0.55.
"""
from __future__ import annotations

from pathlib import Path

import torch
from src.eval.runner import run_measure_subprocess
from src.utils import decode_s10, load_s10_decoder
from src.utils.raw_io import RawWriter
from src.utils.runtime import (
    REPO as ROOT,
    read_json,
    save_json,
    utc,
)

OUT = ROOT / "runs/v27"


class RawWriterV27(RawWriter):
    def __init__(self, arm_name: str, dec=None, device: str = "cuda"):
        super().__init__(arm_name, raw_root=OUT / "raw", kind="v27")
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


def run_measure_v27(arm_name: str, device: str = "cuda") -> dict:
    s = run_measure_subprocess(OUT / "raw" / arm_name, OUT / "measured" / arm_name,
                               arm_name=arm_name, device=device)
    save_json(OUT / "measured" / arm_name / "summary_v27.json", s)
    c = s["cross600"]
    pt = {k[:3]: round(v["csd_top1"] * 100, 1) for k, v in s["per_target_cross120"].items()}
    print(f"[V27 EVAL] {arm_name:28s} | Top1={c['csd_top1']*100:5.2f}% DINO-S={c['dino_s']:.4f} "
          f"DINO-C={c['dino_c']:.4f} LPIPS={c['lpips']:.4f} | Top1:{pt}", flush=True)
    return s
