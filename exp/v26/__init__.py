"""V26: Unified One-Shot PRISM — 少步任意参考 Flow 收敛与机制诊断 (docs/1120 方案).

公共层 (两卡共享): SR-CFG 积分器与评测、Q32 谱诊断、RefAttn 因果诊断、K2 风格损失审计.
分卡队列: exp.v26.gpu3060 (P0 主线) / exp.v26.gpu4090 (K0–K2 诊断).
"""
from __future__ import annotations

from pathlib import Path

from src.eval.runner import run_measure_subprocess
from src.utils import decode_s10, load_s10_decoder
from src.utils.protocol import EVAL_SEED, STYLES, pick_wrong_cross_tid  # noqa: F401
from src.utils.raw_io import RawWriter, export_raw_pngs
from src.utils.runtime import (
    REPO as ROOT,
    bf16_autocast as autocast,
    read_json,
    sanitize_cuda_env,
    save_json,
    seed_all,
    sha256_file,
    tier_micro_accum,
    utc,
)
from exp.v22 import config_hash, file_sha, git_sha, validate_raw_meta

OUT = ROOT / "runs/v26"

# V26 依赖的既有资产锚点 (只读, 不修改 V25 产物)
C2NT_CKPT = ROOT / "runs/v25/oneshot_val/C2nt_sw6_lc0.8.pt"
V21_CACHE = ROOT / "runs/v21/cache/v21_stiefel_ref_cache.pt"
V16_TR_CACHE = ROOT / "runs/v24/cache/train_cnx_v16.pt"
V25_RAW_C2NT_T45 = ROOT / "runs/v25/raw/C2nt_sw6_lc0.8_t45"
V25_MEASURED_C2NT_T45 = ROOT / "runs/v25/measured/C2nt_sw6_lc0.8_t45/summary_v24.json"


class RawWriterV26(RawWriter):
    def __init__(self, arm_name: str, dec=None, device: str = "cuda"):
        super().__init__(arm_name, raw_root=OUT / "raw", kind="v24")
        self._dec = dec
        self._device = device

    def write_batch(self, z_final, tasks_slice: list[dict]) -> None:
        if self._dec is None:
            self._dec = load_s10_decoder(self._device)
        ids = [t["task_id"] for t in tasks_slice]
        rgb01 = decode_s10(self._dec, z_final, batch=16).clamp(0.0, 1.0)
        self.append(ids, rgb01, z_final=z_final.cpu())

    def finalize(self, **extra) -> dict:
        self.finish(**extra)
        return read_json(self.path / "meta.json")


def run_measure_v26(arm_name: str, device: str = "cuda") -> dict:
    s = run_measure_subprocess(OUT / "raw" / arm_name, OUT / "measured" / arm_name,
                               arm_name=arm_name, device=device)
    save_json(OUT / "measured" / arm_name / "summary_v26.json", s)
    c = s["cross600"]
    pt = {k[:3]: round(v["csd_top1"] * 100, 1) for k, v in s["per_target_cross120"].items()}
    print(
        f"[V26 EVAL] {arm_name:24s} | DINO-C={c['dino_c']:.4f} DINO-S={c['dino_s']:.4f} LPIPS={c['lpips']:.4f} "
        f"Aff={c['style_affinity']:.4f} Top1={c['csd_top1']*100:5.2f}% Same={s['same150']['csd_top1']*100:4.1f}% | Top1:{pt}",
        flush=True,
    )
    return s


def arm_summary_path(arm_name: str) -> Path:
    return OUT / "measured" / arm_name / "summary_v26.json"


def arm_done(arm_name: str) -> bool:
    p = arm_summary_path(arm_name)
    if not p.exists():
        return False
    s = read_json(p)
    return bool(s) and int(s.get("n_total", 0)) == 750
