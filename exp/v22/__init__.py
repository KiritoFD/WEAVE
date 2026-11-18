"""V22: PRISM 主模型正确化、受控消融与 One-Shot 任意参考扩展（P1 + P2）。

P0 统一接口（可替换组件）:
  build_style_condition / build_content_condition / make_initial_state /
  predict_guided_velocity / integrate_flow / decode_output
P0 修复: T4 缓存版本绑定、Same150 旁路拆分、反事实锚点对照、缓存完整性校验、V21 无放回参考采样。
P1 消融: A(锚点) / B(CFG 分支) / C(密度门控) / D(原型) + Same150 拆分。
P2: 双分支重训（无 content dropout）、Reference Adapter 与风格编码诊断。
"""
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
    sha256_file,
    tier_micro_accum,
    utc,
)

OUT = ROOT / "runs/v22"


class RawWriterV22(RawWriter):
    def __init__(self, arm_name: str, dec=None, device: str = "cuda"):
        super().__init__(arm_name, raw_root=OUT / "raw", kind="v22")
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


def run_measure_v22(arm_name: str, device: str = "cuda") -> dict:
    s = run_measure_subprocess(OUT / "raw" / arm_name, OUT / "measured" / arm_name, arm_name=arm_name, device=device)
    save_json(OUT / "measured" / arm_name / "summary_v22.json", s)
    c = s["cross600"]
    pt = {k[:3]: round(v["csd_top1"] * 100, 1) for k, v in s["per_target_cross120"].items()}
    pt_ds = {k[:3]: round(v["dino_s"], 3) for k, v in s["per_target_cross120"].items()}
    print(
        f"[V22 EVAL] {arm_name:32s} | Top1={c['csd_top1']*100:5.2f}% "
        f"DINO-S={c['dino_s']:.4f} CLIP-S={c['clip_s']:.4f} Aff={c['style_affinity']:.4f} "
        f"DINO-C={c['dino_c']:.4f} LPIPS={c['lpips']:.4f} Same={s['same150']['csd_top1']*100:4.1f}% "
        f"| Top1:{pt} | D-S:{pt_ds}",
        flush=True,
    )
    return s


def config_hash(cfg: dict, n: int = 12) -> str:
    import hashlib
    import json as _json
    blob = _json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:n]


def git_sha() -> str:
    import subprocess
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def validate_raw_meta(meta: dict, expected_tasks: int = 750) -> bool:
    """Fix 4: raw/meta.json 只有在恰好 expected_tasks 个唯一 task_id 时才算完整。"""
    try:
        ids = [tid for b in meta["batches"] for tid in b["ids"]]
        return len(ids) == expected_tasks and len(set(ids)) == expected_tasks
    except Exception:
        return False


def file_sha(path: Path) -> str:
    p = Path(path)
    return sha256_file(p) if p.exists() else "missing"
