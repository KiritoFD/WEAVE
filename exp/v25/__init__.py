"""V25: Minimal PRISM pull-together (收敛) + V25-M mechanisms (空间门控 / 多强度蒸馏 / One-Shot 监督解耦)."""
from __future__ import annotations

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

OUT = ROOT / "runs/v25"


class RawWriterV25(RawWriter):
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


def run_measure_v25(arm_name: str, device: str = "cuda") -> dict:
    s = run_measure_subprocess(OUT / "raw" / arm_name, OUT / "measured" / arm_name, arm_name=arm_name, device=device)
    save_json(OUT / "measured" / arm_name / "summary_v25.json", s)
    c = s["cross600"]
    pt = {k[:3]: round(v["csd_top1"] * 100, 1) for k, v in s["per_target_cross120"].items()}
    print(
        f"[V25 EVAL] {arm_name:24s} | DINO-C={c['dino_c']:.4f} DINO-S={c['dino_s']:.4f} LPIPS={c['lpips']:.4f} "
        f"Aff={c['style_affinity']:.4f} Top1={c['csd_top1']*100:5.2f}% Same={s['same150']['csd_top1']*100:4.1f}% | Top1:{pt}",
        flush=True,
    )
    return s
