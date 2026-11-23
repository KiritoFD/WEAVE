"""V27 Mainline: Anchor-Free 4-Step Q32-Reflow (Dir 1) & 4-Step Differentiable Rollout Refinement (Dir 2)."""
from __future__ import annotations

import torch

from src.eval.runner import run_measure_subprocess
from src.utils import decode_s10, load_s10_decoder
from src.utils.raw_io import RawWriter
from src.utils.runtime import REPO as ROOT, read_json, save_json, utc

OUT = ROOT / "runs/v27_mainline"
Q32_FLAGSHIP_CKPT = ROOT / "runs/v27/icme_q32_flagship.pt"
P1_CKPT = ROOT / "runs/v19/students/s2_prism1_v2.pt"
B4_CKPT = ROOT / "runs/v6/P1_B4_distill50/P1_B4_distill50.pt"


class RawWriterV27M(RawWriter):
    def __init__(self, arm_name: str, dec=None, device: str = "cuda"):
        super().__init__(arm_name, raw_root=OUT / "raw", kind="v27_mainline")
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


def build_wrong_task_index(tasks: list[dict], sid_t: torch.Tensor, tid_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (tid_wrong, wrong_task_idx) so anchors & conditions both switch to wrong target."""
    from src.utils.protocol import pick_wrong_cross_tid

    tid_wrong = pick_wrong_cross_tid(sid_t, tid_t)
    lut = {(t["source_stem"], int(t["target_style_id"])): idx for idx, t in enumerate(tasks)}
    wrong_idx = torch.tensor(
        [lut[(tasks[i]["source_stem"], int(tid_wrong[i].item()))] for i in range(len(tasks))],
        dtype=torch.long,
    )
    return tid_wrong, wrong_idx


def summarize_eval(arm_name: str, m_dir) -> dict:
    s = read_json(m_dir / "summary_v27m.json")
    c = s["cross600"]
    rows = read_json(m_dir / "eval.json")["rows"]
    imp_min = [r for r in rows if r["src_style"] == "Impressionism" and r["tgt_style"] == "Minimalism"]
    return {
        "csd_top1": round(c["csd_top1"], 4),
        "dino_s": round(c["dino_s"], 4),
        "dino_c": round(c["dino_c"], 4),
        "style_affinity": round(c["style_affinity"], 4),
        "clip_s": round(c["clip_s"], 4),
        "lpips": round(c["lpips"], 4),
        "same150_top1": round(s["same150"]["csd_top1"], 4),
        "imp_to_min_top1": round(sum(int(r["target_top1"]) for r in imp_min) / max(len(imp_min), 1), 4),
        "per_target": {k: round(v["csd_top1"], 4) for k, v in s.get("per_target_cross120", {}).items()},
    }


def run_measure_v27m(arm_name: str, device: str = "cuda") -> dict:
    m_dir = OUT / "measured" / arm_name
    if (m_dir / "summary_v27m.json").exists():
        return summarize_eval(arm_name, m_dir)
    s = run_measure_subprocess(OUT / "raw" / arm_name, m_dir, arm_name=arm_name, device=device)
    save_json(m_dir / "summary_v27m.json", s)
    c = s["cross600"]
    pt = {k[:3]: round(v["csd_top1"] * 100, 1) for k, v in s["per_target_cross120"].items()}
    print(
        f"[V27M EVAL] {arm_name:28s} | Top1={c['csd_top1']*100:5.2f}% DINO-S={c['dino_s']:.4f} "
        f"DINO-C={c['dino_c']:.4f} LPIPS={c['lpips']:.4f} | Top1:{pt}",
        flush=True,
    )
    return summarize_eval(arm_name, m_dir)
