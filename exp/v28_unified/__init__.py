"""V28 PRISM-1-Unified v2: 18h RTX 4090 Teacher Repair, Student Fidelity, and Teacher Duel."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.eval.runner import run_measure_subprocess
from src.utils import decode_s10, load_s10_decoder
from src.utils._local_eval_refs import csd_prototypes
from src.utils.raw_io import RawWriter
from src.utils.runtime import REPO as ROOT, read_json, save_json

OUT = ROOT / "runs/v28_unified"
UNIFIED_OLD_OUT = ROOT / "runs/v27_mainline/prism1_unified"
GATE_OLD_OUT = ROOT / "runs/v27_mainline/unified_teacher_gate"
Q32_FLAGSHIP_CKPT = ROOT / "runs/v27/icme_q32_flagship.pt"
PEAK1500_CKPT = ROOT / "runs/v22/tmatrix/slf_T4Ssnap1500.pt"
V20_D2_CKPT = ROOT / "runs/v20/slf_flow/slf_core_v2.pt"
P1_CKPT = ROOT / "runs/v19/students/s2_prism1_v2.pt"
B4_CKPT = ROOT / "runs/v6/P1_B4_distill50/P1_B4_distill50.pt"
SD0_VAE_DIR = ROOT / "runs/v26/cache/sd0_vae_full"

SEED_PLAN = 20261126
SEED_TRAIN = 20261128
MAX_VRAM_GB = 20.5


def atomic_save(payload, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def recipe_hash(obj: dict) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def peak_vram_gb(device: str = "cuda") -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 3)


def assert_vram_budget(limit_gb: float = MAX_VRAM_GB, device: str = "cuda", tag: str = "") -> float:
    gb = peak_vram_gb(device)
    if gb > limit_gb:
        raise RuntimeError(f"[{tag}] Peak VRAM {gb:.2f} GB exceeded limit {limit_gb:.2f} GB")
    return gb


class RawWriterV28(RawWriter):
    def __init__(self, arm_name: str, dec=None, device: str = "cuda"):
        super().__init__(arm_name, raw_root=OUT / "raw", kind="v28_unified")
        self._dec = dec
        self._device = device

    def write_batch(self, z_final: torch.Tensor, tasks_slice: list[dict]) -> None:
        if self._dec is None:
            self._dec = load_s10_decoder(self._device)
        ids = [t["task_id"] for t in tasks_slice]
        rgb01 = decode_s10(self._dec, z_final, batch=32).clamp(0.0, 1.0)
        self.append(ids, rgb01, z_final=z_final)

    def finalize(self, **extra) -> dict:
        self.finish(**extra)
        return read_json(self.path / "meta.json")


def hard_hits(arm_name: str) -> dict[str, list[int]]:
    rows = read_json(OUT / "measured" / arm_name / "eval.json")["rows"]
    counts = {}
    for sid, key in ((0, "ER"), (4, "Uki")):
        grp = [r for r in rows if r["is_cross"] and r["tgt_style_id"] == sid]
        counts[key] = [sum(int(r["target_top1"]) for r in grp), len(grp)]
    return counts


def compute_csls_k10(arm_name: str) -> dict:
    rows = [r for r in read_json(OUT / "measured" / arm_name / "eval.json")["rows"] if r["is_cross"]]
    src_feats = torch.load(ROOT / "runs/v6_r2/cache/tasks750_source_features.pt", map_location="cpu", weights_only=False)
    tasks = read_json(ROOT / "eval_protocol/tasks_750.json")["tasks"]
    stems = sorted({t["source_stem"] for t in tasks})
    proto = csd_prototypes()
    bg = np.stack([src_feats["raw_csd"][s] for s in stems])
    bg_sim = bg @ proto.T
    r_proto = np.sort(bg_sim, axis=0)[-10:].mean(0)
    x = np.array([r["csd_scores_5"] for r in rows])
    y = np.array([r["tgt_style_id"] for r in rows])
    pred = (2.0 * x - r_proto).argmax(axis=1)
    return {
        "csls_k10": round(float((pred == y).mean()), 4),
        "csls_k10_er": [int(((pred == 0) & (y == 0)).sum()), int((y == 0).sum())],
        "csls_k10_uki": [int(((pred == 4) & (y == 4)).sum()), int((y == 4).sum())],
    }


def summarize_eval_v28(arm_name: str, m_dir: Path) -> dict:
    s = read_json(m_dir / "summary_v28.json")
    c = s["cross600"]
    rows = read_json(m_dir / "eval.json")["rows"]
    imp_min = [r for r in rows if r["src_style"] == "Impressionism" and r["tgt_style"] == "Minimalism"]
    out = {
        "csd_top1": round(c["csd_top1"], 4),
        "dino_s": round(c["dino_s"], 4),
        "dino_c": round(c["dino_c"], 4),
        "style_affinity": round(c["style_affinity"], 4),
        "clip_s": round(c["clip_s"], 4),
        "lpips": round(c["lpips"], 4),
        "same150_top1": round(s["same150"]["csd_top1"], 4),
        "imp_to_min_top1": round(sum(int(r["target_top1"]) for r in imp_min) / max(len(imp_min), 1), 4),
        "per_target": {k: round(v["csd_top1"], 4) for k, v in s.get("per_target_cross120", {}).items()},
        "hits": hard_hits(arm_name),
    }
    out.update(compute_csls_k10(arm_name))
    return out


def run_measure_v28(arm_name: str, device: str = "cuda") -> dict:
    m_dir = OUT / "measured" / arm_name
    if (m_dir / "summary_v28.json").exists():
        return summarize_eval_v28(arm_name, m_dir)
    s = run_measure_subprocess(OUT / "raw" / arm_name, m_dir, arm_name=arm_name, device=device)
    save_json(m_dir / "summary_v28.json", s)
    c = s["cross600"]
    pt = {k[:3]: round(v["csd_top1"] * 100, 1) for k, v in s["per_target_cross120"].items()}
    print(
        f"[V28 EVAL] {arm_name:32s} | Top1={c['csd_top1']*100:5.2f}% DINO-S={c['dino_s']:.4f} "
        f"DINO-C={c['dino_c']:.4f} LPIPS={c['lpips']:.4f} | Top1:{pt}",
        flush=True,
    )
    return summarize_eval_v28(arm_name, m_dir)
