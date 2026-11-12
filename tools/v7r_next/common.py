"""WEAVE-CPT / V7R_NEXT shared config, paths, frozen-asset loading, recording.

All modules import from here; no module re-defines B4 / critic / canon loading.
Runs on 3060 WSL: ROOT = /mnt/i/Latent_Style/WEAVE.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.v6.train_eval_v6 import STYLES  # noqa: E402
from tools.v7r import v7r_common as V7  # noqa: E402

RUNS = ROOT / "runs/v7r_next"
STATE = RUNS / "state"
LOGS = RUNS / "logs"
for _d in (RUNS, STATE, LOGS):
    _d.mkdir(parents=True, exist_ok=True)

B4_CKPT = ROOT / "runs/v6/P1_B4_distill50/P1_B4_distill50.pt"
R1_CKPT = ROOT / "runs/v6_next/probes/r1_r1.pt"
TASKS = ROOT / "eval_protocol/tasks_750.json"
REFS = ROOT / "eval_protocol/cache/references_v1_features.pt"
SRC_CACHE = ROOT / "runs/v6_r2/cache/tasks750_source_features.pt"
PATCH_BANK = ROOT / "runs/v7r/p1_energy/patch_banks.pt"   # E3 legacy, read-only
SD_POOL = V7.SD_POOL                                      # 15,545 real latents
V7R = ROOT / "runs/v7r"                                   # read-only history

BUDGET_TOTAL_SEC = 15 * 3600
SEED = 20261112


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256(path: Path, limit_mb: int = 0) -> str:
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
            n += len(b)
            if limit_mb and n >= limit_mb << 20:
                break
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stage_done(name: str) -> bool:
    return (STATE / f"{name}.done").exists()


def mark_done(name: str, artifacts: list[str]) -> None:
    for a in artifacts:
        if not (ROOT / a).exists():
            raise RuntimeError(f"[{name}] missing artifact: {a}")
    (STATE / f"{name}.done").write_text(
        f"{int(time.time())}\n" + "\n".join(artifacts), encoding="utf-8")


def eta_update() -> None:
    """Work-based ETA from measured per-job wall times (NOT a GPU ETA promise)."""
    rec = RUNS / "timings.json"
    t = read_json(rec) if rec.exists() else {"eval750": [], "train_step": []}
    write_json(RUNS / "eta.json", {
        "budget_total_sec": BUDGET_TOTAL_SEC,
        "measured_mean_eval750_sec": float(np.mean(t["eval750"])) if t["eval750"] else None,
        "measured_train_step_sec": float(np.mean(t["train_step"])) if t["train_step"] else None,
        "n_eval750_done": len(t["eval750"]),
        "n_train_steps_done": len(t["train_step"]),
        "note": "work-based estimate; never wait to fill the budget",
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })


def record_timing(kind: str, sec: float) -> None:
    p = RUNS / "timings.json"
    t = read_json(p) if p.exists() else {"eval750": [], "train_step": []}
    t.setdefault(kind, []).append(float(sec))
    write_json(p, t)
    eta_update()


def load_c3unet(device="cuda"):
    from tools.v6.models_v6 import C3UNet
    m = C3UNet(base_dim=16, num_styles=5, mode="direct", inject_16=True,
               style_keys=False).to(device).eval()
    ck = torch.load(B4_CKPT, map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model"])
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def load_critic(device="cuda"):
    from tools.v6_next.probe_style import LatentStyleNet
    c = LatentStyleNet(width=64).to(device).eval()
    c.load_state_dict(torch.load(R1_CKPT, map_location="cpu",
                                 weights_only=False)["model"])
    for p in c.parameters():
        p.requires_grad_(False)
    return c


def load_canon(device="cuda"):
    from tools.v6.canonicalize import ContentCanonicalizer
    stats = torch.load(ROOT / "runs/v6/cache/canonicalizer_stats.pt",
                       map_location="cpu", weights_only=False)
    return ContentCanonicalizer(stats).to(device).eval()


def tasks_sha() -> str:
    return sha256(TASKS)


# ---- target specificity (P0.1 / all later gates) ---------------------------

def specificity_from_scores(style_scores_5, tid: int) -> float:
    """G(z,t) = S(z,t) - mean_{k!=t} S(z,k)."""
    others = [v for j, v in enumerate(style_scores_5) if j != tid]
    return float(style_scores_5[tid] - sum(others) / 4.0)


def summarize_specificity(rows: list[dict]) -> dict:
    cross = [r for r in rows if r.get("is_cross")]
    same = [r for r in rows if not r.get("is_cross")]
    def agg(rs, key):
        v = [r[key] for r in rs if key in r]
        return {"mean": round(float(np.mean(v)), 5),
                "n": len(v)} if v else None
    return {
        "cross600_dino_s_specificity": agg(cross, "dino_s_specificity"),
        "cross600_dino_s": agg(cross, "dino_s"),
        "same150_dino_s_specificity": agg(same, "dino_s_specificity"),
        "note": "specificity = target-style DINO-S minus mean of other 4 styles; "
                "a DINO-S gain without specificity gain is NOT target-specific.",
    }
