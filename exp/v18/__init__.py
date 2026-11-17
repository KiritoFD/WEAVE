"""WEAVE V18: De-Styled Structural-Content Flow (CCF) & Calibrated LD4+ Amortization."""
from __future__ import annotations

import contextlib
import datetime
import hashlib
import json
import random
import subprocess
import sys
import unicodedata
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/v18"
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
EVAL_SEED = 20261185


def utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path: Path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False),
        encoding="utf-8",
    )
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast(device: str = "cuda"):
    if str(device).startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def canonical_pid(name: str) -> str:
    return unicodedata.normalize(
        "NFC", Path(str(name).replace("\\", "/")).stem.split("__")[-1]
    ).casefold()


def clean_balanced_pool(
    pool_dir: str | Path,
    train_per_style: int = 880,
    hold_per_style: int = 60,
    seed: int = 20261118,
):
    """Load strictly deduplicated, 1:1:1:1:1 balanced latents excluding Test750 & references."""
    tasks = read_json(ROOT / "eval_protocol/tasks_750.json")["tasks"]
    refs = read_json(ROOT / "eval_protocol/references_v1.json")["references"]
    banned = {canonical_pid(t["source_stem"]) for t in tasks}
    banned |= {canonical_pid(r["stem"]) for rows in refs.values() for r in rows}

    def _h10(pid: str) -> int:
        return int(hashlib.md5(pid.encode()).hexdigest()[:8], 16) % 10

    g = torch.Generator().manual_seed(seed)
    splits = {"train": [], "holdout": []}
    manifest = {"seed": seed, "banned_count": len(banned), "styles": {}}

    for sid, style in enumerate(STYLES):
        p = Path(pool_dir) / f"{sid:02d}_{style}.pt"
        d = torch.load(p, map_location="cpu", weights_only=False)
        ids = [str(x) for x in d["ids"]]
        groups: dict[str, list[int]] = {}
        for idx, raw_id in enumerate(ids):
            cp = canonical_pid(raw_id)
            if cp not in banned:
                groups.setdefault(cp, []).append(idx)

        tr_cands, ho_cands = [], []
        for rows in groups.values():
            if any(_h10(ids[r]) < 9 for r in rows):
                pick = next(r for r in rows if _h10(ids[r]) < 9)
                tr_cands.append(pick)
            else:
                ho_cands.append(rows[0])

        n_tr = min(train_per_style, len(tr_cands))
        n_ho = min(hold_per_style, len(ho_cands))
        if n_tr < 500 or n_ho < 30:
            raise ValueError(f"{style}: insufficient clean samples tr={len(tr_cands)} ho={len(ho_cands)}")

        tr_sel = [tr_cands[j] for j in torch.randperm(len(tr_cands), generator=g)[:n_tr].tolist()]
        ho_sel = [ho_cands[j] for j in torch.randperm(len(ho_cands), generator=g)[:n_ho].tolist()]

        for i in tr_sel:
            splits["train"].append((d["latents"][i].float(), sid, ids[i]))
        for i in ho_sel:
            splits["holdout"].append((d["latents"][i].float(), sid, ids[i]))

        manifest["styles"][style] = {
            "pool_total": len(ids),
            "clean_unique": len(groups),
            "train_avail": len(tr_cands),
            "hold_avail": len(ho_cands),
            "train_taken": n_tr,
            "hold_taken": n_ho,
        }

    out = {}
    for sp_name, rows in splits.items():
        out[sp_name] = {
            "z": torch.stack([r[0] for r in rows]),
            "sid": torch.tensor([r[1] for r in rows], dtype=torch.long),
            "ids": [r[2] for r in rows],
        }
    tr_pids = {canonical_pid(x) for x in out["train"]["ids"]}
    ho_pids = {canonical_pid(x) for x in out["holdout"]["ids"]}
    assert not (tr_pids & ho_pids), "train/holdout canonical pid overlap!"
    assert not (tr_pids & banned) and not (ho_pids & banned), "test/ref overlap!"
    return out, manifest


def tasks_and_sources():
    tasks = read_json(ROOT / "eval_protocol/tasks_750.json")["tasks"]
    cache = torch.load(
        ROOT / "runs/v9_local_4070/cache/n2c_z.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert len(tasks) == 750 == len(cache["z_src"])
    return tasks, cache["z_src"].float(), cache


class RawWriterV18:
    """Writes measure()-compatible raw batches atomically."""

    def __init__(self, arm_name: str, raw_root: Path | None = None):
        self.arm_name = arm_name
        self.path = (raw_root or (OUT / "raw")) / arm_name
        self.path.mkdir(parents=True, exist_ok=True)
        self.batches: list[dict] = []
        self.n = 0

    def append(self, ids: list[str], rgb01: torch.Tensor, z_final: torch.Tensor | None = None):
        fname = f"batch_{self.n:05d}.pt"
        u8 = (rgb01.detach().cpu().float() * 255.0).round().clamp(0, 255).byte()
        payload = {"ids": list(ids), "rgb_u8": u8}
        if z_final is not None:
            payload["z_final"] = z_final.detach().cpu().half()
        tmp = self.path / (fname + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(self.path / fname)
        self.batches.append({"offset": self.n, "file": fname, "ids": list(ids)})
        self.n += len(ids)

    def finish(self, **extra):
        assert self.n == 750, f"expected 750 tasks, got {self.n}"
        all_ids = [tid for b in self.batches for tid in b["ids"]]
        assert len(set(all_ids)) == 750, "duplicate task_ids in RawWriterV18"
        meta = {
            "kind": "v18",
            "arm": self.arm_name,
            "n_tasks": 750,
            "decoder": "S10",
            "batches": self.batches,
            **extra,
        }
        save_json(self.path / "meta.json", meta)
        return self.path


def run_measure_process(arm_name: str, device: str = "cuda") -> dict:
    """Run src.eval.measure in a fresh child process (strict VRAM isolation) and summarize."""
    raw_dir = OUT / "raw" / arm_name
    meas_dir = OUT / "measured" / arm_name
    if not (meas_dir / "eval.json").exists():
        r = subprocess.run(
            [
                sys.executable, "-u", "-m", "src.eval.measure",
                "--raw", str(raw_dir),
                "--out", str(meas_dir),
                "--device", device,
                "--no-png",
            ],
            cwd=ROOT,
        )
        if r.returncode != 0:
            raise RuntimeError(f"src.eval.measure failed for {arm_name} (rc={r.returncode})")
    return summarize_measured_arm(arm_name, meas_dir)


def summarize_measured_arm(arm_name: str, meas_dir: Path) -> dict:
    d = read_json(meas_dir / "eval.json")
    rows = d["rows"]
    assert d["n_total"] == 750 and d["n_cross"] == 600 and d["n_same"] == 150

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
    summary = {
        "arm": arm_name,
        "utc": utc(),
        "cross600": _stats(cross_rows),
        "same150": _stats(same_rows),
        "per_target_cross120": {
            s: _stats([r for r in cross_rows if r["tgt_style"] == s]) for s in STYLES
        },
        "confusion_5x5": [
            [
                sum(
                    1
                    for r in cross_rows
                    if r["tgt_style"] == STYLES[i]
                    and int(np.argmax(r["csd_scores_5"])) == j
                )
                for j in range(5)
            ]
            for i in range(5)
        ],
    }
    save_json(meas_dir / "summary_v18.json", summary)
    c = summary["cross600"]
    pt = {k[:3]: round(v["csd_top1"] * 100, 1) for k, v in summary["per_target_cross120"].items()}
    print(
        f"[V18 EVAL] {arm_name:24s} | Cross600 Top1={c['csd_top1']*100:5.2f}% "
        f"Mar={c['csd_margin']:+.4f} DINO-S={c['dino_s']:.4f} DINO-C={c['dino_c']:.4f} "
        f"LPIPS={c['lpips']:.4f} SrcLock={c['pred_eq_src']*100:4.1f}% | {pt}",
        flush=True,
    )
    return summary
