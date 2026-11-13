"""Painting-level train/holdout split (pre-registered: md5(painting_id)%10<9 -> train).

Integrity: no id overlap, counts per style, and zero overlap between the 150
eval task sources and the whole pool (re-verified here, plan section 3).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
POOL_DIR = REPO / "weights_local/sd_pool_orig"
OUT = REPO / "runs/v10_local_4070"


def _h10(pid: str) -> int:
    return int(hashlib.md5(pid.encode()).hexdigest()[:8], 16) % 10


def build_splits(pool_dir: Path = POOL_DIR):
    """-> dict style -> {"train_idx", "holdout_idx", "n_total"} (torch.LongTensor)."""
    splits = {}
    for si, s in enumerate(STYLES):
        d = torch.load(pool_dir / f"{si:02d}_{s}.pt",
                       map_location="cpu", weights_only=False)
        ids = [str(x) for x in d["ids"]]
        tr, ho = [], []
        for i, pid in enumerate(ids):
            (tr if _h10(pid) < 9 else ho).append(i)
        splits[s] = {"train_idx": torch.tensor(tr, dtype=torch.long),
                     "holdout_idx": torch.tensor(ho, dtype=torch.long),
                     "n_total": len(ids)}
    return splits


def check_integrity(splits, tasks_path: Path = REPO / "eval_protocol/tasks_750.json") -> dict:
    report = {"styles": {}, "source_overlap": None}
    pool_stems = set()
    for s, sp in splits.items():
        tr = set(sp["train_idx"].tolist())
        ho = set(sp["holdout_idx"].tolist())
        assert not (tr & ho), f"{s}: train/holdout overlap"
        report["styles"][s] = {"train": len(tr), "holdout": len(ho),
                               "total": sp["n_total"]}
    for si, s in enumerate(STYLES):
        d = torch.load(POOL_DIR / f"{si:02d}_{s}.pt",
                       map_location="cpu", weights_only=False)
        pool_stems.update(Path(str(i)).stem for i in d["ids"])
    tasks = json.loads(Path(tasks_path).read_text(encoding="utf-8"))["tasks"]
    srcs = {t["source_stem"] for t in tasks}
    report["source_overlap"] = len(pool_stems & srcs)
    assert report["source_overlap"] == 0, "eval sources leaked into the pool!"
    return report


if __name__ == "__main__":
    sp = build_splits()
    rep = check_integrity(sp)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(sp, OUT / "splits.pt")
    (OUT / "splits_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
