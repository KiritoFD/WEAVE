"""Clean, NFC-deduplicated, 1:1:1:1:1 balanced latent pool loader with strict Test750/Reference firewall."""
from __future__ import annotations

import hashlib
from pathlib import Path
import torch

from .protocol import STYLES, canonical_pid
from .runtime import REPO, read_json


def split_hash10(pid: str) -> int:
    return int(hashlib.md5(pid.encode("utf-8")).hexdigest()[:8], 16) % 10


def clean_balanced_pool(
    pool_dir: str | Path,
    train_per_style: int = 880,
    hold_per_style: int = 60,
    seed: int = 20261118,
    min_train: int = 50,
    min_hold: int = 10,
) -> tuple[dict, dict]:
    """Load strictly deduplicated, balanced latents excluding all Test750 sources & references_v1."""
    tasks = read_json(REPO / "eval_protocol/tasks_750.json")["tasks"]
    refs = read_json(REPO / "eval_protocol/references_v1.json")["references"]
    banned = {canonical_pid(t["source_stem"]) for t in tasks}
    banned |= {canonical_pid(r["stem"]) for rows in refs.values() for r in rows}

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
            if any(split_hash10(ids[r]) < 9 for r in rows):
                pick = next(r for r in rows if split_hash10(ids[r]) < 9)
                tr_cands.append(pick)
            else:
                ho_cands.append(rows[0])

        n_tr = min(train_per_style, len(tr_cands))
        n_ho = min(hold_per_style, len(ho_cands))
        if n_tr < min_train or n_ho < min_hold:
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
    if tr_pids & ho_pids:
        raise AssertionError("train/holdout canonical PID overlap!")
    if (tr_pids & banned) or (ho_pids & banned):
        raise AssertionError("test/reference PID overlap!")
    return out, manifest
