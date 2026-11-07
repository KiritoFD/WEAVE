"""Check frozen P0 source-feature cache alignment for the C3 quick sweep."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    tasks = json.loads((ROOT / "runs" / "v5" / "v5_eval200_tasks.json").read_text(encoding="utf-8"))
    stems = sorted({t["src_stem"] for t in tasks})
    feats = torch.load(ROOT / "runs" / "v5" / "cache" / "eval_src_csd_feats.pt",
                       map_location="cpu", weights_only=False)
    missing = [s for s in stems if s not in feats]
    extra = [k for k in feats if k not in stems]
    print(f"tasks={len(tasks)} stems={len(stems)} feats={len(feats)} missing={len(missing)} extra={len(extra)}")
    if missing:
        print("MISSING first:", missing[:5])
    if extra:
        print("EXTRA first:", extra[:5])
    if missing:
        raise SystemExit(1)
    v = feats[stems[0]]
    print(f"sample feat shape={tuple(v.shape) if hasattr(v, 'shape') else len(v)} dtype={getattr(v, 'dtype', type(v))}")
    print("ALIGNMENT_OK")


if __name__ == "__main__":
    main()
