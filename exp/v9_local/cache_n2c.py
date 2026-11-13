"""Merge stripped N2C.e gen_latents batches into a per-task cache + verify."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
TASKS = json.loads((REPO / "eval_protocol/tasks_750.json").read_text(encoding="utf-8"))["tasks"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(REPO / "runs/v8/N2C.e/gen_latents"))
    ap.add_argument("--out", default=str(REPO / "runs/v9_local_4070/cache/n2c_z.pt"))
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    order = {t["task_id"]: i for i, t in enumerate(TASKS)}
    z_src = [None] * 750
    z0 = [None] * 750
    z_n2 = [None] * 750
    ids_seen = []
    for p in sorted(src.glob("batch_*.pt")):
        d = torch.load(p, map_location="cpu", weights_only=False)
        for i, tid in enumerate(d["ids"]):
            j = order[tid]
            ids_seen.append(tid)
            z_src[j] = d["z_src"][i]
            z0[j] = d["z0"][i]
            z_n2[j] = d["z_final"][i]
    assert len(ids_seen) == 750 and len(set(ids_seen)) == 750, \
        f"expected 750 unique tasks, got {len(set(ids_seen))}"
    assert all(x is not None for x in z_src + z0 + z_n2)
    assert z_src[0].shape == z0[0].shape == z_n2[0].shape
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"task_ids": [t["task_id"] for t in TASKS],
                "z_src": torch.stack(z_src), "z0": torch.stack(z0),
                "z_n2": torch.stack(z_n2)}, out)
    print("saved", out, out.stat().st_size)


if __name__ == "__main__":
    main()
