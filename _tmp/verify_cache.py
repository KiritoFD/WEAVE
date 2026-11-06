"""Verify pseudo-GT paired cache integrity: entry counts, id uniqueness, tensor shapes, checksums."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path("/mnt/i/Latent_Style/pseudo_gt_24h/pairs")


def digest(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def check(arm: str) -> None:
    d = ROOT / arm
    man_p = d / "manifest.json"
    if not man_p.exists():
        print(f"[{arm}] MISSING manifest")
        return
    man = json.loads(man_p.read_text(encoding="utf-8"))
    records = man.get("records", [])
    print(f"[{arm}] manifest records={len(records)} md5={digest(man_p)[:12]}")
    for k in ("method", "spec", "versions", "arm"):
        if k in man:
            print(f"  {k}={man[k]}")

    shards = sorted(d.glob("shard_*.pt"))
    total = 0
    ids: set[str] = set()
    dup = 0
    shape = None
    dtype = None
    for sp in shards:
        sh = torch.load(sp, map_location="cpu", weights_only=False)
        n = len(sh["meta"])
        total += n
        for m in sh["meta"]:
            sid = m.get("sample_id")
            if sid in ids:
                dup += 1
            ids.add(sid)
        if shape is None and n:
            shape = tuple(sh["z0"][0].shape)
            dtype = str(sh["z0"].dtype)
        print(f"  {sp.name}: rows={n} bytes={sp.stat().st_size} md5={digest(sp)[:12]}")

    miss = sum(1 for r in records if r.get("sample_id") not in ids)
    print(f"[{arm}] shards={len(shards)} rows={total} manifest={len(records)} "
          f"dup_ids={dup} missing_ids={miss} z0_shape={shape} dtype={dtype}")
    if total != len(records) or dup or miss:
        print(f"[{arm}] STATUS=MISMATCH")
    else:
        print(f"[{arm}] STATUS=OK")
    rec_keys = set(records[0].keys()) if records else set()
    print(f"[{arm}] record_keys={sorted(rec_keys)}")


def main() -> int:
    arms = sys.argv[1:] or ["base", "positive", "reverse"]
    for arm in arms:
        if (ROOT / arm).exists():
            check(arm)
        else:
            print(f"[{arm}] not built yet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
