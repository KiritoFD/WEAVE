"""Strip per-row CSD embeddings so evaluation JSONs are git-friendly.

The 768-d `csd_embedding` per row dominates file size (2.8 MB per pair eval).
Aggregates, MMD^2 and every scalar metric are preserved verbatim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--suffix", default="_compact.json")
    args = ap.parse_args()
    for f in args.files:
        p = Path(f)
        d = json.loads(p.read_text(encoding="utf-8"))
        for r in d.get("rows", []):
            r.pop("csd_embedding", None)
        out = p.with_name(p.stem + args.suffix)
        out.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{out.name}  {out.stat().st_size/1024:.0f} KB (from {p.stat().st_size/1048576:.2f} MB)")


if __name__ == "__main__":
    main()
