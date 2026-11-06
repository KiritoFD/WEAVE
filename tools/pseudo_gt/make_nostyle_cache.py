"""生成“无 Style-ID”对照缓存：shard 软链接 + manifest 中 target_style 常量。

用法：python tools/pseudo_gt/make_nostyle_cache.py --src pairs/pattern_R64_reverse --dst pairs/pattern_R64_reverse_nostyle
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.pseudo_gt.build_assets import pgt_root  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--style", default="Early_Renaissance")
    args = ap.parse_args()
    src = pgt_root() / args.src if not Path(args.src).is_absolute() else Path(args.src)
    dst = pgt_root() / args.dst if not Path(args.dst).is_absolute() else Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    for sh in sorted(src.glob("shard_*.pt")):
        link = dst / sh.name
        if not link.exists():
            os.symlink(os.path.relpath(sh, dst), link)
    mf = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    recs = mf["records"]
    for r in recs:
        r["target_style"] = args.style
    (dst / "manifest.json").write_text(
        json.dumps({"count": len(recs), "shards": len(sorted(src.glob("shard_*.pt"))),
                    "dtype": mf.get("dtype", "float16"), "records": recs, "nostyle": True},
                   ensure_ascii=False), encoding="utf-8")
    print(f"[nostyle] {len(recs)} records, {len(sorted(src.glob('shard_*.pt')))} shards -> {dst}")


if __name__ == "__main__":
    main()
