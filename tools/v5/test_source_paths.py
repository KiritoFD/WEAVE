"""Verify the corrected source_latents path unpacking (no GPU model load)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.eval_fast import build_protocol
from tools.v5 import route_c


def main():
    test_images, requests, style_meta = build_protocol(
        ROOT / "data" / "test", max_src_samples=30, ref_pool="first"
    )
    src_paths = [p for _, (_, paths) in sorted(test_images.items()) for p in paths]
    bad = [p for p in src_paths if not isinstance(p, Path) or not p.exists()]
    print(f"styles={len(test_images)} requests={len(requests)} src_paths={len(src_paths)} bad={len(bad)}")
    print(f"first3={[str(p) for p in src_paths[:3]]}")
    req_keys = {(str(r["src_path"]), int(r["tgt_style_id"])) for r in requests}
    missing = [(str(p), t) for p in src_paths for t in range(len(style_meta))
               if (str(p), t) not in req_keys]
    print(f"req_map_coverage missing={len(missing)}")
    assert not bad and len(src_paths) == 150 and len(requests) == 750 and not missing
    print("SOURCE_LATENTS_FIX_OK")


if __name__ == "__main__":
    main()
