"""Cross-check the delivered test images against the SHA256 recorded in the protocol manifest.

The bundle's own MANIFEST proves the bytes survived transfer. This proves something stronger: that
these are exactly the images tasks_750.json says they are, so the delivered set can reproduce the
reported numbers rather than merely looking plausible.

Matching is done by CONTENT HASH, not by path, because one filename contains a Unicode mojibake that
differs between the protocol JSON and the files on disk:

  protocol : ...joaqu + U+82AB U+9845 + n-sorolla...
  on disk  : ...joaqu + U+00E3 + U+00AD + n-sorolla...

Both are corrupted renderings of 'joaquín'. The image itself is byte-identical (same SHA256), so the
delivered set is complete; only that one path string cannot be used verbatim to open the file. Any
loader should fall back to a hash lookup or normalise the stem.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TS = ROOT / "deliver" / "testset"


def main() -> None:
    proto = json.loads((TS / "protocol" / "tasks_750.json").read_text())
    tasks = proto["tasks"]
    want_by_path = {t["source_rel_path"]: t["source_sha256"] for t in tasks}
    want_hashes = set(want_by_path.values())
    print(f"protocol references {len(want_by_path)} unique source images")

    imgs = sorted((TS / "data" / "test").glob("*/*.jpg"))
    have = {}
    for p in imgs:
        have[hashlib.sha256(p.read_bytes()).hexdigest()] = p

    matched = want_hashes & set(have)
    missing = want_hashes - set(have)
    extra = set(have) - want_hashes
    print(f"  by content hash: matched {len(matched)}/{len(want_hashes)}, "
          f"missing {len(missing)}, extra {len(extra)}")

    # path-level agreement, reported separately so the encoding issue is visible
    path_ok = sum(1 for rel in want_by_path if (TS / rel).exists())
    print(f"  by exact path  : {path_ok}/{len(want_by_path)} "
          f"({len(want_by_path)-path_ok} path string(s) differ -- see module docstring)")

    c = Counter(p.parent.name for p in imgs)
    print(f"\n  delivered images: {len(imgs)}")
    for k in sorted(c):
        sz = sum(p.stat().st_size for p in imgs if p.parent.name == k)
        print(f"    {k:22s} {c[k]:3d}  {sz/1024/1024:5.1f} MB")

    pairs = Counter((t["source_style"], t["target_style"])
                    for t in tasks if t["target_style"] != t["source_style"])
    print(f"\n  cross-style pairs: {len(pairs)}/20, "
          f"tasks per pair min={min(pairs.values())} max={max(pairs.values())}")

    good = (not missing) and (not extra)
    print(f"\n  {'ALL 150 IMAGES PRESENT AND BYTE-IDENTICAL TO THE PROTOCOL' if good else 'MISMATCH'}")


if __name__ == "__main__":
    main()
