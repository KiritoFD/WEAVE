"""Push selected local files to 3060 (tar.gz | base64 via ssh stdin)."""
from __future__ import annotations

import base64
import io
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(r"M:\WEAVE")
REMOTE = "/mnt/i/Latent_Style/WEAVE"

FILES = [
    "config_schema.py",
    "flow.py",
    "paths.py",
    "run.py",
    "utils/dataset.py",
    "tools/pseudo_gt/__init__.py",
    "tools/pseudo_gt/masks.py",
    "tools/pseudo_gt/matching.py",
    "tools/pseudo_gt/construct.py",
    "tools/pseudo_gt/cache.py",
    "tools/pseudo_gt/build_assets.py",
    "tools/pseudo_gt/q0_quality.py",
    "tools/pseudo_gt/select.py",
    "tools/pseudo_gt/build_train_cache.py",
    "tools/pseudo_gt/pgt_render.py",
    "configs/pgt/S0_base.json",
    "configs/pgt/S1_positive.json",
    "configs/pgt/S2_reverse.json",
    "tests/test_pseudo_gt_masks.py",
    "tests/test_pseudo_gt_construct.py",
    "tests/test_paired_latent_passthrough.py",
]


def main() -> int:
    files = sys.argv[1:] or FILES
    buf = io.BytesIO()
    count = 0
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for rel in files:
            path = ROOT / rel
            if not path.exists():
                print(f"SKIP missing {rel}")
                continue
            tf.add(path, arcname=rel)
            count += 1
    data = buf.getvalue()
    print(f"payload {count} files, tgz={len(data)}B")

    decode = subprocess.run(
        ["ssh", "3060", 'wsl -e bash -c "base64 -d > /tmp/pgt_sync.tgz"'],
        input=base64.b64encode(data).decode(),
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if decode.returncode != 0:
        print("FAIL upload:", decode.stderr[:400])
        return 1
    extract = subprocess.run(
        ["ssh", "3060", f'wsl -e bash -c "tar -xzf /tmp/pgt_sync.tgz -C {REMOTE} && echo EXTRACTED"'],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if extract.returncode != 0:
        print("FAIL extract:", extract.stderr[:400])
        return 1
    print("OK", extract.stdout.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
