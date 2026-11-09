"""Fetch every eval_*/style_scores.json and manifest.json from 3060 (JSON only)."""
from __future__ import annotations

import subprocess
from pathlib import Path

REMOTE = "/mnt/i/Latent_Style/WEAVE/runs/blog_sbflow"
LOCAL = Path("runs/blog_sbflow")


def remote_dirs() -> list[str]:
    out = subprocess.run(
        ["ssh", "3060", f"wsl ls -1 {REMOTE}"],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    return [d for d in out if d.startswith("eval_")]


def pull(rel: str) -> None:
    dst = LOCAL / rel
    if dst.exists() and dst.stat().st_size > 0:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("wb") as f:
        r = subprocess.run(["ssh", "3060", f"wsl cat {REMOTE}/{rel}"], stdout=f, stderr=subprocess.DEVNULL)
    if r.returncode != 0 or dst.stat().st_size == 0:
        dst.unlink(missing_ok=True)
        print(f"SKIP {rel}")
        return
    print(f"{rel}  {dst.stat().st_size/1024:.0f} KB")


def main() -> None:
    for d in remote_dirs():
        for f in ("manifest.json", "style_scores.json"):
            pull(f"{d}/{f}")


if __name__ == "__main__":
    main()
