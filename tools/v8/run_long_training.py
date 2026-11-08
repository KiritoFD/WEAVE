"""V8 long-training driver: B0 / fixed-B2 to 3k and 8k updates with checkpoints.

Stage P1 of the P0 review. Keeps sigma_mid=0 (CFM), fixed effective batch 120,
cosine LR schedule over the TOTAL step count, and saves checkpoints at the
requested milestones so 800 / 3k / 8k can be compared on the same protocol.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path("/mnt/i/Latent_Style/WEAVE")


def run(cmd, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print(">>>", " ".join(cmd), flush=True)
    with log.open("a", encoding="utf-8") as f:
        f.write("\n>>> " + " ".join(cmd) + "\n")
        f.flush()
        subprocess.run(cmd, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--which", default="B0,B2")
    ap.add_argument("--checkpoints", default="800,3000,8000")
    args = ap.parse_args()

    py = sys.executable
    logs = REPO_ROOT / "runs" / "v8" / "logs"
    want = {w.strip() for w in args.which.split(",") if w.strip()}
    ckpts = ",".join(s for s in args.checkpoints.split(",") if int(s) <= args.steps)

    if "B0" in want:
        run([
            py, "-u", "tools/v8/train.py",
            "--name", "B0", "--tag", f"B0_long{args.steps}",
            "--pairing", "random", "--sigma-mid", "0.0",
            "--steps", str(args.steps), "--microbatch", "16", "--accum", "8",
            "--lr", "2e-4", "--seed", "42", "--ema-decay", "0.999",
            "--checkpoints", ckpts,
        ], logs / f"B0_long{args.steps}.log")

    if "B2" in want:
        run([
            py, "-u", "tools/v8/train.py",
            "--name", "B2", "--tag", f"B2_long{args.steps}",
            "--pairing", "ot", "--sigma-mid", "0.0",
            "--steps", str(args.steps), "--microbatch", "16", "--accum", "8",
            "--lr", "2e-4", "--seed", "42", "--ema-decay", "0.999",
            "--checkpoints", ckpts,
        ], logs / f"B2_long{args.steps}.log")

    print("V8 long training complete")


if __name__ == "__main__":
    main()
