"""One-shot: incrementally encode + tar-stream + verify PNGs for given arms.

Run locally:  python -m exp.v15.pull_png --arms W_dim96_conv_base
Remote encode skips existing PNGs, so repeat calls are near-instant.
"""
import argparse
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EVAL_LOCAL = REPO / "runs/v15/eval"
REMOTE_EVAL = "I:/Latent_Style/WEAVE/runs/v15/eval"


def ssh(cmd: str, timeout: int = 900) -> str:
    return subprocess.run(f'ssh -o BatchMode=yes 3060 "{cmd}"', shell=True,
                          capture_output=True, text=True,
                          timeout=timeout).stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="W_dim96_conv_base")
    a = ap.parse_args()
    arms = "+".join(a.arms.split(","))
    t0 = time.time()

    out = ssh(f"cd /d I:/Latent_Style/WEAVE && "
              f"python -m exp.v15.encode_png --arms {a.arms} "
              f"2>nul | findstr /c:PNG_DONE")
    print(out.strip() or "(no PNG_DONE line — check remote log)", flush=True)

    (EVAL_LOCAL / "png").mkdir(parents=True, exist_ok=True)
    print("tar-stream download...", flush=True)
    p1 = subprocess.Popen(
        f'ssh -o BatchMode=yes 3060 "tar cf - -C {REMOTE_EVAL} '
        f'png/{arms}"',
        shell=True, stdout=subprocess.PIPE)
    p2 = subprocess.Popen(f'tar xf - -C "{EVAL_LOCAL.as_posix()}"', shell=True,
                          stdin=p1.stdout, stderr=subprocess.PIPE)
    p2.wait()
    p1.wait()

    d = EVAL_LOCAL / "png" / arms
    pngs = list(d.rglob("*.png"))
    gb = sum(p.stat().st_size for p in pngs) / 1e9
    print(f"LOCAL {d}: {len(pngs)} pngs {gb:.2f}GB  "
          f"({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
