"""Orchestrate: wait for 3060 eval (EVAL_ALL_DONE) -> tar-download raw
payloads -> run the official local referee (src.eval.measure) per arm/level.

Run locally: python -m exp.v15.orchestrate_redownload
Progress: G:/WEAVE/runs/v15/orchestrate.log
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

REPO = Path("G:/WEAVE")
RUNS = REPO / "runs/v15"
RAW = RUNS / "eval/raw"
LOG = RUNS / "orchestrate.log"
LEVELS = ["0.0", "0.4", "0.7"]
LOG_FH = open(LOG, "a", encoding="utf-8")


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    LOG_FH.write(line + "\n")
    LOG_FH.flush()
    print(line, flush=True)


def run(cmd, **kw):
    log("$ " + cmd if isinstance(cmd, str) else " ".join(cmd))
    return subprocess.run(cmd, shell=isinstance(cmd, str), cwd=REPO,
                          stdout=LOG_FH, stderr=subprocess.STDOUT, **kw)


def main():
    log("=== orchestrate start ===")
    # 1. wait for EVAL_ALL_DONE (poll 5 min, up to 4h)
    for _ in range(48):
        r = subprocess.run(
            'ssh -o BatchMode=yes 3060 '
            '"type I:/Latent_Style/WEAVE/runs/v15/eval_run.log"',
            shell=True, capture_output=True, text=True)
        if "EVAL_ALL_DONE" in r.stdout:
            log("eval complete on 3060")
            break
        log(f"eval not done yet (tail: ...{r.stdout.strip()[-60:]})")
        time.sleep(300)
    else:
        log("TIMEOUT waiting for eval; aborting")
        return

    # 2. tar-stream download of raw payloads
    t0 = time.time()
    run('ssh -o BatchMode=yes 3060 "tar cf - -C '
        'I:/Latent_Style/WEAVE/runs/v15/eval raw" '
        f'> {RUNS / "eval/raw.tar"}')
    size_gb = (RUNS / "eval/raw.tar").stat().st_size / 1e9
    log(f"downloaded raw.tar {size_gb:.1f}GB in {time.time()-t0:.0f}s")

    t0 = time.time()
    run(f"tar xf {RUNS / 'eval/raw.tar'} -C {RUNS / 'eval'}")
    log(f"extracted in {time.time()-t0:.0f}s")
    (RUNS / "eval/raw.tar").unlink()

    # 3. local referee per arm/level
    arms = sorted(p.name for p in RAW.iterdir() if p.is_dir())
    log(f"referee over arms: {arms}")
    for arm in arms:
        for lev in LEVELS:
            raw_dir = RAW / arm / f"t0_{lev}"
            if not (raw_dir / "meta.json").exists():
                log(f"SKIP missing meta: {raw_dir}")
                continue
            out = RUNS / "eval/measured" / arm / f"t0_{lev}"
            if (out / "eval.json").exists():
                log(f"SKIP done: {out}")
                continue
            t1 = time.time()
            r = run(f'python -m src.eval.measure --raw "{raw_dir}" '
                    f'--out "{out}" --device cuda')
            log(f"{arm} t0_{lev}: rc={r.returncode} "
                f"{time.time()-t1:.0f}s")
    log("=== orchestrate done ===")


if __name__ == "__main__":
    main()
