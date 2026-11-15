"""Orchestrate: wait for 3060 PNG encoding -> tar-stream download -> verify.

Run locally: python -m exp.v15.orchestrate_png
Progress: G:/WEAVE/runs/v15/orchestrate.log
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

REPO = Path("G:/WEAVE")
RUNS = REPO / "runs/v15"
PNG_LOCAL = RUNS / "eval/png"
LOG = RUNS / "orchestrate.log"
LOG_FH = open(LOG, "a", encoding="utf-8")


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    LOG_FH.write(line + "\n")
    LOG_FH.flush()
    print(line, flush=True)


def ssh(cmd: str, timeout: int = 120) -> str:
    return subprocess.run(f'ssh -o BatchMode=yes 3060 "{cmd}"', shell=True,
                          capture_output=True, text=True,
                          timeout=timeout).stdout


def main():
    log("=== orchestrate png start ===")
    for _ in range(40):
        out = ssh("type I:/Latent_Style/WEAVE/runs/v15/png_run.log")
        if "PNG_DONE" in out:
            total = [l for l in out.splitlines() if "PNG_DONE" in l][0]
            log(f"remote encoding complete: {total.strip()}")
            break
        log(f"encoding pending (tail ...{out.strip()[-50:]})")
        time.sleep(300)
    else:
        log("TIMEOUT waiting for encoding; aborting")
        return

    PNG_LOCAL.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log("tar-stream download (png/ -> local)...")
    p1 = subprocess.Popen(
        'ssh -o BatchMode=yes 3060 "tar cf - -C '
        'I:/Latent_Style/WEAVE/runs/v15/eval png"',
        shell=True, stdout=subprocess.PIPE)
    p2 = subprocess.Popen(f'tar xf - -C "{PNG_LOCAL.as_posix()}"', shell=True,
                          stdin=p1.stdout, stderr=subprocess.PIPE)
    p2.wait()
    p1.wait()
    log(f"download+extract rc={p2.returncode} {time.time()-t0:.0f}s")

    n = sum(1 for _ in PNG_LOCAL.rglob("*.png"))
    gb = sum(p.stat().st_size for p in PNG_LOCAL.rglob("*.png")) / 1e9
    log(f"local verify: {n} pngs, {gb:.2f}GB at {PNG_LOCAL}")
    log("=== orchestrate png done ===")


if __name__ == "__main__":
    main()
