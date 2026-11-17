"""Independent referee process: measure an already-generated raw dir.

Run: python -m exp.v17.measure_only --arm P2_S1_eval
Reads runs/v17/raw/<arm>/, writes runs/v17/measured/<arm>/ + DONE.json.
Separate process => generation models are fully released before CSD/DINO/
CLIP/LPIPS load, avoiding the 12GB OOM that killed inline measure.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / 'runs/v17'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True)
    a = ap.parse_args()
    raw = RUNS / 'raw' / a.arm
    meta = json.loads((raw / 'meta.json').read_text(encoding='utf-8'))
    measured = RUNS / 'measured' / a.arm
    r = subprocess.run([sys.executable, '-u', '-m', 'src.eval.measure',
                        '--raw', str(raw), '--out', str(measured),
                        '--device', 'cuda', '--no-png'], cwd=REPO)
    if r.returncode != 0:
        print(f'MEASURE_FAIL {a.arm}', flush=True)
        raise SystemExit(1)
    from .eval750 import finish_eval
    summary, done = finish_eval(a.arm, measured, raw, meta, RUNS / 'raw')
    print(f'MEASURE_OK {a.arm} top1={summary["cross600"]["csd_top1"]:.4f} '
          f'dino_s={summary["cross600"]["dino_s"]:.4f}', flush=True)


if __name__ == '__main__':
    main()
