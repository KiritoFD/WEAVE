"""Encode V16 raw payloads (runs/v16/raw/<arm>/) to 512px lossless PNGs.

Run on the 3060: python -m exp.v17.encode_v16_png --arms B_adapter,B_base
Output: runs/v16/png/<arm>/<task_id>.png
"""
import argparse
import time
from pathlib import Path

import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / 'runs/v16/raw'
PNG = REPO / 'runs/v16/png'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', required=True)
    a = ap.parse_args()
    keep = set(a.arms.split(','))
    n = 0
    t0 = time.time()
    for meta in sorted(RAW.glob('*/meta.json')):
        if meta.parent.name not in keep:
            continue
        out_dir = PNG / meta.parent.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for bp in sorted(meta.parent.glob('batch_*.pt')):
            payload = torch.load(bp, map_location='cpu', weights_only=False)
            u8 = payload['rgb_u8']
            for tid, arr in zip(payload['ids'],
                                u8.permute(0, 2, 3, 1).numpy()):
                p = out_dir / (tid.replace('/', '__') + '.png')
                if not p.exists():
                    Image.fromarray(arr).save(p, format='PNG')
                    n += 1
        print(f"{meta.parent.name}: new={n} {time.time()-t0:.0f}s", flush=True)
    total = sum(p.stat().st_size for p in PNG.rglob('*.png'))
    print(f"V16PNG_DONE new={n} total={total/1e9:.2f}GB "
          f"{time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
