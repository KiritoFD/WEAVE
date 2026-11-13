#!/usr/bin/env python3
"""Hist2Style 750-task batch runner (WEAVE reproduction, official inference path).

Reuses the OFFICIAL run.py functions verbatim (load_model / style_histogram /
stylize / configure_performance / select_device) so the model, histogram
conditioning, downsample rule (<=2MP) and output encoding match run.py exactly.
Only the pair enumeration differs: we iterate the frozen WEAVE 750-task
manifest (same content/style pairs as the StyleGallery reproduction) instead of
folder pairing.

Outputs {index:04d}_{sha256(task_id)[:10]}.png + execution.jsonl, same naming
as batch_stylegallery_12g.py / batch_stylegallery_fast.py for cross-baseline
alignment. Resumable (existing png -> SKIP). Per-task failures recorded and
run continues; DONE marker when manifest fully covered.

Speed note: this runner keeps the 1.5M-param model resident, so per-task
cold_wall_s EXCLUDES model load. Official run.py reloads per invocation;
report both numbers (see REPRO.md).
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True, type=Path, help='hist2style repo root')
    ap.add_argument('--manifest', required=True, type=Path)
    ap.add_argument('--out', default='h2s750_outputs', type=Path)
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--max-pixels', type=int, default=2000000)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--max-tasks', type=int, default=0)
    args = ap.parse_args()

    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))          # for `import run`
    import run as h2s

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    ckpt = Path(args.checkpoint) if args.checkpoint else h2s.DEFAULT_CHECKPOINT

    with args.manifest.open(encoding='utf-8', newline='') as fp:
        rows = list(csv.DictReader(fp))
    if args.limit:
        rows = rows[:args.limit]

    h2s.configure_performance()
    device = torch.device(args.device) if args.device != 'auto' else h2s.select_device('auto')

    t_load = time.monotonic()
    model = h2s.load_model(ckpt.resolve(), device)
    load_s = time.monotonic() - t_load
    print(f'[h2s] checkpoint={ckpt} device={device} load={load_s:.1f}s', flush=True)

    hist_cache = {}
    logpath = out / 'execution.jsonl'
    processed = failures = 0
    hit_restart = False

    for i, row in enumerate(rows):
        if i < args.start:
            continue
        tid = row['task_id']
        safe_id = f'{i:04d}_' + hashlib.sha256(tid.encode()).hexdigest()[:10]
        result = out / f'{safe_id}.png'
        if result.is_file():
            print(f'[SKIP] {i+1}/{len(rows)} {tid}', flush=True)
            continue
        refs = row['style_paths'].split(';')
        content_path = Path(row['source_path'])
        style_path = Path(refs[0])  # manifest is refs_per_style=1

        started = time.monotonic()
        success = False
        try:
            if style_path not in hist_cache:
                hist_cache[style_path] = h2s.style_histogram(style_path, device)
            img = h2s.stylize(model, content_path, hist_cache[style_path],
                              device, args.max_pixels)
            img.save(result)
            success = result.is_file()
        except Exception:
            (out / f'{safe_id}.error.log').write_text(traceback.format_exc(),
                                                   encoding='utf8')
            print(f'[ERR] {i+1}/{len(rows)} {tid}', flush=True)

        elapsed = time.monotonic() - started
        metadata = {'index': i, 'task_id': tid,
                    'source_style': row['source_style'],
                    'target_style': row['target_style'],
                    'is_cross': int(row['is_cross']),
                    'source_path': str(content_path), 'ref_paths': refs,
                    'result': str(result) if success else None,
                    'returncode': 0 if success else 1, 'success': bool(success),
                    'infer_wall_s': round(elapsed, 3),
                    'runner': 'hist2style_resident',
                    'checkpoint': str(ckpt), 'max_pixels': args.max_pixels,
                    'device': str(device)}
        with logpath.open('a', encoding='utf8') as fh:
            fh.write(json.dumps(metadata, ensure_ascii=False) + '\n')
        failures += (not success)
        processed += 1
        print(f"[{'OK' if success else 'FAIL'}] {i+1}/{len(rows)} {tid} "
              f"{elapsed:.2f}s", flush=True)

        if args.max_tasks and processed >= args.max_tasks:
            hit_restart = True
            break

    if not hit_restart:
        (out / 'DONE').write_text(
            f'processed={processed} failures={failures} load_s={load_s:.1f}\n')
    print(f'[h2s] exiting processed={processed} failures={failures}', flush=True)


if __name__ == '__main__':
    main()
