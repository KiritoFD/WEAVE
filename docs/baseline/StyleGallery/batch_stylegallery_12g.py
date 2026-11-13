#!/usr/bin/env python3
"""Strict sequential, resumable StyleGallery batch protocol for RTX 3060 12GB.

Runs the official demo_accelerate.py as a fresh subprocess per task.
This is conservative about VRAM and crash isolation, but starts weights for
EVERY TASK; its total wall-clock time is NOT comparable to a warm end-to-end
inference benchmark. Logs the measured duration and nvidia-smi usage anyway.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time


def mem_sampler(stop, samples):
    while not stop.is_set():
        try:
            p = subprocess.run(['nvidia-smi','--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                               capture_output=True, text=True, timeout=5)
            if p.returncode == 0:
                m = re.search(r'\d+', p.stdout)
                if m: samples.append(int(m.group()))
        except (OSError, subprocess.TimeoutExpired):
            return
        stop.wait(0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True, type=Path, help='StyleGallery repo root')
    ap.add_argument('--manifest', required=True, type=Path)
    ap.add_argument('--out', default='sg750_outputs', type=Path)
    ap.add_argument('--mode', default='lcm',choices=['lcm','hyper'])
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--steps', type=int, default=28)
    ap.add_argument('--iters', type=int, default=3)
    ap.add_argument('--noise-steps', type=int, default=15)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--cpu-dino', action='store_true')
    ap.add_argument('--start', type=int, default=0)
    args = ap.parse_args()
    repo = args.repo.resolve()
    out = args.out.resolve(); out.mkdir(parents=True,exist_ok=True)
    with args.manifest.open(encoding='utf-8', newline='') as fp:
        rows = list(csv.DictReader(fp))
    if args.limit: rows = rows[:args.limit]
    default = {'lcm':dict(lr=0.25,c_ratio=0.33,lora='pretrained_models/lcm-lora-sd1.5/lcm-lora-sdv1-5/pytorch_lora_weights.safetensors'),
               'hyper':dict(lr=0.15,c_ratio=0.38,lora='pretrained_models/Hyper-SD15-4steps-lora.safetensors')}
    cfg=default[args.mode]
    env=os.environ.copy()
    env['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    env['CUDA_VISIBLE_DEVICES']='0'
    env['STYLEGALLERY_DINO_CPU']='1' if args.cpu_dino else '0'
    logpath=out/'execution.jsonl'
    for i,row in enumerate(rows):
        if i < args.start: continue
        tid=row['task_id']; safe_id=f'{i:04d}_'+hashlib.sha256(tid.encode()).hexdigest()[:10]
        result=out/f'{safe_id}.png'
        if result.is_file():
            print(f'[SKIP] {i+1}/{len(rows)} {tid}')
            continue
        refs=row['style_paths'].split(';')
        if not refs or any(not Path(p).is_file() for p in refs):
            raise FileNotFoundError(f'bad reference paths for {tid}: {refs}')
        if not Path(row['source_path']).is_file():
            raise FileNotFoundError(row['source_path'])
        taskdir=out/'_work'/safe_id
        taskdir.mkdir(parents=True,exist_ok=True)
        cmd=[os.sys.executable,'demo_accelerate.py',
             '--mode',args.mode,
             '--model_name','pretrained_models/runwayml_stable-diffusion-v1-5',
             '--lora_path',cfg['lora'],
             '--content_image',row['source_path'],
             '--style_images',*refs,
             '--output_folder',str(taskdir),
             '--noise_steps',str(args.noise_steps),
             '--num_optimize_steps',str(args.steps),
             '--iters',str(args.iters),
             '--lr',str(cfg['lr']),
             '--c_ratio',str(cfg['c_ratio']),
             '--seed',str(args.seed)]
        stopped=threading.Event();mem=[]
        sampler=threading.Thread(target=mem_sampler,args=(stopped,mem),daemon=True)
        sampler.start()
        started=time.monotonic()
        lp=out/f'{safe_id}.log'
        with lp.open('w',encoding='utf8') as lfh:
            p=subprocess.run(cmd,cwd=repo,env=env,stdout=lfh,stderr=subprocess.STDOUT)
        elapsed=time.monotonic()-started
        stopped.set();sampler.join(timeout=1.0)
        tmp=taskdir/f'result_{args.mode}.png'
        success=p.returncode==0 and tmp.is_file()
        if success:shutil.copyfile(tmp,result)
        metadata={'index':i,'task_id':tid,'source_style':row['source_style'],
                  'target_style':row['target_style'], 'is_cross':int(row['is_cross']),
                  'source_path':row['source_path'],'ref_paths':refs,
                  'result':str(result) if success else None,'returncode':p.returncode,
                  'success':bool(success),'cold_wall_s':round(elapsed,3),
                  'peak_device_memory_MiB_nvidia_smi':max(mem) if mem else None,
                  'mode':args.mode,'steps':args.steps,'iters':args.iters,
                  'cpu_dino':args.cpu_dino,'seed':args.seed}
        with logpath.open('a',encoding='utf8') as fh:
            fh.write(json.dumps(metadata,ensure_ascii=False)+'\n')
        print(f"[{('OK' if success else 'FAIL')}] {i+1}/{len(rows)} {tid} {elapsed:.1f}s peak={metadata['peak_device_memory_MiB_nvidia_smi']}MiB")
        if not success:
            raise RuntimeError(f'Baseline failed on {tid}, read {lp}. STOP, do not silently skip.')


if __name__=='__main__':main()
