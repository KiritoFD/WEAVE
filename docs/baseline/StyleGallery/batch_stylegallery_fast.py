#!/usr/bin/env python3
"""Fast persistent-process StyleGallery batch runner (WEAVE 12GB reproduction).

Equivalent protocol to batch_stylegallery_12g.py (official demo_accelerate.py
lcm preset: 28 optimize steps x 3 iters, lr=0.25, c_ratio=0.33, noise_steps=15,
single style reference, no depth/SAM, fp16, gradient checkpointing) but keeps
SD1.5/DINOv2/Transformer resident and caches PER-IMAGE cluster features on disk.

Documented deviations from the official per-subprocess script (implementation-
efficiency fixes, not algorithm changes):
  * pipeline.py re-creates a randomly-initialised Transformer inside
    match_content_style_clusters() on EVERY call, so official matching is
    RNG-state dependent (differs between cache-hit and cache-miss runs).
    Here ONE Transformer is created under torch.manual_seed(seed) and reused
    for all images -> deterministic, internally consistent matching.
  * Per-image cluster_data (combined_feature / circle / dinov2_feature) depends
    only on that image, not on the pair -> cached to disk; computed once per
    unique image (<=155) instead of once per task (1500).
  * Model weights stay resident; scheduler swapped DDIM<->LCM per phase to
    preserve official order (inversion/matching on DDIM, optimization on LCM);
    LoRA fuse/unfuse per task as in demo_accelerate.py.
  * torch.manual_seed(seed) re-armed before each process_image_data call that
    would miss the processed_data cache, reproducing fresh-subprocess kmeans RNG.

Output naming / execution.jsonl schema identical to batch_stylegallery_12g.py
so results are interchangeable and resumable. Per-task failures are recorded
and the run continues (a single bad image must not kill an overnight batch);
DONE marker is only written when the manifest is fully covered.
"""
import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import threading
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image


def mem_sampler(stop, samples):
    import subprocess
    while not stop.is_set():
        try:
            p = subprocess.run(['nvidia-smi', '--query-gpu=memory.used',
                                '--format=csv,noheader,nounits'],
                               capture_output=True, text=True, timeout=5)
            if p.returncode == 0:
                m = re.search(r'\d+', p.stdout)
                if m:
                    samples.append((time.monotonic(), int(m.group())))
        except (OSError, subprocess.TimeoutExpired):
            return
        stop.wait(0.5)


def cluster_cache_path(cache_dir, image_path):
    key = hashlib.sha256(os.path.abspath(image_path).encode()).hexdigest()[:24]
    return cache_dir / f'{key}.pt'


def compute_image_cluster_data(pipe, transformer, dinov2_model, dinov2_processor,
                               utils, unet_feature, mask, image_path):
    """Replica of the per-cluster feature extraction inside
    StyleGallery.match_content_style_clusters (single-image part only)."""
    mask = mask.to(pipe.device)
    cluster_data = {}
    for cluster in torch.unique(mask):
        c_mask = (mask == cluster)
        valid_indices = torch.nonzero(c_mask.squeeze(), as_tuple=False)
        if valid_indices.size(0) == 0:
            continue
        masked_features = unet_feature * c_mask.unsqueeze(0).float()
        valid_features = masked_features[0, :, valid_indices[:, 0], valid_indices[:, 1]]
        attended_features = transformer(valid_features.t().float()).t()
        reconstructed_features = torch.zeros_like(masked_features)
        reconstructed_features[0, :, valid_indices[:, 0], valid_indices[:, 1]] = attended_features
        channel_sums = reconstructed_features.sum(dim=(2, 3))
        valid_points = c_mask.sum()
        avg_feature = channel_sums / (valid_points + 1e-8)
        variance_feature = reconstructed_features.var(dim=(2, 3)) / (valid_points + 1e-8)
        combined_feature = torch.cat((avg_feature, variance_feature), dim=1).squeeze()
        points = valid_indices.cpu().numpy()
        center, radius = utils.minimum_enclosing_circle(points)
        dinov2_semantic_feature = pipe.extract_semantic_features(
            image_path, c_mask, dinov2_model, dinov2_processor).squeeze()
        cluster_data[int(cluster.item())] = {
            'combined_feature': combined_feature.detach().cpu(),
            'circle_center': (float(center[0]), float(center[1])),
            'circle_radius': float(radius),
            'dinov2_feature': dinov2_semantic_feature.detach().cpu(),
        }
    return cluster_data


def get_image_cluster_data(pipe, transformer, dinov2, cache_dir, utils,
                           unet_feature, mask, image_path):
    cp = cluster_cache_path(cache_dir, image_path)
    if cp.is_file():
        blob = torch.load(cp, map_location='cpu', weights_only=False)
        return {int(k): v for k, v in blob.items()}
    data = compute_image_cluster_data(
        pipe, transformer, dinov2[0], dinov2[1], utils,
        unet_feature, mask, image_path)
    torch.save(data, cp)
    return data


def pairwise_match(content_data, style_data_list, device, utils):
    """Replica of the pairwise similarity stage of match_content_style_clusters."""
    cluster_matches = {}
    for content_cluster, cdat in content_data.items():
        best_match = {'style_dict_index': 0, 'style_cluster': 0,
                      'similarity': float('-inf')}
        c_combined = cdat['combined_feature'].flatten().to(device)
        c_dinov2 = cdat['dinov2_feature'].mean(dim=0).to(device)
        if c_dinov2.dim() == 0:
            c_dinov2 = torch.zeros(1, device=device)
        for s_idx, s_clusters in enumerate(style_data_list):
            for style_cluster, sdat in s_clusters.items():
                s_combined = sdat['combined_feature'].flatten().to(device)
                s_dinov2 = sdat['dinov2_feature'].mean(dim=0).to(device)
                if s_dinov2.dim() == 0:
                    s_dinov2 = torch.zeros(1, device=device)
                combined_sim = F.cosine_similarity(
                    c_combined.unsqueeze(0), s_combined.unsqueeze(0)).item()
                dinov2_sim = F.cosine_similarity(
                    c_dinov2.unsqueeze(0), s_dinov2.unsqueeze(0)).item()
                overlap_ratio = utils.calculate_overlap_ratio(
                    cdat['circle_center'], cdat['circle_radius'],
                    sdat['circle_center'], sdat['circle_radius'])
                similarity = (combined_sim * 0.25) + dinov2_sim + (overlap_ratio * 0.125)
                if similarity > best_match['similarity']:
                    best_match = {'style_dict_index': s_idx,
                                  'style_cluster': style_cluster,
                                  'similarity': similarity}
        if best_match['similarity'] == float('-inf'):
            best_match['similarity'] = 0.0
        cluster_matches[content_cluster] = best_match
    return cluster_matches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True, type=Path)
    ap.add_argument('--manifest', required=True, type=Path)
    ap.add_argument('--out', default='sg750_fast', type=Path)
    ap.add_argument('--mode', default='lcm', choices=['lcm', 'hyper'])
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--steps', type=int, default=28)
    ap.add_argument('--iters', type=int, default=3)
    ap.add_argument('--noise-steps', type=int, default=15)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--cpu-dino', action='store_true')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--max-tasks', type=int, default=0,
                    help='exit cleanly after N processed tasks (periodic restart)')
    args = ap.parse_args()

    repo = args.repo.resolve()
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    import utils
    from pipeline import StyleGallery
    from diffusers import DDIMScheduler, LCMScheduler
    from basic_module import Transformer
    from transformers import AutoModel, AutoImageProcessor

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = out / '_cluster_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)

    with args.manifest.open(encoding='utf-8', newline='') as fp:
        rows = list(csv.DictReader(fp))
    if args.limit:
        rows = rows[:args.limit]

    cfg = {'lcm': dict(lr=0.25, c_ratio=0.33,
                       lora='pretrained_models/lcm-lora-sd1.5/lcm-lora-sdv1-5/pytorch_lora_weights.safetensors'),
           'hyper': dict(lr=0.15, c_ratio=0.38,
                         lora='pretrained_models/Hyper-SD15-4steps-lora.safetensors')}[args.mode]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)

    print('[fast] loading SD1.5 ...', flush=True)
    ddim_scheduler = DDIMScheduler.from_pretrained(
        'pretrained_models/runwayml_stable-diffusion-v1-5', subfolder='scheduler')
    pipe = StyleGallery.from_pretrained(
        'pretrained_models/runwayml_stable-diffusion-v1-5',
        scheduler=ddim_scheduler, safety_checker=None,
        torch_dtype=torch.float16).to(device)
    pipe.extractor = pipe.unet
    lcm_scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    transformer_box = [None]
    dinov2_box = [None, None]

    def ensure_transformer(width):
        if transformer_box[0] is None:
            torch.manual_seed(args.seed)
            transformer_box[0] = Transformer(
                width=width, layers=8, heads=4).to(
                device=pipe.device, dtype=torch.float32)
        return transformer_box[0]

    def ensure_dinov2():
        if dinov2_box[0] is None:
            ddev = 'cpu' if (args.cpu_dino or
                             os.environ.get('STYLEGALLERY_DINO_CPU', '0') == '1') \
                else str(pipe.device)
            dinov2_box[0] = AutoModel.from_pretrained(
                'pretrained_models/facebook/dinov2-base').eval().to(ddev)
            dinov2_box[1] = AutoImageProcessor.from_pretrained(
                'pretrained_models/facebook/dinov2-base')
        return dinov2_box[0], dinov2_box[1]

    stopped = threading.Event()
    mem = []
    sampler = threading.Thread(target=mem_sampler, args=(stopped, mem), daemon=True)
    sampler.start()

    logpath = out / 'execution.jsonl'
    lora_registered = False
    lora_fused = False
    processed = 0
    failures = 0
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
        content_path = row['source_path']
        taskdir = out / '_work' / safe_id
        taskdir.mkdir(parents=True, exist_ok=True)

        started = time.monotonic()
        success = False
        try:
            # ---- phase 1: matching on BASE (unfused) UNet + DDIM scheduler ----
            pipe.scheduler = ddim_scheduler
            if lora_fused:
                pipe.unfuse_lora()
                lora_fused = False
            with torch.inference_mode():
                null_embeds = pipe.encode_prompt('', device, 1, False)[0]

                def build_dict(image_path, image_name):
                    data_path = utils.get_route(image_path)
                    if not os.path.isfile(data_path):
                        torch.manual_seed(args.seed)  # fresh-subprocess kmeans RNG
                    latent = pipe.image2latent(utils.load_image(image_path, (512, 512)))
                    text_embeds = null_embeds.repeat(latent.shape[0], 1, 1)
                    original_mask = None
                    if image_name.startswith('style'):
                        sm = os.path.join(
                            os.path.dirname(image_path),
                            os.path.splitext(os.path.basename(image_path))[0] + '_json/label.png')
                        if os.path.exists(sm):
                            original_mask = utils.convert_mask_to_array(sm)
                    return pipe.process_image_data(
                        image_path=image_path, latent=latent, text_embeds=text_embeds,
                        steps=args.noise_steps, image_name=image_name,
                        original_mask=original_mask, use_depth=False, use_sam=False)

                content_dict = build_dict(content_path, 'content')
                style_dict_list = [build_dict(r, f'style{j}') for j, r in enumerate(refs)]

                width = content_dict['unet_feature'].shape[1]
                transformer = ensure_transformer(width)
                dinov2_model, dinov2_processor = ensure_dinov2()
                dinov2 = (dinov2_model, dinov2_processor)
                cdata = get_image_cluster_data(
                    pipe, transformer, dinov2, cache_dir, utils,
                    content_dict['unet_feature'], content_dict['mask'], content_path)
                sdata_list = [get_image_cluster_data(
                    pipe, transformer, dinov2, cache_dir, utils,
                    sd['unet_feature'], sd['mask'], sd['image']) for sd in style_dict_list]
                cluster_matches = pairwise_match(cdata, sdata_list, pipe.device, utils)

            # ---- phase 2: LCM LoRA + optimize (official order) ----
            if not lora_registered:
                pipe.load_lora_weights(cfg['lora'])
                lora_registered = True
            if not lora_fused:
                pipe.fuse_lora()
                lora_fused = True
            pipe.scheduler = lcm_scheduler
            controller = utils.Controller(self_layers=(10, 16))
            image = pipe.style_transfer(
                content_dict=content_dict, style_dict=style_dict_list,
                controller=controller, cluster_matches=cluster_matches,
                num_optimize_steps=args.steps, iters=args.iters,
                lr=cfg['lr'], c_ratio=cfg['c_ratio'],
                enable_gradient_checkpoint=True)
            tmp = taskdir / f'result_{args.mode}.png'
            save_image(image, str(tmp))
            success = tmp.is_file()
            if success:
                shutil.copyfile(tmp, result)
        except Exception:
            (out / f'{safe_id}.error.log').write_text(traceback.format_exc(),
                                                   encoding='utf8')
            print(f'[ERR] {i+1}/{len(rows)} {tid}', flush=True)

        elapsed = time.monotonic() - started
        stopped_now = time.monotonic()
        win = [mb for ts, mb in mem if started - 1 <= ts <= stopped_now + 1]
        metadata = {'index': i, 'task_id': tid,
                    'source_style': row['source_style'],
                    'target_style': row['target_style'],
                    'is_cross': int(row['is_cross']),
                    'source_path': content_path, 'ref_paths': refs,
                    'result': str(result) if success else None,
                    'returncode': 0 if success else 1, 'success': bool(success),
                    'cold_wall_s': round(elapsed, 3), 'runner': 'fast_persistent',
                    'peak_device_memory_MiB_nvidia_smi': max(win) if win else None,
                    'mode': args.mode, 'steps': args.steps, 'iters': args.iters,
                    'seed': args.seed}
        with logpath.open('a', encoding='utf8') as fh:
            fh.write(json.dumps(metadata, ensure_ascii=False) + '\n')
        failures += (not success)
        processed += 1
        print(f"[{'OK' if success else 'FAIL'}] {i+1}/{len(rows)} {tid} "
              f"{elapsed:.1f}s", flush=True)

        if args.max_tasks and processed >= args.max_tasks:
            hit_restart = True
            break

    stopped.set()
    sampler.join(timeout=1.0)
    if not hit_restart:
        (out / 'DONE').write_text(
            f'processed={processed} failures={failures}\n')
    print(f'[fast] exiting processed={processed} failures={failures} '
          f'restart={hit_restart}', flush=True)


if __name__ == '__main__':
    main()
