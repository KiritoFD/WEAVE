"""Train-only image audit, bounded shards, and resumable experiment IO."""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch

from exp.deploy_v35.bundle import file_sha256, load_a0_encoder_from_file
from exp.deploy_v35.torch_opt import FastA0Encoder
from src.utils.runtime import REPO


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def save_checkpoint(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.pt')
    torch.save(value, tmp)
    tmp.replace(path)


def load_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as im:
        arr = np.array(ImageOps.exif_transpose(im).convert('RGB').resize((512, 512), Image.Resampling.BICUBIC))
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def fingerprint(rgb: torch.Tensor) -> tuple[str, int, np.ndarray]:
    arr = rgb.permute(1, 2, 0).numpy()
    im = Image.fromarray(arr)
    small = np.asarray(im.convert('L').resize((9, 8), Image.Resampling.BILINEAR))
    dh = int.from_bytes(np.packbits(small[:, 1:] > small[:, :-1]).tobytes(), 'big')
    thumb = np.asarray(im.resize((32, 32), Image.Resampling.BILINEAR), dtype=np.int16)
    return hashlib.sha256(arr.tobytes()).hexdigest(), dh, thumb


def build_manifest(train_root: Path, out: Path, n_train: int, n_calib: int, seed: int) -> dict:
    path = out / 'manifest.json'
    requested = dict(train_root=str(train_root.resolve()), n_train=n_train, n_calib=n_calib, seed=seed)
    if path.exists():
        manifest = json.loads(path.read_text(encoding='utf-8'))
        if manifest['requested'] != requested:
            raise ValueError('Cache configuration changed; use a new output directory')
        return manifest
    task_path = REPO / 'eval_protocol/tasks_750.json'
    tasks = json.loads(task_path.read_text(encoding='utf-8'))['tasks']
    rels = {t['source_rel_path'] for t in tasks}
    rels.update(t['ref_rel_path'] for t in tasks if 'ref_rel_path' in t)
    # Include every reference on disk, regardless of manifest field spelling.
    eval_files = set((REPO / 'data/test').rglob('*.jpg')) | set((REPO / 'data/test').rglob('*.png'))
    eval_files.update(REPO / p for p in rels)
    blocked = [fingerprint(load_rgb(p)) for p in sorted(eval_files) if p.is_file()]
    blocked_sha = {x[0] for x in blocked}
    files = sorted(p for p in train_root.rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'})
    random.Random(seed).shuffle(files)
    selected = {'train': [], 'calib': []}
    seen_sha, seen_groups = set(), set()
    excluded, corrupt = 0, 0
    oversized = []
    audit_start = time.monotonic()
    for scanned, p in enumerate(files, 1):
        if scanned == 1 or scanned % 256 == 0:
            print(f'[audit] scanned={scanned}/{len(files)} train={len(selected["train"])}/{n_train} '
                  f'calib={len(selected["calib"])}/{n_calib} elapsed={time.monotonic()-audit_start:.0f}s', flush=True)
        try:
            sha, dh, thumb = fingerprint(load_rgb(p))
        except Image.DecompressionBombError as exc:
            oversized.append(dict(rel=p.relative_to(train_root).as_posix(), reason=str(exc)))
            write_json(out / 'oversized.json', dict(excluded=oversized))
            print(f'[audit excluded oversized] {p}: {exc}', flush=True)
            continue
        except (OSError, ValueError):
            corrupt += 1
            continue
        near_test = sha in blocked_sha or any(
            (dh ^ h).bit_count() <= 4 and np.abs(thumb - tiny).mean() <= 3.0
            for _, h, tiny in blocked
        )
        if near_test:
            excluded += 1
            continue
        # Keep a content hash group in one split; discard exact/near dHash duplicates.
        if sha in seen_sha or dh in seen_groups:
            continue
        seen_sha.add(sha)
        seen_groups.add(dh)
        split = 'calib' if int(sha[:8], 16) % 32 == 0 else 'train'
        limit = n_calib if split == 'calib' else n_train
        if len(selected[split]) < limit:
            selected[split].append(dict(rel=p.relative_to(train_root).as_posix(), sha256_rgb512=sha, dhash=str(dh)))
        if len(selected['train']) == n_train and len(selected['calib']) == n_calib:
            break
    if len(selected['train']) != n_train or len(selected['calib']) != n_calib:
        raise ValueError(f'Insufficient distinct train-side images: { {k: len(v) for k,v in selected.items()} }')
    manifest = dict(requested=requested, splits=selected, excluded_test=excluded, corrupt=corrupt, oversized=oversized,
                    eval_protocol_sha256=file_sha256(task_path), eval_images=len(blocked),
                    audit='512px exact RGB and dHash+32px near-duplicate exclusion; not an artist-disjoint claim')
    write_json(path, manifest)
    return manifest


@torch.inference_mode()
def cache_encoder(config: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(Path(config['train_root']), out, config['n_train'], config['n_calib'], config['seed'])
    bundle = Path(config['bundle'])
    expected = dict(a0_sha256=file_sha256(bundle / 'a0_encoder.pt'), manifest_sha256=file_sha256(out / 'manifest.json'),
                    preprocess='exif_rgb_bicubic_512', teacher='A0_BF16_mean_scaled_0.18215')
    done = out / 'complete.json'
    if done.exists():
        if json.loads(done.read_text())['provenance'] != expected:
            raise ValueError('Teacher cache provenance mismatch')
        return
    torch.set_num_threads(4)
    a0 = load_a0_encoder_from_file(bundle / 'a0_encoder.pt', dtype=torch.bfloat16)
    teacher = FastA0Encoder(a0).cuda().to(dtype=torch.bfloat16).eval()
    counts = {}
    t0 = time.monotonic()
    for split, records in manifest['splits'].items():
        (out / split).mkdir(exist_ok=True)
        for start in range(0, len(records), 32):
            dest = out / split / f'{start:06d}.pt'
            if dest.exists():
                continue
            rec = records[start:start+32]
            rgb = torch.stack([load_rgb(Path(config['train_root']) / r['rel']) for r in rec])
            zs = []
            for i in range(0, len(rec), config['micro_batch']):
                x = rgb[i:i+config['micro_batch']].cuda().to(torch.bfloat16) / 255.0
                zs.append(teacher(x).cpu().half())
            save_checkpoint(dest, dict(rgb=rgb, z=torch.cat(zs), ids=[r['rel'] for r in rec]))
            if start % 512 == 0:
                print(f'[cache {split}] {start+len(rec)}/{len(records)} elapsed={time.monotonic()-t0:.0f}s', flush=True)
        counts[split] = len(records)
    write_json(done, dict(provenance=expected, counts=counts, seconds=time.monotonic()-t0))


def batches(paths: list[Path], keys: tuple[str, ...], batch: int, seed: int):
    """One CPU shard resident; shuffle shards and rows without random disk seeks."""
    if not paths:
        raise ValueError('Empty training cache')
    rng = random.Random(seed)
    while True:
        order = paths.copy()
        rng.shuffle(order)
        for p in order:
            shard = torch.load(p, map_location='cpu', weights_only=False)
            ids = list(range(len(shard[keys[0]])))
            rng.shuffle(ids)
            for i in range(0, len(ids), batch):
                ix = ids[i:i+batch]
                yield {k: shard[k][ix].cuda(non_blocking=True) for k in keys}


def capture_rng() -> dict:
    return dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(), python=random.getstate())


def restore_rng(state: dict) -> None:
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state(state['cuda'])
    random.setstate(state['python'])


def configure(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    os.environ.setdefault('PYTHONUNBUFFERED', '1')
