"""A0 channel pruning and teacher-coordinate distillation; E/P/S10 stay frozen."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from exp.deploy_v35.a0_prune import StudentEncoder, build_encoder, prune_encoder, taylor_encoder_importance, _haar_high
from exp.deploy_v35.bundle import file_sha256, load_a0_encoder_from_file, load_s10b_from_file
from exp.deploy_v35.distill.data import batches, capture_rng, configure, restore_rng, save_checkpoint, write_json
from src.utils.vae import decode_s10


WIDTHS = {'enc20': [96, 192, 384, 384], 'enc10': [64, 128, 256, 320]}


def load_student(path: Path, device='cuda') -> StudentEncoder:
    ck = torch.load(path, map_location='cpu', weights_only=False)
    model = StudentEncoder(build_encoder(ck['meta']['widths']), nn.Conv2d(8, 4, 1))
    model.load_state_dict(ck['model'], strict=True)
    return model.to(device).eval()


@torch.no_grad()
def validate(model, decoder, paths: list[Path], micro: int, limit: int = 128) -> dict:
    sums = dict(mae=0., mse=0., energy=0., cosine=0., decoded_mae255=0.)
    n = 0
    for path in paths:
        shard = torch.load(path, map_location='cpu', weights_only=False)
        for i in range(0, len(shard['z']), micro):
            x = shard['rgb'][i:i+micro].cuda().float() / 255.
            zt = shard['z'][i:i+micro].cuda().float()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                zs = model(x).float()
            b = len(zs)
            if not torch.isfinite(zs).all():
                raise FloatingPointError('Non-finite distilled encoder')
            sums['mae'] += (zs-zt).abs().flatten(1).mean(1).sum().item()
            sums['mse'] += (zs-zt).square().flatten(1).mean(1).sum().item()
            sums['energy'] += zt.square().flatten(1).mean(1).sum().item()
            sums['cosine'] += F.cosine_similarity(zs.flatten(1), zt.flatten(1)).sum().item()
            # Calibrate on actual S10 responses to teacher and student coordinates.
            ys = decode_s10(decoder, zs, output_cpu=False)
            yt = decode_s10(decoder, zt, output_cpu=False)
            sums['decoded_mae255'] += (ys-yt).abs().flatten(1).mean(1).sum().item()*255.
            n += b
            if n >= limit:
                break
        if n >= limit:
            break
    return dict(n=n, latent_mae=sums['mae']/n, latent_cosine=sums['cosine']/n,
                latent_relative_rmse=(sums['mse']/max(sums['energy'], 1e-12))**.5,
                decoded_mae255=sums['decoded_mae255']/n)


def train(config: dict, variant: str, out: Path) -> None:
    configure(config['seed'])
    out.mkdir(parents=True, exist_ok=True)
    bundle, cache = Path(config['bundle']), Path(config['cache'])
    decoder = load_s10b_from_file(bundle/'s10b_decoder.pt')
    cache_hash = file_sha256(cache/'complete.json')
    last = out/'last.pt'
    opt_state = None
    if last.exists():
        ck = torch.load(last, map_location='cpu', weights_only=False)
        if ck['cache_hash'] != cache_hash:
            raise ValueError('Encoder resume cache mismatch')
        model = load_student(last).train()
        meta, step, elapsed, best = ck['meta'], ck['step'], ck['elapsed'], ck['best']
        opt_state = ck['optimizer']
        restore_rng(ck['rng'])
    else:
        raw = load_a0_encoder_from_file(bundle/'a0_encoder.pt', device='cpu', dtype=torch.float32)
        score_path = out.parent/'taylor_scores.pt'
        if score_path.exists():
            imp = torch.load(score_path, map_location='cpu', weights_only=True)
        else:
            sample = torch.load(sorted((cache/'train').glob('*.pt'))[0], map_location='cpu', weights_only=False)
            imp = taylor_encoder_importance(raw, sample['rgb'][:8].float()/255., sample['z'][:8].float(), bs=1, decoder=decoder)
            torch.save(imp, score_path)
        raw = raw.cpu()
        model, meta = prune_encoder(raw, WIDTHS[variant], importance=imp)
        del raw
        model = model.cuda().train()
        step, elapsed, best = 0, 0., float('inf')
        torch.cuda.empty_cache()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get('encoder_lr', 1e-4), weight_decay=1e-4)
    if opt_state is not None:
        optimizer.load_state_dict(opt_state)
    stream = batches(sorted((cache/'train').glob('*.pt')), ('rgb','z'), config['micro_batch'], config['seed']+step)
    calib = sorted((cache/'calib').glob('*.pt'))
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()

    def checkpoint():
        save_checkpoint(last, dict(model=model.state_dict(), meta=meta, optimizer=optimizer.state_dict(),
                                  step=step, elapsed=elapsed+time.monotonic()-start, best=best,
                                  cache_hash=cache_hash, rng=capture_rng(), variant=variant))

    initial = validate(model, decoder, calib, config['micro_batch'], config.get('calib_limit',128))
    write_json(out/'initial.json', initial)
    running = []
    for step in range(step+1, config['encoder_steps']+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_step = 0.
        for _ in range(config['accum']):
            batch = next(stream)
            x, target = batch['rgb'].float()/255., batch['z'].float()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                pred = model(x).float()
                lat = F.smooth_l1_loss(pred, target, beta=.1)
                high = sum(F.l1_loss(a,b) for a,b in zip(_haar_high(pred),_haar_high(target)))/3.
                loss = lat + .5*high
            # Frozen S10 input gradients, exactly one PQC and /0.18215.
            if step % 4 == 0:
                ys = decode_s10(decoder, pred, enable_grad=True, output_cpu=False)
                yt = decode_s10(decoder, target, output_cpu=False).clone()
                loss = loss + .1*F.l1_loss(ys,yt)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'{variant} non-finite loss step {step}')
            (loss/config['accum']).backward()
            loss_step += loss.detach().item()/config['accum']
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        running.append(loss_step)
        stop = elapsed+time.monotonic()-start >= config['encoder_seconds'] or step == config['encoder_steps']
        if step % 50 == 0 or stop:
            print(f'[{variant}] step={step} loss={sum(running)/len(running):.5f} elapsed={elapsed+time.monotonic()-start:.0f}s',flush=True)
            running.clear()
        if step % 300 == 0 or stop:
            metrics = validate(model, decoder, calib, config['micro_batch'], config.get('calib_limit',128))
            score = metrics['latent_relative_rmse'] + metrics['decoded_mae255']/255.
            if score < best:
                best = score
                save_checkpoint(out/'best.pt', dict(model=model.state_dict(), meta=meta, step=step,
                                                    cache_hash=cache_hash, metrics=metrics, variant=variant))
            write_json(out/'progress.json',dict(step=step, elapsed=elapsed+time.monotonic()-start,
                                              calibration=metrics,best_score=best))
            checkpoint()
        elif step % 100 == 0:
            checkpoint()
        if stop:
            break
    final = load_student(out/'best.pt')
    metrics = validate(final, decoder, calib, config['micro_batch'], config.get('calib_limit',128))
    write_json(out/'complete.json',dict(variant=variant, step=step, widths=meta['widths'], params=meta['params'],
                                      calibration=metrics, best_score=best,
                                      peak_allocated_gb=torch.cuda.max_memory_allocated()/2**30,
                                      seconds=elapsed+time.monotonic()-start,
                                      checkpoint_sha256=file_sha256(out/'best.pt'),
                                      quality_status='P4 test750 still required'))


def install(pipe, path: Path) -> None:
    student = load_student(path).requires_grad_(False)
    pipe.fast_a0 = None
    pipe.a0 = None
    def encode(rgb):
        pipe.stats.a0_calls += 1
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return pipe._get_callable('distilled_a0', len(rgb), student)(rgb.float()).float()
    pipe._encode_a0 = encode
