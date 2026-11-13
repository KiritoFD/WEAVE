"""WEAVE-v4 训练：只学真实作品的条件 latent 分布 p(z|s)。

- 单图真实 latent，不做跨风格配对；
- 唯一损失 = eps MSE；
- EMA、里程碑保存、按 t 分桶的 val 误差、早停；
- 训练侧不加载 VAE / DINO / CLIP。
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from paths import resolve_config_path
from .config import V4Config
from .data import (STYLE_NAMES, build_split, content_clusters, cross_style_topk,
                   duplicate_groups, latent_descriptor, load_bank, save_json)
from .diffusion import Schedule, pred_to_x0_eps, training_loss, v_target
from .model import V4UNet, count_parameters


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="WEAVE-v4 条件扩散先验训练")
    ap.add_argument("--run-name", default="")
    ap.add_argument("--config", default="", help="可选 JSON，字段同 V4Config")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--batch-size", type=int, default=-1)
    ap.add_argument("--grad-accum", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=-1.0)
    ap.add_argument("--bench", action="store_true", help="只测 microbatch 显存/速度后退出")
    ap.add_argument("--bench-batches", type=int, default=8)
    ap.add_argument("--bench-mbs", default="16,32,64,96,128,192,256", help="逗号分隔的 microbatch 列表")
    ap.add_argument("--time-budget-sec", type=float, default=-1.0)
    ap.add_argument("--seed", type=int, default=-1)
    return ap.parse_args()


def build_config(args: argparse.Namespace) -> V4Config:
    payload = {}
    if args.config:
        payload = json.loads(Path(args.config).read_text(encoding="utf-8"))
    cfg = V4Config.from_mapping(payload)
    if args.run_name:
        cfg.run_name = args.run_name
    if args.max_steps > 0:
        cfg.max_steps = args.max_steps
    if args.batch_size > 0:
        cfg.batch_size = args.batch_size
    if args.grad_accum > 0:
        cfg.grad_accum = args.grad_accum
    if args.lr > 0:
        cfg.lr = args.lr
    if args.seed >= 0:
        cfg.seed = args.seed
    cfg.validate()
    return cfg


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module, step: int) -> None:
        d = min(self.decay, (1.0 + step) / (10.0 + step))
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1.0 - d)
            else:
                self.shadow[k].copy_(v)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}


def lr_at(step: int, cfg: V4Config) -> float:
    total = cfg.max_steps
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, total - cfg.warmup_steps)
    cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return cfg.lr * (cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cos)


@torch.no_grad()
def validate(model, val: dict, schedule: Schedule, cfg: V4Config) -> dict:
    model.eval()
    gen = torch.Generator(device=val["latents"].device).manual_seed(cfg.seed + 7777)
    bins = np.zeros(6, dtype=np.float64)
    counts = np.zeros(6, dtype=np.float64)
    total = 0.0
    x0_total = 0.0
    n = val["latents"].shape[0]
    for b in range(cfg.val_batches):
        idx = torch.randint(0, n, (cfg.batch_size,), device=val["latents"].device, generator=gen)
        z0 = val["latents"].index_select(0, idx)
        sid = val["style_ids"].index_select(0, idx)
        t = torch.randint(0, schedule.timesteps, (z0.shape[0],), device=z0.device, generator=gen)
        noise = torch.randn(z0.shape, device=z0.device, dtype=z0.dtype, generator=gen)
        zt = schedule.q_sample(z0, t, noise)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.use_bf16):
            pred = model(zt, t.float(), sid)
        pred = pred.float()
        target = v_target(z0, noise, t, schedule) if cfg.prediction_type == "v" else noise
        err = (pred - target) ** 2
        x0_pred, _ = pred_to_x0_eps(pred, zt, t, schedule, cfg.prediction_type)
        per = err.flatten(1).mean(1)
        x0_total += float(((x0_pred - z0) ** 2).flatten(1).mean(1).sum())
        b_idx = (t * 6 // schedule.timesteps).cpu().numpy()
        for k in range(6):
            sel = b_idx == k
            if sel.any():
                bins[k] += float(per[torch.from_numpy(sel)].sum())
                counts[k] += float(sel.sum())
        total += float(per.sum())
    model.train()
    bin_means = (bins / np.maximum(counts, 1)).tolist()
    denom = max(1, cfg.val_batches * cfg.batch_size)
    return {"val_loss": total / denom, "val_x0_mse": x0_total / denom,
            "val_bins": bin_means, "prediction_type": cfg.prediction_type}


def style_response_probe(model, schedule: Schedule, cfg: V4Config, *, step: int) -> dict:
    """固定噪声 & 固定 t，比较 style 0 与 style 1 的 eps 输出差异（初始应≈0，训练后应>0）。"""
    model.eval()
    gen = torch.Generator(device="cuda").manual_seed(cfg.seed + 999)
    z = torch.randn(4, cfg.latent_channels, 64, 64, device="cuda", generator=gen)
    t = torch.full((4,), 500.0, device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.use_bf16):
        a = model(z, t, torch.zeros(4, dtype=torch.long, device="cuda")).float()
        b = model(z, t, torch.ones(4, dtype=torch.long, device="cuda")).float()
        nul = model(z, t, torch.full((4,), -1, dtype=torch.long, device="cuda")).float()
    model.train()
    return {
        "style01_rel_diff": float((a - b).abs().mean() / (a.abs().mean() + 1e-8)),
        "style0_vs_null_rel_diff": float((a - nul).abs().mean() / (a.abs().mean() + 1e-8)),
        "eps_abs_mean": float(a.abs().mean()),
    }


def bench(cfg: V4Config, mbs: tuple[int, ...] = (16, 32, 64, 96, 128, 192, 256)) -> None:
    bank = load_bank(resolve_config_path(cfg.packed_dir))
    schedule = Schedule.linear(cfg.timesteps, cfg.beta_start, cfg.beta_end, device="cuda")
    model = V4UNet(cfg).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    n = len(bank)
    latents = bank.latents.cuda()
    styles = bank.style_ids.cuda()
    print(f"[bench] params={count_parameters(model)/1e6:.2f}M", flush=True)
    results = []
    for mb in mbs:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        rounds = 3
        for _ in range(rounds):
            idx = torch.randint(0, n, (mb,), device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.use_bf16):
                loss, _ = training_loss(model, latents.index_select(0, idx),
                                        styles.index_select(0, idx), schedule,
                                        cfg_dropout=cfg.cfg_dropout,
                                        prediction_type=cfg.prediction_type)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / rounds
        peak = torch.cuda.max_memory_allocated() / 2**30
        resv = torch.cuda.max_memory_reserved() / 2**30
        results.append({"microbatch": mb, "step_ms": round(dt * 1000, 1),
                        "img_per_s": round(mb / dt, 2),
                        "peak_gib": round(peak, 2), "reserved_gib": round(resv, 2)})
        print(f"[bench] mb={mb:3d}  step={dt*1000:7.1f} ms  {mb/dt:6.2f} img/s  "
              f"peak={peak:.2f} GiB  reserved={resv:.2f} GiB", flush=True)
        del loss
    print(json.dumps(results, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if args.bench:
        mbs = tuple(int(x) for x in args.bench_mbs.split(",") if x.strip())
        bench(cfg, mbs)
        return

    out = resolve_config_path(cfg.out_dir) / cfg.run_name
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train.jsonl"
    log = open(log_path, "a", encoding="utf-8")

    def record(payload: dict) -> None:
        payload["wall"] = time.time()
        log.write(json.dumps(payload, ensure_ascii=False) + "\n")
        log.flush()

    bank = load_bank(resolve_config_path(cfg.packed_dir))
    desc = latent_descriptor(bank.latents)
    dup_group, dup_pairs = duplicate_groups(desc)
    clusters = content_clusters(desc, k=20, seed=cfg.seed)
    train_idx, val_idx, split_report = build_split(
        bank, val_per_style=cfg.val_per_style, seed=cfg.seed, dup_group=dup_group
    )
    neighbors = cross_style_topk(desc, bank.style_ids, k=5)
    save_json(out / "split.json", {
        **split_report,
        "dup_pairs": int(dup_pairs),
        "dup_groups": int(dup_group.max()) + 1,
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
    })
    torch.save({"train_idx": torch.from_numpy(train_idx), "val_idx": torch.from_numpy(val_idx),
                "dup_group": torch.from_numpy(dup_group), "cluster": torch.from_numpy(clusters),
                "neighbors": neighbors, "style_ids": bank.style_ids},
               out / "indices.pt")
    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")

    latents = bank.latents.cuda()
    style_all = bank.style_ids.cuda()
    train_lat = latents.index_select(0, torch.from_numpy(train_idx).cuda())
    train_sid = style_all.index_select(0, torch.from_numpy(train_idx).cuda())
    val = {"latents": latents.index_select(0, torch.from_numpy(val_idx).cuda()),
           "style_ids": style_all.index_select(0, torch.from_numpy(val_idx).cuda())}

    schedule = Schedule.linear(cfg.timesteps, cfg.beta_start, cfg.beta_end, device="cuda")
    model = V4UNet(cfg).cuda()
    ema = EMA(model, cfg.ema_decay)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    params = count_parameters(model)
    print(f"[v4] params={params/1e6:.2f}M train={len(train_idx)} val={len(val_idx)} "
          f"styles={STYLE_NAMES[:bank.num_styles]}", flush=True)
    record({"event": "start", "params": params, "train": int(len(train_idx)),
            "val": int(len(val_idx)), "config": cfg.to_dict(), "split": split_report})

    gen = torch.Generator(device="cuda").manual_seed(cfg.seed)
    best_val = float("inf")
    best_step = 0
    bad_evals = 0
    t0 = time.perf_counter()
    step = 0
    while step < cfg.max_steps:
        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        stat_acc = {}
        for _ in range(cfg.grad_accum):
            idx = torch.randint(0, train_lat.shape[0], (cfg.batch_size,), device="cuda", generator=gen)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.use_bf16):
                loss, stats = training_loss(
                    model, train_lat.index_select(0, idx), train_sid.index_select(0, idx),
                    schedule, cfg_dropout=cfg.cfg_dropout,
                    prediction_type=cfg.prediction_type,
                )
            (loss / cfg.grad_accum).backward()
            loss_acc += float(loss.detach()) / cfg.grad_accum
            for k, v in stats.items():
                stat_acc[k] = stat_acc.get(k, 0.0) + v / cfg.grad_accum
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg)
        opt.step()
        ema.update(model, step)
        step += 1

        if step % 50 == 0:
            record({"event": "train", "step": step, "loss": loss_acc, **stat_acc,
                    "lr": lr_at(step, cfg),
                    "vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                    "elapsed_s": round(time.perf_counter() - t0, 1)})

        if step % cfg.val_every == 0:
            ema_model = _ema_model(cfg, ema)
            v = validate(ema_model, val, schedule, cfg)
            probe = style_response_probe(model, schedule, cfg, step=step)
            record({"event": "val", "step": step, **v, **probe})
            print(f"[v4] step={step} train={loss_acc:.4f} val={v['val_loss']:.4f} "
                  f"bins={[round(x,3) for x in v['val_bins']]} "
                  f"style_diff={probe['style01_rel_diff']:.4f} "
                  f"elapsed={time.perf_counter()-t0:.0f}s", flush=True)
            improved = v["val_loss"] < best_val * (1.0 - cfg.early_stop_min_delta)
            if improved:
                best_val = v["val_loss"]
                best_step = step
                bad_evals = 0
                if cfg.save_best:
                    _save(out / "best.pt", ema, model, cfg, schedule, step, v)
            else:
                bad_evals += 1
                if step >= cfg.early_stop_after and bad_evals >= cfg.early_stop_patience:
                    record({"event": "early_stop", "step": step, "best_step": best_step,
                            "best_val": best_val})
                    print(f"[v4] early stop at {step} (best {best_step} val={best_val:.4f})", flush=True)
                    break

        if step in cfg.milestones:
            _save(out / f"step_{step:05d}.pt", ema, model, cfg, schedule, step,
                  validate(_ema_model(cfg, ema), val, schedule, cfg))
            print(f"[v4] milestone ckpt saved: step_{step:05d}.pt", flush=True)

        if args.time_budget_sec > 0 and time.perf_counter() - t0 > args.time_budget_sec \
                and step % cfg.val_every == 0:
            record({"event": "time_budget_stop", "step": step})
            print(f"[v4] stop on time budget at step {step}", flush=True)
            break

    _save(out / "last.pt", ema, model, cfg, schedule, step, {"val_loss": best_val})
    record({"event": "done", "step": step, "best_step": best_step, "best_val": best_val,
            "elapsed_s": round(time.perf_counter() - t0, 1)})
    print(f"[v4] done step={step} best={best_step}:{best_val:.4f} out={out}", flush=True)


def _ema_model(cfg: V4Config, ema: EMA) -> V4UNet:
    model = V4UNet(cfg).cuda()
    model.load_state_dict({k: v.float() for k, v in ema.state_dict().items()})
    model.eval()
    return model


def _save(path: Path, ema: EMA, model, cfg: V4Config, schedule: Schedule, step: int, val: dict) -> None:
    payload = {
        "variant": "V4DIFF",
        "v4_config": cfg.to_dict(),
        "model": {k: v.half() for k, v in ema.state_dict().items()},
        "raw_model": {k: v.half() for k, v in model.state_dict().items()},
        "step": step,
        "val": val,
        "schedule": {"timesteps": cfg.timesteps, "beta_start": cfg.beta_start,
                     "beta_end": cfg.beta_end, "beta_schedule": cfg.beta_schedule},
        "prediction_type": cfg.prediction_type,
        "cond_mode": cfg.cond_mode,
        "latent_scale_factor": cfg.latent_scale_factor,
    }
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


if __name__ == "__main__":
    main()

