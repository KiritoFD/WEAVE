"""WEAVE-v4 checkpoint 诊断：去噪误差分桶、风格判别探针、记忆检查、纯生成与 swap。

不参与训练，只用真实 latent + 一个独立的线性风格探针；VAE 只在需要看图时加载。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import V4Config
from .data import STYLE_NAMES, load_bank
from .diffusion import Schedule, ddim_sample, sdedit
from .model import V4UNet


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--run-dir", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gen-per-style", type=int, default=16)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--decode", action="store_true", help="用 VAE 解码少量图并保存 PNG")
    ap.add_argument("--val-batches", type=int, default=16)
    return ap.parse_args()


def latent_features(z: torch.Tensor) -> torch.Tensor:
    mean = z.mean(dim=(2, 3))
    std = z.std(dim=(2, 3))
    pooled = F.adaptive_avg_pool2d(z, 4).flatten(1)
    return torch.cat([mean, std, pooled], dim=1)


class StyleProbe(nn.Module):
    """线性风格判别器（独立于扩散主干），只用来检验生成结果是否带目标风格。"""

    def __init__(self, in_dim: int, num_styles: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, num_styles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def fit_probe(feats: torch.Tensor, labels: torch.Tensor, num_styles: int, *, steps: int = 600) -> StyleProbe:
    mu = feats.mean(0, keepdim=True)
    sd = feats.std(0, keepdim=True).clamp_min(1e-6)
    x = (feats - mu) / sd
    probe = StyleProbe(feats.shape[1], num_styles).to(feats.device)
    labels = labels.to(feats.device)
    opt = torch.optim.Adam(probe.parameters(), lr=0.05, weight_decay=1e-3)
    for _ in range(steps):
        logits = probe(x)
        loss = F.cross_entropy(logits, labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    probe.mu, probe.sd = mu, sd  # type: ignore[attr-defined]
    return probe.eval()


@torch.no_grad()
def probe_predict(probe: StyleProbe, feats: torch.Tensor) -> torch.Tensor:
    x = (feats - probe.mu) / probe.sd  # type: ignore[attr-defined]
    return probe(x).argmax(1)


def bin_stats(err: torch.Tensor, t: torch.Tensor, timesteps: int) -> list[float]:
    idx = (t * 6 // timesteps).cpu().numpy()
    per = err.flatten(1).mean(1).cpu().numpy()
    out = []
    for k in range(6):
        sel = idx == k
        out.append(float(per[sel].mean()) if sel.any() else float("nan"))
    return out


@torch.no_grad()
def denoise_report(model, lat, sid, schedule, cfg, *, batches: int) -> dict:
    model.eval()
    gen = torch.Generator(device="cuda").manual_seed(cfg.seed + 123)
    n = lat.shape[0]
    fixed_z, fixed_t, fixed_eps = [], [], []
    for _ in range(batches):
        idx = torch.randint(0, n, (cfg.batch_size,), device="cuda", generator=gen)
        z0 = lat.index_select(0, idx)
        t = torch.randint(0, schedule.timesteps, (z0.shape[0],), device="cuda", generator=gen)
        eps = torch.randn(z0.shape, device="cuda", dtype=z0.dtype, generator=gen)
        fixed_z.append(z0)
        fixed_t.append(t)
        fixed_eps.append(eps)
    z0 = torch.cat(fixed_z)
    t = torch.cat(fixed_t)
    eps = torch.cat(fixed_eps)
    style = sid.index_select(0, torch.randint(0, sid.shape[0], (z0.shape[0],), device="cuda", generator=gen))
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.use_bf16):
        pred_fixed = model(schedule.q_sample(z0, t, eps), t.float(), style).float()
    err_fixed = (pred_fixed - eps) ** 2
    fresh = torch.randn(z0.shape, device="cuda", dtype=z0.dtype, generator=gen)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.use_bf16):
        pred_fresh = model(schedule.q_sample(z0, t, fresh), t.float(), style).float()
        pred_null = model(schedule.q_sample(z0, t, fresh), t.float(),
                          torch.full_like(style, -1)).float()
    err_fresh = (pred_fresh - fresh) ** 2
    err_null = (pred_null - fresh) ** 2
    return {
        "loss_fixed_noise": float(err_fixed.mean()),
        "loss_fresh_noise": float(err_fresh.mean()),
        "loss_fresh_noise_null": float(err_null.mean()),
        "cond_gain": float(err_null.mean() / err_fresh.mean() - 1.0),
        "loss_overfit_gap": float(err_fresh.mean() - err_fixed.mean()),
        "bins_fixed_noise": bin_stats(err_fixed, t, schedule.timesteps),
        "bins_fresh_noise": bin_stats(err_fresh, t, schedule.timesteps),
    }


@torch.no_grad()
def generate(model, schedule, cfg, *, per_style: int, steps: int, cfg_scale: float,
             seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    sids = torch.arange(cfg.num_styles, device="cuda").repeat_interleave(per_style)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    z = torch.randn(sids.shape[0], cfg.latent_channels, 64, 64, device="cuda", generator=gen)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.use_bf16):
        out = ddim_sample(model, z, sids, schedule, t_start=schedule.timesteps - 1,
                          num_steps=steps, cfg_scale=cfg_scale)
    return out.float(), sids


@torch.no_grad()
def swap_report(model, schedule, cfg, val_lat: torch.Tensor, *, steps: int, seed: int,
                n_content: int = 8) -> dict:
    """同一内容 latent、同一噪声，只换 Style-ID：看输出是否有可测变化。"""
    x0 = val_lat[:n_content]
    outs = []
    for sid in range(cfg.num_styles):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.use_bf16):
            outs.append(sdedit(model, x0, torch.full((x0.shape[0],), sid, device="cuda"),
                               schedule, strength=0.45, num_steps=steps, cfg_scale=1.0, generator=gen))
    outs = torch.stack(outs, 0)  # [S, N, 4, 64, 64]
    pair = []
    for a in range(cfg.num_styles):
        for b in range(a + 1, cfg.num_styles):
            pair.append(float((outs[a] - outs[b]).pow(2).mean().sqrt()))
    to_input = [float((outs[s] - x0).pow(2).mean().sqrt()) for s in range(cfg.num_styles)]
    return {"pairwise_style_rms": float(np.mean(pair)),
            "to_input_rms": float(np.mean(to_input))}


@torch.no_grad()
def memorization(model, schedule, cfg, bank_lat: torch.Tensor, gen: torch.Tensor) -> dict:
    """生成样本与训练 latent 的最近余弦相似度：>0.98 视为疑似记忆。"""
    g = F.normalize(gen.flatten(1), dim=1)
    best = []
    for start in range(0, bank_lat.shape[0], 512):
        chunk = F.normalize(bank_lat[start:start + 512].flatten(1), dim=1)
        best.append((g @ chunk.T).max(1).values)
    maxsim = torch.cat(best)
    return {"nn_cos_mean": float(maxsim.mean()), "nn_cos_p95": float(maxsim.quantile(0.95)),
            "nn_cos_max": float(maxsim.max()),
            "frac_above_0.98": float((maxsim > 0.98).float().mean())}


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if str(payload.get("variant")) != "V4DIFF":
        raise SystemExit(f"unexpected variant {payload.get('variant')!r}")
    cfg = V4Config.from_mapping(payload["v4_config"])
    model = V4UNet(cfg).cuda()
    model.load_state_dict({k: v.float() for k, v in payload["model"].items()}, strict=True)
    model.eval()
    schedule = Schedule.linear(cfg.timesteps, cfg.beta_start, cfg.beta_end, device="cuda")

    run_dir = Path(args.run_dir) if args.run_dir else Path(args.ckpt).parent
    idx_path = run_dir / "indices.pt"
    if not idx_path.exists():
        raise SystemExit(f"缺少划分索引 {idx_path}；请先跑训练或指定 --run-dir")
    idx = torch.load(idx_path, map_location="cpu", weights_only=False)
    bank = load_bank(cfg.packed_dir)
    lat = bank.latents.cuda()
    sid = bank.style_ids.cuda()
    tr = lat.index_select(0, idx["train_idx"].cuda())
    tr_sid = sid.index_select(0, idx["train_idx"].cuda())
    va = lat.index_select(0, idx["val_idx"].cuda())
    va_sid = sid.index_select(0, idx["val_idx"].cuda())

    report: dict = {"ckpt": str(args.ckpt), "step": payload.get("step"),
                    "params": sum(p.numel() for p in model.parameters())}
    report["train_denoise"] = denoise_report(model, tr, tr_sid, schedule, cfg, batches=args.val_batches)
    report["val_denoise"] = denoise_report(model, va, va_sid, schedule, cfg, batches=args.val_batches)

    probe = fit_probe(latent_features(tr), tr_sid, cfg.num_styles)
    with torch.no_grad():
        val_acc = float((probe_predict(probe, latent_features(va)) == va_sid).float().mean())
    report["style_probe_val_acc"] = val_acc

    gen, sids = generate(model, schedule, cfg, per_style=args.gen_per_style,
                         steps=args.steps, cfg_scale=args.cfg, seed=args.seed)
    pred = probe_predict(probe, latent_features(gen))
    acc = float((pred == sids).float().mean())
    confusion = np.zeros((cfg.num_styles, cfg.num_styles), dtype=int)
    for t, p in zip(sids.cpu().numpy(), pred.cpu().numpy()):
        confusion[t, p] += 1
    report["gen"] = {"probe_acc": acc, "pred_hist": np.bincount(pred.cpu().numpy(),
                                                               minlength=cfg.num_styles).tolist(),
                     "confusion": confusion.tolist()}
    report["gen"]["memorization"] = memorization(model, schedule, cfg, tr, gen)
    report["swap"] = swap_report(model, schedule, cfg, va, steps=args.steps, seed=args.seed)

    if args.decode:
        from utils.inference import load_vae, decode_latent
        vae = load_vae("cuda", model_id="ema", cache_dir="/mnt/i/Latent_Style/eval_cache/hf")
        from PIL import Image
        img_dir = out / "samples"
        img_dir.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            for i in range(gen.shape[0]):
                img = decode_latent(vae, gen[i:i + 1], "cuda")
                arr = (img[0].float().permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
                name = f"{STYLE_NAMES[int(sids[i])]}_{i:03d}_pred{int(pred[i])}.png"
                Image.fromarray(arr).save(img_dir / name)

    (out / "eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "gen"}, indent=2))
    print("gen:", json.dumps(report["gen"], indent=2))


if __name__ == "__main__":
    main()
