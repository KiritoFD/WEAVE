"""Classifier Guidance 强度扫描（冻结分类器，不训练）。

固定内容/初始噪声/随机种子，只改目标 Style-ID 与引导强度；每个强度都记录：
目标风格 Top-1（独立评估分类器）、margin、生成多样性、训练集最近邻；
可选 VAE 解码存图，用于人工检查纹理伪影。

引导限幅在 ddim_sample 内完成：梯度按样本 L2 归一化，再乘 `scale·sigma_t`，
并只在 ``[t_min, t_max]``（分类器仍有辨别力的噪声区间）内生效。

用法（路径一律相对项目根）:
  python -m v4.guidance_scan --ckpt runs/v4/style_breakthrough/E1/step_04000.pt \
      --guide-clf runs/v4/style_clf/main.pt --eval-clf runs/v4/style_clf/alt.pt \
      --scales 0,0.1,0.2,0.4,0.8 --out runs/v4/style_breakthrough/E1/guidance.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .audit_cond import load_model
from .data import STYLE_NAMES, load_bank
from .diffusion import Schedule, ddim_sample
from .eval_style_ckpt import classify_at_low_noise, clf_scores, diversity_and_nn
from .style_classifier import load_classifier


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def make_batch_guidance(clf, target_sids: torch.Tensor, t_min: int, t_max: int):
    """按样本目标 ID 求 dCE/dz；样本顺序必须在整个采样过程中保持不变。"""

    def fn(z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        z = z.detach().requires_grad_(True)
        with torch.enable_grad():
            logits = clf(z, t)
            loss = F.cross_entropy(logits.float(), target_sids)
            grad, = torch.autograd.grad(loss, z)
        gate = ((t >= t_min) & (t <= t_max)).float().view(-1, 1, 1, 1)
        return grad * gate

    return fn


def save_pngs(gen: torch.Tensor, sids: torch.Tensor, out_dir: Path, *, cache_dir: str,
              prefix: str) -> None:
    from PIL import Image
    from utils.inference import decode_latent, load_vae

    vae = load_vae("cuda", model_id="ema", cache_dir=cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(gen.shape[0]):
        img = decode_latent(vae, gen[i:i + 1].cuda(), "cuda")
        arr = (img[0].float().permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
        Image.fromarray(arr).save(out_dir / f"{prefix}_{STYLE_NAMES[int(sids[i])]}_{i:03d}.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--guide-clf", required=True)
    ap.add_argument("--eval-clf", required=True, help="独立评估分类器")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scales", default="0,0.1,0.2,0.3,0.5,0.8")
    ap.add_argument("--t-min", type=int, default=0)
    ap.add_argument("--t-max", type=int, default=700)
    ap.add_argument("--per-style", type=int, default=4)
    ap.add_argument("--ddim-steps", type=int, default=20)
    ap.add_argument("--cfg-scale", type=float, default=1.0)
    ap.add_argument("--indices", default="runs/v4/arms/B/indices.pt")
    ap.add_argument("--decode", action="store_true", help="解码存图 + DINO 选择性")
    ap.add_argument("--vae-cache", default="eval_cache/hf")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = _device(args.device)
    model, cfg, info = load_model(args.ckpt, "", device)
    schedule = Schedule.linear(cfg.timesteps, cfg.beta_start, cfg.beta_end, device=device.type)
    guide, _ = load_classifier(args.guide_clf, device)
    eval_clf, _ = load_classifier(args.eval_clf, device)
    idx = torch.load(args.indices, map_location="cpu", weights_only=False)
    bank = load_bank(cfg.packed_dir)
    train_lat = bank.latents.index_select(0, idx["train_idx"].long()).to(device)

    sids = torch.arange(cfg.num_styles, device=device).repeat_interleave(args.per_style)
    gen = torch.Generator(device=device).manual_seed(args.seed + 31)
    z = torch.randn(sids.numel(), cfg.latent_channels, 64, 64, device=device, generator=gen)

    rows = []
    for scale in [float(x) for x in args.scales.split(",") if x.strip()]:
        fn = None if scale == 0.0 else make_batch_guidance(guide, sids, args.t_min, args.t_max)
        out = ddim_sample(model, z.clone(), sids, schedule, t_start=cfg.timesteps - 1,
                          num_steps=args.ddim_steps, cfg_scale=args.cfg_scale,
                          prediction_type=cfg.prediction_type,
                          guidance_fn=fn, guidance_scale=scale)
        scores = clf_scores(classify_at_low_noise(eval_clf, out, schedule,
                                                 seed=args.seed)["logits"], sids)
        div = diversity_and_nn(out, sids, train_lat, cfg.num_styles)
        row = {"scale": scale, **scores, **div,
               "latent_abs_mean": round(float(out.abs().mean()), 4),
               "frac_abs_gt6": round(float((out.abs() > 6).float().mean()), 4)}
        rows.append(row)
        print(f"[guide] scale={scale:<4} top1={row['top1']:.4f} margin={row['margin']:+.3f} "
              f"pair_rms={row['within_style_pairwise_rms']} nn_p95={row['nn_cos_p95']:.3f} "
              f"|z|>6={row['frac_abs_gt6']:.4f}", flush=True)
        if args.decode:
            save_pngs(out, sids, Path(args.out).parent / f"samples_s{scale:g}",
                      cache_dir=args.vae_cache,
                      prefix=Path(args.ckpt).stem)

    report = {"ckpt": args.ckpt, "step": info.get("step"), "prediction_type": cfg.prediction_type,
              "guide_clf": args.guide_clf, "eval_clf": args.eval_clf,
              "t_band": [args.t_min, args.t_max], "ddim_steps": args.ddim_steps,
              "cfg_scale": args.cfg_scale, "per_style": args.per_style, "rows": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
