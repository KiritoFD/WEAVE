#!/usr/bin/env python3
"""实验 A：高频可恢复性上限与误差归因。

1) 用真实训练 latent 估计 **Style × 子带 × 通道** 的方差，按同噪声日程构造
   对角高斯线性 MMSE（Wiener）基线，作为「可恢复上限」参照；
2) 与 E4 在同一批噪声下比：子带 MSE / 相对误差 / 高频功率比 / 高频相关系数 /
   E4 相对线性基线的增益；
3) 归因用的 GT 子带替换（不进训练、不进正式推理）：
   - ``pred+trueHF``：LL 用预测、LH/HL/HH 用 GT
   - ``pred+trueLL``：LL 用 GT、HF 用预测
   分别解码测 DINO-C / LPIPS，分离「高频缺失」与「低频结构错误」。

用法：
  PYTHONPATH=<root> python3 tools/12h/v4_p1b_mmse.py \
      --ckpt runs/v4/style_breakthrough/E4/best.pt \
      --out runs/v4/style_breakthrough/p1b
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from paths import hf_cache_root, resolve  # noqa: E402
from v4.config import V4Config  # noqa: E402
from v4.data import STYLE_NAMES, load_bank  # noqa: E402
from v4.diffusion import Schedule, ddim_sample, pred_to_x0_eps  # noqa: E402
from v4.model import V4UNet  # noqa: E402

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def load_image(path: Path, size: int = 512) -> torch.Tensor:
    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.size != (size, size):
            im = im.resize((size, size), Image.BICUBIC)
        arr = np.asarray(im, dtype="uint8").copy()
    return torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0)


def save_image(x: torch.Tensor, path: Path) -> None:
    arr = (x.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


def haar(x: torch.Tensor) -> dict[str, torch.Tensor]:
    a, b = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
    c, d = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
    return {"LL": (a + b + c + d) * 0.5, "LH": (a + b - c - d) * 0.5,
            "HL": (a - b + c - d) * 0.5, "HH": (a - b - c + d) * 0.5}


def ihaar(bands: dict[str, torch.Tensor]) -> torch.Tensor:
    ll, lh, hl, hh = (bands[k] for k in ("LL", "LH", "HL", "HH"))
    a = (ll + lh + hl + hh) * 0.5
    b = (ll + lh - hl - hh) * 0.5
    c = (ll - lh + hl - hh) * 0.5
    d = (ll - lh - hl + hh) * 0.5
    out = torch.empty(ll.shape[0], ll.shape[1], ll.shape[2] * 2, ll.shape[3] * 2,
                      device=ll.device, dtype=ll.dtype)
    out[..., 0::2, 0::2], out[..., 0::2, 1::2] = a, b
    out[..., 1::2, 0::2], out[..., 1::2, 1::2] = c, d
    return out


def rmse(x: torch.Tensor) -> float:
    return float(x.detach().pow(2).mean().sqrt())


def corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().flatten().float()
    b = b.detach().flatten().float()
    a = a - a.mean()
    b = b - b.mean()
    den = a.pow(2).sum().sqrt() * b.pow(2).sum().sqrt()
    return float((a * b).sum() / den.clamp_min(1e-12))


def band_stats(pred: torch.Tensor, gt: torch.Tensor) -> dict:
    pb, gb = haar(pred), haar(gt)
    out = {}
    for k in ("LL", "LH", "HL", "HH"):
        d = pb[k] - gb[k]
        out[k] = {
            "mse": float(d.detach().pow(2).mean()),
            "rel": rmse(d) / max(rmse(gb[k]), 1e-12),
            "power_ratio": rmse(pb[k]) / max(rmse(gb[k]), 1e-12),
            "corr": corr(pb[k], gb[k]),
        }
    return out


def band_variance_map(var_style: torch.Tensor, sid_all: torch.Tensor,
                      h: int, w: int) -> torch.Tensor:
    """把 ``[S,4,C]`` 的按子带方差铺回 ``[B,C,H,W]`` 的空间位置。"""
    out = torch.empty(sid_all.shape[0], var_style.shape[2], h, w, device=var_style.device)
    layouts = [(slice(0, None, 2), slice(0, None, 2)), (slice(0, None, 2), slice(1, None, 2)),
               (slice(1, None, 2), slice(0, None, 2)), (slice(1, None, 2), slice(1, None, 2))]
    for si, sid in enumerate(sid_all.tolist()):
        for bi, (rs, cs) in enumerate(layouts):
            out[si, :, rs, cs] = var_style[sid, bi].view(-1, 1, 1)
    return out


def mmse_estimate(zt: torch.Tensor, var: torch.Tensor, ab: float) -> torch.Tensor:
    """对角高斯 MMSE：ẑ0 = (√ab σ²)/(ab σ² + (1-ab)) · z_t（var 随 style 广播）。"""
    a = float(ab) ** 0.5
    num = a * var
    den = ab * var + (1.0 - ab)
    return num / den * zt


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-dir", default="data/train")
    ap.add_argument("--test-dir", default="data/test")
    ap.add_argument("--var-per-style", type=int, default=10)
    ap.add_argument("--eval-per-style", type=int, default=2)
    ap.add_argument("--ts", default="50,150,300,450")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--attn-impl", default="channel")
    args = ap.parse_args()
    t_wall = time.time()
    ts = [int(v) for v in args.ts.split(",")]
    out_dir = resolve(args.out)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    payload = torch.load(resolve(args.ckpt), map_location="cpu", weights_only=False)
    cfg = V4Config.from_ckpt_payload(payload)
    cfg.attn_impl = args.attn_impl
    model = V4UNet(cfg).cuda()
    model.load_state_dict({k: v.float() for k, v in payload["model"].items()}, strict=True)
    model.eval()
    schedule = Schedule.linear(cfg.timesteps, cfg.beta_start, cfg.beta_end, device="cuda")

    from utils.inference import decode_latent, encode_image, load_vae
    vae = load_vae("cuda", model_id="ema", cache_dir=str(hf_cache_root()))

    # ---- 1) 用训练 latent 估 Style × 子带 × 通道 方差 ----
    bank = load_bank(resolve(args.train_dir))  # 打包好的 [N,4,64,64]
    var = torch.zeros(len(STYLE_NAMES), 4, cfg.latent_channels, device="cuda")
    bands_key = ["LL", "LH", "HL", "HH"]
    for sid in range(len(STYLE_NAMES)):
        idx = torch.nonzero(bank.style_ids == sid).flatten()[: args.var_per_style]
        zs = bank.latents.index_select(0, idx).cuda().float()
        zb = haar(zs)
        for bi, k in enumerate(bands_key):
            stack = zb[k]  # [N,C,H,W]
            var[sid, bi] = stack.var(dim=(0, 2, 3), unbiased=False)
    print("[A] band var mean LL/HH:", float(var[:, 0].mean()), float(var[:, 3].mean()), flush=True)

    # ---- 2) 评测集 ----
    test_dir = resolve(args.test_dir)
    items: list[tuple[int, Path]] = []
    for sid, name in enumerate(STYLE_NAMES):
        files = sorted([p for p in (test_dir / name).iterdir()
                        if p.suffix.lower() in IMAGE_EXTS])[: args.eval_per_style]
        items.extend((sid, p) for p in files)
    z0 = torch.cat([encode_image(vae, load_image(p).unsqueeze(0).cuda()).float()
                    for _, p in items], 0).detach()
    sid_all = torch.tensor([s for s, _ in items], device="cuda")
    print(f"[A] eval n={len(items)} latent={tuple(z0.shape)}", flush=True)

    rows, levels = [], []
    for t in ts:
        g = torch.Generator(device="cuda").manual_seed(args.seed * 10000 + t)
        noise = torch.randn(z0.shape, device="cuda", generator=g)
        t_batch = torch.full((z0.shape[0],), float(t), device="cuda")
        zt = schedule.q_sample(z0, torch.full((z0.shape[0],), t, device="cuda"), noise)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.use_bf16):
            pred = model(zt, t_batch, sid_all)
        x0_e4 = pred_to_x0_eps(pred.float(), zt, t, schedule, cfg.prediction_type)[0].detach()

        ab = float(schedule.alpha_bar[t])
        var_map = band_variance_map(var, sid_all, z0.shape[2], z0.shape[3])
        x0_lin = mmse_estimate(zt, var_map, ab).detach()

        pb, gb, lb = haar(x0_e4), haar(z0), haar(x0_lin)
        # 子带替换
        hf_true = {k: (gb[k] if k != "LL" else pb[k]) for k in ("LL", "LH", "HL", "HH")}
        ll_true = {k: (gb[k] if k == "LL" else pb[k]) for k in ("LL", "LH", "HL", "HH")}
        x0_thf, x0_tll = ihaar(hf_true), ihaar(ll_true)

        lv = {"t": t,
              "e4": band_stats(x0_e4, z0),
              "linear": band_stats(x0_lin, z0),
              "e4_trueHF": band_stats(x0_thf, z0),
              "e4_trueLL": band_stats(x0_tll, z0),
              "gain_vs_linear": {
                  k: (float((pb[k] - gb[k]).pow(2).mean()), float((lb[k] - gb[k]).pow(2).mean()),
                      1.0 - float((pb[k] - gb[k]).pow(2).mean()) /
                      max(float((lb[k] - gb[k]).pow(2).mean()), 1e-12))
                  for k in ("LL", "LH", "HL", "HH")},
              "bands_gt": {k: rmse(gb[k]) for k in ("LL", "LH", "HL", "HH")}}
        levels.append(lv)
        print(f"[A] t={t} e4_HH_rel={lv['e4']['HH']['rel']:.4f} "
              f"lin_HH_rel={lv['linear']['HH']['rel']:.4f} "
              f"e4_corr_HH={lv['e4']['HH']['corr']:.4f} gain_HH={lv['gain_vs_linear']['HH'][2]:+.3f}",
              flush=True)

        for i, (src_sid, path) in enumerate(items):
            tag = f"{i:02d}_{STYLE_NAMES[src_sid]}_t{t}"
            for name, lat in (("e4", x0_e4), ("lin", x0_lin),
                              ("trueHF", x0_thf), ("trueLL", x0_tll)):
                img = decode_latent(vae, lat[i:i + 1])[0].float().cpu()
                save_image(img, out_dir / "images" / f"{tag}_{name}.png")
            rows.append({"t": t, "idx": i, "style": STYLE_NAMES[src_sid], "tag": tag,
                         "src": str(path)})

    del model, vae
    torch.cuda.empty_cache()

    # ---- 3) 图像级指标 ----
    variants = ["e4", "lin", "trueHF", "trueLL"]
    src_x = torch.stack([load_image(Path(r["src"])) for r in rows], 0)
    try:
        import lpips
        from utils.run_evaluation import _lpips_forward_safe
        loss_fn = lpips.LPIPS(net="alex", verbose=False).cuda().eval()
        for v in variants:
            gen = torch.stack([load_image(out_dir / "images" / f"{r['tag']}_{v}.png")
                               for r in rows], 0)
            d = _lpips_forward_safe(loss_fn, gen.cuda(), src_x.cuda(), device="cuda",
                                    chunk_size=4, cpu_fallback=True, tag=f"A-{v}")
            for r, val in zip(rows, d.tolist()):
                r[f"lpips_{v}"] = round(float(val), 5)
        del loss_fn
        torch.cuda.empty_cache()
    except Exception as exc:
        print("[A] lpips failed:", type(exc).__name__, exc, flush=True)

    try:
        from utils.compute_dino_metrics import load_dino
        cache = str(hf_cache_root())
        dino = load_dino("facebook/dinov2-small", "cuda", cache, allow_network=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)

        def feats(x: torch.Tensor) -> torch.Tensor:
            outs = []
            for i in range(0, x.shape[0], 4):
                px = torch.nn.functional.interpolate(
                    x[i:i + 4].cuda(), size=(224, 224), mode="bicubic",
                    align_corners=False, antialias=True)
                o = dino((px - mean) / std)
                outs.append(torch.nn.functional.normalize(o.last_hidden_state[:, 0, :].float(), dim=-1))
            return torch.cat(outs, 0)

        f_src = feats(src_x)
        for v in variants:
            gen = torch.stack([load_image(out_dir / "images" / f"{r['tag']}_{v}.png")
                               for r in rows], 0)
            f = feats(gen)
            for r, val in zip(rows, (f * f_src).sum(-1).tolist()):
                r[f"dino_c_{v}"] = round(float(val), 5)
    except Exception as exc:
        print("[A] dino failed:", type(exc).__name__, exc, flush=True)

    for lv in levels:
        for v in variants:
            dc = [r[f"dino_c_{v}"] for r in rows if r["t"] == lv["t"] and f"dino_c_{v}" in r]
            lp = [r[f"lpips_{v}"] for r in rows if r["t"] == lv["t"] and f"lpips_{v}" in r]
            if dc:
                lv.setdefault(v, {})["dino_c"] = round(float(np.mean(dc)), 5)
            if lp:
                lv.setdefault(v, {})["lpips"] = round(float(np.mean(lp)), 5)

    summary = {"ckpt": args.ckpt, "step": payload.get("step"), "attn_impl": args.attn_impl,
               "ts": ts, "seed": args.seed, "n_eval": len(items),
               "levels": levels, "wall_s": round(time.time() - t_wall, 1)}
    (out_dir / "p1b_metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                             encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
