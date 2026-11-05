#!/usr/bin/env python3
"""P1：定位「模糊/涂抹」产生在哪一步（同风格自重建，不跨风格）。

对固定源图（``data/test`` 每风格 N 张）取真实 latent ``z0``，在
t ∈ {50,150,300,450,700} 上分别测量：

1. 单步 ``x̂0 = pred_to_x0_eps(f(z_t,t,s))[0]`` 与 ``z0`` 的误差（整体 + 幅值分桶）；
2. 从该 t 起完整 DDIM 去噪到 0 后的误差（同口径）；
3. latent 子带 LL/LH/HL/HH 的 RMSE 与相对误差；
4. 解码后的 DINO-C（与源图 DINO CLS 余弦）与 LPIPS，并保存 原图/单步/完整 三联图。

另加「真实噪声 oracle」：stub 网络按解析解返回真 ε（或真 v），走同一套
``ddim_sample`` / ``sdedit``，若 oracle 都对不上 z0，说明采样器数学本身有问题。

用法（远程）：
  PYTHONPATH=<root> python3 tools/12h/v4_p1_localize.py \
      --ckpt runs/v4/style_breakthrough/E4/best.pt \
      --out runs/v4/style_breakthrough/p1

默认 ``--attn-impl channel``：历史 ckpt 是按旧通道注意力训练的，只有用同一实现
加载才与「产出 75 图的那个模型」等价。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools"):
    sys.path.insert(0, str(_p))

from paths import hf_cache_root, resolve  # noqa: E402
from v4.config import V4Config  # noqa: E402
from v4.data import STYLE_NAMES  # noqa: E402
from v4.diffusion import (Schedule, ddim_sample, pred_to_x0_eps,  # noqa: E402
                          sdedit)
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


def haar_bands(x: torch.Tensor) -> dict[str, torch.Tensor]:
    """一级 Haar 分解（偶数边长）；LL 为低频，LH/HL/HH 为三个高频方向。"""
    a, b = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
    c, d = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
    return {
        "LL": (a + b + c + d) * 0.5,
        "LH": (a + b - c - d) * 0.5,
        "HL": (a - b + c - d) * 0.5,
        "HH": (a - b - c + d) * 0.5,
    }


def _rmse(x: torch.Tensor) -> float:
    return float(x.pow(2).mean().sqrt())


def latent_metrics(hat: torch.Tensor, z0: torch.Tensor) -> dict:
    """整体 RMSE/相对误差 + 子带 + 幅值三分桶（|z0| 的 1/3、2/3 分位）。"""
    diff = hat - z0
    out: dict = {
        "rmse": _rmse(diff),
        "rel": _rmse(diff) / max(_rmse(z0), 1e-12),
        "bands": {},
        "buckets": {},
    }
    hb, zb = haar_bands(hat), haar_bands(z0)
    for key in ("LL", "LH", "HL", "HH"):
        out["bands"][key] = {
            "rmse": _rmse(hb[key] - zb[key]),
            "rel": _rmse(hb[key] - zb[key]) / max(_rmse(zb[key]), 1e-12),
            "true_rms": _rmse(zb[key]),
        }
    mag = z0.abs().flatten()
    lo, hi = torch.quantile(mag, torch.tensor([1 / 3, 2 / 3], device=mag.device)).tolist()
    for name, sel in (("small", z0.abs() <= lo),
                      ("mid", (z0.abs() > lo) & (z0.abs() <= hi)),
                      ("large", z0.abs() > hi)):
        out["buckets"][name] = {
            "frac": float(sel.float().mean()),
            "rmse": _rmse(diff[sel]),
            "rel": _rmse(diff[sel]) / max(_rmse(z0[sel]), 1e-12),
        }
    out["bucket_edges"] = [round(lo, 5), round(hi, 5)]
    return out


class OracleModel(nn.Module):
    """解析 oracle：给定 z0，从 z_t 反解真 ε（或真 v），用于验证采样器数学。"""

    def __init__(self, schedule: Schedule, z0: torch.Tensor, prediction_type: str) -> None:
        super().__init__()
        self.schedule = schedule
        self.z0 = z0
        self.prediction_type = prediction_type

    def forward(self, z: torch.Tensor, t: torch.Tensor,
                style_id: torch.Tensor) -> torch.Tensor:
        t_long = t.long()
        ab = self.schedule.alpha_bar[t_long].reshape(-1, 1, 1, 1)
        eps = (z - ab.sqrt() * self.z0) / (1.0 - ab).sqrt().clamp_min(1e-12)
        if self.prediction_type == "v":
            alpha, sigma = self.schedule.alpha_sigma(t_long)
            return alpha * eps - sigma * self.z0
        return eps


@torch.no_grad()
def run_oracle(schedule: Schedule, z0: torch.Tensor, sid: torch.Tensor,
               ts: list[int], num_steps: int, seed: int, strength: float) -> dict:
    """对每个 t 与两种 prediction_type：完整 DDIM 与 sdedit 都应精确还原 z0。"""
    out: dict = {}
    for ptype in ("eps", "v"):
        oracle = OracleModel(schedule, z0, ptype)
        rows = {}
        for t in ts:
            g = torch.Generator(device=z0.device).manual_seed(seed * 10000 + t)
            noise = torch.randn(z0.shape, device=z0.device, generator=g)
            zt = schedule.q_sample(z0, torch.full((z0.shape[0],), t, device=z0.device), noise)
            rec = ddim_sample(oracle, zt, sid, schedule, t_start=t, num_steps=num_steps,
                              prediction_type=ptype)
            rows[f"t{t}"] = {"max_abs": float((rec - z0).abs().max()),
                             "rmse": _rmse(rec - z0)}
        # sdedit 自重建（strength 固定 0.45）
        g = torch.Generator(device=z0.device).manual_seed(seed * 10000 + 999)
        rec = sdedit(oracle, z0, sid, schedule, strength=strength, num_steps=num_steps,
                     prediction_type=ptype, generator=g)
        rows[f"sdedit_s{strength}"] = {"max_abs": float((rec - z0).abs().max()),
                                       "rmse": _rmse(rec - z0)}
        out[ptype] = rows
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--test-dir", default="data/test")
    ap.add_argument("--per-style", type=int, default=2)
    ap.add_argument("--ts", default="50,150,300,450,700")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--strength", type=float, default=0.45)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--attn-impl", default="channel", choices=["channel", "spatial"])
    ap.add_argument("--dino-cache", default="", help="空=用 paths.hf_cache_root()")
    ap.add_argument("--no-image-metrics", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    t_wall = time.time()
    out_dir = resolve(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    img_dir.mkdir(exist_ok=True)
    ts = [int(v) for v in args.ts.split(",") if v.strip()]

    payload = torch.load(resolve(args.ckpt), map_location="cpu", weights_only=False)
    if str(payload.get("variant")) != "V4DIFF":
        raise SystemExit(f"unexpected variant {payload.get('variant')!r}")
    cfg = V4Config.from_mapping(dict(payload["v4_config"], attn_impl=args.attn_impl))
    model = V4UNet(cfg).cuda()
    model.load_state_dict({k: v.float() for k, v in payload["model"].items()}, strict=True)
    model.eval()
    schedule = Schedule.linear(cfg.timesteps, cfg.beta_start, cfg.beta_end, device="cuda")

    test_dir = resolve(args.test_dir)
    sources: list[tuple[int, Path]] = []
    for sid, name in enumerate(STYLE_NAMES):
        files = sorted([p for p in (test_dir / name).iterdir()
                        if p.suffix.lower() in IMAGE_EXTS])[: args.per_style]
        if not files:
            raise SystemExit(f"没有源图: {test_dir / name}")
        sources.extend((sid, p) for p in files)

    from utils.inference import decode_latent, encode_image, load_vae
    vae = load_vae("cuda", model_id="ema", cache_dir=str(hf_cache_root()))

    z0_list: list[torch.Tensor] = []
    for _, path in sources:
        z0_list.append(encode_image(vae, load_image(path).unsqueeze(0).cuda()).float())
    z0 = torch.cat(z0_list, 0)
    sid_all = torch.tensor([s for s, _ in sources], device="cuda")
    print(f"[p1] ckpt={args.ckpt} step={payload.get('step')} "
          f"attn_impl={args.attn_impl} pred={cfg.prediction_type} n_src={len(sources)} "
          f"latent={tuple(z0.shape)}", flush=True)

    oracle = run_oracle(schedule, z0, sid_all, ts, args.steps, args.seed, args.strength)
    print("[p1] oracle:", json.dumps(oracle, ensure_ascii=False), flush=True)

    levels: list[dict] = []
    rows: list[dict] = []
    for ti, t in enumerate(ts):
        t0 = time.time()
        g = torch.Generator(device="cuda").manual_seed(args.seed * 10000 + t)
        noise = torch.randn(z0.shape, device="cuda", generator=g)
        t_batch = torch.full((z0.shape[0],), float(t), device="cuda")
        zt = schedule.q_sample(z0, torch.full((z0.shape[0],), t, device="cuda"), noise)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.use_bf16):
            pred = model(zt, t_batch, sid_all)
        x0_hat = pred_to_x0_eps(pred.float(), zt, t, schedule, cfg.prediction_type)[0]
        t_single = time.time() - t0

        t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.use_bf16):
            x0_full = ddim_sample(model, zt, sid_all, schedule, t_start=t,
                                  num_steps=args.steps, prediction_type=cfg.prediction_type)
        sec_full = time.time() - t0
        x0_full = x0_full.float()

        level = {
            "t": t,
            "single": latent_metrics(x0_hat, z0),
            "full": latent_metrics(x0_full, z0),
            "sec_single": round(t_single, 3),
            "sec_full_ddim": round(sec_full, 3),
            "sec_per_step": round(sec_full / max(args.steps, 1), 4),
        }
        levels.append(level)
        print(f"[p1] t={t} single_rmse={level['single']['rmse']:.5f} "
              f"full_rmse={level['full']['rmse']:.5f} "
              f"LL_rel={level['full']['bands']['LL']['rel']:.4f} "
              f"HH_rel={level['full']['bands']['HH']['rel']:.4f}", flush=True)

        for i, (src_sid, path) in enumerate(sources):
            orig = decode_latent(vae, z0[i:i + 1])[0].float().cpu()
            single = decode_latent(vae, x0_hat[i:i + 1])[0].float().cpu()
            full = decode_latent(vae, x0_full[i:i + 1])[0].float().cpu()
            tag = f"{i:02d}_{STYLE_NAMES[src_sid]}_t{t}"
            save_image(orig, img_dir / f"{tag}_orig.png")
            save_image(single, img_dir / f"{tag}_single.png")
            save_image(full, img_dir / f"{tag}_full.png")
            rows.append({"t": t, "idx": i, "style": STYLE_NAMES[src_sid],
                         "src": str(path), "tag": tag})

    del model, vae
    torch.cuda.empty_cache()

    image_metrics: dict = {"dino_c": {}, "lpips": {}}
    if not args.no_image_metrics:
        src_x = torch.cat([load_image(Path(r["src"])) for r in rows], 0)
        single_x = torch.cat([load_image(img_dir / f"{r['tag']}_single.png") for r in rows], 0)
        full_x = torch.cat([load_image(img_dir / f"{r['tag']}_full.png") for r in rows], 0)

        try:
            import lpips
            from utils.run_evaluation import _lpips_forward_safe
            loss_fn = lpips.LPIPS(net="alex", verbose=False).cuda().eval()
            d_single = _lpips_forward_safe(loss_fn, single_x.cuda(), src_x.cuda(), device="cuda",
                                           chunk_size=2, cpu_fallback=True, tag="p1-single")
            d_full = _lpips_forward_safe(loss_fn, full_x.cuda(), src_x.cuda(), device="cuda",
                                         chunk_size=2, cpu_fallback=True, tag="p1-full")
            for r, a, b in zip(rows, d_single.tolist(), d_full.tolist()):
                r["lpips_single"], r["lpips_full"] = round(float(a), 5), round(float(b), 5)
            del loss_fn
            torch.cuda.empty_cache()
            print("[p1] lpips done", flush=True)
        except Exception as exc:  # 失败留痕
            import traceback
            traceback.print_exc()
            image_metrics["lpips_error"] = f"{type(exc).__name__}: {exc}"

        try:
            from utils.compute_dino_metrics import load_dino
            device = "cuda" if torch.cuda.is_available() else "cpu"
            cache = args.dino_cache or str(hf_cache_root())
            dino = load_dino("facebook/dinov2-small", device, cache, allow_network=False)
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

            @torch.inference_mode()
            def dino_cls(x: torch.Tensor) -> torch.Tensor:
                chunks = []
                for i in range(0, x.shape[0], 2):
                    px = torch.nn.functional.interpolate(
                        x[i:i + 2].to(device), size=(224, 224), mode="bicubic",
                        align_corners=False, antialias=True)
                    px = (px - mean) / std
                    o = dino(px)
                    chunks.append(torch.nn.functional.normalize(
                        o.last_hidden_state[:, 0, :].float(), dim=-1))
                return torch.cat(chunks, 0)

            f_src = dino_cls(src_x)
            f_single = dino_cls(single_x)
            f_full = dino_cls(full_x)
            for r, a, b in zip(rows, (f_single * f_src).sum(-1).tolist(),
                               (f_full * f_src).sum(-1).tolist()):
                r["dino_c_single"], r["dino_c_full"] = round(float(a), 5), round(float(b), 5)
            print("[p1] dino done", flush=True)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            image_metrics["dino_error"] = f"{type(exc).__name__}: {exc}"

    # ---- 汇总 ----
    for level in levels:
        t = level["t"]
        for key, col in (("single", "dino_c_single"), ("full", "dino_c_full"),
                         ("single", "lpips_single"), ("full", "lpips_full")):
            vals = [r[col] for r in rows if r["t"] == t and col in r]
            if vals:
                level.setdefault(key, {})[
                    "dino_c" if col.startswith("dino") else "lpips"] = round(float(np.mean(vals)), 5)

    summary = {
        "ckpt": args.ckpt, "step": payload.get("step"), "attn_impl": args.attn_impl,
        "prediction_type": cfg.prediction_type, "num_steps": args.steps,
        "ts": ts, "seed": args.seed, "n_src": len(sources), "per_style": args.per_style,
        "oracle": oracle, "levels": levels, "image_metrics": image_metrics,
        "wall_s": round(time.time() - t_wall, 1),
    }
    (out_dir / "p1_metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                             encoding="utf-8")
    with (out_dir / "rows.csv").open("w", encoding="utf-8") as f:
        keys = ["t", "idx", "style", "tag", "lpips_single", "lpips_full",
                "dino_c_single", "dino_c_full"]
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
