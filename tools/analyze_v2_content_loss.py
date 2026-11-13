"""离线内容损失拆解：对 step_002500 的 z_c / ẑ / 目标风格参考图做子带、Lab、SSIM、残差分析。"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from tools.legacy_root.config_schema import load_experiment_config  # noqa: E402
from eval_fast import build_protocol, build_source_latents, build_target_latents, _load_eval_image_tensor  # noqa: E402
from tools.legacy_utils.inference import LGTInference, decode_latent, load_vae  # noqa: E402
from tools.legacy_root.wavelet import dwt2_haar  # noqa: E402


def _gaussian_kernel(size: int = 11, sigma: float = 1.5, device="cuda", dtype=torch.float32) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return g[:, None] @ g[None, :]


def ssim_map(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """x/y: (B,3,H,W) in [0,1]; 返回逐像素 SSIM 图（已通道平均）。"""
    c1, c2 = 0.01**2, 0.03**2
    k = _gaussian_kernel(device=x.device, dtype=x.dtype).expand(3, 1, 11, 11)
    mu_x = F.conv2d(x, k, groups=3, padding=5)
    mu_y = F.conv2d(y, k, groups=3, padding=5)
    var_x = F.conv2d(x * x, k, groups=3, padding=5) - mu_x**2
    var_y = F.conv2d(y * y, k, groups=3, padding=5) - mu_y**2
    cov = F.conv2d(x * y, k, groups=3, padding=5) - mu_x * mu_y
    num = (2 * mu_x * mu_y + c1) * (2 * cov + c2)
    den = (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    return (num / den).mean(dim=1, keepdim=True)


def sobel_mag(x: torch.Tensor) -> torch.Tensor:
    lum = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    gx = F.conv2d(lum, kx, padding=1)
    gy = F.conv2d(lum, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


def rgb_to_lab(x: torch.Tensor) -> torch.Tensor:
    """x: (B,3,H,W) sRGB [0,1] -> Lab (D65)."""
    x = torch.clamp(x, 1e-6, 1.0)
    lin = torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).float()
    m = torch.tensor(
        [[0.4124564, 0.3575761, 0.1804375],
         [0.2126729, 0.7151522, 0.0721750],
         [0.0193339, 0.1191920, 0.9503041]],
        device=x.device, dtype=torch.float32,
    )
    xyz = torch.einsum("ij,bjhw->bihw", m, lin) / 0.95047  # X 归一化到 D65 白点
    xyz[:, 1] = xyz[:, 1] / 1.0
    xyz[:, 2] = xyz[:, 2] / 1.08883
    f = torch.where(xyz > 0.008856, xyz.clamp_min(1e-8) ** (1 / 3), 7.787 * xyz + 16 / 116)
    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return torch.stack([L, a, b], dim=1)


def _band_stats(x: torch.Tensor) -> dict[str, float]:
    ll, lh, hl, hh = dwt2_haar(x.float())
    out = {}
    for name, band in (("LL", ll), ("LH", lh), ("HL", hl), ("HH", hh)):
        out[f"energy_{name}"] = float(band.pow(2).mean())
        out[f"std_{name}"] = float(band.std())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/v2/V1/step_002500_bare.pt")
    ap.add_argument("--config", default="configs/v2/V1.json")
    ap.add_argument("--test-dir", default="data/test")
    ap.add_argument("--out", default="runs/v2/analysis/step_002500_content_loss.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--decode-batch", type=int, default=8)
    ap.add_argument("--num-steps", type=int, default=8)
    ap.add_argument("--step-size", type=float, default=1.0)
    ap.add_argument("--style-strength", type=float, default=2.0)
    ap.add_argument("--max-src-samples", type=int, default=30)
    args = ap.parse_args()

    device = args.device
    os.environ["SB_STRICT_CKPT"] = "1"
    lgt = LGTInference(
        model_path=args.checkpoint, device=device, num_steps=args.num_steps, step_size=args.step_size,
        style_strength=args.style_strength, residual_scale=1.0, config_override_path=None,
    )
    cache_root = ROOT.parent / "eval_cache"
    hf_cache = str(cache_root / "hf")
    vae = load_vae(device, model_id="ema", cache_dir=hf_cache)

    test_images, requests, style_meta = build_protocol(
        Path(args.test_dir), max_src_samples=args.max_src_samples, ref_pool="first"
    )
    source_paths = sorted({r["src_path"] for r in requests})
    req_map = {(r["src_path"], r["tgt_style_id"]): r for r in requests}
    ns = SimpleNamespace(source_latent_cache=True, cache_dir=str(cache_root), vae_model="ema", encode_batch=8)
    source_latents = build_source_latents(lgt, vae, source_paths, device, ns, SimpleNamespace(start=lambda *_: None, stop=lambda *_: None))
    target_latents = build_target_latents(lgt, vae, style_meta, device, ns, SimpleNamespace(start=lambda *_: None, stop=lambda *_: None))

    model_scale = float(getattr(lgt.model, "latent_scale_factor", 0.18215))
    vae_scale = float(getattr(getattr(vae, "config", None), "scaling_factor", model_scale))
    scale_out = vae_scale / max(model_scale, 1e-8)
    src_images = {p: _load_eval_image_tensor(p) for p in source_paths}

    stats = {
        "pairs": 0,
        "identity_dist": [], "lab_dL": [], "lab_da": [], "lab_db": [], "ssim": [], "sobel_ratio": [],
        "grid_ssim": torch.zeros(16, 16, dtype=torch.float64, device=device),
        "grid_hf": torch.zeros(16, 16, dtype=torch.float64, device=device),
        "subband": {k: [] for k in ("LL", "LH", "HL", "HH")},
        "subband_fake": {k: [] for k in ("LL", "LH", "HL", "HH")},
        "subband_target": {k: [] for k in ("LL", "LH", "HL", "HH")},
        "resid_band": {k: [] for k in ("LL", "LH", "HL", "HH")},
        "spec_num": 0.0, "spec_den": 0.0,
        "lab_abs_b_gt3": 0.0, "lab_pixels": 0.0,
    }
    resid_norms: list[float] = []

    for b0 in range(0, len(source_paths), args.batch):
        chunk_paths = source_paths[b0:b0 + args.batch]
        lat_c = source_latents[b0:b0 + len(chunk_paths)].to(device=device, dtype=torch.float32)
        img_c = torch.stack([src_images[p] for p in chunk_paths]).to(device)
        with torch.inference_mode():
            x0 = lgt.inversion(lat_c.to(torch.float32))
        for tgt in sorted(style_meta):
            rows = [req_map[(p, tgt)] for p in chunk_paths]
            n = len(rows)
            with torch.inference_mode():
                style_lat = target_latents[tgt].to(device=device, dtype=torch.float32).expand(n, -1, -1, -1).contiguous()
                tgt_ids = torch.full((n,), tgt, dtype=torch.long, device=device)
                z_hat = lgt.generation_with_target_latent(x0, tgt_ids, target_style_latent={"style_latent_tensor": style_lat})
            if abs(scale_out - 1.0) > 1e-4:
                z_hat = z_hat * scale_out
            keep = [i for i, r in enumerate(rows) if not r["same_style"]]
            if not keep:
                continue
            z_c = lat_c[keep].float()
            zh = z_hat[keep].float()
            with torch.inference_mode():
                dec = torch.cat([decode_latent(vae, z_hat[keep][s:s + args.decode_batch], device)
                                 for s in range(0, len(keep), args.decode_batch)], 0)
            gen01 = dec.detach().clamp(0, 1).mul(255).round().div(255).float()
            src01 = img_c[keep]
            if gen01.shape[-2:] != src01.shape[-2:]:
                gen01 = F.interpolate(gen01, size=src01.shape[-2:], mode="bilinear", align_corners=False)
            z_t = target_latents[tgt].to(device=device, dtype=torch.float32).expand(len(keep), -1, -1, -1)

            # 子带 + 残差
            for name, lc, lf, lt in zip(("LL", "LH", "HL", "HH"),
                                        dwt2_haar(z_c), dwt2_haar(zh), dwt2_haar(z_t)):
                stats["subband"][name].append(float(lc.pow(2).mean()))
                stats["subband_fake"][name].append(float(lf.pow(2).mean()))
                stats["subband_target"][name].append(float(lt.pow(2).mean()))
            for name, band in zip(("LL", "LH", "HL", "HH"), dwt2_haar(zh - z_c)):
                stats["resid_band"][name].append(float(band.pow(2).mean()))
            norms = (zh - z_c).flatten(1).norm(dim=1)
            resid_norms.extend(norms.detach().cpu().tolist())
            stats["identity_dist"].append(float(norms.mean()))
            # 残差频谱重心
            spec = torch.fft.fftshift(torch.fft.fft2(zh - z_c, dim=(-2, -1))).abs().mean(dim=(0, 1))
            h, w = spec.shape
            yy = torch.arange(h, device=device, dtype=torch.float32) - h / 2
            xx = torch.arange(w, device=device, dtype=torch.float32) - w / 2
            rad = torch.sqrt(yy[:, None] ** 2 + xx[None, :] ** 2) / math.sqrt((h / 2) ** 2 + (w / 2) ** 2)
            stats["spec_num"] += float((rad * spec).sum())
            stats["spec_den"] += float(spec.sum())

            # Lab
            lab_c, lab_g = rgb_to_lab(src01), rgb_to_lab(gen01)
            d = lab_g - lab_c
            for i, k in enumerate(("lab_dL", "lab_da", "lab_db")):
                stats[k].append(float(d[:, i].mean()))
            stats["lab_abs_b_gt3"] += float((d[:, 2].abs() > 3).sum())
            stats["lab_pixels"] += float(d[:, 2].numel())

            # SSIM / 网格 / 边缘
            sm = ssim_map(src01, gen01)
            stats["ssim"].append(float(sm.mean()))
            gpool = F.adaptive_avg_pool2d(sm, 16)[:, 0].double()
            stats["grid_ssim"] += gpool.sum(dim=0)
            hf = (src01.mean(dim=1, keepdim=True) - F.avg_pool2d(src01.mean(dim=1, keepdim=True), 5, 1, 2))
            gpool_hf = F.adaptive_avg_pool2d(hf.abs(), 16)[:, 0].double()
            stats["grid_hf"] += gpool_hf.sum(dim=0)
            stats["sobel_ratio"].append(float(sobel_mag(gen01).mean() / (sobel_mag(src01).mean() + 1e-12)))
            stats["pairs"] += len(keep)

    n_pairs = stats["pairs"]
    resid = torch.tensor(resid_norms, dtype=torch.float64)
    grid_ssim = (stats["grid_ssim"] / n_pairs).cpu()
    grid_hf = (stats["grid_hf"] / n_pairs).cpu()
    worst_flat = torch.argsort(grid_ssim.flatten())[: max(1, int(0.1 * 256))]

    def _mean_std(vals):
        t = torch.tensor(vals, dtype=torch.float64)
        return {"mean": float(t.mean()), "std": float(t.std(unbiased=False))}

    def _band_table(key):
        return {b: _mean_std(stats[key][b]) for b in ("LL", "LH", "HL", "HH")}

    resid_energies = {b: float(torch.tensor(stats["resid_band"][b]).mean()) for b in ("LL", "LH", "HL", "HH")}
    tot = sum(resid_energies.values()) + 1e-12
    out = {
        "meta": {
            "checkpoint": args.checkpoint,
            "protocol": "paper / 256px / 8 steps / AdaIN 2.0 / B32",
            "pairs_cross": n_pairs,
            "note": "所有量只在 cross-style 对上统计（同风格对被排除）",
        },
        "subband_energy": {
            "content_z_c": _band_table("subband"),
            "fake_z_hat": _band_table("subband_fake"),
            "target_style": _band_table("subband_target"),
        },
        "subband_delta_fake_minus_content": {
            b: float(torch.tensor(stats["subband_fake"][b]).mean() - torch.tensor(stats["subband"][b]).mean())
            for b in ("LL", "LH", "HL", "HH")
        },
        "lab_drift": {k.replace("lab_", "delta_"): _mean_std(stats[k]) for k in ("lab_dL", "lab_da", "lab_db")},
        "lab_fraction_abs_delta_b_gt_3": stats["lab_abs_b_gt3"] / max(stats["lab_pixels"], 1),
        "ssim": _mean_std(stats["ssim"]),
        "sobel_edge_ratio_fake_over_content": _mean_std(stats["sobel_ratio"]),
        "local_grid_16x16": {
            "mean_ssim": grid_ssim.tolist(),
            "mean_hf_energy": grid_hf.tolist(),
            "worst_10pct_cells": [[int(i // 16), int(i % 16)] for i in worst_flat.tolist()],
            "worst_10pct_mean_ssim": float(grid_ssim.flatten()[worst_flat].mean()),
            "best_90pct_mean_ssim": float(grid_ssim.flatten()[torch.argsort(grid_ssim.flatten())[worst_flat.numel():]].mean()),
            "corr_ssim_vs_hf": float(torch.corrcoef(torch.stack([grid_ssim.flatten(), grid_hf.flatten()]))[0, 1]),
        },
        "residual": {
            "norm": {"mean": float(resid.mean()), "std": float(resid.std(unbiased=False)),
                     "p99": float(torch.quantile(resid, 0.99))},
            "band_energy": resid_energies,
            "band_share": {b: resid_energies[b] / tot for b in resid_energies},
            "spectral_centroid_normalized": stats["spec_num"] / max(stats["spec_den"], 1e-12),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    color_score = abs(out["lab_drift"]["delta_db"]["mean"]) + abs(out["lab_drift"]["delta_da"]["mean"]) \
        + abs(out["lab_drift"]["delta_dL"]["mean"]) + out["residual"]["band_share"]["LL"]
    struct_score = (1.0 - out["ssim"]["mean"]) + abs(1.0 - out["sobel_edge_ratio_fake_over_content"]["mean"]) \
        + out["residual"]["band_share"]["LH"] + out["residual"]["band_share"]["HL"] + out["residual"]["band_share"]["HH"]
    verdict = "整幅偏色（低频/LL 主导）" if color_score > struct_score else "局部结构改写（HF 子带与 SSIM 下降主导）"
    note_path = out_path.with_suffix(".NOTE.md")
    note_path.write_text(
        "# step_002500 内容损失拆解说明\n\n"
        f"- 样本：paper 协议 cross-style 对，共 {n_pairs} 对；同风格对排除。\n"
        "- 子带能量：对 4×32×32 latent 做 Haar DWT，各子带 `mean(x^2)`；delta 为 fake 减 content。\n"
        "- Lab：解码图与原图（sRGB→D65 Lab）逐像素差，报告通道均值/方差。\n"
        "- SSIM：11×11 Gaussian（σ=1.5）逐像素图后取均值；Sobel 比值为 fake/content 的边缘幅度均值比。\n"
        "- 网格：SSIM 图 adaptive-avg-pool 到 16×16，统计最差 10% 格子；HF 为原图减去 5×5 均值滤波后的 |.|。\n"
        "- 残差：`‖ẑ−z_c‖` 的 L2 分布（mean/std/p99）；频谱重心按 FFT 半径加权归一化到 [0,1]。\n"
        f"- 判读：color_score={color_score:.4f}，struct_score={struct_score:.4f} → **{verdict}**。\n",
        encoding="utf-8",
    )
    print(f"[analysis] wrote {out_path} and {note_path}")
    print(f"[analysis] pairs={n_pairs} ssim={out['ssim']['mean']:.4f} "
          f"lab_dL={out['lab_drift']['delta_delta_L']['mean']:+.3f} "
          f"lab_db={out['lab_drift']['delta_delta_b']['mean']:+.3f} "
          f"sobel_ratio={out['sobel_edge_ratio_fake_over_content']['mean']:.3f} "
          f"LL_share={out['residual']['band_share']['LL']:.3f} "
          f"verdict={verdict}")


if __name__ == "__main__":
    main()
