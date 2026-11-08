"""WEAVE V7 P1: train a pure denoising-MSE conditional ScoreNet (S1).

This is a controlled ablation against the original V6 ScoreNet:
- identical architecture (ConditionalScoreNet base_dim=64)
- identical training data / crop schedule / sigma schedule / steps / batch
- only difference: margin_weight=0, removing the wrong-class margin loss.

Outputs per-style denoising MSE, per-style wrong-vs-correct gap, and Score-direction
diagnostics on content-matched validation latents, especially for Ukiyo-e.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import paths  # noqa: E402
from tools.v6.canonicalize import build_content_matched_subset  # noqa: E402
from tools.v6.score_diff import train_score_model  # noqa: E402
from tools.v6.train_eval_v6 import build_indomain_style_targets, boot_ci  # noqa: E402
from tools.v7.common import (
    DOCS_V7,
    STYLES,
    V7_ROOT,
    ensure_v7_dirs,
    load_score_model,
    save_json,
)


def evaluate_score_quality(model, val_by_style, X_matched, Y_matched, device: str) -> dict:
    model.eval()
    sigmas = [0.15, 0.30, 0.50, 0.80]
    per_style_mse = {}
    per_style_gap = {}
    per_style_margin = {}
    sigma_mse = {}

    for sigma in sigmas:
        eps_all, pred_all = [], []
        with torch.inference_mode():
            torch.manual_seed(42)
            for sid, s in enumerate(STYLES):
                x = val_by_style[s][:40].to(device).float()
                eps = torch.randn_like(x)
                y = torch.full((x.shape[0],), sid, device=device, dtype=torch.long)
                y_sig = x + sigma * eps
                pred = model(y_sig, sigma, y)
                err = (pred.float() - eps).pow(2).mean(dim=(1, 2, 3))
                eps_all.append(err)
                pred_all.append(pred)
                if sid not in per_style_mse:
                    per_style_mse[s] = []
                    per_style_gap[s] = []
                    per_style_margin[s] = []
                per_style_mse[s].append(float(err.mean().item()))

                # Wrong-style predictions on the SAME noisy latent: measures score separation.
                wrong_errs = []
                for shift in range(1, len(STYLES)):
                    y_wrong = torch.full_like(y, (sid + shift) % len(STYLES))
                    pred_w = model(y_sig, sigma, y_wrong)
                    wrong_errs.append((pred_w.float() - eps).pow(2).mean(dim=(1, 2, 3)))
                wrong = torch.stack(wrong_errs, dim=0).mean(dim=0)
                gap = (wrong - err).cpu().numpy()
                per_style_gap[s].append(float(gap.mean()))
                per_style_margin[s].append(float((gap > 0).mean()))

        sigma_mse[f"{sigma:.2f}"] = float(torch.cat(eps_all).mean().item())

    # Directional cosine of g_{s -> t} at sigma=0.30 on content-matched subset
    X = X_matched.to(device).float()
    Y = Y_matched.to(device).long()
    with torch.inference_mode():
        torch.manual_seed(123)
        eps = torch.randn_like(X)
        z_sig = X + 0.30 * eps
        preds = {}
        for sid in range(len(STYLES)):
            y = torch.full((X.shape[0],), sid, device=device, dtype=torch.long)
            preds[sid] = model(z_sig, 0.30, y)
        # For each sample, compute direction from its true source style to Ukiyo-e and
        # compare with directions toward other targets.
        ukiyo = len(STYLES) - 1
        dir_u, dir_other, cos_u_other = [], [], []
        for sid in range(len(STYLES) - 1):
            mask = Y == sid
            if mask.sum() == 0:
                continue
            g_u = ((preds[sid] - preds[ukiyo]) / 0.30)[mask].flatten(1)
            g_u = torch.nn.functional.normalize(g_u, dim=-1)
            dir_u.append(g_u)
            for tid in range(len(STYLES)):
                if tid == sid or tid == ukiyo:
                    continue
                g_o = ((preds[sid] - preds[tid]) / 0.30)[mask].flatten(1)
                g_o = torch.nn.functional.normalize(g_o, dim=-1)
                dir_other.append(g_o)
                cos_u_other.append(float((g_u * g_o).sum(dim=-1).mean().item()))
        mean_cos = float(np.mean(cos_u_other)) if cos_u_other else 0.0

    return {
        "per_style_denoise_mse": {s: round(float(np.mean(v)), 5) for s, v in per_style_mse.items()},
        "per_style_wrong_minus_correct": {s: round(float(np.mean(v)), 5) for s, v in per_style_gap.items()},
        "per_style_frac_correct_smaller": {s: round(float(np.mean(v)), 4) for s, v in per_style_margin.items()},
        "sigma_mse": {k: round(v, 5) for k, v in sigma_mse.items()},
        "ukiyo_direction_mean_cos_vs_other_targets": round(mean_cos, 4),
        "ukiyo_denoise_mse": round(float(np.mean(per_style_mse["Ukiyo_e"])), 5),
        "ukiyo_wrong_minus_correct": round(float(np.mean(per_style_gap["Ukiyo_e"])), 5),
    }


def main() -> None:
    ensure_v7_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== WEAVE V7 P1 Pure-MSE ScoreNet (S1) (Device: {device}) ===", flush=True)

    asset_root = paths.asset_root()
    orig_dir = asset_root / "pseudo_gt_24h" / "orig_mean_fp16"
    latents_by_style, train_by_style, val_by_style = {}, {}, {}
    for sid, s in enumerate(STYLES):
        d = torch.load(orig_dir / f"{sid:02d}_{s}.pt", map_location="cpu", weights_only=False)
        x = d["latents"].float()
        latents_by_style[s] = x
        val_by_style[s] = x[:100]
        train_by_style[s] = x[100:]

    style_targets = build_indomain_style_targets(train_by_style, asset_root, device=device, top_k=64)
    X_matched, Y_matched = build_content_matched_subset(latents_by_style, asset_root, per_style=120)

    print("[S1] Training pure denoising-MSE conditional ScoreNet (margin_weight=0)...", flush=True)
    model, train_stats = train_score_model(
        train_by_style,
        style_targets,
        base_dim=64,
        steps=1000,
        microbatch=100,
        accum=2,
        lr=3e-4,
        seed=42,
        margin_weight=0.0,
        device=device,
    )

    out_ckpt = V7_ROOT / "scorenet_s1_pure_mse.pt"
    torch.save({"model": model.state_dict(), "train_stats": train_stats}, out_ckpt)
    print(f"[S1] Saved checkpoint to {out_ckpt}", flush=True)

    # Compare quality of old vs new ScoreNet under identical held-out noise.
    s1_quality = evaluate_score_quality(model, val_by_style, X_matched, Y_matched, device)
    old_model = load_score_model(V7_ROOT.parent / "v6" / "score_diff" / "score_diff_net.pt", device=device)
    old_quality = evaluate_score_quality(old_model, val_by_style, X_matched, Y_matched, device)

    out = {
        "protocol": {
            "S1": "pure denoising MSE, margin_weight=0",
            "V6": "original denoising MSE + 0.5 * wrong-class margin on even steps",
            "data": "identical pseudo_gt_24h/orig_mean_fp16 train split, 1000 steps, base_dim=64",
        },
        "S1_train_stats": train_stats,
        "S1_quality": s1_quality,
        "V6_original_quality": old_quality,
    }
    out_json = V7_ROOT / "p1_pure_mse_scorenet.json"
    save_json(out, out_json)

    lines = []
    lines.append("# WEAVE V7 P1：纯去噪 MSE ScoreNet (S1) 对照报告\n")
    lines.append("> 目标：验证原 V6 ScoreNet 中的错误类别 margin 损失是否扭曲了真实条件 Score 估计，")
    lines.append("> 尤其关注 Ukiyo-e 等弱类的去噪误差、Score 方向可分辨性与噪声尺度敏感性。\n")
    lines.append("## 一、总体去噪 MSE（越低越符合标准条件 Score 最优解）\n")
    lines.append("| $\\sigma$ | 原 V6 (MSE + margin) | S1 (纯 MSE) | 差值 (V6 - S1) |")
    lines.append("|---:|---:|---:|---:|")
    for sig in ["0.15", "0.30", "0.50", "0.80"]:
        a = old_quality["sigma_mse"][sig]
        b = s1_quality["sigma_mse"][sig]
        lines.append(f"| {sig} | {a:.5f} | **{b:.5f}** | {a-b:+.5f} |")

    lines.append("\n## 二、五类风格各自去噪误差与条件可分辨性\n")
    lines.append("| 风格 | 原 V6 MSE | S1 MSE | 原 V6 wrong-correct | S1 wrong-correct | S1 正确率 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for s in STYLES:
        lines.append(f"| **{s}** | {old_quality['per_style_denoise_mse'][s]:.5f} | **{s1_quality['per_style_denoise_mse'][s]:.5f}** | "
                     f"{old_quality['per_style_wrong_minus_correct'][s]:+.5f} | {s1_quality['per_style_wrong_minus_correct'][s]:+.5f} | "
                     f"{s1_quality['per_style_frac_correct_smaller'][s]*100:.1f}% |")

    lines.append("\n## 三、Ukiyo-e 条件 Score 专项诊断\n")
    lines.append(f"- 原 V6 Ukiyo-e 去噪 MSE：**{old_quality['ukiyo_denoise_mse']:.5f}**；S1：**{s1_quality['ukiyo_denoise_mse']:.5f}**。\n")
    lines.append(f"- 原 V6 Ukiyo-e wrong-correct gap：**{old_quality['ukiyo_wrong_minus_correct']:+.5f}**；S1：**{s1_quality['ukiyo_wrong_minus_correct']:+.5f}**。\n")
    lines.append(f"- Ukiyo-e 目标方向与其他目标方向平均余弦：原 V6 **{old_quality['ukiyo_direction_mean_cos_vs_other_targets']:.4f}**，S1 **{s1_quality['ukiyo_direction_mean_cos_vs_other_targets']:.4f}**（越低越不易与其他画风混淆）。\n")

    lines.append("\n## 四、结论\n")
    lines.append("S1 使用完全标准的高斯去噪目标，具有明确的统计最优解。若其 Ukiyo-e / 弱类去噪误差与方向分离度明显改善，")
    lines.append("则 RDMD 的真实 Score 监督器应优先采用 S1；若改善有限，则 Ukiyo-e 弱类瓶颈更可能来自数据分布或 CSD 原型偏置，")
    lines.append("而非 margin 损失本身。\n")

    out_md = DOCS_V7 / "P1_PURE_MSE_SCORENET_REPORT.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[V7 P1] Saved {out_json} and {out_md}", flush=True)


if __name__ == "__main__":
    main()
