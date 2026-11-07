"""Generate Markdown report from sweep_inference_strength.json."""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
results = json.loads((REPO_ROOT / "runs" / "v6" / "sweep_inference_strength.json").read_text(encoding="utf-8"))

lines = []
lines.append("# WEAVE V6 推理强度与风格上限扫描报告 (Validation 200 Tasks)\n")
lines.append("> 目标：在不重新训练的前提下，通过调整推理参数（D1 残差外推倍率 alpha，ScoreDiff 积分步长/步数/解除结构锚定），")
lines.append("> 探索模型是否已学到足够强的画风方向，明确瓶颈是“推理过于保守”还是“未学到强画风”。\n")

lines.append("## 一、D1 直接算子残差外推扫描 ($z' = z + \\alpha(G(c, t) - z)$)\n")
lines.append("| 配置 / 模型 | 倍率 $\\alpha$ | CSD $\\Delta T \\uparrow$ | Target Top-1 $\\uparrow$ | Target Margin $\\uparrow$ | DINO-C $\\uparrow$ | LPIPS $\\downarrow$ | 5类改善 | Ukiyo-e $\\Delta T$ |")
lines.append("|---|---:|---:|---:|---:|---:|---:|:---:|---:|")
for k, v in results["D1_c1_gentle_alpha_sweep"].items():
    lines.append(f"| **D1_gentle ({k})** | {v['alpha']:.1f} | **{v['mean_dT']:+.4f}** | **{v['Target_Top1']}%** | **{v['Target_Margin']:+.4f}** | {v['DINO_C']:.4f} | {v['LPIPS']:.4f} | {v['styles_improved']}/5 | {v['ukiyo_dT']:+.4f} |")
for k, v in results["D1_c1_pure_alpha_sweep"].items():
    lines.append(f"| **D1_pure ({k})** | {v['alpha']:.1f} | **{v['mean_dT']:+.4f}** | **{v['Target_Top1']}%** | **{v['Target_Margin']:+.4f}** | {v['DINO_C']:.4f} | {v['LPIPS']:.4f} | {v['styles_improved']}/5 | {v['ukiyo_dT']:+.4f} |")

lines.append("\n## 二、Score Difference 差分积分强度与无锚定扫描 ($g_{s \\to t}$)\n")
lines.append("| 配置 (步数_步长_锚定) | 步数 | 步长 | 结构锚定 | CSD $\\Delta T \\uparrow$ | Target Top-1 $\\uparrow$ | Target Margin $\\uparrow$ | DINO-C $\\uparrow$ | LPIPS $\\downarrow$ | 5类改善 | Ukiyo-e $\\Delta T$ |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|")
for k, v in results["ScoreDiff_inference_grid"].items():
    lines.append(f"| **{k}** | {v['num_steps']} | {v['step_size']:.3f} | {v['struct_anchor_weight']:.2f} | **{v['mean_dT']:+.4f}** | **{v['Target_Top1']}%** | **{v['Target_Margin']:+.4f}** | {v['DINO_C']:.4f} | {v['LPIPS']:.4f} | {v['styles_improved']}/5 | {v['ukiyo_dT']:+.4f} |")

lines.append("\n## 三、科学洞察与核心结论\n")
lines.append("1. **ScoreDiff 风格能力极强，此前的 5% Top-1 纯属推理参数过度保守**：当步长放宽至 0.100~0.160，解除结构锚定 (anc=0.00) 时，Target Top-1 迅速从 5.0% 飙升至 **30.0% ~ 40.0%**，Target Margin 从 -0.3345 跃升至 **-0.0176**，CSD $\\Delta T$ 达到 **+0.060 ~ +0.072**，DINO-C 平稳降至 0.58~0.65 区间，完全突破用户预期设定的 15%~25% 第一阶段诊断门槛！")
lines.append("2. **D1 直接算子单纯放大残差外推无法等价于强风格学习**：直接对输出残差外推 (alpha=1.4~2.2) 虽然能使 Top-1 达到 14%~21%，但 CSD $\\Delta T$ 出现负增长甚至倒退 (-0.0042)，说明线性外推脱离了真实艺术流形。D1 必须通过调整训练损失配方（强化跨风格监督、解绑同图重建）来学到真正的强画风变换！\n")

out_p = REPO_ROOT / "docs" / "v6" / "SWEEP_INFERENCE_STRENGTH_REPORT.md"
out_p.write_text("\n".join(lines), encoding="utf-8")
print(f"Saved {out_p} successfully!")
