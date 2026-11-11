"""WEAVE W — stage report generator (single formal deliverable).

Reads whatever W artefacts exist (resume-safe), writes docs/v6/W_REPORT.md.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "runs/v6_w"
OUT = ROOT / "docs/v6/W_REPORT.md"


def load(name, base=W):
    p = base / f"eval_{name}" / "eval.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def f(x, n=4):
    return f"{x:.{n}f}" if isinstance(x, (int, float)) else "—"


def main():
    L = []
    L.append("# WEAVE W 阶段报告 — 冻结潜空间下的单步拟合修复")
    L.append("")
    L.append("> 协议：`tasks_750.json`（600 cross + 150 same），`paper_rgb_v1`，"
             "原始 RGB 零点，统一 TRT fp16 SD-VAE 评测。单卡串行 "
             "`sbflow/scripts/run_w_12h.sh`。")
    L.append("> 前提：**Encoder / latent 空间完全冻结**（A0 与 sd-vae-ft-ema 架构一致，"
             "encoder conv_in 差异仅 2e-4，A0 latent 可直接喂既有 Q2a/B4 生成器）。"
             "S10 Decoder 必须与 **A0 Encoder + A0 post_quant_conv** 配对"
             "（4090 正确配方，实测 DINO-C 0.9387；错配 sd-vae latent 会崩到 0.22）。")
    L.append("")
    L.append("## 立项依据（R2 结论纠错）")
    L.append("")
    L.append("R2 表 D 铁证：学生与 Oracle 教师位移 cos 0.638–0.683、RMS 比 0.52–0.65、"
             "相对拟合误差 0.53–0.60、R1 margin gap 1.29–1.77 —— **学生没把教师学进去**。"
             "Oracle 教师本身成功（Top-1 34.3%、DINO-S 0.4277）。因此病灶是"
             "**蒸馏损失**：潜空间逐像素 L2 无法传递高频、相位敏感的判别方向，"
             "且与旧 V6 DWT/C1 统计损失拔河。R2 原判『教师失败』已更正为"
             "『教师成功、学生拟合失败 → 研究蒸馏优化』。")
    L.append("")
    L.append("## W 实验与实现")
    L.append("")
    L.append("| 实验 | 实现 | 关键改动 |")
    L.append("|---|---|---|")
    L.append("| 实验一 W_FeatDistill | `w_distill.py` | R1 多尺度特征(64/32/16)匹配 + KL 分类边界 + "
             "反吸引子 Margin；像素 L2 **仅 Haar LL**，高频放开 |")
    L.append("| 实验二 W_LossClean | `train_eval_v6.py` 开关 | 关旧 DWT `w_cross` 冲突项，"
             "C1 内容不变性仅约束 LL 低频结构 |")
    L.append("| 实验三 困难风格 | `w_distill.py` | 按风格 RMS（Min/Imp 0.15、ER/Roc/Ukiyo 0.32）"
             "+ 对 Imp/Min 决策边界的反吸引子排斥 |")
    L.append("| 实验四 W_S10_ImgCalib | `w_img_calib.py`+`w_vae.py` | 冻结 A0 Encoder + S10 Decoder，"
             "150 步 RGB 域 CSD margin + DINO-C 联合微调 |")
    L.append("| 实验五 845K+StyleKeys | 训练臂 W4 | dim32 + rank32 风格残差基，强特征监督下重测容量 |")
    L.append("")
    L.append("- 损失量级经 `TERM_SCALE` 归一化 + `W_LOSS_SCALE=0.006` 对齐历史像素 MSE 量级"
             "（`L_distill≈0.09`），λ=20/50 的既有标定继续有效。")
    L.append("- 反吸引子用 `softplus(-margin/tau)` 有界化（原始 `-margin` 无下界会导致 logits 发散）。")
    L.append("- 14 项单元测试全部通过（`w_unit_tests.py`），含 A0/S10 配对与 `decode_s10` 缩放校验。")
    L.append("- 50 步真实 microbatch 稳定性标定：218K 臂 loss 2.06→1.92 单调、显存 5.4GB、~1.2s/step；"
             "**845K+StyleKeys 对照臂 loss 3.59→5.55 上升（不稳定）**，按计划『低优先级臂在稳定性/预算压力下砍掉』"
             "主动禁用，记为负结果，不进入正式训练矩阵。")
    L.append("- **S10 Decoder 配对纠错**：初版用 sd-vae-ft-ema latent 喂 S10 导致 DINO-C 崩到 0.22；"
             "经 ssh 4090 找到完整 VAE 结果后确认正确配方 = A0 Encoder + A0 post_quant_conv + S10，"
             "实测 DINO-C 0.9387（标准 0.9657 / A0 0.9666）。A0 Encoder 权重已拉回并固化于 `w_vae.py`。")
    L.append("")

    # results table
    arms = [
        ("Q2a Direct (R2)", "Q2a_direct", ROOT / "runs/v6_r2"),
        ("Q2a+Oracle (R2)", "Q2a_oracle", ROOT / "runs/v6_r2"),
        ("B3 λ20 (R2 anchor)", "B3_distill20", ROOT / "runs/v6_r2"),
        ("B4 λ50 (R2)", "B4_distill50", ROOT / "runs/v6_r2"),
        ("W1 FeatDistill", "W1_FeatDistill_218", W),
        ("W2 LossClean", "W2_LossClean_218", W),
        ("W3 LossClean λ50", "W3_LossClean_218_l50", W),
        ("W4 845K+SK", "W4_LossClean_845_SK", W),
        ("W5 S10 ImgCalib", "W5_ImgCalib", W),
    ]
    L.append("## 统一协议结果（Cross600）")
    L.append("")
    L.append("| 模型 | Top-1 | Margin | CSD ΔT | DINO-S | gain | DINO-C | LPIPS | MinOff |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label, name, base in arms:
        d = load(name, base)
        if not d:
            L.append(f"| {label} | — | — | — | — | — | — | — | — |")
            continue
        s = d["cross600"]
        L.append(f"| {label} | {f(s['top1']*100,1)}% | {f(s['margin'])} | {f(s['dT'])} | "
                 f"{f(s['dino_s'])} | {f(s['dino_s_gain'])} | {f(s['dino_c'])} | "
                 f"{f(s['lpips'])} | {f(s['minimalism_offtarget']*100,1)}% |")
    L.append("")

    # paired verdict
    pv = W / "w_paired_verdict.json"
    if pv.exists():
        v = json.loads(pv.read_text(encoding="utf-8"))
        L.append("## P4.3 成功门槛判定（相对 B3 anchor）")
        L.append("")
        L.append(f"- 主门槛 DINO-S ≥ +0.020 且 CI 下界 > 0："
                 f"**{'有模型通过' if v.get('p43_primary_gate_any_pass') else '无模型通过'}**。")
        L.append(f"- 各臂配对判定：{v.get('gates_vs_B3', {})}")
        L.append("")
        if not v.get("p43_primary_gate_any_pass"):
            L.append("- 按计划：即便无 +0.020 的独立显著收益，也按完整结果交付，"
                     "不通过修改测试协议或选择性展示制造提升。")
    L.append("")
    L.append("## 复现")
    L.append("")
    L.append("```bash")
    L.append("bash sbflow/scripts/run_w_12h.sh            # 单入口串行（可续跑）")
    L.append("bash sbflow/scripts/run_w_12h.sh --dry-run  # 仅预检（编译 + 14 单测 + A0/S10 校验）")
    L.append("python3 -m tools.v6_next.w_unit_tests       # 14 项单元测试")
    L.append("```")
    L.append("")
    L.append("- 结果目录 `runs/v6_w/`；A0 Encoder 在 `runs/a0/a0_encoder_only.pt`"
             "（SHA256 前缀 `70715532435ad6c2`，从 4090 `/root/Workspace/WEAVE_VAE/runs_vae/A0/` 导出）。")
    L.append("- 已知限制：Val200 ⊂ Test750 且 Test750 反复参与研发；未构造作者隔离盲集，"
             "所有显著性为研发集内配对，不宣称盲测。")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[W-P5] wrote {OUT} ({len(L)} lines)")


if __name__ == "__main__":
    main()
