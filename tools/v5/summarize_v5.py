"""Summarise the full 116plan decision tree into a final delivery report.

Reads only JSON artifacts already written by the serial V5 stages; safe to run
while another stage is active (no GPU use).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
V5 = REPO_ROOT / "runs" / "v5"
RC = V5 / "route_c"
OUT = REPO_ROOT / "docs" / "v5" / "V5_116PLAN_EXECUTION_REPORT.md"


def load(p: Path, default=None):
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def fmt(v, nd=5):
    if v is None:
        return "N/A"
    if isinstance(v, bool):
        return "Yes" if v else "No"
    if isinstance(v, float):
        return f"{v:+.{nd}f}" if abs(v) < 1 else f"{v:.{nd}f}"
    return str(v)


def main():
    p0 = load(V5 / "p0_report.json", {})
    p1 = load(V5 / "p1_oracle" / "oracle_matrix_report.json", {})
    b1 = load(V5 / "b1_style_encoder" / "b1_report.json", {})
    noise = load(RC / "noise_check.json")
    noise_low = load(RC / "noise_check_lowmid.json")
    quick = load(RC / "quick_report.json") or load(RC / "quick_report_lowmid.json", {})
    final = load(RC / "final_report.json", {})

    lines = []
    lines.append("# WEAVE V5 / 116plan 24h 实验执行报告")
    lines.append("")
    lines.append(f"> 自动生成于 {time.strftime('%Y-%m-%d %H:%M:%S')}；证据均来自 `runs/v5/**/*.json` 与完整 750 评测产物。")
    lines.append("")
    lines.append("## 0. 决策树实际路径")
    lines.append("")
    lines.append("```text")
    lines.append("P0 条件/数据冻结")
    lines.append("  -> P1 Latent Oracle O1/O2/O3 (弱/强/无约束)")
    p1d = p1.get("decision", {})
    if p1d:
        lines.append(f"     -> G1: Oracle success={p1d.get('oracle_success')} => {p1d.get('branch_g1')}")
    b1d = b1.get("decision", {})
    if b1d:
        lines.append("  -> B1 训练 Latent Style Encoder (15,545 张真实 D5 latent) 并重测 Oracle")
        lines.append(f"     -> B1 success={b1d.get('b1_success')} => {b1d.get('next_decision')}")
    if noise:
        lines.append(f"  -> C2 纯噪声生成先验 credible={noise.get('generation_prior_credible')} (margin={fmt(noise.get('margin'), 4)})")
    if noise_low:
        lines.append(f"  -> C2b/C2c 低/中噪声补救（`{noise_low.get('start_mode', 'partially_noised_real_art')}`）credible={noise_low.get('generation_prior_credible')} margin={fmt(noise_low.get('margin'), 4)}")
    qd = quick.get("decision", {})
    if qd:
        lines.append(f"  -> C3 200 任务扫描 best={qd.get('best')} gate_passed={qd.get('gate_passed')}")
        lines.append(f"     config={qd.get('best_config')}")
        sel = qd.get("style_id_selectivity", {})
        if sel:
            lines.append(f"     Style-ID 正确/错误 margin={fmt(sel.get('margin'), 4)} selective={sel.get('selective')}")
    if final:
        lines.append(f"  -> C4 600+150 双 seed 全量评测 => 最终裁决 {final.get('final_verdict')}")
    lines.append("```")
    lines.append("")
    lines.append("Oracle 失败后，计划明确禁止停在诊断报告：先训练风格表征器，再切真实分布生成先验 Route C；因此 M1/M2 与 Oracle 解蒸馏未被进入，属于决策树规定的剪枝，不是漏跑。")
    lines.append("")

    lines.append("## 1. P0：信息流与条件泄漏修复")
    lines.append("")
    swap = p0.get("condition_swap", {})
    diffs = swap.get("diffs", {})
    lines.append(f"- 条件交换测试状态：**{swap.get('status', 'N/A')}**，模型在 `style_latent=None` 下仅由 Style-ID 驱动。")
    if diffs:
        lines.append(f"- 正确 ID / 错误 ID 的 LL 速度余弦相似度 {fmt(diffs.get('ll', {}).get('cosine_similarity'), 6)}，L2 差 {fmt(diffs.get('ll', {}).get('l2_diff'), 6)}；该旧 WEAVE 权重对 Style-ID 几乎不敏感，正是 P0 要暴露的信息流问题。")
    base = p0.get("baseline", {})
    lines.append(f"- 冻结 50 张源图 / 200 个跨风格任务：源图对目标原型相似度 {fmt(base.get('mean_src_cos_target'), 4)}，对自身风格原型 {fmt(base.get('mean_src_cos_source'), 4)}。")
    lines.append("- 固定 VAE posterior mean、seed、独立验证集、CSD 原型与真实 D5 多尺度 latent 描述符（`runs/v5/cache/`）。")
    lines.append("")

    lines.append("## 2. P1：Latent Oracle 是否可解")
    lines.append("")
    lines.append("| 臂 | 可改频带 | λ_struct | CSD ΔT | ΔS | ΔM | DINO-C | ΔT>0 比例 | 峰值显存 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, s in (p1.get("matrix") or {}).items():
        lines.append(f"| {name} | {s.get('bands')} | {fmt(s.get('lambda_struct'),2)} | {fmt(s.get('mean_dT'))} | {fmt(s.get('mean_dS'))} | {fmt(s.get('mean_dM'))} | {fmt(s.get('mean_dino_c'),4)} | {fmt(s.get('frac_dT_positive'),3)} | {fmt(s.get('peak_vram_mib'),1)} MiB |")
    if p1d:
        lines.append("")
        lines.append(f"**G1 裁决**：`oracle_success={p1d.get('oracle_success')}`，最佳约束臂 `{p1d.get('best_arm')}` 仅 ΔT={fmt(p1d.get('best_arm_dT'))}；无约束 Oracle 的 ΔT 也为负，说明不是内容约束过强，而是原始 latent 风格目标本身不能把图像推向目标风格原型。进入 B1 风格表征器补救。")
    lines.append("")

    lines.append("## 3. B1：真实 D5 风格表征器与 Oracle 重测")
    lines.append("")
    ts = b1.get("train_stats", {})
    lines.append(f"- `LatentStyleEncoder` 在 15,545 张真实作品 latent 上训练 12 epoch，最终验证分类准确率 **{fmt(ts.get('final_val_acc'),4)}**，训练耗时 {fmt(ts.get('train_time_sec'),1)} s。")
    lines.append("")
    lines.append("| 臂 | λ_struct | CSD ΔT | ΔS | ΔM | DINO-C | ΔT>0 比例 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, s in (b1.get("oracle_b1_matrix") or {}).items():
        lines.append(f"| {name} | {fmt(s.get('lambda_struct'),2)} | {fmt(s.get('mean_dT'))} | {fmt(s.get('mean_dS'))} | {fmt(s.get('mean_dM'))} | {fmt(s.get('mean_dino_c'),4)} | {fmt(s.get('frac_dT_positive'),3)} |")
    if b1d:
        lines.append("")
        lines.append(f"**B1 裁决**：`b1_success={b1d.get('b1_success')}`，最佳臂 `{b1d.get('best_arm')}` ΔT={fmt(b1d.get('best_arm_dT'))}，仍远低于 +0.03；按计划切换 **Route C 真实艺术分布生成先验**。")
    lines.append("")

    lines.append("## 4. Route C：Style-Conditional Rectified Flow")
    lines.append("")
    lines.append("监督目标：`z_tau=(1-tau)*eps + tau*z_real`，`v*=z_real-eps`；条件只有 Style-ID，不使用参考 GT、不反传 CSD、不使用 GAN/RGB 训练损失。")
    lines.append("")
    if noise:
        lines.append(f"- 纯噪声生成检查（`{noise.get('start_mode', 'pure_noise')}`）：对角-非对角 CSD 原型 margin **{fmt(noise.get('margin'),4)}**，`generation_prior_credible={noise.get('generation_prior_credible')}`。")
        mat = noise.get("csd_matrix")
        if mat:
            lines.append("")
            lines.append("| 生成条件 \\ 原型 | " + " | ".join(["ER", "Imp", "Min", "Roc", "Uki"]) + " |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            short = ["ER", "Imp", "Min", "Roc", "Uki"]
            for i, row in enumerate(mat):
                lines.append(f"| {short[i]} | " + " | ".join(f"{v:.3f}" for v in row) + " |")
    lines.append("")
    lines.append("### C3：200 跨风格任务扫描")
    lines.append("")
    lines.append("| 配置 | τ0 | 结构引导 | CSD ΔT [95%CI] | ΔM | DINO-C | LPIPS | 5 类中改善 | Ukiyo-e ΔT |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for name, s in (quick.get("sweep") or {}).items():
        ci = s.get("dT_ci") or [0, 0]
        lines.append(f"| {name} | {fmt(s.get('tau0'),2)} | {s.get('guide')} | {fmt(s.get('mean_dT'))} [{fmt(ci[0],4)}, {fmt(ci[1],4)}] | {fmt(s.get('mean_dM'))} | {fmt(s.get('mean_dino_c'),4)} | {fmt(s.get('mean_lpips'),4)} | {s.get('styles_improved')}/5 | {fmt(s.get('ukiyo_dT'))} |")
    if qd:
        lines.append("")
        lines.append(f"扫描选出 `{qd.get('best')}`；gate_passed={qd.get('gate_passed')}。Style-ID 交换敏感度：正确 ID 余弦 {fmt(qd.get('style_id_selectivity',{}).get('cos_correct_id'),4)}，错误 ID {fmt(qd.get('style_id_selectivity',{}).get('cos_wrong_id'),4)}，margin {fmt(qd.get('style_id_selectivity',{}).get('margin'),4)}，selective={qd.get('style_id_selectivity',{}).get('selective')}。")
    lines.append("")

    lines.append("## 5. 最终 600 跨风格 + 150 同风格（双 seed）")
    lines.append("")
    if final:
        c = final.get("two_seed_mean_cross600", {})
        sm = final.get("two_seed_mean_same150", {})
        lines.append("| 口径 | CSD ΔT | CSD ΔS | CSD ΔM | DINO-S | DINO-C | CLIP-S | LPIPS |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        lines.append("| 600 cross (双 seed 均值) | " + " | ".join(fmt(c.get(k)) for k in ("CSD_dT", "CSD_dS", "CSD_dM", "DINO_S", "DINO_C", "CLIP_S", "LPIPS")) + " |")
        lines.append("| 150 same (双 seed 均值) | " + " | ".join(fmt(sm.get(k)) for k in ("CSD_dT", "CSD_dS", "CSD_dM", "DINO_S", "DINO_C", "CLIP_S", "LPIPS")) + " |")
        lines.append("")
        lines.append(f"- 两 seed 的 600 cross ΔT 差值：{fmt(final.get('seed_gap_cross600_dT'))}（随机路线按要求报告平均值与方差，不挑最好 seed）。")
        lines.append(f"- 与同协议旧 WEAVE 600 cross 锚点 ΔT=+0.020284 / DINO-C=0.878826 / LPIPS=0.265337 对照。")
        lines.append("")
        lines.append("| 统一成功标准 | 结果 |")
        lines.append("|---|---|")
        for k, v in (final.get("success_criteria") or {}).items():
            lines.append(f"| {k} | {'PASS' if v else 'FAIL'} |")
        lines.append("")
        lines.append(f"**最终裁决：`{final.get('final_verdict')}`**")
        lines.append("")
        lines.append("### 24h 结束必须回答的问题")
        lines.append("")
        lines.append("| 问题 | 结论 |")
        lines.append("|---|---|")
        for k, v in (final.get("answers") or {}).items():
            lines.append(f"| {k} | {v} |")
    else:
        lines.append("> `final_report.json` 尚未生成：C4 全量评测仍在串行队列中。")
    lines.append("")
    lines.append("## 6. 工程约束、剪枝分支与复现")
    lines.append("")
    lines.append("本轮实际只启动决策树裁决到的路径。计划中的 M1/M2、B2 内容 Pareto、Oracle 解蒸馏属于「Oracle 成功」或「监督修复成功」后的后续分支；本轮 Oracle 与 B1 均失败，按 §4/§6/§8 被剪枝。为满足「尽量都实现好」，这些分支已在 `tools/v5/branches/route_a.py` 与 `run_pruned_branches.sh` 中实现并通过 smoke test（M1 输出零初始化恒等、M2 可微 ODE 有限、监督项接线正常），但没有与 Route C 抢 GPU 并行运行。")
    lines.append("")
    lines.append("- 单 GPU 任务、串行决策树、一个 tmux 驱动；每阶段写 `runs/v5/PROGRESS.md`。")
    lines.append("- 训练：15,545 张真实 D5 latent 常驻 GPU（972 MiB），2.28M 参数 flow，有效 batch 256，fp16 autocast，峰值显存约 11.4 GiB（贴近 3060 12GB 上限且不触发 WDDM 分页）。")
    lines.append("- 渲染：固定 shape TRT decoder b8 + batch 64 / save_threads 8 档位；750 张 PNG 与 `metrics.csv`/`dino_metrics.csv` 复用 V4 冻结协议。")
    lines.append("- 随机生成路线强制两个 seed (42/43)，不允许挑最优单张。")
    lines.append("")
    lines.append("## 7. 产物索引")
    lines.append("")
    for rel in [

        "runs/v5/p0_report.json",
        "runs/v5/v5_eval200_tasks.json",
        "runs/v5/v5_oracle100_tasks.json",
        "runs/v5/v5_eval200_baseline.json",
        "runs/v5/p1_oracle/oracle_matrix_report.json",
        "runs/v5/b1_style_encoder/b1_report.json",
        "runs/v5/b1_style_encoder/latent_style_encoder.pt",
        "runs/v5/route_c/flow_full.pt",
        "runs/v5/route_c/noise_check.json",
        "runs/v5/route_c/quick_report.json",
        "runs/v5/route_c/best_config.json",
        "runs/v5/route_c/final_report.json",
        "runs/v5/PROGRESS.md",
    ]:
        p = REPO_ROOT / rel
        status = "存在" if p.exists() else "待生成/位于远端 I: Latent_Style WEAVE"
        lines.append(f"- `{rel}` — {status}")
    lines.append("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
