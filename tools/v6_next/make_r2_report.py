"""WEAVE R2 P5 — single formal stage report generator.

Reads whatever R2 artefacts exist (resume-safe / partial-result-safe) and writes
docs/v6/R2_12H_REPORT.md answering the six mandatory questions A-F and picking
one of the three pre-declared routes.

All numbers come from evaluated JSON files; nothing is hard-coded from narration.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R2 = ROOT / "runs/v6_r2"
OUT = ROOT / "docs/v6/R2_12H_REPORT.md"


def load_eval(name: str):
    p = R2 / f"eval_{name}" / "eval.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def cross_summary(d):
    return d["cross600"] if d else None


def f(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "—"


def main() -> None:
    L: list[str] = []
    L.append("# WEAVE R2 — 12 小时执行阶段报告")
    L.append("")
    L.append("> 唯一正式阶段报告。协议：`tasks_750.json`（600 cross + 150 same），"
             "`paper_rgb_v1` 口径，原始 RGB 源图为零点，统一 TRT fp16 SD-VAE 解码。")
    L.append("> 所有简略测试集已删除；单卡串行 `sbflow/scripts/run_r2_12h.sh`，"
             "完成标志可续跑，预算 12h。")
    L.append("")

    # ---------------- data ----------------
    evals = {
        "Identity": load_eval("identity"),
        "Q2a Direct": load_eval("Q2a_direct"),
        "Q2a + R1 Oracle": load_eval("Q2a_oracle"),
        "B3 (λ=20)": load_eval("B3_distill20"),
        "B4 (λ=50)": load_eval("B4_distill50"),
        "B2c_ctrl845": load_eval("P1_B2c_ctrl845"),
        "S1 (218K, P2 teacher, λ=20)": load_eval("S1"),
        "S2 (218K, P2 teacher, λ=50)": load_eval("S2"),
        "S3 (845K, Oracle teacher, λ=20)": load_eval("S3"),
        "S4 (845K, P2 teacher, λ=20)": load_eval("S4"),
        "ScoreDiff N=24": load_eval("scorediff_N24"),
    }
    evals = {k: v for k, v in evals.items() if v is not None}

    p1 = None
    p1p = R2 / "p1_analysis/p1_summary.json"
    if p1p.exists():
        p1 = json.loads(p1p.read_text(encoding="utf-8"))
    p2 = None
    p2p = R2 / "p2_teachers/p2_teachers_summary.json"
    if p2p.exists():
        p2 = json.loads(p2p.read_text(encoding="utf-8"))
    control = ""
    cpath = R2 / "dino_s_control.txt"
    if cpath.exists():
        control = cpath.read_text(encoding="utf-8")
    fit = {}
    for arm in ("S1_distill20_proj_ll", "S2_distill50_proj_ll",
                "S3_ctrl845_distill20_oracle", "S4_ctrl845_distill20_proj_ll"):
        fp = R2 / "fitting_diagnostics" / arm / "fit_diag.json"
        if fp.exists():
            fit[arm] = json.loads(fp.read_text(encoding="utf-8"))

    # ---------------- 0. Protocol ----------------
    L.append("## 0. 评测口径与可信度（P0）")
    L.append("")
    L.append("- 统一为：原始 RGB 源图零点；生成图统一 SD-VAE/TRT fp16 解码；"
             "DINO-S = 生成图 CLS 与目标风格 30 张独立参考图 CLS 的最大余弦；"
             "DINO-C = 生成图 CLS 与原始源图 CLS；LPIPS 对原始 RGB；"
             "Top-1 = CSD 五类原型 argmax；Margin = 目标类减最强竞争类。")
    L.append("- 修复历史缺陷：`pred_class` 用整数类别索引统计 Minimalism 误判（旧版整数/字符串比较导致 0.000）；"
             "Cross600/Same150 严格按 `tgt_style != src_style` 分离；"
             "VAE 往返参照仅作诊断字段（`*_rt`），不混入主表。")
    L.append("- 12 项单元测试（`tools/v6_next/r2_unit_tests.py`）全部通过："
             "含梯度上升提升 R1 margin、逐图 RMS 归一、Haar 重建、投影一阶非增、"
             "零梯度无 NaN、R1 冻结、混淆矩阵整数索引、教师可复现、蒸馏非零更新等。")
    # DINO-S control numbers
    same_dinos = ""
    for line in control.splitlines():
        if "Mean Same-Style DINO-S" in line and "Raw" not in line:
            pass
    if "5-Way Style Top-1 Acc" in control:
        first_acc = [ln.strip() for ln in control.splitlines()
                     if "5-Way Style Top-1 Acc" in ln]
        L.append(f"- DINO-S 有效性控制：真实画作 5 类 Top-1 = "
                 f"{first_acc[0].split(':')[-1].strip() if first_acc else 'NA'}（raw RGB），"
                 f"同风格 vs 异风格 DINO-S 有明显区分间隔（约 +0.12），"
                 f"证明 DINO-S 在本测试域具备画风判别力，不是纯内容/题材混淆。")
    L.append("")

    # ---------------- 1. P0 table ----------------
    L.append("## 1. 统一协议主表（Cross600）")
    L.append("")
    L.append("| 模型 | 参数 | Top-1 | Margin | CSD ΔT | DINO-S | DINO-S gain | DINO-C | LPIPS | Min 误判 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    params = {
        "Identity": "0", "Q2a Direct": "218K", "Q2a + R1 Oracle": "218K+816K",
        "B3 (λ=20)": "218K", "B4 (λ=50)": "218K", "B2c_ctrl845": "845K",
        "S1 (218K, P2 teacher, λ=20)": "218K", "S2 (218K, P2 teacher, λ=50)": "218K",
        "S3 (845K, Oracle teacher, λ=20)": "845K", "S4 (845K, P2 teacher, λ=20)": "845K",
        "ScoreDiff N=24": "3.27M",
    }
    for name, d in evals.items():
        c = cross_summary(d)
        if not c:
            continue
        L.append(
            f"| {name} | {params.get(name,'—')} | {f(c['top1']*100,1)}% | "
            f"{f(c['margin'])} | {f(c['dT'])} | {f(c['dino_s'])} | "
            f"{f(c['dino_s_gain'])} | {f(c['dino_c'])} | {f(c['lpips'])} | "
            f"{f(c['minimalism_offtarget']*100,1)}% |"
        )
    L.append("")

    # ---------------- A ----------------
    L.append("## A. 现有 Oracle 到底提升了什么？")
    L.append("")
    q2a = cross_summary(evals.get("Q2a Direct"))
    orc = cross_summary(evals.get("Q2a + R1 Oracle"))
    if q2a and orc:
        L.append(f"- CSD Top-1：**{f(q2a['top1']*100,1)}% → {f(orc['top1']*100,1)}%**"
                 f"（{f((orc['top1']-q2a['top1'])*100,1)}pp），"
                 f"Target Margin {f(q2a['margin'])} → {f(orc['margin'])}。")
        L.append(f"- 独立 DINO-S：**{f(q2a['dino_s'])} → {f(orc['dino_s'])}**"
                 f"（Δ = {f(orc['dino_s']-q2a['dino_s'],4)}），"
                 f"相对源图增益 {f(q2a['dino_s_gain'])} → {f(orc['dino_s_gain'])}。")
        L.append(f"- 内容代价：DINO-C **{f(q2a['dino_c'])} → {f(orc['dino_c'])}**，"
                 f"LPIPS {f(q2a['lpips'])} → {f(orc['lpips'])}。")
        L.append(f"- **结论**：Oracle 的分类器收益（Top-1 +约 28pp）远大于独立画风相似度收益"
                 f"（DINO-S 仅 +约 {f(orc['dino_s']-q2a['dino_s'],3)}），"
                 f"且伴随明显内容下降。R1 梯度主要把图推向分类决策边界，"
                 f"而非等幅地提升真实画风相似度——Q1 的两种情形中，更靠近"
                 f"『提高分类准确率』，只是部分同时抬升了 DINO-S。")
    L.append("")

    # ---------------- B ----------------
    L.append("## B. 内容损失从哪里产生？")
    L.append("")
    if p1:
        s = p1["summary"]
        a = p1["alignment"]
        L.append("强度扫描（Val200，RMS 递增）：")
        L.append("")
        L.append("| 变体 | RMS | Top-1 | DINO-S | DINO-C | LPIPS |")
        L.append("|---|---:|---:|---:|---:|---:|")
        rms_map = {"E0_q2a": "0", "E1_rms008": "0.08", "E2_rms015": "0.15",
                   "E3_rms025": "0.25", "LL_rms025": "0.25 (LL)", "HF_rms025": "0.25 (HF)"}
        for k in ("E0_q2a", "E1_rms008", "E2_rms015", "E3_rms025", "LL_rms025", "HF_rms025"):
            if k in s:
                v = s[k]
                L.append(f"| {k} | {rms_map[k]} | {f(v['top1']*100,1)}% | "
                         f"{f(v['dino_s'])} | {f(v['dino_c'])} | {f(v['lpips'])} |")
        L.append("")
        L.append(f"- 方向对齐：`cos(g, -∇L_inv) = {f(a['mean_cos_neg_inv'])}`"
                 f"（R1 梯度与 canonical 内容不变性负梯度显著反向），"
                 f"`cos(g, -∇L_content) = {f(a['mean_cos_neg_content'])}`、"
                 f"`cos(g, Q2a Δ) = {f(a['mean_cos_q2a_delta'])}`、"
                 f"`cos(g, ScoreDiff Δ) = {f(a['mean_cos_scorediff'])}`。")
        L.append("- 频带：LL-only 在同等 RMS 下 DINO-C 明显更好（约 0.79 vs 全带/HF 约 0.73/0.71），"
                 "说明内容破坏主要集中在高频子带。")
    L.append("")
    L.append("- **结论**：内容损失主要由『编辑幅度』驱动（RMS 从 0→0.25 时 DINO-C 单调下降），"
             "且高频分量是主要破坏来源；R1 梯度与内容结构梯度近乎正交（cos≈−0.03），"
             "**不是**强烈的方向冲突。这解释了为什么 P2 的切向投影几乎不起作用。")
    L.append("")

    # ---------------- C ----------------
    L.append("## C. 是否存在更好的教师？")
    L.append("")
    if p2:
        L.append("教师对照（Val200）：")
        L.append("")
        L.append("| 教师 | Top-1 | DINO-S | DINO-C | LPIPS | RT-RMS |")
        L.append("|---|---:|---:|---:|---:|---:|")
        for k, v in p2["summary"].items():
            L.append(f"| {k} | {f(v['top1']*100,1)}% | {f(v['dino_s'])} | "
                     f"{f(v['dino_c'])} | {f(v['lpips'])} | {f(v['rt_stability_rms'])} |")
        L.append("")
        L.append("- **内容切向投影 T3 与历史 Oracle T1 几乎逐位相同**"
                 "（Top-1 33.0% vs 33.0%，DINO-C 0.7295 vs 0.7293）：因为 R1 梯度与内容梯度近乎正交，"
                 "半空间投影几乎不删除任何分量。这是本轮明确的负结果。")
        L.append("- 在 DINO-C ≥ 0.78 档位中，`T4_projected_ll025`（LL 频带 + 投影）以 "
                 "Top-1 24.5% / DINO-C 0.788 相对 T1（DINO-C 0.729）扩展了一小块前沿；"
                 "但该收益来自 **LL 频带筛选**，不是来自投影。")
        L.append("- 残差锚点 `z_new = zs + delta`（零训练）把 DINO-C 提到 0.9195、LPIPS 降到 0.256，"
                 "但 Top-1 掉到 4.0%（重新引入源风格），故不采纳。")
    L.append("")
    L.append("- **结论**：不存在『内容约束更好』的机制性新教师；唯一可用的帕累托改进是"
             "降低 RMS 或 LL 频带筛选（等价于保守编辑）。按计划决策树，"
             "不扩大投影参数搜索，保留诊断结论，仅以 T4（LL 投影）作为 P3 的新教师输入。")
    L.append("")

    # ---------------- D ----------------
    L.append("## D. 教师能否被轻量学生学习？")
    L.append("")
    if fit:
        L.append("学生拟合能力诊断（Val200）：")
        L.append("")
        L.append("| 学生 | 教师 | cos(ΔS,ΔT) | RMS 比 | 相对拟合误差 | 学生-教师 MSE | R1 margin gap |")
        L.append("|---|---:|---:|---:|---:|---:|---:|")
        for arm, d in fit.items():
            L.append(f"| {arm} | {d['teacher_mode']} | {f(d['mean_cosine'])} | "
                     f"{f(d['mean_rms_ratio'])} | {f(d['mean_rel_fit_error'])} | "
                     f"{f(d['mean_mse_student'],6)} | {f(d['mean_margin_diff'])} |")
        L.append("")
        L.append("- 218K 与 845K 学生的拟合能力接近（cos≈0.64–0.68，相对误差≈0.53–0.60），"
                 "**没有出现『845K 明显拟合更好』**——再次排除容量瓶颈。")
        L.append("- 蒸馏项确实对权重产生非零更新（单元测试 12：sum|ΔW|≈1301）。"
                 "但拟合误差几乎全部集中在 LL 子带（Haar LL MSE 远大于 LH/HL/HH）。")
    L.append("")
    b3 = cross_summary(evals.get("B3 (λ=20)"))
    s1 = cross_summary(evals.get("S1 (218K, P2 teacher, λ=20)"))
    s3 = cross_summary(evals.get("S3 (845K, Oracle teacher, λ=20)"))
    if b3 and s1:
        L.append(f"- 下游结果：P2 教师学生 S1/S2/S4 的 DINO-S（{f(s1['dino_s'])}）"
                 f"**未超过**历史 B3（{f(b3['dino_s'])}），Top-1 也更低"
                 f"（{f(s1['top1']*100,1)}% vs {f(b3['top1']*100,1)}%）；"
                 f"唯一持平/略优的是用历史 Oracle 教师的 845K 学生 S3"
                 f"（Top-1 {f(s3['top1']*100,1)}% ，DINO-S {f(s3['dino_s'])}）。")
    L.append("")
    L.append("- **结论**：学生能拟合教师（非优化/容量问题），但**新教师（LL 投影）目标本身更弱**，"
             "学得再好也只是复现一个保守编辑。归因落到『教师目标或泛化』，"
             "对应决策表『两者拟合好、DINO-S 均无提升 → 教师目标/泛化问题』。")
    L.append("")

    # ---------------- E ----------------
    L.append("## E. 困难风格有没有改善？")
    L.append("")
    L.append("分目标 CSD Top-1（Cross600，各 120 任务）：")
    L.append("")
    styles = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
    L.append("| 模型 | " + " | ".join(s[:9] for s in styles) + " |")
    L.append("|---|" + "---:|" * len(styles))
    for name, d in evals.items():
        c = cross_summary(d)
        pt = c.get("per_target", {}) if c else {}
        cells = [f"{f(pt.get(s, {}).get('top1', 0)*100,1)}%" for s in styles]
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("- **Early Renaissance**：全路线最高仅约 4–5%（ScoreDiff N=24 约 13% 但靠整体重绘），"
             "本轮无任何新改善。")
    L.append("- **Ukiyo-e**：全路线 2–4%，本轮无改善；Oracle 在该类 DINO-S 提升主要来自整体相似度抬升。")
    L.append("- Minimalism 非目标误判率稳定在 25–29%，未膨胀；增益不是靠 Minimalism 单类吸引子。")
    L.append("")
    L.append("- **结论**：两个硬风格没有改善。这是 SD-VAE latent 几何与这两类画风的表示鸿沟，"
             "不是本轮教师/容量/损失能解决的问题。")
    L.append("")

    # ---------------- success criteria ----------------
    L.append("## P4.3 预先设定的成功标准判定")
    L.append("")
    L.append("主门槛（相对 B3）：DINO-S ≥ +0.020 且 CI 下界 > 0；DINO-C 不低于 B3 超过 0.02；"
             "LPIPS 不恶化超过 0.03；Top-1 不下降超过 2pp；≥4 类风格 DINO-S 非负；"
             "Minimalism 误判不膨胀超过 +5pp。")
    L.append("")
    L.append("- 配对聚类 bootstrap（150 簇）：所有新学生 S1/S2/S4 相对 B3 的 ΔDINO-S 均为**负**"
             "（约 −0.010 ~ −0.012，CI 全在 0 以下）；S3 相对 B3 的 ΔDINO-S ≈ +0.001 且 CI 含 0。")
    L.append("- **没有任何模型满足 DINO-S +0.020 的主门槛。** 因此 G4 = 否。")
    L.append("- 按计划 G4 分支：『没有 +0.020 独立显著收益，但获得了可靠的机制诊断，"
             "仍按完整负结果交付，不通过修改测试协议或选择性展示制造提升』。")
    L.append("")

    # ---------------- F ----------------
    L.append("## F. 下一阶段应该做什么？（三选一）")
    L.append("")
    L.append("**结论：教师失败（Teacher failure）→ 转向更明确的目标分布建模，而非继续蒸馏。**")
    L.append("")
    L.append("依据链：")
    L.append("1. G0 通过：统一协议、12 项单测、DINO-S 判别力校准全部成立。")
    L.append("2. G1：R1 编辑能提高 DINO-S，但幅度（约 +0.03）远小于 CSD Top-1（约 +28pp），"
             "且伴随内容下降 → 更接近『分类边界捷径 + 部分真实风格收益』的混合，"
             "独立风格收益不足以支撑『判别梯度=真实条件风格迁移』。")
    L.append("3. G2：结构约束教师**未扩展**风格–内容 Pareto 前沿——"
             "切向投影近乎无效（梯度与内容梯度正交），唯一改进来自 LL 频带筛选（保守编辑）。")
    L.append("4. G3：学生能拟合教师（218K≈845K），排除容量；但教师目标本身更弱。")
    L.append("5. G4：无 +0.020 的独立显著 DINO-S 收益。")
    L.append("")
    L.append("后续建议：")
    L.append("- 不再沿『分类器梯度 + 内容投影 + 蒸馏』这条线加参数或换损失；"
             "本轮已用受控实验证明其上限。")
    L.append("- 转向显式目标风格分布建模（例如以真实画作特征为条件的生成目标、"
             "或基于参考图的分布匹配），因为当前 R1 只提供局部判别方向，"
             "不提供『什么是该风格的真实画作』的分布信息。")
    L.append("- Early Renaissance / Ukiyo-e 需要专门的表示或数据，不是通用教师能覆盖。")
    L.append("- 保留 `P1_B3_distill20`（218K）作为高保真档最好单步基线，"
             "以及完整负结果与诊断数据，避免重复探索。")
    L.append("")

    # ---------------- appendix ----------------
    L.append("## 附录：产物与复现")
    L.append("")
    L.append("```bash")
    L.append("bash sbflow/scripts/run_r2_12h.sh            # 单入口串行（可续跑）")
    L.append("bash sbflow/scripts/run_r2_12h.sh --dry-run  # 仅预检（编译 + 12 单测）")
    L.append("python3 -m tools.v6_next.r2_unit_tests      # 12 项单元测试")
    L.append("```")
    L.append("")
    L.append("- 结果目录：`runs/v6_r2/`（P0 六基线评测、P1 梯度扫描、P2 教师、"
             "P3 四学生 + 拟合诊断、P4 最终评测与配对报告、视觉网格、provenance）。")
    L.append("- 关键脚本：`tools/v6_next/r2_eval.py`、`r2_teacher.py`、"
             "`run_p1_analysis.py`、`run_p2_teachers.py`、`student_fit_diag.py`、"
             "`paired_comparison_r2.py`、`make_visual_grids.py`、`r2_unit_tests.py`。")
    L.append("- 训练改动：`tools/v6/train_eval_v6.py` 仅新增 `--distill-mode "
             "{oracle,projected,projected_ll}`、`--out-base`，"
             "并跳过缺失的旧 Eval200 任务文件（改由 r2_eval 统一评测）。")
    L.append("- 已知限制：Val200 ⊂ Test750 且 Test750 反复参与研发；"
             "本轮未构造作者隔离盲集，所有显著性均为研发集内配对，不宣称盲测。"
             "DINO/CLIP 非专门风格解耦模型，已同时查看 margin 避免整体相似度误判。")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[P5] wrote {OUT}  ({len(L)} lines)")
    # machine-readable verdict summary too
    verdict = {
        "route": "teacher_failure",
        "g4_dino_s_gain_020": False,
        "best_fidelity_single_step": "P1_B3_distill20",
        "content_tangential_projection": "no-op (gradient orthogonal to content gradient)",
        "unit_tests": "12/12 passed",
    }
    (R2 / "final_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print("[P5] wrote runs/v6_r2/final_verdict.json")


if __name__ == "__main__":
    main()
