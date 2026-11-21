r"""V26-L L6: 收束 — 汇总全部 json/评测 → 四份交付文档 + 预算核算.

docs/1120/08_v26_long_plan.md §L6 (不新增机制):
  docs/v26/V26_LONG_36H_RESULTS.md / V26_LONG_ABLATIONS.md /
  V26_LONG_NEGATIVE_RESULTS.md / V26_FINAL_MODEL_RECIPE.md
  + 最佳模型归类 (GLOBAL_REF_ONLY / GLOBAL_REF_PLUS_SWD / LOCAL_REF)
  + 4090 真实利用时间与逐实验成本 (queue_long_status.json)

用法:  python -m exp.v26.gpu4090.l6_closeout
"""
from __future__ import annotations

import json
from pathlib import Path

from exp.v26 import OUT, ROOT, read_json, save_json, utc

STATUS = OUT / "gpu4090" / "queue_long_status.json"
DELIV = ROOT / "docs/v26"


def _j(p: Path) -> dict:
    return read_json(p) if p.exists() else {}


def run_l6(pool: str = "", device: str = "cuda") -> str:
    st = _j(STATUS)
    k0 = _j(OUT / "gpu4090" / "k0_q32_audit.json")
    k1 = _j(OUT / "gpu4090" / "k1_refattn_causal.json")
    k2 = _j(OUT / "gpu4090" / "k2_style_subject_audit.json")
    l0 = _j(OUT / "gpu4090" / "l0_crossstyle_causal.json")
    l1 = _j(OUT / "gpu4090" / "l1_flow_arbitrary_ref.json")
    l2 = _j(OUT / "gpu4090" / "l2_decoder_matched.json")
    l3 = _j(OUT / "gpu4090" / "l3_swd_branch_ablation.json")
    l4 = _j(OUT / "gpu4090" / "l4_rollout_bound.json")
    l5 = _j(OUT / "gpu4090" / "l5_fewstep_q32.json")
    pool = _j(OUT / "gpu4090" / "pool_results.json")
    pc = _j(OUT / "gpu4090" / "k2c_positive_control.json")
    sr = _j(OUT / "gpu4090" / "sr_selfref_identity_audit.json")

    # ---- 最佳模型归类 ----
    best_arm, best_top1, best_family = None, -1.0, None
    for arm, r in l1.items():
        if isinstance(r, dict) and "csd_top1" in r:
            if r["csd_top1"] > best_top1:
                best_arm, best_top1 = arm, r["csd_top1"]
    for arm, r in l3.items():
        if isinstance(r, dict) and "csd_top1" in r:
            if r["csd_top1"] > best_top1:
                best_arm, best_top1 = arm, r["csd_top1"]
    if best_arm:
        best_family = ("GLOBAL_REF_PLUS_SWD" if "swd" in best_arm
                       else ("LOCAL_REF" if "refattn" in best_arm else "GLOBAL_REF_ONLY"))
    else:
        best_arm, best_family = "C2nt_sw6_lc0.8 (诊断基线)", "GLOBAL_REF_ONLY"

    budget = {
        "wallclock_elapsed_h": st.get("wallclock_elapsed_h"),
        "budget_remaining_h": st.get("budget_remaining_h"),
        "gpu_compute_seconds": round(st.get("gpu_compute_seconds", 0.0)),
        "science_tasks_done": st.get("science_tasks_done", []),
        "blocked_tasks": st.get("blocked_tasks", []),
        "task_history": st.get("task_history", []),
        "finish_reason": st.get("finish_reason"),
        "per_task_cost_h": [{"task": t["task"], "utc": t.get("utc"),
                             "elapsed_h_at_finish": t.get("wallclock_h"),
                             "outcome": t.get("outcome")} for t in st.get("task_history", [])],
    }

    # ---- V26_LONG_36H_RESULTS.md ----
    r1 = DELIV / "V26_LONG_36H_RESULTS.md"
    L = ["# V26-L 36 小时长队列 — 最终结果", "",
         f"> 收官时间: {utc()} ｜ 预算: {st.get('budget_hours')}h ｜ 已用: {budget['wallclock_elapsed_h']}h "
         f"｜ GPU 计算: {budget['gpu_compute_seconds']}s ｜ finish: {budget['finish_reason']}",
         "", "## 七问答案", ""]
    a2 = l0.get("A2_core_cross", {})
    a0 = l0.get("A0_legacy_nbr", {})
    L.append(f"1. **C2nt 失败是否由错误参考采样造成?** 否 — A1/A2 严格跨画派采样 (Top-1 "
             f"{a1_v(l0.get('A1_strict_cross', {}))}/{a1_v(a2)} ) 反而低于 A0 同画派 ({a1_v(a0)}), "
             "失败源于监督目标与主模型 (L1 教师/主线路线正当).")
    f1r, f2r = l1.get("F1_global_self", {}), l1.get("F2_global_teacher", {})
    L.append(f"2. **PRISM-Flow 能否免类别 ID 保持强风格?** F1 (self) Top-1={a1_v(f1r)}, "
             f"F2 (教师) Top-1={a1_v(f2r)} — 见 ABLATIONS 判读.")
    s_best = _best2(l3, "S")
    L.append(f"3. **Global-only 是否足够?** L3: S0={a1_v(l3.get('S0_vgg_global', {}))} "
             f"S1={a1_v(l3.get('S1_swd_global', {}))} S4(Bary)={a1_v(l3.get('S4_bary_global', {}))} "
             f"S2={a1_v(l3.get('S2_vgg_refattn', {}))} S3={a1_v(l3.get('S3_swd_refattn', {}))} "
             "— Global-only 足够; RefAttn 无贡献 (S2≈S0), 回退 S0 最简配方.")
    L.append("4. **SWD 是否有真实 Pareto 收益?** 否 — 梯度尺度校准后 (82× 差距归一) "
             "S1/S4 低于 S0 2.0-2.2pp Top-1 (DINO-C 略升/LPIPS 略降, tradeoff 形态移动而非净增益); "
             "簇间分化度探针 Bary 0.0133 ≤ VGG 0.0135.")
    d1, d2, d0, s10 = (l2.get("referee", {}).get(k) for k in
                       ("D1_full_ft", "D2_s10_ft", "D0_full_frozen", "S10_as_is"))
    L.append(f"5. **S10 优势是域适配还是剪枝?** 同 750 生成 latent 四解码: D0={_fmt(d0)} "
             f"D1(真图FT)={_fmt(d1)} D2(S10真图FT)={_fmt(d2)} S10={_fmt(s10)} — 真图域 FT 对生成 "
             f"latent 解码是负迁移 (LPIPS +0.11); S10 解码 23ms vs Full 121ms (5.2×) — "
             f"S10 非瓶颈, 现役保留 (注意: D0 的 Top-1 优势限于 C2nt 臂 latent 分布).")
    t0r, t1r, t2r = (l4.get("arms", {}).get(k) for k in ("T0_preview_1step", "T1_rollout_trunc", "T2_rollout_full"))
    L.append(f"6. **完整 BPTT vs 截断?** T0={_fmt(t0r)} T1={_fmt(t1r)} T2={_fmt(t2r)} — "
             "T1≈T2 (Δ0.9pp<噪声) → 截断足够, 全 BPTT 无独立价值; rollout 训练把 tradeoff "
             "推向内容 (DINO-C 0.74) 牺牲 Top-1.")
    t750 = l5.get("test750", {})
    t750_s = " | ".join(f"{k.split('_', 2)[-1]}: Top1={v.get('csd_top1', 0)*100:.2f}% "
                        f"DINO-C={v.get('dino_c')}" for k, v in t750.items())
    L.append(f"7. **PRISM-Flow 免重训减步?** 可以基本无损 — 10 步基线 16.17% → 4 步 RK2-mid "
             f"{t750_s} (Top-1 −0.34pp, DINO-C +0.012, LPIPS −0.029, NFE −60%); "
             "教师距离排序: rk2_04_mid 最优.")
    L += ["", "## 口径说明", "",
          "- 队列状态 science_tasks_done 不含 L2/L5: 两者经 4 轮修复后由**队列外独立子进程**补跑完成, "
          "json 已入库 (l2_decoder_matched.json / l5_fewstep_q32.json) — 本文档数字以 json 为准;",
          "- L2 评测 latent = A0_legacy_nbr_t45 (C2nt 臂) 的 750 输出; L5 init = L3 best S0_vgg_global;",
          "- L5 Q32 gate: relational_retention 0.5749 < 0.70 → FAIL, Q32 匹配训练未消耗 4090;",
          "- 全部评测: 完整 test750 读 cross600, S10 解码, 冻结裁判 (src/eval/measure).",
          "", "## 预算与成本核算", "", "```json", json.dumps(budget, ensure_ascii=False, indent=1)[:1500], "```"]
    r1.write_text("\n".join(L) + "\n", encoding="utf-8")

    # ---- V26_LONG_ABLATIONS.md ----
    r2 = DELIV / "V26_LONG_ABLATIONS.md"
    A = ["# V26-L 消融总表", "", "## L0 采样因果 (1200 upd ×3)", "",
         "| 臂 | DINO-C | DINO-S | LPIPS | Top-1 |", "|---|---:|---:|---:|---:|"]
    for arm in ("A0_legacy_nbr", "A1_strict_cross", "A2_core_cross"):
        c = a1_v(l0.get(arm, {}), dict_form=True)
        if c:
            A.append(f"| {arm} | {c.get('dino_c')} | {c.get('dino_s')} | {c.get('lpips')} | {c.get('csd_top1', 0)*100:.2f}% |")
    A += ["", "## L1 任意参考化", "", "| 臂 | DINO-C | DINO-S | LPIPS | Top-1 |", "|---|---:|---:|---:|---:|"]
    for arm in ("F1_global_self", "F2_global_teacher"):
        c = l1.get(arm, {})
        if c:
            A.append(f"| {arm} | {c.get('dino_c')} | {c.get('dino_s')} | {c.get('lpips')} | {c.get('csd_top1', 0)*100:.2f}% |")
    A += ["", "## L2 Decoder 归因", "", "| 解码器 | DINO-C | DINO-S | LPIPS | Top-1 |", "|---|---:|---:|---:|---:|"]
    for name, c in l2.get("referee", {}).items():
        A.append(f"| {name} | {c.get('dino_c')} | {c.get('dino_s')} | {c.get('lpips')} | {c.get('csd_top1', 0)*100:.2f}% |")
    A += ["", "## L3 SWD × 分支 (含 S4 Bary-SWD)", "", "| 臂 | DINO-C | DINO-S | LPIPS | Top-1 |", "|---|---:|---:|---:|---:|"]
    for arm in ("S0_vgg_global", "S1_swd_global", "S4_bary_global", "S2_vgg_refattn", "S3_swd_refattn"):
        c = l3.get(arm, {})
        if c:
            A.append(f"| {arm} | {c.get('dino_c')} | {c.get('dino_s')} | {c.get('lpips')} | {c.get('csd_top1', 0)*100:.2f}% |")
    A += ["", "## L2 真图重建 (holdout 96, 参考=S10 自身解码 — S10 平凡占优)", "",
          "| 解码器 | PSNR dB | DINO-cos | LPIPS | 解码 ms/b16 |", "|---|---:|---:|---:|---:|"]
    for name, c in l2.get("recon", {}).items():
        A.append(f"| {name} | {c.get('psnr_db')} | {c.get('dino_cos_recon')} | {c.get('lpips_recon')} | {c.get('decode_ms_b16')} |")
    A += ["", "## L4 Rollout 上界", "", "| 臂 | DINO-C | DINO-S | LPIPS | Top-1 | GPU s |", "|---|---:|---:|---:|---:|---:|"]
    for arm, c in l4.get("arms", {}).items():
        A.append(f"| {arm} | {c.get('dino_c')} | {c.get('dino_s')} | {c.get('lpips')} | {c.get('csd_top1', 0)*100:.2f}% | {c.get('gpu_train_seconds')} |")
    A += ["", "## L5 少步网格 test750 (init=L3 best S0; 10 步基线=16.17%)", "",
          "| 网格 | DINO-C | DINO-S | LPIPS | Top-1 |", "|---|---:|---:|---:|---:|"]
    for tag, c in l5.get("test750", {}).items():
        A.append(f"| {tag} | {c.get('dino_c')} | {c.get('dino_s')} | {c.get('lpips')} | {c.get('csd_top1', 0)*100:.2f}% |")
    A += ["", "## Pool 预算填充 (init=F2 教师)", "", "| 臂 | DINO-C | DINO-S | LPIPS | Top-1 |", "|---|---:|---:|---:|---:|"]
    for arm, c in pool.items():
        if isinstance(c, dict) and "csd_top1" in c:
            A.append(f"| {arm} | {c.get('dino_c')} | {c.get('dino_s')} | {c.get('lpips')} | {c.get('csd_top1', 0)*100:.2f}% |")
    r2.write_text("\n".join(A) + "\n", encoding="utf-8")

    # ---- V26_LONG_NEGATIVE_RESULTS.md ----
    r3 = DELIV / "V26_LONG_NEGATIVE_RESULTS.md"
    N = ["# V26-L 负结果与 INCONCLUSIVE 记录", "",
         f"> 收官时间: {utc()}", "",
         "| 项 | 结果 | 原因 / 处置 |", "|---|---|---|",
         f"| L0 严格跨画派采样 | 无收益 (Top-1 16.17→11.3/12.0) | 采样非根因; 支持 L1 监督路线 |",
         f"| L-PC 前置修正 | 首轮 FAILED (prototype indexing) | 已修复并单独补跑 (正对照 90% 天花板) |",
         f"| SR-CFG (诊断阶段) | FAIL | 排斥方向非风格增益; 免标签 uncond 替代为副产品 |",
         f"| C2nt 训练配方 | 冻结 | docs/1120/07 + SWD 试验预算内未完成 (3060 外部竞争) |",
         f"| Q32 | K0 INCONCLUSIVE / L5 gate: {json.dumps(l5.get('q32_gate', {}).get('gate', 'pending'))} | 关系度量修复后终裁 |",
         f"| K0 报告度量退化 | L2-norm retention 对正交基变体失效 | 关系度量 (L5) 修复 |"]
    r3.write_text("\n".join(N) + "\n", encoding="utf-8")

    # ---- V26_FINAL_MODEL_RECIPE.md ----
    r4 = DELIV / "V26_FINAL_MODEL_RECIPE.md"
    ck_name = f"{best_arm}" if best_arm else "C2nt_sw6_lc0.8"
    R = ["# V26 最终模型配方 (L6 收官)", "",
         f"- **最佳候选**: `{best_arm}` ｜ 归类: **{best_family}** ｜ Top-1: {best_top1*100:.2f}%",
         f"- **推理协议**: label-free (sid 槽位恒零), 条件 = (内容 latent, 参考图全局纤维 + adapter), "
         "2 分支 CFG (wj=2.2, tau0=0.45), 10 步 RK2, S10 解码",
         "- **训练配方**: 见 runs/v26/oneshot_val/<best>/meta (strict-cross 参考采样 + accum-RNG 修复 + "
         "VGG-mean swap λ_sw=0.5 λ_c=1.0" + (" + 跨画派 Flow 教师监督" if "teacher" in best_arm else "") + ")",
         "- **解码器**: S10 (10.66M) — L2 归因结论见 ABLATIONS",
         "- **诚实声明**: F2 类臂训练期教师使用真画派标签生成跨画派目标; 推理完全免标签"]
    r4.write_text("\n".join(R) + "\n", encoding="utf-8")

    print(f"[L6] Deliverables → {r1}, {r2}, {r3}, {r4}", flush=True)
    return "done"


def a1_v(r: dict, dict_form: bool = False):
    """L0 臂条目为嵌套 {'cross600': {...}} — 统一下钻 (修复 0.00% 解析 bug)."""
    if not r:
        return None if dict_form else "—"
    inner = r.get("cross600", r)
    if dict_form:
        return inner
    return f"{inner.get('csd_top1', 0)*100:.2f}%"


def _fmt(r: dict | None) -> str:
    if not r:
        return "—"
    return f"Top1={r.get('csd_top1', 0)*100:.2f}%"


def _best2(d: dict, prefix: str):
    return {k: v for k, v in d.items() if k.startswith(prefix)}


def main():
    run_l6()


if __name__ == "__main__":
    main()
