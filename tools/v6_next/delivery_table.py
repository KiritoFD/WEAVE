"""Build the consolidated delivery table from the measured JSONs, with provenance per row.

Every row states whether the number was MEASURED IN THIS SESSION (and from which file) or QUOTED
from another source, because mixing the two silently is the easiest way to misreport a comparison.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/mnt/i/Latent_Style/WEAVE")
STY = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
MIN = STY.index("Minimalism")


def cross_split(rows):
    return ([r for r in rows if r["tgt_style"] != r["src_style"]],
            [r for r in rows if r["tgt_style"] == r["src_style"]])


def from_eval750(name):
    p = ROOT / f"runs/v6_next/eval_750_{name}/eval.json"
    if not p.exists():
        return None
    rows = json.loads(p.read_text())["rows"]
    cross, same = cross_split(rows)
    if not cross:
        return None
    conf = [[0] * 5 for _ in range(5)]
    for r in cross:
        conf[STY.index(r["tgt_style"])][int(r["pred_class"])] += 1
    hits = sum(conf[i][MIN] for i in range(5) if i != MIN)
    tot = sum(sum(conf[i]) for i in range(5) if i != MIN)
    return {
        "top1": sum(r["target_top1"] for r in cross) / len(cross),
        "dT": sum(r["dT"] for r in cross) / len(cross),
        "margin": sum(r["target_margin"] for r in cross) / len(cross),
        "dino": sum(r["dino_c"] for r in cross) / len(cross),
        "lpips": sum(r["lpips"] for r in cross) / len(cross),
        "per_target": {s: (sum(r["target_top1"] for r in cross if r["tgt_style"] == s) /
                           max(1, sum(1 for r in cross if r["tgt_style"] == s))) for s in STY},
        "min_offdiag": hits / max(1, tot),
        "same_top1": (sum(r["target_top1"] for r in same) / len(same)) if same else None,
        "same_dS": (sum(r["dS"] for r in same) / len(same)) if same else None,
        "n_cross": len(cross), "n_same": len(same),
    }


def from_p0(variant):
    p = ROOT / "runs/v6_next/amortize/p0_variants.json"
    if not p.exists():
        return None
    v = json.loads(p.read_text())["variants"].get(variant)
    if not v:
        return None
    conf = v.get("confusion")
    mo = None
    if conf:
        h = sum(conf[i][MIN] for i in range(5) if i != MIN)
        t = sum(sum(conf[i]) for i in range(5) if i != MIN)
        mo = h / max(1, t)
    return {"top1": v.get("top1"), "dT": v.get("dT"), "margin": v.get("m_target"),
            "dino": v.get("dino_c"), "lpips": v.get("lpips"),
            "per_target": {s: (v.get("per_target") or {}).get(s, {}).get("top1")
                           for s in STY},
            "min_offdiag": mo,
            "same_top1": v.get("same_top1"), "same_dS": v.get("same_dS"),
            "n_cross": v.get("n_cross"), "n_same": v.get("n_same")}


def arm_report(name):
    p = ROOT / f"runs/v6/{name}/report.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    e = d.get("eval200") or {}
    pt = e.get("per_target_top1") or {}
    conf = e.get("confusion_target_vs_pred")
    mo = None
    if conf:
        h = sum(conf[i][MIN] for i in range(5) if i != MIN)
        t = sum(sum(conf[i]) for i in range(5) if i != MIN)
        mo = h / max(1, t)
    return {"top1": sum(pt.values()) / len(pt) if pt else None,
            "dT": e.get("mean_dT"), "margin": e.get("mean_dM"),
            "dino": e.get("mean_dino_c"), "lpips": e.get("mean_lpips"),
            "per_target": {s: pt.get(s) for s in STY}, "min_offdiag": mo,
            "same_dS": e.get("same50_mean_dS"), "params": d.get("n_params"),
            "distill_lambda": d.get("distill_lambda")}


rows = []
rows.append(("Q2a Direct", "218K", from_p0("A0"), "MEASURED", "P0 A0, Test600 (750 protocol)"))
rows.append(("Q2a + R1 Oracle (1 step)", "218K + R1", from_p0("A1"), "MEASURED",
             "P0 A1, Test600 (750 protocol)"))
rows.append(("Q2a + R1 Oracle (4 steps dyn)", "218K + R1", from_p0("A2"), "MEASURED",
             "P0 A2, Test600 (750 protocol)"))
rows.append(("V6 ScoreDiff N=24", "3.27M", from_p0("A4"), "MEASURED",
             "P0 A4, Test600 (750 protocol)"))
rows.append(("V6 ScoreDiff N=12 strong", "3.27M", from_eval750("scorediff_N12"), "MEASURED",
             "eval_750_scorediff_N12"))
rows.append(("B0 218K control (continued)", "218K", from_eval750("P1_B0_ctrl218"), "MEASURED",
             "eval_750_P1_B0_ctrl218"))
rows.append(("B3 distilled lambda=20", "218K", from_eval750("P1_B3_distill20"), "MEASURED",
             "eval_750_P1_B3_distill20"))
rows.append(("B4 distilled lambda=50", "218K", from_eval750("P1_B4_distill50"), "MEASURED",
             "eval_750_P1_B4_distill50"))

print("# V6 下一阶段 · 统一结果表（交付用）\n")
print("**评测器**：`paper_rgb_v1` 评分链路（CSD / DINOv2 / LPIPS + TRT fp16 VAE 解码 → uint8）")
print("**协议**：750 任务 = **600 cross + 150 same**，150 源图，每风格 120/120 均衡")
print("**Cross600 指标一律为 cross-only**（按 `tgt_style != src_style` 分离；")
print("`eval_tiers` 的行里不含 `is_cross`，用默认值会把 750 全算成 cross，Top-1 虚高 3.4 倍）\n")
print("| 模型 | 参数量 | Cross600 Top-1 | DINO-C | LPIPS | ΔT | Margin | Minimalism 误判 | 来源 |")
print("|---|---|---:|---:|---:|---:|---:|---:|---|")
for name, params, d, prov, src in rows:
    if d is None:
        print(f"| {name} | {params} | — | — | — | — | — | — | MISSING ({src}) |")
        continue
    t1 = f"{d['top1']*100:.1f}%" if d.get("top1") is not None else "—"
    dn = f"{d['dino']:.3f}" if d.get("dino") is not None else "—"
    lp = f"{d['lpips']:.3f}" if d.get("lpips") is not None else "—"
    dt = f"{d['dT']:+.4f}" if d.get("dT") is not None else "—"
    mg = f"{d['margin']:+.4f}" if d.get("margin") is not None else "—"
    mo = f"{d['min_offdiag']*100:.1f}%" if d.get("min_offdiag") is not None else "—"
    print(f"| {name} | {params} | **{t1}** | {dn} | {lp} | {dt} | {mg} | {mo} | {prov}: `{src}` |")

print("\n## 分目标 Top-1（Cross600）\n")
print("| 模型 | Early_Ren | Impressionism | Minimalism | Rococo | Ukiyo_e |")
print("|---|---:|---:|---:|---:|---:|")
for name, params, d, prov, src in rows:
    if d is None:
        continue
    cells = []
    for s in STY:
        v = (d.get("per_target") or {}).get(s)
        cells.append(f"{v*100:.1f}%" if v is not None else "—")
    print(f"| {name} | " + " | ".join(cells) + " |")

print("\n## 同风格（Same150）\n")
print("| 模型 | Same Top-1 | Same ΔS |")
print("|---|---:|---:|")
for name, params, d, prov, src in rows:
    if d is None:
        continue
    st = f"{d['same_top1']*100:.1f}%" if d.get("same_top1") is not None else "—"
    sd = f"{d['same_dS']:+.4f}" if d.get("same_dS") is not None else "—"
    print(f"| {name} | {st} | {sd} |")

print("\n## 未在本轮测量、仅作参照（来源不是本次运行）\n")
print("| 模型 | 参数量 | Top-1 | DINO-C | 来源 |")
print("|---|---|---:|---:|---|")
print("| V6 D1_T1 | 1.88M | 6.5% | 0.870 | 早期 `paper_rgb_v1` 结果，非本轮 |")
print("| 历史 WEAVE 原版 | 待核实 | 16.7% | 0.808 | 已发表系统，非本轮 |")
print("| L11 Direct | 218K | ~3% | ~0.89 | 本轮 Heldout400（400 任务），"
      "**不是** Cross600 协议 |")
