"""Ranked comparison of Q2a + R1 Oracle against the baselines on the formal 750-task protocol.

Every row is measured on tasks_750.json (600 cross + 150 same, 150 sources, 30 per style), scored
through the same paper_rgb_v1 path, and reported cross-only (split by target != source, because
eval_tiers rows carry no is_cross field).

Two rankings are given because one is not enough: ranking on Top-1 alone rewards destroying content,
so a second ranking applies the plan's content constraint DINO-C >= 0.85.

Paired source-clustered bootstrap is computed against two baselines -- Q2a Direct (the Oracle's own
base) and L11 (the original lightweight baseline) -- with task alignment asserted before pairing.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tools.v6_24h.stats import cluster_ci, paired_cluster_ci

ROOT = Path(__file__).resolve().parents[2]
STY = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
MIN = STY.index("Minimalism")


def p0_rows():
    p = ROOT / "runs/v6_next/amortize/p0_variants.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text()).get("rows") or {}


def eval750_rows(name):
    p = ROOT / f"runs/v6_next/eval_750_{name}/eval.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("rows")


def norm(r):
    """Unify the two row schemas.

    amortize_p0 writes   {top1: int, argmax: str, m_target: float}
    eval_tiers writes    {target_top1: int, pred_class: int, target_margin: float}
    Reading only one schema silently KeyErrors on the other, so both are accepted explicitly.
    """
    top1 = r.get("target_top1")
    if top1 is None:
        top1 = r.get("top1")
    pred = r.get("pred_class")
    if pred is None:
        pred = r.get("argmax")
    pred_idx = STY.index(pred) if isinstance(pred, str) else int(pred)
    marg = r.get("target_margin")
    if marg is None:
        marg = r.get("m_target")
    return {
        "id": r["task_id"],
        "src": r["src_style"], "tgt": r["tgt_style"],
        "stem": r["src_stem"],
        "top1": float(top1),
        "dT": float(r["dT"]),
        "dino": float(r["dino_c"]),
        "lpips": float(r["lpips"]),
        "marg": float(marg) if marg is not None else float("nan"),
        "pred": pred_idx,
    }


def cross_only(rows):
    return [r for r in rows if r["tgt"] != r["src"]]


def summarise(rows):
    cross = cross_only(rows)
    t1 = np.array([r["top1"] for r in cross])
    cl = [r["stem"] for r in cross]
    ci = cluster_ci(t1, cl)
    conf = np.zeros((5, 5), dtype=int)
    for r in cross:
        conf[STY.index(r["tgt"])][r["pred"]] += 1
    hits = sum(conf[i][MIN] for i in range(5) if i != MIN)
    tot = sum(sum(conf[i]) for i in range(5) if i != MIN)
    return {
        "n": len(cross),
        "top1": float(t1.mean()), "ci": ci,
        "dT": float(np.mean([r["dT"] for r in cross])),
        "dino": float(np.mean([r["dino"] for r in cross])),
        "lpips": float(np.mean([r["lpips"] for r in cross])),
        "margin": float(np.mean([r["marg"] for r in cross])),
        "min_off": hits / max(1, tot),
        "per_target": {s: float(np.mean([r["top1"] for r in cross if r["tgt"] == s]))
                       for s in STY},
        "rows": cross,
    }


def main() -> None:
    arms = []

    p0 = p0_rows()
    for key, label in (("A0", "Q2a Direct (Oracle base)"),
                       ("A1", "**Q2a + R1 Oracle, 1 step**"),
                       ("A2", "Q2a + R1 Oracle, 4 steps dyn"),
                       ("A4", "ScoreDiff N=24")):
        if key in p0:
            arms.append((label, summarise([norm(r) for r in p0[key]])))

    for name, label in (("L11_k1024_dim16", "L11 Direct (original baseline)"),
                        ("Q2a_embed_style", "Q2a Direct (750 re-eval)"),
                        ("scorediff_N12", "ScoreDiff N=12 strong"),
                        ("P1_B0_ctrl218", "B0 control (continued 800 steps)"),
                        ("P1_B3_distill20", "B3 distilled lambda=20"),
                        ("P1_B4_distill50", "B4 distilled lambda=50")):
        rows = eval750_rows(name)
        if rows:
            arms.append((label, summarise([norm(r) for r in rows])))

    if not arms:
        print("no data"); return

    # ---- ranking by Top-1 ----
    ranked = sorted(arms, key=lambda x: -x[1]["top1"])
    print("=" * 116)
    print("正式排名评估集 = tasks_750.json  (600 cross + 150 same, 150 sources, 30/style)")
    print("指标一律 cross-only；CI 为源图聚类 bootstrap（150 簇，4000 次）")
    print("=" * 116)
    print(f"  {'#':>2s}  {'模型':30s}{'Cross600 Top-1':>15s}{'95% CI':>22s}"
          f"{'DINO-C':>8s}{'LPIPS':>8s}{'Min 误判':>9s}")
    for i, (label, s) in enumerate(ranked, 1):
        print(f"  {i:>2d}  {label:30s}{s['top1']*100:14.1f}%"
              f"{'[%.1f, %.1f]' % (s['ci']['ci'][0]*100, s['ci']['ci'][1]*100):>22s}"
              f"{s['dino']:8.3f}{s['lpips']:8.3f}{s['min_off']*100:8.1f}%")

    # ---- content-constrained ranking ----
    print("\n" + "=" * 116)
    print("内容受限排名（DINO-C >= 0.85，计划定义的保真档）")
    print("=" * 116)
    ok = [a for a in arms if a[1]["dino"] >= 0.85]
    for i, (label, s) in enumerate(sorted(ok, key=lambda x: -x[1]["top1"]), 1):
        print(f"  {i:>2d}  {label:30s}{s['top1']*100:14.1f}%   DINO-C {s['dino']:.3f}")
    print(f"\n  被排除（DINO-C < 0.85，强风格档）:")
    for label, s in sorted([a for a in arms if a[1]["dino"] < 0.85],
                           key=lambda x: -x[1]["top1"]):
        print(f"      {label:30s}{s['top1']*100:14.1f}%   DINO-C {s['dino']:.3f}")

    # ---- paired comparisons ----
    def paired_vs(base_label, base_rows, cand_label, cand_rows):
        if len(base_rows) != len(cand_rows):
            print(f"  {cand_label}: row count differs ({len(cand_rows)} vs {len(base_rows)})")
            return
        ids_a = [r["id"] for r in base_rows]
        ids_b = [r["id"] for r in cand_rows]
        if ids_a != ids_b:
            order = {k: i for i, k in enumerate(ids_b)}
            if set(ids_a) != set(ids_b):
                print(f"  {cand_label}: task sets differ, cannot pair")
                return
            cand_rows = [cand_rows[order[k]] for k in ids_a]
        cl = [r["stem"] for r in base_rows]
        ci_t = paired_cluster_ci([r["top1"] for r in cand_rows],
                                 [r["top1"] for r in base_rows], cl)
        ci_d = paired_cluster_ci([r["dino"] for r in cand_rows],
                                 [r["dino"] for r in base_rows], cl)
        print(f"  {cand_label:34s} dTop-1 {ci_t['diff']*100:+6.1f}pp "
              f"[{ci_t['ci'][0]*100:+5.1f},{ci_t['ci'][1]*100:+5.1f}] "
              f"excl0={str(ci_t['excludes_zero']):5s} | "
              f"dDINO {ci_d['diff']:+.3f} [{ci_d['ci'][0]:+.3f},{ci_d['ci'][1]:+.3f}]")

    oracle = next((s for l, s in arms if l.startswith("**Q2a + R1 Oracle, 1")), None)
    q2a = next((s for l, s in arms if l.startswith("Q2a Direct (Oracle base)")), None)
    l11 = next((s for l, s in arms if l.startswith("L11")), None)

    if q2a:
        print("\n" + "=" * 116)
        print("配对差分 vs Q2a Direct（Oracle 自己的基线）")
        print("=" * 116)
        for label, s in arms:
            if s is q2a:
                continue
            paired_vs("Q2a", q2a["rows"], label.replace("*", ""), s["rows"])

    if l11:
        print("\n" + "=" * 116)
        print("配对差分 vs L11 Direct（原始轻量基线）")
        print("=" * 116)
        for label, s in arms:
            if s is l11:
                continue
            paired_vs("L11", l11["rows"], label.replace("*", ""), s["rows"])

    # ---- per target for the Oracle and both baselines ----
    print("\n" + "=" * 116)
    print("分目标 Top-1（cross only）")
    print("=" * 116)
    show = [a for a in arms if a[0].startswith(("Q2a Direct (Oracle base)", "**Q2a",
                                               "L11", "ScoreDiff N=24"))]
    print(f"  {'模型':32s}" + "".join(f"{s[:9]:>11s}" for s in STY))
    for label, s in show:
        print(f"  {label.replace('*', ''):32s}"
              + "".join(f"{s['per_target'][x]*100:10.1f}%" for x in STY))

    if oracle and q2a:
        print("\n" + "=" * 116)
        print("结论")
        print("=" * 116)
        d = (oracle["top1"] - q2a["top1"]) * 100
        print(f"  Oracle 1 步把 Cross600 Top-1 从 {q2a['top1']*100:.1f}% 提到 "
              f"{oracle['top1']*100:.1f}%（{d:+.1f}pp，{oracle['top1']/max(q2a['top1'],1e-9):.1f} 倍）")
        print(f"  代价：DINO-C {q2a['dino']:.3f} → {oracle['dino']:.3f}"
              f"（{oracle['dino']-q2a['dino']:+.3f}），LPIPS "
              f"{q2a['lpips']:.3f} → {oracle['lpips']:.3f}")
        print(f"  Minimalism 误判：{q2a['min_off']*100:.1f}% → {oracle['min_off']*100:.1f}%")
        if l11:
            print(f"  相对 L11 基线：{l11['top1']*100:.1f}% → {oracle['top1']*100:.1f}%"
                  f"（{(oracle['top1']-l11['top1'])*100:+.1f}pp）")


if __name__ == "__main__":
    main()
