"""V11 P2: how much of the transfer actually comes from the EDM analytic term?

No training. The same checkpoint (exp01 equiv4000) and the same Quick200 tasks are
generated three times, scaling the network residual by 0 / 0.5 / 1:

    residual 0.0 -> pure analytic statistics transport  T(z) = mu_j + (sigma_j/sigma_i)(z-mu_i)
    residual 0.5 -> half the learned correction
    residual 1.0 -> the full model (V10 baseline)

Reported per target style: dT, margin, top1, corrected SWD, DINO-C. Plus
Delta_learned = dT(full) - dT(EDM-only), and the per-task agreement between the two.

IMPORTANT (per the execution decision): a small Delta_learned does NOT prove the network
only learned statistical transport. It only says the learned residual gives no clear net
benefit on THIS task set and metric. The residual may encode changes that do not convert
into target-style gain, so per-task differences and content metrics are reported too.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
SCALES = [0.0, 0.5, 1.0]


def run(cmd: List[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as f:
        f.write("\n>>> " + " ".join(map(str, cmd)) + "\n")
        f.flush()
        r = subprocess.run([str(c) for c in cmd], cwd=str(ROOT), stdout=f,
                           stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"rc={r.returncode}; see {log}")


def tag(scale: float) -> str:
    return f"rs{scale:.2f}".replace(".", "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/blog_sbflow/suite40/exp01/model_equiv4000.pt")
    ap.add_argument("--out-root", default="runs/blog_sbflow/v11")
    ap.add_argument("--subset", default="quick200")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--skip-gen", action="store_true")
    args = ap.parse_args()

    out_root = ROOT / args.out_root
    ckpt = ROOT / args.ckpt if not Path(args.ckpt).is_absolute() else Path(args.ckpt)
    if not ckpt.exists():
        raise SystemExit(f"missing checkpoint {ckpt}")

    scores: Dict[float, dict] = {}
    for sc in SCALES:
        t = tag(sc)
        gen = out_root / f"p2_{t}_gen"
        score = out_root / f"p2_{t}_score"
        if not args.skip_gen and not (gen / "manifest.json").exists():
            print(f"[p2] generating residual_scale={sc} ...", flush=True)
            run([args.python, "-m", "sbflow.gen5", "--repo-root", ".",
                 "--ckpt", str(ckpt), "--out", str(gen), "--subset", args.subset,
                 "--steps", str(args.steps), "--method", "heun", "--batch", "16",
                 "--residual-scale", str(sc)], out_root / "logs" / f"p2_{t}_gen.log")
        if not (score / "style_scores.json").exists():
            print(f"[p2] scoring residual_scale={sc} ...", flush=True)
            run([args.python, "-m", "sbflow.score_cpu", "--repo-root", ".",
                 "--gen", str(gen), "--out", str(score), "--batch", "8",
                 "--threads", str(args.threads)], out_root / "logs" / f"p2_{t}_score.log")
        scores[sc] = json.loads((score / "style_scores.json").read_text())

    # ---- report ----
    def overall(sc):
        return scores[sc]["aggregate"]["overall"]

    def by_target(sc):
        return scores[sc]["aggregate"]["by_target"]

    lines = ["# V11 P2: EDM 解析项 vs 学习残差的贡献", "",
             f"checkpoint: `{ckpt.relative_to(ROOT)}`，Quick200 同一 200 任务，"
             f"{args.steps}-step Heun。", "",
             "## 总体", "",
             "| residual | Top-1 | Margin | ΔT | ΔS | SWD(修正) | MMD(修正) | DINO-C | LPIPS |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for sc in SCALES:
        o = overall(sc)
        lines.append(f"| {sc:.1f} | {o['target_top1']:.4f} | {o['target_margin']:+.4f} | "
                     f"{o['csd_dT']:+.4f} | {o['csd_dS']:+.4f} | {o['csd_swd']:.5f} | "
                     f"{o['csd_mmd2_unbiased']:.5f} | {o['dino_c']:.4f} | "
                     f"{o['lpips_to_input']:.4f} |")

    d_learned = overall(1.0)["csd_dT"] - overall(0.0)["csd_dT"]
    lines += ["", f"**Δ_learned (ΔT) = ΔT(full) − ΔT(EDM-only) = {d_learned:+.4f}**", ""]

    lines += ["## 分目标风格", ""]
    for metric, name in (("csd_dT", "ΔT"), ("target_margin", "Margin"),
                         ("target_top1", "Top-1"), ("csd_swd", "SWD")):
        lines += [f"### {name}", "", "| residual | " + " | ".join(
            s.replace("Early_Renaissance", "ER").replace("Impressionism", "Imp")
             .replace("Minimalism", "Min").replace("Rococo", "Roc").replace("Ukiyo_e", "Uki")
            for s in STYLES) + " |",
            "|---:|" + "---:|" * len(STYLES)]
        for sc in SCALES:
            bt = by_target(sc)
            row = " | ".join(f"{bt[s][metric]:.4f}" if s in bt else "—" for s in STYLES)
            lines.append(f"| {sc:.1f} | {row} |")
        lines.append("")

    # per-target learned contribution
    lines += ["### 每目标的学习残差贡献 Δ_learned(ΔT)", "",
              "| 目标 | ΔT(EDM-only) | ΔT(full) | Δ_learned |", "|---|---:|---:|---:|"]
    b0, b1 = by_target(0.0), by_target(1.0)
    for s in STYLES:
        if s in b0 and s in b1:
            d = b1[s]["csd_dT"] - b0[s]["csd_dT"]
            lines.append(f"| {s} | {b0[s]['csd_dT']:+.4f} | {b1[s]['csd_dT']:+.4f} | "
                         f"{d:+.4f} |")

    # per-task agreement between EDM-only and full
    r0 = {r["task_id"]: r for r in scores[0.0]["rows"]}
    r1 = {r["task_id"]: r for r in scores[1.0]["rows"]}
    common = sorted(set(r0) & set(r1))
    a = np.array([r0[k]["csd_dT"] for k in common])
    b = np.array([r1[k]["csd_dT"] for k in common])
    corr = float(np.corrcoef(a, b)[0, 1]) if len(common) > 2 else float("nan")
    same_sign = int((np.sign(a) == np.sign(b)).sum())
    lp0 = np.array([r0[k]["lpips_to_input"] for k in common])
    lp1 = np.array([r1[k]["lpips_to_input"] for k in common])
    dc0 = np.array([r0[k]["dino_c"] for k in common])
    dc1 = np.array([r1[k]["dino_c"] for k in common])
    lines += ["", "## 逐任务一致性（EDM-only vs full）", "",
              f"- 任务数：{len(common)}",
              f"- 逐任务 ΔT 相关系数：**{corr:.4f}**",
              f"- ΔT 同号任务数：{same_sign}/{len(common)}",
              f"- 逐任务 ΔT 平均绝对差：{float(np.mean(np.abs(a - b))):.5f}",
              f"- LPIPS 平均差 (full − EDM-only)：{float(np.mean(lp1 - lp0)):+.5f}",
              f"- DINO-C 平均差 (full − EDM-only)：{float(np.mean(dc1 - dc0)):+.5f}",
              ""]

    verdict = []
    if abs(d_learned) < 0.01:
        verdict.append(
            "Δ_learned < 0.01：**在当前 Quick200 任务与 CSD 指标下**，学习残差没有提供"
            "明显的目标风格净收益。这**不构成**'网络只学会了统计迁移'的证明——"
            "残差可能学到了其他变化而未能转化为目标风格收益，需结合逐任务差异与内容指标判断。")
    else:
        verdict.append(f"Δ_learned = {d_learned:+.4f}，学习残差有可测净贡献（方向见分目标表）。")
    hard = [s for s in ("Rococo", "Early_Renaissance", "Ukiyo_e")
            if s in b0 and s in b1]
    hard_pos = [s for s in hard if b1[s]["csd_dT"] > 0]
    if not hard_pos:
        verdict.append(f"三个困难目标（{', '.join(hard)}）在 full 模型下 ΔT 仍全部为负。")
    lines += ["## 判读", ""] + [f"- {v}" for v in verdict]

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "p2_edm_control.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_root / "p2_edm_control.json").write_text(
        json.dumps({"scales": SCALES, "delta_learned_dT": d_learned,
                    "per_task_dT_corr": corr, "same_sign": same_sign,
                    "n_tasks": len(common),
                    "overall": {str(sc): overall(sc) for sc in SCALES},
                    "by_target": {str(sc): by_target(sc) for sc in SCALES}},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
