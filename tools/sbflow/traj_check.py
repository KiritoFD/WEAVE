"""Separate 'the supervision target is wrong' from 'train/infer trajectories differ'.

The differentiable ODE that produces L_SWD endpoints uses `--ode-steps-train 2` with
Euler steps, while evaluation uses 4-step Heun. If a distribution-supervision arm looks
bad, we must know whether the loss pointed at a different place than inference lands.

This generates and scores the SAME checkpoint under both trajectories and reports the
difference. If |dT(2-Euler) - dT(4-Heun)| is the same order as the arm's effect, the
trajectory mismatch is a plausible explanation and the supervision should be judged
under the matching trajectory instead.

Usage
    python -m sbflow.traj_check --ckpt runs/blog_sbflow/v11/d1/model_step4000.pt
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
TRAJ = [("e2", 2, "euler"), ("h4", 4, "heun")]


def run(cmd, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as f:
        f.write("\n>>> " + " ".join(map(str, cmd)) + "\n")
        f.flush()
        r = subprocess.run([str(c) for c in cmd], cwd=str(ROOT), stdout=f,
                           stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"rc={r.returncode}; see {log}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-root", default="runs/blog_sbflow/v11/traj")
    ap.add_argument("--tag", default="")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    ck = Path(args.ckpt)
    if not ck.is_absolute():
        ck = ROOT / ck
    if not ck.exists():
        raise SystemExit(f"missing {ck}")
    tag = args.tag or ck.parent.name
    root = ROOT / args.out_root / tag

    scores = {}
    for name, steps, method in TRAJ:
        gen = root / f"gen_{name}"
        score = root / f"score_{name}"
        if not (gen / "manifest.json").exists():
            run([args.python, "-m", "sbflow.gen5", "--repo-root", ".", "--ckpt", str(ck),
                 "--out", str(gen), "--subset", "quick200", "--steps", str(steps),
                 "--method", method, "--batch", "16"], root / "logs" / f"{name}_gen.log")
        if not (score / "style_scores.json").exists():
            run([args.python, "-m", "sbflow.score_cpu", "--repo-root", ".", "--gen",
                 str(gen), "--out", str(score), "--batch", "8",
                 "--threads", str(args.threads)], root / "logs" / f"{name}_score.log")
        scores[name] = json.loads((score / "style_scores.json").read_text())

    lines = [f"# 轨迹一致性检查：{tag}", "",
             f"checkpoint: `{ck.relative_to(ROOT)}`", "",
             "| 轨迹 | Top-1 | Margin | ΔT | ΔS | SWD | DINO-C |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, steps, method in TRAJ:
        o = scores[name]["aggregate"]["overall"]
        lines.append(f"| {steps}-step {method} | {o['target_top1']:.4f} | "
                     f"{o['target_margin']:+.4f} | {o['csd_dT']:+.4f} | "
                     f"{o['csd_dS']:+.4f} | {o['csd_swd']:.5f} | {o['dino_c']:.4f} |")

    a = scores["e2"]["aggregate"]["overall"]
    b = scores["h4"]["aggregate"]["overall"]
    dd = {k: b[k] - a[k] for k in ("target_top1", "target_margin", "csd_dT",
                                   "csd_dS", "csd_swd", "dino_c")}
    lines += ["", "## 差异（4-step Heun − 2-step Euler）", "",
              "| 指标 | 差值 |", "|---|---:|"]
    for k, v in dd.items():
        lines.append(f"| {k} | {v:+.5f} |")

    # per-task agreement
    r1 = {r["task_id"]: r for r in scores["e2"]["rows"]}
    r2 = {r["task_id"]: r for r in scores["h4"]["rows"]}
    common = sorted(set(r1) & set(r2))
    x = np.array([r1[k]["csd_dT"] for k in common])
    y = np.array([r2[k]["csd_dT"] for k in common])
    corr = float(np.corrcoef(x, y)[0, 1]) if len(common) > 2 else float("nan")
    lines += ["", f"- 逐任务 ΔT 相关：**{corr:.4f}**",
              f"- 逐任务 ΔT 平均绝对差：{float(np.mean(np.abs(x - y))):.5f}",
              "",
              "判读：若该平均绝对差与某组的效应同量级，则该组的结论必须"
              "在匹配轨迹下重新评估，不能直接判为'监督目标无效'。"]

    root.mkdir(parents=True, exist_ok=True)
    (root / "traj_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "traj_check.json").write_text(
        json.dumps({"checkpoint": str(ck), "diff_heun_minus_euler": dd,
                    "per_task_dT_corr": corr,
                    "per_task_dT_mad": float(np.mean(np.abs(x - y)))},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
