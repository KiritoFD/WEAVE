"""Checkpoint-trend evaluation: does style quality degrade while loss keeps falling?

Execution book section 6 explicitly requires not judging by the last checkpoint only.
For the main baseline (and optionally any arm), this evaluates the saved
1000 / 2000 / 3000 / 4000 equivalent-step checkpoints on Quick200 and reports
Top-1 / Margin / dT / dS / SWD per step alongside the training loss.

Designed to run AFTER the main suite finishes so it never competes for the GPU.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / "runs/blog_sbflow/suite40"
EQUIVS = [1000, 2000, 3000, 4000]


def run(cmd, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as f:
        f.write("\n>>> " + " ".join(map(str, cmd)) + "\n")
        f.flush()
        r = subprocess.run([str(c) for c in cmd], cwd=str(ROOT), stdout=f,
                           stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"rc={r.returncode}; see {log}")


def loss_at(exp: Path, equiv: int) -> float | None:
    p = exp / "train_summary.json"
    if not p.exists():
        return None
    hist = json.loads(p.read_text()).get("loss_history", [])
    # equivalent step == absolute step for the baseline (B512 / equiv_batch 512)
    for h in hist:
        if h.get("step") == equiv:
            return float(h.get("loss"))
    return None


def evaluate(aid: str, python: str, threads: int) -> list:
    exp = SUITE / f"exp{aid}"
    rows = []
    for eq in EQUIVS:
        ck = exp / f"model_equiv{eq}.pt"
        if not ck.exists():
            print(f"  {aid} equiv{eq}: checkpoint missing, skip", flush=True)
            continue
        gen = SUITE / f"trend_gen_{aid}_{eq}"
        score = SUITE / f"trend_score_{aid}_{eq}"
        if not (gen / "manifest.json").exists():
            run([python, "-m", "sbflow.gen5", "--repo-root", ".", "--ckpt", str(ck),
                 "--out", str(gen), "--subset", "quick200", "--steps", "4",
                 "--method", "heun", "--batch", "16"],
                SUITE / "logs" / f"trend_{aid}_{eq}_gen.log")
        if not (score / "style_scores.json").exists():
            run([python, "-m", "sbflow.score_cpu", "--repo-root", ".",
                 "--gen", str(gen), "--out", str(score), "--batch", "8",
                 "--threads", str(threads)],
                SUITE / "logs" / f"trend_{aid}_{eq}_score.log")
        o = json.loads((score / "style_scores.json").read_text())["aggregate"]["overall"]
        rows.append({"equiv_step": eq, "loss": loss_at(exp, eq),
                     "top1": o["target_top1"], "margin": o["target_margin"],
                     "dT": o["csd_dT"], "dS": o["csd_dS"], "swd": o["csd_swd"],
                     "dino_c": o["dino_c"]})
        print(f"  {aid} equiv{eq}: loss={rows[-1]['loss']} top1={o['target_top1']:.4f} "
              f"margin={o['target_margin']:+.4f} dT={o['csd_dT']:+.4f}", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="01",
                    help="comma-separated arms to trend-test, default only the baseline")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    out = {"arms": {}}
    for aid in arms:
        print(f"[trend] arm {aid}", flush=True)
        rows = evaluate(aid, args.python, args.threads)
        out["arms"][aid] = rows
        if len(rows) >= 2:
            first, last = rows[0], rows[-1]
            degraded = (last["top1"] < first["top1"] or last["margin"] < first["margin"]
                        or last["dT"] < first["dT"])
            out["arms"][aid + "_verdict"] = {
                "style_degraded_while_trained_longer": degraded,
                "first": first["equiv_step"], "last": last["equiv_step"],
            }
            print(f"  verdict: {'DEGRADED' if degraded else 'no degradation detected'}",
                  flush=True)
    p = Path(args.out) if args.out else SUITE / "checkpoint_trend.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 中间 checkpoint 趋势（执行书第 6 节）", "",
             "| arm | equiv step | train loss | Top-1 | Margin | ΔT | ΔS | SWD | DINO-C |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for aid, rows in out["arms"].items():
        if not isinstance(rows, list):
            continue
        for r in rows:
            lines.append(f"| {aid} | {r['equiv_step']} | "
                         f"{r['loss'] if r['loss'] is not None else '—'} | "
                         f"{r['top1']:.4f} | {r['margin']:+.4f} | {r['dT']:+.4f} | "
                         f"{r['dS']:+.4f} | {r['swd']:.4f} | {r['dino_c']:.4f} |")
    (SUITE / "checkpoint_trend.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[trend] saved {p}", flush=True)


if __name__ == "__main__":
    main()
