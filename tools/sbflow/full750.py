"""Full750 official evaluation for the suite40 finalists.

The frozen `tasks_750.json` has 600 cross-style + 150 same-style tasks. Full750 means:
  * run the generator on the 600 cross-style tasks (gen5 --subset full600),
  * treat the 150 same-style tasks as STRICT identity (no network call),
  * score both with the same frozen CPU evaluator, and
  * keep the 600 / 150 / 750 aggregates separate so identity cannot hide a bad
    cross-style result.

Three distinct winning mechanisms plus the exp01 baseline are selected from Quick200
using the execution book's priority: 20-direction macro margin/dT first, then the
three difficult targets, then per-target SWD/MMD, with DINO-C/LPIPS as guards.
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
SUITE = ROOT / "runs/blog_sbflow/suite40"
HARD = ["Rococo", "Early_Renaissance", "Ukiyo_e"]
MECHANISM_GROUP = {"A": "batch", "B": "flow", "C": "condition", "D": "distribution",
                   "E": "online", "F": "structure", "G": "ot"}


def quick_score(aid: str) -> dict | None:
    p = SUITE / f"score_{aid}_quick200" / "style_scores.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def rank_key(aid: str, d: dict) -> tuple:
    """Execution-book priority, expressed as one sortable tuple (larger is better)."""
    a = d["aggregate"]
    by_dir = a["by_direction"]
    macro_margin = float(np.mean([v["target_margin"] for v in by_dir.values()]))
    macro_dt = float(np.mean([v["csd_dT"] for v in by_dir.values()]))
    hard_dt = float(np.mean([a["by_target"][s]["csd_dT"] for s in HARD
                             if s in a["by_target"]]))
    swd = -float(np.mean([a["by_target"][s]["csd_swd"] for s in a["by_target"]]))
    mmd = -float(np.mean([a["by_target"][s]["csd_mmd2_unbiased"] for s in a["by_target"]]))
    # DINO-C / LPIPS are structural guards, not primary objectives.
    guard = float(a["overall"]["dino_c"]) - float(a["overall"]["lpips_to_input"]) * 0.1
    return (macro_margin, macro_dt, hard_dt, swd, mmd, guard)


def select_finalists(top_k: int = 3) -> List[str]:
    state = json.loads((SUITE / "suite_state.json").read_text())
    ranked = []
    for aid, rec in state.items():
        e = rec.get("eval")
        if not e or "target_top1" not in e or rec.get("status") != "done":
            continue
        d = quick_score(aid)
        if d is None:
            continue
        ranked.append((rank_key(aid, d), aid, rec.get("group", "")))
    ranked.sort(reverse=True)
    chosen, groups = [], set()
    for _, aid, group in ranked:
        g = MECHANISM_GROUP.get(group, group)
        if g in groups or aid == "01":
            continue
        chosen.append(aid)
        groups.add(g)
        if len(chosen) >= top_k:
            break
    return ["01"] + chosen


def identity_rows() -> List[dict]:
    """Strict identity: no generated image, no CSD shift, perfect content match."""
    raw = json.loads((ROOT / "eval_protocol/tasks_750.json").read_text())
    out = []
    for t in raw["tasks"]:
        if t.get("is_cross", t["source_style"] != t["target_style"]):
            continue
        out.append({"task_id": t["task_id"], "source": t["source_style"],
                    "target": t["target_style"],
                    "direction": f"{t['source_style']}->{t['target_style']}",
                    "variant": "generated", "csd_dT": 0.0, "csd_dS": 0.0,
                    "target_margin": 0.0, "target_top1": 0, "lpips_to_input": 0.0,
                    "dino_c": 1.0, "csd_embedding": None, "strict_identity": True})
    return out


def run(cmd: List[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as f:
        f.write("\n>>> " + " ".join(map(str, cmd)) + "\n")
        f.flush()
        r = subprocess.run([str(c) for c in cmd], cwd=str(ROOT), stdout=f,
                           stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"failed rc={r.returncode}: see {log}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="", help="override finalists, e.g. 01,04,08")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--score-threads", type=int, default=4)
    ap.add_argument("--status-only", action="store_true")
    args = ap.parse_args()

    finalists = [a.strip() for a in args.arms.split(",") if a.strip()] \
        if args.arms else select_finalists(args.top_k)
    print(f"[full750] finalists={finalists}", flush=True)
    if args.status_only:
        return

    from tools.sbflow.score_cpu import CpuFeatures, _reference_csd, aggregate, mmd2, swd
    feat = CpuFeatures(ROOT, args.score_threads)
    refs = _reference_csd(ROOT, feat,
                          ROOT / "eval_protocol/cache/references_v1_csd_cpu.pt")
    gamma = float(refs["rbf_gamma"])

    results = {}
    for aid in finalists:
        out_dir = SUITE / f"full750_{aid}"
        gen_dir = SUITE / f"gen_{aid}_full600"
        score_dir = SUITE / f"score_{aid}_full600"
        ck = SUITE / f"exp{aid}" / f"model_step{_final_step(aid)}.pt"
        if not ck.exists():
            print(f"[full750] {aid}: missing {ck}, skip", flush=True)
            continue
        if not (gen_dir / "manifest.json").exists():
            run([args.python, "-m", "sbflow.gen5", "--repo-root", ".",
                 "--ckpt", str(ck), "--out", str(gen_dir), "--subset", "full600",
                 "--steps", "4", "--method", "heun", "--batch", "16"],
                SUITE / "logs" / f"{aid}_full600_gen.log")
        if not (score_dir / "style_scores.json").exists():
            run([args.python, "-m", "sbflow.score_cpu", "--repo-root", ".",
                 "--gen", str(gen_dir), "--out", str(score_dir),
                 "--batch", "8", "--threads", str(args.score_threads)],
                SUITE / "logs" / f"{aid}_full600_score.log")
        cross = json.loads((score_dir / "style_scores.json").read_text())
        cross_rows = cross["rows"]
        same_rows = identity_rows()
        # Official 750 aggregate: real 600 cross-style metrics + 150 exact identity.
        all_rows = cross_rows + same_rows
        keys = ("csd_dT", "csd_dS", "target_margin", "target_top1",
                "lpips_to_input", "dino_c")
        overall750 = {k: float(np.mean([r[k] for r in all_rows])) for k in keys}
        cross600 = {k: float(np.mean([r[k] for r in cross_rows])) for k in keys}
        same150 = {k: float(np.mean([r[k] for r in same_rows])) for k in keys}
        results[aid] = {"cross600": cross600, "same150": same150,
                        "official750": overall750,
                        "by_direction": cross["aggregate"]["by_direction"],
                        "by_target": cross["aggregate"]["by_target"],
                        "checkpoint": str(ck), "git_sha": _git()}
        print(f"[full750] {aid}: cross600 top1={cross600['target_top1']:.4f} "
              f"margin={cross600['target_margin']:+.4f} dT={cross600['csd_dT']:+.4f} | "
              f"official750 top1={overall750['target_top1']:.4f}", flush=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "results.json").write_text(
            json.dumps(results[aid], ensure_ascii=False, indent=2), encoding="utf-8")
    (SUITE / "full750_results.json").write_text(
        json.dumps({"finalists": finalists, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    _write_summary(finalists, results)
    print(f"[full750] saved {SUITE / 'full750_results.json'}", flush=True)


def _final_step(aid: str) -> int:
    cfg = json.loads((ROOT / "configs/sbflow" / f"exp{aid}.json").read_text())
    return cfg["steps"] + (3000 if cfg.get("fork_from") else 0)


def _git() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT),
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def _write_summary(finalists, results):
    lines = ["# Full750 终评", "",
             "正式口径：600 跨风格生成 + 150 同风格严格 Identity，两者分开报告。", "",
             "| ID | cross600 Top-1 | cross600 Margin | cross600 ΔT | official750 Top-1 | "
             "Rococo ΔT | EarlyRen ΔT | Ukiyo-e ΔT | NFE |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for aid in finalists:
        if aid not in results:
            continue
        r = results[aid]
        c = r["cross600"]
        bt = r["by_target"]
        lines.append(f"| {aid} | {c['target_top1']:.4f} | {c['target_margin']:+.4f} | "
                     f"{c['csd_dT']:+.4f} | {r['official750']['target_top1']:.4f} | "
                     f"{bt.get('Rococo', {}).get('csd_dT', float('nan')):+.4f} | "
                     f"{bt.get('Early_Renaissance', {}).get('csd_dT', float('nan')):+.4f} | "
                     f"{bt.get('Ukiyo_e', {}).get('csd_dT', float('nan')):+.4f} | 16 |")
    (SUITE / "full750_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
