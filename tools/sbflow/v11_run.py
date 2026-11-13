"""V11 P1 runner: four corrected algorithm arms, strictly serial, then compare.

Control: the existing plain-bridge 4000-equivalent checkpoint
(`suite40/exp01/model_equiv4000.pt`). It is NOT retrained because the bidir objective was
untouched by every V11 fix. Its Quick200 rows are rescored with the corrected per-target
aggregate so the comparison uses the same metric.

Usage
    python -m sbflow.v11_run --arms d1,d2,e1,e2
    python -m sbflow.v11_run --status-only
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CFG = ROOT / "configs" / "v11"
CACHE_ROOT = "/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16"
CACHES = [f"{CACHE_ROOT}/{f}" for f in
          ("00_Early_Renaissance.pt", "01_Impressionism.pt", "02_Minimalism.pt",
           "03_Rococo.pt", "04_Ukiyo_e.pt")]
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
HARD = ["Rococo", "Early_Renaissance", "Ukiyo_e"]


def run(cmd: List[str], log: Path) -> float:
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with log.open("a", encoding="utf-8") as f:
        f.write("\n>>> " + " ".join(map(str, cmd)) + "\n")
        f.flush()
        r = subprocess.run([str(c) for c in cmd], cwd=str(ROOT), stdout=f,
                           stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"rc={r.returncode}; see {log}")
    return time.perf_counter() - t0


def resolve(aid: str) -> dict:
    c = json.loads((CFG / f"{aid}.json").read_text(encoding="utf-8"))
    out = ROOT / "runs/blog_sbflow/v11" / aid
    c["output"] = str(out)
    c["resume"] = str(ROOT / c["resume"]) if not Path(c["resume"]).is_absolute() else c["resume"]
    if c.get("style_enc_ckpt"):
        c["style_enc_ckpt"] = str(ROOT / c["style_enc_ckpt"])
    return c


def control_aggregate() -> dict:
    """Rescore the existing control Quick200 rows with the corrected aggregate."""
    from tools.sbflow.score_cpu import aggregate
    refs = _refs()
    p = ROOT / "runs/blog_sbflow/suite40/score_01_quick200/style_scores.json"
    d = json.loads(p.read_text())
    return aggregate(d["rows"], refs, float(refs["rbf_gamma"]))


def _refs():
    import torch
    return torch.load(ROOT / "eval_protocol/cache/references_v1_csd_cpu.pt",
                      map_location="cpu", weights_only=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="d1,d2,e1,e2")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--status-only", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    root = ROOT / "runs/blog_sbflow/v11"
    (root / "logs").mkdir(parents=True, exist_ok=True)
    state_file = root / "p1_state.json"
    state: Dict[str, dict] = json.loads(state_file.read_text()) if state_file.exists() else {}
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    if args.status_only:
        for a in sorted(state):
            v = state[a]
            e = v.get("eval") or {}
            extra = (f"top1={e['target_top1']:.4f} margin={e['target_margin']:+.4f} "
                     f"dT={e['csd_dT']:+.4f}") if "target_top1" in e else ""
            print(f"  {a:3s} {v.get('status', '?'):18s} {extra}")
        return

    for aid in arms:
        cfg = resolve(aid)
        cp = root / f"{aid}.json"
        cp.write_text(json.dumps(cfg, ensure_ascii=False, indent=2, sort_keys=True),
                      encoding="utf-8")
        rec = state.get(aid, {"arm": aid, "note": cfg.get("note", "")})
        state[aid] = rec
        try:
            if not args.skip_train:
                rec["status"] = "training"
                state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
                rec["train_seconds"] = round(
                    run([args.python, "-m", "sbflow.train5", "--config", str(cp),
                         "--caches", *CACHES], root / "logs" / f"{aid}_train.log"), 1)
            ck = root / aid / f"model_step{3000 + cfg['steps']}.pt"
            if not ck.exists():
                raise FileNotFoundError(f"missing final checkpoint {ck}")
            rec["checkpoint"] = str(ck)
            s = root / aid / "train_summary.json"
            if s.exists():
                summ = json.loads(s.read_text())
                rec["sec_per_update"] = summ.get("sec_per_update")
                rec["peak_vram_mib"] = summ.get("peak_vram_mib")
                rec["exposures"] = summ.get("endpoint_exposures")
                hist = summ.get("loss_history", [])
                for key in ("dist_grad_norm", "bridge_grad_norm", "rho_swd_over_bridge"):
                    vals = [h[key] for h in hist if h.get(key) is not None]
                    if vals:
                        rec[key] = vals[0]
            gen = root / f"gen_{aid}_quick200"
            score = root / f"score_{aid}_quick200"
            if not (gen / "manifest.json").exists():
                run([args.python, "-m", "sbflow.gen5", "--repo-root", ".", "--ckpt", str(ck),
                     "--out", str(gen), "--subset", "quick200", "--steps", "4",
                     "--method", "heun", "--batch", "16"],
                    root / "logs" / f"{aid}_gen.log")
            if not (score / "style_scores.json").exists():
                run([args.python, "-m", "sbflow.score_cpu", "--repo-root", ".", "--gen",
                     str(gen), "--out", str(score), "--batch", "8",
                     "--threads", str(args.threads)], root / "logs" / f"{aid}_score.log")
            agg = json.loads((score / "style_scores.json").read_text())["aggregate"]
            o = agg["overall"]
            rec["eval"] = {k: o[k] for k in ("target_top1", "target_margin", "csd_dT",
                                             "csd_dS", "csd_swd", "csd_mmd2_unbiased",
                                             "dino_c", "lpips_to_input", "n")}
            rec["by_target"] = agg["by_target"]
            rec["status"] = "done"
            print(f"[v11] {aid} done: top1={o['target_top1']:.4f} "
                  f"margin={o['target_margin']:+.4f} dT={o['csd_dT']:+.4f} "
                  f"rho={rec.get('rho_swd_over_bridge')}", flush=True)
        except Exception as e:  # noqa: BLE001
            rec["status"] = "failed"
            rec["error"] = f"{type(e).__name__}: {e}"
            print(f"[v11] {aid} FAILED: {rec['error']}", flush=True)
        state[aid] = rec
        state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- comparison against the control ----
    try:
        ctrl = control_aggregate()
    except Exception as e:  # noqa: BLE001
        print(f"[v11] control rescore unavailable: {e}")
        ctrl = None

    lines = ["# V11 P1：修正后算法对照", "",
             "全部从 `exp01/model_equiv3000.pt` 分叉，追加 1000 步，"
             "固定 Batch512 五风格混合采样、4000 等效 LR 周期（无 warmup 重启）。",
             "对照为同一 3000-step 前缀续训到 4000 步的普通 Bridge"
             "（`exp01/model_equiv4000.pt`，其 bidir 目标未受任何 V11 修复影响，故不重训）。", ""]
    if ctrl:
        o = ctrl["overall"]
        lines += ["| 组 | Top-1 | Margin | ΔT | ΔS | SWD | MMD | DINO-C | ρ | 训练秒 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                  f"| control(exp01) | {o['target_top1']:.4f} | {o['target_margin']:+.4f} | "
                  f"{o['csd_dT']:+.4f} | {o['csd_dS']:+.4f} | {o['csd_swd']:.5f} | "
                  f"{o['csd_mmd2_unbiased']:.5f} | {o['dino_c']:.4f} | — | — |"]
    for aid in arms:
        rec = state.get(aid, {})
        e = rec.get("eval") or {}
        if "target_top1" not in e:
            lines.append(f"| {aid} | {rec.get('status', '?')} | — | — | — | — | — | — | — | "
                         f"{rec.get('train_seconds', 0):.0f} |")
            continue
        rho = rec.get("rho_swd_over_bridge")
        lines.append(f"| {aid} | {e['target_top1']:.4f} | {e['target_margin']:+.4f} | "
                     f"{e['csd_dT']:+.4f} | {e['csd_dS']:+.4f} | {e['csd_swd']:.5f} | "
                     f"{e['csd_mmd2_unbiased']:.5f} | {e['dino_c']:.4f} | "
                     f"{rho:.3f} | {rec.get('train_seconds', 0):.0f} |"
                     if rho is not None else
                     f"| {aid} | {e['target_top1']:.4f} | {e['target_margin']:+.4f} | "
                     f"{e['csd_dT']:+.4f} | {e['csd_dS']:+.4f} | {e['csd_swd']:.5f} | "
                     f"{e['csd_mmd2_unbiased']:.5f} | {e['dino_c']:.4f} | — | "
                     f"{rec.get('train_seconds', 0):.0f} |")

    if ctrl:
        lines += ["", "## 相对对照的变化（正值=更好）", "",
                  "| 组 | ΔTop-1 | ΔMargin | ΔΔT | ΔSWD | Rococo ΔT | ER ΔT | Ukiyo ΔT |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        co = ctrl["overall"]
        cb = ctrl["by_target"]
        for aid in arms:
            rec = state.get(aid, {})
            e = rec.get("eval") or {}
            if "target_top1" not in e:
                continue
            bt = rec.get("by_target") or {}
            def g(s, k):
                return bt.get(s, {}).get(k, float("nan"))
            lines.append(
                f"| {aid} | {e['target_top1'] - co['target_top1']:+.4f} | "
                f"{e['target_margin'] - co['target_margin']:+.4f} | "
                f"{e['csd_dT'] - co['csd_dT']:+.4f} | "
                f"{e['csd_swd'] - co['csd_swd']:+.5f} | "
                f"{g('Rococo', 'csd_dT'):+.4f} | "
                f"{g('Early_Renaissance', 'csd_dT'):+.4f} | "
                f"{g('Ukiyo_e', 'csd_dT'):+.4f} |")
        lines += ["", "判定要求：ΔT 与 Margin 同向改善，且三个困难目标要有真实收益；"
                      "仅总体 SWD 下降不算有效。"]

    (root / "p1_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "p1_results.json").write_text(
        json.dumps({"control": ctrl, "arms": state}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
