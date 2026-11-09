"""Serial executor for the 12h / 40-arm five-style Schrodinger Flow suite.

One GPU process at a time; CPU scoring is spawned in the background with a bounded
thread count and folded back at the end. State is checkpointed after every arm, so a
crash or SSH disconnect can be resumed with --resume-suite.

Usage
    python -m sbflow.suite40 --arms core
    python -m sbflow.suite40 --arms all --resume-suite
    python -m sbflow.suite40 --status-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs" / "sbflow"
CACHE_ROOT = "/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16"
CACHES = [f"{CACHE_ROOT}/{f}" for f in
          ("00_Early_Renaissance.pt", "01_Impressionism.pt", "02_Minimalism.pt",
           "03_Rococo.pt", "04_Ukiyo_e.pt")]
CORE = ["01", "04", "08", "10", "24", "27", "32", "35", "37"]


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT),
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def sha256(o) -> str:
    return hashlib.sha256(json.dumps(o, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def select(spec: str) -> List[str]:
    if spec == "all":
        return [f"{i:02d}" for i in range(1, 41)]
    if spec == "core":
        return list(CORE)
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(f"{i:02d}" for i in range(int(lo), int(hi) + 1))
        else:
            out.append(f"{int(part):02d}")
    return out


def run(cmd: List[str], log: Path) -> float:
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with log.open("a", encoding="utf-8") as f:
        f.write("\n>>> " + " ".join(map(str, cmd)) + "\n")
        f.flush()
        r = subprocess.run([str(c) for c in cmd], cwd=str(ROOT), stdout=f,
                           stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"command failed rc={r.returncode}; see {log}")
    return time.perf_counter() - t0


def spawn(cmd: List[str], log: Path):
    log.parent.mkdir(parents=True, exist_ok=True)
    f = log.open("a", encoding="utf-8")
    f.write("\n>>> " + " ".join(map(str, cmd)) + "\n")
    f.flush()
    return subprocess.Popen([str(c) for c in cmd], cwd=str(ROOT), stdout=f,
                            stderr=subprocess.STDOUT)


def resolve(cfg: dict, root: Path) -> dict:
    c = dict(cfg)
    c["experiment_id"] = f"exp{cfg['id']}"
    c["output"] = str(root / f"exp{cfg['id']}")
    if c.get("fork_from"):
        src = c["fork_from"]
        src_id = src.replace("exp", "")
        # Prefer the stable equivalent-step checkpoint; fall back to absolute step.
        cand = root / f"exp{src_id}" / "model_equiv3000.pt"
        if not cand.exists() and cfg.get("equiv_batch") == 512:
            cand = root / f"exp{src_id}" / "model_step3000.pt"
        c["resume"] = str(cand)
    else:
        c.pop("resume", None)
    if c.get("style_enc_ckpt"):
        c["style_enc_ckpt"] = str(root / "shared_style_encoder.pt")
    return c


def status(state: dict, root: Path) -> None:
    from collections import Counter

    print(f"=== suite40 status ({root}) ===")
    print("  counts:", dict(Counter(v.get("status") for v in state.values())))
    for k in sorted(state):
        v = state[k]
        e = v.get("eval") or {}
        if "target_top1" in e:
            print(f"  {k} {v['status']:18s} top1={e['target_top1']:.4f} "
                  f"margin={e['target_margin']:+.4f} dT={e['csd_dT']:+.4f} "
                  f"train={v.get('train_seconds', 0):.0f}s")
        else:
            print(f"  {k} {v['status']:18s} {v.get('error', '')[:80]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/blog_sbflow/suite40")
    ap.add_argument("--arms", default="core")
    ap.add_argument("--status-only", action="store_true")
    ap.add_argument("--resume-suite", action="store_true")
    ap.add_argument("--only-train", action="store_true")
    ap.add_argument("--eval-subset", default="quick200")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--score-threads", type=int, default=4)
    args = ap.parse_args()

    root = ROOT / args.root if not Path(args.root).is_absolute() else Path(args.root)
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    state_file = root / "suite_state.json"
    state: Dict[str, dict] = json.loads(state_file.read_text()) if state_file.exists() else {}
    if args.status_only:
        status(state, root)
        return

    ids = select(args.arms)
    pending = []
    print(f"[suite40] arms={ids} eval={args.eval_subset} git={git_sha()} root={root}", flush=True)

    # Pretrain the single shared frozen style encoder before any learned-SWD arm.
    enc = root / "shared_style_encoder.pt"
    if any(json.loads((CONFIGS / f"exp{i}.json").read_text()).get("style_enc_ckpt") for i in ids) \
            and not enc.exists():
        print("[suite40] pretraining shared style encoder", flush=True)
        run([args.python, "-m", "sbflow.pretrain_style_encoder", "--out", str(enc),
             "--epochs", "60"], root / "logs/style_encoder.log")

    for aid in ids:
        cfg = json.loads((CONFIGS / f"exp{aid}.json").read_text(encoding="utf-8"))
        rcfg = resolve(cfg, root)
        cfg_path = root / "configs" / f"exp{aid}.json"
        cfg_path.write_text(json.dumps(rcfg, ensure_ascii=False, indent=2, sort_keys=True),
                            encoding="utf-8")
        rec = {"id": aid, "group": cfg["group"], "note": cfg["note"],
               "git_sha": git_sha(), "config_sha256": sha256(rcfg), "config": rcfg,
               "status": "training", "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
        state[aid] = rec
        state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            if rcfg.get("resume") and not Path(rcfg["resume"]).exists():
                raise FileNotFoundError(f"fork checkpoint missing: {rcfg['resume']}")
            end_step = rcfg["steps"] + (3000 if rcfg.get("resume") else 0)
            final = root / f"exp{aid}" / f"model_step{end_step}.pt"
            if not (args.resume_suite and final.exists()):
                cmd = [args.python, "-m", "sbflow.train5", "--config", str(cfg_path),
                       "--caches", *CACHES]
                rec["train_seconds"] = round(run(cmd, root / "logs" / f"{aid}_train.log"), 1)
            summary = root / f"exp{aid}" / "train_summary.json"
            if summary.exists():
                s = json.loads(summary.read_text())
                rec.update({"train_seconds": s.get("train_seconds", rec.get("train_seconds")),
                            "peak_vram_mib": s.get("peak_vram_mib"),
                            "exposures": s.get("endpoint_exposures"),
                            "sec_per_update": s.get("sec_per_update")})
            rec["checkpoint"] = str(final)
            rec["status"] = "trained"
            if not args.only_train:
                gen_dir = root / f"gen_{aid}_{args.eval_subset}"
                if not (gen_dir / "manifest.json").exists():
                    run([args.python, "-m", "sbflow.gen5", "--repo-root", ".",
                         "--ckpt", str(final), "--out", str(gen_dir),
                         "--subset", args.eval_subset, "--steps", "4",
                         "--method", "heun", "--batch", "16"],
                        root / "logs" / f"{aid}_gen.log")
                score_dir = root / f"score_{aid}_{args.eval_subset}"
                score_cmd = [args.python, "-m", "sbflow.score_cpu", "--repo-root", ".",
                             "--gen", str(gen_dir), "--out", str(score_dir),
                             "--batch", "8", "--threads", str(args.score_threads)]
                if not (score_dir / "style_scores.json").exists():
                    p = spawn(score_cmd, root / "logs" / f"{aid}_score.log")
                    pending.append((aid, p, score_dir))
                    rec["status"] = "awaiting_score"
                    raise _Deferred()
                state[aid] = rec
                state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
            rec["status"] = "done"
        except _Deferred:
            pass
        except Exception as e:  # noqa: BLE001
            rec["status"] = "failed"
            rec["error"] = f"{type(e).__name__}: {e}"
            print(f"[suite40] {aid} FAILED: {rec['error']}", flush=True)
        finally:
            state[aid] = rec
            state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
            print(f"[suite40] {aid} {rec['status']} train={rec.get('train_seconds', 0):.0f}s",
                  flush=True)

    if pending:
        print(f"[suite40] waiting for {len(pending)} CPU scorer(s)", flush=True)
        for aid, p, score_dir in pending:
            rc = p.wait()
            rec = state[aid]
            f = score_dir / "style_scores.json"
            if rc == 0 and f.exists():
                o = json.loads(f.read_text())["aggregate"]["overall"]
                rec["eval"] = {k: o[k] for k in
                               ("target_top1", "target_margin", "csd_dT", "csd_dS",
                                "csd_swd", "csd_mmd2_unbiased", "dino_c", "n")}
                rec["status"] = "done"
            else:
                rec["status"] = "failed"
                rec["error"] = f"scorer rc={rc}"
            state[aid] = rec
            state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
    write_results(state, root)
    status(state, root)


class _Deferred(Exception):
    pass


def write_results(state: dict, root: Path) -> None:
    (root / "results.json").write_text(json.dumps(state, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    lines = ["# suite40 结果", "",
             "| ID | 状态 | Top-1 | Margin | ΔT | ΔS | SWD | MMD | DINO-C | 训练秒 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k in sorted(state):
        v = state[k]
        e = v.get("eval") or {}
        if "target_top1" in e:
            lines.append(f"| {k} | {v['status']} | {e['target_top1']:.4f} | "
                         f"{e['target_margin']:+.4f} | {e['csd_dT']:+.4f} | "
                         f"{e['csd_dS']:+.4f} | {e['csd_swd']:.4f} | "
                         f"{e['csd_mmd2_unbiased']:.4f} | {e['dino_c']:.4f} | "
                         f"{v.get('train_seconds', 0):.0f} |")
        else:
            lines.append(f"| {k} | {v['status']} | — | — | — | — | — | — | — | "
                         f"{v.get('train_seconds', 0):.0f} |")
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
