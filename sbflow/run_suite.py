"""Serial experiment driver for the 40-arm V10 suite.

For each arm it runs, strictly serially:
  1. `sbflow.train5` (its own output directory, checkpoints every --save-interval)
  2. `sbflow.gen5 --subset quick200` (or full600 for the gate arms)
  3. `sbflow.score_cpu` (pure CPU, does not block the GPU for the next arm)
  4. appends the per-arm record required by docs/bridgev10.md section 7

State is written to `<root>/suite_state.json` after every arm, so the driver can be
restarted and will skip completed arms. Per-arm logs go to `<root>/logs/<id>.log`.

Usage
    python -m sbflow.run_suite --root runs/blog_sbflow/v10 --arms 04 --full600
    python -m sbflow.run_suite --root runs/blog_sbflow/v10 --arms A
    python -m sbflow.run_suite --root runs/blog_sbflow/v10 --arms impl --resume
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from .arms import select

REPO = Path(__file__).resolve().parents[1]
# Canonical SD-VAE pseudo-GT caches (4x64x64, sf=0.18215), 15,545 rows across five styles.
CACHE_ROOT = "/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16"
CACHE_FILES = ["00_Early_Renaissance.pt", "01_Impressionism.pt", "02_Minimalism.pt",
               "03_Rococo.pt", "04_Ukiyo_e.pt"]
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]


def caches() -> List[str]:
    out = [f"{CACHE_ROOT}/{f}" for f in CACHE_FILES]
    missing = [p for p in out if not Path(p).exists()]
    if missing:
        raise SystemExit(f"missing SD-VAE caches: {missing}")
    return out


def run(cmd: List[str], log: Path) -> float:
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with log.open("a", encoding="utf-8") as f:
        f.write("\n>>> " + " ".join(cmd) + "\n")
        f.flush()
        r = subprocess.run(cmd, cwd=str(REPO), stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}); see {log}")
    return time.perf_counter() - t0


def spawn(cmd: List[str], log: Path) -> subprocess.Popen:
    """Start a process without waiting; used for CPU scoring so the GPU is never idle."""
    log.parent.mkdir(parents=True, exist_ok=True)
    f = log.open("a", encoding="utf-8")
    f.write("\n>>> " + " ".join(cmd) + "\n")
    f.flush()
    return subprocess.Popen(cmd, cwd=str(REPO), stdout=f, stderr=subprocess.STDOUT)


def arm_flags(cfg: dict) -> List[str]:
    def flag(name, val):
        key = "--" + name.replace("_", "-")
        if isinstance(val, bool):
            return [key] if val else []
        return [key, str(val)]

    keys = ["objective", "edm", "heads_out", "style_cond", "target", "time_sampling",
            "sampler", "ot_ratio", "ot_cost", "eps", "batch", "steps", "width",
            "depth", "heads", "patch", "warmup", "lr", "ema_decay", "pair_rank",
            "local_conv", "dist_loss", "style_enc", "lambda_dist", "online",
            "online_steps", "online_direction", "online_mix", "online_replay",
            "online_raw_teacher", "online_warmup", "pivot"]
    out: List[str] = []
    for k in keys:
        if k in cfg:
            out.extend(flag(k, cfg[k]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/blog_sbflow/v10")
    ap.add_argument("--arms", default="04")
    ap.add_argument("--full600", action="store_true", help="run full600 instead of quick200")
    ap.add_argument("--eval-subset", default="", help="override: quick200|full600|same150|none")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--only-train", action="store_true")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--async-score", action="store_true", default=True,
                    help="run CPU scoring in the background so the GPU never idles")
    ap.add_argument("--sync-score", dest="async_score", action="store_false")
    ap.add_argument("--score-threads", type=int, default=6)
    args = ap.parse_args()

    root = REPO / args.root
    logs = root / "logs"
    state_file = root / "suite_state.json"
    root.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    state: Dict[str, dict] = {}
    if state_file.exists():
        state = json.loads(state_file.read_text(encoding="utf-8"))
    subset = args.eval_subset or ("full600" if args.full600 else "quick200")

    arms = select(args.arms)
    cache_list = caches()
    pending: List[tuple] = []
    print(f"[suite] {len(arms)} arm(s): {[a['experiment_id'] for a in arms]}", flush=True)
    print(f"[suite] caches={len(cache_list)} eval_subset={subset} root={root}", flush=True)

    for cfg in arms:
        aid = cfg["experiment_id"]
        entry = state.get(aid, {})
        if args.resume and entry.get("status") == "done":
            print(f"[suite] skip {aid} (done)", flush=True)
            continue

        exp_dir = root / f"exp_{aid}"
        ckpt = exp_dir / f"model_step{cfg['steps']}.pt"
        rec = {"experiment_id": aid, "group": cfg["group"], "note": cfg["note"],
               "config": cfg, "git_commit": git_commit(), "seed": 42, "styles": len(STYLES),
               "status": "training", "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
        state[aid] = rec
        state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

        try:
            if not ckpt.exists():
                cmd = [args.python, "-m", "sbflow.train5",
                       "--experiment-id", aid,
                       "--caches", *cache_list,
                       "--output", str(exp_dir),
                       "--seed", "42",
                       "--log-interval", "50",
                       "--save-interval", str(max(100, cfg["steps"] // 4)),
                       *arm_flags(cfg)]
                secs = run(cmd, logs / f"{aid}_train.log")
                summary = json.loads((exp_dir / "train_summary.json").read_text())
                rec.update({"train_seconds": round(secs, 1),
                            "sec_per_update": summary.get("sec_per_update"),
                            "peak_vram_mib": summary.get("peak_vram_mib"),
                            "endpoint_exposures": summary.get("endpoint_exposures"),
                            "n_params": summary.get("n_params")})
            else:
                print(f"[suite] {aid}: checkpoint exists, skipping training", flush=True)
            rec["checkpoint"] = str(ckpt.relative_to(REPO))

            if not args.only_train and subset != "none":
                gen_dir = root / f"gen_{aid}_{subset}"
                if not (gen_dir / "manifest.json").exists():
                    cmd = [args.python, "-m", "sbflow.gen5", "--repo-root", ".",
                           "--ckpt", str(ckpt), "--out", str(gen_dir),
                           "--subset", subset, "--steps", "4", "--method", "heun",
                           "--batch", "16"]
                    run(cmd, logs / f"{aid}_gen.log")
                score_dir = root / f"score_{aid}_{subset}"
                score_cmd = [args.python, "-m", "sbflow.score_cpu", "--repo-root", ".",
                             "--gen", str(gen_dir), "--out", str(score_dir),
                             "--batch", "8", "--threads", str(args.score_threads)]
                if not (score_dir / "style_scores.json").exists():
                    if args.async_score:
                        # CPU scoring runs in the background so the GPU starts the next
                        # arm immediately; the driver waits for these at the very end.
                        p = spawn(score_cmd, logs / f"{aid}_score.log")
                        pending.append((aid, p, score_dir))
                        rec["eval"] = {"subset": subset, "status": "scoring_async",
                                       "pid": p.pid}
                        raise _Deferred()
                    run(score_cmd, logs / f"{aid}_score.log")
                sc = json.loads((score_dir / "style_scores.json").read_text())
                o = sc["aggregate"]["overall"]
                rec["eval"] = {"subset": subset, **{k: o[k] for k in
                                                   ("target_top1", "target_margin", "csd_dT",
                                                    "csd_dS", "csd_mmd2_unbiased", "csd_swd",
                                                    "dino_c", "n")}}
            rec["status"] = "done"
            rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        except _Deferred:
            rec["status"] = "trained_awaiting_score"
        except Exception as e:  # noqa: BLE001
            rec["status"] = "failed"
            rec["error"] = f"{type(e).__name__}: {e}"
            print(f"[suite] {aid} FAILED: {rec['error']}", flush=True)
        finally:
            state[aid] = rec
            state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
            if rec.get("eval"):
                e = rec["eval"]
                print(f"[suite] {aid} {rec['status']:6s} top1={e['target_top1']:.4f} "
                      f"margin={e['target_margin']:+.4f} dT={e['csd_dT']:+.4f} "
                      f"dS={e['csd_dS']:+.4f} swd={e['csd_swd']:.4f} "
                      f"train={rec.get('train_seconds', 0):.0f}s", flush=True)
            else:
                print(f"[suite] {aid} {rec['status']}", flush=True)

    done = [k for k, v in state.items() if v.get("status") == "done"]
    print(f"[suite] finished. done={len(done)}/{len(state)} state={state_file}", flush=True)

    # Collect deferred CPU scoring, then fold the metrics back into the state.
    if pending:
        print(f"[suite] waiting for {len(pending)} background scorer(s) ...", flush=True)
        for aid, p, score_dir in pending:
            rc = p.wait()
            rec = state.get(aid, {})
            f = score_dir / "style_scores.json"
            if rc == 0 and f.exists():
                sc = json.loads(f.read_text())
                o = sc["aggregate"]["overall"]
                rec["eval"] = {"subset": sc["protocol"].get("subset"),
                               **{k: o[k] for k in ("target_top1", "target_margin", "csd_dT",
                                                    "csd_dS", "csd_mmd2_unbiased", "csd_swd",
                                                    "dino_c", "n")}}
                rec["status"] = "done"
                rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                e = rec["eval"]
                print(f"[suite] {aid} scored top1={e['target_top1']:.4f} "
                      f"margin={e['target_margin']:+.4f} dT={e['csd_dT']:+.4f}", flush=True)
            else:
                rec["status"] = "failed"
                rec["error"] = f"scorer rc={rc}"
                print(f"[suite] {aid} scoring FAILED rc={rc}", flush=True)
            state[aid] = rec
        state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


class _Deferred(Exception):
    """Internal: this arm was trained and its CPU scoring continues in the background."""


def git_commit() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO),
                           capture_output=True, text=True)
        return r.stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


if __name__ == "__main__":
    main()
