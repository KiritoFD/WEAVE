"""Future S1 serial profile of actual logical updates; never runs on import.

The controller is CPU-only. It launches one explicitly enabled GPU worker at a
time, each from identical initialization and seed. No timed worker produces a
formal training checkpoint or selects weights using test750.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from exp.stage1_recipe.training.oneshot_p import ROOT, Session, arm_recipe, asset_root, atomic_json, read_config, resolve


def summarize(rows: list[dict], arms: tuple[str, ...], *, train_seconds: float = 4500.,
              minimum_updates: int = 600, safety_factor: float = .90) -> dict:
    """Choose one shared stable micro/cp config using the slowest arm's rate."""
    if not 0 < safety_factor <= 1:
        raise ValueError("Profile safety_factor must be in (0,1]")
    options = []
    for micro in sorted({r["micro"] for r in rows}):
        for checkpointing in sorted({r["checkpointing"] for r in rows}):
            subset = [r for r in rows if r.get("micro") == micro and
                      r.get("checkpointing") == checkpointing and r.get("stable")]
            by_arm = {r["arm"]: r for r in subset}
            if not all(arm in by_arm for arm in arms):
                continue
            rates = [float(by_arm[a]["seconds_per_update_p95"]) for a in arms]
            if not all(math.isfinite(v) and v > 0 for v in rates):
                continue
            conservative = max(rates) / safety_factor
            options.append({"micro": micro, "checkpointing": checkpointing,
                            "seconds_per_update": conservative,
                            "samples_per_s": min(float(by_arm[a]["samples_per_s"]) for a in arms),
                            "common_updates": math.floor(train_seconds/conservative)})
    if not options:
        return {"status": "no_stable_configuration", "variants": rows,
                "reduce_arms_required": True, "common_updates": 0}
    best = max(options, key=lambda x: x["samples_per_s"])
    return {**best, "status": "ready" if best["common_updates"] >= minimum_updates else "reduce_arms_required",
            "reduce_arms_required": best["common_updates"] < minimum_updates,
            "training_seconds": train_seconds, "safety_factor": safety_factor,
            "minimum_updates": minimum_updates, "variants": rows, "stable_options": options}


def worker(config_path: str, arm: str, micro: int, checkpointing: bool,
           output: Path, warmup: int, updates: int) -> None:
    if os.environ.get("WEAVE_STAGE1_GPU_ALLOWED") != "1":
        raise RuntimeError("S1 GPU profiling is disabled until explicitly launched by the serial queue")
    root = ROOT
    config = read_config(config_path)
    config["paths"]["p_initial"] = str(asset_root(config) / config["paths"]["p_candidates"][0])
    config["paths"]["cache_root"] = str(asset_root(config) / config["paths"]["cache_root"])
    if arm == "P2":
        # LR has no effect on activation shapes. This is a throughput assumption,
        # not a Seen-Dev LR choice; queue resolves the real P2 LR only after P0/P1.
        config.setdefault("arms", {}).setdefault("P2", {}).setdefault("lr_p", 1e-4)
    recipe = arm_recipe(config, arm)
    runtime = Session(config, recipe, micro=micro, checkpointing=checkpointing, root=root)
    torch, np = runtime.torch, runtime.np
    # Each isolated worker freshly loaded the same checkpoint, optimizer and RNG.
    for _ in range(warmup):
        runtime.update()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    durations, resident = [], []
    for _ in range(updates):
        start = time.perf_counter()
        runtime.update()
        torch.cuda.synchronize()
        durations.append(time.perf_counter()-start)
        free, total = torch.cuda.mem_get_info()
        resident.append((total-free)/2**30)
    peak = torch.cuda.max_memory_allocated()/2**30
    reserved = torch.cuda.max_memory_reserved()/2**30
    card_peak = max(resident)
    result = {"profile_only": True, "arm": arm, "family": "p", "micro": micro,
              "checkpointing": checkpointing, "warmup_updates": warmup,
              "measured_updates": updates, "logical_batch": 96,
              "seconds_per_update": float(np.mean(durations)),
              "seconds_per_update_p95": float(np.percentile(durations, 95)),
              "samples_per_s": 96./float(np.mean(durations)),
              "peak_allocated_gib": peak, "peak_reserved_gib": reserved,
              "card_used_peak_gib": card_peak,
              "stable": max(peak, card_peak) <= 10. and all(math.isfinite(d) for d in durations),
              "no_formal_weights_saved": True,
              "p2_lr_profile_assumption_only": arm == "P2" and recipe["lr_p"] == 1e-4}
    atomic_json(output, result)
    print(json.dumps(result), flush=True)
    del runtime
    gc.collect()
    torch.cuda.empty_cache()


def run_profile(args: argparse.Namespace) -> None:
    if os.environ.get("WEAVE_STAGE1_GPU_ALLOWED") != "1":
        raise RuntimeError("S1 GPU profiling requires explicit WEAVE_STAGE1_GPU_ALLOWED=1")
    root = ROOT
    output = Path(args.output)
    if not output.is_absolute(): output = root / output
    config = read_config(args.config)
    asset = asset_root(config)
    config["paths"]["cache_root"] = str(asset / config["paths"]["cache_root"])
    settings = config.get("profile", {})
    warmup = args.profile_warmup if args.profile_warmup is not None else int(settings.get("warmup", 2))
    steps = args.profile_steps if args.profile_steps is not None else int(settings.get("steps", 6))
    training_seconds = args.training_seconds if args.training_seconds is not None else float(config.get("training", {}).get("train_budget_s", 4500))
    if warmup < 1 or steps < 2:
        raise ValueError("Profile needs warmup and at least two full measured updates")
    work = output.parent/"profile_workers"
    work.mkdir(parents=True, exist_ok=True)
    families = ("e", "p") if args.family == "both" else (args.family,)
    profiles = {}
    for family in families:
        arms = ("E0", "E1", "E2") if family == "e" else ("P0", "P1", "P2")
        rows = []
        micros = settings.get("micro_" + family, [1, 2, 4] if family == "e" else [4, 8, 16])
        for micro in micros:
            for cp in settings.get("checkpointing", [False, True]):
                for arm in arms:
                    name = f"{arm}_m{micro}_cp{int(cp)}"
                    result_path = work/(name+".json")
                    cmd = [sys.executable, "-u", "-m"]
                    if family == "p":
                        cmd += ["exp.stage1_recipe.training.profile", "--config", args.config,
                                "--profile-worker", arm, "--micro", str(micro),
                                "--output", str(result_path), "--profile-steps", str(steps),
                                "--profile-warmup", str(warmup)]
                    else:
                        cmd += ["exp.stage1_recipe.training.reference_e", "--config", args.config,
                                "--profile", "--profile-output", str(result_path),
                                "--profile-steps", str(steps),
                                "--profile-warmup", str(warmup),
                                "--arm", arm, "--run-dir", str(work/name), "--steps", "600",
                                "--deadline-epoch", str(min(args.deadline_epoch, time.time()+600)),
                                "--micro", str(micro)]
                    cmd[cmd.index("--config") + 1] = str(Path(args.config).resolve())
                    if family == "p":
                        cmd += ["--asset-root", str(asset)]
                    if cp:
                        cmd.append("--checkpointing")
                    if time.time() >= args.deadline_epoch:
                        raise TimeoutError("S1 budget expired before all candidate profiles were measured")
                    log = work/(name+".log")
                    with log.open("wb") as fh:
                        run = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                             stdin=subprocess.DEVNULL,
                                             timeout=max(1, min(600, args.deadline_epoch-time.time())))
                    if run.returncode:
                        rows.append({"arm": arm, "micro": micro, "checkpointing": cp,
                                     "stable": False, "error": f"worker exit {run.returncode}",
                                     "log": str(log)})
                    else:
                        measured = json.loads(result_path.read_text(encoding="utf-8"))
                        # E's CLI uses this compatible scalar field; 10% margin is
                        # used if it has not reported a separate p95 quantile.
                        measured.setdefault("seconds_per_update_p95", measured["seconds_per_update"])
                        measured.update(arm=arm, micro=micro, checkpointing=cp)
                        rows.append(measured)
                    atomic_json(output, {"training_profiles": profiles,
                                "in_progress_family": family, "partial_variants": rows,
                                "status": "profiling", "updated_epoch": time.time()})
        profiles[family] = summarize(rows, arms, train_seconds=training_seconds,
                minimum_updates=int(config.get("training", {}).get("min_updates", 600)),
                safety_factor=float(settings.get("safety_factor", .90)))
        atomic_json(output, {"training_profiles": profiles, "status": "profiling",
                            "updated_epoch": time.time()})
    status = "ready" if all(p["status"] == "ready" for p in profiles.values()) else "reduce_arms_required"
    atomic_json(output, {"training_profiles": profiles, "status": status,
                        "updated_epoch": time.time(), "profile_only": True,
                        "note": "Actual GPU measurements; no quality claim and no automatic P-stage cancellation"})
    if any(p["status"] == "no_stable_configuration" for p in profiles.values()):
        raise RuntimeError("S1 failed: no shared stable micro/checkpoint configuration; see profile variants")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--family", choices=("e", "p", "both"), default="both")
    ap.add_argument("--deadline-epoch", type=float, default=float("inf"))
    ap.add_argument("--profile-steps", type=int)
    ap.add_argument("--profile-warmup", type=int)
    ap.add_argument("--training-seconds", type=float)
    ap.add_argument("--profile-worker", choices=("P0", "P1", "P2"), help=argparse.SUPPRESS)
    ap.add_argument("--micro", type=int, default=4)
    ap.add_argument("--checkpointing", action="store_true")
    return ap


if __name__ == "__main__":
    parsed = parser().parse_args()
    if parsed.profile_worker:
        if parsed.profile_steps is None or parsed.profile_steps < 2 or parsed.profile_warmup is None or parsed.profile_warmup < 1:
            raise ValueError("Profile worker needs explicit warmup and at least two measured updates")
        worker(parsed.config, parsed.profile_worker, parsed.micro, parsed.checkpointing,
               resolve(parsed.output), parsed.profile_warmup, parsed.profile_steps)
    else:
        run_profile(parsed)
