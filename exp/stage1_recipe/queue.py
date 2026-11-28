"""Exclusive, deadline-persistent serial queue. Default mode only prints a plan."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
os.environ.setdefault("PYTHONUTF8", "1")
import signal
import subprocess
import sys
import time
from pathlib import Path

from exp.stage1_recipe.config import (ROOT, STAGES, audit_root, defaults, digest, read,
                                     resolve, resolve_root, validate, write)

CODE_ROOT = Path(__file__).resolve().parents[2]


def pid_alive(pid: int) -> bool:
    """Read-only liveness check; never os.kill(pid, 0) on Windows."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        code = wintypes.DWORD()
        try:
            ok = kernel.GetExitCodeProcess(handle, ctypes.byref(code))
            return bool(ok and code.value == 259)
        finally:
            kernel.CloseHandle(handle)
    return Path(f"/proc/{pid}").exists()


def assert_idle(config):
    """Executed only after --execute, immediately before claiming the window."""
    result = subprocess.run(["nvidia-smi", "--query-gpu=name,pci.device_id,memory.used,utilization.gpu",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    lines = result.stdout.strip().splitlines()
    if len(lines) != 1:
        raise RuntimeError("This recipe expects one physical GPU")
    name, pci, memory, util = [x.strip() for x in lines[0].split(",")]
    cfg = config["gpu"]
    if cfg.get("required_name") and cfg["required_name"].lower() not in name.lower() and pci.lower() != "0x250110de":
        raise RuntimeError(f"Expected RTX3060, found {name} / {pci}")
    processes = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name",
                                "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout.strip()
    # Windows WDDM lists desktop graphics applications as compute apps. Check
    # actual Python/CUDA jobs; desktop composition cannot define card ownership.
    workers = [line for line in processes.splitlines() if any(
        token in line.lower() for token in ("python", "comfy", "blender", "llama", "ollama"))]
    if workers or float(memory) > cfg["idle_memory_mib_max"]:
        raise RuntimeError(f"GPU occupied: {workers}; memory={memory} MiB")
    if os.name != "nt" and float(util) > cfg["idle_util_max"]:
        raise RuntimeError("GPU utilization exceeds idle threshold")
    return {"name": name, "pci_device_id": pci, "memory_mib": float(memory),
            "utilization": float(util), "checked_epoch": time.time(), "worker_lines": workers}


@contextlib.contextmanager
def lock_window(out: Path, clear_stale=False):
    path = out / "queue.lock"
    if path.exists():
        old = read(path)
        if pid_alive(int(old["pid"])):
            raise RuntimeError("Another queue owns this output directory")
        if not clear_stale:
            raise RuntimeError("Stale queue lock: inspect it, then use --clear-stale-lock")
        path.unlink()
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({"pid": os.getpid(), "created_epoch": time.time()}, f)
    try:
        yield
    finally:
        if path.exists() and read(path).get("pid") == os.getpid():
            path.unlink()


class Queue:
    def __init__(self, config_path: Path, out: Path, manifest: dict):
        self.config_path, self.out, self.manifest = config_path, out, manifest
        self.config = read(config_path)
        self.root = ROOT
        self.cache = resolve_root(self.config["paths"]["cache_root"], self.root)
        self.stop = False
        self.active_run = None
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: setattr(self, "stop", True))

    def run(self, label, module, args, deadline, run_dir=None, gpu=True, allow_other_gpu_processes=False):
        state_path = self.out / "workers" / (label + ".json")
        command_args = []
        for index, value in enumerate(args):
            path = Path(value) if isinstance(value, Path) else None
            if path is not None and not path.is_absolute():
                base = self.model_root if index > 0 and str(args[index - 1]) == "--resume-from" else self.root
                path = base / path
                value = path
            command_args.append(str(value))
        command = [sys.executable, "-u", "-m", module, *command_args]
        if state_path.exists():
            old = read(state_path)
            if old.get("command") != command:
                raise ValueError(f"Worker command changed on resume: {label}")
            if old.get("status") == "complete":
                return
            if old.get("pid") and pid_alive(old["pid"]):
                raise RuntimeError(f"Existing worker still alive: {label}")
            deadline = min(deadline, old["deadline_epoch"])
        if self.stop or (self.out / "STOP").exists():
            raise InterruptedError("Queue STOP requested")
        if time.time() >= deadline - 10:
            raise TimeoutError(f"No time left for {label}; deadline is never reset")
        # Recheck after each owned process has exited; never race an existing Qwen/F7 worker.
        shared = bool(allow_other_gpu_processes or getattr(self, "allow_shared_gpu", False))
        if gpu and not shared:
            assert_idle(self.config)
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1",
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        if gpu:
            env["WEAVE_STAGE1_GPU_ALLOWED"] = "1"
            if shared:
                env["WEAVE_STAGE1_ALLOW_EXTERNAL_GPU_WORKERS"] = "1"
        else:
            env.pop("WEAVE_STAGE1_GPU_ALLOWED", None)
        log = self.out / "logs" / (label + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("ab", buffering=0) as fh:
            child = subprocess.Popen(command, cwd=self.root, env=env, stdin=subprocess.DEVNULL,
                                     stdout=fh, stderr=subprocess.STDOUT)
            self.active_run = run_dir
            state = {"label": label, "command": command, "pid": child.pid, "status": "running",
                     "deadline_epoch": deadline, "started_epoch": time.time(), "log": str(log)}
            write(state_path, state)
            last_step = None
            last_advance = time.time()
            while child.poll() is None:
                now = time.time()
                if run_dir is not None:
                    progress = run_dir / "heartbeat.json"
                    if not progress.exists():
                        progress = run_dir / "state.json"
                    if progress.exists():
                        row = read(progress)
                        step = row.get("added_steps", row.get("step"))
                        if step != last_step:
                            last_step, last_advance = step, now
                        state.update(observed_step=step, heartbeat_mtime=progress.stat().st_mtime,
                                     last_step_advance_epoch=last_advance,
                                     progress_fresh=now - progress.stat().st_mtime < 600)
                state.update(worker_alive=child.poll() is None, updated_epoch=now,
                             log_bytes=log.stat().st_size, stalled=now - last_advance > 600)
                write(state_path, state)
                if self.stop or (self.out / "STOP").exists() or now >= deadline:
                    if run_dir is not None:
                        (run_dir / "STOP").touch()
                    try:
                        child.wait(timeout=40)
                    except subprocess.TimeoutExpired:
                        # Only this queue's child tree. Never stop unrelated card users.
                        if os.name == "nt":
                            subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"], capture_output=True)
                        else:
                            child.terminate()
                        child.wait(timeout=15)
                    state.update(status="interrupted" if self.stop else "deadline", finished_epoch=time.time())
                    write(state_path, state)
                    raise TimeoutError(f"{label} ended without a complete result; inspect preserved checkpoints")
                time.sleep(5)
        rc = child.returncode
        state.update(status="complete" if rc == 0 else "failed", rc=rc, worker_alive=False,
                     finished_epoch=time.time())
        write(state_path, state)
        self.active_run = None
        if rc:
            raise RuntimeError(f"{label} exited {rc}; inspect {log}")

    def eval_jobs(self, jobs, label, deadline):
        paths = []
        for i, job in enumerate(jobs):
            folder = self.out / "evaluations" / label / job["name"]
            spec = folder / "job.json"
            if spec.exists() and read(spec) != job:
                raise ValueError("Immutable evaluation job changed")
            write(spec, job)
            base = ["--config", self.config_path, "--job", spec, "--output", folder,
                    "--deadline-epoch", deadline]
            self.run(f"{label}_{job['name']}_generate", "exp.stage1_recipe.evaluation.generate", ["generate", *base], deadline)
            self.run(f"{label}_{job['name']}_score", "exp.stage1_recipe.evaluation.generate", ["score", *base], deadline)
            paths.append(folder / "metrics.json")
        return paths

    def jobs(self, name, family, *, p=None, e=None, p_ema=None, e_ema=None, split="seen_dev", knobs=None, **kwargs):
        ev = self.config["evaluation"]
        knobs = knobs if knobs is not None else (ev["e_lambdas"] if family == "e" else ev["taus"])
        return [{"name": f"{name}_{int(k*100):03d}", "route": name, "family": family,
                 "p_checkpoint": p, "e_checkpoint": e, "p_ema": p_ema, "e_ema": e_ema,
                 "split": split, "n": ev["seen_n"] if split == "seen_dev" else 0,
                 ("lambda" if family == "e" else "tau"): k, **kwargs} for k in knobs]

    def persist_config(self, config, name):
        path = self.out / "configs" / (name + ".json")
        if path.exists() and read(path) != config:
            raise ValueError(f"Frozen per-stage configuration changed: {path}")
        write(path, config)
        return path

    def s1(self, deadline):
        from exp.stage1_recipe.evaluation.selection import seen_score
        base_paths = []
        candidates = self.config["paths"]["p_candidates"]
        for i, ck in enumerate(candidates):
            jobs = self.jobs(f"initial_P{i}", "p", p=ck, knobs=[0., *self.config["evaluation"]["taus"]])
            base_paths.append(self.eval_jobs(jobs, "S1_baseline", deadline))
        comparison = seen_score(base_paths[1], base_paths[0], self.config["evaluation"]["content_anchors"])
        if comparison["score"] is None:
            raise RuntimeError("P1/P2 Seen-Dev curves have no bracketed comparison; inspect before choosing")
        selected = 1 if comparison["score"] > 0 else 0
        self.config["paths"]["p_initial"] = candidates[selected]
        self.config["training"]["p"]["ratios_stage"] = "p2" if selected == 1 else "p1"
        write(self.out / "p_initial_selection.json", {"selected": candidates[selected], "selected_index": selected,
                                                      "comparison": comparison, "baseline_paths": [[str(p) for p in pp] for pp in base_paths],
                                                      "selection_data": "seen_dev_only"})
        self.config_path = self.persist_config(self.config, "s1_initial")
        profile = self.out / "profile.json"
        self.run("S1_profile", "exp.stage1_recipe.training.profile",
                 ["--config", self.config_path, "--output", profile, "--family", "both",
                  "--deadline-epoch", deadline], deadline)
        measured = read(profile)
        if measured["status"] != "ready":
            raise RuntimeError("Common updates below600 or no stable profile; reduce the matrix explicitly before starting a new window")
        for f in ("e", "p"):
            x = measured["training_profiles"][f]
            self.config["training"][f].update(micro=x["micro"], checkpointing=x["checkpointing"],
                                                common_updates=x["common_updates"])
        self.config_path = self.persist_config(self.config, "training_common")
        # E-only control for assessing independent E ability, distinct from P+A3.
        paths = self.eval_jobs(self.jobs("initial_E", "e", e=self.config["paths"]["e_initial"], e_ema="online"), "S1_E_baseline", deadline)
        write(self.out / "e_initial_selection.json", {"baseline_paths": list(map(str, paths))})

    def train_arm(self, arm, stage_deadline, *, run_name=None, config=None, steps=None, train_seconds=4500, reserve=900,
                  resume_from=None):
        from exp.stage1_recipe.evaluation.selection import seen_score
        family = "e" if arm.startswith("E") else "p"
        config = config or self.config
        settings = config["training"][family]
        steps = int(steps or settings["common_updates"])
        if steps < config["training"]["min_updates"]:
            raise ValueError("Fewer than600 updates: revise the matrix rather than forcing all arms")
        name = run_name or arm
        out = self.out / "training" / name
        out.mkdir(parents=True, exist_ok=True)
        cfgpath = self.persist_config(config, name)
        train_deadline = min(time.time() + train_seconds, stage_deadline - reserve)
        # Freeze the training cutoff separately; retry does not extend it.
        cutoff_path = out / "cutoff.json"
        if cutoff_path.exists():
            train_deadline = min(train_deadline, read(cutoff_path)["deadline_epoch"])
        write(cutoff_path, {"deadline_epoch": train_deadline})
        args = ["--config", cfgpath, "--arm", arm, "--run-dir", out, "--steps", steps,
                "--micro", settings["micro"], "--deadline-epoch", train_deadline]
        if resume_from is not None:
            args.extend(["--resume-from", resume_from])
        if settings["checkpointing"]:
            args.append("--checkpointing")
        module = "exp.stage1_recipe.training." + ("reference_e" if family == "e" else "oneshot_p")
        self.run(name + "_train", module, args, min(train_deadline + 60, stage_deadline), out)
        checkpoint = out / "matched.pt"
        if not checkpoint.exists():
            raise RuntimeError(f"{name} did not reach its common update checkpoint")
        baseline = read(self.out / "p_initial_selection.json")
        # Formal V36 P+A3 comparison uses the original P2 model, irrespective of continuation initialization.
        baseline_paths = [Path(p) for p in baseline["baseline_paths"][1]]
        scores = []
        for decay in settings["ema_decays"]:
            if family == "e":
                pj = self.jobs(name + "_EP_" + str(decay), "ep", p=config["paths"]["p_candidates"][1],
                               e=str(checkpoint), e_ema=str(decay))
                ej = self.jobs(name + "_E_" + str(decay), "e", e=str(checkpoint), e_ema=str(decay))
                ep = self.eval_jobs(pj, name + "_seen", stage_deadline)
                standalone = self.eval_jobs(ej, name + "_seen_e", stage_deadline)
                ebase = [Path(p) for p in read(self.out / "e_initial_selection.json")["baseline_paths"]]
                e_result = seen_score(standalone, ebase, [.70, .75, .80, .85])
            else:
                pj = self.jobs(name + "_P_" + str(decay), "p", p=str(checkpoint), p_ema=str(decay),
                               knobs=[0., *config["evaluation"]["taus"]])
                ep = self.eval_jobs(pj, name + "_seen", stage_deadline)
                e_result = None
            result = seen_score(ep, baseline_paths, config["evaluation"]["content_anchors"])
            result["eligible_ep_or_p"] = result["eligible"]
            result["eligible_e_only"] = bool(e_result and e_result["eligible"])
            result["eligible"] = result["eligible"] or result["eligible_e_only"]
            scores.append({"arm": arm, "run": name, "ema": str(decay), "family": "ep" if family == "e" else "p",
                           "checkpoint": str(checkpoint), "comparison": result,
                           "e_only_comparison": e_result, "paths": list(map(str, ep)), "jobs": pj})
        write(self.out / "screens" / (name + ".json"), scores)
        return scores

    def resolve_p_lr(self):
        choices = [x for arm in ("P0", "P1") for x in read(self.out / "screens" / (arm + ".json"))
                   if x["comparison"]["score"] is not None]
        if not choices:
            raise RuntimeError("No bracketed P0/P1 Seen-Dev LR comparison")
        best = max(choices, key=lambda r: r["comparison"]["score"])
        source = self.config["arms"][best["arm"]]
        self.config["arms"]["P2"].update(lr_p=source["lr_p"], lr_style=source["lr_style"])
        self.config["arms"]["P3"].update(lr_p=source["lr_p"], lr_style=2 * source["lr_p"])
        write(self.out / "p_lr_selection.json", {"winner": best, "source": "seen_dev_only", "p2_forks_common_initial": True})

    def r0(self, deadline):
        from exp.stage1_recipe.evaluation.selection import seen_score

        mode = self.config["r0"].get("mode", "auto")
        if mode not in ("auto", "second_seed", "extend", "backup_e", "backup_p"):
            raise ValueError("R0 mode must be auto/second_seed/extend/backup_e/backup_p")
        decision_path, plan_path = self.out / "r0_decision.json", self.out / "r0_plan.json"
        if decision_path.exists() and read(decision_path).get("status") == "complete":
            return

        def best(rows):
            valid = [x for x in rows if x["comparison"].get("score") is not None]
            return max(valid, key=lambda x: x["comparison"]["score"]) if valid else None

        if plan_path.exists():
            plan = read(plan_path)
            if plan["requested_mode"] != mode:
                raise ValueError("Frozen R0 mode changed on resume")
            deadline = min(deadline, plan["deadline_epoch"])
        else:
            # Ignore R0 screens so a retry cannot select itself or change the pair.
            all_scores = [row for p in sorted((self.out / "screens").glob("*.json"))
                          if not p.stem.startswith("R0_") for row in read(p)]
            by_arm = {arm: best([x for x in all_scores if x["arm"] == arm])
                      for arm in ("E0", "E1", "E2", "P0", "P1", "P2")}
            lr_selection = self.out / "p_lr_selection.json"
            p_control = read(lr_selection)["winner"]["arm"] if lr_selection.exists() else None
            controls = {"E1": "E0", "E2": "E1", "P1": "P0", "P2": p_control}
            pairs = []
            for candidate, control in controls.items():
                c, b = by_arm.get(candidate), by_arm.get(control)
                if c is None or b is None:
                    continue
                direct = seen_score([Path(p) for p in c["paths"]], [Path(p) for p in b["paths"]],
                                    self.config["evaluation"]["content_anchors"])
                if direct["eligible"]:
                    pairs.append({"candidate": c, "control": b, "direct_comparison": direct})

            selected = max(pairs, key=lambda x: x["direct_comparison"]["score"]) if pairs else None
            actual_mode = "second_seed" if mode == "auto" else mode
            if mode in ("auto", "second_seed", "extend") and selected is None:
                write(decision_path, {"status": "complete", "requested_mode": mode,
                      "mode": "frozen_diagnostics", "reason": "no_supported_candidate_control_gain",
                      "confirmation": False, "independent_replication": False,
                      "no_blind_extension": True, "preserve_I0_F0": True})
                if deadline - time.time() >= 60:
                    self.run("R0_contract", "exp.stage1_recipe.training.reference_e",
                 ["--config", self.config_path, "--arm", "E1", "--run-dir", self.out / "R0_audit",
                          "--audit-contract", "--audit-output", self.out / "R0_contract.json"], deadline)
                return

            cfg = json.loads(json.dumps(self.config))
            if actual_mode == "backup_e":
                candidate_arm, control_arm = "E3", "E1"
                cfg["arms"]["E3"].update(a_q32="proxy", a_weight=.75, b_style_weight=1.)
                original_pair = None
            elif actual_mode == "backup_p":
                if p_control not in ("P0", "P1"):
                    raise ValueError("backup_p needs the frozen P0/P1 Seen-Dev LR choice")
                candidate_arm, control_arm = "P3", p_control
                lr = float(cfg["arms"][p_control]["lr_p"])
                cfg["arms"]["P3"].update(lr_p=lr, lr_style=2*lr, q_noise=.05)
                original_pair = None
            else:
                candidate_arm, control_arm = selected["candidate"]["arm"], selected["control"]["arm"]
                original_pair = selected
            if actual_mode != "extend":
                cfg["seed"] += int(cfg["r0"].get("seed_offset", 1))
                for arm in (control_arm, candidate_arm):
                    cfg["arms"][arm]["seed"] = cfg["seed"]
            family = "e" if candidate_arm.startswith("E") else "p"
            profile = read(self.out / "profile.json")["training_profiles"][family]
            started = time.time()
            half = min(3600., max(0., deadline - started)/2.)
            train_seconds = min(3000., max(0., half-600.))
            common = int(int(profile["common_updates"]) * train_seconds / 4500.)
            if common < int(self.config["training"]["min_updates"]):
                write(decision_path, {"status": "complete", "requested_mode": mode,
                      "mode": actual_mode, "reason": "insufficient_r0_budget", "confirmation": False,
                      "independent_replication": False, "candidate": candidate_arm, "control": control_arm,
                      "common_additional_updates": common, "training_seconds_per_arm": train_seconds,
                      "preserve_I0_F0": True, "no_training_started": True})
                return

            runs = []
            for i, arm in enumerate((control_arm, candidate_arm)):
                name = "R0_" + actual_mode + "_" + arm
                resume = None
                initial_added = 0
                arm_cfg = cfg
                if actual_mode == "extend":
                    source = selected["control"] if arm == control_arm else selected["candidate"]
                    source_dir = self.out / "training" / source["run"]
                    # Continue the equal-update checkpoint, not unequal wall-time endpoints.
                    resume = Path(source["checkpoint"])
                    if not resume.is_absolute():
                        resume = self.root / resume
                    # Preserve the original trainer's exact config hash, optimizer,
                    # sampler and seed; extension is explicitly not a second seed.
                    arm_cfg = read(self.out / "configs" / (source["run"] + ".json"))
                    if not resume.is_file():
                        raise FileNotFoundError(f"R0 continuation needs full checkpoint: {resume}")
                    source_state = read(source_dir / "state.json")
                    matched = source_state.get("matched_steps", source_state.get("matched_updates_target"))
                    if matched is None:
                        raise ValueError("R0 extension requires a declared matched-update checkpoint")
                    initial_added = int(matched)
                cfgpath = self.persist_config(arm_cfg, name)
                runs.append({"arm": arm, "run": name, "config": str(cfgpath),
                             "resume_from": str(resume) if resume else None,
                             "initial_added_steps": initial_added, "steps": initial_added+common,
                             "deadline_epoch": min(started+(i+1)*half, deadline)})
            plan = {"requested_mode": mode, "mode": actual_mode, "candidate": candidate_arm,
                    "control": control_arm, "original_pair": original_pair,
                    "common_additional_updates": common, "training_seconds_per_arm": train_seconds,
                    "calibration_reserve_seconds": 600, "runs": runs,
                    "started_epoch": started, "deadline_epoch": deadline}
            write(plan_path, plan)

        write(decision_path, {**plan, "status": "running", "confirmation": False,
                             "independent_replication": False, "preserve_I0_F0": True})
        results = {}
        try:
            for case in plan["runs"]:
                cfg = read(case["config"])
                scores = self.train_arm(case["arm"], min(case["deadline_epoch"], deadline),
                        run_name=case["run"], config=cfg, steps=case["steps"],
                        train_seconds=plan["training_seconds_per_arm"], reserve=600,
                        resume_from=case["resume_from"])
                results[case["arm"]] = best(scores)
        except TimeoutError as exc:
            write(decision_path, {**plan, "status": "complete", "reason": "incomplete_r0_budget",
                  "error": str(exc), "confirmation": False, "independent_replication": False,
                  "completed_results": results, "preserve_I0_F0": True})
            return
        candidate, control = results[plan["candidate"]], results[plan["control"]]
        comparison = seen_score([Path(p) for p in candidate["paths"]], [Path(p) for p in control["paths"]],
                                self.config["evaluation"]["content_anchors"]) if candidate and control else None
        confirms = bool(comparison and comparison["eligible"] and comparison["score"] > 0 and
                        all(r["delta_s"] >= -.005 for r in comparison["matched"]))
        write(decision_path, {**plan, "status": "complete", "confirmation": confirms,
              "independent_replication": confirms and plan["mode"] == "second_seed",
              "evidence_type": "independent_second_seed" if plan["mode"] == "second_seed" else
                               "continued_training" if plan["mode"] == "extend" else "new_backup_factor",
              "candidate_result": candidate, "control_result": control, "direct_comparison": comparison,
              "reason": "seen_gain_supported" if confirms else "gain_not_confirmed_or_regressed",
              "selection_data": "seen_dev_only", "formal_ci_still_required": True,
              "preserve_I0_F0": True})

    def calibrate(self, deadline):
        from exp.stage1_recipe.evaluation.selection import seen_score
        all_scores = [r for p in (self.out / "screens").glob("*.json") if not p.stem.startswith("R0_") for r in read(p)]
        supported = [r for r in all_scores if r["comparison"]["eligible"]]
        replication = read(self.out / "r0_decision.json") if (self.out / "r0_decision.json").exists() else {}
        if replication.get("mode") == "second_seed" and replication.get("reason") == "gain_not_confirmed_or_regressed":
            failed = replication["candidate"]
            supported = [r for r in supported if r["arm"] != failed]
        if replication.get("mode") in ("extend", "backup_e", "backup_p") and replication.get("confirmation"):
            supported.append(replication["candidate_result"])
        score = lambda r: (r["comparison"]["score"] if r["comparison"]["score"] is not None else
                           (r.get("e_only_comparison") or {}).get("score", float("-inf")))
        es, ps = [r for r in supported if r["family"] == "ep"], [r for r in supported if r["family"] == "p"]
        if es and ps:
            e, p = max(es, key=score), max(ps, key=score)
            name = "combo_" + e["run"] + "_" + p["run"]
            jobs = self.jobs(name, "ep", p=p["checkpoint"], e=e["checkpoint"],
                             p_ema=p["ema"], e_ema=e["ema"])
            paths = self.eval_jobs(jobs, "I0_complementarity", deadline)
            strongest = max((e, p), key=score)
            direct = seen_score(paths, [Path(x) for x in strongest["paths"]], self.config["evaluation"]["content_anchors"])
            if direct["eligible"]:
                baseline_paths = [Path(x) for x in read(self.out / "p_initial_selection.json")["baseline_paths"][1]]
                comparison = seen_score(paths, baseline_paths, self.config["evaluation"]["content_anchors"])
                supported.append({**e, "run": name, "arm": "EP_combo", "jobs": jobs,
                                  "paths": list(map(str, paths)), "comparison": comparison,
                                  "complementarity_vs_best_individual": direct})
        supported.sort(key=score, reverse=True)
        selected = supported[:self.config["evaluation"]["formal_max_candidates"]]
        baseline = [Path(p) for p in read(self.out / "p_initial_selection.json")["baseline_paths"][1]]
        final = []
        if selected:
            baseline_probes = self.out / "probes/baseline_P_A3"
            baseline_probe_job = self.jobs("baseline_P_A3", "p", p=self.config["paths"]["p_candidates"][1], knobs=[.5])[0]
            write(baseline_probes / "job.json", baseline_probe_job)
            self.run("I0_probe_baseline", "exp.stage1_recipe.evaluation.probes",
                     ["--config", self.config_path, "--job", baseline_probes / "job.json", "--output", baseline_probes,
                      "--deadline-epoch", deadline], deadline)
        for r in selected:
            original = r["jobs"][0]
            base = {k: v for k, v in original.items() if k not in ("name", "route", "tau", "split", "n")}
            specs = [{}, {"wj": self.config["evaluation"]["wj"] * .9},
                     {"wj": self.config["evaluation"]["wj"] * 1.1}, {"wc": self.config["evaluation"]["wc"] * 1.1},
                     {"alpha_y": .75}, {"alpha_c": .75}]
            variants = []
            for i, change in enumerate(specs):
                job = {**base, "name": r["run"] + f"_I{i}", "route": r["run"], "tau": .5,
                       "split": "seen_dev", "n": self.config["evaluation"]["seen_n"], **change}
                path = self.eval_jobs([job], "I0_local", deadline)[0]
                variants.append((job, read(path)["means"]))
            # Single-point screening uses a declared style-minus-content-deviation proxy,
            # then the winner must improve the whole bracketed curve, never the test set.
            original_c = variants[0][1]["dino_c"]
            best = max(variants, key=lambda x: x[1]["dino_s"] - abs(x[1]["dino_c"] - original_c))[0]
            jobs = [{**best, "name": r["run"] + f"_Ibest_{int(t*100)}", "tau": t}
                    for t in self.config["evaluation"]["taus"]]
            paths = self.eval_jobs(jobs, "I0_curve", deadline)
            check = seen_score(paths, baseline, self.config["evaluation"]["content_anchors"])
            if check["eligible"] or r["comparison"].get("eligible_e_only"):
                probes = self.out / "probes" / r["run"]
                probe_job = probes / "job.json"
                write(probe_job, best)
                self.run("I0_probe_" + r["run"], "exp.stage1_recipe.evaluation.probes",
                         ["--config", self.config_path, "--job", probe_job, "--output", probes,
                          "--deadline-epoch", deadline], deadline)
                final.append({**r, "calibrated_jobs": jobs, "calibration": check,
                              "r0_evidence": replication, "probe_directory": str(probes)})
        write(self.out / "final_selection.json", {"candidates": final, "data": "seen_dev_only",
                                                  "formal_limit": self.config["evaluation"]["formal_max_candidates"]})

    def formal(self, deadline):
        baseline = self.jobs("baseline_P_A3", "p", p=self.config["paths"]["p_candidates"][1], split="core750")
        selected = read(self.out / "final_selection.json")["candidates"]
        rows = {"baseline": [], "candidates": [], "selection_frozen_before_test": True}
        # Baseline is always complete; reuse is allowed only after immutable-job provenance matches.
        for split in ("core750", "artist_ood", "movement_ood"):
            jobs = [{**x, "split": split, "n": 0} for x in baseline]
            paths = self.eval_jobs(jobs, "F0_baseline_" + split, deadline)
            rows["baseline"].append({"split": split, "paths": list(map(str, paths))})
        for r in selected:
            case = {"name": r["run"], "family": r["family"], "splits": []}
            for split in ("core750", "artist_ood", "movement_ood"):
                jobs = [{**x, "split": split, "n": 0} for x in r["calibrated_jobs"]]
                paths = self.eval_jobs(jobs, "F0_" + r["run"] + "_" + split, deadline)
                case["splits"].append({"split": split, "paths": list(map(str, paths))})
            if r["family"] == "ep":
                e_jobs = self.jobs(r["run"] + "_Eonly", "e", e=r["checkpoint"], e_ema=r["ema"], split="core750")
                e_paths = self.eval_jobs(e_jobs, "F0_" + r["run"] + "_Eonly", deadline)
                case["e_only"] = list(map(str, e_paths))
            rows["candidates"].append(case)
        write(self.out / "formal_index.json", rows)
        self.run("F0_report", "exp.stage1_recipe.evaluation.final",
                 ["--config", self.config_path, "--run-dir", self.out], deadline, gpu=False)

    def execute(self):
        started = self.manifest["started_epoch"]
        cumulative = 0
        for stage, _ in STAGES:
            cumulative += self.config["stage_budgets"][stage]
            deadline = min(started + cumulative, time.time() + self.config["stage_budgets"][stage],
                           self.manifest["deadline_epoch"])
            done = self.out / "stages" / (stage + ".json")
            if done.exists() and read(done).get("status") == "complete":
                if stage == "S1":
                    self.config_path = self.out / "configs/training_common.json"
                    self.config = read(self.config_path)
                if stage == "P2":
                    self.resolve_p_lr()
                continue
            if done.exists():
                deadline = min(deadline, read(done)["deadline_epoch"])
            if time.time() >= deadline:
                raise TimeoutError(f"Persistent {stage} deadline elapsed")
            write(done, {"status": "running", "started_epoch": time.time(), "deadline_epoch": deadline})
            if stage == "S0":
                self.run("S0_cpu", "exp.stage1_recipe.evaluation.audit",
                         ["--config", self.config_path, "--out", self.out / "S0_cpu_audit.json",
                          "--validate-training-assets"], deadline, gpu=False)
                inventory = read(self.out / "S0_cpu_audit.json")["lpips_repair"]
                spec = self.config.get("s0", {}).get("lpips_repair_spec")
                repair_status = {}
                for target in ("f1", "f6"):
                    if not inventory[target]["ready"] and not spec:
                        repair_status[target] = {"status": "missing_assets", "missing": inventory[target]["missing"]}
                        continue
                    args = ["--repair-lpips", "--repair-target", target, "--device", "cpu",
                            "--out", self.out / "lpips_repair" / target]
                    if spec:
                        args += ["--repair-spec", resolve_root(spec, self.root)]
                    self.run("S0_lpips_" + target, "exp.stage1_recipe.evaluation.audit", args, deadline, gpu=False)
                    repair_status[target] = {"status": "recomputed_from_existing_outputs"}
                write(self.out / "S0_result.json", {"lpips_repair": repair_status,
                    "lpips_complete": all(x["status"] == "recomputed_from_existing_outputs" for x in repair_status.values()),
                    "source_fixes": "prepared", "old_lpips_must_remain_invalid_until_recomputed": True})
                self.run("S0_contract", "exp.stage1_recipe.training.reference_e",
                         ["--config", self.config_path, "--arm", "E1", "--run-dir", self.out / "S0_contract",
                          "--audit-contract", "--audit-output", self.out / "S0_contract.json"], deadline)
            elif stage == "S1":
                self.s1(deadline)
            elif stage in ("E0", "E1", "E2", "P0", "P1", "P2"):
                if stage == "P2":
                    self.resolve_p_lr()
                self.train_arm(stage, deadline)
            elif stage == "R0":
                self.r0(deadline)
            elif stage == "I0":
                self.calibrate(deadline)
            else:
                self.formal(deadline)
            record = {"status": "complete", "finished_epoch": time.time(), "deadline_epoch": deadline}
            if stage == "S0":
                record["evidence_status"] = read(self.out / "S0_result.json")
            write(done, record)
        write(self.out / "completion.json", {"status": "complete", "finished_epoch": time.time(),
                                              "deadline_epoch": self.manifest["deadline_epoch"],
                                              "adoption_requires_final_gates": True})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path)
    p.add_argument("--execute", action="store_true", help="Launch the serial 16-hour stage-one queue")
    p.add_argument("--allow-shared-gpu", action="store_true", help="Explicitly allow existing CUDA GUI/worker processes; required for the current desktop session")
    p.add_argument("--clear-stale-lock", action="store_true")
    p.add_argument("--preflight", action="store_true", help="Read-only asset audit on a prepared checkout; does not query GPU")
    args = p.parse_args()
    config = read(args.config) if args.config else defaults()
    validate(config)
    if args.preflight:
        result = audit_root(ROOT, config)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not result["ready"]:
            raise SystemExit(2)
        return
    if not args.execute:
        print(json.dumps({"mode": "plan_only_no_gpu_query", "hours": 16,
                          "stages": config["stage_budgets"], "config_sha256": digest(config)}, indent=2))
        return
    if args.config is None:
        p.error("Explicit execution requires a saved --config")
    repo_root = ROOT
    audit = audit_root(repo_root, config)
    if not audit["ready"]:
        raise FileNotFoundError("Stage-one training assets are incomplete before GPU check: " +
                                ", ".join(audit["missing"]))
    idle = assert_idle(config)
    out = resolve_root(config["paths"]["output_root"], repo_root)
    out.mkdir(parents=True, exist_ok=True)
    with lock_window(out, args.clear_stale_lock):
        manifest_path = out / "manifest.json"
        if manifest_path.exists():
            manifest = read(manifest_path)
            if manifest["config_sha256"] != digest(config):
                raise ValueError("Frozen config changed; use a new run output root")
        else:
            now = time.time()
            manifest = {"config_sha256": digest(config), "started_epoch": now,
                        "deadline_epoch": now + 16 * 3600, "idle_check": idle,
                        "argv": sys.argv, "no_external_workers_stopped": True}
            write(manifest_path, manifest)
        queue = Queue(args.config.resolve(), out, manifest)
        queue.allow_shared_gpu = bool(args.allow_shared_gpu)
        queue.execute()


if __name__ == "__main__":
    main()
