"""Opt-in, resumable stage-1 Reference-E training. Import/help never touches CUDA."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
os.environ.setdefault("PYTHONUTF8", "1")
import random
import shutil
import signal
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def save_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def copy_checkpoint(source, destination):
    destination = Path(destination)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, tmp)
    tmp.replace(destination)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def warmup_factor(added_step: int, warmup: int = 100):
    return min(1., max(0., added_step / max(1, warmup)))


def cpu_grad_vector(model):
    return torch.cat([(p.grad.detach().float().flatten().cpu() if p.grad is not None else
                       torch.zeros(p.numel(), dtype=torch.float32)) for p in model.parameters()])


def component_gradient_probe(components, model):
    params = tuple(p for p in model.parameters() if p.requires_grad)
    grads = {}
    for key in ("style", "content"):
        gg = torch.autograd.grad(components[key], params, retain_graph=True, allow_unused=True)
        grads[key] = torch.cat([torch.zeros(p.numel()) if g is None else g.detach().float().flatten().cpu()
                               for p, g in zip(params, gg)])
    a, b = grads["style"], grads["content"]
    return {"style_grad_norm": float(a.norm()), "content_grad_norm": float(b.norm()),
            "style_content_grad_cosine": float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-12))}


@torch.no_grad()
def audit_contract(obj, micro=1):
    from exp.stage1_recipe.training.e_objectives import proxy_q_metrics, slices
    from src.utils.structure import latent_to_structure_64
    ac, _ = obj.sample()["A"]
    rows = []
    for sl in slices(len(ac), micro):
        ids = ac[sl]
        target_z, target_q = obj.tensor("z_gt", ids), obj.tensor("q32", ids)
        rgb = obj.decode(target_z, micro)
        proxy_z, proxy_q = obj.proxy(rgb, ids, 0, sl.start)
        target_c = latent_to_structure_64(target_z, obj.priors["luma_w"], obj.priors["luma_b"])
        proxy_c = latent_to_structure_64(proxy_z, obj.priors["luma_w"], obj.priors["luma_b"])
        rows.append({"source_ids": ids.tolist(), **proxy_q_metrics(target_q, proxy_q),
                     "latent_rms_change": float((target_z - proxy_z).square().mean().sqrt()),
                     "c64_rms_change": float((target_c - proxy_c).square().mean().sqrt())})
    return {"kind": "stage1_A_input_contract", "arm": obj.arm, "seed": obj.seed,
            "source": "same deterministic proxy RGB feeds encoder and Q32 extractor",
            "q32_mode": obj.arm_config.get("a_q32", "target" if obj.arm == "E0" else "proxy"),
            "c64_and_luma": "ReferenceECoreNet derives both from the actual z_proxy argument",
            "reference_latent_and_stat": "independent style condition, intentionally not content source",
            "target_latent_and_rgb": "supervision only", "full_dataset_reencoded": False,
            "rows": rows}


def logical_update(obj, model, opt, added_step, *, micro, checkpointing, lr, warmup, inspect):
    """Run exactly one A/B/C update; used by both S1 profiling and formal arms."""
    started = time.perf_counter()
    for group in opt.param_groups:
        group["lr"] = lr * warmup_factor(added_step, warmup)
    opt.zero_grad(set_to_none=True)
    totals, details, task_grads, probes = {}, {}, {}, []
    current = None
    before = cpu_grad_vector(model) if inspect else None
    last_task = None
    for task, loss, logs, components in obj.task_losses(model, added_step, micro, checkpointing, inspect):
        if task != last_task and last_task is not None and inspect:
            now = cpu_grad_vector(model)
            task_grads[last_task] = now - before
            before = now
        if inspect and components is not None:
            probes.append(component_gradient_probe(components, model))
        loss.backward()
        totals[task] = totals.get(task, 0.) + float(loss.detach())
        details.setdefault(task, []).append(logs)
        last_task = task
    if inspect:
        now = cpu_grad_vector(model)
        task_grads[last_task] = now - before
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5, error_if_nonfinite=True)
    opt.step()
    torch.cuda.synchronize()
    row = {"added_steps": added_step, "step": 4500 + added_step,
           "loss": totals, "details": details, "grad_norm_preclip": float(norm),
           "lr": opt.param_groups[0]["lr"], "seconds_per_update": time.perf_counter() - started}
    if inspect:
        total = sum(task_grads.values())
        row["weighted_task_grad_norm"] = {k: float(v.norm()) for k, v in task_grads.items()}
        row["task_gradient_projection"] = {k: float((g @ total) / total.square().sum().clamp_min(1e-12))
                                           for k, g in task_grads.items()}
        row["task_gradient_cosine"] = {a + b: float((task_grads[a] @ task_grads[b]) /
            (task_grads[a].norm() * task_grads[b].norm()).clamp_min(1e-12))
            for a, b in (("A", "B"), ("A", "C"), ("B", "C"))}
        row["b_style_content_gradient"] = probes
    return row


def train(args):
    if os.environ.get("WEAVE_STAGE1_GPU_ALLOWED") != "1":
        raise RuntimeError("GPU execution disabled; set WEAVE_STAGE1_GPU_ALLOWED=1 only inside the exclusive stage-1 worker")
    # Heavy project modules are deliberately imported after the explicit GPU gate.
    from exp.stage1_recipe.training.e_objectives import EObjectives
    from exp.v36_art_foundation.ecore_model import ReferenceECoreNet
    from src.utils.runtime import sanitize_cuda_env

    root = ROOT
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    data_root = ROOT
    cache_value = Path(config["paths"]["cache_root"])
    if not (data_root / cache_value).exists() and (Path("I:/WEAVE") / cache_value).exists():
        data_root = Path("I:/WEAVE")
    arm_cfg = config["arms"][args.arm]
    settings = config.get("training", {}).get("e", {})
    out = Path(args.run_dir)
    if not out.is_absolute(): out = root / out
    out.mkdir(parents=True, exist_ok=True)
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    seed = int(arm_cfg.get("seed", config.get("seed", 20263800)))
    latest = out / "latest.pt"
    state_path = out / "state.json"
    # Resume uses the original deadline. A new command never grants extra time.
    previous = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() and not args.profile else None
    limit = float(args.deadline_epoch) if args.deadline_epoch is not None else time.time() + 4500
    if previous:
        if previous.get("config_hash") != signature or previous.get("arm") != args.arm:
            raise ValueError("Training config/arm changed; use a new run directory")
        limit = min(limit, float(previous["deadline_epoch"]))
        if previous.get("status") in ("complete", "deadline"):
            return previous
    if time.time() >= limit - 20:
        raise TimeoutError("Reference-E deadline has no remaining save reserve")
    sanitize_cuda_env()
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    obj = EObjectives(config, args.arm, root)
    if args.audit_contract:
        result = audit_contract(obj, args.micro)
        audit_path = Path(args.audit_output)
        write_json(audit_path if audit_path.is_absolute() else root / audit_path, result)
        print(json.dumps(result), flush=True)
        return result
    if data_root != root:
        obj.cache_root = data_root / cache_value
        obj.meta = torch.load(obj.cache_root / "seen_train_index.pt", map_location="cpu", weights_only=False)
        obj.priors = torch.load(obj.cache_root / "v36_seen_train_priors.pt", map_location="cpu", weights_only=False)
        obj.priors = {k: v.to("cuda").float() if torch.is_tensor(v) else v for k, v in obj.priors.items()}
        obj.arrays = {k: np.load(data_root / obj.meta[k + "_rel"], mmap_mode="r") for k in ("z_gt", "q32", "stat", "stat_crop")}
        obj.valid_idx = np.asarray(obj.meta["valid_idx"])
        probs = np.asarray(obj.meta["sample_probs"])[obj.valid_idx]
        obj.probs = probs / probs.sum()
        obj.artist_nbrs = np.asarray(obj.meta["artist_nbrs"])
        obj.has_nbr = obj.artist_nbrs[:, 0] >= 0
    if args.profile:
        # Each micro/checkpointing candidate measures fresh proxy-cache population;
        # otherwise later candidates would appear faster only because the first filled it.
        obj.proxy_cache = out / "profile_proxy_cache"
        obj.proxy_cache.mkdir(parents=True, exist_ok=True)
    model = ReferenceECoreNet(obj.priors["stat_mean"], obj.priors["stat_std"]).cuda().train()
    initial = Path(config["paths"]["e_initial"])
    if not initial.is_absolute(): initial = data_root / initial
    external_resume = Path(args.resume_from) if args.resume_from else None
    if external_resume is not None and not external_resume.is_absolute(): external_resume = data_root / external_resume
    if external_resume is not None and (args.profile or args.deadline_epoch is None):
        raise ValueError("R0 --resume-from requires a separate explicit stage deadline and cannot be used for profiling")
    resume_source = latest if latest.exists() and not args.profile else external_resume
    ck = torch.load(resume_source or initial, map_location="cpu", weights_only=False)
    resumed = resume_source is not None
    if not resumed and (int(ck["step"]) != 4500 or "ecore" not in ck):
        raise ValueError("Reference-E initial checkpoint must contain online ecore at step 4500")
    model.load_state_dict(ck["online"] if resumed else ck["ecore"])
    decays = tuple(float(v) for v in settings.get("ema_decays", [.995, .999]))
    emas = {str(v): copy.deepcopy(model).eval().requires_grad_(False) for v in decays}
    opt = torch.optim.AdamW(model.parameters(), lr=float(arm_cfg.get("lr", 1e-4)),
                           betas=(.9, .99), weight_decay=1e-4)
    added, history = 0, []
    if resumed:
        if ck["config_hash"] != signature or ck["arm"] != args.arm:
            raise ValueError("Checkpoint and requested config differ")
        added = int(ck["added_steps"])
        if resume_source == latest:
            limit = min(limit, ck["deadline_epoch"])
        for name, ema in emas.items():
            ema.load_state_dict(ck["emas"][name])
        opt.load_state_dict(ck["optimizer"])
        obj.load_state_dict(ck["sampler"])
        restore_rng(ck["rng"])
        history = ck.get("history", [])
    del ck
    if resumed and not (out / "matched.pt").exists() and added >= args.steps:
        raise ValueError("Continuation matched target must exceed source added_steps; do not relabel inherited updates")
    warmup = int(settings.get("warmup_steps", 100))
    lr = float(arm_cfg.get("lr", 1e-4))
    save_every = int(settings.get("save_every", 250))
    monitor_every = int(settings.get("monitor_every", 250))
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    torch.cuda.reset_peak_memory_stats()
    start_added = added
    started = time.perf_counter()
    stage_started = previous.get("started_epoch", time.time()) if previous else time.time()
    profile_times = []

    def save(status, snapshot=False):
        row = {"arm": args.arm, "status": status, "step": 4500 + added, "added_steps": added,
               "pid": os.getpid(), "started_epoch": stage_started, "updated_epoch": time.time(),
               "deadline_epoch": limit, "config_hash": signature, "seed": seed,
               "matched_steps": args.steps, "matched_reached": (out / "matched.pt").exists(),
               "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
               "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
               "exposure": obj.exposure_summary(), "profile_only": False,
               "updates_per_s": (added - start_added) / max(1e-8, time.perf_counter() - started)}
        checkpoint = {"schema": "stage1_reference_e_v1", "status": status, "step": 4500 + added, "added_steps": added,
                      "arm": args.arm, "seed": seed, "config_hash": signature, "config": config,
                      "online": model.state_dict(), "emas": {k: v.state_dict() for k, v in emas.items()},
                      "optimizer": opt.state_dict(), "scheduler": {"kind": "warmup_constant", "warmup_steps": warmup,
                                                                   "base_lr": lr, "last_step": added},
                      "rng": rng_state(), "sampler": obj.state_dict(), "deadline_epoch": limit,
                      "history": history, "matched_steps": args.steps,
                      "initial": str(initial), "initial_online_step": 4500,
                      "continuation_source": str(external_resume) if external_resume else None,
                      "fresh_optimizer_at_initial": True}
        save_torch(latest, checkpoint)
        if snapshot:
            # Small evaluation artifacts preserve actual matched-step and end-of-time weights.
            for decay, ema in emas.items():
                save_torch(out / "snapshots" / f"step{added:06d}_ema{decay}.pt",
                           {"step": 4500 + added, "added_steps": added, "ecore_ema": ema.state_dict(),
                            "ema_decay": float(decay), "seed": seed, "arm": args.arm,
                            "config_hash": signature, "source_latest": str(latest)})
        write_json(state_path, row)
        return row

    if not args.profile:
        save("running", snapshot=True)
    iterations = args.profile_warmup + args.profile_steps if args.profile else (args.max_steps or float("inf"))
    while (added - start_added if args.profile else added) < iterations and time.time() < limit - 20:
        if stop[0] or (out / "STOP").exists():
            break
        update_started = time.perf_counter()
        added += 1
        inspect = (not args.profile) and (added == 1 or added % monitor_every == 0)
        row = logical_update(obj, model, opt, added, micro=args.micro, checkpointing=args.checkpointing,
                             lr=lr, warmup=warmup, inspect=inspect)
        with torch.no_grad():
            online = [p.detach() for p in model.parameters()]
            for decay, ema in emas.items():
                torch._foreach_lerp_(list(ema.parameters()), online, 1 - float(decay))
                for a, b in zip(ema.buffers(), model.buffers()):
                    a.copy_(b)
        torch.cuda.synchronize()
        row["seconds_per_update"] = time.perf_counter() - update_started
        if args.profile:
            if added - start_added > args.profile_warmup:
                profile_times.append(row["seconds_per_update"])
        else:
            row.update(arm=args.arm, peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                       updated_epoch=time.time())
            with (out / "train.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps({k: v for k, v in row.items() if k != "details"}), flush=True)
            if inspect:
                history.append(row)
            # Cheap heartbeat is distinct from expensive recoverable checkpoints.
            write_json(state_path, {"arm": args.arm, "status": "running", "pid": os.getpid(),
                       "step": 4500 + added, "added_steps": added, "deadline_epoch": limit,
                       "started_epoch": stage_started, "updated_epoch": time.time(), "config_hash": signature,
                       "last_step_seconds": row["seconds_per_update"]})
            if added == args.steps:
                save("running", snapshot=True)
                copy_checkpoint(latest, out / "matched.pt")
            elif added % save_every == 0:
                save("running", snapshot=True)
        if torch.cuda.max_memory_allocated() > 10 * 2**30:
            if not args.profile:
                save("vram_limit", snapshot=True)
            raise MemoryError("Reference-E exceeded the 10GiB allocated budget; choose a smaller micro")
    if args.profile:
        if len(profile_times) != args.profile_steps:
            raise TimeoutError("Profile did not complete all declared measured updates")
        seconds = float(np.median(profile_times))
        result = {"arm": args.arm, "micro": args.micro, "checkpointing": args.checkpointing,
                  "profile_only": True, "profile_steps": args.profile_steps, "warmup": args.profile_warmup,
                  "seconds_per_update": seconds,
                  "seconds_per_update_p95": float(np.percentile(profile_times, 95)),
                  "samples_per_s": sum(int(settings.get(k, v)) for k, v in
                      (("batch_a", 6), ("batch_b", 4), ("batch_c", 4))) / seconds,
                  "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                  "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                  "stable": torch.cuda.max_memory_allocated() <= 10 * 2**30,
                  "formal_checkpoint_written": False}
        result["proxy_cache_mode"] = "isolated_per_profile_run; cold population for unseen updates"
        profile_path = Path(args.profile_output)
        write_json(profile_path if profile_path.is_absolute() else root / profile_path, result)
        print(json.dumps(result), flush=True)
        return result
    status = "interrupted" if stop[0] or (out / "STOP").exists() else ("complete" if (out / "matched.pt").exists() else "insufficient_updates")
    result = save(status, snapshot=True)
    if status == "complete":
        copy_checkpoint(latest, out / "equal_time.pt")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--arm", choices=("E0", "E1", "E2", "E3"), required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--resume-from", help="Explicit R0 continuation from a full latest.pt into a new run directory")
    parser.add_argument("--steps", type=int, default=1000, help="Matched additional-update checkpoint after shared step4500; continue to deadline for equal-time result")
    parser.add_argument("--max-steps", type=int, help="Optional hard additional-update cap; default continues to the deadline")
    parser.add_argument("--deadline-epoch", type=float)
    parser.add_argument("--micro", type=int, choices=(1, 2, 4, 6), default=1)
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-steps", type=int, default=5)
    parser.add_argument("--profile-warmup", type=int, default=1)
    parser.add_argument("--profile-output", default="runs/stage1_recipe/e_profile.json")
    parser.add_argument("--audit-contract", action="store_true", help="Audit a small deterministic A batch without training")
    parser.add_argument("--audit-output", default="runs/stage1_recipe/e_contract_audit.json")
    args = parser.parse_args()
    if args.steps < 1 or args.profile_steps < 1 or args.micro < 1:
        parser.error("Update counts and micro must be positive")
    train(args)


if __name__ == "__main__":
    main()
