"""Deadline-aware E0/E1 continuation with full RNG state and exposure accounting."""
import argparse
import copy
import json
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch
from exp.v36_art_foundation import CACHE_DIR, CKPT_DIR, ROOT
from exp.v36_art_foundation.ecore_model import ReferenceECoreNet
from exp.v36_art_foundation.train_ecore import _atomic_save, _eval_seen_dev_ecore
from exp.v37_editing import OUT, SEED, atomic_json, read_manifest
from exp.v37_editing.training.objectives import Objectives, grad_vector
from src.utils.runtime import sanitize_cuda_env, seed_all


def dev_pack():
    d = json.loads((CACHE_DIR / "v36_caches_ready.json").read_text())["dev_info"]
    z, q, s = [np.load(ROOT / d["paths"][k], mmap_mode="r") for k in ("z_gt", "q32", "stat")]
    ii = np.arange(32)
    return {"zc": torch.from_numpy(np.array(z[ii])).float(), "q32": torch.from_numpy(np.array(q[ii])).float(),
            "zra": torch.from_numpy(np.array(z[(ii + 123) % len(z)])).float(),
            "zrb": torch.from_numpy(np.array(z[(ii + 317) % len(z)])).float(),
            "sta": torch.from_numpy(np.array(s[(ii + 123) % len(z)])).float(),
            "stb": torch.from_numpy(np.array(s[(ii + 317) % len(z)])).float(),
            "eps": torch.randn((32, 4, 64, 64), generator=torch.Generator().manual_seed(20263699))}


def setup(route):
    sanitize_cuda_env()
    seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    obj = Objectives(route)
    model = ReferenceECoreNet(obj.pool.priors["stat_mean"], obj.pool.priors["stat_std"]).cuda().train()
    return obj, model


def audit(batches=12):
    obj, model = setup("E0")
    ck = torch.load(CKPT_DIR / "v36_ecore_latest.pt", map_location="cpu", weights_only=False)
    assert ck["step"] == 4500
    model.load_state_dict(ck["ecore"])
    del ck
    rows = []
    torch.cuda.reset_peak_memory_stats()
    for i in range(batches):
        torch.cuda.synchronize()
        started = time.perf_counter()
        grads, losses, details = {}, {}, {}
        for task, loss, detail in obj.task_losses(model, 4501 + i):
            model.zero_grad(set_to_none=True)
            loss.backward()
            grads[task] = grad_vector(model)
            losses[task], details[task] = float(loss.detach()), detail
        total = sum(grads.values())
        row = {"batch": i, "loss": losses, "details": details,
               "grad_norm": {k: float(v.norm()) for k, v in grads.items()},
               "grad_cos": {a + b: float(torch.nn.functional.cosine_similarity(grads[a], grads[b], dim=0))
                            for a, b in (("A", "B"), ("A", "C"), ("B", "C"))},
               "projection_share": {k: float((v @ total) / total.square().sum().clamp_min(1e-12)) for k, v in grads.items()}}
        torch.cuda.synchronize()
        row["seconds"] = time.perf_counter() - started
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = {"kind": "frozen_4500_train_weights_no_optimizer_updates", "rows": rows,
              "warm_step_seconds_mean": float(np.mean([r["seconds"] for r in rows[2:]])),
              "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
              "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
              "source_exposures": obj.exposure_summary(),
              "source_hashes": read_manifest()["source_hashes"],
              "findings": ["Task A uses target Q32 with destylized input; retained in E0 and E1 A",
                           "Task C strength U(.10,.70), not zero; exact identity only at lambda=0",
                           "E delta has no tanh/clamp; amplitude not mathematically capped",
                           "P style LayerNorm followed by L2 normalization and per-layer AdaGN",
                           "V36 sampler inspected: Euler first + AB2, actual 6 batched forwards, 18 CFG branch-equivalents"]}
    atomic_json(OUT / "audit.json", report)


def inverse_smoke():
    obj, model = setup("E1")
    ck = torch.load(OUT / "checkpoints/initial4500.pt", weights_only=False, map_location="cpu")
    model.load_state_dict(ck["ecore"])
    del ck
    # Only the first 24 fully materialized examples are used for this preflight.
    for k in ("target", "target_refs", "other_ref", "same_artist"):
        obj.inverse[k] = obj.inverse[k][:24]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, betas=(.9, .99), weight_decay=1e-4)
    losses = {}
    for task, loss, _ in obj.task_losses(model, 4501):
        loss.backward()
        losses[task] = float(loss.detach())
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5, error_if_nonfinite=True)
    opt.step()
    atomic_json(OUT / "inverse_preflight.json", {"loss": losses, "grad_norm": float(norm),
                "saved_training_weights": False, "n_fixed_pairs": 24})


def train(route, max_steps=15000, budget_s=9000, extension=False):
    manifest = read_manifest()
    stage_start = time.time()
    limit = min(stage_start + budget_s, manifest["eval_reserved_from_epoch"] - 30)
    out = OUT / "checkpoints"
    out.mkdir(exist_ok=True)
    runstate = OUT / f"{route}_state.json"
    prior_state = json.loads(runstate.read_text()) if runstate.exists() else None
    if prior_state and not extension:
        limit = min(limit, prior_state["stage_deadline_epoch"])
        if prior_state.get("status") == "complete":
            return
    obj, model = setup(route)
    latest = out / f"{route}_latest.pt"
    initial = out / "initial4500.pt"
    ck = torch.load(latest if latest.exists() else initial, map_location="cpu", weights_only=False)
    step = int(ck["step"])
    if not latest.exists():
        assert step == 4500, "Both arms must fork the same 4500-step weights"
    model.load_state_dict(ck["ecore"])
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    ema.load_state_dict(ck["ecore_ema"])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(.9, .99), weight_decay=1e-4)
    opt.load_state_dict(ck["opt"])
    history = ck.get("v37_history", [])
    if "rng" in ck:
        st = ck["rng"]
        obj.rng.bit_generator.state = st["numpy_generator"]
        obj.gen.set_state(st["proxy_generator"])
        random.setstate(st["python"])
        np.random.set_state(st["numpy_global"])
        torch.set_rng_state(st["torch_cpu"])
        torch.cuda.set_rng_state_all(st["torch_cuda"])
        obj.exposed = ck["exposed"]
    # V36 lacks complete torch RNG state; both new arms begin at the same declared seed.
    del ck
    dp = dev_pack()
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    start_step, started = step, time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    def save(status, snapshot=False):
        rng = {"numpy_generator": obj.rng.bit_generator.state, "proxy_generator": obj.gen.get_state(),
               "python": random.getstate(), "numpy_global": np.random.get_state(),
               "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all()}
        _atomic_save({"step": step, "ecore": model.state_dict(), "ecore_ema": ema.state_dict(),
                      "opt": opt.state_dict(), "rng": rng, "exposed": obj.exposed,
                      "v37_history": history, "route": route, "initial": str(initial),
                      "stage_deadline_epoch": limit}, latest)
        if snapshot:
            _atomic_save({"step": step, "ecore_ema": ema.state_dict(), "route": route,
                          "history": history}, out / f"{route}_step{step:05d}.pt")
        atomic_json(runstate, {"route": route, "step": step, "status": status,
                              "stage_deadline_epoch": limit, "updated_epoch": time.time(),
                              "additional_exposure": obj.exposure_summary(), "history": history,
                              "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30})

    save("running")
    while step < max_steps and time.time() < limit - 15 and not stop[0]:
        step += 1
        extra = step - 4500
        lr = 1e-4 * min(1., extra / 250.) * max(.5, 1. - .5 * max(0, extra - 250) / 15750)
        for pg in opt.param_groups:
            pg["lr"] = lr
        opt.zero_grad(set_to_none=True)
        losses, details, gradients = {}, {}, {}
        inspect = step == start_step + 1 or step % 500 == 0
        before = grad_vector(model) if inspect else None
        for task, loss, detail in obj.task_losses(model, step):
            loss.backward()
            losses[task], details[task] = float(loss.detach()), detail
            if inspect:
                now = grad_vector(model)
                gradients[task] = float((now - before).norm())
                before = now
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5, error_if_nonfinite=True)
        opt.step()
        with torch.no_grad():
            torch._foreach_lerp_(list(ema.parameters()), [p.detach() for p in model.parameters()], .005)
        if step % 100 == 0 or inspect:
            row = {"step": step, "loss": losses, "lr": lr, "grad_norm_preclip": float(norm),
                   "task_grad_norm": gradients, "updates_per_s": (step - start_step) / (time.perf_counter() - started),
                   "peak_gib": torch.cuda.max_memory_allocated() / 2**30, "details": details}
            print(json.dumps(row), flush=True)
        if step % 500 == 0:
            # Evaluation must not perturb subsequent training randomness.
            cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
            d = _eval_seen_dev_ecore(ema, obj.priors, obj.dec, obj.critic, dp)
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state_all(cuda_rng)
            d.update(step=step, route=route)
            history.append(d)
            print("DEV " + json.dumps(d), flush=True)
        if step % 250 == 0:
            save("running", snapshot=step % 500 == 0 or step in (7500, 10500, 15000))
    save("complete" if not stop[0] else "interrupted", snapshot=True)
    _atomic_save({"step": step, "ecore_ema": ema.state_dict(), "history": history,
                  "route": route}, out / f"{route}_final.pt")
    print(f"{route} stopped safely at {step}; absolute experiment deadline unchanged", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--audit", action="store_true")
    p.add_argument("--smoke-inverse", action="store_true")
    p.add_argument("--route", choices=("E0", "E1"), default="E0")
    p.add_argument("--budget-s", type=int, default=9000)
    p.add_argument("--max-steps", type=int, default=15000)
    p.add_argument("--extension", action="store_true")
    a = p.parse_args()
    if a.audit: audit()
    elif a.smoke_inverse: inverse_smoke()
    else: train(a.route, a.max_steps, a.budget_s, a.extension)
