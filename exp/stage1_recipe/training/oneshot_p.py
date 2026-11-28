"""Controlled V36 One-shot P continuations; imports/help never inspect a GPU.

No worker starts on import. All three main arms fork the same declared V36
checkpoint. Sampling is synchronous at the logical effective batch boundary so
saved sampler and RNG state describe exactly the next update (no prefetch debt).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
os.environ.setdefault("PYTHONUTF8", "1")
import random
import signal
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RATIOS = {"p1": (.30, .40, .15, .15), "p2": (.10, .30, .30, .30)}
EMA_DECAYS = (.999, .9995)


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def run_root(config_path: str | Path) -> Path:
    return ROOT


def asset_root(config: dict) -> Path:
    env = os.environ.get("WEAVE_STAGE1_ASSET_ROOT")
    if env:
        return Path(env)
    cache = Path(config["paths"]["cache_root"])
    if resolve(cache).exists():
        return ROOT
    legacy_root = Path("I:/WEAVE")
    if (legacy_root / cache).exists():
        return legacy_root
    return ROOT


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def arm_recipe(config: dict, arm: str) -> dict:
    if arm not in ("P0", "P1", "P2", "P3"):
        raise ValueError(f"Not a One-shot P arm: {arm}")
    defaults = {"P0": (1e-4, 1.5e-4, .05), "P1": (5e-5, 7.5e-5, .05)}
    raw = dict(config.get("arms", {}).get(arm, {}))
    if arm in defaults:
        lr, style_lr, sigma = defaults[arm]
    else:
        # Adaptive arms must be concretized from Seen-Dev, never silently pick LR.
        if "lr_p" not in raw:
            raise ValueError(f"{arm} requires resolved arms.{arm}.lr_p from Seen-Dev selection")
        lr = float(raw["lr_p"])
        style_lr = lr * (2.0 if arm == "P3" else 1.5)
        sigma = .025 if arm == "P2" else .05
    training = config.get("training", {})
    shared = {**config.get("p", {}), **training.get("p", {})}
    for source, target in (("batch_size", "effective_batch"), ("warmup_steps", "warmup"),
                           ("grad_clip", "clip"), ("y16_dropout", "y16_perturb")):
        if source in shared:
            shared[target] = shared.pop(source)
    result = {"lr_p": lr, "lr_style": style_lr, "q_noise": sigma,
              "effective_batch": 96, "warmup": 100, "clip": 1.,
              "content_dropout": .30, "style_dropout": .15, "y16_perturb": .40,
              "seed": int(config.get("seed", 20263801)), "ratios_stage": "p2",
              "save_every": 250, "monitor_every": 250,
              "reserve_seconds": int(training.get("save_reserve_s", 60)),
              **shared, **raw}
    if result["effective_batch"] != 96:
        raise ValueError("Stage 1 fixes One-shot P effective batch at 96")
    if result["warmup"] != 100:
        raise ValueError("All short-window arms use the same 100-update warmup")
    if tuple(float(v) for v in result.get("ema_decays", EMA_DECAYS)) != EMA_DECAYS:
        raise ValueError("Stage 1 fixes P EMA decays at .999/.9995")
    for key, value in (("content_dropout", .30), ("style_dropout", .15), ("y16_perturb", .40)):
        if float(result[key]) != value:
            raise ValueError(f"{key} must remain {value} in stage 1")
    return result


def identity_hash(config: dict, recipe: dict, arm: str) -> str:
    frozen = {"paths": config["paths"], "p": recipe, "arm": arm,
              "sampler": "logical_effective_batch_sync_v1"}
    return hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()


def task_ratios(recipe: dict) -> tuple[float, ...]:
    explicit = recipe.get("task_ratios")
    if explicit is None:
        stage = recipe.get("ratios_stage", "p2")
        if stage not in DEFAULT_RATIOS:
            raise ValueError(f"Unknown common initialization stage {stage!r}")
        explicit = DEFAULT_RATIOS[stage]
    ratios = tuple(float(v) for v in explicit)
    if len(ratios) != 4 or any(v < 0 for v in ratios) or abs(sum(ratios)-1.) > 1e-8:
        raise ValueError("T0/T1/T2/T3 ratios must be four nonnegative fractions summing to one")
    return ratios


class Session:
    """Explicitly constructed training runtime; CUDA is touched only here."""

    def __init__(self, config: dict, recipe: dict, *, micro: int, checkpointing: bool,
                 device: str = "cuda", root: Path = ROOT):
        if device.startswith("cuda") and os.environ.get("WEAVE_STAGE1_GPU_ALLOWED") != "1":
            raise RuntimeError("GPU execution disabled; the serial queue must explicitly enable WEAVE_STAGE1_GPU_ALLOWED=1")
        import numpy as np
        import torch
        from exp.v36_art_foundation import train_foundation as legacy
        from exp.v36_art_foundation.models import FoundationPCoreNet, StyleMLPEncoder
        from src.utils.runtime import sanitize_cuda_env

        self.np, self.torch, self.legacy = np, torch, legacy
        self.root = Path(root)
        self.asset_root = asset_root(config)
        self.device, self.recipe, self.micro = device, recipe, int(micro)
        self.eff = int(recipe["effective_batch"])
        if self.micro <= 0 or self.eff % self.micro:
            raise ValueError("micro must divide effective batch 96")
        sanitize_cuda_env()
        self.seed = int(recipe["seed"])
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.benchmark = True
        cache = Path(config["paths"].get("cache_root", "runs/v36_art_foundation/cache"))
        if not cache.is_absolute(): cache = self.asset_root / cache
        self.cache_root = cache
        old_cache = legacy.CACHE_DIR
        try:
            legacy.CACHE_DIR = cache
            self.data = legacy._MemmapDataset(seed=self.seed)
        finally:
            legacy.CACHE_DIR = old_cache
        initial = Path(config["paths"]["p_initial"])
        if not initial.is_absolute(): initial = self.asset_root / initial
        if not initial.is_file():
            raise FileNotFoundError(f"Common P initialization missing: {initial}")
        self.initial = initial
        ck = torch.load(initial, map_location="cpu", weights_only=False)
        self.initial_step = int(ck.get("step", 0))
        self.initial_state_label = "EMA" if "pcore_ema" in ck else "online"
        self.pcore = FoundationPCoreNet(
            base_dim=208, style_dim=256, cond_dim=384, use_gamma=True,
            normalize_style=True, per_layer_adagn=True, use_checkpoint=checkpointing,
        ).to(device).train()
        priors = self.data.priors
        self.style = StyleMLPEncoder(1561, 512, 256, priors["stat_mean"], priors["stat_std"]).to(device).train()
        self.pcore.load_state_dict(ck.get("pcore_ema", ck.get("pcore")), strict=True)
        self.style.load_state_dict(ck.get("style_enc_ema", ck.get("style_enc")), strict=True)
        self.ratios = task_ratios(recipe)
        self.params = list(self.pcore.parameters()) + list(self.style.parameters())
        self.opt = torch.optim.AdamW([
            {"params": list(self.pcore.parameters()), "lr": float(recipe["lr_p"])},
            {"params": list(self.style.parameters()), "lr": float(recipe["lr_style"])},
        ], betas=(.9, .999), weight_decay=.01)
        self.emas = {}
        for decay in EMA_DECAYS:
            self.emas[str(decay)] = (
                copy.deepcopy(self.pcore).eval().requires_grad_(False),
                copy.deepcopy(self.style).eval().requires_grad_(False),
            )
        self.rng = np.random.default_rng(self.seed + 333)
        self.cursor = {"pos": 0, "perm": self.rng.permutation(self.data.valid_idx)}
        self.updates = 0
        self.scheduler = {"kind": "100_update_warmup_then_constant", "last_step": 0,
                          "base_lrs": [float(recipe["lr_p"]), float(recipe["lr_style"])]}
        # Re-seed after model construction: all arms see common sampling/noise.
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(self.seed)
        del ck

    def rng_state(self) -> dict:
        t = self.torch
        return {"python": random.getstate(), "numpy": self.np.random.get_state(),
                "sampler_numpy": self.rng.bit_generator.state,
                "torch_cpu": t.get_rng_state(),
                "torch_cuda": t.cuda.get_rng_state_all() if self.device.startswith("cuda") else []}

    def restore_rng(self, state: dict) -> None:
        random.setstate(state["python"])
        self.np.random.set_state(state["numpy"])
        self.rng.bit_generator.state = state["sampler_numpy"]
        self.torch.set_rng_state(state["torch_cpu"])
        if self.device.startswith("cuda"):
            self.torch.cuda.set_rng_state_all(state["torch_cuda"])

    def checkpoint(self) -> dict:
        return {"pcore": self.pcore.state_dict(), "style_enc": self.style.state_dict(),
                "emas": {k: {"pcore": p.state_dict(), "style_enc": s.state_dict()}
                         for k, (p, s) in self.emas.items()},
                "optimizer": self.opt.state_dict(), "scheduler": self.scheduler,
                "rng": self.rng_state(), "sampler": copy.deepcopy(self.cursor),
                "updates": self.updates, "initial_step": self.initial_step,
                "added_steps": self.updates,
                "initial_state_label": self.initial_state_label,
                "source_exposures": self.updates * self.eff,
                "sampler_contract": "full96_once_per_update_no_async_prefetch"}

    def restore(self, ck: dict) -> None:
        if "micro" in ck and int(ck["micro"]) != self.micro:
            raise ValueError("Resume micro changed; internal stochastic forwards would no longer reproduce the saved trajectory")
        if "checkpointing" in ck and bool(ck["checkpointing"]) != bool(self.pcore.use_checkpoint):
            raise ValueError("Resume checkpointing configuration changed")
        self.pcore.load_state_dict(ck["pcore"], strict=True)
        self.style.load_state_dict(ck["style_enc"], strict=True)
        for key, (p, s) in self.emas.items():
            p.load_state_dict(ck["emas"][key]["pcore"], strict=True)
            s.load_state_dict(ck["emas"][key]["style_enc"], strict=True)
        self.opt.load_state_dict(ck["optimizer"])
        self.scheduler = ck["scheduler"]
        self.cursor = ck["sampler"]
        self.updates = int(ck["updates"])
        self.restore_rng(ck["rng"])

    def prepare_batch(self) -> dict:
        """Sample & perturb the full96 once; micro only controls activation use."""
        t, dev, n, r = self.torch, self.device, self.eff, self.recipe
        z, y, q, stat = self.data.sample_batch_fp16(self.eff, self.ratios, self.rng, self.cursor, pin=False)
        z, y, q, stat = [v.to(dev).float() for v in (z, y, q, stat)]
        y, _ = self.legacy._blur_or_drop_y16(y, drop_prob=float(r["y16_perturb"]))
        q = q + float(r["q_noise"]) * t.randn_like(q)
        q = q - q.mean(dim=(2, 3), keepdim=True)
        dc = t.rand(n, device=dev) < float(r["content_dropout"])
        ds = t.rand(n, device=dev) < float(r["style_dropout"])
        layout = t.where(dc, t.zeros(n, device=dev), .25 + .75 * t.rand(n, device=dev))
        gamma = t.where(t.rand(n, device=dev) < .40, t.ones(n, device=dev),
                        t.empty(n, device=dev).uniform_(.15, 1.))
        tau = t.rand(n, device=dev).clamp(.02, .98)
        eps = t.randn_like(z)
        zt = (1. - tau[:, None, None, None]) * eps + tau[:, None, None, None] * z
        return dict(z1=z, y=y, q=q, stat=stat, dc=dc, ds=ds, layout=layout,
                    gamma=gamma, tau=tau, zt=zt, velocity=z-eps)

    def update(self, *, inspect: bool = False) -> dict:
        from src.utils.runtime import bf16_autocast
        t = self.torch
        b = self.prepare_batch()
        self.pcore.train()
        self.style.train()
        self.opt.zero_grad(set_to_none=True)
        next_step = self.updates + 1
        scale = min(1., next_step / float(self.recipe["warmup"]))
        for pg, base in zip(self.opt.param_groups, self.scheduler["base_lrs"]):
            pg["lr"] = base * scale
        before = [p.detach().clone() for p in self.params] if inspect else None
        loss = t.zeros((), device=self.device)
        for start in range(0, self.eff, self.micro):
            sl = slice(start, start+self.micro)
            with bf16_autocast(self.device):
                style = self.style(b["stat"][sl])
                out = self.pcore(b["zt"][sl], b["y"][sl], b["q"][sl], style, b["tau"][sl],
                                 drop_style=b["ds"][sl], drop_content=b["dc"][sl],
                                 layout_scale=b["layout"][sl], gamma=b["gamma"][sl])
                part = self.legacy.pseudo_huber_loss(out, b["velocity"][sl], delta=.05)
            weight = self.micro / self.eff
            (part * weight).backward()
            loss = loss + part.detach() * weight
        norm = t.nn.utils.clip_grad_norm_(self.params, float(self.recipe["clip"]), error_if_nonfinite=True)
        self.opt.step()
        with t.no_grad():
            for decay, (p, s) in self.emas.items():
                self.legacy._ema_update(p, self.pcore, float(decay))
                self.legacy._ema_update(s, self.style, float(decay))
        self.updates = next_step
        self.scheduler["last_step"] = next_step
        result = {"loss_fm": float(loss), "grad_norm_preclip": float(norm),
                  "lr_p": self.opt.param_groups[0]["lr"], "lr_style": self.opt.param_groups[1]["lr"]}
        if inspect:
            n_p = len(list(self.pcore.parameters()))
            for name, low, high in (("pcore", 0, n_p), ("style", n_p, len(self.params))):
                num = sum((p.detach()-old).float().square().sum() for p, old in zip(self.params[low:high], before[low:high]))
                den = sum(old.float().square().sum() for old in before[low:high])
                result[name+"_relative_update"] = float((num/den.clamp_min(1e-20)).sqrt())
        return result

    def response_monitor(self) -> dict:
        """Small fixed Seen-Dev velocity probe, not a generative quality score."""
        from src.utils.runtime import bf16_autocast
        t, np = self.torch, self.np
        state = self.rng_state()
        try:
            # Fixed indices from training-side Seen-Dev only, never Core750.
            ready = json.loads((self.cache_root / "v36_caches_ready.json").read_text(encoding="utf-8"))
            info = ready["dev_info"]
            n = min(4, int(info["n"]))
            ids = np.arange(n)
            ids_b = (ids+123) % int(info["n"])
            def a(key, ix):
                path = Path(info["paths"][key])
                if not path.is_absolute(): path = self.asset_root / path
                mm = np.load(path, mmap_mode="r")
                return t.from_numpy(np.array(mm[ix])).to(self.device).float()
            z, y, q, st, st_b = a("z_gt", ids), a("y16", ids), a("q32", ids), a("stat", ids), a("stat", ids_b)
            gen = t.Generator(device=self.device).manual_seed(self.seed+911)
            eps = t.randn(z.shape, device=self.device, generator=gen)
            tau = t.full((n,), .35, device=self.device)
            rows = {}
            with t.no_grad(), bf16_autocast(self.device):
                for key, (p, s) in self.emas.items():
                    sa, sb = s(st), s(st_b)
                    va = p(.65*eps+.35*z, y, q, sa, tau)
                    vb = p(.65*eps+.35*z, y, q, sb, tau)
                    rows[key] = {"style_embedding_delta_rms": float((sa-sb).float().square().mean().sqrt()),
                                 "reference_velocity_delta_rms": float((va-vb).float().square().mean().sqrt()),
                                 "conditional_fm_error_t35": float(self.legacy.pseudo_huber_loss(va,z-eps))}
            return {"scope": "Seen-Dev4 fixed velocity diagnostic, not frontier quality", "emas": rows}
        finally:
            self.restore_rng(state)


def train(args: argparse.Namespace) -> None:
    config = read_config(args.config)
    recipe = arm_recipe(config, args.arm)
    root = run_root(args.config)
    data_root = asset_root(config)
    config["paths"]["cache_root"] = str(data_root / config["paths"]["cache_root"])
    if config["paths"].get("p_initial") and not Path(config["paths"]["p_initial"]).is_absolute():
        config["paths"]["p_initial"] = str(data_root / config["paths"]["p_initial"])
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute(): run_dir = root / run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path, latest = run_dir / "state.json", run_dir / "latest.pt"
    old = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    deadline = min(float(args.deadline_epoch), float(old.get("deadline_epoch", args.deadline_epoch)))
    signature = identity_hash(config, recipe, args.arm)
    if old and old.get("signature") != signature:
        raise ValueError("Run recipe changed; use a new run directory")
    if old.get("status") == "complete":
        print(json.dumps({"status": "already_complete", "arm": args.arm}), flush=True)
        return
    train_deadline = deadline - float(recipe["reserve_seconds"])
    if time.time() >= train_deadline:
        raise TimeoutError("No training budget remains; deadline is persistent across resume")
    # Pending explicit execution is the first point at which a CUDA runtime exists.
    runtime = Session(config, recipe, micro=args.micro, checkpointing=args.checkpointing, root=root)
    runtime.cache_root = Path(config["paths"].get("cache_root", "runs/v36_art_foundation/cache"))
    if not runtime.cache_root.is_absolute(): runtime.cache_root = data_root / runtime.cache_root
    t = runtime.torch
    if latest.exists():
        ck = t.load(latest, map_location="cpu", weights_only=False)
        if ck["signature"] != signature:
            raise ValueError("Checkpoint does not belong to this declared experiment")
        runtime.restore(ck)
        deadline = min(deadline, float(ck["deadline_epoch"]))
        del ck
    elif args.resume_from:
        # R0 continuation uses a NEW run directory and an explicit new deadline.
        # It restores weights, EMA, optimizer, scheduler, RNG and sampler. This
        # is continued training, never an independent seed replication.
        resume = Path(args.resume_from)
        if not resume.is_absolute(): resume = data_root / resume
        ck = t.load(resume, map_location="cpu", weights_only=False)
        if ck["signature"] != signature:
            raise ValueError("Continuation must preserve the exact original arm recipe and seed")
        runtime.restore(ck)
        del ck
    train_deadline = deadline - float(recipe["reserve_seconds"])
    if time.time() >= train_deadline:
        raise TimeoutError("Saved training deadline expired; resume cannot extend it")
    stop = [False]
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.__setitem__(0, True))
    started = time.perf_counter()
    initial_updates = runtime.updates
    t.cuda.reset_peak_memory_stats()
    history = list(old.get("history", []))

    def save(status: str, name: str = "latest.pt") -> None:
        payload = {**runtime.checkpoint(), "arm": args.arm, "recipe": recipe,
                   "signature": signature, "deadline_epoch": deadline,
                   "common_initial": config["paths"]["p_initial"], "history": history,
                   "matched_updates_target": args.steps, "micro": args.micro,
                   "checkpointing": args.checkpointing,
                   "continuation_source": args.resume_from}
        runtime.legacy._atomic_torch_save(payload, run_dir/name)
        atomic_json(state_path, {"status": status, "arm": args.arm, "step": runtime.updates,
                    "updates": runtime.updates, "added_steps": runtime.updates, "initial_step": runtime.initial_step,
                    "pid": os.getpid(),
                    "signature": signature, "deadline_epoch": deadline,
                    "updated_epoch": time.time(), "source_exposures": runtime.updates*runtime.eff,
                    "matched_updates_target": args.steps, "matched_saved": (run_dir/"matched.pt").exists(),
                    "matched_reached": (run_dir/"matched.pt").exists(),
                    "insufficient_updates": runtime.updates < 600,
                    "history": history, "checkpoint": str((run_dir/name).relative_to(ROOT))})

    save("running")
    try:
        while time.time() < train_deadline-30 and not stop[0] and not (run_dir/"STOP").exists():
            if args.max_steps and runtime.updates >= args.max_steps:
                break
            inspect = (runtime.updates+1) % int(recipe["monitor_every"]) == 0
            row = runtime.update(inspect=inspect)
            elapsed = time.perf_counter()-started
            row.update(arm=args.arm, step=runtime.updates, updated_epoch=time.time(),
                       samples_per_s=(runtime.updates-initial_updates)*runtime.eff/max(elapsed,1e-9),
                       peak_allocated_gib=t.cuda.max_memory_allocated()/2**30,
                       peak_reserved_gib=t.cuda.max_memory_reserved()/2**30)
            atomic_json(run_dir/"heartbeat.json", row)
            if inspect or runtime.updates == initial_updates+1:
                if row["peak_allocated_gib"] > 10.:
                    raise RuntimeError("Peak allocated exceeds the declared 10 GiB cap; reduce micro/checkpoint configuration")
                row["seen_dev"] = runtime.response_monitor()
                history.append(row)
                with (run_dir/"metrics.jsonl").open("a",encoding="utf-8") as fh:
                    fh.write(json.dumps(row,ensure_ascii=False)+"\n")
                print(json.dumps(row,ensure_ascii=False),flush=True)
            if runtime.updates == args.steps:
                save("running", "matched.pt")
            if runtime.updates % int(recipe["save_every"]) == 0:
                save("running")
        status = "interrupted" if stop[0] or (run_dir/"STOP").exists() else "complete"
        save(status)
        save(status, "equal_time.pt")
        atomic_json(run_dir/"completion.json", {"status": status,
                    "matched_reached": (run_dir/"matched.pt").exists(), "updates": runtime.updates,
                    "added_steps": runtime.updates, "insufficient_updates": runtime.updates < 600,
                    "deadline_epoch": deadline, "finished_epoch": time.time(),
                    "quality_status": "not_evaluated", "history": history})
    except BaseException as exc:
        # An exception may happen halfway through a logical update. Preserve the
        # last atomic completed-update checkpoint; its RNG/cursor are resumable.
        prior = json.loads(state_path.read_text(encoding="utf-8"))
        atomic_json(run_dir/"failure.json", {"failed_epoch": time.time(),
                    "error_type": type(exc).__name__, "error": str(exc),
                    "in_memory_updates": runtime.updates,
                    "resume_updates": prior["added_steps"],
                    "checkpoint_preserved": str(latest.relative_to(ROOT))})
        prior.update(status="failed", updated_epoch=time.time())
        atomic_json(state_path, prior)
        raise


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--asset-root", type=Path)
    ap.add_argument("--arm", choices=("P0","P1","P2","P3"), required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--steps", type=int, required=True, help="Common additional-update snapshot; continues to time deadline")
    ap.add_argument("--max-steps", type=int, default=0, help="Optional explicit upper bound; 0 means train to time deadline")
    ap.add_argument("--resume-from", help="R0 continuation from full latest.pt into a new run directory, with an explicit new deadline")
    ap.add_argument("--deadline-epoch", type=float, required=True)
    ap.add_argument("--micro", type=int, default=4)
    ap.add_argument("--checkpointing", action="store_true")
    return ap


if __name__ == "__main__":
    parsed = parser().parse_args()
    if parsed.asset_root is not None:
        os.environ["WEAVE_STAGE1_ASSET_ROOT"] = str(parsed.asset_root)
    if parsed.steps < 600:
        raise ValueError("Common update target below600: reduce arms, do not force the full matrix")
    train(parsed)
