"""WEAVE-v2 独立训练入口（V1–V5）。

不修改 ``run.py`` / ``trainer.py`` 的默认路径；只有显式运行本模块才会训练 v2。
所有产物写到 ``runs/v2/...``。

用法示例::

    python -m v2.train --config configs/v2/V3.json --variant V3 --steps 30 --synthetic
    python -m v2.train --config configs/v2/V5.json --variant V5 --stage distill \
        --teacher runs/v2/V3/step_020000.pt --steps 1000

V0 直接使用原入口，不经过本文件。
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from config_schema import load_experiment_config
from flow import FlowMatchingObjective
from model import build_model_from_config
from wavelet import dwt2_haar

from .config import V2Config, expand_variant_defaults
from .distill import DistillObjective
from .legacy_adapter import DifferentiableLegacyGenerator
from .model import V2Backbone, V2Generator, count_parameters
from .objective import V2DistributionObjective, identity_rows
from .style_stats import load_style_stats


class SyntheticLatentDataset(Dataset):
    """CPU smoke 用；真实训练不要使用。"""

    def __init__(self, length: int = 64, num_styles: int = 5, size: int = 32, identity_ratio: float = 0.1) -> None:
        self.length = int(length)
        self.num_styles = int(num_styles)
        self.size = int(size)
        self.identity_ratio = float(identity_ratio)
        g = torch.Generator().manual_seed(1234)
        self.content = torch.randn(self.length, 4, self.size, self.size, generator=g) * 0.2
        self.targets = [torch.randn(self.length, 4, self.size, self.size, generator=g) * 0.2 for _ in range(num_styles)]
        self.ids = torch.randint(0, self.num_styles, (self.length,), generator=g)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        target_id = int(self.ids[idx])
        identity = float(torch.rand(1).item() < self.identity_ratio)
        if identity:
            target_id = idx % self.num_styles
            target = self.content[idx]
            content_id = target_id
        else:
            target = self.targets[target_id][idx]
            content_id = idx % self.num_styles
        return {
            "content": self.content[idx],
            "target_style": target,
            "target_style_id": torch.tensor(target_id, dtype=torch.long),
            "source_style_id": torch.tensor(content_id, dtype=torch.long),
            "identity_mask": torch.tensor(identity, dtype=torch.float32),
        }


def _build_real_dataset(config, device: torch.device):
    import inspect

    from utils.dataset import AdaCUTLatentDataset

    data = config.data
    params = inspect.signature(AdaCUTLatentDataset.__init__).parameters
    kwargs: dict[str, Any] = {}
    for name in params:
        if name == "self":
            continue
        if hasattr(data, name):
            kwargs[name] = getattr(data, name)
    # 必须项与设备显式覆盖，避免 config 缺省导致的隐式默认
    kwargs["data_root"] = data.data_root
    kwargs["style_subdirs"] = data.style_subdirs
    kwargs["device"] = str(device)
    kwargs.setdefault("batch_size_hint", int(config.training.batch_size))
    kwargs["latent_cache_mode"] = str(getattr(data, "latent_cache_mode", "off"))
    kwargs["latent_cache_dir"] = str(getattr(data, "latent_cache_dir", "") or "")
    return AdaCUTLatentDataset(**kwargs)


def _move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def _infinite(loader: DataLoader) -> Iterator[dict[str, Any]]:
    while True:
        for batch in loader:
            yield batch


def _style_stats(v2cfg: V2Config, num_styles: int, device: torch.device):
    if not v2cfg.style_stats_path:
        return None
    bank = load_style_stats(v2cfg.style_stats_path, device=device)
    if bank.shape[0] < num_styles:
        raise ValueError(f"style stats bank has {bank.shape[0]} rows < num_styles={num_styles}")
    return bank[:num_styles]


def _bare_state_dict(model: nn.Module) -> dict:
    """V1 的 DifferentiableLegacyGenerator 会带 ``model.`` 前缀；存盘统一存裸键。"""
    if isinstance(model, DifferentiableLegacyGenerator):
        return model.model.state_dict()
    return model.state_dict()


def _checkpoint_payload(variant: str, v2cfg: V2Config, step: int, model: nn.Module, objective: nn.Module | None) -> dict:
    payload = {
        "schema_version": "v2-bare-keys-v1",
        "variant": variant,
        "v2_config": v2cfg.to_dict(),
        "step": int(step),
        "model": _bare_state_dict(model),
    }
    if objective is not None:
        payload["objective"] = {
            "nce_encoder": objective.nce_encoder.state_dict(),
            "nce": objective.nce.state_dict(),
            "discriminator": objective.discriminator.state_dict(),
        }
    return payload


def _save(payload: dict, out_dir: Path, step: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_dir / f"step_{step:06d}.pt")


def _subband_energies(x: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        ll, lh, hl, hh = dwt2_haar(x.float())
        return {
            "LL": float(ll.pow(2).mean()),
            "LH": float(lh.pow(2).mean()),
            "HL": float(hl.pow(2).mean()),
            "HH": float(hh.pow(2).mean()),
        }


def _diagnose(
    step: int,
    *,
    content: torch.Tensor,
    target: torch.Tensor,
    z_hat: torch.Tensor,
    ids: torch.Tensor,
    source_ids: Any,
    conditioning: dict[str, Any],
    g: dict[str, Any],
    d: dict[str, Any],
    step_seconds: list[float],
    metrics_path: Path,
) -> dict[str, Any]:
    """每 diag_every 步记录：恒等距离、D 真假 logit/准确率、四项损失、四子带能量、显存/速度。"""
    with torch.no_grad():
        dev = content.device
        ident = identity_rows(
            content.shape[0], target_style_id=ids, source_style_id=source_ids, conditioning=conditioning, device=dev
        )
        real = target.clone()
        if bool(ident.any()):
            real[ident] = content[ident]
        dist = (z_hat.float() - content.float()).flatten(1).norm(dim=1).mean().item()
        e_real = _subband_energies(real)
        e_fake = _subband_energies(z_hat)

    row: dict[str, Any] = {
        "step": int(step),
        "identity_dist": dist,
        "adv": _as_float(g["loss_adv"]),
        "patchnce": _as_float(g["loss_patchnce"]),
        "geometry": _as_float(g["loss_geometry"]),
        "identity": _as_float(g["loss_identity"]),
        "g_loss": _as_float(g["loss"]),
        "identity_fraction": _as_float(g["identity_fraction"]),
        "d_real_logit": _as_float(d["d_real_logit"]),
        "d_fake_logit": _as_float(d["d_fake_logit"]),
        "d_acc": _as_float(d["d_acc"]),
        "d_loss": _as_float(d["loss"]),
    }
    for band in ("LL", "LH", "HL", "HH"):
        row[f"E_real_{band}"] = e_real[band]
        row[f"E_fake_{band}"] = e_fake[band]
    if torch.cuda.is_available():
        row["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
        row["peak_mem_reserved_gb"] = torch.cuda.max_memory_reserved() / 1e9
    if step_seconds:
        row["sec_per_step"] = sum(step_seconds) / len(step_seconds)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        " | ".join(
            [
                f"[diag step {step}]",
                f"||z-zc||={dist:.4f}",
                f"adv={row['adv']:.3f}",
                f"nce={row['patchnce']:.3f}",
                f"geo={row['geometry']:.4f}",
                f"id={row['identity']:.4f}",
                f"D(real)={row['d_real_logit']:.3f}",
                f"D(fake)={row['d_fake_logit']:.3f}",
                f"Dacc={row['d_acc']:.3f}",
                f"E_real(LL/LH/HL/HH)={e_real['LL']:.4f}/{e_real['LH']:.4f}/{e_real['HL']:.4f}/{e_real['HH']:.4f}",
                f"E_fake(LL/LH/HL/HH)={e_fake['LL']:.4f}/{e_fake['LH']:.4f}/{e_fake['HL']:.4f}/{e_fake['HH']:.4f}",
                f"peak={row.get('peak_mem_gb', 0.0):.2f}GB",
                f"s/step={row.get('sec_per_step', 0.0):.3f}",
            ]
        ),
        flush=True,
    )
    return row


def _as_float(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().float().mean().item())
    return float(value)


def _accumulate_into(dst: dict[str, float], src: dict[str, Any], weight: float) -> None:
    """把一次 microbatch 的标量指标按权重累加进 ``dst``（用于梯度累积下的日志）。"""
    for key, value in src.items():
        if torch.is_tensor(value):
            value = float(value.detach().float().mean().item())
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            dst[key] = dst.get(key, 0.0) + float(value) * weight


def _fm_diagnose(step: int, merged: dict[str, float], step_seconds: list[float], metrics_path: Path) -> None:
    row: dict[str, Any] = {"step": int(step)}
    row.update({k: float(v) for k, v in merged.items()})
    if torch.cuda.is_available():
        row["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
        row["peak_mem_reserved_gb"] = torch.cuda.max_memory_reserved() / 1e9
    if step_seconds:
        row["sec_per_step"] = sum(step_seconds) / len(step_seconds)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        " | ".join(
            [
                f"[fm diag step {step}]",
                f"loss={row.get('loss', 0.0):.4f}",
                f"peak={row.get('peak_mem_gb', 0.0):.2f}GB",
                f"s/step={row.get('sec_per_step', 0.0):.3f}",
            ]
        ),
        flush=True,
    )


def _log(step: int, steps: int, metrics: dict[str, Any], t0: float) -> None:
    if step % 10 != 0 and step != 1 and step != steps:
        return
    parts = [f"step {step}/{steps}"]
    for key, value in metrics.items():
        if torch.is_tensor(value):
            value = float(value.detach().float().mean().item())
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            parts.append(f"{key}={float(value):.4f}")
    parts.append(f"elapsed={time.time() - t0:.1f}s")
    print(" | ".join(parts), flush=True)


def train_generator_variant(
    config,
    v2cfg: V2Config,
    variant: str,
    *,
    steps: int,
    device: torch.device,
    synthetic: bool,
    out_dir: Path,
    save_every: int,
    diag_every: int = 500,
    save_at: tuple[int, ...] = (),
) -> None:
    """V1（旧主干 + 新目标）与 V3/V4（新主干 + 新目标）。"""
    num_styles = int(config.model.num_styles)
    stats = _style_stats(v2cfg, num_styles, device)

    if variant == "V1":
        legacy = build_model_from_config(config.model, bridge_cfg=config.bridge)
        generator = DifferentiableLegacyGenerator(legacy, num_steps=int(v2cfg.legacy_ode_steps))
    else:
        generator = V2Generator(v2cfg, latent_channels=int(config.model.latent_channels), num_styles=num_styles)
    generator = generator.to(device)

    objective = V2DistributionObjective(
        v2cfg, latent_channels=int(config.model.latent_channels), num_styles=num_styles
    ).to(device)
    opt_g = torch.optim.Adam(
        list(generator.parameters()) + list(objective.nce_encoder.parameters()) + list(objective.nce.parameters()),
        lr=float(v2cfg.lr),
        betas=(0.5, 0.999),
    )
    opt_d = torch.optim.Adam(objective.discriminator.parameters(), lr=float(v2cfg.lr_disc), betas=(0.5, 0.999))

    dataset = (
        SyntheticLatentDataset(num_styles=num_styles, size=32)
        if synthetic
        else _build_real_dataset(config, device)
    )
    loader = DataLoader(dataset, batch_size=int(v2cfg.batch_size), shuffle=True, drop_last=True, num_workers=0)
    it = _infinite(loader)
    generator.train()
    objective.train()
    metrics_path = out_dir / "metrics.jsonl"
    step_seconds: list[float] = []
    accum = max(1, int(v2cfg.accumulation_steps))
    t0 = time.time()
    for step in range(1, int(steps) + 1):
        step_t0 = time.time()
        d_merged: dict[str, float] = {}
        g_merged: dict[str, float] = {}
        g_ctx: dict[str, Any] = {}

        # ---- D 阶段：accum 个 microbatch 累积后更新一次 ----
        # 必须在 D 反向之前清零：G 阶段会反传穿过 D 并留下 D 梯度，
        # 若留到下一次 D 更新，D 会被反向优化（Dacc 恒 0.5、adv 一路走负）。
        opt_d.zero_grad(set_to_none=True)
        for _ in range(accum):
            batch = _move(next(it), device)
            content = batch["content"]
            target = batch["target_style"]
            ids = batch["target_style_id"]
            source_ids = batch.get("source_style_id")
            conditioning = {"identity_mask": batch.get("identity_mask")}
            with torch.no_grad():
                fake_for_d = generator(content, style_id=ids, style_stats=stats)["z_hat"]
            d = objective.compute_discriminator_loss(
                content, target, ids, fake_for_d, source_style_id=source_ids, conditioning=conditioning
            )
            (d["loss"] / accum).backward()
            _accumulate_into(d_merged, d, 1.0 / accum)
        opt_d.step()

        # ---- G 阶段：另取 accum 个 microbatch（样本曝光量与有效 batch 对齐）----
        opt_g.zero_grad(set_to_none=True)
        for _ in range(accum):
            batch = _move(next(it), device)
            content = batch["content"]
            target = batch["target_style"]
            ids = batch["target_style_id"]
            source_ids = batch.get("source_style_id")
            conditioning = {"identity_mask": batch.get("identity_mask")}
            g = objective.compute_generator_loss(
                generator, content, target, ids, source_style_id=source_ids, conditioning=conditioning, style_stats=stats
            )
            (g["loss"] / accum).backward()
            _accumulate_into(g_merged, {k: v for k, v in g.items() if k != "z_hat"}, 1.0 / accum)
            g_ctx = {
                "content": content, "target": target, "z_hat": g["z_hat"], "ids": ids,
                "source_ids": source_ids, "conditioning": conditioning,
            }
        torch.nn.utils.clip_grad_norm_(list(generator.parameters()) + list(objective.nce_encoder.parameters()), 1.0)
        opt_g.step()
        opt_g.zero_grad(set_to_none=True)
        step_seconds.append(time.time() - step_t0)

        metrics = dict(g_merged)
        metrics.update({f"d_{k}": v for k, v in d_merged.items()})
        _log(step, steps, metrics, t0)
        is_diag = diag_every > 0 and (step % diag_every == 0 or step == steps or step == 1)
        if is_diag:
            _diagnose(
                step,
                content=g_ctx["content"],
                target=g_ctx["target"],
                z_hat=g_ctx["z_hat"],
                ids=g_ctx["ids"],
                source_ids=g_ctx["source_ids"],
                conditioning=g_ctx["conditioning"],
                g=g_merged,
                d=d_merged,
                step_seconds=step_seconds,
                metrics_path=metrics_path,
            )
            step_seconds.clear()
        if step in save_at or (save_every > 0 and step % save_every == 0):
            _save(_checkpoint_payload(variant, v2cfg, step, generator, objective), out_dir, step)

    if steps not in save_at:
        _save(_checkpoint_payload(variant, v2cfg, steps, generator, objective), out_dir, steps)


def train_fm_variant(
    config,
    v2cfg: V2Config,
    variant: str,
    *,
    steps: int,
    device: torch.device,
    synthetic: bool,
    out_dir: Path,
    save_every: int,
    diag_every: int = 500,
    save_at: tuple[int, ...] = (),
) -> None:
    """V2：原 FM 目标 + 新主干（velocity 模式）。"""
    num_styles = int(config.model.num_styles)
    backbone = V2Backbone(
        v2cfg,
        latent_channels=int(config.model.latent_channels),
        num_styles=num_styles,
        mode="velocity",
    ).to(device)
    objective = FlowMatchingObjective(config)
    opt = torch.optim.AdamW(backbone.parameters(), lr=float(v2cfg.lr))
    dataset = (
        SyntheticLatentDataset(num_styles=num_styles, size=32)
        if synthetic
        else _build_real_dataset(config, device)
    )
    loader = DataLoader(dataset, batch_size=int(v2cfg.batch_size), shuffle=True, drop_last=True, num_workers=0)
    it = _infinite(loader)
    backbone.train()
    metrics_path = out_dir / "metrics.jsonl"
    step_seconds: list[float] = []
    accum = max(1, int(v2cfg.accumulation_steps))
    t0 = time.time()
    for step in range(1, int(steps) + 1):
        step_t0 = time.time()
        merged: dict[str, float] = {}
        for _ in range(accum):
            batch = _move(next(it), device)
            metrics = objective.compute(
                backbone,
                content=batch["content"],
                target_style=batch["target_style"],
                target_style_id=batch["target_style_id"],
                source_style_id=batch.get("source_style_id"),
                conditioning={},
            )
            (metrics["loss"] / accum).backward()
            _accumulate_into(merged, metrics, 1.0 / accum)
        torch.nn.utils.clip_grad_norm_(backbone.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step_seconds.append(time.time() - step_t0)
        _log(step, steps, merged, t0)
        if diag_every > 0 and (step % diag_every == 0 or step == steps or step == 1):
            _fm_diagnose(step, merged, step_seconds, metrics_path)
            step_seconds.clear()
        if step in save_at or (save_every > 0 and step % save_every == 0):
            _save(_checkpoint_payload(variant, v2cfg, step, backbone, None), out_dir, step)
    if steps not in save_at:
        _save(_checkpoint_payload(variant, v2cfg, steps, backbone, None), out_dir, steps)


def train_distill(
    config,
    v2cfg: V2Config,
    *,
    teacher_path: str,
    steps: int,
    device: torch.device,
    synthetic: bool,
    out_dir: Path,
    save_every: int,
) -> None:
    """V5：G_φ (stopgrad) 端点 -> FM 蒸馏。必须显式 --stage distill。"""
    if not v2cfg.distill_enabled:
        raise RuntimeError("V5 distillation requires v2.distill_enabled=true (explicit opt-in)")
    if not teacher_path:
        raise RuntimeError("V5 distillation requires --teacher <checkpoint.pt>")
    num_styles = int(config.model.num_styles)
    teacher = V2Generator(v2cfg, latent_channels=int(config.model.latent_channels), num_styles=num_styles)
    payload = torch.load(teacher_path, map_location=device)
    teacher_variant = str(payload.get("variant", ""))
    if teacher_variant not in {"V3", "V4"}:
        raise RuntimeError(
            f"teacher variant {teacher_variant!r} is not a cross-style-evaluated V2 generator; "
            "pseudo targets / legacy checkpoints are not allowed as teachers"
        )
    teacher.load_state_dict(payload["model"], strict=True)
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = V2Backbone(
        v2cfg,
        latent_channels=int(config.model.latent_channels),
        num_styles=num_styles,
        mode="velocity",
    ).to(device)
    objective = DistillObjective(v2cfg)
    opt = torch.optim.AdamW(student.parameters(), lr=float(v2cfg.lr))
    stats = _style_stats(v2cfg, num_styles, device)
    dataset = (
        SyntheticLatentDataset(num_styles=num_styles, size=32)
        if synthetic
        else _build_real_dataset(config, device)
    )
    loader = DataLoader(dataset, batch_size=int(v2cfg.batch_size), shuffle=True, drop_last=True, num_workers=0)
    it = _infinite(loader)
    student.train()
    t0 = time.time()
    for step in range(1, int(steps) + 1):
        batch = _move(next(it), device)
        result = objective.compute(
            student,
            teacher,
            batch["content"],
            batch["target_style_id"],
            style_stats=stats,
        )
        opt.zero_grad(set_to_none=True)
        result["loss"].backward()
        opt.step()
        _log(step, steps, result, t0)
        if save_every > 0 and step % save_every == 0:
            _save(_checkpoint_payload("V5", v2cfg, step, student, None), out_dir, step)
    _save(_checkpoint_payload("V5", v2cfg, steps, student, None), out_dir, steps)


def train_swd_variant(
    config,
    v2cfg: V2Config,
    *,
    steps: int,
    device: torch.device,
    out_dir: Path,
    save_every: int,
    diag_every: int = 50,
    save_at: tuple[int, ...] = (),
) -> None:
    """V3S：单步残差 + 局部 SWD + 整图 MMD + 结构约束 + Identity。

    每步统一目标风格；真实特征来自离线库；生成侧 decode 保持梯度。
    """
    from utils.inference import load_vae

    from .data512 import StyleLatentPool
    from .features import global_descriptor, local_descriptors, normalize_global, normalize_local
    from .distances import swd2
    from .objective_swd import SWDObjective, decode_latents, load_library

    if not v2cfg.swd_library:
        raise SystemExit("V3S 需要 v2.swd_library（先跑 tools/build_style_library.py）")
    library = load_library(v2cfg.swd_library)
    pool = StyleLatentPool(v2cfg.latent_cache)
    num_styles = int(config.model.num_styles)
    if pool.num_styles != num_styles:
        raise SystemExit(f"latent pool styles={pool.num_styles} != model.num_styles={num_styles}")

    vae = load_vae(
        device=str(device), model_id="ema", compile_dtype=torch.float32,
        enable_xformers=True, compile_decoder=False,
    )
    vae.requires_grad_(False)
    vae.eval()

    generator = V2Generator(v2cfg, latent_channels=int(config.model.latent_channels), num_styles=num_styles).to(device)
    generator.train()
    objective = SWDObjective(v2cfg, library, device=device, seed=int(v2cfg.seed))
    opt = torch.optim.Adam(generator.parameters(), lr=float(v2cfg.lr), betas=(0.5, 0.999))
    rng = torch.Generator(device="cpu").manual_seed(int(v2cfg.seed))

    metrics_path = out_dir / "metrics.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    accum = max(1, int(v2cfg.accumulation_steps))
    b = int(v2cfg.batch_size)
    print(json.dumps({
        "variant": "V3S", "steps": steps, "microbatch": b, "accum": accum,
        "effective_batch": b * accum, "library": v2cfg.swd_library,
        "latent": v2cfg.latent_cache, "w_swd": v2cfg.w_swd, "w_mmd": v2cfg.w_mmd,
        "w_content": v2cfg.w_content, "eps": v2cfg.content_eps,
        "w_identity": v2cfg.w_identity, "zero_init_head": v2cfg.residual_head_zero_init,
    }, ensure_ascii=False), flush=True)

    step_seconds: list[float] = []
    t0 = time.time()
    for step in range(1, int(steps) + 1):
        step_t0 = time.time()
        target_style = (step - 1) % num_styles
        merged: dict[str, float] = {}
        for _ in range(accum):
            content = pool.sample_content(b, gen=rng).to(device)
            identity = pool.sample_style(target_style, min(2, b), gen=rng).to(device)
            row = objective.compute(
                generator, vae, content,
                target_style=target_style,
                identity_latents=identity,
                patch_samples=int(v2cfg.swd_patch_samples),
                real_images=int(v2cfg.swd_real_images),
            )
            (row.loss / accum).backward()
            _accumulate_into(merged, {
                "loss": row.loss, "swd": row.swd, "mmd": row.mmd,
                "structure": row.structure, "hinge": row.hinge, "identity": row.identity,
            }, 1.0 / accum)
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step_seconds.append(time.time() - step_t0)
        _log(step, steps, merged, t0)

        is_diag = diag_every > 0 and (step % diag_every == 0 or step == steps or step == 1)
        if is_diag:
            _swd_diagnose(
                step, generator, vae, objective, pool, target_style,
                batch=b, patch_samples=int(v2cfg.swd_patch_samples),
                real_images=int(v2cfg.swd_real_images), rng=rng,
                metrics_path=metrics_path, step_seconds=step_seconds,
            )
            step_seconds.clear()
        if step in save_at or (save_every > 0 and step % save_every == 0):
            _save(_checkpoint_payload("V3S", v2cfg, step, generator, None), out_dir, step)
    if steps not in save_at:
        _save(_checkpoint_payload("V3S", v2cfg, steps, generator, None), out_dir, steps)


@torch.no_grad()
def _swd_diagnose(
    step, generator, vae, objective, pool, target_style, *,
    batch, patch_samples, real_images, rng, metrics_path, step_seconds,
) -> None:
    """Distribution Gap（生成 vs 源图到目标风格分布）与 Content Drift（结构差异）。"""
    from .distances import content_structure_loss
    from .features import local_descriptors, normalize_local

    content = pool.sample_content(batch, gen=rng).to(next(generator.parameters()).device)
    tgt = torch.full((batch,), int(target_style), dtype=torch.long, device=content.device)
    z_hat = generator(content, style_id=tgt)["z_hat"]
    x_src = decode_latents(vae, content)
    x_hat = decode_latents(vae, z_hat)

    real_locals, _real_glob = objective.real_samples(target_style, n_images=real_images)
    gaps = []
    for tag, x in (("gen", x_hat), ("src", x_src)):
        descs = normalize_local(local_descriptors(x), objective.stats)
        flat = []
        for d in descs:
            b_, c_, h_, w_ = d.shape
            d = d.reshape(b_, c_, h_ * w_).transpose(1, 2).reshape(b_ * h_ * w_, c_)
            flat.append(d)
        total = 0.0
        for k in range(len(flat)):
            total += float(swd2(flat[k], real_locals[k]).item())
        gaps.append((tag, total))
    drift = float(content_structure_loss(x_hat, x_src).item())
    row = {
        "step": int(step),
        "gap_gen": gaps[0][1],
        "gap_src": gaps[1][1],
        "content_drift": drift,
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0,
        "sec_per_step": sum(step_seconds) / len(step_seconds) if step_seconds else 0.0,
    }
    with metrics_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(" | ".join([
        f"[swd diag {step}]",
        f"gap_gen={row['gap_gen']:.3f}",
        f"gap_src={row['gap_src']:.3f}",
        f"drift={row['content_drift']:.4f}",
        f"peak={row['peak_mem_gb']:.2f}GB",
        f"s/step={row['sec_per_step']:.3f}",
    ]), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="WEAVE-v2 training (V1-V5)")
    parser.add_argument("--config", required=True, help="WEAVE config json (with a top-level v2 section)")
    parser.add_argument("--variant", default=None, choices=["V1", "V2", "V3", "V4", "V5", "V3S"])
    parser.add_argument("--stage", default="gen", choices=["gen", "distill"])
    parser.add_argument("--teacher", default="", help="V5 teacher checkpoint (V3 step_*.pt)")
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0, help="microbatch size")
    parser.add_argument("--accumulation-steps", type=int, default=0, help="梯度累积 microbatch 数；0=用配置")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--out", default="")
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--save-at", default="", help="逗号分隔的指定存盘步数，如 1000,2500,5000")
    parser.add_argument("--diag-every", type=int, default=500, help="诊断记录间隔步数；0 关闭")
    args = parser.parse_args(argv)
    save_at = tuple(sorted({int(x) for x in str(args.save_at).split(",") if x.strip()}))
    if args.save_every == 1000 and save_at:
        args.save_every = 0

    config = load_experiment_config(args.config)
    v2cfg = V2Config.from_experiment(config)
    variant = args.variant or v2cfg.variant
    v2cfg.variant = variant
    expand_variant_defaults(v2cfg)
    if args.steps > 0:
        v2cfg.steps = int(args.steps)
    if v2cfg.steps <= 0:
        v2cfg.steps = int(getattr(config.training, "stop_after_global_steps", 0) or 0)
    if v2cfg.steps <= 0:
        v2cfg.steps = int(config.training.num_epochs) * 1000
    if args.batch_size > 0:
        v2cfg.batch_size = int(args.batch_size)
    if args.accumulation_steps > 0:
        v2cfg.accumulation_steps = int(args.accumulation_steps)
    device = torch.device(args.device)
    out_dir = Path(args.out) if args.out else Path(v2cfg.out_dir) / variant

    print(json.dumps({"variant": variant, "steps": v2cfg.steps, "batch": v2cfg.batch_size,
                      "accumulation_steps": v2cfg.accumulation_steps,
                      "synthetic": args.synthetic,
                      "out": str(out_dir), "device": str(device)}, ensure_ascii=False))
    if variant in {"V1", "V3", "V4"}:
        train_generator_variant(
            config, v2cfg, variant, steps=v2cfg.steps, device=device,
            synthetic=args.synthetic, out_dir=out_dir, save_every=args.save_every,
            diag_every=args.diag_every, save_at=save_at,
        )
    elif variant == "V2":
        train_fm_variant(
            config, v2cfg, variant, steps=v2cfg.steps, device=device,
            synthetic=args.synthetic, out_dir=out_dir, save_every=args.save_every,
            diag_every=args.diag_every, save_at=save_at,
        )
    elif variant == "V3S":
        train_swd_variant(
            config, v2cfg, steps=v2cfg.steps, device=device, out_dir=out_dir,
            save_every=args.save_every, diag_every=args.diag_every, save_at=save_at,
        )
    elif variant == "V5":
        if args.stage != "distill":
            raise SystemExit("V5 必须显式 --stage distill 且提供 --teacher；不会自动启用")
        train_distill(
            config, v2cfg, teacher_path=args.teacher, steps=v2cfg.steps, device=device,
            synthetic=args.synthetic, out_dir=out_dir, save_every=args.save_every,
        )
    else:
        raise SystemExit(f"unsupported variant {variant}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
