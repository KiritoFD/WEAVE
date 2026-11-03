#!/usr/bin/env python3
"""A2/A3/A4 saturation diagnostics for the WEAVE velocity field.

P0 corrections (2026-11-04):
  A3 真实训练监督 : endpoint, x_t 和 velocity 目标全部来自
                    FlowMatchingObjective._build_training_target()
                    (SAT / LL-partial AdaIN / train AdaIN 等开关全部生效),
                    不再用原始 target-content 自造监督.
  A2 标签方差     : 固定在 t=0 (x_t == content, 与候选目标无关) 与固定风格条件
                    (同一 style_id + 同一 style_latent), 对 Top1/4/8 个候选目标
                    的 velocity 标签计算 cos(v,u) 的均值/标准差/极值 + 标签两两
                    cosine, 直接量化非配对监督的不确定性.
  A4 真实轨迹     : 使用 model.integrate_transport() 真实推理路径 (含 per-step
                    AdaIN / EOTA / progressive 等真实语义), 不再用自写 Euler
                    与 "style_latent * 系数" 的假 AdaIN.

所有量只来自模型本身 (无外部教师). A2/A3/A4 的 endpoint 均通过训练 objective 构造.

CPU smoke:  python tools/diagnose_saturation.py --smoke --n 4 --steps 4
Server:     python tools/diagnose_saturation.py --config config.json \
                --checkpoint runs/.../latest.pt --data-root data/train --device cuda
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config_schema import load_experiment_config
from flow import FlowMatchingObjective
from model import build_model_from_config, count_parameters
from wavelet import dwt2_haar

SUBBANDS = ("ll", "lh", "hl", "hh")
TIME_BUCKETS = ((0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0))


def _load_checkpoint(model: torch.nn.Module, path: str, device: torch.device) -> None:
    """Strict checkpoint load.

    Prints the expected / missing / unexpected key lists and fails (SystemExit)
    if any list is non-empty. No `strict=False` fallback: a diagnostic run on a
    partially loaded model is worse than no run at all.
    """
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise SystemExit(f"[ckpt] {path}: payload is {type(payload).__name__}, expected dict")
    state = payload.get("model_state_dict")
    if state is None:
        state = payload.get("state_dict")
    if state is None:
        state = payload
    if not isinstance(state, dict):
        raise SystemExit(f"[ckpt] {path}: state dict is {type(state).__name__}, expected dict")

    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        clean = key
        for prefix in ("_orig_mod.", "module.", "model."):
            if clean.startswith(prefix):
                clean = clean[len(prefix):]
        cleaned[clean] = value

    expected = list(model.state_dict().keys())
    missing = [key for key in expected if key not in cleaned]
    unexpected = [key for key in cleaned if key not in expected]
    print(f"[ckpt] path={path}")
    print(f"[ckpt] expected={len(expected)} provided={len(cleaned)} "
          f"missing={len(missing)} unexpected={len(unexpected)}")
    print(f"[ckpt] missing_keys={missing}")
    print(f"[ckpt] unexpected_keys={unexpected}")

    try:
        result = model.load_state_dict(cleaned, strict=False)
    except RuntimeError as exc:  # shape mismatch etc. (strict=False still raises)
        raise SystemExit(f"[ckpt] strict load failed (shape mismatch): {exc}") from exc
    if result.missing_keys or result.unexpected_keys:
        raise SystemExit(
            "[ckpt] strict load failed: expected/missing/unexpected must all be empty"
        )
    print("[ckpt] strict load OK")


def _load_latents(root: Path, styles: list[str], n: int, seed: int) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    stacks: list[torch.Tensor] = []
    for style in styles:
        files = sorted((root / style).glob("*.pt"))
        if not files:
            raise FileNotFoundError(f"no *.pt latents under {root / style}")
        idx = torch.randperm(len(files), generator=g)[:n]
        items = []
        for i in idx.tolist():
            value = torch.load(files[i], map_location="cpu", weights_only=False)
            if isinstance(value, dict):
                value = value.get("latent", next(iter(value.values())))
            value = value.float()
            if value.ndim == 3:
                value = value.unsqueeze(0)
            value = value[:4]
            items.append(value)
        stacks.append(torch.cat(items, dim=0))
    return stacks


def _subbands(x: torch.Tensor) -> dict[str, torch.Tensor]:
    ll, lh, hl, hh = dwt2_haar(x)
    return {"ll": ll, "lh": lh, "hl": hl, "hh": hh}


def _cos_ratio(v: torch.Tensor, u: torch.Tensor) -> tuple[float, float]:
    vf, uf = v.flatten(1), u.flatten(1)
    vn = vf.norm(dim=1).clamp_min(1e-8)
    un = uf.norm(dim=1).clamp_min(1e-8)
    cos = F.cosine_similarity(vf, uf, dim=1, eps=1e-8)
    ratio = vn / un
    return float(cos.mean()), float(ratio.mean())


def _pairwise_cos(vecs: list[torch.Tensor]) -> float:
    """Mean cosine between every pair of flattened label tensors."""
    if len(vecs) < 2:
        return 0.0
    flat = [v.reshape(-1) for v in vecs]
    vals = []
    for i in range(len(flat)):
        for j in range(i + 1, len(flat)):
            vals.append(float(F.cosine_similarity(flat[i], flat[j], dim=0, eps=1e-8)))
    return sum(vals) / len(vals)


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / max(1, len(values) - 1)
    return {
        "mean": mean,
        "std": math.sqrt(max(0.0, var)),
        "min": min(values),
        "max": max(values),
    }


def _velocity(model, h, t: float, style_id: int, style_latent) -> dict:
    return model(h, t=float(t), style_id=style_id, style_latent=style_latent)


def model_has_hh(model) -> bool:
    for name in ("head_hh",):
        if getattr(model, name, None) is not None:
            return True
    return False


def _cfg_get(model, key: str, default):
    """Same lookup precedence as model.integrate_transport: model_cfg, then bridge_cfg."""
    mcfg = getattr(model, "model_cfg", None)
    bcfg = getattr(model, "bridge_cfg", None)
    if mcfg is not None and hasattr(mcfg, key):
        return getattr(mcfg, key)
    if bcfg is not None and hasattr(bcfg, key):
        return getattr(bcfg, key)
    return default


@contextlib.contextmanager
def _temporary_cfg(model, overrides: dict):
    """Temporarily override model_cfg (or bridge_cfg) attributes, restore afterwards."""
    mcfg = getattr(model, "model_cfg", None)
    targets = [mcfg, getattr(model, "bridge_cfg", None)]
    saved = []  # (obj, key, existed, value)
    for obj in targets:
        if obj is None:
            continue
        for key, value in overrides.items():
            existed = hasattr(obj, key)
            saved.append((obj, key, existed, getattr(obj, key, None) if existed else None))
            setattr(obj, key, value)
    try:
        yield
    finally:
        for obj, key, existed, value in reversed(saved):
            if existed:
                setattr(obj, key, value)
            else:
                try:
                    delattr(obj, key)
                except AttributeError:
                    pass


def analysis_a3(
    model,
    objective: FlowMatchingObjective,
    content: torch.Tensor,
    target_style: torch.Tensor,
    *,
    style_id: int,
    samples_per_bucket: int = 3,
    seed: int = 0,
):
    """Time-bucket cosine and magnitude ratio, per subband, against the REAL target."""
    g = torch.Generator().manual_seed(seed)
    endpoint = objective._build_training_target(content, target_style, target_style)
    endpoint_sub = _subbands(endpoint)
    content_sub = _subbands(content)
    u_sub = {k: endpoint_sub[k] - content_sub[k] for k in SUBBANDS}
    records = []
    for lo, hi in TIME_BUCKETS:
        cos_rows = {k: [] for k in SUBBANDS}
        ratio_rows = {k: [] for k in SUBBANDS}
        for _ in range(samples_per_bucket):
            t = lo + (hi - lo) * float(torch.rand(1, generator=g))
            h = content * (1.0 - t) + endpoint * t
            v = _velocity(model, h, t, style_id, target_style)
            for k in SUBBANDS:
                if k == "hh" and not model_has_hh(model):
                    continue
                cos, ratio = _cos_ratio(v[k], u_sub[k])
                cos_rows[k].append(cos)
                ratio_rows[k].append(ratio)
        for k in SUBBANDS:
            if not cos_rows[k]:
                continue
            cos_stats = _stats(cos_rows[k])
            records.append({
                "bucket": f"[{lo},{hi}]", "subband": k,
                "cos_mean": cos_stats["mean"], "cos_std": cos_stats["std"],
                "ratio_mean": sum(ratio_rows[k]) / len(ratio_rows[k]),
            })
    return records


def analysis_a2(
    model,
    objective: FlowMatchingObjective,
    content: torch.Tensor,
    candidates: list[torch.Tensor],
    *,
    style_id: int,
    k_list=(1, 4, 8),
):
    """Non-paired label uncertainty at t=0 with a fixed style condition.

    At t=0 x_t == content for every candidate, so the prediction v is computed
    once and compared against the candidate velocity labels. The only source of
    variation is the non-paired target, which is exactly what we want to measure.
    """
    if not candidates:
        raise ValueError("A2 needs at least one candidate target")
    style_ref = candidates[0]
    batch = min(content.shape[0], *(c.shape[0] for c in candidates))
    content_b = content[:batch]
    targets = [
        objective._build_training_target(content_b, cand[:batch], style_ref[:batch])
        for cand in candidates
    ]
    content_sub = _subbands(content_b)
    u_sub_all = [
        {k: _subbands(tgt)[k] - content_sub[k] for k in SUBBANDS} for tgt in targets
    ]
    # t=0 -> x_t = content (independent of candidate).
    v = _velocity(model, content_b, 0.0, style_id, style_ref[:batch])
    rows = []
    for k in k_list:
        ks = min(k, len(targets))
        for sb in SUBBANDS:
            if sb == "hh" and not model_has_hh(model):
                continue
            cos_vals = []
            ratio_vals = []
            for u in u_sub_all[:ks]:
                cos, ratio = _cos_ratio(v[sb], u[sb])
                cos_vals.append(cos)
                ratio_vals.append(ratio)
            cos_stats = _stats(cos_vals)
            ratio_stats = _stats(ratio_vals)
            rows.append({
                "topk": ks, "subband": sb,
                "t": 0.0, "n_candidates": ks,
                "cos_mean": cos_stats["mean"], "cos_std": cos_stats["std"],
                "cos_min": cos_stats["min"], "cos_max": cos_stats["max"],
                "ratio_mean": ratio_stats["mean"], "ratio_std": ratio_stats["std"],
                "label_pairwise_cos_mean": _pairwise_cos([u[sb] for u in u_sub_all[:ks]]),
            })
    return rows


A4_MODES = (
    # (name, overrides) — empty overrides == the config's real inference path.
    ("net_only", {
        "endpoint_adain_scale": 0.0,
        "endpoint_adain_scale_ll": 0.0,
        "endpoint_adain_scale_lh": 0.0,
        "endpoint_adain_scale_hl": 0.0,
        "endpoint_adain_scale_hh": 0.0,
        "endpoint_adain_only_last_step": False,
        "progressive_alpha_schedule": "none",
        "zero_step_wct_enabled": False,
    }),
    ("real_ode", {}),
    ("last_step_adain", {
        "endpoint_adain_only_last_step": True,
        "progressive_alpha_schedule": "none",
    }),
    ("every_step_adain", {
        "endpoint_adain_only_last_step": False,
        "progressive_alpha_schedule": "none",
    }),
)


def analysis_a4(
    model,
    objective: FlowMatchingObjective,
    content: torch.Tensor,
    target_style: torch.Tensor,
    *,
    style_id: int,
    steps: int,
):
    """Teacher forcing vs the real ODE rollout (model.integrate_transport)."""
    endpoint = objective._build_training_target(content, target_style, target_style)
    content_sub = _subbands(content)
    endpoint_sub = _subbands(endpoint)
    u = {k: endpoint_sub[k] - content_sub[k] for k in SUBBANDS}

    grid = [i / 8 for i in range(9)]
    tf_cos = {k: [] for k in SUBBANDS}
    for t in grid:
        h = content * (1.0 - t) + endpoint * t
        v = _velocity(model, h, t, style_id, target_style)
        for k in SUBBANDS:
            if k == "hh" and not model_has_hh(model):
                continue
            cos, _ = _cos_ratio(v[k], u[k])
            tf_cos[k].append(cos)
    tf_mean = sum(sum(v) / len(v) for v in tf_cos.values() if v) / max(
        1, sum(1 for v in tf_cos.values() if v)
    )

    rows = []
    for name, overrides in A4_MODES:
        with _temporary_cfg(model, dict(overrides)):
            effective_scale = float(_cfg_get(model, "endpoint_adain_scale", 0.0))
            out = model.integrate_transport(
                content, style_id, num_steps=steps, style_latent=target_style
            )
        err = float((out - endpoint).flatten(1).norm(dim=1).mean())
        out_sub = _subbands(out)
        cos_to_endpoint = []
        for k in SUBBANDS:
            if k == "hh" and not model_has_hh(model):
                continue
            cos_to_endpoint.append(
                _cos_ratio(out_sub[k] - content_sub[k], u[k])[0]
            )
        cos_to_raw = []
        raw_sub = _subbands(target_style)
        raw_u = {k: raw_sub[k] - content_sub[k] for k in SUBBANDS}
        for k in SUBBANDS:
            if k == "hh" and not model_has_hh(model):
                continue
            cos_to_raw.append(_cos_ratio(out_sub[k] - content_sub[k], raw_u[k])[0])
        rows.append({
            "mode": name, "steps": steps,
            "effective_adain_scale": effective_scale,
            "endpoint_l2": err,
            "endpoint_cos_to_training_target": sum(cos_to_endpoint) / max(1, len(cos_to_endpoint)),
            "endpoint_cos_to_raw_target": sum(cos_to_raw) / max(1, len(cos_to_raw)),
            "teacher_forcing_cos_mean": tf_mean,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=False, default=None)
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--data-root", default="data/train")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/diagnose_saturation.json")
    ap.add_argument("--smoke", action="store_true", help="random latents + random weights, CPU")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-candidates", type=int, default=8)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device if not args.smoke else "cpu")

    if args.smoke:
        cfg_path = args.config or "configs/hf_oriented_internal_early_stop.json"
        cfg = load_experiment_config(cfg_path)
        cfg.model.base_dim = 16
        cfg.model.num_res_blocks = 2
        objective = FlowMatchingObjective(cfg)
        model = build_model_from_config(cfg.model, bridge_cfg=cfg.bridge).to(device).eval()
        content = torch.randn(4, 4, 64, 64, device=device)
        target = torch.randn_like(content)
        candidates = [torch.randn_like(content) for _ in range(4)]
        style_id = 0
        print(f"[smoke] params={count_parameters(model)}")
    else:
        if not args.config:
            raise SystemExit("--config is required unless --smoke")
        cfg = load_experiment_config(args.config)
        objective = FlowMatchingObjective(cfg)
        model = build_model_from_config(cfg.model, bridge_cfg=cfg.bridge).to(device)
        if args.checkpoint:
            _load_checkpoint(model, args.checkpoint, device)
        model.eval()
        styles = list(cfg.data.style_subdirs)
        stacks = _load_latents(Path(args.data_root), styles, args.n, args.seed)
        content = stacks[0][: args.n].to(device)
        target = stacks[1 % len(stacks)][: args.n].to(device)
        pool = stacks[1 % len(stacks)]
        n = min(args.n, pool.shape[0])
        k = max(1, args.n_candidates)
        shift = max(1, pool.shape[0] // k)
        candidates = [pool.roll(shifts=i * shift, dims=0)[:n].to(device) for i in range(k)]
        style_id = 1 % len(stacks)
        print(f"[data] content={tuple(content.shape)} target={tuple(target.shape)} style_id={style_id}")

    with torch.no_grad():
        a3 = analysis_a3(model, objective, content, target, style_id=style_id, seed=args.seed)
        a2 = analysis_a2(model, objective, content, candidates, style_id=style_id)
        a4 = analysis_a4(model, objective, content, target, style_id=style_id, steps=args.steps)

    payload = {
        "config": args.config, "checkpoint": args.checkpoint,
        "n": args.n, "steps": args.steps, "seed": args.seed, "smoke": args.smoke,
        "endpoint_source": "FlowMatchingObjective._build_training_target",
        "a2_setting": "t=0, fixed style_id + fixed style_latent",
        "a4_setting": "model.integrate_transport (real inference path)",
        "A2_candidate_uncertainty": a2,
        "A3_time_buckets": a3,
        "A4_trajectory_vs_teacher_forcing": a4,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"A2": a2[:4], "A3": a3[:6], "A4": a4}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
