#!/usr/bin/env python3
"""A2/A3/A4 saturation diagnostics for the WEAVE velocity field.

A3  time-bucket cosine / magnitude ratio   : cos(v, u) and ||v||/||u|| on
                                            t in [0,.1] [.1,.25] [.25,.5]
                                            [.5,.75] [.75,1], per subband.
A2  non-paired label uncertainty           : Top1/Top4/Top8 candidate targets,
                                            cosine spread + consistency, per subband.
A4  teacher-forcing vs ODE trajectory      : 1-step, 8-step, 8-step AdaIN@0/2,
                                            8-step AdaIN only on the last step.

All quantities are computed from the model alone (no external teacher).
CPU smoke:  python tools/diagnose_saturation.py --smoke --n 4 --steps 4
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config_schema import load_experiment_config
from model import build_model_from_config, count_parameters
from wavelet import dwt2_haar, idwt2_haar

SUBBANDS = ("ll", "lh", "hl", "hh")
TIME_BUCKETS = ((0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0))


def _load_checkpoint(model: torch.nn.Module, path: str, device: torch.device) -> None:
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload.get("model_state_dict", payload.get("state_dict", payload))
    cleaned = {}
    for key, value in state.items():
        clean = key
        for prefix in ("_orig_mod.", "module.", "model."):
            if clean.startswith(prefix):
                clean = clean[len(prefix):]
        cleaned[clean] = value
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"[ckpt] loaded {path} | missing={len(missing)} unexpected={len(unexpected)}")


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
    cos = torch.nn.functional.cosine_similarity(vf, uf, dim=1, eps=1e-8)
    ratio = vn / un
    return float(cos.mean()), float(ratio.mean())


def _velocity(model, h, t: float, style_id: int, style_latent, adain_scale: float) -> dict:
    if adain_scale == 0.0:
        sl = torch.zeros_like(style_latent)
    elif adain_scale != 1.0:
        sl = style_latent * float(adain_scale)
    else:
        sl = style_latent
    return model(h, t=float(t), style_id=style_id, style_latent=sl)


def _euler(model, h, *, steps: int, style_id: int, style_latent, adain_scale: float,
           last_step_adain_only: bool = False) -> torch.Tensor:
    steps = max(1, int(steps))
    dt = 1.0 / steps
    for i in range(steps):
        t = i * dt
        scale = adain_scale
        if last_step_adain_only and i != steps - 1:
            scale = 0.0
        v = _velocity(model, h, t, style_id, style_latent, scale)
        ll, lh, hl, hh = dwt2_haar(h)
        ll = ll + v["ll"] * dt
        lh = lh + v["lh"] * dt
        hl = hl + v["hl"] * dt
        if "hh" in v:
            hh = hh + v["hh"] * dt
        h = idwt2_haar(ll, lh, hl, hh)
    return h


def analysis_a3(model, content, target, *, style_id: int, samples_per_bucket: int = 3, seed: int = 0):
    """Time-bucket cosine and magnitude ratio, per subband."""
    g = torch.Generator().manual_seed(seed)
    u_sub = {k: _subbands(target)[k] - _subbands(content)[k] for k in SUBBANDS}
    records = []
    for lo, hi in TIME_BUCKETS:
        rows = {k: {"cos": [], "ratio": []} for k in SUBBANDS}
        for _ in range(samples_per_bucket):
            t = lo + (hi - lo) * float(torch.rand(1, generator=g))
            h = content * (1.0 - t) + target * t
            v = _velocity(model, h, t, style_id, target, 1.0)
            v_sub = _subbands_out(v, h)
            for k in SUBBANDS:
                if k not in v_sub or k not in u_sub or u_sub[k] is None:
                    continue
                cos, ratio = _cos_ratio(v_sub[k], u_sub[k])
                rows[k]["cos"].append(cos)
                rows[k]["ratio"].append(ratio)
        for k in SUBBANDS:
            if rows[k]["cos"]:
                records.append({
                    "bucket": f"[{lo},{hi}]", "subband": k,
                    "cos_mean": sum(rows[k]["cos"]) / len(rows[k]["cos"]),
                    "ratio_mean": sum(rows[k]["ratio"]) / len(rows[k]["ratio"]),
                })
    return records


def _subbands_out(v: dict, h: torch.Tensor) -> dict[str, torch.Tensor]:
    ll, lh, hl, hh = dwt2_haar(h)
    zeros = torch.zeros_like(hh)
    out = {
        "ll": torch.cat([ll, lh, hl, hh], dim=1),
        "lh": torch.cat([ll, lh, hl, hh], dim=1),
        "hl": torch.cat([ll, lh, hl, hh], dim=1),
        "hh": torch.cat([ll, lh, hl, hh], dim=1),
    }
    # Subband-velocity tensors are already (B,C,H/2,W/2); keep the raw fields.
    out["ll"] = v["ll"]
    out["lh"] = v["lh"]
    out["hl"] = v["hl"]
    out["hh"] = v.get("hh", zeros)
    return out


def analysis_a2(model, content, candidates, *, style_id: int, t: float, k_list=(1, 4, 8)):
    """Non-paired label uncertainty over Top-K candidate targets."""
    u_sub_all = []
    for target in candidates:
        u = {kk: _subbands(target)[kk] - _subbands(content)[kk] for kk in SUBBANDS}
        u_sub_all.append(u)
    h = content * (1.0 - t) + candidates[0] * t
    v = _velocity(model, h, t, style_id, candidates[0], 1.0)
    v_sub = _subbands_out(v, h)
    rows = []
    for k in k_list:
        ks = min(k, len(candidates))
        for sb in SUBBANDS:
            cos_vals = []
            ratio_vals = []
            if sb == "hh" and not model_has_hh(model):
                continue
            for u in u_sub_all[:ks]:
                cos, ratio = _cos_ratio(v_sub[sb], u[sb])
                cos_vals.append(cos)
                ratio_vals.append(ratio)
            mean = sum(cos_vals) / len(cos_vals)
            var = sum((c - mean) ** 2 for c in cos_vals) / max(1, len(cos_vals) - 1)
            rows.append({
                "topk": ks, "subband": sb, "cos_mean": mean,
                "cos_std": math.sqrt(max(0.0, var)),
                "ratio_mean": sum(ratio_vals) / len(ratio_vals),
            })
    return rows


def analysis_a4(model, content, target, *, style_id: int, steps: int):
    """Teacher forcing vs free ODE rollout under different AdaIN schedules."""
    u = {k: _subbands(target)[k] - _subbands(content)[k] for k in SUBBANDS}
    grid = [i / 8 for i in range(9)]
    tf_cos = {k: [] for k in SUBBANDS}
    for t in grid:
        h = content * (1.0 - t) + target * t
        v = _velocity(model, h, t, style_id, target, 1.0)
        v_sub = _subbands_out(v, h)
        for k in SUBBANDS:
            if k == "hh" and not model_has_hh(model):
                continue
            cos, _ = _cos_ratio(v_sub[k], u[k])
            tf_cos[k].append(cos)
    tf_mean = {k: (sum(v) / len(v) if v else 0.0) for k, v in tf_cos.items()}

    modes = [
        ("1step_adain0", 1, 0.0, False),
        ("8step_adain0", steps, 0.0, False),
        ("8step_adain2", steps, 2.0, False),
        ("8step_last_adain", steps, 1.0, True),
        ("8step_adain1", steps, 1.0, False),
    ]
    rows = []
    for name, n_steps, scale, last_only in modes:
        out_latent = _euler(model, content, steps=n_steps, style_id=style_id,
                            style_latent=target, adain_scale=scale,
                            last_step_adain_only=last_only)
        err = float((out_latent - target).flatten(1).norm(dim=1).mean())
        out_sub = _subbands(out_latent)
        cos_to_target = []
        for k in SUBBANDS:
            if k == "hh" and not model_has_hh(model):
                continue
            cos_to_target.append(_cos_ratio(out_sub[k] - _subbands(content)[k], u[k])[0])
        rows.append({
            "mode": name, "steps": n_steps, "adain_scale": scale,
            "last_step_adain_only": last_only,
            "endpoint_l2": err,
            "endpoint_cos_to_target": sum(cos_to_target) / max(1, len(cos_to_target)),
            "teacher_forcing_cos_mean": sum(tf_mean.values()) / max(1, len(tf_mean)),
        })
    return rows


def model_has_hh(model) -> bool:
    for name in ("head_hh",):
        if getattr(model, name, None) is not None:
            return True
    return False


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
        model = build_model_from_config(cfg.model, bridge_cfg=cfg.bridge).to(device)
        content = torch.randn(4, 4, 64, 64, device=device)
        target = torch.randn_like(content)
        candidates = [torch.randn_like(content) for _ in range(4)]
        styles = ["style0", "style1"]
        style_id = 0
        print(f"[smoke] params={count_parameters(model)}")
    else:
        if not args.config:
            raise SystemExit("--config is required unless --smoke")
        cfg = load_experiment_config(args.config)
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
        a3 = analysis_a3(model, content, target, style_id=style_id, seed=args.seed)
        a2 = analysis_a2(model, content, candidates, style_id=style_id, t=0.5)
        a4 = analysis_a4(model, content, target, style_id=style_id, steps=args.steps)

    payload = {
        "config": args.config, "checkpoint": args.checkpoint,
        "n": args.n, "steps": args.steps, "seed": args.seed, "smoke": args.smoke,
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
