"""V7 P1 validity: the four mandatory measurements (V7_IMPLEMENTATION_PLAN.md P1.3).

1. 五风格可分性          -> from p1_energy/energy_audit.json (recomputed if absent)
2. 内容保持扰动稳定性    -> small Gaussian / LL-only perturbations must move the
                            energy less than the inter-style margin
3. 优化方向一致性        -> small-step energy-only descent on B4 direct outputs of
                            40 stratified tasks must improve (or not degrade)
                            independent DINO-S
4. 各风格均衡            -> per-style AUROC / per-style ΔDINO-S balance check

Only *_sd energies are valid optimization targets for generated SD-VAE latents;
*_a0 energies are对照 statistics.

Writes runs/v7/p1_energy/energy_validity.json with `best_energy` and the
GO_ENERGY / NO_ENERGY verdict.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from tools.v7.v7_common import (
    B4_CKPT, P1_DIR, ROOT, STYLES, load_c3unet, load_canon, load_critic,
    model_direct, rms_normalize, write_json,
)
from wavelet import dwt2_haar, idwt2_haar

SRC_CACHE = ROOT / "runs" / "v6_r2" / "cache" / "tasks750_source_features.pt"
DEFAULT_REFS = ROOT / "eval_protocol" / "cache" / "references_v1_features.pt"

# Predefined pass thresholds (recorded in the output; not tuned post-hoc)
PERTURB_RMS = (0.02, 0.05)
OPT_STEPS = 4
OPT_PER_STEP_RMS = 0.02  # small-step optimization probe (8x smaller than P2 arms)
DINO_S_NONDEGRADE = -0.005
MIN_STYLES_IMPROVE = 3


def load_energy(kind: str, tag: str):
    p = P1_DIR / f"{kind}_{tag}.pt"
    return torch.load(p, map_location="cpu", weights_only=False) if p.exists() else None


def perturb(z: torch.Tensor, rms: float, mode: str, gen: torch.Generator) -> torch.Tensor:
    noise = torch.randn(z.shape, generator=gen, device=gen.device, dtype=z.dtype)
    noise = noise.to(z.device)
    noise = rms_normalize(noise) * rms
    if mode == "ll":
        ll, lh, hl, hh = dwt2_haar(noise)
        noise = idwt2_haar(ll, torch.zeros_like(lh), torch.zeros_like(hl),
                           torch.zeros_like(hh))
        noise = noise / (noise.flatten(1).pow(2).mean(dim=1).sqrt()
                         .view(-1, 1, 1, 1).clamp(min=1e-8)) * rms
    return z + noise


def main() -> int:
    ap = argparse.ArgumentParser(description="V7 P1 energy validity")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tasks-per-style", type=int, default=8)
    args = ap.parse_args()
    device = args.device
    t0 = time.time()

    from tools.v5.route_c import decode_trt, load_trt, load_vae
    from tools.v6_next.w_distill import R1FeatureExtractor
    from tools.v6.train_eval_v6 import encode_dino, load_dino_model

    audit_path = P1_DIR / "energy_audit.json"
    if not audit_path.exists():
        raise SystemExit("energy_audit.json missing — run p1_energy.py first")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audits_by_key = {(a["kind"], a["pool"]): a for a in audit["audits"]}

    critic = load_critic(device=device)
    feat_ext = R1FeatureExtractor(critic)

    # ---------------- Measurement 2: perturbation stability ----------------
    sd_pool = Path("/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16")
    val_by_style = {}
    for sid, s in enumerate(STYLES):
        d = torch.load(sd_pool / f"{sid:02d}_{s}.pt", map_location="cpu",
                       weights_only=False)
        val_by_style[s] = d["latents"].float()[:40]
    perturb_rows = {}
    for key, a in audits_by_key.items():
        kind, tag = key
        stats = load_energy(kind, tag)
        if stats is None:
            continue
        per_rms = {}
        for rms in PERTURB_RMS:
            deltas = {"gauss": [], "ll": []}
            gen = torch.Generator(device="cpu").manual_seed(1234 + int(rms * 1000))
            for sid, s in enumerate(STYLES):
                z = val_by_style[s][:40].to(device)
                ids = torch.full((z.shape[0],), sid, dtype=torch.long, device=device)
                with torch.no_grad():
                    e0 = (s1_loss_per_sample(z, ids, stats) if kind == "S1"
                          else s2_loss_per_sample(z, ids, stats, feat_ext))
                for mode in deltas:
                    zp = perturb(z, rms, mode, gen)
                    with torch.no_grad():
                        e1 = (s1_loss_per_sample(zp, ids, stats) if kind == "S1"
                              else s2_loss_per_sample(zp, ids, stats, feat_ext))
                    deltas[mode].extend((e1 - e0).detach().cpu().tolist())
            margin = a.get("margin_median", a["margin_mean"])
            per_rms[str(rms)] = {
                "delta_mean_gauss": float(np.mean(deltas["gauss"])),
                "delta_mean_ll": float(np.mean(deltas["ll"])),
                "inter_style_margin": float(margin),
                "stable_gauss": bool(np.mean(deltas["gauss"]) < margin),
                "stable_ll": bool(np.mean(deltas["ll"]) < margin),
            }
        perturb_rows[f"{kind}_{tag}"] = per_rms

    # ---------------- Measurement 3: optimization direction ----------------
    tasks = json.loads((ROOT / "eval_protocol" / "tasks_750.json").read_text(encoding="utf-8"))["tasks"]
    tasks = [t for t in tasks if t.get("is_cross")]
    by_tgt = {}
    for t in tasks:
        if len(by_tgt.setdefault(t["target_style"], [])) < args.tasks_per_style:
            by_tgt[t["target_style"]].append(t)
    sel = [t for s in STYLES for t in by_tgt.get(s, [])]
    src_cache = torch.load(SRC_CACHE, map_location="cpu", weights_only=False)
    ref_data = torch.load(DEFAULT_REFS, map_location="cpu", weights_only=False)
    ref_dino = {s: F.normalize(ref_data["styles"][s]["dino_cls"].float(), dim=-1).numpy()
                for s in STYLES}  # (30, 384) per style

    vae = load_vae(device)
    trt = load_trt(vae, device)
    dino = load_dino_model(device=device)
    b4 = load_c3unet(B4_CKPT, device)
    canon = load_canon(device)

    z0_all, ids_all, dino_s_base = [], [], []
    with torch.inference_mode():
        for i in range(0, len(sel), 8):
            grp = sel[i:i + 8]
            z = torch.cat([src_cache["z_src"][t["source_stem"]].to(device) for t in grp], 0)
            tg = torch.tensor([t["target_style_id"] for t in grp], device=device)
            z0 = model_direct(b4, z, canon, "c1_gentle", tg)
            img = decode_trt(trt, vae, z0, batch=8)
            u8 = img.mul(255).round().clamp(0, 255).to(torch.uint8).cpu()
            arr = u8.permute(0, 2, 3, 1).numpy()
            pils = [Image.fromarray(arr[j]) for j in range(len(grp))]
            dg, _ = encode_dino(dino, pils, batch=8, device=device)
            dg = F.normalize(dg.float(), dim=-1)
            for j, t in enumerate(grp):
                z0_all.append(z0[j].cpu())
                ids_all.append(int(t["target_style_id"]))
                dino_s_base.append(float(np.max(dg[j].cpu().numpy() @ ref_dino[t["target_style"]].T)))
    z0_all = torch.stack(z0_all, 0).to(device)
    ids_all_t = torch.tensor(ids_all, dtype=torch.long, device=device)

    def dino_s_of(zb: torch.Tensor) -> list[float]:
        out = []
        with torch.inference_mode():
            for i in range(0, zb.shape[0], 8):
                img = decode_trt(trt, vae, zb[i:i + 8], batch=8)
                u8 = img.mul(255).round().clamp(0, 255).to(torch.uint8).cpu()
                arr = u8.permute(0, 2, 3, 1).numpy()
                pils = [Image.fromarray(arr[j]) for j in range(arr.shape[0])]
                dg, _ = encode_dino(dino, pils, batch=8, device=device)
                dg = F.normalize(dg.float(), dim=-1).cpu().numpy()
                for j in range(arr.shape[0]):
                    out.append(float(np.max(dg[j] @ ref_dino[STYLES[int(ids_all[i + j])]].T)))
        return out

    opt_rows = {}
    for kind, tag in [("S1", "sd"), ("S2", "sd")]:
        stats = load_energy(kind, tag)
        if stats is None:
            continue
        z = z0_all.clone()
        for _ in range(OPT_STEPS):
            zr = z.detach().clone().requires_grad_(True)
            if kind == "S1":
                loss = s1_loss(zr, ids_all_t, stats)
            else:
                loss = s2_loss(zr, ids_all_t, stats, feat_ext)
            g = torch.autograd.grad(loss, zr)[0]
            z = (zr - OPT_PER_STEP_RMS * rms_normalize(g)).detach()
        ds = dino_s_of(z)
        delta = np.array(ds) - np.array(dino_s_base)
        per_style = {s: float(np.mean([delta[i] for i in range(len(ids_all))
                                       if ids_all[i] == sid]))
                     for sid, s in enumerate(STYLES)}
        e_before = float(s1_loss(z0_all, ids_all_t, stats) if kind == "S1"
                         else s2_loss(z0_all, ids_all_t, stats, feat_ext))
        e_after = float(s1_loss(z, ids_all_t, stats) if kind == "S1"
                        else s2_loss(z, ids_all_t, stats, feat_ext))
        opt_rows[f"{kind}_{tag}"] = {
            "delta_dino_s_mean": float(delta.mean()),
            "delta_dino_s_per_style": per_style,
            "n_styles_improved": int(sum(1 for v in per_style.values() if v > 0)),
            "energy_before": e_before, "energy_after": e_after,
            "energy_reduced": bool(e_after < e_before),
            "pass": bool(delta.mean() >= DINO_S_NONDEGRADE
                         and (delta.mean() >= 0
                              or sum(1 for v in per_style.values() if v > 0) >= MIN_STYLES_IMPROVE)),
        }
        print(f"[p1valid] {kind}_{tag}: ΔDINO-S={delta.mean():+.4f} "
              f"energy {e_before:.4f}->{e_after:.4f}", flush=True)

    # ---------------- Aggregate verdict ----------------
    candidates = {}
    for key in [f"{k}_{t}" for k in ("S1", "S2") for t in ("sd", "a0")]:
        a = audits_by_key.get(tuple(key.split("_")))
        p = perturb_rows.get(key, {})
        o = opt_rows.get(key)
        sep = a["separable"] if a else False
        bal = bool(a and min(a["auroc_per_style"].values()) > 0.5) if a else False
        stable = bool(p) and all(v["stable_gauss"] and v["stable_ll"]
                                 for v in p.values())
        # measurement 3 only applies to *_sd (generated latents live in SD space)
        opt_pass = o["pass"] if (o is not None) else (key.endswith("_a0"))
        candidates[key] = {
            "separable": sep, "balanced": bal, "perturbation_stable": stable,
            "optimization_consistent": opt_pass,
            "delta_dino_s_mean": (o or {}).get("delta_dino_s_mean"),
            "valid": bool(sep and bal and stable and opt_pass),
        }
    valid_keys = [k for k, v in candidates.items() if v["valid"]]
    if valid_keys:
        best = max(valid_keys, key=lambda k: candidates[k]["delta_dino_s_mean"] or -9)
        verdict = "GO_ENERGY"
    else:
        best = None
        verdict = "NO_ENERGY"

    write_json({
        "verdict": verdict,
        "best_energy": best,
        "candidates": candidates,
        "audit": {f"{k[0]}_{k[1]}": v for k, v in audits_by_key.items()},
        "perturbation": perturb_rows,
        "optimization": opt_rows,
        "thresholds": {
            "perturb_rms": PERTURB_RMS,
            "opt_steps": OPT_STEPS,
            "opt_per_step_rms": OPT_PER_STEP_RMS,
            "dino_s_nondegrade": DINO_S_NONDEGRADE,
            "min_styles_improve": MIN_STYLES_IMPROVE,
        },
        "notes": [
            "Optimization consistency is only measured for *_sd energies: "
            "generated latents (B4 direct) live in the SD-VAE latent space; "
            "*_a0 energies are对照 statistics and cannot drive SD-latent edits.",
        ],
        "wall_sec": round(time.time() - t0, 1),
    }, P1_DIR / "energy_validity.json")
    print(f"[p1valid] verdict={verdict} best={best} ({time.time()-t0:.0f}s)", flush=True)
    return 0


def s1_loss_per_sample(z, ids, stats):
    from tools.v7.v7_common import s1_loss_per_sample as f
    return f(z, ids, stats)


def s2_loss_per_sample(z, ids, stats, feat_ext):
    from tools.v7.v7_common import s2_loss_per_sample as f
    return f(z, ids, stats, feat_ext)


def s1_loss(z, ids, stats):
    from tools.v7.v7_common import s1_loss as f
    return f(z, ids, stats)


def s2_loss(z, ids, stats, feat_ext):
    from tools.v7.v7_common import s2_loss as f
    return f(z, ids, stats, feat_ext)


if __name__ == "__main__":
    raise SystemExit(main())
