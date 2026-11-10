"""WEAVE R2 Unified Evaluation Pipeline (tasks_750, 600 cross + 150 same).

Implements Section P0 of WEAVE R2 12-Hour Execution Plan:
- Raw RGB source preprocessed to 512x512 as ground-truth zero point.
- Output decoded via standard SD-VAE / TRT fp16.
- CSD: Raw RGB source feature zero point (with VAE-roundtrip diagnostic).
- DINO-C: Generated CLS vs Raw RGB source CLS.
- DINO-S: Generated CLS vs 30 target-style references (disjoint, self-match masked).
- LPIPS: Generated RGB vs Raw RGB source.
- Target Top-1 & Margin: True 5-way argmax and (s_target - max_{k!=target} s_k).
- Full 5x5 confusion matrix with integer indexing (fixed string comparison defect).
- Cluster bootstrap 95% CI with 150 source clusters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from tools.v5.route_c import csd_prototypes, decode_trt, load_trt, load_vae
from tools.v6.canonicalize import ContentCanonicalizer
from tools.v6.models_v6 import C3UNet, ConditionalScoreNet
from tools.v6.train_eval_v6 import STYLES, csd_model, encode_dino, load_dino_model
from tools.v6_24h.stats import cluster_ci, paired_cluster_ci
from tools.v6_next.probe_style import LatentStyleNet
from utils.run_evaluation import _load_eval_image_tensor, to_lpips_input

DEFAULT_TASKS = ROOT / "eval_protocol" / "tasks_750.json"
DEFAULT_REFS = ROOT / "eval_protocol" / "cache" / "references_v1_features.pt"
CANON_STATS = ROOT / "runs" / "v6" / "cache" / "canonicalizer_stats.pt"


def load_generator(args, device: str):
    """Load model according to generator mode."""
    if args.gen in ("direct", "oracle"):
        ck = torch.load(ROOT / args.ckpt, map_location="cpu", weights_only=False)
        a = ck.get("args", {})
        model = C3UNet(
            base_dim=a.get("base_dim", 16),
            num_styles=5,
            mode=a.get("mode", "direct"),
            no_style=bool(a.get("no_style", False)),
            content_mode=a.get("content_mode", "full"),
            inject_64=not a.get("no_inject_64", False),
            inject_32=not a.get("no_inject_32", False),
            inject_16=not a.get("no_inject_16", False),
            style_keys=bool(a.get("style_keys", False)),
            style_keys_n=int(a.get("style_keys_n", 8) or 8),
            style_keys_rank=int(a.get("style_keys_rank", 32) or 32),
        ).to(device)
        model.load_state_dict(ck["model"])
        model.eval()

        canon_mode = args.canon_mode or a.get("canon_mode", "c1_gentle")
        canon = ContentCanonicalizer(
            torch.load(CANON_STATS, map_location="cpu", weights_only=False)
        ).to(device)

        critic = None
        if args.gen == "oracle":
            ck_c = torch.load(
                ROOT / args.oracle_critic, map_location="cpu", weights_only=False
            )
            critic = LatentStyleNet(width=ck_c.get("width", 64)).to(device).eval()
            critic.load_state_dict(ck_c["model"])
            for p in critic.parameters():
                p.requires_grad_(False)

        return {
            "kind": args.gen,
            "model": model,
            "canon": canon,
            "canon_mode": canon_mode,
            "critic": critic,
            "oracle_rms": args.oracle_rms,
            "oracle_steps": args.oracle_steps,
        }

    elif args.gen == "scorediff":
        from tools.v6.score_diff import edit_with_score_difference

        sd_ck = torch.load(
            ROOT / args.sd_ckpt, map_location="cpu", weights_only=False
        )
        sd_model = (
            ConditionalScoreNet(base_dim=args.sd_base_dim, num_styles=5)
            .to(device)
            .eval()
        )
        sd_model.load_state_dict(sd_ck["model"])
        return {
            "kind": "scorediff",
            "model": sd_model,
            "steps": args.sd_steps,
            "lr": args.sd_lr,
            "anchor": args.sd_anchor,
            "fn": edit_with_score_difference,
        }

    elif args.gen == "identity":
        return {"kind": "identity"}

    raise ValueError(f"Unknown generator mode: {args.gen}")


def run_generator(gen_pack, z_batch, sid_batch, tid_batch):
    """Execute generator to produce target latents."""
    kind = gen_pack["kind"]
    if kind == "identity":
        return z_batch

    if kind == "direct":
        canon = gen_pack["canon"]
        model = gen_pack["model"]
        c = canon(z_batch, mode=gen_pack["canon_mode"], augment=False)
        return model(c, 0.0, c, tid_batch)

    if kind == "oracle":
        canon = gen_pack["canon"]
        model = gen_pack["model"]
        critic = gen_pack["critic"]
        steps = gen_pack["oracle_steps"]
        total_rms = gen_pack["oracle_rms"]
        per_rms = total_rms / max(1, steps)

        with torch.inference_mode(False), torch.enable_grad():
            zb = z_batch.clone()
            tb = tid_batch.clone()
            c = canon(zb, mode=gen_pack["canon_mode"], augment=False)
            z = model(c, 0.0, c, tb)
            for _ in range(steps):
                zr = z.detach().clone().requires_grad_(True)
                q = critic(zr.float()).float()
                oh = F.one_hot(tb, 5).bool()
                m = q.gather(1, tb[:, None]).squeeze(1) - torch.logsumexp(
                    q.masked_fill(oh, -1e9), dim=1
                )
                g = torch.autograd.grad(m.sum(), zr)[0]
                D = g[0].numel()
                gn = g.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
                z = (z + per_rms * (g / (gn / (D**0.5) + 1e-12))).detach()
        return z.detach()

    if kind == "scorediff":
        return gen_pack["fn"](
            gen_pack["model"],
            z_batch,
            sid_batch,
            tid_batch,
            num_steps=gen_pack["steps"],
            step_size=gen_pack["lr"],
            struct_anchor_weight=gen_pack["anchor"],
            seed=42,
        )


def build_source_cache(tasks, vae, trt, csd, dino, cache_dir: Path, device: str):
    """Precompute raw RGB features, VAE posterior mean latents, and round-trip features."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "tasks750_source_features.pt"
    if cache_path.exists():
        print(f"[P0] Loading cached source features from {cache_path}", flush=True)
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    print(
        f"[P0] Precomputing unified source features for 150 tasks...", flush=True
    )
    uniq = {}
    for t in tasks:
        k = t["source_stem"]
        uniq.setdefault(k, t["source_rel_path"])
    keys = sorted(uniq)

    sf = float(vae.config.scaling_factor)
    vae_dtype = next(vae.parameters()).dtype

    cache = {
        "z_src": {},
        "raw_rgb": {},
        "raw_csd": {},
        "raw_dino": {},
        "rt_rgb": {},
        "rt_csd": {},
        "rt_dino": {},
    }

    with torch.inference_mode():
        for s in range(0, len(keys), 16):
            chunk = keys[s : s + 16]
            raw_pils = [
                Image.open(ROOT / uniq[k]).convert("RGB").resize((512, 512))
                for k in chunk
            ]
            x_01 = torch.stack(
                [
                    torch.from_numpy(np.array(p)).permute(2, 0, 1).float()
                    / 255.0
                    for p in raw_pils
                ],
                0,
            ).to(device)
            # VAE expects input normalized to [-1, 1]
            x_pm1 = x_01 * 2.0 - 1.0
            dist = vae.encode(x_pm1.to(vae_dtype)).latent_dist
            z = (dist.mean * sf).float()

            # Raw RGB features
            f_raw = csd_model.feats_pils(csd, raw_pils, batch=16, device=device)
            f_raw = f_raw / np.linalg.norm(f_raw, axis=1, keepdims=True)
            d_raw, _ = encode_dino(dino, raw_pils, batch=16, device=device)
            d_raw = F.normalize(d_raw.float(), dim=-1).cpu().numpy()

            # VAE Round-trip reconstruction
            img_rt = decode_trt(trt, vae, z, batch=8)
            u8_rt = (
                img_rt.mul(255.0).round().clamp(0, 255).to(torch.uint8).cpu()
            )
            arr_rt = u8_rt.permute(0, 2, 3, 1).numpy()
            rt_pils = [Image.fromarray(arr_rt[i]) for i in range(len(chunk))]

            f_rt = csd_model.feats_pils(csd, rt_pils, batch=16, device=device)
            f_rt = f_rt / np.linalg.norm(f_rt, axis=1, keepdims=True)
            d_rt, _ = encode_dino(dino, rt_pils, batch=16, device=device)
            d_rt = F.normalize(d_rt.float(), dim=-1).cpu().numpy()

            for i, k in enumerate(chunk):
                cache["z_src"][k] = z[i : i + 1].cpu()
                cache["raw_rgb"][k] = x_01[i].cpu()
                cache["raw_csd"][k] = f_raw[i]
                cache["raw_dino"][k] = d_raw[i]
                cache["rt_rgb"][k] = (
                    torch.from_numpy(arr_rt[i].copy())
                    .permute(2, 0, 1)
                    .float()
                    / 255.0
                )
                cache["rt_csd"][k] = f_rt[i]
                cache["rt_dino"][k] = d_rt[i]

            print(
                f"  encoded sources: {min(s+16, len(keys))}/{len(keys)}",
                flush=True,
            )

    torch.save(cache, cache_path)
    print(f"[P0] Saved source feature cache to {cache_path}", flush=True)
    return cache


def evaluate_tasks(
    args, tasks, gen_pack, src_cache, ref_data, proto_csd, models, device: str
):
    """Main evaluation loop for 750 tasks."""
    vae, trt, csd, dino, lpips_fn = (
        models["vae"],
        models["trt"],
        models["csd"],
        models["dino"],
        models["lpips"],
    )

    # Reference DINO CLS features per style (30, 384)
    ref_dino = {
        s: F.normalize(ref_data["styles"][s]["dino_cls"].float(), dim=-1).to(
            device
        )
        for s in STYLES
    }
    # Reference stems if available (for self-match masking)
    ref_stems = {
        s: [
            r.get("stem")
            for r in ref_data.get("references", {}).get(s, [])
            if "stem" in r
        ]
        for s in STYLES
    }

    batch_size = args.batch_size
    rows = []
    t0 = time.time()

    all_dino_feats = []
    all_csd_feats = []

    with torch.inference_mode():
        for s in range(0, len(tasks), batch_size):
            grp = tasks[s : s + batch_size]
            z = torch.cat(
                [src_cache["z_src"][t["source_stem"]].to(device) for t in grp],
                0,
            )
            sg = torch.tensor(
                [t["source_style_id"] for t in grp], device=device
            )
            tg = torch.tensor(
                [t["target_style_id"] for t in grp], device=device
            )

            z_gen = run_generator(gen_pack, z, sg, tg)
            img_gen = decode_trt(trt, vae, z_gen, batch=min(8, len(grp)))
            u8_gen = (
                img_gen.mul(255.0).round().clamp(0, 255).to(torch.uint8).cpu()
            )
            gen01 = u8_gen.float().div(255.0)
            arr_gen = u8_gen.permute(0, 2, 3, 1).numpy()
            pils_gen = [Image.fromarray(arr_gen[i]) for i in range(len(grp))]

            # Extract features of generated images
            fg = csd_model.feats_pils(csd, pils_gen, batch=16, device=device)
            fg = fg / np.linalg.norm(fg, axis=1, keepdims=True)
            dg, _ = encode_dino(dino, pils_gen, batch=16, device=device)
            dg = F.normalize(dg.float(), dim=-1)

            all_csd_feats.append(fg)
            all_dino_feats.append(dg.cpu().numpy())

            # Stack reference images for LPIPS
            raw_rgb_stack = torch.stack(
                [src_cache["raw_rgb"][t["source_stem"]] for t in grp]
            ).to(device)
            rt_rgb_stack = torch.stack(
                [src_cache["rt_rgb"][t["source_stem"]] for t in grp]
            ).to(device)

            lp_raw = (
                lpips_fn(
                    to_lpips_input(gen01.to(device)),
                    to_lpips_input(raw_rgb_stack),
                )
                .flatten()
                .cpu()
                .numpy()
            )
            lp_rt = (
                lpips_fn(
                    to_lpips_input(gen01.to(device)),
                    to_lpips_input(rt_rgb_stack),
                )
                .flatten()
                .cpu()
                .numpy()
            )

            for i, t in enumerate(grp):
                k = t["source_stem"]
                tid = t["target_style_id"]
                sid = t["source_style_id"]
                tgt_s = t["target_style"]
                src_s = t["source_style"]
                is_cross = bool(t.get("is_cross", tid != sid))

                # 1. CSD classification & margin
                sc = fg[i] @ proto_csd.T
                pred_c = int(sc.argmax())
                t_top1 = int(pred_c == tid)
                other_max = max(sc[idx] for idx in range(5) if idx != tid)
                t_margin = float(sc[tid] - other_max)

                # CSD delta T and delta S
                f_raw = src_cache["raw_csd"][k]
                f_rt = src_cache["rt_csd"][k]
                dT = float(sc[tid] - f_raw @ proto_csd[tid])
                dS = float(sc[sid] - f_raw @ proto_csd[sid])
                dT_rt = float(sc[tid] - f_rt @ proto_csd[tid])
                dS_rt = float(sc[sid] - f_rt @ proto_csd[sid])

                # 2. DINO-C (content fidelity)
                d_raw = src_cache["raw_dino"][k]
                d_rt = src_cache["rt_dino"][k]
                dino_c = float(dg[i].cpu().numpy() @ d_raw)
                dino_c_rt = float(dg[i].cpu().numpy() @ d_rt)

                # 3. DINO-S (style fidelity to 30 references)
                # Compute cosine to all 30 references of target style
                sims = (dg[i] @ ref_dino[tgt_s].T).cpu().numpy()
                # Exclude self if same style and stem matches
                if (
                    not is_cross
                    and ref_stems.get(tgt_s)
                    and k in ref_stems[tgt_s]
                ):
                    match_idx = ref_stems[tgt_s].index(k)
                    sims[match_idx] = -1e9
                dino_s = float(sims.max())

                # Baseline source image DINO-S to target references
                sims_src = (
                    torch.from_numpy(d_raw).to(device) @ ref_dino[tgt_s].T
                ).cpu().numpy()
                if (
                    not is_cross
                    and ref_stems.get(tgt_s)
                    and k in ref_stems[tgt_s]
                ):
                    match_idx = ref_stems[tgt_s].index(k)
                    sims_src[match_idx] = -1e9
                dino_s_raw = float(sims_src.max())
                dino_s_gain = float(dino_s - dino_s_raw)

                rows.append({
                    "task_id": t["task_id"],
                    "src_stem": k,
                    "src_style": src_s,
                    "tgt_style": tgt_s,
                    "src_style_id": sid,
                    "tgt_style_id": tid,
                    "is_cross": is_cross,
                    "target_top1": t_top1,
                    "pred_class": pred_c,
                    "target_margin": t_margin,
                    "dT": dT,
                    "dS": dS,
                    "dT_rt": dT_rt,
                    "dS_rt": dS_rt,
                    "dino_s": dino_s,
                    "dino_s_raw": dino_s_raw,
                    "dino_s_gain": dino_s_gain,
                    "dino_c": dino_c,
                    "dino_c_rt": dino_c_rt,
                    "lpips": float(lp_raw[i]),
                    "lpips_rt": float(lp_rt[i]),
                })

            if (s // batch_size) % 10 == 0:
                print(
                    f"  evaluated tasks: {min(s + batch_size, len(tasks))}/{len(tasks)} in {time.time()-t0:.1f}s",
                    flush=True,
                )

    all_csd_feats = np.concatenate(all_csd_feats, axis=0)
    all_dino_feats = np.concatenate(all_dino_feats, axis=0)
    return rows, all_csd_feats, all_dino_feats


def summarize_results(rows, out_dir: Path):
    """Compute aggregate statistics, CI, confusion matrix, and save deliverables."""
    cross = [r for r in rows if r["is_cross"]]
    same = [r for r in rows if not r["is_cross"]]
    n_cross = len(cross)
    n_same = len(same)

    MIN_IDX = STYLES.index("Minimalism")

    # 1. Cross600 Metrics
    t1_arr = np.array([r["target_top1"] for r in cross], dtype=float)
    stems_cross = [r["src_stem"] for r in cross]
    ci_top1 = cluster_ci(t1_arr, stems_cross)

    dt_arr = np.array([r["dT"] for r in cross], dtype=float)
    ci_dt = cluster_ci(dt_arr, stems_cross)

    dinos_arr = np.array([r["dino_s"] for r in cross], dtype=float)
    ci_dinos = cluster_ci(dinos_arr, stems_cross)

    dinoc_arr = np.array([r["dino_c"] for r in cross], dtype=float)
    lpips_arr = np.array([r["lpips"] for r in cross], dtype=float)
    margin_arr = np.array([r["target_margin"] for r in cross], dtype=float)

    # 5x5 CSD confusion matrix on Cross600
    conf = np.zeros((5, 5), dtype=int)
    for r in cross:
        conf[r["tgt_style_id"], r["pred_class"]] += 1

    # Off-target minimalism attractor rate: target != Minimalism, but pred == Minimalism
    off_min_hits = sum(conf[i, MIN_IDX] for i in range(5) if i != MIN_IDX)
    off_min_tot = sum(conf[i].sum() for i in range(5) if i != MIN_IDX)
    off_min_rate = (
        float(off_min_hits / off_min_tot) if off_min_tot > 0 else 0.0
    )

    # Per-target breakdowns
    per_target = {}
    for sid, s in enumerate(STYLES):
        sub = [r for r in cross if r["tgt_style"] == s]
        per_target[s] = {
            "n": len(sub),
            "top1": (
                float(np.mean([r["target_top1"] for r in sub])) if sub else 0.0
            ),
            "margin": (
                float(np.mean([r["target_margin"] for r in sub]))
                if sub
                else 0.0
            ),
            "dT": float(np.mean([r["dT"] for r in sub])) if sub else 0.0,
            "dino_s": (
                float(np.mean([r["dino_s"] for r in sub])) if sub else 0.0
            ),
            "dino_s_gain": (
                float(np.mean([r["dino_s_gain"] for r in sub])) if sub else 0.0
            ),
            "dino_c": (
                float(np.mean([r["dino_c"] for r in sub])) if sub else 0.0
            ),
            "lpips": float(np.mean([r["lpips"] for r in sub])) if sub else 0.0,
        }

    # 20 directed transfer pairs breakdown
    pairs = {}
    for r in cross:
        p_key = f"{r['src_style']}->{r['tgt_style']}"
        pairs.setdefault(p_key, []).append(r)
    pairs_summary = {
        k: {
            "top1": float(np.mean([x["target_top1"] for x in v])),
            "dT": float(np.mean([x["dT"] for x in v])),
            "dino_s": float(np.mean([x["dino_s"] for x in v])),
            "dino_c": float(np.mean([x["dino_c"] for x in v])),
            "lpips": float(np.mean([x["lpips"] for x in v])),
        }
        for k, v in pairs.items()
    }

    # 2. Same150 Metrics
    same_metrics = {}
    if same:
        same_metrics = {
            "n": n_same,
            "same_top1": float(np.mean([r["target_top1"] for r in same])),
            "same_margin": float(np.mean([r["target_margin"] for r in same])),
            "same_dS": float(np.mean([r["dS"] for r in same])),
            "same_dino_c": float(np.mean([r["dino_c"] for r in same])),
            "same_dino_s": float(np.mean([r["dino_s"] for r in same])),
            "same_lpips": float(np.mean([r["lpips"] for r in same])),
        }

    res = {
        "n_total": len(rows),
        "n_cross": n_cross,
        "n_same": n_same,
        "cross600": {
            "top1": float(t1_arr.mean()),
            "top1_ci95": ci_top1["ci"],
            "margin": float(margin_arr.mean()),
            "dT": float(dt_arr.mean()),
            "dT_ci95": ci_dt["ci"],
            "dino_s": float(dinos_arr.mean()),
            "dino_s_ci95": ci_dinos["ci"],
            "dino_s_gain": float(np.mean([r["dino_s_gain"] for r in cross])),
            "dino_c": float(dinoc_arr.mean()),
            "lpips": float(lpips_arr.mean()),
            "minimalism_offtarget": off_min_rate,
            "confusion_matrix": conf.tolist(),
            "per_target": per_target,
            "directed_pairs": pairs_summary,
            "diagnostics": {
                "mean_dT_rt": float(np.mean([r["dT_rt"] for r in cross])),
                "mean_dino_c_rt": float(
                    np.mean([r["dino_c_rt"] for r in cross])
                ),
                "mean_lpips_rt": float(np.mean([r["lpips_rt"] for r in cross])),
            },
        },
        "same150": same_metrics,
        "rows": rows,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "eval.json"
    out_file.write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 90)
    print(f"Evaluation Completed! Wrote {out_file}")
    print(
        f"Cross600: Top-1={res['cross600']['top1']*100:.1f}% "
        f"CI[{ci_top1['ci'][0]*100:.1f}, {ci_top1['ci'][1]*100:.1f}] | "
        f"Margin={res['cross600']['margin']:+.4f} | "
        f"dT={res['cross600']['dT']:+.4f} | "
        f"DINO-S={res['cross600']['dino_s']:.4f} (gain {res['cross600']['dino_s_gain']:+.4f}) | "
        f"DINO-C={res['cross600']['dino_c']:.4f} | "
        f"LPIPS={res['cross600']['lpips']:.4f}"
    )
    print(f"Off-target Minimalism Attractor Rate: {off_min_rate*100:.1f}%")
    if same:
        print(
            f"Same150 : Top-1={same_metrics['same_top1']*100:.1f}% | "
            f"Margin={same_metrics['same_margin']:+.4f} | "
            f"dS={same_metrics['same_dS']:+.4f} | "
            f"DINO-C={same_metrics['same_dino_c']:.4f} | "
            f"LPIPS={same_metrics['same_lpips']:.4f}"
        )
    print("=" * 90 + "\n")
    return res


def main():
    ap = argparse.ArgumentParser(description="WEAVE R2 Unified Evaluation")
    ap.add_argument("--tasks", default=str(DEFAULT_TASKS))
    ap.add_argument("--ref-features", default=str(DEFAULT_REFS))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--gen",
        choices=["direct", "oracle", "scorediff", "identity"],
        default="direct",
    )
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--canon-mode", default="")
    ap.add_argument(
        "--oracle-critic", default="runs/v6_next/probes/r1_r1.pt"
    )
    ap.add_argument("--oracle-rms", type=float, default=0.25)
    ap.add_argument("--oracle-steps", type=int, default=1)
    ap.add_argument(
        "--sd-ckpt", default="runs/v6/score_diff/score_diff_net.pt"
    )
    ap.add_argument("--sd-steps", type=int, default=24)
    ap.add_argument("--sd-lr", type=float, default=0.100)
    ap.add_argument("--sd-anchor", type=float, default=0.08)
    ap.add_argument("--sd-base-dim", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--cache-dir", default="runs/v6_r2/cache")
    ap.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    ap.add_argument("--limit", type=int, default=0, help="Smoke test limit")
    args = ap.parse_args()

    device = args.device
    tasks_data = json.loads(
        (ROOT / args.tasks).read_text(encoding="utf-8")
    )
    tasks = tasks_data["tasks"]
    if args.limit:
        tasks = tasks[: args.limit]
        print(f"[P0] SMOKE TEST: limiting to first {args.limit} tasks")

    # Load shared evaluator stack
    print("[P0] Loading VAE, TRT, CSD, DINO, and LPIPS models...", flush=True)
    vae = load_vae(device)
    trt = load_trt(vae, device)
    csd, _, _ = csd_model.load_csd(device)
    dino = load_dino_model(device=device)
    import lpips

    lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    models = {
        "vae": vae,
        "trt": trt,
        "csd": csd,
        "dino": dino,
        "lpips": lpips_fn,
    }

    # Precompute / load source features cache
    src_cache = build_source_cache(
        tasks_data["tasks"],
        vae,
        trt,
        csd,
        dino,
        ROOT / args.cache_dir,
        device,
    )

    # Load references features and CSD prototypes
    ref_data = torch.load(
        ROOT / args.ref_features, map_location="cpu", weights_only=False
    )
    proto_csd = csd_prototypes()

    # Load generator
    gen_pack = load_generator(args, device)

    # Evaluate
    rows, csd_feats, dino_feats = evaluate_tasks(
        args, tasks, gen_pack, src_cache, ref_data, proto_csd, models, device
    )

    # Summarize and save
    out_dir = ROOT / args.out_dir
    res = summarize_results(rows, out_dir)

    # Save feature arrays for downstream analysis
    torch.save(
        {
            "csd_feats": csd_feats,
            "dino_feats": dino_feats,
            "task_ids": [r["task_id"] for r in rows],
        },
        out_dir / "features.pt",
    )
    print(f"[P0] Saved features cache to {out_dir / 'features.pt'}")


if __name__ == "__main__":
    main()
