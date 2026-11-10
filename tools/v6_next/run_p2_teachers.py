"""WEAVE R2 P2: Content-Constrained R1 Gradient Teacher Construction & Diagnostics.

Implements Section P2 of WEAVE R2 12-Hour Execution Plan:
- T0: Q2a Direct (no gradient baseline)
- T1: Historical Oracle (unconstrained R1 gradient, RMS=0.25)
- T2: R1 Gradient with reduced step (RMS=0.15)
- T3: Content Tangential Projection (removes gradient component conflicting with L_C)
- T4: Optimal frequency band + Content Tangential Projection
- Residual Anchor: z_anchor = zs + (G_Q2a(c,t) - c)
- Diagnostics: VAE roundtrip stability RMS(z_rt - z_edit), real style nearest neighbor distance.
"""
from __future__ import annotations

import argparse
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
from tools.v6.models_v6 import C3UNet
from tools.v6.train_eval_v6 import STYLES, csd_model, encode_dino, load_dino_model
from tools.v6_next.probe_style import LatentStyleNet
from tools.v6_next.r2_teacher import (
    compute_content_gradient,
    compute_content_losses,
    compute_r1_gradient,
    filter_haar_subbands,
    project_tangential,
    rms_normalize,
)
from utils.run_evaluation import to_lpips_input

DEFAULT_TASKS = ROOT / "eval_protocol" / "tasks_750.json"
DEFAULT_REFS = ROOT / "eval_protocol" / "cache" / "references_v1_features.pt"
SRC_CACHE_PATH = ROOT / "runs" / "v6_r2" / "cache" / "tasks750_source_features.pt"


def main():
  ap = argparse.ArgumentParser(description="WEAVE R2 P2 Teacher Construction")
  ap.add_argument("--tasks", default=str(DEFAULT_TASKS))
  ap.add_argument("--ref-features", default=str(DEFAULT_REFS))
  ap.add_argument("--src-cache", default=str(SRC_CACHE_PATH))
  ap.add_argument(
      "--q2a-ckpt", default="runs/v6/Q2a_embed_style/Q2a_embed_style.pt"
  )
  ap.add_argument("--critic-ckpt", default="runs/v6_next/probes/r1_r1.pt")
  ap.add_argument("--out-dir", default="runs/v6_r2/p2_teachers")
  ap.add_argument(
      "--device", default="cuda" if torch.cuda.is_available() else "cpu"
  )
  ap.add_argument(
      "--all-tasks",
      action="store_true",
      help="Run on full 600 cross instead of Val200",
  )
  args = ap.parse_args()

  device = args.device
  out_dir = ROOT / args.out_dir
  out_dir.mkdir(parents=True, exist_ok=True)

  tasks_data = json.loads((ROOT / args.tasks).read_text(encoding="utf-8"))[
      "tasks"
  ]
  cross_tasks = [t for t in tasks_data if t["is_cross"]]
  if not args.all_tasks:
    val_cand = [
        ROOT / "runs/v5/cache/eval_src_latents.pt",
        ROOT / "deliver/testset/refs/eval_src_latents.pt",
    ]
    val_file = next(p for p in val_cand if p.exists())
    val_stems = sorted(
        list(torch.load(val_file, map_location="cpu", weights_only=False).keys())
    )
    selected_tasks = [t for t in cross_tasks if t["source_stem"] in val_stems]
    print(
        f"[P2] Selected Val200: {len(selected_tasks)} cross tasks from 50"
        " sources"
    )
  else:
    selected_tasks = cross_tasks
    print(f"[P2] Selected Full Cross600: {len(selected_tasks)} tasks")

  print("[P2] Loading Models and Evaluators...", flush=True)
  ck_q = torch.load(ROOT / args.q2a_ckpt, map_location="cpu", weights_only=False)
  q2a_args = ck_q.get("args", {})
  q2a = C3UNet(
      base_dim=q2a_args.get("base_dim", 16),
      num_styles=5,
      mode="direct",
      inject_64=True,
      inject_32=True,
      inject_16=True,
  ).to(device)
  q2a.load_state_dict(ck_q["model"])
  q2a.eval()

  canon = ContentCanonicalizer(
      torch.load(
          ROOT / "runs/v6/cache/canonicalizer_stats.pt",
          map_location="cpu",
          weights_only=False,
      )
  ).to(device)

  ck_c = torch.load(
      ROOT / args.critic_ckpt, map_location="cpu", weights_only=False
  )
  critic = LatentStyleNet(width=ck_c.get("width", 64)).to(device).eval()
  critic.load_state_dict(ck_c["model"])
  for p in critic.parameters():
    p.requires_grad_(False)

  vae = load_vae(device)
  trt = load_trt(vae, device)
  csd, _, _ = csd_model.load_csd(device)
  dino = load_dino_model(device=device)
  import lpips

  lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

  src_cache = torch.load(
      ROOT / args.src_cache, map_location="cpu", weights_only=False
  )
  ref_data = torch.load(
      ROOT / args.ref_features, map_location="cpu", weights_only=False
  )
  ref_dino = {
      s: F.normalize(ref_data["styles"][s]["dino_cls"].float(), dim=-1).to(
          device
      )
      for s in STYLES
  }
  proto_csd = csd_prototypes()

  # Teachers to evaluate
  teachers = [
      ("T0_q2a_base", "Q2a Direct (no gradient baseline)"),
      ("T1_oracle_rms025", "Historical Oracle (unconstrained R1, RMS=0.25)"),
      ("T2_oracle_rms015", "R1 gradient with reduced step (RMS=0.15)"),
      ("T3_projected_rms025", "Content Tangential Projection (RMS=0.25)"),
      ("T3_projected_rms015", "Content Tangential Projection (RMS=0.15)"),
      ("T4_projected_ll025", "LL-band + Content Tangential Projection (RMS=0.25)"),
      ("T_residual_anchor", "Q2a Residual Anchor: zs + delta(c,t)"),
  ]

  print(f"[P2] Evaluating {len(teachers)} teacher candidates...")
  results = {t[0]: [] for t in teachers}
  diagnostics = {
      t[0]: {"rt_stability_rms": [], "l_ss": [], "l_inv": []} for t in teachers
  }

  batch_size = 8
  t0 = time.time()
  sf = float(vae.config.scaling_factor)

  for s in range(0, len(selected_tasks), batch_size):
    grp = selected_tasks[s : s + batch_size]
    zs = torch.cat(
        [src_cache["z_src"][t["source_stem"]].to(device) for t in grp], 0
    )
    tg = torch.tensor([t["target_style_id"] for t in grp], device=device)
    sg = torch.tensor([t["source_style_id"] for t in grp], device=device)

    # 1. Base Q2a prediction
    with torch.inference_mode():
      c = canon(zs, mode="c1_gentle", augment=False)
      z0 = q2a(c, 0.0, c, tg).detach()

    # 2. R1 Gradient & Content Gradient
    g_s, _ = compute_r1_gradient(z0, tg, critic)
    g_s_norm = rms_normalize(g_s)

    g_c, _ = compute_content_gradient(
        z0, zs, canon_fn=canon, canon_mode="c1_gentle"
    )

    # Tangential projection
    d_proj = project_tangential(g_s, g_c)
    d_proj_norm = rms_normalize(d_proj)

    # LL filtered projection
    d_proj_ll = filter_haar_subbands(d_proj_norm, band="ll")

    # Generate each teacher's target latent z_t
    z_candidates = {}
    z_candidates["T0_q2a_base"] = z0
    z_candidates["T1_oracle_rms025"] = z0 + 0.25 * g_s_norm
    z_candidates["T2_oracle_rms015"] = z0 + 0.15 * g_s_norm
    z_candidates["T3_projected_rms025"] = z0 + 0.25 * d_proj_norm
    z_candidates["T3_projected_rms015"] = z0 + 0.15 * d_proj_norm
    z_candidates["T4_projected_ll025"] = z0 + 0.25 * d_proj_ll
    # Residual Anchor: zs + (z0 - c)
    z_candidates["T_residual_anchor"] = zs + (z0 - c)

    raw_rgb_stack = torch.stack(
        [src_cache["raw_rgb"][t["source_stem"]] for t in grp]
    ).to(device)

    for t_name, _ in teachers:
      z_t = z_candidates[t_name]

      # Diagnostics: VAE round-trip stability
      with torch.inference_mode():
        img_t = decode_trt(trt, vae, z_t, batch=min(8, len(grp)))
        # Re-encode
        img_pm1 = img_t * 2.0 - 1.0
        dist_rt = vae.encode(img_pm1.to(vae.dtype)).latent_dist
        z_rt = (dist_rt.mean * sf).float()
        rt_rms = (
            (z_rt - z_t)
            .flatten(1)
            .pow(2)
            .mean(dim=1)
            .sqrt()
            .cpu()
            .numpy()
        )

        l_ss_val, l_inv_val = compute_content_losses(
            z_t, zs, canon, "c1_gentle"
        )
        diagnostics[t_name]["rt_stability_rms"].extend([float(x) for x in rt_rms])
        diagnostics[t_name]["l_ss"].extend([float(x) for x in l_ss_val.cpu()])
        diagnostics[t_name]["l_inv"].extend([float(x) for x in l_inv_val.cpu()])

        # Metric extraction
        u8_t = img_t.mul(255.0).round().clamp(0, 255).to(torch.uint8).cpu()
        gen01 = u8_t.float().div(255.0)
        arr_t = u8_t.permute(0, 2, 3, 1).numpy()
        pils_t = [Image.fromarray(arr_t[i]) for i in range(len(grp))]

        ft = csd_model.feats_pils(csd, pils_t, batch=16, device=device)
        ft = ft / np.linalg.norm(ft, axis=1, keepdims=True)
        dt_feat, _ = encode_dino(dino, pils_t, batch=16, device=device)
        dt_feat = F.normalize(dt_feat.float(), dim=-1)

        lp_t = (
            lpips_fn(to_lpips_input(gen01.to(device)), to_lpips_input(raw_rgb_stack))
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

        sc = ft[i] @ proto_csd.T
        pred_c = int(sc.argmax())
        t_top1 = int(pred_c == tid)
        other_max = max(sc[idx] for idx in range(5) if idx != tid)
        t_margin = float(sc[tid] - other_max)

        f_raw = src_cache["raw_csd"][k]
        dT = float(sc[tid] - f_raw @ proto_csd[tid])

        d_raw = src_cache["raw_dino"][k]
        dino_c = float(dt_feat[i].cpu().numpy() @ d_raw)

        sims = (dt_feat[i] @ ref_dino[tgt_s].T).cpu().numpy()
        dino_s = float(sims.max())
        sims_src = (
            torch.from_numpy(d_raw).to(device) @ ref_dino[tgt_s].T
        ).cpu().numpy()
        dino_s_gain = float(dino_s - sims_src.max())

        results[t_name].append({
            "task_id": t["task_id"],
            "src_stem": k,
            "src_style": src_s,
            "tgt_style": tgt_s,
            "target_top1": t_top1,
            "target_margin": t_margin,
            "dT": dT,
            "dino_s": dino_s,
            "dino_s_gain": dino_s_gain,
            "dino_c": dino_c,
            "lpips": float(lp_t[i]),
            "rt_rms": float(rt_rms[i]),
            "l_ss": float(l_ss_val[i].cpu()),
            "l_inv": float(l_inv_val[i].cpu()),
        })

    if (s // batch_size) % 5 == 0:
      print(
          f"  processed {min(s + batch_size, len(selected_tasks))}/{len(selected_tasks)} tasks in {time.time()-t0:.1f}s",
          flush=True,
      )

  # Summarize
  summary = {}
  for t_name, desc in teachers:
    rows_t = results[t_name]
    summary[t_name] = {
        "description": desc,
        "top1": float(np.mean([r["target_top1"] for r in rows_t])),
        "margin": float(np.mean([r["target_margin"] for r in rows_t])),
        "dT": float(np.mean([r["dT"] for r in rows_t])),
        "dino_s": float(np.mean([r["dino_s"] for r in rows_t])),
        "dino_s_gain": float(np.mean([r["dino_s_gain"] for r in rows_t])),
        "dino_c": float(np.mean([r["dino_c"] for r in rows_t])),
        "lpips": float(np.mean([r["lpips"] for r in rows_t])),
        "rt_stability_rms": float(
            np.mean(diagnostics[t_name]["rt_stability_rms"])
        ),
        "l_ss": float(np.mean(diagnostics[t_name]["l_ss"])),
        "l_inv": float(np.mean(diagnostics[t_name]["l_inv"])),
        "per_target": {
            s: {
                "top1": float(
                    np.mean(
                        [r["target_top1"] for r in rows_t if r["tgt_style"] == s]
                    )
                ),
                "dino_s": float(
                    np.mean([r["dino_s"] for r in rows_t if r["tgt_style"] == s])
                ),
                "dino_c": float(
                    np.mean([r["dino_c"] for r in rows_t if r["tgt_style"] == s])
                ),
                "margin": float(
                    np.mean(
                        [r["target_margin"] for r in rows_t if r["tgt_style"] == s]
                    )
                ),
            }
            for s in STYLES
        },
    }

  out_pack = {
      "summary": summary,
      "results": results,
  }

  (out_dir / "p2_teachers_summary.json").write_text(
      json.dumps(out_pack, indent=2), encoding="utf-8"
  )
  print(
      f"[P2] Successfully saved teacher evaluation to"
      f" {out_dir / 'p2_teachers_summary.json'}"
  )

  print("\n" + "=" * 115)
  print("P2 TEACHER COMPARISON & CONTENT-CONSTRAINED PARETO (Val200)")
  print("=" * 115)
  header = (
      f"  {'Teacher':22s} | {'Top-1':>7s} | {'Margin':>8s} | {'ΔT':>7s} |"
      f" {'DINO-S (gain)':>17s} | {'DINO-C':>7s} | {'LPIPS':>7s} | {'RT RMS':>7s} | {'L_ss':>7s}"
  )
  print(header)
  print("  " + "-" * 110)
  for t_name, desc in teachers:
    s = summary[t_name]
    gain_str = f"{s['dino_s']:.4f} ({s['dino_s_gain']:+.4f})"
    print(
        f"  {t_name:22s} | {s['top1']*100:6.1f}% | {s['margin']:+8.4f} |"
        f" {s['dT']:+7.4f} | {gain_str:>17s} | {s['dino_c']:7.4f} |"
        f" {s['lpips']:7.4f} | {s['rt_stability_rms']:7.4f} | {s['l_ss']:7.4f}"
    )

  print("\n" + "=" * 115)
  print("DECISION: Which teacher expands the Pareto frontier in DINO-C >= 0.78?")
  print("=" * 115)
  valid_teachers = [
      (k, v) for k, v in summary.items() if v["dino_c"] >= 0.78 and k != "T0_q2a_base"
  ]
  if valid_teachers:
    best_t = max(valid_teachers, key=lambda x: (x[1]["top1"], x[1]["dino_s"]))
    print(
        f"  Best teacher in DINO-C >= 0.78: {best_t[0]} ({best_t[1]['description']})"
    )
    print(
        f"    Top-1: {best_t[1]['top1']*100:.1f}%, DINO-S: {best_t[1]['dino_s']:.4f}, DINO-C: {best_t[1]['dino_c']:.4f}"
    )
  else:
    print("  No teacher candidate achieved DINO-C >= 0.78")


if __name__ == "__main__":
  main()
