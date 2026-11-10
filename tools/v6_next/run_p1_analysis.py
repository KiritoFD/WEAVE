"""WEAVE R2 P1: Comprehensive R1 Gradient & Editing Analysis.

Answers Q1 & Q2 of the WEAVE R2 Plan:
- Scans edit strengths: E0 (RMS=0), E1 (0.08), E2 (0.15), E3 (0.25).
- Frequency subband isolation: LL-only (1st order Haar LL), HF-only (LH+HL+HH).
- Measures R1 margin, CSD Top-1/Margin/dT, DINO-S gain, DINO-C, LPIPS.
- Computes Haar 4-subband displacement energy breakdown.
- Measures content structure loss changes (L_ss self-similarity, L_inv canonical).
- Computes directional alignment cosines with Q2a delta, -∇L_inv, -∇L_ss, ScoreDiff.
- Separate breakdowns for Early Renaissance and Ukiyo-e.
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
from tools.v6.models_v6 import C3UNet, ConditionalScoreNet
from tools.v6.train_eval_v6 import STYLES, csd_model, encode_dino, load_dino_model
from tools.v6_next.probe_style import LatentStyleNet
from tools.v6_next.r2_teacher import (
    compute_content_gradient,
    compute_content_losses,
    compute_r1_gradient,
    cosine_similarity,
    filter_haar_subbands,
    rms_normalize,
    subband_energies,
)
from utils.run_evaluation import to_lpips_input

DEFAULT_TASKS = ROOT / "eval_protocol" / "tasks_750.json"
DEFAULT_REFS = ROOT / "eval_protocol" / "cache" / "references_v1_features.pt"
SRC_CACHE_PATH = ROOT / "runs" / "v6_r2" / "cache" / "tasks750_source_features.pt"


def main():
  ap = argparse.ArgumentParser(description="WEAVE R2 P1 Analysis")
  ap.add_argument("--tasks", default=str(DEFAULT_TASKS))
  ap.add_argument("--ref-features", default=str(DEFAULT_REFS))
  ap.add_argument("--src-cache", default=str(SRC_CACHE_PATH))
  ap.add_argument(
      "--q2a-ckpt", default="runs/v6/Q2a_embed_style/Q2a_embed_style.pt"
  )
  ap.add_argument("--critic-ckpt", default="runs/v6_next/probes/r1_r1.pt")
  ap.add_argument(
      "--sd-ckpt", default="runs/v6/score_diff/score_diff_net.pt"
  )
  ap.add_argument("--out-dir", default="runs/v6_r2/p1_analysis")
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

  # 1. Load tasks (filter to Val200 = 200 cross tasks from 50 sources unless --all-tasks)
  tasks_data = json.loads((ROOT / args.tasks).read_text(encoding="utf-8"))[
      "tasks"
  ]
  cross_tasks = [t for t in tasks_data if t["is_cross"]]
  if not args.all_tasks:
    # Use 50 Val200 sources
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
        f"[P1] Selected Val200: {len(selected_tasks)} cross tasks from 50"
        " sources"
    )
  else:
    selected_tasks = cross_tasks
    print(f"[P1] Selected Full Cross600: {len(selected_tasks)} tasks")

  # 2. Load models
  print("[P1] Loading Q2a, R1 Critic, ScoreDiff, and Evaluators...", flush=True)
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

  sd_ck = torch.load(ROOT / args.sd_ckpt, map_location="cpu", weights_only=False)
  sd_model = ConditionalScoreNet(base_dim=64, num_styles=5).to(device).eval()
  sd_model.load_state_dict(sd_ck["model"])
  from tools.v6.score_diff import edit_with_score_difference

  vae = load_vae(device)
  trt = load_trt(vae, device)
  csd, _, _ = csd_model.load_csd(device)
  dino = load_dino_model(device=device)
  import lpips

  lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

  # Load caches
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

  # 3. Define variants
  variants = [
      ("E0_q2a", {"rms": 0.0, "band": "all"}),
      ("E1_rms008", {"rms": 0.08, "band": "all"}),
      ("E2_rms015", {"rms": 0.15, "band": "all"}),
      ("E3_rms025", {"rms": 0.25, "band": "all"}),
      ("LL_rms025", {"rms": 0.25, "band": "ll"}),
      ("HF_rms025", {"rms": 0.25, "band": "hf"}),
  ]

  print(f"[P1] Starting evaluation of {len(variants)} variants on Val200...")
  results = {v[0]: [] for v in variants}
  alignment_stats = []

  batch_size = 8
  t0 = time.time()

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

    # 2. R1 Gradient & margin at z0
    g, m0 = compute_r1_gradient(z0, tg, critic)
    g_norm = rms_normalize(g)

    # 3. Directional Alignment Diagnostics (at z0)
    delta_q2a = z0 - zs
    cos_q2a = cosine_similarity(g, delta_q2a).cpu().numpy()

    # Content structure gradients
    g_content, l_c_curr = compute_content_gradient(
        z0, zs, canon_fn=canon, canon_mode="c1_gentle"
    )
    cos_neg_content = cosine_similarity(g, -g_content).cpu().numpy()

    # Self-similarity gradient alone
    with torch.enable_grad():
      zr_ss = z0.detach().clone().float().requires_grad_(True)
      l_ss_only, l_inv_only = compute_content_losses(zr_ss, zs, canon, "c1_gentle")
      g_ss = torch.autograd.grad(l_ss_only.sum(), zr_ss)[0]
      g_inv = torch.autograd.grad(l_inv_only.sum(), zr_ss)[0]
    cos_neg_ss = cosine_similarity(g, -g_ss).cpu().numpy()
    cos_neg_inv = cosine_similarity(g, -g_inv).cpu().numpy()

    # ScoreDiff direction (12 steps for comparison direction)
    with torch.inference_mode():
      z_sd = edit_with_score_difference(
          sd_model,
          zs,
          sg,
          tg,
          num_steps=12,
          step_size=0.160,
          struct_anchor_weight=0.08,
          seed=42,
      )
    delta_sd = z_sd - zs
    cos_sd = cosine_similarity(g, delta_sd).cpu().numpy()

    for i, t in enumerate(grp):
      alignment_stats.append({
          "task_id": t["task_id"],
          "source_style": t["source_style"],
          "target_style": t["target_style"],
          "cos_q2a_delta": float(cos_q2a[i]),
          "cos_neg_content": float(cos_neg_content[i]),
          "cos_neg_ss": float(cos_neg_ss[i]),
          "cos_neg_inv": float(cos_neg_inv[i]),
          "cos_scorediff": float(cos_sd[i]),
      })

    # 4. Generate & score each variant
    raw_rgb_stack = torch.stack(
        [src_cache["raw_rgb"][t["source_stem"]] for t in grp]
    ).to(device)

    for v_name, v_cfg in variants:
      rms_val = v_cfg["rms"]
      band = v_cfg["band"]

      if rms_val == 0.0:
        z_v = z0
      else:
        g_filt = filter_haar_subbands(g_norm, band=band)
        z_v = z0 + rms_val * g_filt

      # Measure R1 margin at z_v
      with torch.no_grad():
        q_v = critic(z_v.float()).float()
        oh = F.one_hot(tg, 5).bool()
        m_v = q_v.gather(1, tg[:, None]).squeeze(1) - torch.logsumexp(
            q_v.masked_fill(oh, -1e9), dim=1
        )
        r1_margin_delta = (m_v - m0).cpu().numpy()

      # Measure Haar energy breakdown of edit displacement (z_v - z0)
      disp = z_v - z0
      energies = subband_energies(disp)

      # Measure content structure losses
      with torch.no_grad():
        l_ss_v, l_inv_v = compute_content_losses(z_v, zs, canon, "c1_gentle")

      # Decode to image
      with torch.inference_mode():
        img_v = decode_trt(trt, vae, z_v, batch=min(8, len(grp)))
        u8_v = img_v.mul(255.0).round().clamp(0, 255).to(torch.uint8).cpu()
        gen01 = u8_v.float().div(255.0)
        arr_v = u8_v.permute(0, 2, 3, 1).numpy()
        pils_v = [Image.fromarray(arr_v[i]) for i in range(len(grp))]

        fv = csd_model.feats_pils(csd, pils_v, batch=16, device=device)
        fv = fv / np.linalg.norm(fv, axis=1, keepdims=True)
        dv, _ = encode_dino(dino, pils_v, batch=16, device=device)
        dv = F.normalize(dv.float(), dim=-1)

        lp_v = (
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

        sc = fv[i] @ proto_csd.T
        pred_c = int(sc.argmax())
        t_top1 = int(pred_c == tid)
        other_max = max(sc[idx] for idx in range(5) if idx != tid)
        t_margin = float(sc[tid] - other_max)

        f_raw = src_cache["raw_csd"][k]
        dT = float(sc[tid] - f_raw @ proto_csd[tid])
        dS = float(sc[sid] - f_raw @ proto_csd[sid])

        d_raw = src_cache["raw_dino"][k]
        dino_c = float(dv[i].cpu().numpy() @ d_raw)

        # DINO-S to 30 target references
        sims = (dv[i] @ ref_dino[tgt_s].T).cpu().numpy()
        dino_s = float(sims.max())
        sims_src = (
            torch.from_numpy(d_raw).to(device) @ ref_dino[tgt_s].T
        ).cpu().numpy()
        dino_s_gain = float(dino_s - sims_src.max())

        results[v_name].append({
            "task_id": t["task_id"],
            "src_stem": k,
            "src_style": src_s,
            "tgt_style": tgt_s,
            "target_top1": t_top1,
            "target_margin": t_margin,
            "dT": dT,
            "dS": dS,
            "dino_s": dino_s,
            "dino_s_gain": dino_s_gain,
            "dino_c": dino_c,
            "lpips": float(lp_v[i]),
            "r1_margin_delta": float(r1_margin_delta[i]),
            "prop_ll": float(energies["prop_ll"][i].cpu()),
            "prop_lh": float(energies["prop_lh"][i].cpu()),
            "prop_hl": float(energies["prop_hl"][i].cpu()),
            "prop_hh": float(energies["prop_hh"][i].cpu()),
            "l_ss": float(l_ss_v[i].cpu()),
            "l_inv": float(l_inv_v[i].cpu()),
        })

    if (s // batch_size) % 5 == 0:
      print(
          f"  processed {min(s + batch_size, len(selected_tasks))}/{len(selected_tasks)} tasks in {time.time()-t0:.1f}s",
          flush=True,
      )

  # 5. Summarize and produce reports
  summary = {}
  for v_name in results:
    rows_v = results[v_name]
    summary[v_name] = {
        "top1": float(np.mean([r["target_top1"] for r in rows_v])),
        "margin": float(np.mean([r["target_margin"] for r in rows_v])),
        "dT": float(np.mean([r["dT"] for r in rows_v])),
        "dino_s": float(np.mean([r["dino_s"] for r in rows_v])),
        "dino_s_gain": float(np.mean([r["dino_s_gain"] for r in rows_v])),
        "dino_c": float(np.mean([r["dino_c"] for r in rows_v])),
        "lpips": float(np.mean([r["lpips"] for r in rows_v])),
        "r1_margin_delta": float(
            np.mean([r["r1_margin_delta"] for r in rows_v])
        ),
        "energy_prop": {
            "ll": float(np.mean([r["prop_ll"] for r in rows_v])),
            "lh": float(np.mean([r["prop_lh"] for r in rows_v])),
            "hl": float(np.mean([r["prop_hl"] for r in rows_v])),
            "hh": float(np.mean([r["prop_hh"] for r in rows_v])),
        },
        "content_losses": {
            "l_ss": float(np.mean([r["l_ss"] for r in rows_v])),
            "l_inv": float(np.mean([r["l_inv"] for r in rows_v])),
        },
        "per_target": {
            s: {
                "top1": float(
                    np.mean(
                        [r["target_top1"] for r in rows_v if r["tgt_style"] == s]
                    )
                ),
                "dino_s": float(
                    np.mean([r["dino_s"] for r in rows_v if r["tgt_style"] == s])
                ),
                "dino_c": float(
                    np.mean([r["dino_c"] for r in rows_v if r["tgt_style"] == s])
                ),
                "margin": float(
                    np.mean(
                        [r["target_margin"] for r in rows_v if r["tgt_style"] == s]
                    )
                ),
            }
            for s in STYLES
        },
    }

  # Alignment summary
  align_summary = {
      "mean_cos_q2a_delta": float(
          np.mean([a["cos_q2a_delta"] for a in alignment_stats])
      ),
      "mean_cos_neg_content": float(
          np.mean([a["cos_neg_content"] for a in alignment_stats])
      ),
      "mean_cos_neg_ss": float(
          np.mean([a["cos_neg_ss"] for a in alignment_stats])
      ),
      "mean_cos_neg_inv": float(
          np.mean([a["cos_neg_inv"] for a in alignment_stats])
      ),
      "mean_cos_scorediff": float(
          np.mean([a["cos_scorediff"] for a in alignment_stats])
      ),
      "per_target": {
          s: {
              "cos_q2a": float(
                  np.mean([
                      a["cos_q2a_delta"]
                      for a in alignment_stats
                      if a["target_style"] == s
                  ])
              ),
              "cos_neg_content": float(
                  np.mean([
                      a["cos_neg_content"]
                      for a in alignment_stats
                      if a["target_style"] == s
                  ])
              ),
              "cos_neg_ss": float(
                  np.mean([
                      a["cos_neg_ss"]
                      for a in alignment_stats
                      if a["target_style"] == s
                  ])
              ),
              "cos_neg_inv": float(
                  np.mean([
                      a["cos_neg_inv"]
                      for a in alignment_stats
                      if a["target_style"] == s
                  ])
              ),
              "cos_scorediff": float(
                  np.mean([
                      a["cos_scorediff"]
                      for a in alignment_stats
                      if a["target_style"] == s
                  ])
              ),
          }
          for s in STYLES
      },
  }

  out_pack = {
      "summary": summary,
      "alignment": align_summary,
      "detailed_results": results,
      "detailed_alignment": alignment_stats,
  }

  (out_dir / "p1_summary.json").write_text(
      json.dumps(out_pack, indent=2), encoding="utf-8"
  )
  print(f"[P1] Successfully saved analysis to {out_dir / 'p1_summary.json'}")

  # Print Markdown Table for review
  print("\n" + "=" * 105)
  print("P1 R1 GRADIENT EDITING & SUBBAND SCAN (Val200)")
  print("=" * 105)
  print(
      f"  {'Variant':12s} | {'Top-1':>7s} | {'Margin':>8s} | {'ΔT':>7s} |"
      f" {'DINO-S (gain)':>17s} | {'DINO-C':>7s} | {'LPIPS':>7s} |"
      f" {'ΔR1-M':>7s} | {'LL/HF Energy':>14s}"
  )
  print("  " + "-" * 100)
  for v_name in variants:
    s = summary[v_name[0]]
    en = s["energy_prop"]
    en_str = (
        f"{en['ll']*100:.0f}% / {((en['lh']+en['hl']+en['hh'])*100):.0f}%"
        if v_name[0] != "E0_q2a"
        else "—"
    )
    gain_str = f"{s['dino_s']:.4f} ({s['dino_s_gain']:+.4f})"
    print(
        f"  {v_name[0]:12s} | {s['top1']*100:6.1f}% | {s['margin']:+8.4f} |"
        f" {s['dT']:+7.4f} | {gain_str:>17s} | {s['dino_c']:7.4f} |"
        f" {s['lpips']:7.4f} | {s['r1_margin_delta']:+7.3f} | {en_str:>14s}"
    )

  print("\n" + "=" * 105)
  print("DIRECTIONAL ALIGNMENT DIAGNOSTICS (Cosine with R1 Gradient)")
  print("=" * 105)
  print(f"  cos(g, Q2a Δ)       : {align_summary['mean_cos_q2a_delta']:+.4f}")
  print(
      f"  cos(g, -∇L_content) : {align_summary['mean_cos_neg_content']:+.4f}"
      " (Negative means R1 gradient directly conflicts with content structure!)"
  )
  print(f"  cos(g, -∇L_ss)      : {align_summary['mean_cos_neg_ss']:+.4f}")
  print(f"  cos(g, -∇L_inv)     : {align_summary['mean_cos_neg_inv']:+.4f}")
  print(f"  cos(g, ScoreDiff Δ) : {align_summary['mean_cos_scorediff']:+.4f}")

  print("\nPer-Target Cosine with Content Negative Gradient (-∇L_content):")
  for s in STYLES:
    c_s = align_summary["per_target"][s]["cos_neg_content"]
    print(
        f"  {s:20s}: {c_s:+.4f}"
        f"  {'⚠️ Severe Content Conflict' if c_s < -0.1 else ''}"
    )


if __name__ == "__main__":
  main()
