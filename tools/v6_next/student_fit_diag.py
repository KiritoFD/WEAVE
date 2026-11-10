"""Student Fitting Diagnostics against Teacher (Section P3.4 of WEAVE R2 Plan).

Measures:
1. Teacher displacement delta_T = z_teacher - z0
2. Student displacement delta_S = z_student - z0
3. Displacement cosine cos(delta_S, delta_T)
4. Displacement RMS ratio RMS(delta_S) / RMS(delta_T)
5. Relative fit error MSE(z_student, z_teacher) / MSE(z_teacher, z0)
6. Haar subband fitting errors (LL, LH, HL, HH)
7. R1 target margin mimicry: |m_R1(z_student) - m_R1(z_teacher)|
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from wavelet import dwt2_haar

from tools.v6.canonicalize import ContentCanonicalizer
from tools.v6.models_v6 import C3UNet
from tools.v6.train_eval_v6 import STYLES
from tools.v6_next.probe_style import LatentStyleNet
from tools.v6_next.r2_teacher import (
    compute_content_gradient,
    compute_content_losses,
    compute_r1_gradient,
    cosine_similarity,
    filter_haar_subbands,
    project_tangential,
    rms_normalize,
)

DEFAULT_TASKS = ROOT / "eval_protocol" / "tasks_750.json"
SRC_CACHE_PATH = ROOT / "runs" / "v6_r2" / "cache" / "tasks750_source_features.pt"


def main():
  ap = argparse.ArgumentParser(description="Student Fitting Diagnostics")
  ap.add_argument("--student-ckpt", required=True)
  ap.add_argument(
      "--teacher-ckpt", default="runs/v6/Q2a_embed_style/Q2a_embed_style.pt"
  )
  ap.add_argument("--critic-ckpt", default="runs/v6_next/probes/r1_r1.pt")
  ap.add_argument(
      "--teacher-mode",
      choices=["oracle", "projected", "projected_ll"],
      default="projected_ll",
  )
  ap.add_argument("--teacher-rms", type=float, default=0.25)
  ap.add_argument("--tasks", default=str(DEFAULT_TASKS))
  ap.add_argument("--src-cache", default=str(SRC_CACHE_PATH))
  ap.add_argument("--out-dir", default="")
  ap.add_argument(
      "--device", default="cuda" if torch.cuda.is_available() else "cpu"
  )
  args = ap.parse_args()

  device = args.device

  # Load Student
  ck_s = torch.load(
      ROOT / args.student_ckpt, map_location="cpu", weights_only=False
  )
  s_args = ck_s.get("args", {})
  student = C3UNet(
      base_dim=s_args.get("base_dim", 16),
      num_styles=5,
      mode="direct",
      inject_64=True,
      inject_32=True,
      inject_16=True,
  ).to(device)
  student.load_state_dict(ck_s["model"])
  student.eval()

  # Load Teacher Base (Q2a)
  ck_t = torch.load(
      ROOT / args.teacher_ckpt, map_location="cpu", weights_only=False
  )
  t_args = ck_t.get("args", {})
  teacher_base = C3UNet(
      base_dim=t_args.get("base_dim", 16),
      num_styles=5,
      mode="direct",
      inject_64=True,
      inject_32=True,
      inject_16=True,
  ).to(device)
  teacher_base.load_state_dict(ck_t["model"])
  teacher_base.eval()

  canon = ContentCanonicalizer(
      torch.load(
          ROOT / "runs/v6/cache/canonicalizer_stats.pt",
          map_location="cpu",
          weights_only=False,
      )
  ).to(device)

  # Critic
  ck_c = torch.load(
      ROOT / args.critic_ckpt, map_location="cpu", weights_only=False
  )
  critic = LatentStyleNet(width=ck_c.get("width", 64)).to(device).eval()
  critic.load_state_dict(ck_c["model"])
  for p in critic.parameters():
    p.requires_grad_(False)

  # Tasks (Val200)
  tasks_data = json.loads((ROOT / args.tasks).read_text(encoding="utf-8"))[
      "tasks"
  ]
  val_cand = [
      ROOT / "runs/v5/cache/eval_src_latents.pt",
      ROOT / "deliver/testset/refs/eval_src_latents.pt",
  ]
  val_file = next(p for p in val_cand if p.exists())
  val_stems = sorted(
      list(torch.load(val_file, map_location="cpu", weights_only=False).keys())
  )
  selected_tasks = [
      t
      for t in tasks_data
      if t["is_cross"] and t["source_stem"] in val_stems
  ]

  src_cache = torch.load(
      ROOT / args.src_cache, map_location="cpu", weights_only=False
  )

  cosines = []
  rms_ratios = []
  rel_fit_errors = []
  mse_students = []
  mse_denoms = []
  haar_errors = {"ll": [], "lh": [], "hl": [], "hh": []}
  margin_diffs = []

  batch_size = 8
  with torch.no_grad():
    for s in range(0, len(selected_tasks), batch_size):
      grp = selected_tasks[s : s + batch_size]
      zs = torch.cat(
          [src_cache["z_src"][t["source_stem"]].to(device) for t in grp], 0
      )
      tg = torch.tensor([t["target_style_id"] for t in grp], device=device)

      c = canon(zs, mode="c1_gentle", augment=False)
      z0 = teacher_base(c, 0.0, c, tg)
      z_S = student(c, 0.0, c, tg)

      # Build Teacher z_T
      g_s, _ = compute_r1_gradient(z0, tg, critic)
      g_s_norm = rms_normalize(g_s)

      if args.teacher_mode == "oracle":
        z_T = z0 + args.teacher_rms * g_s_norm
      elif args.teacher_mode in ("projected", "projected_ll"):
        g_c, _ = compute_content_gradient(
            z0, zs, canon_fn=canon, canon_mode="c1_gentle"
        )
        d_proj = project_tangential(g_s, g_c)
        d_proj_norm = rms_normalize(d_proj)
        if args.teacher_mode == "projected_ll":
          d_proj_norm = filter_haar_subbands(d_proj_norm, band="ll")
        z_T = z0 + args.teacher_rms * d_proj_norm

      delta_T = z_T - z0
      delta_S = z_S - z0

      # Cosine
      cos_ts = cosine_similarity(delta_S, delta_T).cpu().numpy()
      cosines.extend([float(x) for x in cos_ts])

      # RMS Ratio
      rms_T = (
          delta_T.flatten(1)
          .pow(2)
          .mean(dim=1)
          .sqrt()
          .clamp(min=1e-8)
          .cpu()
          .numpy()
      )
      rms_S = delta_S.flatten(1).pow(2).mean(dim=1).sqrt().cpu().numpy()
      rms_ratios.extend([float(s / t) for s, t in zip(rms_S, rms_T)])

      # Relative Fit Error: ||z_S - z_T||^2 / ||z_T - z0||^2
      mse_num = (z_S - z_T).flatten(1).pow(2).mean(dim=1).cpu().numpy()
      mse_den = delta_T.flatten(1).pow(2).mean(dim=1).clamp(min=1e-8).cpu().numpy()
      rel_fit_errors.extend([float(n / d) for n, d in zip(mse_num, mse_den)])
      mse_students.extend([float(x) for x in mse_num])
      mse_denoms.extend([float(x) for x in mse_den])

      # Haar Subband Errors
      s_ll, s_lh, s_hl, s_hh = dwt2_haar(z_S.float())
      t_ll, t_lh, t_hl, t_hh = dwt2_haar(z_T.float())
      haar_errors["ll"].extend([
          float(x)
          for x in (s_ll - t_ll).flatten(1).pow(2).mean(dim=1).cpu().numpy()
      ])
      haar_errors["lh"].extend([
          float(x)
          for x in (s_lh - t_lh).flatten(1).pow(2).mean(dim=1).cpu().numpy()
      ])
      haar_errors["hl"].extend([
          float(x)
          for x in (s_hl - t_hl).flatten(1).pow(2).mean(dim=1).cpu().numpy()
      ])
      haar_errors["hh"].extend([
          float(x)
          for x in (s_hh - t_hh).flatten(1).pow(2).mean(dim=1).cpu().numpy()
      ])

      # Margin mimicry
      q_S = critic(z_S.float()).float()
      q_T = critic(z_T.float()).float()
      oh = F.one_hot(tg, 5).bool()
      m_S = q_S.gather(1, tg[:, None]).squeeze(1) - torch.logsumexp(
          q_S.masked_fill(oh, -1e9), dim=1
      )
      m_T = q_T.gather(1, tg[:, None]).squeeze(1) - torch.logsumexp(
          q_T.masked_fill(oh, -1e9), dim=1
      )
      margin_diffs.extend([float(x) for x in (m_S - m_T).abs().cpu().numpy()])

  report = {
      "student_ckpt": args.student_ckpt,
      "teacher_mode": args.teacher_mode,
      "mean_cosine": float(np.mean(cosines)),
      "mean_rms_ratio": float(np.mean(rms_ratios)),
      "mean_rel_fit_error": float(np.mean(rel_fit_errors)),
      "mean_mse_student": float(np.mean(mse_students)),
      "mean_mse_teacher_disp": float(np.mean(mse_denoms)),
      "haar_mse": {k: float(np.mean(v)) for k, v in haar_errors.items()},
      "mean_margin_diff": float(np.mean(margin_diffs)),
  }

  print("=" * 80)
  print(f"P3.4 STUDENT FITTING DIAGNOSTICS: {Path(args.student_ckpt).stem}")
  print(f"Teacher Mode: {args.teacher_mode} (RMS={args.teacher_rms})")
  print("=" * 80)
  print(f"  Displacement Cosine cos(ΔS, ΔT) : {report['mean_cosine']:+.4f}")
  print(f"  Displacement RMS Ratio          : {report['mean_rms_ratio']:.4f}")
  print(f"  Relative Fit Error ||S-T||²/||T-0||²: {report['mean_rel_fit_error']:.4f}")
  print(f"  Absolute Student MSE to Teacher : {report['mean_mse_student']:.6f}")
  print(f"  Haar Subband Fitting MSE:")
  print(
      f"    LL: {report['haar_mse']['ll']:.6f} | LH:"
      f" {report['haar_mse']['lh']:.6f} | HL: {report['haar_mse']['hl']:.6f} |"
      f" HH: {report['haar_mse']['hh']:.6f}"
  )
  print(
      f"  R1 Margin Gap |m(S) - m(T)|     : {report['mean_margin_diff']:.4f}"
  )
  print("=" * 80)

  if args.out_dir:
    out_p = Path(args.out_dir) / "fit_diag.json"
    out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out_p}")


if __name__ == "__main__":
  main()
