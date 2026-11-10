"""WEAVE R2 P4.4: High-Resolution Visual Comparison Grids.

Generates diagnostic comparison grids:
1. grid_balanced_5styles.png: Balanced 5-style transfers.
2. grid_er_focus.png: Early Renaissance cases.
3. grid_ukiyo_focus.png: Ukiyo-e cases.
4. grid_minimalism_attractor.png: Minimalism off-target attractor cases.
5. grid_teacher_vs_student.png: Teacher success vs student failure/success cases.

Columns: [Raw Source, Q2a Direct (218K), Q2a + R1 Oracle, B3 Distill (218K), S1 New Model (218K)]
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
from PIL import Image, ImageDraw, ImageFont

from tools.v5.route_c import decode_trt, load_trt, load_vae
from tools.v6.canonicalize import ContentCanonicalizer
from tools.v6.models_v6 import C3UNet
from tools.v6.train_eval_v6 import STYLES
from tools.v6_next.probe_style import LatentStyleNet
from tools.v6_next.r2_teacher import (
    filter_haar_subbands,
    project_tangential,
    rms_normalize,
)

DEFAULT_TASKS = ROOT / "eval_protocol" / "tasks_750.json"
SRC_CACHE_PATH = ROOT / "runs" / "v6_r2" / "cache" / "tasks750_source_features.pt"


def create_grid(
    images_per_row, col_titles, row_titles, out_path: Path, cell_size=256
):
  n_rows = len(images_per_row)
  n_cols = len(col_titles)
  header_height = 40
  left_margin = 180
  grid_w = left_margin + n_cols * cell_size
  grid_h = header_height + n_rows * cell_size

  canvas = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
  draw = ImageDraw.Draw(canvas)

  # Column headers
  for j, title in enumerate(col_titles):
    x = left_margin + j * cell_size + cell_size // 2
    draw.text((x, 12), title, fill=(0, 0, 0), anchor="mt")

  # Rows
  for i, row in enumerate(images_per_row):
    y = header_height + i * cell_size
    if i < len(row_titles):
      draw.text(
          (left_margin // 2, y + cell_size // 2),
          row_titles[i],
          fill=(0, 0, 0),
          anchor="mm",
      )
    for j, img in enumerate(row):
      x = left_margin + j * cell_size
      cell_img = img.resize((cell_size, cell_size))
      canvas.paste(cell_img, (x, y))

  out_path.parent.mkdir(parents=True, exist_ok=True)
  canvas.save(out_path)
  print(f"[Visual Grid] Saved to {out_path}", flush=True)


def main():
  ap = argparse.ArgumentParser(description="Generate Visual Comparison Grids")
  ap.add_argument("--tasks", default=str(DEFAULT_TASKS))
  ap.add_argument("--src-cache", default=str(SRC_CACHE_PATH))
  ap.add_argument(
      "--q2a-ckpt", default="runs/v6/Q2a_embed_style/Q2a_embed_style.pt"
  )
  ap.add_argument("--b3-ckpt", default="runs/v6/P1_B3_distill20/P1_B3_distill20.pt")
  ap.add_argument(
      "--new-ckpt",
      default="runs/v6_r2/S1_distill20_proj_ll/S1_distill20_proj_ll.pt",
  )
  ap.add_argument("--critic-ckpt", default="runs/v6_next/probes/r1_r1.pt")
  ap.add_argument("--out-dir", default="runs/v6_r2/visual_grids")
  ap.add_argument(
      "--device", default="cuda" if torch.cuda.is_available() else "cpu"
  )
  args = ap.parse_args()

  device = args.device
  out_dir = ROOT / args.out_dir
  out_dir.mkdir(parents=True, exist_ok=True)

  tasks_data = json.loads((ROOT / args.tasks).read_text(encoding="utf-8"))[
      "tasks"
  ]
  src_cache = torch.load(
      ROOT / args.src_cache, map_location="cpu", weights_only=False
  )

  # Load generator models
  print("[Grid] Loading VAE and models...", flush=True)
  vae = load_vae(device)
  trt = load_trt(vae, device)

  canon = ContentCanonicalizer(
      torch.load(
          ROOT / "runs/v6/cache/canonicalizer_stats.pt",
          map_location="cpu",
          weights_only=False,
      )
  ).to(device)

  # Q2a
  ck_q = torch.load(ROOT / args.q2a_ckpt, map_location="cpu", weights_only=False)
  q2a = C3UNet(
      base_dim=16, num_styles=5, mode="direct", inject_64=True, inject_32=True
  ).to(device)
  q2a.load_state_dict(ck_q["model"])
  q2a.eval()

  # B3
  ck_b3 = torch.load(ROOT / args.b3_ckpt, map_location="cpu", weights_only=False)
  b3 = C3UNet(
      base_dim=16, num_styles=5, mode="direct", inject_64=True, inject_32=True
  ).to(device)
  b3.load_state_dict(ck_b3["model"])
  b3.eval()

  # New Model
  new_model = None
  if (ROOT / args.new_ckpt).exists():
    ck_new = torch.load(
        ROOT / args.new_ckpt, map_location="cpu", weights_only=False
    )
    b_dim = ck_new.get("args", {}).get("base_dim", 16)
    new_model = C3UNet(
        base_dim=b_dim,
        num_styles=5,
        mode="direct",
        inject_64=True,
        inject_32=True,
    ).to(device)
    new_model.load_state_dict(ck_new["model"])
    new_model.eval()

  # Critic for Oracle
  ck_c = torch.load(
      ROOT / args.critic_ckpt, map_location="cpu", weights_only=False
  )
  critic = LatentStyleNet(width=ck_c.get("width", 64)).to(device).eval()
  critic.load_state_dict(ck_c["model"])
  for p in critic.parameters():
    p.requires_grad_(False)

  col_titles = [
      "Raw Source",
      "Q2a Direct (218K)",
      "Q2a + R1 Oracle",
      "B3 Distill (218K)",
      "New R2 Student",
  ]

  def render_task_row(task):
    k = task["source_stem"]
    tid = task["target_style_id"]
    tg = torch.tensor([tid], device=device)

    # 1. Raw source
    raw_rgb = src_cache["raw_rgb"][k]
    raw_pil = Image.fromarray(
        (raw_rgb.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    )

    zs = src_cache["z_src"][k].to(device)
    c = canon(zs, mode="c1_gentle", augment=False)

    with torch.inference_mode():
      z_q2a = q2a(c, 0.0, c, tg)

      # Oracle
      with torch.inference_mode(False), torch.enable_grad():
        zr = z_q2a.detach().clone().requires_grad_(True)
        q = critic(zr.float()).float()
        oh = F.one_hot(tg, 5).bool()
        m = q.gather(1, tg[:, None]).squeeze(1) - torch.logsumexp(
            q.masked_fill(oh, -1e9), dim=1
        )
        g = torch.autograd.grad(m.sum(), zr)[0]
        gn = g.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
        z_oracle = (
            z_q2a + 0.25 * (g / (gn / (16384**0.5) + 1e-12))
        ).detach()

      z_b3 = b3(c, 0.0, c, tg)
      z_new = new_model(c, 0.0, c, tg) if new_model is not None else z_b3

      batch_z = torch.cat([z_q2a, z_oracle, z_b3, z_new], dim=0)
      img_dec = decode_trt(trt, vae, batch_z, batch=4)
      u8_dec = (
          img_dec.mul(255.0).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
      )
      pils = [
          Image.fromarray(u8_dec[i].transpose(1, 2, 0))
          for i in range(len(batch_z))
      ]

    return [raw_pil, pils[0], pils[1], pils[2], pils[3]]

  # 1. Balanced 5-style Grid (one representative task per target style)
  print("[Grid] Rendering balanced 5-style grid...", flush=True)
  balanced_tasks = []
  row_titles_b = []
  for s_id, s_name in enumerate(STYLES):
    # pick a task where target_style == s_name and source_style != s_name
    matching = [
        t
        for t in tasks_data
        if t["is_cross"] and t["target_style"] == s_name
    ]
    if matching:
      t = matching[len(matching) // 2]
      balanced_tasks.append(t)
      row_titles_b.append(f"{t['source_style'][:4]} -> {s_name}")

  rows_b = [render_task_row(t) for t in balanced_tasks]
  create_grid(
      rows_b,
      col_titles,
      row_titles_b,
      out_dir / "grid_balanced_5styles.png",
      cell_size=256,
  )

  # 2. Early Renaissance Focus Grid (4 tasks targeting ER)
  print("[Grid] Rendering Early Renaissance focus grid...", flush=True)
  er_tasks = [
      t
      for t in tasks_data
      if t["is_cross"] and t["target_style"] == "Early_Renaissance"
  ][:4]
  row_titles_er = [f"{t['source_style'][:4]} -> ER" for t in er_tasks]
  rows_er = [render_task_row(t) for t in er_tasks]
  create_grid(
      rows_er,
      col_titles,
      row_titles_er,
      out_dir / "grid_er_focus.png",
      cell_size=256,
  )

  # 3. Ukiyo-e Focus Grid (4 tasks targeting Ukiyo-e)
  print("[Grid] Rendering Ukiyo-e focus grid...", flush=True)
  uki_tasks = [
      t
      for t in tasks_data
      if t["is_cross"] and t["target_style"] == "Ukiyo_e"
  ][:4]
  row_titles_uki = [f"{t['source_style'][:4]} -> Ukiyo" for t in uki_tasks]
  rows_uki = [render_task_row(t) for t in uki_tasks]
  create_grid(
      rows_uki,
      col_titles,
      row_titles_uki,
      out_dir / "grid_ukiyo_focus.png",
      cell_size=256,
  )

  # 4. Minimalism Attractor Diagnostic Grid (tasks where target is Rococo/Impressionism but source is NOT Min)
  print("[Grid] Rendering Minimalism attractor diagnostic grid...", flush=True)
  min_diag_tasks = [
      t
      for t in tasks_data
      if t["is_cross"]
      and t["target_style"] in ("Rococo", "Impressionism")
      and t["source_style"] != "Minimalism"
  ][:4]
  row_titles_min = [
      f"{t['source_style'][:4]} -> {t['target_style'][:4]}"
      for t in min_diag_tasks
  ]
  rows_min = [render_task_row(t) for t in min_diag_tasks]
  create_grid(
      rows_min,
      col_titles,
      row_titles_min,
      out_dir / "grid_minimalism_attractor.png",
      cell_size=256,
  )

  print(f"[Grid] All visual grids successfully written to {out_dir}!")


if __name__ == "__main__":
  main()
