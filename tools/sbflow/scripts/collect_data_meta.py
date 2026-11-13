#!/usr/bin/env python3
"""Collect exact provenance fields for data.md."""
import glob, json, os
from pathlib import Path
import torch

REPO = Path("/mnt/i/Latent_Style/WEAVE")
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]

print("=== D5 pseudo-GT cache metadata ===")
for c in sorted(glob.glob("/mnt/i/Latent_Style/**/pseudo_gt_24h/orig_mean_fp16/*.pt", recursive=True)):
    d = torch.load(c, map_location="cpu", weights_only=False)
    print(f"{os.path.basename(c):32s} schema={d.get('schema')} vae={d.get('vae')} "
          f"sf={d.get('scaling_factor')} count={d.get('count')} subdir={d.get('subdir')} "
          f"latents={tuple(d['latents'].shape)} {d['latents'].dtype}")

print("\n=== DCAE cache metadata (V9) ===")
for s in ("Impressionism", "Ukiyo_e"):
    p = REPO / "runs/blog_sbflow/cache" / f"{s}.pt"
    if p.exists():
        d = torch.load(p, map_location="cpu", weights_only=False)
        print(f"{s:16s} ae={d['ae']} sf={d['scaling_factor']} image_size={d['image_size']} "
              f"latent={tuple(d['latent'].shape)} {d['latent'].dtype} "
              f"thumb={tuple(d['thumb'].shape)} {d['thumb'].dtype} n_ids={len(d['ids'])}")

print("\n=== eval protocol ===")
raw = json.loads((REPO / "eval_protocol/tasks_750.json").read_text())
t = raw["tasks"]
cross = [x for x in t if x.get("is_cross", x["source_style"] != x["target_style"])]
same = [x for x in t if not x.get("is_cross", x["source_style"] != x["target_style"])]
print(f"tasks_750: total={len(t)} cross={len(cross)} same={len(same)}")
print(f"keys of a task: {sorted(t[0].keys())}")
refs = json.loads((REPO / "eval_protocol/references_v1.json").read_text())
print(f"references_v1 keys={list(refs.keys())[:6]}")
if "styles" in refs:
    for s, v in refs["styles"].items():
        n = len(v.get("files", v.get("paths", []))) if isinstance(v, dict) else len(v)
        print(f"  ref {s}: {n}")

print("\n=== directory sizes ===")
for d in ("datasets/wikiart_new_512", "datasets/distinct5_512", "data/test"):
    p = REPO / d
    if p.exists():
        n = sum(1 for _ in p.rglob("*") if _.suffix.lower() in (".jpg", ".png"))
        print(f"  {d}: {n} images")
