"""DINO-S Validity Control (Section P0.3 of WEAVE R2 Plan).

Validates whether DINO-S can reliably distinguish real paintings of target styles:
1. Real raw paintings DINO-S on same-style vs cross-style references.
2. VAE round-trip reconstruction DINO-S and distortion effect.
3. Target style discrimination margin and 5-way classification accuracy.
4. Per-style breakdown across all 5 styles.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
DEFAULT_TASKS = ROOT / "eval_protocol" / "tasks_750.json"
DEFAULT_REFS = ROOT / "eval_protocol" / "cache" / "references_v1_features.pt"
CACHE_PATH = ROOT / "runs" / "v6_r2" / "cache" / "tasks750_source_features.pt"


def main():
  ap = argparse.ArgumentParser(description="DINO-S Validity Control Probe")
  ap.add_argument("--tasks", default=str(DEFAULT_TASKS))
  ap.add_argument("--ref-features", default=str(DEFAULT_REFS))
  ap.add_argument("--src-cache", default=str(CACHE_PATH))
  args = ap.parse_args()

  tasks = json.loads((ROOT / args.tasks).read_text(encoding="utf-8"))["tasks"]
  refs = torch.load(
      ROOT / args.ref_features, map_location="cpu", weights_only=False
  )
  ref_dino = {
      s: F.normalize(refs["styles"][s]["dino_cls"].float(), dim=-1).numpy()
      for s in STYLES
  }
  ref_stems = {
      s: [
          r.get("stem")
          for r in refs.get("references", {}).get(s, [])
          if "stem" in r
      ]
      for s in STYLES
  }

  cache = torch.load(
      ROOT / args.src_cache, map_location="cpu", weights_only=False
  )

  # Collect unique sources
  uniq_sources = {}
  for t in tasks:
    k = t["source_stem"]
    if k not in uniq_sources:
      uniq_sources[k] = {
          "style": t["source_style"],
          "style_id": t["source_style_id"],
      }

  print("=" * 85)
  print(
      "P0.3 DINO-S VALIDITY CONTROL: REAL PAINTINGS VS INDEPENDENT REFERENCES"
  )
  print(
      f"Total Sources: {len(uniq_sources)} (30 per style) | Total References:"
      f" {sum(len(v) for v in ref_dino.values())} (30 per style)"
  )
  print("=" * 85)

  for kind, name in [
      ("raw_dino", "Raw RGB Real Paintings"),
      ("rt_dino", "VAE Round-trip Real Paintings"),
  ]:
    print(f"\n--- {name} ---")
    same_scores = []
    cross_scores = []
    margins = []
    top1_correct = 0

    per_style = {
        s: {"same": [], "cross": [], "margin": [], "top1": 0} for s in STYLES
    }

    for k, info in uniq_sources.items():
      s_name = info["style"]
      s_id = info["style_id"]
      d_feat = cache[kind][k]  # (384,)

      # Compute similarity to all 5 styles
      style_max_sims = []
      for tid, t_name in enumerate(STYLES):
        sims = d_feat @ ref_dino[t_name].T
        if t_name == s_name and k in ref_stems.get(s_name, []):
          match_idx = ref_stems[s_name].index(k)
          sims[match_idx] = -1e9
        style_max_sims.append(float(sims.max()))

      same_s = style_max_sims[s_id]
      other_max = max(style_max_sims[i] for i in range(5) if i != s_id)
      margin = same_s - other_max
      is_top1 = int(np.argmax(style_max_sims) == s_id)

      same_scores.append(same_s)
      other_sims = [style_max_sims[i] for i in range(5) if i != s_id]
      cross_scores.extend(other_sims)
      margins.append(margin)
      top1_correct += is_top1

      per_style[s_name]["same"].append(same_s)
      per_style[s_name]["cross"].extend(other_sims)
      per_style[s_name]["margin"].append(margin)
      per_style[s_name]["top1"] += is_top1

    mean_same = np.mean(same_scores)
    mean_cross = np.mean(cross_scores)
    mean_margin = np.mean(margins)
    acc = top1_correct / len(uniq_sources)

    print(
        f"Mean Same-Style DINO-S  : {mean_same:.4f} (Independent ground truth"
        " upper bound)"
    )
    print(
        f"Mean Cross-Style DINO-S : {mean_cross:.4f} (Raw source cross-style zero"
        " point)"
    )
    print(
        f"Style Distinction Margin: {mean_margin:+.4f} (Same minus Cross"
        " strongest)"
    )
    print(
        f"5-Way Style Top-1 Acc   : {acc*100:.1f}% ({top1_correct}/{len(uniq_sources)})"
    )

    print("\nPer-Style Breakdown:")
    print(
        f"  {'Style':20s} | {'Same DINO-S':>11s} | {'Cross DINO-S':>12s} |"
        f" {'Margin':>10s} | {'Top-1':>7s}"
    )
    print("  " + "-" * 70)
    for s in STYLES:
      ps = per_style[s]
      s_same = np.mean(ps["same"])
      s_cross = np.mean(ps["cross"])
      s_margin = np.mean(ps["margin"])
      s_acc = ps["top1"] / len(ps["same"]) * 100
      print(
          f"  {s:20s} | {s_same:11.4f} | {s_cross:12.4f} | {s_margin:+10.4f} |"
          f" {s_acc:6.1f}%"
      )


if __name__ == "__main__":
  main()
