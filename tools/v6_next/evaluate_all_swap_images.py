"""Pure offline evaluation script for all generated swap images.

Reads generated images from disk (deliver/swap_images/{variant}/{arm}/) and
computes official protocol metrics:
  - Cross600: Top-1, dT, dS, Target Margin, DINO-C, LPIPS, Per-target Top-1, Minimalism misclassification rate
  - Same150: Top-1, dS
Uses local cached models (CSD, local DINOv2 snapshot, AlexNet LPIPS). Zero network calls.
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path("/root/Workspace/WEAVE_VAE")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "12h"))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import csd_model
from tools.pseudo_gt.matching import load_dino_model, encode_dino
import lpips

SWAP_IMAGES_DIR = ROOT / "deliver/swap_images"
TASKS_PATH = ROOT / "deliver/testset/protocol/tasks_750.json"
PROTO_PATH = ROOT / "deliver/testset/refs/csd_proto.pt"
OUT_JSON = ROOT / "deliver/swap_eval_results.json"

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
MIN_IDX = STYLES.index("Minimalism")


def load_img_tensor(path):
    im = Image.open(path).convert("RGB")
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1) * 2.0 - 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", default="sdvae,a0,s10,cedit")
    parser.add_argument("--arms", default="Q2a,Oracle1,B4,B3,B0")
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()

    dev = "cuda"

    # 1. Load protocol tasks
    spec = json.loads(TASKS_PATH.read_text(encoding="utf-8"))
    tasks = spec["tasks"]
    print(f"[Eval] Loaded {len(tasks)} tasks from {TASKS_PATH}")

    # 2. Load CSD prototypes
    proto_d = torch.load(PROTO_PATH, map_location="cpu", weights_only=False)
    proto = np.stack([np.asarray(proto_d[s], dtype=np.float32) for s in STYLES])
    proto = proto / np.linalg.norm(proto, axis=-1, keepdims=True)
    print(f"[Eval] CSD prototypes loaded from {PROTO_PATH}, shape: {proto.shape}")

    # 3. Load evaluation models
    print("[Eval] Loading CSD model...")
    csd, _, _ = csd_model.load_csd(dev)
    print("[Eval] Loading DINOv2 model (offline local snapshot)...")
    dino = load_dino_model(device=dev)
    print("[Eval] Loading LPIPS model...")
    lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    for p in lpips_fn.parameters():
        p.requires_grad = False

    variant_names = [v.strip() for v in args.variants.split(",") if v.strip()]
    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]

    all_results = {}
    summary_rows = []

    print("\n" + "=" * 90)
    print(f"EVALUATING GENERATED IMAGES: {len(variant_names)} variants x {len(arm_names)} arms")
    print("=" * 90)

    for vname in variant_names:
        v_dir = SWAP_IMAGES_DIR / vname
        if not v_dir.exists():
            print(f"[Skip] Variant directory not found: {v_dir}")
            continue

        print(f"\n>>> Evaluating Variant: {vname}")
        recon_dir = v_dir / "recon"
        if not recon_dir.exists():
            print(f"[Error] Missing recon directory for {vname}: {recon_dir}")
            continue

        # Cache source recon features
        recon_files = sorted(list(recon_dir.glob("*.png")))
        print(f"  [{vname}] Caching {len(recon_files)} source reference features...")
        t_ref_start = time.time()

        src_pils = {}
        src_tensors = {}
        for p in recon_files:
            stem = p.stem
            im = Image.open(p).convert("RGB")
            src_pils[stem] = im
            arr = np.asarray(im, dtype=np.float32) / 255.0
            src_tensors[stem] = torch.from_numpy(arr).permute(2, 0, 1) * 2.0 - 1.0

        stems_list = list(src_pils.keys())
        pils_ordered = [src_pils[s] for s in stems_list]

        # Batch CSD
        f_csd_src = csd_model.feats_pils(csd, pils_ordered, batch=args.batch, device=dev)
        f_csd_src = f_csd_src / np.linalg.norm(f_csd_src, axis=-1, keepdims=True)
        csd_src_map = {stems_list[k]: f_csd_src[k] for k in range(len(stems_list))}

        # Batch DINO
        with torch.inference_mode():
            f_dino_src, _ = encode_dino(dino, pils_ordered, batch=args.batch, device=dev)
            f_dino_src = f_dino_src.float().cpu().numpy()
        dino_src_map = {stems_list[k]: f_dino_src[k] for k in range(len(stems_list))}

        print(f"  [{vname}] Source features cached in {time.time() - t_ref_start:.1f}s")

        all_results[vname] = {}

        for arm_name in arm_names:
            arm_dir = v_dir / arm_name
            manifest_path = arm_dir / "manifest.json"
            if not manifest_path.exists():
                print(f"  [Skip] Missing manifest for {vname}/{arm_name}")
                continue

            t_arm_start = time.time()
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            task_records = []
            img_paths = [arm_dir / rec["file_name"] for rec in manifest]

            # Batch process all 750 task images
            n_tasks = len(manifest)
            all_lpips = []
            all_csd = []
            all_dino = []

            for i in range(0, n_tasks, args.batch):
                chunk_manifest = manifest[i:i + args.batch]
                chunk_paths = img_paths[i:i + args.batch]
                chunk_pils = [Image.open(p).convert("RGB") for p in chunk_paths]

                # CSD features
                f_csd = csd_model.feats_pils(csd, chunk_pils, batch=args.batch, device=dev)
                f_csd = f_csd / np.linalg.norm(f_csd, axis=-1, keepdims=True)
                all_csd.append(f_csd)

                # DINO features
                with torch.inference_mode():
                    f_dino, _ = encode_dino(dino, chunk_pils, batch=args.batch, device=dev)
                    all_dino.append(f_dino.float().cpu().numpy())

                # LPIPS
                chunk_tensors = torch.stack([
                    torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1) * 2.0 - 1.0
                    for im in chunk_pils
                ]).to(dev)
                ref_tensors = torch.stack([
                    src_tensors[rec["source_stem"]] for rec in chunk_manifest
                ]).to(dev)

                with torch.inference_mode():
                    lp_vals = lpips_fn(chunk_tensors, ref_tensors).flatten().cpu().numpy()
                all_lpips.append(lp_vals)

                del chunk_tensors, ref_tensors
                torch.cuda.empty_cache()

            all_csd = np.concatenate(all_csd, axis=0)
            all_dino = np.concatenate(all_dino, axis=0)
            all_lpips = np.concatenate(all_lpips, axis=0)

            # Compute per-task metrics
            cross_rows = []
            same_rows = []
            per_target_hits = defaultdict(list)
            min_misclass = []

            for j, rec in enumerate(manifest):
                stem = rec["source_stem"]
                ti = rec["target_style_id"]
                si = STYLES.index(rec["source_style"])
                is_cross = rec["is_cross"]

                f_task = all_csd[j]
                d_task = all_dino[j]
                lp_task = float(all_lpips[j])

                f_ref = csd_src_map[stem]
                d_ref = dino_src_map[stem]

                scores = f_task @ proto.T
                pred_class = int(np.argmax(scores))
                target_top1 = int(pred_class == ti)
                other_scores = np.delete(scores, ti)
                target_margin = float(scores[ti] - np.max(other_scores))

                dT = float(scores[ti] - f_ref[ti])
                dS = float(scores[si] - f_ref[si])
                dino_c = float(d_task @ d_ref)

                task_res = {
                    "task_idx": rec["task_idx"],
                    "task_id": rec["task_id"],
                    "source_stem": stem,
                    "source_style": rec["source_style"],
                    "target_style": rec["target_style"],
                    "is_cross": is_cross,
                    "target_top1": target_top1,
                    "pred_class": pred_class,
                    "pred_style": STYLES[pred_class],
                    "target_margin": target_margin,
                    "dT": dT,
                    "dS": dS,
                    "dino_c": dino_c,
                    "lpips": lp_task,
                }

                if is_cross:
                    cross_rows.append(task_res)
                    per_target_hits[rec["target_style"]].append(target_top1)
                    if rec["target_style"] != "Minimalism":
                        min_misclass.append(int(pred_class == MIN_IDX))
                else:
                    same_rows.append(task_res)

            # Aggregates
            cross_top1 = float(np.mean([r["target_top1"] for r in cross_rows]))
            cross_dT = float(np.mean([r["dT"] for r in cross_rows]))
            cross_dS = float(np.mean([r["dS"] for r in cross_rows]))
            cross_margin = float(np.mean([r["target_margin"] for r in cross_rows]))
            cross_dino = float(np.mean([r["dino_c"] for r in cross_rows]))
            cross_lpips = float(np.mean([r["lpips"] for r in cross_rows]))

            same_top1 = float(np.mean([r["target_top1"] for r in same_rows])) if same_rows else 0.0
            same_dS = float(np.mean([r["dS"] for r in same_rows])) if same_rows else 0.0
            min_mis_rate = float(np.mean(min_misclass)) if min_misclass else 0.0

            target_top1_map = {st: float(np.mean(per_target_hits[st])) for st in STYLES}

            arm_summary = {
                "variant": vname,
                "arm": arm_name,
                "n_cross": len(cross_rows),
                "n_same": len(same_rows),
                "cross_top1": round(cross_top1, 4),
                "cross_dT": round(cross_dT, 5),
                "cross_dS": round(cross_dS, 5),
                "cross_margin": round(cross_margin, 5),
                "cross_dino_c": round(cross_dino, 4),
                "cross_lpips": round(cross_lpips, 4),
                "same_top1": round(same_top1, 4),
                "same_dS": round(same_dS, 5),
                "min_misclass_rate": round(min_mis_rate, 4),
                "per_target_top1": {k: round(v, 4) for k, v in target_top1_map.items()},
            }

            all_results[vname][arm_name] = {
                "summary": arm_summary,
                "tasks": cross_rows + same_rows,
            }
            summary_rows.append(arm_summary)

            print(f"  [{vname}] [{arm_name:<8}] Cross Top-1={cross_top1*100:5.2f}%  dT={cross_dT:+.4f}  "
                  f"margin={cross_margin:+.4f}  DINO-C={cross_dino:.4f}  LPIPS={cross_lpips:.4f}  "
                  f"MinMis={min_mis_rate*100:5.2f}%  in {time.time() - t_arm_start:.1f}s")

    # Save complete JSON
    OUT_JSON.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\n[Saved] Full evaluation results to {OUT_JSON}")

    # Print summary table
    print("\n" + "=" * 105)
    print("FINAL SUMMARY TABLE: VAE Variants x WEAVE Model Arms (Official Protocol, 600 Cross / 150 Same)")
    print("=" * 105)
    header = f"{'Variant':<8} {'Arm':<9} {'Top-1':>7} {'dT':>8} {'dS':>8} {'Margin':>9} {'DINO-C':>8} {'LPIPS':>8} {'MinMis':>8} {'SameT1':>8}"
    print(header)
    print("-" * len(header))
    for s in summary_rows:
        print(f"{s['variant']:<8} {s['arm']:<9} {s['cross_top1']*100:6.1f}% {s['cross_dT']:+8.4f} {s['cross_dS']:+8.4f} "
              f"{s['cross_margin']:+9.4f} {s['cross_dino_c']:8.4f} {s['cross_lpips']:8.4f} {s['min_misclass_rate']*100:7.1f}% {s['same_top1']*100:7.1f}%")

    print("\n" + "=" * 105)
    print("PER-TARGET TOP-1 BREAKDOWN (Cross-600)")
    print("=" * 105)
    t_header = f"{'Variant':<8} {'Arm':<9} " + " ".join([f"{st[:10]:>12}" for st in STYLES])
    print(t_header)
    print("-" * len(t_header))
    for s in summary_rows:
        pt = s["per_target_top1"]
        line = f"{s['variant']:<8} {s['arm']:<9} " + " ".join([f"{pt.get(st, 0.0)*100:11.1f}%" for st in STYLES])
        print(line)


if __name__ == "__main__":
    main()
