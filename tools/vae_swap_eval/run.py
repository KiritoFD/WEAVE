"""Evaluate only ranked WEAVE arms on tasks_750; never write image files.

Reuse completed Oracle/B3/B4 PNGs. ScoreDiff N24/N12 is generated and scored
in memory. All variants share RAW source RGB references and one metric stack.
"""
import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("WEAVE_ASSET_ROOT", str(ROOT))
os.environ.setdefault("WEAVE_EVAL_CACHE_ROOT", str(ROOT / "eval_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image

from metrics import Metrics, compare, pixels, read_rgb, source_paths, summarize, write_results

VARIANTS = ["sdvae", "a0", "s10", "cedit"]
ARMS = ["Oracle1", "B4", "B3", "ScoreDiff_N24", "ScoreDiff_N12"]
OUT = ROOT / "deliver/vae_ranked_eval/results.json"


def score_existing(metrics, tasks, variant, arm, batch):
    folder = ROOT / "deliver/swap_images" / variant / arm
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    by_id = {r["task_id"]: r for r in manifest}
    assert set(by_id) == {t["task_id"] for t in tasks}
    rows = []
    for i in range(0, len(tasks), batch):
        grp = tasks[i:i + batch]
        pils = [read_rgb(folder / by_id[t["task_id"]]["file_name"]) for t in grp]
        rows.extend(metrics.score(grp, pils))
    return rows


def score_diff_variant(metrics, tasks, paths, variant, wanted, batch):
    from tools.v6_next.generate_all_swap_images import build_variant, decode_chunked
    from tools.v6.models_v6 import ConditionalScoreNet
    from tools.v6.score_edit import edit_with_score_difference

    cfg = build_variant(variant, "cuda")
    model = ConditionalScoreNet(base_dim=64, num_styles=5).cuda().eval()
    ck = torch.load(ROOT / "deliver/checkpoints/runs/v6/score_diff/score_diff_net.pt",
                    map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    del ck
    latents = {}
    stems = sorted(paths)
    with torch.inference_mode():
        for i in range(0, len(stems), 4):
            chunk = stems[i:i + 4]
            x = pixels([read_rgb(paths[s]) for s in chunk]).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = cfg["enc"](x)
            z = z.float() * cfg["sf"]
            for j, stem in enumerate(chunk):
                latents[stem] = z[j].cpu()
    try:
        for arm in wanted:
            steps, lr = (24, 0.100) if arm == "ScoreDiff_N24" else (12, 0.160)
            rows = []
            for i in range(0, len(tasks), batch):
                grp = tasks[i:i + batch]
                z = torch.stack([latents[t["source_stem"]] for t in grp]).cuda()
                si = torch.tensor([t["source_style_id"] for t in grp], device="cuda")
                ti = torch.tensor([t["target_style_id"] for t in grp], device="cuda")
                # Match the ranked recipes: batch 8, seed 42, anchor 0.08.
                y = edit_with_score_difference(model, z, si, ti, num_steps=steps,
                                               step_size=lr, struct_anchor_weight=0.08, seed=42)
                rgb = decode_chunked(cfg["dec"], y / cfg["sf"], chunk_size=4)
                arr = (rgb.clamp(-1, 1).add(1).mul(127.5).round().byte()
                       .permute(0, 2, 3, 1).numpy())
                rows.extend(metrics.score(grp, [Image.fromarray(a) for a in arr]))
                if (i // batch + 1) % 20 == 0:
                    print(f"[Progress] {variant}/{arm}: {min(i+batch,750)}/750", flush=True)
            yield arm, rows
    finally:
        del model, cfg, latents
        gc.collect()
        torch.cuda.empty_cache()


def save_report(result):
    for variant, cells in result["results"].items():
        for arm, cell in cells.items():
            baseline = result["results"].get("sdvae", {}).get(arm)
            if baseline and variant != "sdvae":
                cell["delta_vs_sdvae"] = compare(cell["rows"], baseline["rows"])
    write_results(OUT, result)
    lines = ["# Ranked WEAVE: paired VAE replacement evaluation", "",
             "750 frozen tasks per configuration: 600 cross / 150 same. No new image files.",
             "Raw RGB source reference; CSD ViT-L, DINOv2-small, AlexNet LPIPS.",
             "All VAE variants use paired encoder/decoder, with frozen WEAVE weights.",
             "The sdvae baseline is measured in this run (PyTorch BF16); historical TRT numbers are separate.", "",
             "Historical eval_tiers.py feeds [0,1] images directly to VAE.encode and uses round-trip references. "
             "This run uses the VAE's [-1,1] input and raw RGB metric references; compare VAE variants "
             "against this run's sdvae, not against the archived ranking.", "",
             "| Model | VAE | Top-1 | change pp [95% CI] | DINO-C | LPIPS | Same dS |",
             "|---|---|---:|---|---:|---:|---:|"]
    for arm in ARMS:
        for variant in VARIANTS:
            cell = result["results"].get(variant, {}).get(arm)
            if not cell:
                continue
            s = cell["summary"]
            c = s["cross600"]
            change = "baseline"
            if "delta_vs_sdvae" in cell:
                d = cell["delta_vs_sdvae"]["target_top1"]
                change = f"{d['mean']*100:+.2f} [{d['ci95'][0]*100:+.2f}, {d['ci95'][1]*100:+.2f}]"
            lines.append(f"| {arm} | {variant} | {c['target_top1']*100:.2f}% | {change} | "
                         f"{c['dino_c']:.4f} | {c['lpips']:.4f} | {s['same150']['dS']:+.4f} |")
    (OUT.parent / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--variants", default=",".join(VARIANTS))
    args = ap.parse_args()
    arms, variants = args.arms.split(","), args.variants.split(",")
    assert set(arms) <= set(ARMS) and set(variants) <= set(VARIANTS)
    task_path = ROOT / "deliver/testset/protocol/tasks_750.json"
    tasks = json.loads(task_path.read_text(encoding="utf-8"))["tasks"]
    assert len(tasks) == len({t["task_id"] for t in tasks}) == 750
    paths = source_paths(tasks)
    result = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {
        "protocol": "tasks_750_raw_rgb", "tasks_sha256": hashlib.sha256(task_path.read_bytes()).hexdigest(),
        "new_image_files": 0, "results": {}}
    assert result["tasks_sha256"] == hashlib.sha256(task_path.read_bytes()).hexdigest()
    metrics = Metrics(tasks, paths, batch=16)
    start = time.time()
    for variant in variants:
        cells = result["results"].setdefault(variant, {})
        for arm in arms:
            if arm.startswith("ScoreDiff") or arm in cells:
                continue
            rows = score_existing(metrics, tasks, variant, arm, batch=16)
            cells[arm] = {"summary": summarize(rows), "rows": rows, "input": "existing_png"}
            save_report(result)
            print(f"[DONE] {variant}/{arm} {cells[arm]['summary']['cross600']}", flush=True)
    for variant in variants:
        cells = result["results"].setdefault(variant, {})
        wanted = [arm for arm in arms if arm.startswith("ScoreDiff") and arm not in cells]
        if wanted:
            for arm, rows in score_diff_variant(metrics, tasks, paths, variant, wanted, batch=8):
                cells[arm] = {"summary": summarize(rows), "rows": rows, "input": "memory_only"}
                save_report(result)
                print(f"[DONE] {variant}/{arm} {cells[arm]['summary']['cross600']}", flush=True)
    print(f"ALL_DONE seconds={time.time()-start:.1f} new_image_files=0", flush=True)


if __name__ == "__main__":
    main()
