"""Supplement 750-task ranked VAE results with style metrics and B4 causal controls.

No PNG output. Cache only embeddings, scores and four encoder/decoder hybrid cells.
"""
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

torch.set_num_threads(8)
from metrics import STYLES, pixels, read_rgb, source_paths, write_results
from run import ARMS, VARIANTS, score_diff_variant
from style_metrics import StyleMetrics

OUT = ROOT / "deliver/vae_ranked_eval/style_comparison.json"
CACHE = ROOT / "deliver/vae_ranked_eval/feature_cache"


def source_tasks(tasks):
    return sorted([t for t in tasks if t["source_style_id"] == t["target_style_id"]],
                  key=lambda t: t["source_stem"])


def score_pils(metrics, tasks, pils, cell, with_csd=False):
    cached = metrics.read_cell(cell, tasks) if not with_csd else None
    if cached is not None:
        return cached
    metrics.start_cell()
    rows = []
    for i in range(0, len(tasks), 16):
        grp, images = tasks[i:i+16], pils[i:i+16]
        scored = metrics.score(grp, images)
        if with_csd:
            csd = metrics.csd_rows(grp, images)
            for r, c in zip(scored, csd):
                r.update(c)
        rows.extend(scored)
    metrics.save_cell(cell)
    return rows


def join_csd(rows, previous):
    lookup = {r["task_id"]: r for r in previous}
    keys = ["target_top1", "target_margin", "dT", "dS", "dino_c", "lpips"]
    for row in rows:
        row.update({k: lookup[row["task_id"]][k] for k in keys})
    return rows


def existing_cell(metrics, tasks, variant, arm):
    cached = metrics.read_cell(f"{variant}_{arm}", tasks)
    if cached is not None:
        return cached
    folder = ROOT / "deliver/swap_images" / variant / arm
    m = json.loads((folder / "manifest.json").read_text())
    lookup = {r["task_id"]: r["file_name"] for r in m}
    assert set(lookup) == {t["task_id"] for t in tasks}
    metrics.start_cell()
    rows = []
    for i in range(0, 750, 16):
        grp = tasks[i:i+16]
        rows.extend(metrics.score(grp, [read_rgb(folder / lookup[t["task_id"]]) for t in grp]))
    metrics.save_cell(f"{variant}_{arm}")
    return rows


def save(result):
    write_results(OUT, result)


def hybrid_cells(metrics, tasks, paths, result):
    """2x2 interventions for A0/C_edit around SD-VAE, representative B4 only."""
    from tools.v6_next.generate_all_swap_images import build_variant, decode_chunked
    from tools.v6.models_v6 import C3UNet
    from tools.v6.canonicalize import ContentCanonicalizer

    pairs = [("sdvae", "a0"), ("a0", "sdvae"),
             ("sdvae", "cedit"), ("cedit", "sdvae")]
    stats = torch.load(ROOT / "runs/v6/cache/canonicalizer_stats.pt", map_location="cpu", weights_only=False)
    canon = ContentCanonicalizer(stats).cuda().eval()
    gen = C3UNet(base_dim=16, num_styles=5, mode="direct", inject_16=True, style_keys=False).cuda().eval()
    ck = torch.load(ROOT / "deliver/checkpoints/runs/v6/P1_B4_distill50/P1_B4_distill50.pt",
                    map_location="cpu", weights_only=False)
    gen.load_state_dict(ck["model"])
    del ck
    metrics.enable_csd()
    with torch.inference_mode():
        for enc_name, dec_name in pairs:
            key = f"E_{enc_name}__D_{dec_name}"
            if key in result["hybrids"]:
                continue
            enc, dec = build_variant(enc_name, "cuda"), build_variant(dec_name, "cuda")
            stems, latents = sorted(paths), {}
            for i in range(0, 150, 8):
                chunk = stems[i:i+8]
                x = pixels([read_rgb(paths[s]) for s in chunk]).cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    z = enc["enc"](x)
                for j, stem in enumerate(chunk):
                    latents[stem] = (z[j].float() * enc["sf"]).cpu()
            metrics.start_cell()
            rows = []
            for i in range(0, 750, 16):
                grp = tasks[i:i+16]
                z = torch.stack([latents[t["source_stem"]] for t in grp]).cuda()
                ti = torch.tensor([t["target_style_id"] for t in grp], device="cuda")
                c = canon(z, mode="c1_gentle", augment=False)
                out = gen(c, 0.0, c, ti)
                rgb = decode_chunked(dec["dec"], out / dec["sf"], chunk_size=8)
                u8 = rgb.clamp(-1,1).add(1).mul(127.5).round().byte().permute(0,2,3,1).numpy()
                pils = [Image.fromarray(x) for x in u8]
                if enc_name == "sdvae" and dec_name == "a0" and i == 0:
                    # One batch checks that the factorial shares the archived diagonal.
                    diag = decode_chunked(enc["dec"], out / enc["sf"], chunk_size=8)
                    diag = diag.clamp(-1,1).add(1).mul(127.5).round().byte().permute(0,2,3,1).numpy()
                    folder = ROOT / "deliver/swap_images/sdvae/B4"
                    lookup = {r["task_id"]: r["file_name"] for r in
                              json.loads((folder / "manifest.json").read_text())}
                    archived = np.stack([np.array(read_rgb(folder / lookup[t["task_id"]])) for t in grp])
                    error = int(np.abs(diag.astype(int)-archived.astype(int)).max())
                    result["factorial_diagonal_max_u8_error"] = error
                    assert error <= 1, error
                scored = metrics.score(grp, pils)
                for r, cr in zip(scored, metrics.csd_rows(grp, pils)):
                    r.update(cr)
                rows.extend(scored)
            assert len(rows) == 750
            metrics.save_cell(key)
            result["hybrids"][key] = rows
            save(result)
            print(f"[HYBRID DONE] {key}", flush=True)
            del enc, dec, latents, z, c, out, rgb
            gc.collect()
            torch.cuda.empty_cache()


def main():
    start = time.time()
    task_path = ROOT / "deliver/testset/protocol/tasks_750.json"
    tasks = json.loads(task_path.read_text())["tasks"]
    previous = json.loads((OUT.parent / "results.json").read_text())
    assert hashlib.sha256(task_path.read_bytes()).hexdigest() == previous["tasks_sha256"]
    paths = source_paths(tasks)
    result = json.loads(OUT.read_text()) if OUT.exists() else {
        "tasks_sha256": previous["tasks_sha256"], "results": {}, "reconstruction": {},
        "hybrids": {}, "new_image_files": 0}
    metrics = StyleMetrics(CACHE)
    result["metric_definitions"] = metrics.metadata
    src_tasks = source_tasks(tasks)
    source = score_pils(metrics, src_tasks, [read_rgb(paths[t["source_stem"]]) for t in src_tasks], "sources")
    metrics.source_scores = {r["source_stem"]: {n: np.asarray(r[n+"_scores"])
                                                for n in ["dino", "clip"]} for r in source}
    result["sources"] = source
    save(result)
    for variant in VARIANTS:
        cells = result["results"].setdefault(variant, {})
        if variant not in result["reconstruction"]:
            folder = ROOT / "deliver/swap_images" / variant / "recon"
            recon = score_pils(metrics, src_tasks,
                               [read_rgb(folder / (t["source_stem"]+".png")) for t in src_tasks],
                               variant+"_recon")
            result["reconstruction"][variant] = recon
            save(result)
        for arm in ["Oracle1", "B4", "B3"]:
            if arm in cells:
                continue
            rows = existing_cell(metrics, tasks, variant, arm)
            cells[arm] = join_csd(rows, previous["results"][variant][arm]["rows"])
            save(result)
            print(f"[STYLE DONE] {variant}/{arm}", flush=True)
    # Causal controls are cheap decoder passes; complete them before ScoreDiff.
    hybrid_cells(metrics, tasks, paths, result)
    for variant in VARIANTS:
        cells = result["results"][variant]
        wanted = []
        for arm in ["ScoreDiff_N24", "ScoreDiff_N12"]:
            if arm in cells:
                continue
            cached = metrics.read_cell(f"{variant}_{arm}", tasks)
            if cached is not None:
                cells[arm] = join_csd(cached, previous["results"][variant][arm]["rows"])
                save(result)
            else:
                wanted.append(arm)
        if wanted:
            metrics.start_cell()
            for arm, rows in score_diff_variant(metrics, tasks, paths, variant, wanted, batch=8):
                metrics.save_cell(f"{variant}_{arm}")
                cells[arm] = join_csd(rows, previous["results"][variant][arm]["rows"])
                save(result)
                print(f"[STYLE DONE] {variant}/{arm}", flush=True)
    # Real-source CSD calibration is needed only once, without generated images.
    if "source_csd" not in result:
        raw = [read_rgb(paths[t["source_stem"]]) for t in src_tasks]
        result["source_csd"] = metrics.csd_rows(src_tasks, raw)
        save(result)
    print(f"ALL_STYLE_DONE seconds={time.time()-start:.1f} new_image_files=0", flush=True)


if __name__ == "__main__":
    main()
