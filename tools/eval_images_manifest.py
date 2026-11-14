"""Evaluate external-baseline PNG outputs on the WEAVE 750-task protocol.

Drop-in image-dir counterpart of src.eval.measure.measure(): identical
features (CSD/DINOv2-small/CLIP-B32), identical reference bank, identical
source-feature cache, identical enrich_rows/summarize_rows. The only
differences are the input side:

  * images come from an output directory of PNGs named
    ``{index:04d}_{sha256(task_id)[:10]}.png`` (the naming used by
    batch_stylegallery_12g.py / batch_stylegallery_fast.py / batch_h2s_750.py),
    resolved through the same manifest CSV those runners consumed;
  * every image is resized to 512x512 (eval_protocol IMAGE_SIZE) before
    feature extraction, so baselines that emit native-resolution output
    (e.g. Hist2Style) are compared on equal footing;
  * LPIPS is computed against the same cached raw_rgb source tensors,
    exactly as measure() does.

Usage (on the WEAVE repo, GPU optional):
  python3 tools/eval_images_manifest.py \
      --manifest sg750_manifest.csv --images h2s750_outputs \
      --out runs/baselines/hist2style/eval --device cuda
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.measure import REPO, STYLES
from src.eval.metrics import enrich_rows, summarize_rows
from src.utils import FeatureStack, reference_bank

IMAGE_SIZE = 512
CHUNK = 32


def safe_id(index: int, task_id: str) -> str:
    return f"{index:04d}_" + hashlib.sha256(task_id.encode()).hexdigest()[:10]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--images", required=True, type=Path,
                    help="dir containing {index:04d}_{hash}.png outputs")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--missing", default="error",
                    choices=["error", "skip"],
                    help="what to do when a task's png is absent")
    args = ap.parse_args()

    out = args.out.resolve()
    (out / "images").mkdir(parents=True, exist_ok=True)

    with args.manifest.open(encoding="utf-8", newline="") as fp:
        rows_in = list(csv.DictReader(fp))
    if args.limit:
        rows_in = rows_in[:args.limit]

    tasks = json.loads((REPO / "eval_protocol/tasks_750.json")
                       .read_text(encoding="utf-8"))["tasks"]
    by_id = {t["task_id"]: t for t in tasks}
    src = torch.load(REPO / "runs/v6_r2/cache/tasks750_source_features.pt",
                     map_location="cpu", weights_only=False)
    ref_manifest = json.loads(
        (REPO / "eval_protocol/references_v1.json")
        .read_text(encoding="utf-8"))["references"]

    device = args.device if torch.cuda.is_available() or args.device == "cpu" \
        else "cpu"
    stack = FeatureStack(device)
    bank = reference_bank(device)
    proto = stack.csd_prototypes
    rd, rc = bank["dino"], bank["clip"]

    # locate images first so missing files are reported up front
    pending = []
    missing = []
    for i, row in enumerate(rows_in):
        t = by_id[row["task_id"]]
        png = args.images / f"{safe_id(i, row['task_id'])}.png"
        if png.is_file():
            pending.append((i, row, t, png))
        else:
            missing.append((i, row["task_id"], str(png)))
    if missing:
        msg = f"{len(missing)} images missing; first: {missing[0]}"
        if args.missing == "error":
            raise FileNotFoundError(msg)
        print(f"[warn] {msg} (skipping)")

    # incremental resume: rows/features are appended per chunk so a crash
    # never loses finished work. eval_rows.jsonl + features_partial.pt in
    # `out` are the resume state; eval.json/features.pt are written once
    # at the end.
    rows_path = out / "eval_rows.jsonl"
    feats_path = out / "features_partial.pt"
    rows: list = []
    feats = {"dino": [], "csd": [], "clip": []}
    done_ids: set = set()
    if rows_path.is_file():
        with rows_path.open(encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    rows.append(r)
                    done_ids.add(r["task_id"])
        if feats_path.is_file():
            prev = torch.load(feats_path, map_location="cpu",
                              weights_only=False)
            feats = prev
        print(f"[resume] {len(rows)} rows / {len(done_ids)} tasks loaded")

    pending = [p for p in pending
               if p[1]["task_id"] not in done_ids]
    if done_ids:
        print(f"[resume] {len(pending)} tasks remaining")

    t0 = time.time()
    for c0 in range(0, len(pending), args.chunk):
        chunk = pending[c0:c0 + args.chunk]
        imgs = []
        group = []
        u8s = []
        for i, row, t, png in chunk:
            im = Image.open(png).convert("RGB")
            if im.size != (IMAGE_SIZE, IMAGE_SIZE):
                im = im.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
            imgs.append(im)
            group.append(t)
            u8s.append(torch.from_numpy(np.asarray(im)).permute(2, 0, 1))
            # keep protocol-identical image copy for downstream consumers
            im.save(out / "images" / (t["task_id"].replace("/", "__") + ".png"))
        u8 = torch.stack(u8s)  # uint8 NCHW

        feats_out = stack.extract(imgs)
        df, cf, clf = feats_out["dino"], feats_out["csd"], feats_out["clip"]
        scores_by_style = []
        for style, p in zip(STYLES, rd):
            sim = df @ p.T
            stems = [r["stem"] for r in ref_manifest[style]]
            for i, t in enumerate(group):
                if t["source_stem"] in stems:
                    sim[i, stems.index(t["source_stem"])] = -1e9
            scores_by_style.append(sim.max(1).values)
        scores = torch.stack(scores_by_style, 1).cpu().numpy()
        cs = cf @ proto.T
        clips = (clf @ rc.T).cpu().numpy()
        raw_rgb = torch.stack(
            [src["raw_rgb"][t["source_stem"]] for t in group]).to(device)
        lr = stack.lpips_vs(u8, raw_rgb).cpu().tolist()

        for i, t in enumerate(group):
            ss, ts = t["source_style_id"], t["target_style_id"]
            rows.append({
                "task_id": t["task_id"], "src_stem": t["source_stem"],
                "src_style": t["source_style"], "tgt_style": t["target_style"],
                "src_style_id": int(ss), "tgt_style_id": int(ts),
                "is_cross": ss != ts,
                "style_scores_5": scores[i].tolist(),
                "clip_scores_5": clips[i].tolist(),
                "csd_scores_5": cs[i].tolist(),
                "dino_s": float(scores[i, ts]),
                "dino_c": float(df[i].cpu().numpy()
                                @ src["raw_dino"][t["source_stem"]]),
                "lpips": float(lr[i]),
                "target_top1": int(cs[i].argmax() == ts),
                "target_margin": float(
                    cs[i, ts] - np.delete(cs[i], ts).max()),
                "dT": float(cs[i, ts]
                            - src["raw_csd"][t["source_stem"]] @ proto[ts]),
            })
        feats["dino"].append(df.cpu()); feats["csd"].append(torch.tensor(cf))
        feats["clip"].append(clf.cpu())
        # flush this chunk immediately: rows -> jsonl, features -> partial pt
        chunk_rows = rows[len(rows) - len(group):]
        with rows_path.open("a", encoding="utf-8") as fp:
            for r in chunk_rows:
                fp.write(json.dumps(r, ensure_ascii=False) + "\n")
        torch.save(feats, feats_path)
        print(f"  eval {len(rows)} rows done, "
              f"{len(pending) - c0 - len(chunk)} left "
              f"{time.time() - t0:.1f}s", flush=True)

    rows = enrich_rows(rows, styles=STYLES)
    full = len(rows) == 750 and not args.limit
    summary = summarize_rows(rows, styles=STYLES, require_full=full)
    (out / "eval.json").write_text(json.dumps({
        "n_total": len(rows), "n_cross": sum(r["is_cross"] for r in rows),
        "n_same": sum(not r["is_cross"] for r in rows),
        "n_missing": len(missing), "rows": rows, "summary": summary,
        "image_dir": str(args.images.resolve()),
        "manifest": str(args.manifest.resolve()),
        "resize": f"{IMAGE_SIZE}x{IMAGE_SIZE} bilinear",
    }, ensure_ascii=False), encoding="utf-8")
    torch.save({"task_ids": [r["task_id"] for r in rows],
                "dino": torch.cat(feats["dino"]),
                "csd": torch.cat(feats["csd"]),
                "clip": torch.cat(feats["clip"])}, out / "features.pt")
    print(json.dumps(summary, indent=1)[:2000])


if __name__ == "__main__":
    main()
