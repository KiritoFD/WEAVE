"""WEAVE evaluation: raw latents/images in, metrics and PNGs out.

Completely separate from src/infer. Consumes whatever src/infer stored, so
metrics can be recomputed without touching the GPU generation path. Runs on
CPU or a free GPU; the encoder/feature models are the only GPU need.
"""
from __future__ import annotations

from pathlib import Path
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]


def _unused_feature_models(device: str):
    """Retained only so stale imports fail loudly rather than silently."""
    raise NotImplementedError("feature loading moved to src.utils.FeatureStack")


def measure(raw_dir: Path, out: Path, *, device: str = "cpu", write_png: bool = True,
            limit: int = 0) -> list[dict]:
    from .metrics import enrich_rows, summarize_rows
    from src.utils import FeatureStack, reference_bank

    raw_dir, out = Path(raw_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if write_png:
        (out / "images").mkdir(exist_ok=True)

    meta = json.loads((raw_dir / "meta.json").read_text(encoding="utf-8"))
    tasks = json.loads((REPO / "eval_protocol/tasks_750.json").read_text(encoding="utf-8"))["tasks"]
    by_id = {t["task_id"]: t for t in tasks}
    src = torch.load(REPO / "runs/v6_r2/cache/tasks750_source_features.pt",
                     map_location="cpu", weights_only=False)
    ref_manifest = json.loads((REPO / "eval_protocol/references_v1.json").read_text(encoding="utf-8"))["references"]

    stack = FeatureStack(device)
    bank = reference_bank(device)
    proto = stack.csd_prototypes
    rd, rc = bank["dino"], bank["clip"]

    from PIL import Image
    rows, feats = [], {"dino": [], "csd": [], "clip": []}
    t0 = time.time()
    for entry in meta["batches"]:
        if limit and len(rows) >= limit:
            break
        payload = torch.load(raw_dir / entry["file"], map_location="cpu", weights_only=False)
        group = [by_id[i] for i in payload["ids"]]
        if "rgb_u8" not in payload:
            raise ValueError("raw batch has no rgb_u8; re-run inference with decoding enabled")
        u8 = payload["rgb_u8"]
        imgs = [Image.fromarray(a) for a in u8.permute(0, 2, 3, 1).numpy()]
        if write_png:
            for t, im in zip(group, imgs):
                im.save(out / "images" / (t["task_id"].replace("/", "__") + ".png"))

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
        raw_rgb = torch.stack([src["raw_rgb"][t["source_stem"]] for t in group]).to(device)
        lr = stack.lpips_vs(u8, raw_rgb).cpu().tolist()

        for i, t in enumerate(group):
            ss, ts = t["source_style_id"], t["target_style_id"]
            rows.append({
                "task_id": t["task_id"], "src_stem": t["source_stem"],
                "src_style": t["source_style"], "tgt_style": t["target_style"],
                "src_style_id": int(ss), "tgt_style_id": int(ts), "is_cross": ss != ts,
                "style_scores_5": scores[i].tolist(), "clip_scores_5": clips[i].tolist(),
                "csd_scores_5": cs[i].tolist(), "dino_s": float(scores[i, ts]),
                "dino_c": float(df[i].cpu().numpy() @ src["raw_dino"][t["source_stem"]]),
                "lpips": float(lr[i]), "target_top1": int(cs[i].argmax() == ts),
                "target_margin": float(cs[i, ts] - np.delete(cs[i], ts).max()),
                "dT": float(cs[i, ts] - src["raw_csd"][t["source_stem"]] @ proto[ts]),
            })
        feats["dino"].append(df.cpu()); feats["csd"].append(torch.tensor(cf))
        feats["clip"].append(clf.cpu())
        print(f"  measure {len(rows)} {time.time() - t0:.1f}s", flush=True)

    rows = enrich_rows(rows, styles=STYLES)
    full = len(rows) == 750 and not limit
    summary = summarize_rows(rows, styles=STYLES, require_full=full)
    (out / "eval.json").write_text(json.dumps({
        "n_total": len(rows), "n_cross": sum(r["is_cross"] for r in rows),
        "n_same": sum(not r["is_cross"] for r in rows), "rows": rows,
        "summary": summary}, ensure_ascii=False), encoding="utf-8")
    torch.save({"task_ids": [r["task_id"] for r in rows],
                "dino": torch.cat(feats["dino"]), "csd": torch.cat(feats["csd"]),
                "clip": torch.cat(feats["clip"])}, out / "features.pt")
    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Measure metrics from stored raw outputs")
    ap.add_argument("--raw", required=True, help="directory written by src.infer.run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-png", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    rows = measure(Path(a.raw), Path(a.out), device=a.device,
                   write_png=not a.no_png, limit=a.limit)
    print(f"EVAL_DONE {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
