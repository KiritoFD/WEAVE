"""Process-isolated referee evaluator for the independent V18 holdout-300 split.

Splits the 300 unseen holdout paintings (60 per style) into:
  - 150 disjoint holdout sources (30 per style -> 600 Cross + 150 Same = 750 tasks)
  - 150 disjoint holdout style references (30 per style -> DINO-S / CLIP-S reference bank)
Zero overlap with train_4400, tasks_750, or references_v1.json.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .metrics import enrich_rows, summarize_rows
from .runner import summarize_measured_arm
from src.utils.protocol import STYLES
from src.utils.raw_io import save_png_batch_fast
from src.utils.runtime import REPO, run_isolated_python, save_json

HOLDOUT_SPEC_PATH = REPO / "runs/v34_frontier/holdout/holdout_750_spec.pt"


@torch.no_grad()
def build_or_load_holdout_spec(*, device: str = "cuda", cache_path: Path = HOLDOUT_SPEC_PATH) -> dict[str, Any]:
    """Build or load the disjoint 150-source / 150-ref holdout-750 protocol + referee features."""
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    from src.utils import FeatureStack, decode_s10, load_s10_decoder

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    v18 = torch.load(REPO / "runs/v18/cache/ccf_pool_cache.pt", map_location="cpu", weights_only=False)
    ho = v18["holdout"]
    sid = ho["sid"].long()

    src_indices_list: list[int] = []
    ref_indices_list: list[int] = []
    for s in range(5):
        s_idx = (sid == s).nonzero(as_tuple=True)[0].tolist()
        if len(s_idx) < 60:
            raise ValueError(f"Expected >= 60 holdout items for style {STYLES[s]}, got {len(s_idx)}")
        src_indices_list.extend(s_idx[:30])
        ref_indices_list.extend(s_idx[30:60])

    src_idx = torch.tensor(src_indices_list, dtype=torch.long)
    ref_idx = torch.tensor(ref_indices_list, dtype=torch.long)

    dec = load_s10_decoder(device).to(torch.bfloat16).eval().requires_grad_(False)
    stack = FeatureStack(device)
    proto = stack.csd_prototypes

    def _decode_and_extract(indices: torch.Tensor) -> dict[str, Any]:
        u8_all, df_all, cf_all, clf_all = [], [], [], []
        for st in range(0, len(indices), 16):
            sl = indices[st:st + 16]
            rgb = decode_s10(dec, ho["z"][sl].to(device).float(), dtype=torch.bfloat16, batch=len(sl), output_cpu=False).clamp(0.0, 1.0)
            u8 = (rgb * 255.0).round().to(torch.uint8).cpu()
            imgs = [Image.fromarray(a) for a in u8.permute(0, 2, 3, 1).numpy()]
            f_out = stack.extract(imgs, batch=8)
            u8_all.append(u8)
            df_all.append(f_out["dino"].cpu())
            cf_all.append(torch.from_numpy(f_out["csd"]).cpu())
            clf_all.append(f_out["clip"].cpu())
        return {
            "u8": torch.cat(u8_all, 0),
            "dino": torch.cat(df_all, 0),
            "csd": torch.cat(cf_all, 0),
            "clip": torch.cat(clf_all, 0),
        }

    src_feats = _decode_and_extract(src_idx)
    ref_feats = _decode_and_extract(ref_idx)

    # Organize ref features per style: 5 x (30, D)
    ref_dino = [ref_feats["dino"][s * 30:(s + 1) * 30].clone() for s in range(5)]
    ref_clip = torch.stack([ref_feats["clip"][s * 30:(s + 1) * 30].mean(dim=0) for s in range(5)], dim=0)
    ref_clip = ref_clip / ref_clip.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    tasks = []
    for k, orig_i in enumerate(src_idx.tolist()):
        ss = int(sid[orig_i].item())
        stem = f"holdout_{k:03d}_{STYLES[ss][:3]}_{ho['ids'][orig_i]}"
        for ts in range(5):
            tid_int = k * 5 + ts
            tasks.append({
                "task_id": f"ho_{tid_int:04d}",
                "source_pos": k,
                "orig_holdout_idx": int(orig_i),
                "source_stem": stem,
                "source_style": STYLES[ss],
                "target_style": STYLES[ts],
                "source_style_id": ss,
                "target_style_id": ts,
            })

    del dec, stack
    gc.collect()
    torch.cuda.empty_cache()

    payload = {
        "tasks": tasks,
        "src_idx": src_idx,
        "ref_idx": ref_idx,
        "src_u8": src_feats["u8"],
        "src_dino": src_feats["dino"],
        "src_csd": src_feats["csd"],
        "ref_dino": ref_dino,
        "ref_clip": ref_clip,
        "csd_proto": torch.as_tensor(np.asarray(proto)).cpu(),
    }
    torch.save(payload, cache_path)
    return payload


def measure_holdout(
    raw_dir: Path,
    out: Path,
    *,
    spec_path: Path = HOLDOUT_SPEC_PATH,
    device: str = "cuda",
    write_png: bool = False,
    micro_batch: int = 8,
) -> list[dict[str, Any]]:
    """Run frozen FeatureStack referee on holdout raw batches without touching train/test750."""
    from src.utils import FeatureStack

    raw_dir, out = Path(raw_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if write_png:
        (out / "images").mkdir(exist_ok=True)

    spec = build_or_load_holdout_spec(device=device, cache_path=spec_path)
    tasks = spec["tasks"]
    by_id = {t["task_id"]: t for t in tasks}
    meta = json.loads((raw_dir / "meta.json").read_text(encoding="utf-8"))

    stack = FeatureStack(device)
    proto = stack.csd_prototypes
    rd = [r.to(device) for r in spec["ref_dino"]]
    rc = spec["ref_clip"].to(device)
    src_u8 = spec["src_u8"]
    src_dino = spec["src_dino"].numpy()
    src_csd = spec["src_csd"].numpy()
    proto_np = np.asarray(proto)

    rows = []
    t0 = time.time()
    for entry in meta["batches"]:
        payload = torch.load(raw_dir / entry["file"], map_location="cpu", weights_only=False)
        group = [by_id[i] for i in payload["ids"]]
        u8 = payload["rgb_u8"]
        imgs = [Image.fromarray(a) for a in u8.permute(0, 2, 3, 1).numpy()]
        if write_png:
            save_png_batch_fast(u8, [t["task_id"] for t in group], out / "images")

        feats_out = stack.extract(imgs, batch=micro_batch)
        df, cf, clf = feats_out["dino"], feats_out["csd"], feats_out["clip"]
        scores = torch.stack([(df @ p.T).max(1).values for p in rd], 1).cpu().numpy()
        cs = cf @ proto_np.T
        clips = (clf @ rc.T).cpu().numpy()
        raw_rgb = torch.stack([src_u8[t["source_pos"]] for t in group])
        lr = stack.lpips_vs(u8, raw_rgb, batch=micro_batch).cpu().tolist()

        for i, t in enumerate(group):
            ss, ts = int(t["source_style_id"]), int(t["target_style_id"])
            spos = int(t["source_pos"])
            p_cls = int(cs[i].argmax())
            rows.append({
                "task_id": t["task_id"],
                "src_stem": t["source_stem"],
                "src_style": t["source_style"],
                "tgt_style": t["target_style"],
                "src_style_id": ss,
                "tgt_style_id": ts,
                "is_cross": ss != ts,
                "pred_class": p_cls,
                "style_scores_5": scores[i].tolist(),
                "clip_scores_5": clips[i].tolist(),
                "csd_scores_5": cs[i].tolist(),
                "dino_s": float(scores[i, ts]),
                "dino_c": float(df[i].cpu().numpy() @ src_dino[spos]),
                "lpips": float(lr[i]),
                "target_top1": int(p_cls == ts),
                "target_margin": float(cs[i, ts] - np.delete(cs[i], ts).max()),
                "dT": float(cs[i, ts] - src_csd[spos] @ proto_np[ts]),
            })
        print(f"  holdout measure {len(rows)} {time.time() - t0:.1f}s", flush=True)

    rows = enrich_rows(rows, styles=STYLES, require_full=True)
    summary = summarize_rows(rows, styles=STYLES, require_full=True)
    (out / "eval.json").write_text(json.dumps({
        "n_total": len(rows),
        "n_cross": sum(r["is_cross"] for r in rows),
        "n_same": sum(not r["is_cross"] for r in rows),
        "rows": rows,
        "summary": summary,
    }, ensure_ascii=False), encoding="utf-8")
    return rows


def run_holdout_measure_subprocess(
    raw_dir: Path | str,
    meas_dir: Path | str,
    *,
    arm_name: str | None = None,
    device: str = "cuda",
    micro_batch: int = 8,
    force: bool = False,
) -> dict[str, Any]:
    raw_p, meas_p = Path(raw_dir), Path(meas_dir)
    name = arm_name or raw_p.name
    if force or not (meas_p / "eval.json").exists():
        rc = run_isolated_python(
            "src.eval.holdout",
            [
                "--raw", str(raw_p),
                "--out", str(meas_p),
                "--device", device,
                "--micro-batch", str(micro_batch),
            ],
        )
        if rc != 0:
            raise RuntimeError(f"src.eval.holdout failed for {name} (rc={rc})")
    summary = summarize_measured_arm(name, meas_p)
    save_json(meas_p / "summary_holdout.json", summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure independent V18 holdout-750 outputs")
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--write-png", action="store_true")
    args = ap.parse_args()
    rows = measure_holdout(Path(args.raw), Path(args.out), device=args.device, write_png=args.write_png, micro_batch=args.micro_batch)
    print(f"HOLDOUT_EVAL_DONE {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
