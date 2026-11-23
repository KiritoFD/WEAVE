"""Paired 150-source cluster bootstrap CI & qualitative grid exporter for exp/v27_mainline -> docs/exp1123/figures/."""
from __future__ import annotations

import argparse
from pathlib import Path
from PIL import Image, ImageDraw
import torch

from exp.v18 import tasks_and_sources
from exp.v27.icme_step1_decoder_closedset import _cluster_paired_ci
from exp.v27_mainline import OUT, ROOT
from src.utils.runtime import read_json, save_json, utc


def compute_mainline_cis(pairs: list[tuple[str, str, str]]) -> dict:
    out = {"updated_utc": utc(), "comparisons": {}}
    for label, arm_a, arm_b in pairs:
        da = OUT / "measured" / arm_a
        db = OUT / "measured" / arm_b
        if (da / "eval.json").exists() and (db / "eval.json").exists():
            out["comparisons"][label] = _cluster_paired_ci(da, db)
    save_json(OUT / "mainline_paired_cis.json", out)
    return out


def export_qual_strip(arms: list[str], out_png: Path) -> None:
    """Export 4-row qualitative comparison strip (ER->Uki, Imp->Min, Uki->Imp, Roc->ER) across arms."""
    tasks, _, _ = tasks_and_sources()
    picks = [
        "Early_Renaissance/Early_Renaissance__benozzo-gozzoli_madonna-and-child-1485/Ukiyo_e",
        "Impressionism/Impressionism__armand-guillaumin_valley-in-pontgibaud-1890/Minimalism",
        "Ukiyo_e/Ukiyo_e__hiroshige_hakone-kosuizu/Impressionism",
        "Rococo/Rococo__francois-boucher_the-toilet-of-venus-1751/Early_Renaissance",
    ]
    tid_to_idx = {t["task_id"]: i for i, t in enumerate(tasks)}
    # Fallback if Rococo stem differs
    pick_indices = []
    for p in picks:
        if p in tid_to_idx:
            pick_indices.append(tid_to_idx[p])
        else:
            roc = next(i for i, t in enumerate(tasks) if t["source_style"] == "Rococo" and t["target_style"] == "Early_Renaissance")
            pick_indices.append(roc)

    # Load raw rgb_u8 per arm for pick_indices
    arm_imgs: dict[str, list[Image.Image]] = {}
    for arm in arms:
        raw_dir = OUT / "raw" / arm
        meta = read_json(raw_dir / "meta.json")
        imgs = {}
        off = 0
        for b in meta["batches"]:
            d = torch.load(raw_dir / b["file"], map_location="cpu", weights_only=False)
            for j, tid_s in enumerate(d["ids"]):
                if tid_to_idx[tid_s] in pick_indices:
                    arr = d["rgb_u8"][j].permute(1, 2, 0).numpy()
                    imgs[tid_to_idx[tid_s]] = Image.fromarray(arr).resize((256, 256), Image.Resampling.LANCZOS)
            off += len(d["ids"])
        arm_imgs[arm] = [imgs[idx] for idx in pick_indices]

    src_imgs = [
        Image.open(ROOT / tasks[idx]["source_rel_path"]).convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
        for idx in pick_indices
    ]

    cols = 1 + len(arms)
    rows = len(pick_indices)
    pad_t, pad_l = 32, 8
    W = pad_l + cols * 258
    H = pad_t + rows * 258
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    headers = ["Source"] + arms
    for c_i, h in enumerate(headers):
        draw.text((pad_l + c_i * 258 + 8, 8), h[:28], fill=(20, 20, 20))
    for r_i in range(rows):
        canvas.paste(src_imgs[r_i], (pad_l, pad_t + r_i * 258))
        for c_i, arm in enumerate(arms):
            canvas.paste(arm_imgs[arm][r_i], (pad_l + (c_i + 1) * 258, pad_t + r_i * 258))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)
    print(f"[Viz] Saved {out_png}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--out_png", default=str(OUT / "qual_strip.png"))
    args = ap.parse_args()
    export_qual_strip(args.arms, Path(args.out_png))
    if len(args.arms) >= 2:
        pairs = [(f"{args.arms[-1]}_minus_{a}", args.arms[-1], a) for a in args.arms[:-1]]
        compute_mainline_cis(pairs)


if __name__ == "__main__":
    main()
