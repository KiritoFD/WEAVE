"""V11-B inference: apply wavelet stroke flow to all 750 LD4 outputs.

x0 = existing S10-decoded WEAVE LD4 PNGs (v9 dev+test eval/images union,
which together cover tasks_750 exactly). For each task we keep LL2 fixed,
regenerate the chosen detail band(s) with the style-conditioned flow,
reconstruct via iDWT, and write {idx:04d}_{sha256(task_id)[:10]}.png —
the same naming eval_images_manifest.py consumes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import unicodedata
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .bands import dwt2, idwt2, sobel_mag
from .sample_flow import edit_bands, load_flow

REPO = Path(__file__).resolve().parents[2]
TASKS = REPO / "eval_protocol" / "tasks_750.json"
LD4_DIRS = [REPO / "runs/v9_local_4070/dev/LD4/eval/images",
            REPO / "runs/v9_local_4070/test/LD4/eval/images"]
SIGMAS = REPO / "docs/v11b/P0_PROFILE.json"


def safe_id(index: int, task_id: str) -> str:
    return f"{index:04d}_" + hashlib.sha256(task_id.encode()).hexdigest()[:10]


def collect_ld4():
    """filename-stem -> png path. Key is the exact PNG stem; callers
    look up by task_id.replace('/', '__')."""
    lut = {}
    for d in LD4_DIRS:
        for p in d.glob("*.png"):
            lut[unicodedata.normalize("NFC", p.stem)] = p
    return lut


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--width", type=int, default=56)
    ap.add_argument("--band", choices=["H1", "H2", "both"], default="H1")
    ap.add_argument("--tau0", type=float, default=0.6)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--edge-decay", type=float, default=0.5)
    ap.add_argument("--rms-clip", type=float, default=1.5)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--shuffle-style", action="store_true",
                    help="P4 ablation: feed a wrong style id "
                         "((sid+2)%5) to test style conditioning")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)
    sigmas = json.loads(SIGMAS.read_text())["sigmas"]
    model = load_flow(args.ckpt, device=device, width=args.width)
    lut = collect_ld4()
    tasks = json.loads(TASKS.read_text(encoding="utf-8"))["tasks"]

    STYLE2ID = {s: i for i, s in enumerate(
        ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo",
         "Ukiyo_e"])}
    log = args.out / "execution.jsonl"
    todo = list(enumerate(tasks))[args.start:]
    if args.limit:
        todo = todo[:args.limit]
    t0 = time.time()
    for n_done, (i, t) in enumerate(todo):
        tid = t["task_id"]
        png = args.out / f"{safe_id(i, tid)}.png"
        if png.exists():
            continue
        src = lut.get(unicodedata.normalize("NFC", tid.replace("/", "__")))
        if src is None:
            print(f"[missing] {i} {tid}", flush=True)
            continue
        im = Image.open(src).convert("RGB")
        x = torch.from_numpy(np.asarray(im, dtype=np.float32)
                             / 255.0).permute(2, 0, 1)[None].to(device)
        d = dwt2(x)
        bb = {"LL2": d["LL2"], "H2": d["H2"], "H1": d["H1"],
              "LL1": d["LL1"],
              "edge1": sobel_mag(d["LL1"].mean(1, keepdim=True)),
              "edge2": sobel_mag(d["LL2"].mean(1, keepdim=True))}
        sid = STYLE2ID[t["target_style"]]
        if args.shuffle_style:
            sid = (sid + 2) % 5
        bands = ["H1", "H2"] if args.band == "both" else [args.band]
        for b in bands:
            bb[b] = edit_bands(model, bb, sid, sigmas, band=b,
                               tau0=args.tau0, K=args.K, cfg_w=args.cfg,
                               alpha=args.alpha,
                               edge_decay=args.edge_decay,
                               rms_clip_mult=args.rms_clip,
                               device=device)
        y = idwt2(bb["LL2"], bb["H2"], bb["H1"]).clamp(0, 1)
        arr = (y[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
        Image.fromarray(arr).save(png)
        el = time.time() - t0
        msg = f"[{i+1}/{len(tasks)}] {tid} {el/(n_done+1):.2f}s/img"
        print(msg.encode("ascii", "replace").decode(), flush=True)
        with log.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps({"index": i, "task_id": tid,
                                 "band": args.band, "tau0": args.tau0,
                                 "K": args.K, "cfg": args.cfg,
                                 "alpha": args.alpha,
                                 "result": str(png)}) + "\n")
    print(f"[done] {args.out} elapsed={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
