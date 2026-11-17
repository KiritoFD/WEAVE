"""RGB-H2 post-training of the V11b wavelet flow on REAL RGB images.

Drop-in for missing ZIP script. Trains WaveletFlow (flow_v1 arch) on
style-classified RGB crops from --image-roots (e.g. datasets/distinct5_512/train,
datasets/wikiart_new_512). Saves final.pt in the same ckpt format
{"model","args","sigmas","step"} so h2_apply_4070lp can load it.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as Fn

from exp.v11b.bands import dwt2, sobel_mag
from exp.v11b.model import WaveletFlow
from exp.v11b.train_flow import fm_step_full

REPO = Path(__file__).resolve().parents[2]
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]


def collect_images(roots, crop, device, max_per_style=None):
    """Load style-folder jpgs -> uint8 tensors (N,3,crop,crop) + style ids."""
    from PIL import Image
    import numpy as np
    xs, ys = [], []
    rng = np.random.default_rng(0)
    for si, s in enumerate(STYLES):
        files = []
        for r in roots:
            d = Path(r) / s
            if d.is_dir():
                files += sorted(d.glob("*.jpg")) + sorted(d.glob("*.png"))
        if max_per_style:
            files = files[:max_per_style]
        for f in files:
            try:
                im = Image.open(f).convert("RGB")
            except Exception:
                continue
            w, h = im.size
            if w < crop or h < crop:
                continue
            x0 = int(rng.integers(0, w - crop + 1))
            y0 = int(rng.integers(0, h - crop + 1))
            arr = np.asarray(im.crop((x0, y0, x0 + crop, y0 + crop)),
                             dtype=np.float32) / 255.0
            xs.append(torch.from_numpy(arr).permute(2, 0, 1))
            ys.append(si)
    X = torch.stack(xs)          # keep on CPU; per-batch H2D transfer
    y = torch.tensor(ys, dtype=torch.long)
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--image-roots", nargs="+", required=True)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--width", type=int, default=56)
    ap.add_argument("--micro", type=int, default=2)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--updates", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--p-drop", type=float, default=0.15)
    ap.add_argument("--max-per-style", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = a.device
    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    sigmas = json.loads((REPO / "docs/v11b/P0_PROFILE.json")
                        .read_text())["sigmas"]
    ck = torch.load(a.source, map_location="cpu", weights_only=False)
    model = WaveletFlow(width=a.width).to(dev)
    model.load_state_dict(ck["model"] if "model" in ck else ck)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr,
                            weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.updates)

    print("loading images...", flush=True)
    X, y = collect_images(a.image_roots, a.crop, dev,
                          a.max_per_style or None)
    print(f"images {X.shape} styles={torch.bincount(y).tolist()}",
          flush=True)

    def make_batch():
        idx = torch.randint(0, len(X), (a.micro,))
        crops = X[idx].to(dev)                            # (B,3,c,c)
        d = dwt2(crops)
        batch = {"H1": d["H1"], "H2": d["H2"], "LL1": d["LL1"],
                 "LL2": d["LL2"],
                 "edge1": sobel_mag(d["LL1"].mean(1, keepdim=True)),
                 "edge2": sobel_mag(d["LL2"].mean(1, keepdim=True)),
                 "style_id": y[idx].to(dev)}
        return batch

    curve = out / "train_curves.jsonl"
    opt_step = micro_in = 0
    run = 0.0
    t0 = time.time()
    peak = 0
    while opt_step < a.updates:
        batch = make_batch()
        loss, _ = fm_step_full(model, batch, sigmas, dev, p_drop=a.p_drop)
        (loss / a.accum).backward()
        run += loss.item()
        micro_in += 1
        if micro_in % a.accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            opt_step += 1
            peak = max(peak, torch.cuda.memory_allocated() // 2**20)
            if opt_step % 100 == 0:
                rec = {"opt_step": opt_step,
                       "loss": round(run / (a.accum * 100), 5),
                       "sec": round(time.time() - t0, 1),
                       "peak_mb": peak}
                with curve.open("a") as fp:
                    fp.write(json.dumps(rec) + "\n")
                print(f"[h2] {opt_step}/{a.updates} loss={rec['loss']} "
                      f"{rec['sec']}s peak={peak}MB", flush=True)
                run = 0.0
    torch.save({"model": model.state_dict(),
                "args": vars(a), "sigmas": sigmas, "step": opt_step},
               out / "final.pt")
    print(f"DONE {opt_step} {time.time()-t0:.0f}s peak={peak}MB", flush=True)


if __name__ == "__main__":
    main()
