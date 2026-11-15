"""Metric calibration: NCM top1 of REAL paintings (decoded z_src) on the same
DINO reference-bank readout used by eval_arms.  If real paintings also land at
~0.2, style_top1 is a weak metric; if they land high, chance-level top1 in the
arms means the generations are genuinely style-agnostic.

Run: python -m exp.v15.eval_source_baseline
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]


def main():
    from src.utils import decode_s10, load_s10_decoder
    from src.utils._local_eval_refs import encode_dino, load_dino_model
    from PIL import Image

    tasks = json.loads((REPO / "eval_protocol/tasks_750.json")
                       .read_text(encoding="utf-8"))["tasks"]
    cache = torch.load(REPO / "runs/v9_local_4070/cache/n2c_z.pt",
                       map_location="cpu", weights_only=False)
    z_src = cache["z_src"]
    refs = json.loads((REPO / "eval_protocol/references_v1.json")
                      .read_text(encoding="utf-8"))["references"]
    bank = torch.load(REPO / "eval_protocol/cache/references_v1_features.pt",
                      map_location="cpu", weights_only=False)
    rd = [torch.nn.functional.normalize(
        bank["styles"][s]["dino_cls"].float(), dim=-1).cuda() for s in STYLES]
    ref_stems = {s: [r["stem"] for r in refs[s]] for s in STYLES}
    dino_model = load_dino_model(device="cuda")
    s10 = load_s10_decoder("cuda")

    tops, selfs = [], []
    with torch.inference_mode():
        for bi in range(0, len(tasks), 64):
            idx = list(range(bi, min(bi + 64, len(tasks))))
            group = [tasks[i] for i in idx]
            rgb = decode_s10(s10, z_src[idx].cuda(), batch=8)
            u8 = rgb.mul(255).round().clamp_(0, 255).byte()
            imgs = [Image.fromarray(x) for x in u8.permute(0, 2, 3, 1).cpu().numpy()]
            df, _ = encode_dino(dino_model, imgs, batch=16, device="cuda")
            df = torch.nn.functional.normalize(df.float(), dim=-1).cpu()
            gen = df.numpy()
            sc = np.stack([(torch.as_tensor(gen).cuda() @ p.T).cpu().numpy()
                           for p in rd], axis=1)
            for i, t in enumerate(group):
                ss = int(t["source_style_id"])
                stems = ref_stems[t["source_style"]]
                sims = sc[i].copy()
                for si in range(5):
                    for r, stem in enumerate(ref_stems[STYLES[si]]):
                        if stem == t["source_stem"]:
                            sims[si, r] = -1e9
                tops.append(int(np.argmax(sims.max(axis=1)) == ss))
                selfs.append(float(sims[ss].max()))
    print(f"REAL_PAINTING_NCM_TOP1 = {np.mean(tops):.4f}  (n={len(tops)})")
    print(f"REAL_PAINTING_SELF_SIM = {np.mean(selfs):.4f}")
    print(f"chance = 0.2000")


if __name__ == "__main__":
    main()
