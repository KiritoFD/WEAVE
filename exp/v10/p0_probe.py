"""P0 probes (plan §10): split integrity, S10 decode consistency, feature throughput.

Run: python -m exp.v10.p0_probe
Writes runs/v10_local_4070/p0_report.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from .featfn import holdout_latents, local_image_path
from .splits import STYLES, build_splits, check_integrity

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs/v10_local_4070"
PROBE_N = 8  # per style


def s10_decode_probe(device: str) -> dict:
    """decode(z_real) vs the local original image, for pool paintings whose
    512px file exists locally (~8/style; the full image set lives on the 3060)."""
    from src.utils import decode_s10, load_s10_decoder
    from PIL import Image
    import numpy as np
    from .splits import POOL_DIR

    s10 = load_s10_decoder(device)
    res = {"per_style": {}, "mae_all": [], "lpips_all": []}
    lp = None
    try:
        import lpips
        lp = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    except Exception as e:  # pragma: no cover
        res["lpips_note"] = f"lpips unavailable: {e}"
    for si, s in enumerate(STYLES):
        d = torch.load(POOL_DIR / f"{si:02d}_{s}.pt",
                       map_location="cpu", weights_only=False)
        pairs = []
        for i, f in enumerate(d["files"]):
            lp_ = local_image_path(str(f))
            if lp_.exists():
                pairs.append((i, lp_))
            if len(pairs) >= PROBE_N:
                break
        maes, lp_vals = [], []
        for i, path in pairs:
            z = d["latents"][i:i + 1].float().to(device)
            rgb = decode_s10(s10, z, batch=1)[0]              # (3,512,512) [0,1]
            img = Image.open(path).convert("RGB")
            ref = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 255.0
            if ref.shape[-1] != 512 or ref.shape[-2] != 512:
                ref = torch.nn.functional.interpolate(
                    ref[None], size=(512, 512), mode="bilinear",
                    align_corners=False)[0]
            mae = (rgb.cpu() - ref).abs().mean().item() * 255.0
            maes.append(mae)
            if lp is not None:
                a = rgb[None].to(device) * 2 - 1
                b = ref[None].to(device) * 2 - 1
                lp_vals.append(lp(a, b).item())
        res["per_style"][s] = {"mae_255": round(sum(maes) / len(maes), 3),
                               "lpips": round(sum(lp_vals) / len(lp_vals), 4) if lp_vals else None,
                               "n": len(maes)}
        res["mae_all"].extend(maes)
        res["lpips_all"].extend(lp_vals)
        print(f"[p0-decode] {s}: mae {res['per_style'][s]['mae_255']}/255 "
              f"lpips {res['per_style'][s]['lpips']}", flush=True)
    del s10
    torch.cuda.empty_cache()
    return res


def throughput_probe(device: str) -> dict:
    from .features import f_feats, patchify, q_feats
    lat, _, _ = holdout_latents(1, 256, seed=0)   # Impressionism holdout
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(0, 256, 128):
        z = lat[i:i + 128].float().to(device)
        p = patchify(z)
        _ = q_feats(p); _ = f_feats(p)
    torch.cuda.synchronize()
    dt = time.time() - t0
    return {"latents_per_sec": round(256 / dt, 1), "patches_per_sec": round(256 * 64 / dt, 1)}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "device": device}
    splits = build_splits()
    report["split_integrity"] = check_integrity(splits)
    torch.save(splits, OUT / "splits.pt")
    print("split integrity:", json.dumps(report["split_integrity"]), flush=True)
    report["feature_throughput"] = throughput_probe(device)
    print("throughput:", report["feature_throughput"], flush=True)
    report["s10_decode"] = s10_decode_probe(device)
    (OUT / "p0_report.json").write_text(json.dumps(report, indent=1))
    print("P0 report saved ->", OUT / "p0_report.json")


if __name__ == "__main__":
    main()
