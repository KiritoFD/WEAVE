"""V14 Experiment 2: style gradient band decomposition.

Computes DINO style gradient on decoded RGB, decomposes into Haar bands,
and measures per-band energy + executability.
Run on the 3060: python band_energy.py
"""
from __future__ import annotations
import json, math, time
from pathlib import Path
import torch
import torch.nn.functional as Fn
import numpy as np
from PIL import Image

REPO = Path("I:/Latent_Style/WEAVE")
DINO_SNAP = Path("I:/Latent_Style/eval_cache/hf/hub/models--facebook--dinov2-small/snapshots/ed25f3a31f01632728cabb09d1542f84ab7b0056")
S10_PATH = REPO / "runs/vae_decoder_prune/s10_p5_lowlr/decoder.pt"
OUT = REPO / "runs/v14"
N_PER_STYLE = 10

_DINO = None
_S10 = None

def _dino(device):
    global _DINO
    if _DINO is None:
        from transformers import AutoModel
        _DINO = AutoModel.from_pretrained(str(DINO_SNAP)).to(device).eval()
    return _DINO

def _s10(device):
    global _S10
    if _S10 is None:
        import sys
        sys.path.insert(0, str(REPO))
        from tools.vae_suite.decoder_prune import build_decoder, StudentDecoder
        ck = torch.load(str(S10_PATH), map_location="cpu", weights_only=False)
        pqc = torch.nn.Conv2d(4, 4, 1)
        pqc.load_state_dict(ck["post_quant_conv"])
        dec = build_decoder(ck["meta"]["widths_forward"])
        dec.load_state_dict(ck["state_dict"])
        _S10 = StudentDecoder(pqc, dec).to(device).eval().half()
        for p in _S10.parameters():
            p.requires_grad_(False)
    return _S10

def dino_style_grad(model_dino, model_s10, z, target_sid, device):
    """Compute grad_z of DINO CLS feature norm (proxy for style alignment)."""
    for p in model_dino.parameters():
        p.requires_grad_(False)
    zi = z.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        rgb = model_s10((zi / 0.18215).float()).float()
        x = Fn.interpolate(rgb, size=(224, 224), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        x = (x - mean) / std
        out = model_dino(x, return_dict=True)
        cls = out.last_hidden_state[:, 0, :]  # CLS token
        L = cls.norm(dim=1).mean()
        gx = torch.autograd.grad(L, zi)[0]
    return gx.detach()

def haar_decompose_grad(gx):
    """Decompose gradient into 2-level Haar bands."""
    ll1 = Fn.avg_pool2d(gx, 2)
    h1 = gx - Fn.interpolate(ll1, size=gx.shape[-2:], mode="nearest")
    ll2 = Fn.avg_pool2d(ll1, 2)
    h2 = ll1 - Fn.interpolate(ll2, size=ll1.shape[-2:], mode="nearest")
    return {"LL2": ll2, "H2": h2, "H1": h1, "residual": gx - h1 - h2}

def main():
    device = "cuda"
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    model_dino = _dino(device)
    model_s10 = _s10(device)
    STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
    IMG_ROOT = Path("I:/Latent_Style/WEAVE/datasets/wikiart_new_512")
    POOL = Path("I:/Latent_Style/pseudo_gt_24h/orig_mean_fp16")

    import hashlib
    def _h10(pid):
        return int(hashlib.md5(pid.encode()).hexdigest()[:8], 16) % 10

    # collect holdout paintings
    holdout = []
    for si, s in enumerate(STYLES):
        d = torch.load(str(POOL / f"{si:02d}_{s}.pt"), map_location="cpu", weights_only=False)
        for i, pid in enumerate(d["ids"]):
            img = IMG_ROOT / s / Path(str(d["files"][i])).name
            if img.exists() and _h10(str(pid)) >= 9:
                holdout.append((si, str(img), d["latents"][i:i+1]))
                break  # just 1 per style for the gradient analysis
        if len(holdout) >= 5:
            break

    # use 5 holdout paintings (one per style)
    results = {"per_band_energy": {}, "per_style_cosine": {}}
    all_grads = {}
    for si, (style_idx, img_path, z_real) in enumerate(holdout[:5]):
        z = z_real.float().to(device)
        gx = dino_style_grad(model_dino, model_s10, z, style_idx, device)
        bands = haar_decompose_grad(gx)
        energy = {}
        for band_name, bt in bands.items():
            energy[band_name] = round(bt.square().mean().item(), 6)
        total = sum(v for v in energy.values())
        for k, v in energy.items():
            energy[k + "_frac"] = round(v / max(total, 1e-10), 4)
        all_grads[style_idx] = gx
        results.setdefault("per_band_energy", {})[f"style{style_idx}"] = energy
        print(f"[band] style{style_idx} {energy}", flush=True)

    # aggregate: mean fraction per band across styles
    agg = {}
    for band in ("LL2", "H2", "H1", "residual"):
        fracs = [results["per_band_energy"][f"style{si}"].get(band + "_frac", 0)
                 for si in range(5)]
        agg[band + "_frac_mean"] = round(sum(fracs) / len(fracs), 4)
    # also raw energy
    for band in ("LL2", "H2", "H1", "residual"):
        energies = [results["per_band_energy"][f"style{si}"].get(band, 0)
                    for si in range(5)]
        agg[band + "_energy_mean"] = round(sum(energies) / len(energies), 6)
    results["aggregate"] = agg

    # cross-style gradient cosine (correct vs wrong)
    cosines = []
    for si_a in range(5):
        for si_b in range(5):
            if si_a >= len(all_grads) or si_b >= len(all_grads) or si_a == si_b:
                continue
            ga = all_grads[si_a].flatten()
            gb = all_grads[si_b].flatten()
            cos = torch.nn.functional.cosine_similarity(
                ga[None], gb[None], dim=1).item()
            cosines.append({"a": si_a, "b": si_b, "cos": round(cos, 4)})
    if cosines:
        import statistics
        same_style_cos = [c["cos"] for c in cosines if c["a"] == c["b"]]
        cross_style_cos = [c["cos"] for c in cosines if c["a"] != c["b"]]
        results["grad_cosine"] = {
            "same_style": round(statistics.mean(same_style_cos), 4) if same_style_cos else None,
            "cross_style": round(statistics.mean(cross_style_cos), 4) if cross_style_cos else None,
        }

    (OUT / "band_energy.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))
    print("saved ->", OUT / "band_energy.json")

if __name__ == "__main__":
    main()
