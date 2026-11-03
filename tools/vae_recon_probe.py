"""Route D / P0 diagnostic: VAE reconstruction ceiling probe.

Question answered (review doc 路线D): can the latent space even CARRY the
high-frequency brushwork we want to generate? Encode real target paintings
with the training VAE, decode, and compare local texture + style statistics
before/after reconstruction. If real paintings lose their HF art features
through the VAE round-trip, strengthening latent HH supervision has limited
headroom and an image-space refiner / VAE upgrade should be considered.

Outputs JSON with per-image and aggregate metrics:
  * hf_energy_ratio: HH-subband energy(recon) / energy(orig) in image space
  * swd_<layer>: patch-descriptor SWD between orig and recon features
    (distribution-level texture degradation, same metric family as Route A)
  * edge_iou: structural preservation sanity check
  * verdict: "latent_ok" if median hf_energy_ratio >= 0.85 else "vae_limited"

Smoke mode (--smoke): synthetic images + tiny_conv extractor + tiny decoder
round-trip; proves the metric plumbing on CPU.

Usage (server):
    python tools/vae_recon_probe.py --style_root I:/datasets/d5/test --vae taesd --out runs/vae_probe.json
Usage (smoke):
    python tools/vae_recon_probe.py --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import torch
import torch.nn.functional as F


def probe_pairs(
    orig: torch.Tensor,
    recon: torch.Tensor,
    extractor,
    layers: list[str],
    *,
    patch_size: int = 8,
    num_projections: int = 64,
) -> dict:
    """Compute per-image probe metrics between original and reconstruction."""
    from swd import sliced_wasserstein_distance
    from features import extract_patch_descriptors, sobel_edges
    from wavelet import dwt2_haar

    with torch.no_grad():
        _, _, _, hh_o = dwt2_haar(orig.float())
        _, _, _, hh_r = dwt2_haar(recon.float())
        e_o = hh_o.pow(2).flatten(1).sum(-1)
        e_r = hh_r.pow(2).flatten(1).sum(-1)
        hf_ratio = (e_r / e_o.clamp_min(1e-12)).clamp(0.0, 4.0)

        fo = extractor.forward_layers(orig, layers)
        fr = extractor.forward_layers(recon, layers)
        swd_vals = {}
        for layer in layers:
            d_o = extract_patch_descriptors(fo[layer], patch_size, max(1, patch_size // 2)).reshape(-1, fo[layer].shape[1] * patch_size * patch_size)
            d_r = extract_patch_descriptors(fr[layer], patch_size, max(1, patch_size // 2)).reshape(-1, fr[layer].shape[1] * patch_size * patch_size)
            swd_vals[f"swd_{layer}"] = float(sliced_wasserstein_distance(d_r, d_o, num_projections=num_projections))

        eo = sobel_edges(orig.float()) > 0.1
        er = sobel_edges(recon.float()) > 0.1
        if eo.shape[-2:] != er.shape[-2:]:
            er = F.interpolate(er.float(), size=eo.shape[-2:], mode="nearest") > 0.5
        inter = (eo & er).flatten(1).sum(-1).float()
        union = (eo | er).flatten(1).sum(-1).float().clamp_min(1.0)
        edge_iou = inter / union

    return {
        "hf_energy_ratio": [float(v) for v in hf_ratio],
        "edge_iou": [float(v) for v in edge_iou],
        **swd_vals,
    }


def _summarise(records: list[dict]) -> dict:
    import statistics

    keys = [k for k in records[0] if isinstance(records[0][k], list)] + \
           [k for k in records[0] if isinstance(records[0][k], float)]
    agg: dict = {}
    flat: dict[str, list[float]] = {}
    for rec in records:
        for k, v in rec.items():
            flat.setdefault(k, []).extend(v if isinstance(v, list) else [v])
    for k, vals in flat.items():
        agg[f"{k}_mean"] = statistics.fmean(vals)
        agg[f"{k}_median"] = statistics.median(vals)
    agg["verdict"] = "latent_ok" if agg.get("hf_energy_ratio_median", 0.0) >= 0.85 else "vae_limited"
    agg["n_images"] = len(flat.get("hf_energy_ratio", []))
    return agg


def run_smoke(out: Path) -> None:
    from features import build_feature_extractor

    torch.manual_seed(0)
    extractor = build_feature_extractor("tiny_conv")
    orig = torch.rand(4, 3, 32, 32)
    recon = (orig * 0.9 + 0.05 * torch.rand_like(orig)).clamp(0, 1)  # stand-in round-trip
    rec = probe_pairs(orig, recon, extractor, ["conv2", "conv3"])
    summary = _summarise([rec])
    payload = {"smoke": True, "probe": rec, "summary": summary}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"[smoke] vae recon probe OK -> {out}; verdict={summary['verdict']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--style_root", type=str, default="", help="dir with style subdirs of real paintings")
    ap.add_argument("--vae", type=str, default="taesd", help="taesd|sd15 — MUST match the training data VAE")
    ap.add_argument("--vae_dir", type=str, default="", help="sd15 only: local AutoencoderKL snapshot dir")
    ap.add_argument("--extractor", type=str, default="vgg16")
    ap.add_argument("--layers", type=str, default="relu2_2,relu3_3")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--max_per_style", type=int, default=64)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    if args.smoke:
        run_smoke(out)
        return

    from PIL import Image
    from torchvision import transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from features import build_feature_extractor

    extractor = build_feature_extractor(args.extractor, device=device)
    layers = [s.strip() for s in args.layers.split(",") if s.strip()]

    if args.vae == "taesd":
        from diffusers import AutoencoderTiny

        vae = AutoencoderTiny.from_pretrained(str(_ROOT / "data" / "taesd"), local_files_only=True).to(device).eval()
        scale = 1.0
    else:
        from diffusers import AutoencoderKL

        if args.vae_dir:
            vae = AutoencoderKL.from_pretrained(args.vae_dir, local_files_only=True).to(device).eval()
        else:
            vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").to(device).eval()
        scale = float(vae.config.scaling_factor)
    for p in vae.parameters():
        p.requires_grad_(False)

    records: list[dict] = []
    to_tensor = transforms.ToTensor()
    root = Path(args.style_root)
    for sdir in sorted(d for d in root.iterdir() if d.is_dir()):
        paths = sorted(p for p in sdir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))[: args.max_per_style]
        for p in paths:
            im = Image.open(p).convert("RGB").resize((args.size, args.size), Image.BICUBIC)
            t = to_tensor(im).unsqueeze(0).to(device)
            with torch.no_grad():
                if args.vae == "taesd":
                    z = vae.encode(t * 2.0 - 1.0).latents * scale
                    rec = vae.decode(z / scale).sample
                else:
                    z = vae.encode(t * 2.0 - 1.0).latent_dist.mean * scale
                    rec = vae.decode(z / scale).sample
                rec = (rec.float() * 0.5 + 0.5).clamp(0.0, 1.0)
            records.append(probe_pairs(t, rec, extractor, layers))
        print(f"[{sdir.name}] probed {len(paths)} images")

    payload = {
        "vae": args.vae, "extractor": args.extractor, "layers": layers, "size": args.size,
        "summary": _summarise(records),
        "note": "verdict=vae_limited means real brushwork HF does not survive the VAE round-trip; "
                "prefer image-space refiner / VAE upgrade over stronger latent HH supervision.",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"verdict={payload['summary']['verdict']} -> {out}")


if __name__ == "__main__":
    main()
