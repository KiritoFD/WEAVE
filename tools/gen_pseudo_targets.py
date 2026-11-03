"""Route B: offline pseudo-target generation + content-consistency filtering.

Pipeline (server):
  1. Load content/style images from the dataset roots.
  2. Run a pluggable teacher (StyTR2 / AesPA-Net / AdaAttN / existing WEAVE
     checkpoint / --teacher fallback_stats) to create content-aligned
     stylised candidates.
  3. Score each candidate for content-structure consistency
     (self-similarity correlation + edge IoU, see pseudo_targets.py).
  4. Encode survivors to latent with the training VAE and write the cache
     consumed by ``bridge.endpoint_mode = "pseudo" | "mixed"``.

Smoke mode (--smoke): synthetic tensors, tiny_conv extractor, tiny decoder
inverse is unavailable so we cache the content latent itself perturbed —
proves the plumbing end-to-end on CPU in seconds without data/GPU.

Usage (server):
    python tools/gen_pseudo_targets.py \
        --content_root I:/datasets/d5/train --style_root I:/datasets/d5/train \
        --teacher fallback_stats --vae taesd --out data/pseudo_d5 --size 512
Usage (smoke):
    python tools/gen_pseudo_targets.py --smoke
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import torch


def run_smoke(out: Path) -> None:
    from features import build_feature_extractor
    from pseudo_targets import (
        FallbackStatsTeacher, PseudoTargetCache, PseudoTargetCacheWriter,
        score_content_consistency,
    )

    torch.manual_seed(0)
    extractor = build_feature_extractor("tiny_conv")
    teacher = FallbackStatsTeacher()
    writer = PseudoTargetCacheWriter(out, shard_size=4)
    n_content, n_style = 6, 3
    kept = 0
    for ci in range(n_content):
        content_img = torch.rand(1, 3, 32, 32)
        for sid in range(n_style):
            style_img = torch.rand(1, 3, 32, 32)
            pseudo = teacher.stylize(content_img, style_img)
            score = score_content_consistency(pseudo, content_img, extractor, selfsim_layer="conv3")
            # Smoke: cache a stand-in "latent" (real path encodes via VAE).
            latent = torch.randn(1, 4, 16, 16)
            writer.add(latent, [f"0/{ci}"], [sid], [float(score[0])])
            kept += 1
    writer.close(score_threshold=0.0, meta={"smoke": True})
    cache = PseudoTargetCache(out)
    assert cache.count == kept, (cache.count, kept)
    sample = cache.get("0/0", 0)
    assert sample is not None and sample.shape == (4, 16, 16)
    print(f"[smoke] pseudo-target cache OK: {cache.count} entries at {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--content_root", type=str, default="")
    ap.add_argument("--style_root", type=str, default="")
    ap.add_argument("--teacher", type=str, default="fallback_stats",
                    help="fallback_stats | stytr2 | aespa | adaattn | weave_ckpt:<path>")
    ap.add_argument("--extractor", type=str, default="dinov2", help="scoring backbone: dinov2|vgg16|tiny_conv")
    ap.add_argument("--selfsim_layer", type=str, default="block6")
    ap.add_argument("--vae", type=str, default="taesd", help="taesd|sd15")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--score_threshold", type=float, default=0.35)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    if args.smoke:
        run_smoke(out)
        return

    # --- server path ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from features import build_feature_extractor
    from pseudo_targets import PseudoTargetCacheWriter, score_content_consistency

    extractor = build_feature_extractor(args.extractor, device=device)

    # Teacher mounting: external repos are integrated by wrapping their
    # inference callables via pseudo_targets.CallableTeacher. fallback_stats
    # needs nothing external.
    if args.teacher == "fallback_stats":
        from pseudo_targets import FallbackStatsTeacher

        teacher = FallbackStatsTeacher()
    else:
        raise SystemExit(
            f"Teacher {args.teacher!r} is not bundled. Wrap your StyTR2/AesPA-Net/AdaAttN "
            "checkpoint with pseudo_targets.CallableTeacher and pass it in a small driver "
            "script (see module docstring), or use --teacher fallback_stats."
        )

    # VAE encoder for latent caching (same VAE family as training data!).
    if args.vae == "taesd":
        from diffusers import AutoencoderTiny

        vae = AutoencoderTiny.from_pretrained(str(_ROOT / "data" / "taesd"), local_files_only=True).to(device).eval()
        scale = 1.0
    else:
        from diffusers import AutoencoderKL

        vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").to(device).eval()
        scale = float(vae.config.scaling_factor)
    for p in vae.parameters():
        p.requires_grad_(False)

    import cv2

    content_root, style_root = Path(args.content_root), Path(args.style_root)
    style_dirs = sorted([d for d in style_root.iterdir() if d.is_dir()])
    writer = PseudoTargetCacheWriter(out)
    total, kept = 0, 0
    for sid, sdir in enumerate(style_dirs):
        style_paths = sorted([p for p in sdir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
        content_paths = sorted([p for p in (content_root / sdir.name).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
        for ci, cpath in enumerate(content_paths):
            content_img = cv2.imread(str(cpath))[:, :, ::-1]
            content_img = cv2.resize(content_img, (args.size, args.size))
            content_t = torch.from_numpy(content_img).permute(2, 0, 1).float().div_(255.0).unsqueeze(0).to(device)
            for spath in style_paths[:4]:  # up to 4 references per style per content
                style_img = cv2.imread(str(spath))[:, :, ::-1]
                style_img = cv2.resize(style_img, (args.size, args.size))
                style_t = torch.from_numpy(style_img).permute(2, 0, 1).float().div_(255.0).unsqueeze(0).to(device)
                pseudo = teacher.stylize(content_t, style_t)
                score = score_content_consistency(pseudo, content_t, extractor, selfsim_layer=args.selfsim_layer)
                total += 1
                if float(score[0]) < args.score_threshold:
                    continue
                with torch.no_grad():
                    if args.vae == "taesd":
                        lat = vae.encode(pseudo * 2.0 - 1.0).latents * scale
                    else:
                        lat = vae.encode(pseudo * 2.0 - 1.0).latent_dist.mean * scale
                writer.add(lat.cpu(), [f"{sid}/{ci}"], [sid], [float(score[0])])
                kept += 1
        print(f"[{sdir.name}] kept {kept}/{total} (threshold {args.score_threshold})")
    writer.close(score_threshold=args.score_threshold,
                 meta={"teacher": args.teacher, "extractor": args.extractor, "vae": args.vae, "size": args.size})
    print(f"Done. kept {kept}/{total} -> {out}")


if __name__ == "__main__":
    main()
