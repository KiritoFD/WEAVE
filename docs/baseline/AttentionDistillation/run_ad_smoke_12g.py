#!/usr/bin/env python3
"""12GB smoke runner for Attention Distillation (SD1.5).

Drives the repo's own style-transfer entry point (webui/runner.py ->
ADPipeline.optimize) in-process for ONE image pair. Steps=20 is a VRAM gate,
NOT the paper default (200). Fails loudly on any exception/OOM.
"""
import argparse
import json
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--content", required=True, type=Path)
    ap.add_argument("--style", required=True, type=Path)
    ap.add_argument("--model", default="pretrained_models/stable-diffusion-v1-5")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--out", type=Path, default=Path("ad_smoke_out"))
    ap.add_argument("--content-weight", type=float, default=0.25)
    ap.add_argument("--lr", type=float, default=0.05)
    args = ap.parse_args()

    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    args.out.mkdir(parents=True, exist_ok=True)

    # ---- locate the official entry; verify against the pinned commit ----
    # Expected (per audit): webui/runner.py exposes run_style_transfer which
    # builds ADPipeline and calls optimize(). If the import surface differs,
    # fix HERE and record the diff in provenance — do not patch the algorithm.
    try:
        from webui.runner import run_style_transfer  # type: ignore
    except Exception as e:
        print(f"[FATAL] cannot import webui.runner.run_style_transfer: {e}")
        print("Inspect <repo>/webui/runner.py at the pinned commit and adapt.")
        sys.exit(2)

    t0 = time.monotonic()
    try:
        result = run_style_transfer(
            content_path=str(args.content),
            style_path=str(args.style),
            model_path=str(repo / args.model),
            num_steps=args.steps,
            precision=args.precision,
            content_weight=args.content_weight,
            lr=args.lr,
            output_dir=str(args.out),
            enable_gradient_checkpoint=True,
        )
    except TypeError:
        # signature drift fallback: try the minimal documented call
        result = run_style_transfer(
            str(args.content), str(args.style), str(repo / args.model),
            num_steps=args.steps)
    except Exception:
        raise
    dt = time.monotonic() - t0

    pngs = sorted(args.out.rglob("*.png"))
    meta = {"steps": args.steps, "precision": args.precision,
            "wall_s": round(dt, 2), "outputs": [str(p) for p in pngs],
            "content": str(args.content), "style": str(args.style)}
    (args.out / "smoke_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))
    if not pngs:
        print("[FATAL] no PNG produced", file=sys.stderr)
        sys.exit(1)
    print("SMOKE OK (VRAM gate only — not a quality result)")


if __name__ == "__main__":
    main()
