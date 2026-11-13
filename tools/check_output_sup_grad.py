"""Route A memory-optimisation guard: verify the output-level supervision loss
still back-propagates through the unrolled Euler ODE path and the decoder.

Run on the server with the real checkpoint + local DINOv2 snapshot:

    python3 tools/check_output_sup_grad.py \
        --config runs/nextgen_routeA_post/config_routeA.json \
        --checkpoint exp/dino_s_break/brk_a_ll03_10ep/epoch_0010.pt \
        --extractor dinov2 --extractor-local-dir /mnt/i/Latent_Style/weights/dinov2-base \
        --feature-res 384 --chunk-rows 4 --ode-steps 2

Exit code 0 only if every assertion passes:
  * style / content sub-losses are finite and non-zero;
  * grad w.r.t. the content latent (ODE start) is non-zero -> the whole
    unrolled inference path is differentiable;
  * grad w.r.t. at least one model parameter is non-zero -> supervision
    reaches the network, not just the frozen feature extractor.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.legacy_root.config_schema import load_experiment_config  # noqa: E402
from tools.legacy_root.features import build_feature_extractor  # noqa: E402
from tools.legacy_root.model import build_model_from_config  # noqa: E402
from tools.legacy_root.output_supervision import OutputSupervision, build_latent_decoder  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--extractor", default="")
    ap.add_argument("--extractor-local-dir", default="")
    ap.add_argument("--decoder", default="")
    ap.add_argument("--style-layers", default="")
    ap.add_argument("--content-layers", default="")
    ap.add_argument("--selfsim-layer", default="")
    ap.add_argument("--feature-res", type=int, default=-1)
    ap.add_argument("--chunk-rows", type=int, default=-1)
    ap.add_argument("--ode-steps", type=int, default=-1)
    ap.add_argument("--patch-size", type=int, default=-1)
    ap.add_argument("--patch-stride", type=int, default=-1)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_experiment_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    extractor_name = args.extractor or str(getattr(cfg.bridge, "output_supervision_extractor", "tiny_conv"))
    local_dir = args.extractor_local_dir or str(getattr(cfg.bridge, "output_supervision_extractor_local_dir", ""))
    extractor = build_feature_extractor(extractor_name, device=device, local_dir=local_dir)
    decoder = build_latent_decoder(
        args.decoder or str(getattr(cfg.bridge, "output_supervision_decoder", "tiny")), device=device,
    )
    res = args.feature_res if args.feature_res >= 0 else int(getattr(cfg.bridge, "output_supervision_feature_res", 0))
    chunk = args.chunk_rows if args.chunk_rows >= 0 else int(getattr(cfg.bridge, "output_supervision_chunk_rows", 0))
    steps = args.ode_steps if args.ode_steps >= 0 else int(getattr(cfg.bridge, "output_supervision_ode_steps", 2))

    style_layers = (args.style_layers or str(getattr(cfg.bridge, "output_supervision_style_layers", "conv2,conv3"))).split(",")
    content_layers = (
        args.content_layers or str(getattr(cfg.bridge, "output_supervision_content_layers", "conv1,conv2"))
    ).split(",")
    selfsim = args.selfsim_layer or str(getattr(cfg.bridge, "output_supervision_selfsim_layer", "conv3"))

    model = build_model_from_config(cfg.model, bridge_cfg=cfg.bridge, use_checkpointing=False).to(device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state = state.get("model_state_dict", state)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[ckpt] missing={len(missing)} unexpected={len(unexpected)}")
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)

    sup = OutputSupervision(
        extractor, decoder,
        style_layers=style_layers,
        content_perceptual_layers=content_layers,
        content_selfsim_layer=selfsim,
        ode_steps=steps,
        swd_projections=int(getattr(cfg.bridge, "output_supervision_swd_projections", 64)),
        patch_size=(args.patch_size if args.patch_size > 0 else int(getattr(cfg.bridge, "output_supervision_patch_size", 8))),
        patch_stride=(args.patch_stride if args.patch_stride > 0 else int(getattr(cfg.bridge, "output_supervision_patch_stride", 4))),
        w_perceptual=float(getattr(cfg.bridge, "output_supervision_w_perceptual", 1.0)),
        w_selfsim=float(getattr(cfg.bridge, "output_supervision_w_selfsim", 1.0)),
        w_edge=float(getattr(cfg.bridge, "output_supervision_w_edge", 0.5)),
        grad_checkpoint=False,  # keep the raw graph so grad w.r.t. the input is visible
        microbatch=0,
        feature_res=res,
        chunk_rows=chunk,
    ).to(device)
    sup.train()

    torch.manual_seed(0)
    content = torch.randn(1, 4, 64, 64, device=device, requires_grad=True)
    style_latent = torch.randn(1, 4, 64, 64, device=device)
    style_id = torch.zeros(1, dtype=torch.long, device=device)
    with torch.no_grad():
        style_img = decoder.decode_image(style_latent)
        content_img = decoder.decode_image(content)

    logs = sup(
        model, content_latent=content, style_image=style_img, content_image=content_img,
        style_id=style_id, style_latent=style_latent,
    )
    loss = logs["loss_style_out"] + logs["loss_content_out"]
    loss.backward()

    latent_grad = float(content.grad.abs().sum()) if content.grad is not None else 0.0
    param_grad = sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
    report = {
        "extractor": extractor_name,
        "decoder": type(decoder).__name__,
        "feature_res": res,
        "chunk_rows": chunk,
        "ode_steps": steps,
        "style_loss": float(logs["loss_style_out"]),
        "content_loss": float(logs["loss_content_out"]),
        "style_layers": style_layers,
        "content_layers": content_layers,
        "selfsim_layer": selfsim,
        "latent_grad_abs_sum": latent_grad,
        "model_param_grad_abs_sum": param_grad,
        "peak_gb": (torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else 0.0,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))

    ok = (
        torch.isfinite(loss).item()
        and float(logs["loss_style_out"]) > 0.0
        and float(logs["loss_content_out"]) > 0.0
        and latent_grad > 0.0
        and param_grad > 0.0
    )
    print("GRAD_CHECK", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
