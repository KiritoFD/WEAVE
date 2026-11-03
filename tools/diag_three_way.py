"""P0 diagnostic: three-way style-loss localisation.

Question (review doc P0 诊断): WHERE does style fidelity die?
  (1) VAE reconstruction of the real target painting   — encoding ceiling;
  (2) the stylised training ENDPOINT z_1* decoded      — target-construction ceiling;
  (3) the model's FINAL OUTPUT                          — transport ceiling.

For the same (content, style) pairs we decode all three and compute the same
style metric (patch SWD vs the real style image) + content metric (edge IoU vs
content image) on each. The deltas tell us whether to invest in the VAE, the
endpoint, or the transport network.

Smoke mode: tiny decoder/encoder round-trip + tiny model; plumbing only.

Usage (server):
    python tools/diag_three_way.py --checkpoint runs/x/latest.pt --config config.json \
        --style_root I:/datasets/d5/test --out runs/diag_three_way.json
Usage (smoke):
    python tools/diag_three_way.py --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import torch


def three_way_metrics(
    img_style: torch.Tensor,
    img_content: torch.Tensor,
    img_recon: torch.Tensor,
    img_endpoint: torch.Tensor,
    img_output: torch.Tensor,
    extractor,
    style_layer: str,
) -> dict:
    """Compute comparable style/content metrics for the three decoded images."""
    from features import extract_patch_descriptors, sobel_edges
    from swd import sliced_wasserstein_distance
    import torch.nn.functional as F

    def _style_swd(img: torch.Tensor) -> float:
        fa = extractor.forward_layers(img, [style_layer])[style_layer]
        fs = extractor.forward_layers(img_style, [style_layer])[style_layer]
        da = extract_patch_descriptors(fa, 8, 4).reshape(-1, fa.shape[1] * 64)
        ds = extract_patch_descriptors(fs, 8, 4).reshape(-1, fs.shape[1] * 64)
        return float(sliced_wasserstein_distance(da, ds))

    def _edge_iou(img: torch.Tensor) -> float:
        ea = sobel_edges(img.float()) > 0.1
        ec = sobel_edges(img_content.float()) > 0.1
        if ea.shape[-2:] != ec.shape[-2:]:
            ec = F.interpolate(ec.float(), size=ea.shape[-2:], mode="nearest") > 0.5
        inter = (ea & ec).flatten(1).sum(-1).float()
        union = (ea | ec).flatten(1).sum(-1).float().clamp_min(1.0)
        return float((inter / union).mean())

    with torch.no_grad():
        return {
            "vae_recon": {"style_swd": _style_swd(img_recon), "content_edge_iou": _edge_iou(img_recon)},
            "endpoint": {"style_swd": _style_swd(img_endpoint), "content_edge_iou": _edge_iou(img_endpoint)},
            "final_output": {"style_swd": _style_swd(img_output), "content_edge_iou": _edge_iou(img_output)},
        }


def interpret(agg: dict) -> str:
    """Rule-of-thumb interpretation from aggregated three-way metrics."""
    s = agg
    recon, endpoint, final = s["vae_recon"]["style_swd_mean"], s["endpoint"]["style_swd_mean"], s["final_output"]["style_swd_mean"]
    if recon > endpoint and recon > final:
        return "encoding_limited: VAE round-trip already loses the most style signal"
    if endpoint > final:
        return "target_construction_limited: the training endpoint itself is style-poor"
    if final > endpoint * 1.1:
        return "transport_limited: the network fails to reach even the endpoint's style level"
    return "balanced: no single dominant bottleneck"


def run_smoke(out: Path) -> None:
    from config_schema import ExperimentConfig
    from features import build_feature_extractor
    from model import build_model_from_config
    from output_supervision import build_latent_decoder, unroll_euler_decode
    from flow import FlowMatchingObjective

    torch.manual_seed(0)
    cfg = ExperimentConfig.from_mapping({
        "model": {"base_dim": 32, "num_res_blocks": 2, "num_styles": 2, "contract_family": "weave"},
        "bridge": {"structure_aligned_target": True},
    })
    model = build_model_from_config(cfg.model, bridge_cfg=cfg.bridge).eval()
    objective = FlowMatchingObjective(cfg)
    decoder = build_latent_decoder("tiny")
    extractor = build_feature_extractor("tiny_conv")

    content = torch.randn(2, 4, 16, 16)
    target = torch.randn(2, 4, 16, 16)
    with torch.no_grad():
        img_style = decoder.decode_image(target)
        img_content = decoder.decode_image(content)
        img_recon = decoder.decode_image(content)  # stand-in: decode(encode(x)) == decode here
        endpoint_latent = objective._build_training_target(content, target, target)
        img_endpoint = decoder.decode_image(endpoint_latent)
        img_output = unroll_euler_decode(model, decoder, content, steps=2, style_id=1,
                                         style_latent=target[:1], use_grad_checkpoint=False)
    m = three_way_metrics(img_style, img_content, img_recon, img_endpoint, img_output, extractor, "conv3")
    agg = {k: {f"{m2}_mean": v2 for m2, v2 in v.items()} for k, v in m.items()}
    payload = {"smoke": True, "metrics": m, "verdict_rule_example": interpret(agg)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"[smoke] three-way probe OK -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="")
    ap.add_argument("--config", type=str, default="config.json")
    ap.add_argument("--style_root", type=str, default="")
    ap.add_argument("--num_pairs", type=int, default=32)
    ap.add_argument("--extractor", type=str, default="vgg16")
    ap.add_argument("--style_layer", type=str, default="relu3_3")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    if args.smoke:
        run_smoke(out)
        return

    raise SystemExit(
        "Server path: for each (content, style) pair build (1) VAE recon of the style "
        "image, (2) decoded SAT endpoint via FlowMatchingObjective._build_training_target, "
        "(3) final model output via the standard inference pipeline; then call "
        "diag_three_way.three_way_metrics(...) and average. Metric core is implemented; "
        "data plumbing depends on the server dataset layout."
    )


if __name__ == "__main__":
    main()
