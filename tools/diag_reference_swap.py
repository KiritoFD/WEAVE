"""P0 diagnostic: reference-swap interpretability probe.

Question (review doc P0): with content and Style-ID FIXED, does swapping the
reference image within the SAME style category produce interpretable,
style-consistent changes in the output? If yes, the reference pathway actually
carries instance-level style information; if outputs barely change, the model
is still Style-ID-dominated.

For each content image and style s: run inference K times with K different
same-style reference images, then report:
  * output pairwise LPIPS-less diversity: std/mean of per-pixel variance
    across the K outputs (do outputs move at all?);
  * style-feature alignment: SWD between each output and its own reference
    vs SWD against the OTHER references (does the output follow ITS
    reference?). alignment_margin = mean(swd_cross - swd_own) > 0 is good.

Smoke mode: tiny model + tiny decoder + tiny extractor, synthetic latents;
checks plumbing only. Real runs need a trained checkpoint + VAE + dataset.

Usage (server):
    python tools/diag_reference_swap.py --checkpoint runs/x/latest.pt --config config.json \
        --refs_per_style 4 --out runs/diag_reference_swap.json
Usage (smoke):
    python tools/diag_reference_swap.py --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import torch


def analyse_outputs(outputs: list[torch.Tensor], ref_feats: list[torch.Tensor], extractor, layer: str) -> dict:
    """outputs[K] images (1,3,H,W); ref_feats[K] feature maps of references."""
    from features import extract_patch_descriptors
    from swd import sliced_wasserstein_distance

    K = len(outputs)
    out_feats = [extractor.forward_layers(o, [layer])[layer] for o in outputs]
    swd_own, swd_cross = [], []
    for i in range(K):
        d_o = extract_patch_descriptors(out_feats[i], 8, 4).reshape(-1, out_feats[i].shape[1] * 64)
        d_r = extract_patch_descriptors(ref_feats[i], 8, 4).reshape(-1, ref_feats[i].shape[1] * 64)
        swd_own.append(float(sliced_wasserstein_distance(d_o, d_r)))
        for j in range(K):
            if j != i:
                d_x = extract_patch_descriptors(ref_feats[j], 8, 4).reshape(-1, ref_feats[j].shape[1] * 64)
                swd_cross.append(float(sliced_wasserstein_distance(d_o, d_x)))
    stack = torch.stack(outputs).flatten(2)  # (K, 3, HW)
    diversity = float(stack.var(dim=0).mean())
    return {
        "output_diversity": diversity,
        "swd_own_mean": sum(swd_own) / len(swd_own),
        "swd_cross_mean": sum(swd_cross) / max(1, len(swd_cross)),
        "alignment_margin": (sum(swd_cross) / max(1, len(swd_cross))) - (sum(swd_own) / len(swd_own)),
        "interpretable": bool(diversity > 1e-6 and (sum(swd_cross) / max(1, len(swd_cross))) > (sum(swd_own) / len(swd_own))),
    }


def run_smoke(out: Path) -> None:
    from config_schema import ExperimentConfig
    from features import build_feature_extractor
    from model import build_model_from_config
    from output_supervision import build_latent_decoder, unroll_euler_decode

    torch.manual_seed(0)
    cfg = ExperimentConfig.from_mapping({
        "model": {"base_dim": 32, "num_res_blocks": 2, "num_styles": 2, "contract_family": "weave",
                  "target_latent_hf_head_fusion_enabled": True},
        "bridge": {},
    })
    model = build_model_from_config(cfg.model, bridge_cfg=cfg.bridge).eval()
    decoder = build_latent_decoder("tiny")
    extractor = build_feature_extractor("tiny_conv")
    content = torch.randn(1, 4, 16, 16)
    K = 3
    outputs, ref_feats = [], []
    with torch.no_grad():
        for _ in range(K):
            ref_latent = torch.randn(1, 4, 16, 16)  # same style id, different reference
            img = unroll_euler_decode(model, decoder, content, steps=2, style_id=1, style_latent=ref_latent,
                                      use_grad_checkpoint=False)
            outputs.append(img)
            ref_feats.append(extractor.forward_layers(decoder.decode_image(ref_latent), ["conv3"])["conv3"])
    report = analyse_outputs(outputs, ref_feats, extractor, "conv3")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"smoke": True, "report": report}, indent=2))
    print(f"[smoke] reference-swap probe OK -> {out}; interpretable={report['interpretable']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="")
    ap.add_argument("--config", type=str, default="config.json")
    ap.add_argument("--refs_per_style", type=int, default=4)
    ap.add_argument("--num_contents", type=int, default=16)
    ap.add_argument("--extractor", type=str, default="vgg16")
    ap.add_argument("--layer", type=str, default="relu3_3")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    if args.smoke:
        run_smoke(out)
        return

    raise SystemExit(
        "Server path: load the checkpoint with utils/inference.py's pipeline, run K="
        f"{args.refs_per_style} same-style reference swaps per content, decode, then call "
        "diag_reference_swap.analyse_outputs(outputs, ref_feats, extractor, layer=...) and "
        "aggregate over contents. The function is fully implemented; the data-plumbing "
        "wrapper is intentionally thin because eval data layout is server-specific."
    )


if __name__ == "__main__":
    main()
