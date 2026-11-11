"""Export a self-contained Diffusers-compatible Decoder wrapper.

Given an SD0-S10 checkpoint (state_dict + post_quant_conv), produce a directory
that can be loaded WITHOUT any WEAVE repo imports:

    from diffusers import AutoencoderKL
    import torch, json
    cfg = json.load(open("<export_dir>/config.json"))
    dec = build_sd0_s10_decoder(cfg)          # see exported decoder.py
    dec.load_state_dict(torch.load("<export_dir>/decoder.pt"))
    # usage: x = vae.encode(img).latent_dist.mean;  y = dec(x)

The wrapper applies post_quant_conv EXACTLY ONCE and expects UNSCALED latent.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

ROOT = Path("/root/Workspace/WEAVE_VAE")
OUT = ROOT / "runs/sd0_s10"


def export(ckpt_path: Path, name: str, encoder_hash: dict, split_meta: dict):
    d = OUT / f"export_{name}"
    d.mkdir(parents=True, exist_ok=True)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    torch.save({"state_dict": ck["state_dict"],
                "post_quant_conv": ck["post_quant_conv"]},
               d / "decoder.pt")
    cfg = {
        "widths_forward": [64, 128, 192, 256],
        "params": 10657923,
        "in_channels": 4, "out_channels": 3,
        "latent_convention": "UNSCALED posterior mean; no scaling_factor applied",
        "scaling_factor_reference": 0.18215,
        "post_quant_conv": "included, applied exactly once inside the wrapper",
        "encoder": "/root/Workspace/xy/DiT/data/pretrained/sd-vae-ft-ema",
        "encoder_hash": encoder_hash,
        "source_checkpoint": str(ckpt_path),
        "source_meta": ck.get("meta", {}),
        "input_range": "latent (B,4,64,64) FP32/BF16",
        "output_range": "RGB (B,3,512,512), approx [-1,1]",
        "data": {"split": split_meta.get("source"),
                 "n_train": split_meta.get("n_train"),
                 "n_holdout": split_meta.get("n_holdout"),
                 "n_test256": split_meta.get("n_test256")},
    }
    (d / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    # standalone loader (no repo imports)
    (d / "decoder.py").write_text('''"""Standalone SD0-S10 decoder (no WEAVE imports)."""
import json
import torch
import torch.nn as nn


def build_sd0_s10_decoder(cfg):
    from diffusers.models.autoencoders.vae import Decoder
    dec = Decoder(
        in_channels=4, out_channels=3,
        up_block_types=["UpDecoderBlock2D"] * 4,
        block_out_channels=cfg["widths_forward"],
        layers_per_block=2, norm_num_groups=32, act_fn="silu",
    )
    pqc = nn.Conv2d(4, 4, 1)

    class _W(nn.Module):
        def __init__(self, pqc, dec):
            super().__init__()
            self.post_quant_conv = pqc
            self.decoder = dec
        def forward(self, z):
            return self.decoder(self.post_quant_conv(z))
    return _W(pqc, dec)


def load(path, map_location="cpu"):
    cfg = json.load(open(f"{path}/config.json"))
    m = build_sd0_s10_decoder(cfg)
    ck = torch.load(f"{path}/decoder.pt", map_location=map_location,
                    weights_only=False)
    m.decoder.load_state_dict(ck["state_dict"])
    m.post_quant_conv.load_state_dict(ck["post_quant_conv"])
    return m.eval()
''', encoding="utf-8")
    return d


def verify(d: Path, dev=None):
    if dev is None:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    """Load the standalone wrapper and confirm it reproduces the training path."""
    import sys
    sys.path.insert(0, str(d))
    import decoder as standalone
    from diffusers import AutoencoderKL
    from tools.vae_suite.decoder_prune import StudentDecoder, build_decoder
    m = standalone.load(d, map_location="cpu").to(dev).eval()
    # reference: our training wrapper
    ck = torch.load(d / "decoder.pt", map_location="cpu", weights_only=False)
    dec = build_decoder([64, 128, 192, 256])
    dec.load_state_dict(ck["state_dict"])
    pq = torch.nn.Conv2d(4, 4, 1)
    pq.load_state_dict(ck["post_quant_conv"])
    ref = StudentDecoder(pq.to(dev), dec.to(dev)).to(dev).eval()
    z = torch.randn(2, 4, 64, 64, device=dev)
    with torch.no_grad():
        a, b = m(z), ref(z)
    diff = float((a - b).abs().max())
    print(f"  [verify] standalone vs training wrapper max|diff| = {diff:.3e}",
          flush=True)
    assert diff < 1e-5, "standalone wrapper does not match the training path"
    return diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", default="c0_best.pt,c1_best.pt")
    args = ap.parse_args()
    split = json.loads((OUT / "full/split.json").read_text())
    for name in args.checkpoints.split(","):
        f = OUT / name.strip()
        if not f.exists():
            print(f"[skip] {f} missing", flush=True)
            continue
        tag = Path(name.strip()).stem
        d = export(f, tag, split["encoder_hash"], split)
        print(f"[export] {d}", flush=True)
        verify(d)


if __name__ == "__main__":
    main()
