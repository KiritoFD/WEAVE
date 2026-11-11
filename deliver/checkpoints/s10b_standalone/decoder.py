"""Standalone S10B loader. No WEAVE / A0 imports."""
import json, torch, torch.nn as nn
from diffusers.models.autoencoders.vae import Decoder


def build(cfg):
    dec = Decoder(in_channels=4, out_channels=3,
                  up_block_types=["UpDecoderBlock2D"] * 4,
                  block_out_channels=cfg["widths_forward"],
                  layers_per_block=2, norm_num_groups=32, act_fn="silu")
    pq = nn.Conv2d(4, 4, 1)
    class _W(nn.Module):
        def __init__(self, pq, dec):
            super().__init__(); self.post_quant_conv = pq; self.decoder = dec
        def forward(self, z):
            return self.decoder(self.post_quant_conv(z))
    return _W(pq, dec)


def load(path, map_location="cpu"):
    cfg = json.load(open(f"{path}/config.json"))
    m = build(cfg)
    ck = torch.load(f"{path}/decoder.pt", map_location=map_location,
                    weights_only=False)
    m.decoder.load_state_dict(ck["state_dict"])
    m.post_quant_conv.load_state_dict(ck["post_quant_conv"])
    return m.eval()
