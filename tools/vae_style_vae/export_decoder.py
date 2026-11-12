"""Export the joint style-VAE's decoder as a standalone checkpoint.

C_edit_best.pt stores {encoder, decoder, style_head}; post_quant_conv was frozen
from A0 and is not saved there. This writes the same layout as the s10_* files
({state_dict, post_quant_conv, meta}) so the decoder can be loaded and used by
tools/vae_suite/decoder_prune.StudentDecoder without the encoder.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/Workspace/WEAVE_VAE")

import torch
from diffusers import AutoencoderKL

A0_DIR = "/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae"
RUN = Path("/root/Workspace/WEAVE_VAE/runs/vae_style_vae")


def export(src_name, out_name, meta_extra):
    ck = torch.load(RUN / src_name, map_location="cpu", weights_only=False)
    vae = AutoencoderKL.from_pretrained(A0_DIR, torch_dtype=torch.float32)
    dec = ck["decoder"]
    n = sum(v.numel() for v in dec.values())
    meta = {"widths_forward": [64, 128, 192, 256], "params": n,
            "needs_post_quant_conv": True, "source": src_name, **meta_extra}
    out = RUN / out_name
    torch.save({"state_dict": dec,
                "post_quant_conv": vae.post_quant_conv.state_dict(),
                "meta": meta}, out)
    print(f"[{out_name}] params={n/1e6:.3f}M  {out.stat().st_size/1e6:.1f} MB  {meta}",
          flush=True)
    return out


def main():
    ck = torch.load(RUN / "C_edit_best.pt", map_location="cpu", weights_only=False)
    print("C_edit_best meta:", ck.get("meta"), flush=True)
    export("C_edit_best.pt", "C_edit_decoder_only.pt", ck.get("meta", {}))
    export("B_style_best.pt", "B_style_decoder_only.pt",
           torch.load(RUN / "B_style_best.pt", map_location="cpu",
                      weights_only=False).get("meta", {}))


if __name__ == "__main__":
    main()
