"""Build the Cold-SD0 initialisation using the ORIGINAL S10 pruning METHOD.

Reusing A0's channel-index NUMBERS on the official SD0 decoder is wrong: the two
decoders have different weights, so the same index refers to different semantic
channels (measured: Cold LPIPS 0.808, a catastrophic mismatch). The plan asks for
the original S10 *strategy* -- Taylor importance + weight inheritance -- applied
to the official SD0 decoder, in the SD0 latent space.

    imp = taylor_importance(D_SD0, plan, z_SD0, x, pre=pqc_SD0)
    C, meta = prune_decoder(D_SD0, WIDTHS, importance=imp)

Saves C_decoder.pt for P1 and p0_baseline.py to consume.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL

ROOT = Path("/root/Workspace/WEAVE_VAE")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "vae_suite"))

from tools.vae_suite.decoder_prune import (StudentDecoder, make_plan,
                                           prune_decoder, taylor_importance)
from tools.vae_suite.train_sd0_s10 import load_data

SD0 = "/root/Workspace/xy/DiT/data/pretrained/sd-vae-ft-ema"
OUT = ROOT / "runs/sd0_s10"
WIDTHS = [64, 128, 192, 256]


def main():
    dev = "cuda"
    sp, ds_tr, _, _ = load_data()
    official = AutoencoderKL.from_pretrained(SD0, torch_dtype=torch.float32).to(dev).eval()
    # calibration batch from the SD0 latent cache (128 real images)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ds_tr), 128, replace=False)
    z = torch.stack([ds_tr[int(i)]["z"] for i in idx])
    x = torch.stack([ds_tr[int(i)]["x"] for i in idx])
    print(f"[Cold] Taylor calibration on {len(idx)} SD0 latents", flush=True)
    imp = taylor_importance(official.decoder, make_plan(official.decoder),
                            z, x, dev, n_batches=8, bs=2,
                            pre=official.post_quant_conv)
    dec, meta = prune_decoder(official.decoder.cpu(), WIDTHS, importance=imp)
    torch.save({"state_dict": dec.state_dict(),
                "post_quant_conv": official.post_quant_conv.state_dict(),
                "meta": {**meta, "source": "official SD0 Taylor-pruned",
                         "needs_post_quant_conv": True}},
               OUT / "cold_init.pt")
    print(f"[Cold] saved cold_init.pt params={meta['params']:,}", flush=True)
    print(f"  channel indices (first5): "
          f"{ {k: list(v)[:5] for k,v in meta['channel_index'].items()} }",
          flush=True)
    # immediate sanity: reconstruction on 32 holdout images
    _, _, ds_ho, _ = load_data()
    import lpips
    lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    sd = StudentDecoder(official.post_quant_conv, dec.to(dev)).to(dev).eval()
    b = [ds_ho[int(i)] for i in np.random.default_rng(3).choice(len(ds_ho), 32,
                                                                 replace=False)]
    xb = torch.stack([r["x"] for r in b]).to(dev)
    zb = torch.stack([r["z"] for r in b]).to(dev)
    with torch.no_grad():
        y = sd(zb).float()
    print(f"[Cold] zero-step L1={float((y-xb).abs().mean()):.5f} "
          f"LPIPS={float(lp(y.clamp(-1,1), xb).mean()):.5f} "
          f"(must be far below the 0.808 index-mismatch result)", flush=True)


if __name__ == "__main__":
    main()
