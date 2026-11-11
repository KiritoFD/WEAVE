"""P0 zero-step baseline audit on OFFICIAL SD0 latents.

Compares three initialisation candidates WITHOUT any training:
  W       = Warm-A0 s10_p5_lowlr + A0 post_quant_conv
  C       = Cold-SD0 (official decoder pruned with the ORIGINAL S10 indices)
  W-PQC   = W's S10 weights + official SD0 post_quant_conv   (interface probe)
  official= the 49.49M official decoder (reference)

W-PQC is inference-only: if it degrades badly it is dropped, no training.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/root/Workspace/WEAVE_VAE")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "12h"))

from tools.vae_suite.eval_sd0_s10 import (csd_keep, recon_metrics, ssim_simple)
from tools.vae_suite.train_sd0_s10 import load_data

OUT = ROOT / "runs/sd0_s10"


def main():
    dev = "cuda"
    sp, ds_tr, ds_ho, ds_te = load_data()
    from tools.vae_suite.eval_sd0_s10 import build_variant
    from diffusers import AutoencoderKL
    from tools.vae_suite.decoder_prune import StudentDecoder, build_decoder
    from tools.vae_suite.sd0_losses import prune_with_indices

    SD0 = "/root/Workspace/xy/DiT/data/pretrained/sd-vae-ft-ema"
    official = AutoencoderKL.from_pretrained(SD0, torch_dtype=torch.float32).to(dev).eval()
    import lpips
    lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    import csd_model
    csd, _, _ = csd_model.load_csd(dev); csd = csd.float().eval()

    idx = torch.load(ROOT / "runs/vae_decoder_prune/s10_p1_l1/decoder.pt",
                     map_location="cpu", weights_only=False)["meta"]["channel_index"]
    wck = torch.load(ROOT / "runs/vae_decoder_prune/s10_p5_lowlr/decoder.pt",
                     map_location="cpu", weights_only=False)

    variants = {}
    variants["official"] = lambda z: official.decoder(official.post_quant_conv(z))

    # W
    dW = build_decoder([64, 128, 192, 256]).to(dev)
    dW.load_state_dict(wck["state_dict"])
    pqW = torch.nn.Conv2d(4, 4, 1).to(dev)
    pqW.load_state_dict(wck["post_quant_conv"])
    variants["W"] = StudentDecoder(pqW, dW).to(dev).eval()

    # W-PQC
    pqS = torch.nn.Conv2d(4, 4, 1).to(dev)
    pqS.load_state_dict({k: v.clone() for k, v in
                         official.post_quant_conv.state_dict().items()})
    variants["W-PQC"] = StudentDecoder(pqS, dW).to(dev).eval()

    # C: Cold-SD0 built by build_cold_sd0.py -- Taylor importance recomputed
    # on the OFFICIAL SD0 decoder, then prune+inherit. Reusing A0's index
    # NUMBERS was tried and is wrong; the resulting 0.845 LPIPS is the normal
    # UNTRAINED post-pruning state (A0's own untrained Taylor prune measures
    # 0.848 on these same SD0 latents), i.e. exactly what P1 must recover from.
    cold_path = OUT / "cold_init.pt"
    if not cold_path.exists():
        raise FileNotFoundError("run build_cold_sd0.py before p0_baseline.py")
    cck = torch.load(cold_path, map_location="cpu", weights_only=False)
    dC = build_decoder([64, 128, 192, 256])
    dC.load_state_dict(cck["state_dict"])
    pqC = torch.nn.Conv2d(4, 4, 1).to(dev)
    pqC.load_state_dict(cck["post_quant_conv"])
    variants["C"] = StudentDecoder(pqC, dC.to(dev)).to(dev).eval()

    res = {}
    print("=" * 96)
    print("P0 ZERO-STEP BASELINE (official SD0 latents, frozen holdout)")
    print("=" * 96, flush=True)
    for name, dec in variants.items():
        v = recon_metrics(dec, ds_ho, dev, lp, csd, n=256, tag="holdout")
        res[name] = v
        print(f"  {name:<10}{json.dumps(v)}", flush=True)

    # W-PQC gate: inference-only; if clearly worse than W, drop it
    wpqc_ok = (res["W-PQC"]["lpips"] <= res["W"]["lpips"] + 0.02 and
               res["W-PQC"]["l1"] <= res["W"]["l1"] + 0.02)
    out = {
        "variants": res,
        "W_PQC_keep": bool(wpqc_ok),
        "decision": ("W-PQC dropped (no full training)" if not wpqc_ok else
                     "W-PQC comparable; keep for reference only"),
        "cold_vs_warm_lpips_delta": round(res["C"]["lpips"] - res["W"]["lpips"], 5),
        "cold_note": ("Cold is the UNTRAINED Taylor-pruned official SD0 decoder; "
                      "its large zero-step LPIPS is the expected post-pruning "
                      "state (A0's identical untrained prune measures 0.848 on "
                      "these SD0 latents). P1 exists to recover it."),
        "encoder_hash": sp["encoder_hash"],
    }
    (OUT / "p0_baseline.json").write_text(json.dumps(out, indent=2),
                                          encoding="utf-8")
    print(f"\n  W-PQC keep = {wpqc_ok} | Cold-Warm LPIPS delta = "
          f"{out['cold_vs_warm_lpips_delta']:+.5f}", flush=True)
    print(f"[Saved] {OUT / 'p0_baseline.json'}", flush=True)


if __name__ == "__main__":
    main()
