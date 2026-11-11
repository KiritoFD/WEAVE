"""Staged generation-only profile on the paper's own RTX 3060 12GB.

Reuses the SAME hardware / VAE loader / WEAVE ckpt / batch-8 as
docs/model_probe/generation_only_timing_summary.json (97.3 ms/img), then swaps
ONLY the decoder. Stages:

    image --E--> z --WEAVE(8-step Euler)--> z' --D--> image

Variants (all share the SAME encoder + WEAVE outputs):
    official  : sd-vae-ft-ema 49.49M decoder   (paper baseline decoder)
    old_s10   : A0-trained S10 10.66M decoder  (ranking #4's decoder)
    c0        : new SD0-S10 best_recon
    c1        : new SD0-S10 best_overall

Timing: torch.cuda.Event, warmup then N iterations of batch B, report
ms/batch, ms/img, and each stage's share. Also records peak VRAM.

Run on 3060 (Windows python):
    python tools\\profile_staged_3060.py --batch 8 --iters 20
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LEDGER = ROOT / "docs/model_probe/generation_only_timing_summary.json"
WEAVE_CKPT = ROOT / "archive/runs_submission/canonical_oriented_epoch4/epoch_0004.pt"
PROFILE_DIR = ROOT / "deliver/sd0_s10_profile"
OLD_S10 = ROOT / "runs/vae_decoder_prune/s10_p5_lowlr/decoder.pt"
C0 = PROFILE_DIR / "sd0_s10_best_recon.pt"
C1 = PROFILE_DIR / "sd0_s10_best_overall.pt"
SD0_DIR = PROFILE_DIR                  # config.json + safetensors uploaded flat

STYLES = 5


def sync():
    torch.cuda.synchronize()


def timed(fn, iters, warmup=5):
    for _ in range(warmup):
        fn()
    sync()
    ss = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ee = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        ss[i].record()
        fn()
        ee[i].record()
    sync()
    return sum(a.elapsed_time(b) for a, b in zip(ss, ee)) / iters


def load_official_vae(dev):
    """Prefer the exact sd-vae-ft-ema dir we uploaded; else repo load_vae."""
    if (SD0_DIR / "config.json").exists() and (SD0_DIR / "diffusion_pytorch_model.safetensors").exists():
        from diffusers import AutoencoderKL
        return AutoencoderKL.from_pretrained(
            str(SD0_DIR), torch_dtype=torch.float16, local_files_only=True
        ).to(dev).eval()
    from utils.inference import load_vae
    return load_vae(device=dev, model_id="ema", enable_xformers=False)


def build_decoders(vae, dev):
    out = {"official": lambda z: vae.decode(z).sample}
    try:
        from tools.vae_suite.decoder_prune import StudentDecoder, build_decoder
    except Exception as e:
        print(f"[warn] decoder_prune unavailable: {e}", flush=True)
        return out
    for name, f in (("old_s10", OLD_S10), ("c0", C0), ("c1", C1)):
        if not f.exists():
            print(f"[skip] {name}: {f} missing", flush=True)
            continue
        ck = torch.load(str(f), map_location="cpu", weights_only=False)
        d = build_decoder([64, 128, 192, 256]).to(dev).eval()
        d.load_state_dict(ck["state_dict"])
        pq = torch.nn.Conv2d(4, 4, 1).to(dev).eval()
        pq.load_state_dict(ck["post_quant_conv"])
        sd = StudentDecoder(pq, d).to(dev).half().eval()
        out[name] = lambda z, _sd=sd: _sd(z)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--num-steps", type=int, default=8)
    args = ap.parse_args()
    dev = "cuda"
    B = args.batch

    print(f"[profile] {torch.cuda.get_device_name(0)} "
          f"{torch.cuda.get_device_properties(0).total_memory/2**30:.1f}GB "
          f"torch={torch.__version__} batch={B} iters={args.iters} "
          f"num_steps={args.num_steps}", flush=True)

    vae = load_official_vae(dev)
    for p in vae.parameters():
        p.requires_grad_(False)
    sf = float(getattr(vae.config, "scaling_factor", 0.18215))
    decoders = build_decoders(vae, dev)

    from utils.inference import LGTInference
    inf = LGTInference(str(WEAVE_CKPT), device=dev, num_steps=args.num_steps,
                       step_size=1.0, style_strength=1.0)
    inf.model.eval()
    inf.model.solver_type = "euler"
    for p in inf.model.parameters():
        p.requires_grad_(False)

    # dummy 512px image batch in [-1,1]
    x = torch.randn(B, 3, 512, 512, device=dev, dtype=torch.float16)
    style_ids = torch.arange(STYLES, device=dev, dtype=torch.long).repeat(
        B // STYLES + 1)[:B]

    # ---- stage 1: encoder ----
    @torch.no_grad()
    def enc():
        return vae.encode(x).latent_dist.mean
    z = enc()
    enc_ms = timed(enc, args.iters, args.warmup)
    print(f"  encoder     {enc_ms:8.2f} ms/batch ({enc_ms/B:7.2f} ms/img)", flush=True)

    # ---- stage 2: WEAVE 8-step (paper uses bf16 autocast) ----
    @torch.no_grad()
    def wv():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return inf.model.integrate(z.float(), style_id=style_ids,
                                       num_steps=args.num_steps)
    zgen = wv()
    wv_ms = timed(wv, args.iters, args.warmup)
    print(f"  weave({args.num_steps})  {wv_ms:8.2f} ms/batch ({wv_ms/B:7.2f} ms/img)",
          flush=True)
    zgen = zgen.half()

    # ---- stage 3: decoder ----
    results = {}
    for name, dec in decoders.items():
        @torch.no_grad()
        def df(_d=dec):
            return _d(zgen)
        try:
            torch.cuda.reset_peak_memory_stats()
            d_ms = timed(df, args.iters, args.warmup)
            peak = torch.cuda.max_memory_allocated() / 2**30
        except Exception as e:
            print(f"  decoder {name:<9} FAILED: {type(e).__name__}: {e}", flush=True)
            torch.cuda.empty_cache()
            continue
        total = enc_ms + wv_ms + d_ms
        results[name] = {
            "encoder_ms_batch": round(enc_ms, 3),
            "weave_ms_batch": round(wv_ms, 3),
            "decoder_ms_batch": round(d_ms, 3),
            "total_ms_batch": round(total, 3),
            "ms_per_img": round(total / B, 3),
            "encoder_share": round(enc_ms / total, 4),
            "weave_share": round(wv_ms / total, 4),
            "decoder_share": round(d_ms / total, 4),
            "decoder_peak_GB": round(peak, 2),
        }
        print(f"  decoder {name:<9} {d_ms:8.2f} ms/batch ({d_ms/B:7.2f} ms/img) "
              f"| total {total:8.2f} ({total/B:7.2f}/img) "
              f"| E/W/D {enc_ms/total:.1%}/{wv_ms/total:.1%}/{d_ms/total:.1%}",
              flush=True)
        torch.cuda.empty_cache()

    if "official" in results:
        base = results["official"]["decoder_ms_batch"]
        for k, v in results.items():
            v["decoder_speedup_vs_official"] = round(base / v["decoder_ms_batch"], 3)
        btot = results["official"]["total_ms_batch"]
        for k, v in results.items():
            v["total_speedup_vs_official"] = round(btot / v["total_ms_batch"], 3)

    ledger = json.loads(LEDGER.read_text(encoding="utf-8")) if LEDGER.exists() else None
    out = {
        "hardware": torch.cuda.get_device_name(0),
        "gpu_mem_GB": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
        "torch": torch.__version__, "batch": B, "iters": args.iters,
        "warmup": args.warmup, "num_steps": args.num_steps,
        "backend": "PyTorch, encoder/decoder FP16, WEAVE bf16 autocast, cuda.Event",
        "weave_ckpt": str(WEAVE_CKPT.relative_to(ROOT)),
        "encoder": "sd-vae-ft-ema (official), FIXED across all variants",
        "scaling_factor": sf,
        "results": results,
        "paper_ledger": ledger,
        "paper_reference": {
            "hardware": "RTX 3060 12GB (same machine)",
            "backend": "ONNX FP16 encoder + TRT FP16 decoder, ws=2GB, batch 8",
            "total_ms_per_img": 97.3,
            "encoder_plus_weave_ms_per_img": 18.7,
            "decoder_trt_ms_per_img": 78.6,
            "note": "paper uses TRT decoder; this profile uses PyTorch decoder, "
                    "so absolute decode ms differ, but stage shares and "
                    "decoder-relative speedups are measured on the SAME GPU.",
        },
    }
    outp = PROFILE_DIR / f"staged_profile_3060_b{B}.json"
    outp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[Saved] {outp}", flush=True)


if __name__ == "__main__":
    main()
