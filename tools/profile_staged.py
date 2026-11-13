"""Local staged inference profile on this machine's GPU (RTX 4070 Laptop 8GB).

Segments (paper's `generation_only_timing_summary.json` anatomy):
    encoder : official SD0 VAE encode(x).latent_dist.mean
    weave   : LGTInference.generation(z, target_style_id, num_steps=8)   [paper: 8-step Euler]
    decoder : one of {official SD0, old A0-S10, new C0, new C1}

Timing uses torch.cuda.Event, warmup then `iters` measured iterations, and
reports ms/batch, ms/img, and each stage's share of the total.

Paper baseline for reference (NOT comparable in absolute ms):
    RTX 3060 12GB, ONNX FP16 encoder + TRT FP16 decoder, batch 8, 750 imgs
    total 97.3 ms/img = encode 37.2 + weave 11.3 + decode 78.6 (ledger ablation)
    i.e. shares 38.2% / 11.6% / 80.8% are NOT additive in the ledger (the
    ledger's "encoder+WEAVE 18.7 ms/img" is a fused stage); we report OUR OWN
    shares on OUR hardware, and compare the RATIO structure only.

Usage:
    python tools/profile_staged.py --batch 1 --iters 20
    python tools/profile_staged.py --batch 8 --iters 10
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

SD0 = ROOT / "deliver/checkpoints/sd0_vae"
OLD_S10 = ROOT / "deliver/checkpoints/runs/vae_decoder_prune/s10_p5_lowlr/decoder.pt"
NEW_DIR = ROOT / "deliver/checkpoints/runs/sd0_s10"
WEAVE_CKPT = ROOT / "exp/dino_s_break/brk_a_ll03_10ep/epoch_0010.pt"
LEDGER = ROOT / "docs/model_probe/generation_only_timing_summary.json"


def cuda_event_ms(fn, iters, warmup=5):
    """Return mean ms per call, measured with CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    total_ms = sum(s.elapsed_time(e) for s, e in zip(starts, ends))
    return total_ms / iters, (time.perf_counter() - t0) * 1000 / iters


def build_decoders(dev):
    from diffusers import AutoencoderKL
    from tools.vae_suite.decoder_prune import StudentDecoder, build_decoder

    official = AutoencoderKL.from_pretrained(
        str(SD0), torch_dtype=torch.float32, local_files_only=True).to(dev).eval()
    for p in official.parameters():
        p.requires_grad_(False)
    variants = {"official": lambda z: official.decode(z).sample}
    for name, f in (("old_s10", OLD_S10),
                    ("c0", NEW_DIR / "sd0_s10_best_recon.pt"),
                    ("c1", NEW_DIR / "sd0_s10_best_overall.pt")):
        if not f.exists():
            print(f"[skip] {name}: {f} missing", flush=True)
            continue
        ck = torch.load(f, map_location="cpu", weights_only=False)
        d = build_decoder([64, 128, 192, 256])
        d.load_state_dict(ck["state_dict"])
        pq = torch.nn.Conv2d(4, 4, 1).to(dev)
        pq.load_state_dict(ck["post_quant_conv"])
        sd = StudentDecoder(pq, d.to(dev)).to(dev).eval()
        variants[name] = lambda z, _sd=sd: _sd(z)
    return official, variants


def build_weave(dev, num_steps):
    from tools.legacy_utils.inference import LGTInference
    inf = LGTInference(str(WEAVE_CKPT), device=dev, num_steps=num_steps,
                       step_size=1.0, style_strength=1.0)
    inf.model.eval()
    for p in inf.model.parameters():
        p.requires_grad_(False)
    return inf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--num-steps", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    assert torch.cuda.is_available(), "CUDA required"

    print(f"[profile] device={torch.cuda.get_device_name(0)} "
          f"mem={torch.cuda.get_device_properties(0).total_memory/2**30:.1f}GB "
          f"batch={args.batch} iters={args.iters} num_steps={args.num_steps}",
          flush=True)

    official, decoders = build_decoders(dev)
    weave = build_weave(dev, args.num_steps)
    B = args.batch
    x = torch.randn(B, 3, 512, 512, device=dev)
    tgt = torch.zeros(B, dtype=torch.long, device=dev)

    # ---- stage 1: encoder ----
    @torch.no_grad()
    def enc():
        return official.encode(x).latent_dist.mean
    z = enc()
    enc_ms, _ = cuda_event_ms(enc, args.iters, args.warmup)
    print(f"  encoder   {enc_ms:8.2f} ms/batch  ({enc_ms/B:7.2f} ms/img)", flush=True)

    # ---- stage 2: WEAVE 8-step ----
    @torch.no_grad()
    def wv():
        return weave.generation(z, tgt, num_steps=args.num_steps)
    zgen = wv()
    wv_ms, _ = cuda_event_ms(wv, args.iters, args.warmup)
    print(f"  weave({args.num_steps}) {wv_ms:8.2f} ms/batch  ({wv_ms/B:7.2f} ms/img)",
          flush=True)

    # ---- stage 3: decoder variants ----
    results = {}
    for name, dec in decoders.items():
        @torch.no_grad()
        def df(_dec=dec):
            return _dec(zgen)
        try:
            d_ms, _ = cuda_event_ms(df, args.iters, args.warmup)
        except torch.cuda.OutOfMemoryError:
            print(f"  decoder {name:<9} OOM at batch {B}", flush=True)
            torch.cuda.empty_cache()
            continue
        total = enc_ms + wv_ms + d_ms
        results[name] = {
            "encoder_ms_per_batch": round(enc_ms, 3),
            "weave_ms_per_batch": round(wv_ms, 3),
            "decoder_ms_per_batch": round(d_ms, 3),
            "total_ms_per_batch": round(total, 3),
            "ms_per_img": round(total / B, 3),
            "encoder_share": round(enc_ms / total, 4),
            "weave_share": round(wv_ms / total, 4),
            "decoder_share": round(d_ms / total, 4),
            "speedup_vs_official_decoder": None,
        }
        print(f"  decoder {name:<9} {d_ms:8.2f} ms/batch  ({d_ms/B:7.2f} ms/img) "
              f"| total {total:8.2f} ({total/B:7.2f}/img) "
              f"| share E/W/D = {enc_ms/total:.1%}/{wv_ms/total:.1%}/{d_ms/total:.1%}",
              flush=True)
        torch.cuda.empty_cache()

    if "official" in results:
        base = results["official"]["decoder_ms_per_batch"]
        for k, v in results.items():
            v["speedup_vs_official_decoder"] = round(base / v["decoder_ms_per_batch"], 3)

    # ---- paper ledger (reference only, NOT comparable absolute ms) ----
    ledger = json.loads(LEDGER.read_text(encoding="utf-8")) if LEDGER.exists() else None
    out = {
        "hardware": torch.cuda.get_device_name(0),
        "gpu_mem_GB": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
        "torch": torch.__version__,
        "batch": B, "iters": args.iters, "warmup": args.warmup,
        "num_steps": args.num_steps,
        "backend": "PyTorch FP32, torch.cuda.Event timing",
        "weave_ckpt": str(WEAVE_CKPT.relative_to(ROOT)),
        "weave_ckpt_note": ("paper main-table ckpt is epoch_0004 (NOT present locally); "
                            "epoch_0010 from the same training source is used. "
                            "Architecture/8-step compute is identical, so timing is "
                            "representative; weight values do not affect speed."),
        "results": results,
        "paper_ledger_reference_only": ledger,
        "comparability_note": (
            "Paper: RTX 3060 12GB, ONNX FP16 encoder + TRT FP16 decoder, batch 8, "
            "generation-only end-to-end 97.3 ms/img. This profile: local 4070 Laptop "
            "8GB, PyTorch FP32, configurable batch. Absolute ms are NOT comparable; "
            "only the stage-share structure and decoder-relative speedups are."),
    }
    outp = ROOT / f"deliver/staged_profile_b{B}.json"
    outp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[Saved] {outp}", flush=True)


if __name__ == "__main__":
    main()
