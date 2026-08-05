"""
Comprehensive timing benchmark for WEAVE pipeline.
Measures every stage separately for both SD1.5 VAE and TAESD VAE paths.
"""
import argparse, time, sys, json, gc
from pathlib import Path
import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEVICE = "cuda"
N_WARMUP = 3
N_BENCH = 50  # benchmark on 50 images for speed

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def load_vae_sd15():
    from utils.inference import load_vae
    return load_vae(device=DEVICE, model_id="ema", enable_xformers=False)

def load_vae_taesd():
    from diffusers import AutoencoderTiny
    path = ROOT / "data" / "taesd"
    vae = AutoencoderTiny.from_pretrained(str(path), local_files_only=True)
    vae = vae.to(device=DEVICE, dtype=torch.float16).eval()
    vae.requires_grad_(False)
    return vae

def load_weave_model(ckpt_path, config_override=None):
    from utils.inference import LGTInference
    inf = LGTInference(str(ckpt_path), device=DEVICE, num_steps=1, config_override_path=config_override)
    inf.model.eval()
    inf.model.solver_type = "euler"
    return inf

def benchmark_encode(vae, vae_name, images, batch_size):
    """Benchmark VAE encode stage."""
    n = len(images)
    # Warmup
    for _ in range(N_WARMUP):
        _ = vae.encode(images[:batch_size].to(DEVICE, dtype=torch.float16))
    sync()
    
    times = []
    for i in range(0, n, batch_size):
        batch = images[i:i+batch_size].to(DEVICE, dtype=torch.float16)
        t0 = time.perf_counter()
        out = vae.encode(batch)
        if hasattr(out, 'latents'):
            lat = out.latents
        else:
            lat = out.latent_dist.sample()
        sync()
        times.append(time.perf_counter() - t0)
    
    return {
        "stage": f"VAE Encode ({vae_name})",
        "total_s": sum(times),
        "ms_per_img": 1000 * sum(times) / n,
        "imgs_per_s": n / sum(times),
        "batch_size": batch_size,
    }

def benchmark_decode(vae, vae_name, latents, batch_size, scaling_factor):
    """Benchmark VAE decode stage."""
    n = len(latents)
    # Warmup
    for _ in range(N_WARMUP):
        _ = vae.decode(latents[:batch_size].to(DEVICE, dtype=torch.float16) / scaling_factor).sample
    sync()
    
    times = []
    for i in range(0, n, batch_size):
        batch = latents[i:i+batch_size].to(DEVICE, dtype=torch.float16)
        t0 = time.perf_counter()
        _ = vae.decode(batch / scaling_factor).sample
        sync()
        times.append(time.perf_counter() - t0)
    
    return {
        "stage": f"VAE Decode ({vae_name})",
        "total_s": sum(times),
        "ms_per_img": 1000 * sum(times) / n,
        "imgs_per_s": n / sum(times),
        "batch_size": batch_size,
    }

def benchmark_weave_latent_gen(inf, latents, batch_size):
    """Benchmark WEAVE latent style transfer."""
    n = len(latents)
    style_ids = torch.arange(5, device=DEVICE, dtype=torch.long).repeat(batch_size // 5 + 1)[:batch_size]
    
    # Warmup
    for _ in range(N_WARMUP):
        z = latents[:batch_size].to(DEVICE, dtype=torch.float32)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _ = inf.model.integrate(z, style_id=style_ids, num_steps=1)
    sync()
    
    times = []
    for i in range(0, n, batch_size):
        b_size = min(batch_size, n - i)
        z = latents[i:i+b_size].to(DEVICE, dtype=torch.float32)
        s_ids = style_ids[:b_size]
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _ = inf.model.integrate(z, style_id=s_ids, num_steps=1)
        sync()
        times.append(time.perf_counter() - t0)
    
    return {
        "stage": "WEAVE Latent Transfer",
        "total_s": sum(times),
        "ms_per_img": 1000 * sum(times) / n,
        "imgs_per_s": n / sum(times),
        "batch_size": batch_size,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "runs/submission/canonical_oriented_epoch4/epoch_0004.pt"))
    ap.add_argument("--taesd_checkpoint", default=str(ROOT / "runs/taesd/brk_a_15ep/epoch_0015.pt"))
    ap.add_argument("--n", type=int, default=N_BENCH)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--output", default=str(ROOT / "runs/taesd/eval/benchmark_timing.json"))
    args = ap.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}  VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")
    print(f"Benchmark: {args.n} images, batch={args.batch}\n")

    # Create dummy images (512x512 RGB, normalized to [-1,1])
    print("Creating dummy images...")
    dummy_images = torch.randn(args.n, 3, 512, 512, dtype=torch.float32)
    
    results = {
        "device": torch.cuda.get_device_name(0),
        "vram_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        "n_images": args.n,
        "batch_size": args.batch,
        "timings": []
    }

    # =====================================================
    # PART 1: SD1.5 VAE Pipeline
    # =====================================================
    print("=" * 60)
    print("PART 1: SD1.5 VAE Pipeline")
    print("=" * 60)
    
    # Load SD1.5 VAE
    print("\nLoading SD1.5 VAE...")
    vae_sd15 = load_vae_sd15()
    vae_sd15 = vae_sd15.to(device=DEVICE, dtype=torch.float16)
    
    # 1a. SD1.5 Encode
    print("\n[1a] SD1.5 VAE Encode...")
    r = benchmark_encode(vae_sd15, "SD1.5", dummy_images, args.batch)
    results["timings"].append(r)
    print(f"  {r['total_s']:.3f}s  {r['ms_per_img']:.1f}ms/img  {r['imgs_per_s']:.1f} img/s")
    
    # Get SD1.5 latents
    print("  Computing SD1.5 latents for downstream...")
    with torch.inference_mode():
        sd15_latents = []
        for i in range(0, args.n, args.batch):
            batch = dummy_images[i:i+args.batch].to(DEVICE, dtype=torch.float16)
            out = vae_sd15.encode(batch).latent_dist.sample()
            sd15_latents.append(out.float().cpu())
    sd15_latents = torch.cat(sd15_latents, dim=0)
    print(f"  Latents shape: {tuple(sd15_latents.shape)}")
    
    # 1b. WEAVE Latent Transfer (SD1.5 latents)
    print("\n[1b] WEAVE Latent Transfer (SD1.5 latents)...")
    inf_sd15 = load_weave_model(args.checkpoint)
    r = benchmark_weave_latent_gen(inf_sd15, sd15_latents, args.batch)
    results["timings"].append(r)
    print(f"  {r['total_s']:.3f}s  {r['ms_per_img']:.1f}ms/img  {r['imgs_per_s']:.1f} img/s")
    
    # Get styled latents
    print("  Computing styled latents for decode...")
    style_ids = torch.arange(5, device=DEVICE, dtype=torch.long).repeat(args.batch // 5 + 1)[:args.batch]
    styled_sd15 = []
    with torch.inference_mode():
        for i in range(0, args.n, args.batch):
            b_size = min(args.batch, args.n - i)
            z = sd15_latents[i:i+b_size].to(DEVICE, dtype=torch.float32)
            s_ids = style_ids[:b_size]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = inf_sd15.model.integrate(z, style_id=s_ids, num_steps=1)
            styled_sd15.append(out.float().cpu())
    styled_sd15 = torch.cat(styled_sd15, dim=0)
    del inf_sd15; gc.collect(); torch.cuda.empty_cache()
    
    # 1c. SD1.5 Decode
    print("\n[1c] SD1.5 VAE Decode...")
    r = benchmark_decode(vae_sd15, "SD1.5", styled_sd15, args.batch, 0.18215)
    results["timings"].append(r)
    print(f"  {r['total_s']:.3f}s  {r['ms_per_img']:.1f}ms/img  {r['imgs_per_s']:.1f} img/s")
    
    # SD1.5 totals
    sd15_encode = results["timings"][-3]["total_s"]
    sd15_weave = results["timings"][-2]["total_s"]
    sd15_decode = results["timings"][-1]["total_s"]
    sd15_total = sd15_encode + sd15_weave + sd15_decode
    results["timings"].append({
        "stage": "SD1.5 Pipeline TOTAL",
        "total_s": sd15_total,
        "ms_per_img": 1000 * sd15_total / args.n,
        "imgs_per_s": args.n / sd15_total,
        "breakdown": f"encode={sd15_encode:.2f}s weave={sd15_weave:.2f}s decode={sd15_decode:.2f}s"
    })
    print(f"\n  SD1.5 TOTAL: {sd15_total:.3f}s ({sd15_encode:.2f}+{sd15_weave:.2f}+{sd15_decode:.2f})")
    
    del vae_sd15, styled_sd15, sd15_latents
    gc.collect(); torch.cuda.empty_cache()
    
    # =====================================================
    # PART 2: TAESD VAE Pipeline
    # =====================================================
    print("\n" + "=" * 60)
    print("PART 2: TAESD VAE Pipeline")
    print("=" * 60)
    
    # Load TAESD VAE
    print("\nLoading TAESD VAE...")
    vae_taesd = load_vae_taesd()
    
    # 2a. TAESD Encode
    print("\n[2a] TAESD VAE Encode...")
    r = benchmark_encode(vae_taesd, "TAESD", dummy_images, args.batch)
    results["timings"].append(r)
    print(f"  {r['total_s']:.3f}s  {r['ms_per_img']:.1f}ms/img  {r['imgs_per_s']:.1f} img/s")
    
    # Get TAESD latents
    print("  Computing TAESD latents...")
    with torch.inference_mode():
        taesd_latents = []
        for i in range(0, args.n, args.batch):
            batch = dummy_images[i:i+args.batch].to(DEVICE, dtype=torch.float16)
            out = vae_taesd.encode(batch).latents
            taesd_latents.append(out.float().cpu())
    taesd_latents = torch.cat(taesd_latents, dim=0)
    
    # 2b. WEAVE Latent Transfer (TAESD latents, TAESD-trained model)
    print("\n[2b] WEAVE Latent Transfer (TAESD latents)...")
    import tempfile, json
    override_dict = {"model": {"solver_family": "euler_legacy", "latent_scale_factor": 1.0}}
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as f:
        json.dump(override_dict, f)
        override_path = f.name
    inf_taesd = load_weave_model(args.taesd_checkpoint, override_path)
    r = benchmark_weave_latent_gen(inf_taesd, taesd_latents, args.batch)
    results["timings"].append(r)
    print(f"  {r['total_s']:.3f}s  {r['ms_per_img']:.1f}ms/img  {r['imgs_per_s']:.1f} img/s")
    
    # Get styled latents
    styled_taesd = []
    with torch.inference_mode():
        for i in range(0, args.n, args.batch):
            b_size = min(args.batch, args.n - i)
            z = taesd_latents[i:i+b_size].to(DEVICE, dtype=torch.float32)
            s_ids = style_ids[:b_size]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = inf_taesd.model.integrate(z, style_id=s_ids, num_steps=1)
            styled_taesd.append(out.float().cpu())
    styled_taesd = torch.cat(styled_taesd, dim=0)
    del inf_taesd; gc.collect(); torch.cuda.empty_cache()
    
    # 2c. TAESD Decode
    print("\n[2c] TAESD VAE Decode...")
    r = benchmark_decode(vae_taesd, "TAESD", styled_taesd, args.batch, 1.0)
    results["timings"].append(r)
    print(f"  {r['total_s']:.3f}s  {r['ms_per_img']:.1f}ms/img  {r['imgs_per_s']:.1f} img/s")
    
    # TAESD totals
    taesd_encode = results["timings"][-3]["total_s"]
    taesd_weave = results["timings"][-2]["total_s"]
    taesd_decode = results["timings"][-1]["total_s"]
    taesd_total = taesd_encode + taesd_weave + taesd_decode
    results["timings"].append({
        "stage": "TAESD Pipeline TOTAL",
        "total_s": taesd_total,
        "ms_per_img": 1000 * taesd_total / args.n,
        "imgs_per_s": args.n / taesd_total,
        "breakdown": f"encode={taesd_encode:.2f}s weave={taesd_weave:.2f}s decode={taesd_decode:.2f}s"
    })
    print(f"\n  TAESD TOTAL: {taesd_total:.3f}s ({taesd_encode:.2f}+{taesd_weave:.2f}+{taesd_decode:.2f})")
    
    # =====================================================
    # SUMMARY
    # =====================================================
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Stage':<25} {'SD1.5':>12} {'TAESD':>12} {'Speedup':>10}")
    print("-" * 60)
    for stage_name, sd15_idx, taesd_idx in [
        ("VAE Encode", 0, 7),
        ("WEAVE Transfer", 1, 8),
        ("VAE Decode", 2, 9),
        ("TOTAL", 3, 10),
    ]:
        sd15_t = results["timings"][sd15_idx]["total_s"]
        taesd_t = results["timings"][taesd_idx]["total_s"]
        sp = sd15_t / taesd_t if taesd_t > 0 else 0
        print(f"{stage_name:<25} {sd15_t:>8.3f}s {taesd_t:>8.3f}s {sp:>8.1f}x")
    
    results["comparison"] = {
        "sd15_total_s": sd15_total,
        "taesd_total_s": taesd_total,
        "speedup": sd15_total / taesd_total,
    }
    
    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()