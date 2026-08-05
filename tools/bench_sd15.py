"""SD1.5 pipeline stage-by-stage timing benchmark."""
import time, sys, json, gc
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEVICE = "cuda"
N = 50
B = 8

def sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()

# 1. Load VAE
from utils.inference import load_vae
print("Loading SD1.5 VAE..."); t0 = time.perf_counter()
vae = load_vae(device=DEVICE, model_id="ema", enable_xformers=False)
vae = vae.to(device=DEVICE, dtype=torch.float16)
print(f"  VAE loaded in {time.perf_counter()-t0:.1f}s")

# 2. Load WEAVE
from utils.inference import LGTInference
ckpt = str(ROOT / "runs/submission/canonical_oriented_epoch4/epoch_0004.pt")
print("Loading WEAVE..."); t0 = time.perf_counter()
inf = LGTInference(ckpt, device=DEVICE, num_steps=1)
inf.model.eval(); inf.model.solver_type = "euler"
print(f"  WEAVE loaded in {time.perf_counter()-t0:.1f}s  params={sum(p.numel() for p in inf.model.parameters()):,}")

# 3. Dummy images
imgs = torch.randn(N, 3, 512, 512)
style_ids = torch.arange(5, device=DEVICE, dtype=torch.long).repeat(B//5+1)[:B]

# Warmup
print("\nWarmup...")
for _ in range(3):
    b = imgs[:B].to(DEVICE, dtype=torch.float16)
    z = vae.encode(b).latent_dist.sample() * 0.18215
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _ = inf.model.integrate(z, style_id=style_ids, num_steps=1)
    _ = vae.decode(z / 0.18215).sample
sync(); print("  done")

# === STAGE 1: VAE Encode ===
print(f"\n[Stage 1] VAE Encode ({N} images, batch={B})...")
t_enc = 0.0
for i in range(0, N, B):
    b = imgs[i:i+B].to(DEVICE, dtype=torch.float16)
    t0 = time.perf_counter()
    z = vae.encode(b).latent_dist.sample() * 0.18215
    sync()
    t_enc += time.perf_counter() - t0
    if i == 0: latents = z.float().cpu()
    else: latents = torch.cat([latents, z.float().cpu()], 0)
print(f"  {t_enc:.3f}s  {1000*t_enc/N:.1f}ms/img  {N/t_enc:.1f} img/s")

# === STAGE 2: WEAVE Latent Transfer ===
print(f"\n[Stage 2] WEAVE Latent Transfer ({N} images, batch={B})...")
t_weave = 0.0
for i in range(0, N, B):
    b_sz = min(B, N-i)
    z = latents[i:i+b_sz].to(DEVICE, dtype=torch.float32)
    s = style_ids[:b_sz]
    t0 = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        styled = inf.model.integrate(z, style_id=s, num_steps=1)
    sync()
    t_weave += time.perf_counter() - t0
    if i == 0: styled_all = styled.float().cpu()
    else: styled_all = torch.cat([styled_all, styled.float().cpu()], 0)
print(f"  {t_weave:.3f}s  {1000*t_weave/N:.1f}ms/img  {N/t_weave:.1f} img/s")

del inf; gc.collect(); torch.cuda.empty_cache()

# === STAGE 3: VAE Decode ===
print(f"\n[Stage 3] VAE Decode ({N} images, batch={B})...")
t_dec = 0.0
for i in range(0, N, B):
    b_sz = min(B, N-i)
    z = styled_all[i:i+b_sz].to(DEVICE, dtype=torch.float16)
    t0 = time.perf_counter()
    _ = vae.decode(z / 0.18215).sample
    sync()
    t_dec += time.perf_counter() - t0
print(f"  {t_dec:.3f}s  {1000*t_dec/N:.1f}ms/img  {N/t_dec:.1f} img/s")

total = t_enc + t_weave + t_dec
print(f"\n{'='*55}")
print(f" SUMMARY (PyTorch fp16, {N} images, batch={B})")
print(f"{'='*55}")
print(f" Stage 1: VAE Encode         {t_enc:>8.3f}s  ({1000*t_enc/N:>6.1f} ms/img)  {100*t_enc/total:>5.1f}%")
print(f" Stage 2: WEAVE Transfer     {t_weave:>8.3f}s  ({1000*t_weave/N:>6.1f} ms/img)  {100*t_weave/total:>5.1f}%")
print(f" Stage 3: VAE Decode         {t_dec:>8.3f}s  ({1000*t_dec/N:>6.1f} ms/img)  {100*t_dec/total:>5.1f}%")
print(f" {'─'*45}")
print(f" TOTAL                       {total:>8.3f}s  ({1000*total/N:>6.1f} ms/img)")

# Scale to 750
scale = 750/N
print(f"\n Projected to 750 images:")
print(f" Stage 1: VAE Encode         {t_enc*scale:>8.1f}s")
print(f" Stage 2: WEAVE Transfer     {t_weave*scale:>8.1f}s")
print(f" Stage 3: VAE Decode         {t_dec*scale:>8.1f}s")
print(f" TOTAL                       {total*scale:>8.1f}s")

out = {
    "device": torch.cuda.get_device_name(0),
    "n": N, "batch": B,
    "stage1_encode_s": round(t_enc, 3),
    "stage2_weave_s": round(t_weave, 3),
    "stage3_decode_s": round(t_dec, 3),
    "total_s": round(total, 3),
    "projected_750": {
        "encode_s": round(t_enc*scale, 1),
        "weave_s": round(t_weave*scale, 1),
        "decode_s": round(t_dec*scale, 1),
        "total_s": round(total*scale, 1),
    }
}
Path(ROOT / "runs/taesd/eval/sd15_pipeline_timing.json").write_text(json.dumps(out, indent=2))
print(f"\nSaved.")