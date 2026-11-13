"""P0b: measure the discriminative interval of dino_s (ceiling) with a frozen protocol."""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "trt"):
    sys.path.insert(0, str(_p))
import eval_fast as ef
import trt_vae_decoder
from tools.legacy_utils.inference import encode_image, load_vae, LGTInference

ap = argparse.ArgumentParser()
ap.add_argument("--test-dir", default=str(ROOT / "data" / "test"))
ap.add_argument("--checkpoint", default=str(ROOT / "exp" / "dino_s_break" / "brk_a_ll03_10ep" / "epoch_0010.pt"))
ap.add_argument("--out", default=str(ROOT / "runs" / "p0" / "ceiling.json"))
ap.add_argument("--image-size", type=int, default=512)
ap.add_argument("--cache-dir", default="/mnt/i/Latent_Style/eval_cache")
ap.add_argument("--engine-dir", default=str(ROOT / "runs" / "trt" / "engines"))
a = ap.parse_args()
dev = "cuda"

test_images, requests, style_meta = ef.build_protocol(Path(a.test_dir), max_src_samples=30, ref_pool="first")
ns = argparse.Namespace(
    disable_clip=False, disable_lpips=True, disable_dino=False, test_dir=str(a.test_dir),
    clip_model="openai/clip-vit-base-patch32", clip_local_only=True,
    clip_cache_dir=str(Path(a.cache_dir) / "hf" / "hub"), clip_skip_processor=False, amp=False,
    lpips_net="alex", lpips_chunk=32, dino_model="facebook/dinov2-small",
    dino_cache_dir=str(Path(a.cache_dir) / "hf"), allow_network=False,
    ref_cache="", max_ref_cache=16, dino_max_refs=30, image_size=a.image_size,
    metric_batch=32, ref_batch=32, only_clip_style=True,
    clip_style_idt_baseline=0.639920825263, clip_vectors=False,
)
timer = ef.Timer(sync_cuda=False)
engine = ef.MetricEngine(ns, test_images, style_meta, dev)
engine.load(timer)
engine.prepare(timer, [])  # only ref features; no sources needed

vae = load_vae(dev, model_id="ema", cache_dir=str(Path(a.cache_dir) / "hf"), compile_decoder=False).to(torch.float32)
trt = trt_vae_decoder.TrtVaeDecoder(a.engine_dir, vae, precision="fp16", device=dev)

sids = sorted(style_meta)
names = {sid: style_meta[sid]["name"] for sid in sids}
refs = {sid: style_meta[sid]["all"][:30] for sid in sids}
dino_cls = {sid: engine.ref_dino_cls[sid].float() for sid in sids}


def self_excluded_max(cls: torch.Tensor) -> np.ndarray:
    c = F.normalize(cls.float(), dim=-1)
    sim = c @ c.T
    sim.fill_diagonal_(-1e9)
    return sim.max(dim=1).values.numpy()


out = {"styles": [names[s] for s in sids]}
raw = {}
for sid in sids:
    vals = self_excluded_max(dino_cls[sid])
    raw[names[sid]] = float(np.mean(vals))
out["ceiling_raw"] = raw
out["ceiling_raw_macro"] = float(np.mean(list(raw.values())))

# VAE round-trip of the pool images: the actually reachable ceiling with this decoder
scale = float(getattr(vae.config, "scaling_factor", 0.18215))
rr_cls, rr_clip, raw_clip = {}, {}, {}
for sid in sids:
    imgs = [_load for _load in refs[sid]]
    tens = torch.stack([ef._load_eval_image_tensor(p, size=a.image_size) for p in imgs]).to(dev)
    lat = encode_image(vae, tens, dev)
    lat = lat * (0.18215 / scale)
    dec = []
    for s in range(0, lat.shape[0], 8):
        ch = lat[s:s + 8]
        if ch.shape[0] < 8:
            ch = torch.cat([ch, ch[-1:].expand(8 - ch.shape[0], -1, -1, -1)], 0)
        dec.append(trt.decode(ch)[:lat[s:s + 8].shape[0]])
    gen = torch.cat(dec, 0)
    u8 = gen.clamp(0, 1).mul(255).round().to(torch.uint8)
    arrs = [Image.fromarray(u8[i].permute(1, 2, 0).cpu().numpy()) for i in range(u8.shape[0])]
    with torch.inference_mode():
        cls_rr, _ = engine._dino_forward(arrs)
        clip_rr = engine._clip_images(u8.float().div_(255.0)).cpu()
        clip_raw = engine._clip_images(tens).cpu()
    rr_cls[sid] = cls_rr.float()
    rr_clip[sid] = clip_rr
    raw_clip[sid] = clip_raw

rr = {names[sid]: float(np.mean(self_excluded_max(rr_cls[sid]))) for sid in sids}
out["ceiling_vae_roundtrip"] = rr
out["ceiling_vae_roundtrip_macro"] = float(np.mean(list(rr.values())))

# CLIP-S ceiling (proto = normalised mean of first 16 refs, self excluded)
clip_ce_off = {}
for sid in sids:
    feats = raw_clip[sid]
    n = feats.shape[0]
    vals = []
    for i in range(n):
        idx = [j for j in range(min(16, n)) if j != i]
        if i >= 16:
            idx = list(range(min(16, n)))
        proto = F.normalize(feats[idx].mean(0, keepdim=True), dim=-1)
        vals.append(float(feats[i] @ proto[0]))
    clip_ce_off[names[sid]] = float(np.mean(vals))
out["clip_ceiling_raw"] = clip_ce_off

# Cross-style: target-style refs judged against a WRONG (source) style pool
cross = {}
for sid_t in sids:
    for sid_s in sids:
        if sid_t == sid_s:
            continue
        ct = F.normalize(dino_cls[sid_t].float(), dim=-1)
        cs = F.normalize(dino_cls[sid_s].float(), dim=-1)
        cross[f"{names[sid_t]}<-pool:{names[sid_s]}"] = float((ct @ cs.T).max(dim=1).values.mean())
out["cross_style"] = cross
out["cross_style_mean"] = float(np.mean(list(cross.values())))

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))

