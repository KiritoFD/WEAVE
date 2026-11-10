"""Pure inference and image generation across all VAE variants and model arms.

Zero evaluation metrics, zero network requests (NO DINO, NO HF calls).
Generates and writes all output images directly to disk for offline evaluation.

Variants:
  sdvae   sd-vae-ft-ema encoder + decoder
  a0      A0 fine-tuned encoder + A0 decoder
  s10     A0 fine-tuned encoder + S10 (10.66M) decoder
  cedit   C_edit StyleVAE encoder + C_edit decoder

Arms:
  Q2a     Direct baseline (Q2a_embed_style)
  Oracle1 Q2a + R1 Oracle 1-step (RMS=0.25)
  B4      Distilled lambda=50 (P1_B4_distill50)
  B3      Distilled lambda=20 (P1_B3_distill20)
  B0      Continued control (P1_B0_ctrl218)
"""
import argparse
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path("/root/Workspace/WEAVE_VAE")
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from PIL import Image

from tools.v6.canonicalize import STYLES, ContentCanonicalizer
from tools.v6.models_v6 import C3UNet
from tools.v6_next.probe_style import LatentStyleNet
from tools.vae_suite.decoder_prune import build_decoder, StudentDecoder

SDVAE = "/root/Workspace/xy/DiT/data/pretrained/sd-vae-ft-ema"
A0_DIR = str(ROOT / "runs_vae/A0/final_diffusers_vae")
S10_CK = ROOT / "runs/vae_decoder_prune/s10_p5_lowlr/decoder.pt"
CED_CK = ROOT / "runs/vae_style_vae/C_edit_best.pt"
TEST_DIR = ROOT / "deliver/testset/data/test"
TASKS_PATH = ROOT / "deliver/testset/protocol/tasks_750.json"
CANON_STATS = ROOT / "runs/v6/cache/canonicalizer_stats.pt"
OUT_ROOT = ROOT / "deliver/swap_images"

ARMS = {
    "Q2a": "deliver/checkpoints/runs/v6/Q2a_embed_style/Q2a_embed_style.pt",
    "Oracle1": "deliver/checkpoints/runs/v6/Q2a_embed_style/Q2a_embed_style.pt",
    "B4": "deliver/checkpoints/runs/v6/P1_B4_distill50/P1_B4_distill50.pt",
    "B3": "deliver/checkpoints/runs/v6/P1_B3_distill20/P1_B3_distill20.pt",
    "B0": "deliver/checkpoints/runs/v6/P1_B0_ctrl218/P1_B0_ctrl218.pt",
}
R1_CK = "deliver/checkpoints/runs/v6_next/probes/r1_r1.pt"
RMS = 0.25


def load_img(path, size=512):
    im = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.asarray(im, dtype=np.uint8)
    return torch.from_numpy(arr).permute(2, 0, 1).float().div_(127.5).sub_(1.0)


def save_image_worker(args):
    img_tensor, out_path = args
    u8 = (img_tensor.clamp(-1.0, 1.0) * 127.5 + 127.5).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    im = Image.fromarray(u8)
    im.save(out_path, compress_level=1)


def decode_chunked(dec_fn, raw_z, chunk_size=4):
    """Safely decode latents in small chunks (bs=4 max for SD-VAE/A0 on 24GB)."""
    outs = []
    with torch.inference_mode():
        for i in range(0, raw_z.shape[0], chunk_size):
            chunk = raw_z[i:i + chunk_size]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                y_chunk = dec_fn(chunk).float()
            outs.append(y_chunk.cpu())
            del y_chunk, chunk
            torch.cuda.empty_cache()
    return torch.cat(outs, dim=0)

def rank_direction(critic, z, tgt):
    """Unit-RMS direction that increases the target margin."""
    zr = z.detach().clone().requires_grad_(True)
    q = critic(zr.float()).float()
    oh = F.one_hot(tgt, len(STYLES)).bool()
    mg = (q.gather(1, tgt[:, None]).squeeze(1)
          - torch.logsumexp(q.masked_fill(oh, torch.finfo(q.dtype).min), dim=1))
    mg.sum().backward()
    g = zr.grad.detach()
    D = g[0].numel()
    gn = g.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
    return g / (gn / (D ** 0.5) + 1e-12)


def build_variant(vname, dev):
    """Instantiate paired Encoder and Decoder for the specified variant."""
    if vname == "sdvae":
        vae = AutoencoderKL.from_pretrained(SDVAE, torch_dtype=torch.float32).to(dev).eval()
        return {
            "enc": lambda x: vae.encode(x).latent_dist.mean,
            "dec": lambda z: vae.decode(z).sample,
            "sf": float(vae.config.scaling_factor),
        }
    elif vname == "a0":
        vae = AutoencoderKL.from_pretrained(A0_DIR, torch_dtype=torch.float32).to(dev).eval()
        return {
            "enc": lambda x: vae.encode(x).latent_dist.mean,
            "dec": lambda z: vae.decode(z).sample,
            "sf": float(vae.config.scaling_factor),
        }
    elif vname == "s10":
        vae = AutoencoderKL.from_pretrained(A0_DIR, torch_dtype=torch.float32).to(dev).eval()
        ck = torch.load(S10_CK, map_location="cpu", weights_only=False)
        d = build_decoder(ck["meta"]["widths_forward"])
        d.load_state_dict(ck["state_dict"])
        s10 = StudentDecoder(vae.post_quant_conv, d.to(dev)).to(dev).eval()
        return {
            "enc": lambda x: vae.encode(x).latent_dist.mean,
            "dec": lambda z: s10(z),
            "sf": float(vae.config.scaling_factor),
        }
    elif vname == "cedit":
        vae = AutoencoderKL.from_pretrained(A0_DIR, torch_dtype=torch.float32).to(dev).eval()
        ck = torch.load(CED_CK, map_location="cpu", weights_only=False)
        vae.encoder.load_state_dict(ck["encoder"])
        d = build_decoder([64, 128, 192, 256])
        d.load_state_dict(ck["decoder"])
        cedit_dec = StudentDecoder(vae.post_quant_conv, d.to(dev)).to(dev).eval()
        return {
            "enc": lambda x: vae.encode(x).latent_dist.mean,
            "dec": lambda z: cedit_dec(z),
            "sf": float(vae.config.scaling_factor),
        }
    else:
        raise ValueError(f"Unknown variant {vname}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", default="sdvae,a0,s10,cedit")
    parser.add_argument("--arms", default="Q2a,Oracle1,B4,B3,B0")
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    dev = "cuda"

    # 1. Load tasks
    spec = json.loads(TASKS_PATH.read_text(encoding="utf-8"))
    tasks = spec["tasks"]
    print(f"[Init] Loaded {len(tasks)} tasks from {TASKS_PATH}")

    # 2. Map unique sources to paths
    src_paths = {}
    for t in tasks:
        stem = t["source_stem"]
        if stem not in src_paths:
            style = t["source_style"]
            p = TEST_DIR / style / f"{stem}.jpg"
            if not p.exists():
                cands = list(TEST_DIR.glob(f"*/{stem}.jpg"))
                if cands:
                    p = cands[0]
                else:
                    raise FileNotFoundError(f"Missing test image for stem: {stem}")
            src_paths[stem] = p
    print(f"[Init] Found all {len(src_paths)} unique source images on disk")

    # 3. Load pre-computed canonicalizer stats
    if not CANON_STATS.exists():
        raise FileNotFoundError(f"Missing canonicalizer stats at {CANON_STATS}")
    st = torch.load(CANON_STATS, map_location="cpu", weights_only=False)
    canon = ContentCanonicalizer(st).to(dev).eval()
    print(f"[Init] Canonicalizer loaded from {CANON_STATS}")

    # 4. Load Critic for Oracle1
    critic = None
    if "Oracle1" in args.arms.split(","):
        r1_path = ROOT / R1_CK
        r1_ck = torch.load(r1_path, map_location="cpu", weights_only=False)
        critic = LatentStyleNet(width=r1_ck.get("width", 64)).to(dev).eval()
        critic.load_state_dict(r1_ck["model"])
        for p in critic.parameters():
            p.requires_grad = False
        print(f"[Init] Critic R1 loaded from {r1_path}")

    # 5. Load Generator Models
    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]
    generators = {}
    for name in arm_names:
        ck_path = ROOT / ARMS[name]
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        g = C3UNet(base_dim=16, num_styles=5, mode="direct", inject_16=True, style_keys=False).to(dev).eval()
        g.load_state_dict(ck["model"])
        generators[name] = g
        print(f"[Init] Generator arm '{name}' loaded ({ck['n_params']} params) from {ck_path}")

    # Threadpool for parallel PNG saving
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)

    variant_names = [v.strip() for v in args.variants.split(",") if v.strip()]
    print("\n" + "=" * 90)
    print(f"STARTING IMAGE GENERATION: {len(variant_names)} variants x {len(arm_names)} arms x {len(tasks)} tasks")
    print("=" * 90)

    total_start = time.time()
    total_images_saved = 0

    for vname in variant_names:
        v_start = time.time()
        print(f"\n>>> Running Variant: {vname}")
        v_models = build_variant(vname, dev)
        sf = v_models["sf"]
        enc_fn = v_models["enc"]
        dec_fn = v_models["dec"]

        # ---------------------------------------------------------
        # Phase 1: Encode all 150 source images + Reconstruct
        # ---------------------------------------------------------
        recon_dir = OUT_ROOT / vname / "recon"
        recon_dir.mkdir(parents=True, exist_ok=True)

        sorted_stems = sorted(src_paths.keys())
        z_src = {}
        save_jobs = []

        with torch.no_grad():
            for i in range(0, len(sorted_stems), 8):
                chunk = sorted_stems[i:i + 8]
                x_b = torch.stack([load_img(src_paths[s]) for s in chunk]).to(dev)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    z_b = enc_fn(x_b)
                y_rec = decode_chunked(dec_fn, z_b, chunk_size=8)

                # Scale latent for v6 models
                for j, s in enumerate(chunk):
                    z_src[s] = (z_b[j].float() * sf).cpu()
                    out_p = recon_dir / f"{s}.png"
                    save_jobs.append((y_rec[j], out_p))

                del x_b, z_b, y_rec
                torch.cuda.empty_cache()

        list(pool.map(save_image_worker, save_jobs))
        total_images_saved += len(save_jobs)
        print(f"  [{vname}] 150 source images encoded and reconstructed in {time.time() - v_start:.1f}s")

        # ---------------------------------------------------------
        # Phase 2: Generate swap images for each Arm
        # ---------------------------------------------------------
        for arm_name in arm_names:
            arm_start = time.time()
            arm_dir = OUT_ROOT / vname / arm_name
            arm_dir.mkdir(parents=True, exist_ok=True)
            gen = generators[arm_name]

            arm_save_jobs = []
            manifest = []

            for i in range(0, len(tasks), args.bs):
                batch_tasks = tasks[i:i + args.bs]
                zs = torch.stack([z_src[t["source_stem"]] for t in batch_tasks]).to(dev).float()
                tg = torch.tensor([t["target_style_id"] for t in batch_tasks], device=dev, dtype=torch.long)

                with torch.inference_mode():
                    c = canon(zs, mode="c1_gentle", augment=False)
                    z0 = gen(c, 0.0, c, tg)

                if arm_name == "Oracle1":
                    unit_g = rank_direction(critic, z0, tg)
                    zgen = z0 + RMS * unit_g
                else:
                    zgen = z0

                with torch.inference_mode():
                    raw = zgen / sf
                y = decode_chunked(dec_fn, raw, chunk_size=8)

                for j, t in enumerate(batch_tasks):
                    task_idx = i + j
                    stem = t["source_stem"]
                    tgt_style = t["target_style"]
                    fn = f"{task_idx:04d}_{stem}__to__{tgt_style}.png"
                    out_file = arm_dir / fn
                    arm_save_jobs.append((y[j], out_file))
                    manifest.append({
                        "task_idx": task_idx,
                        "task_id": t["task_id"],
                        "source_stem": stem,
                        "source_style": t["source_style"],
                        "target_style": tgt_style,
                        "target_style_id": t["target_style_id"],
                        "is_cross": t.get("is_cross", t["source_style"] != tgt_style),
                        "file_name": fn,
                    })

                del zs, tg, c, z0, zgen, raw, y
                torch.cuda.empty_cache()

            # Save generated images in parallel
            list(pool.map(save_image_worker, arm_save_jobs))
            total_images_saved += len(arm_save_jobs)

            # Save manifest
            (arm_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"  [{vname}] [{arm_name}] 750 images generated and saved in {time.time() - arm_start:.1f}s")

        del v_models
        torch.cuda.empty_cache()

    pool.shutdown()
    total_time = time.time() - total_start
    print("\n" + "=" * 90)
    print(f"ALL IMAGES GENERATED: {total_images_saved} images written to {OUT_ROOT} in {total_time:.1f}s ({total_time / 60:.1f} min)")
    print("=" * 90)


if __name__ == "__main__":
    main()
