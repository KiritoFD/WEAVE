"""Generate ALL 750 task images with Q2A + Oracle1 under each SD0-S10 decoder.

    x --E_SD0--> z (unscaled) --> z*sf (WEAVE space)
      -> canon(z*sf, c1_gentle) -> Q2A(c,0,c,tgt)
      -> + RMS * rank_direction(R1 critic)          [Oracle1]
      -> zgen / sf (UNSCALED) -> Decoder -> PNG

Outputs (per decoder variant):
    runs/sd0_s10/q2a_oracle_images/<variant>/cross/<task_id>.png   (600)
    runs/sd0_s10/q2a_oracle_images/<variant>/same/<task_id>.png    (150)
    runs/sd0_s10/q2a_oracle_images/<variant>/latents.pt            (all 750 z')
    runs/sd0_s10/q2a_oracle_images/<variant>/manifest.json
    runs/sd0_s10/q2a_oracle_images/sources/<stem>.png              (150 src round-trips)

Variants: sdvae (official), old_s10, c0, c1.
The generator latents z' are computed ONCE and shared across all variants, so
the only difference between directories is the decoder.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import AutoencoderKL

ROOT = Path("/root/Workspace/WEAVE_VAE")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "12h"))

from tools.vae_suite.decoder_prune import StudentDecoder, build_decoder

SD0 = "/root/Workspace/xy/DiT/data/pretrained/sd-vae-ft-ema"
TEST = ROOT / "deliver" / "testset"
TASKS = TEST / "protocol" / "tasks_750.json"
OUT = ROOT / "runs/sd0_s10/q2a_oracle_images"
Q2A_CK = ROOT / "deliver/checkpoints/runs/v6/Q2a_embed_style/Q2a_embed_style.pt"
R1_CK = ROOT / "deliver/checkpoints/runs/v6_next/probes/r1_r1.pt"
CANON_STATS = ROOT / "runs/v6/cache/canonicalizer_stats.pt"
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
RMS = 0.25
VARIANT_CK = {
    "old_s10": ROOT / "runs/vae_decoder_prune/s10_p5_lowlr/decoder.pt",
    "c0": ROOT / "runs/sd0_s10/sd0_s10_best_recon.pt",
    "c1": ROOT / "runs/sd0_s10/sd0_s10_best_overall.pt",
}


def resolve(rel, stem):
    p = TEST / rel
    if p.exists():
        return p
    c = list((TEST / "data" / "test").rglob(Path(rel).name))
    if c:
        return c[0]
    c = list((TEST / "data" / "test").rglob(f"*{stem}*"))
    if c:
        return c[0]
    raise FileNotFoundError(rel)


def load_img(p):
    im = Image.open(p).convert("RGB").resize((512, 512), Image.BICUBIC)
    return torch.from_numpy(np.asarray(im, np.uint8)).permute(2, 0, 1).float().div_(127.5).sub_(1.0)


def to_uint8(y):
    return (y.clamp(-1, 1).add(1).mul(127.5).round().clamp(0, 255).byte()
            .permute(1, 2, 0).cpu().numpy())


def rank_direction(critic, z, tgt):
    """Identical to tools/v6_next/vae_swap_full.py:rank_direction."""
    zr = z.detach().clone().requires_grad_(True)
    q = critic(zr.float()).float()
    oh = F.one_hot(tgt, len(STYLES)).bool()
    mg = (q.gather(1, tgt[:, None]).squeeze(1)
          - torch.logsumexp(q.masked_fill(oh, torch.finfo(q.dtype).min), dim=1))
    mg.sum().backward()
    g = zr.grad.detach()
    D = g[0].numel()
    gn = g.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
    return g / (gn / D ** 0.5 + 1e-12)


def build_decoder_variant(name, official, dev):
    if name == "sdvae":
        return lambda z: official.decode(z).sample
    ck = torch.load(VARIANT_CK[name], map_location="cpu", weights_only=False)
    d = build_decoder([64, 128, 192, 256])
    d.load_state_dict(ck["state_dict"])
    pq = torch.nn.Conv2d(4, 4, 1)
    pq.load_state_dict(ck["post_quant_conv"])
    sd = StudentDecoder(pq.to(dev), d.to(dev)).to(dev).eval()
    return lambda z: sd(z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="sdvae,old_s10,c0,c1")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    dev = args.device
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sources").mkdir(exist_ok=True)

    spec = json.loads(TASKS.read_text(encoding="utf-8"))
    tasks = spec["tasks"]
    if args.limit:
        tasks = tasks[:args.limit]
    print(f"[gen] tasks={len(tasks)} cross={sum(t['is_cross'] for t in tasks)} "
          f"same={sum(not t['is_cross'] for t in tasks)}", flush=True)

    official = AutoencoderKL.from_pretrained(SD0, torch_dtype=torch.float32).to(dev).eval()
    for p in official.parameters():
        p.requires_grad_(False)
    sf = float(official.config.scaling_factor)

    from tools.v6.canonicalize import ContentCanonicalizer
    from tools.v6.models_v6 import C3UNet
    from tools.v6_next.probe_style import LatentStyleNet
    import torch.nn.functional as F

    canon = ContentCanonicalizer(torch.load(CANON_STATS, map_location="cpu",
                                            weights_only=False)).to(dev).eval()
    q2 = torch.load(Q2A_CK, map_location="cpu", weights_only=False)
    gen = C3UNet(base_dim=16, num_styles=5, mode="direct", inject_16=True,
                 style_keys=False).to(dev)
    gen.load_state_dict(q2["model"]); gen.eval()
    critic = LatentStyleNet(width=64).to(dev).eval()
    critic.load_state_dict(torch.load(R1_CK, map_location="cpu",
                                      weights_only=False)["model"])
    for p in critic.parameters():
        p.requires_grad = False

    # ---- resolve + encode the 150 unique source images ONCE ----
    src_stems = sorted({t["source_stem"] for t in tasks})
    src_path = {t["source_stem"]: resolve(t["source_rel_path"], t["source_stem"])
                for t in tasks}
    z_src = {}
    t0 = time.time()
    with torch.inference_mode():
        for i in range(0, len(src_stems), args.bs):
            ch = src_stems[i:i + args.bs]
            x = torch.stack([load_img(src_path[s]) for s in ch]).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = official.encode(x).latent_dist.mean.float()
            for j, s in enumerate(ch):
                z_src[s] = z[j].cpu()
            del x, z
            torch.cuda.empty_cache()
    print(f"[gen] {len(z_src)} source latents cached in {time.time()-t0:.0f}s",
          flush=True)

    # ---- compute Oracle1 latents z' for ALL 750 tasks ONCE ----
    z_gen = {}
    t0 = time.time()
    for i in range(0, len(tasks), args.bs):
        grp = tasks[i:i + args.bs]
        zs = torch.stack([z_src[t["source_stem"]] for t in grp]).to(dev) * sf
        tg = torch.tensor([t["target_style_id"] for t in grp], device=dev)
        with torch.inference_mode():
            c = canon(zs, mode="c1_gentle", augment=False)
            z0 = gen(c, 0.0, c, tg)
        g0 = rank_direction(critic, z0, tg)     # needs live graph
        zgen = z0 + RMS * g0
        for j, t in enumerate(grp):
            z_gen[t["task_id"]] = zgen[j].detach().cpu()   # SCALED space
        del zs, tg, c, z0, g0, zgen
        if (i // args.bs) % 20 == 0:
            print(f"  [latent] {i}/{len(tasks)} {time.time()-t0:.0f}s", flush=True)
        torch.cuda.empty_cache()
    torch.save(z_gen, OUT / "oracle_latents_scaled.pt")
    print(f"[gen] Oracle1 latents for {len(z_gen)} tasks computed "
          f"({time.time()-t0:.0f}s)", flush=True)
    del gen, critic, canon
    torch.cuda.empty_cache()

    # ---- source round-trip images (for later metric computation) ----
    for name in args.variants.split(","):
        name = name.strip()
        dec = build_decoder_variant(name, official, dev)
        with torch.inference_mode():
            for i in range(0, len(src_stems), args.bs):
                ch = src_stems[i:i + args.bs]
                z = torch.stack([z_src[s] for s in ch]).to(dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    y = dec(z).float()
                for j, s in enumerate(ch):
                    Image.fromarray(to_uint8(y[j])).save(OUT / "sources" / f"{s}.png")
                del z, y
                torch.cuda.empty_cache()
        print(f"[gen] {name}: source round-trips saved", flush=True)
        del dec
        torch.cuda.empty_cache()
        break   # sources only need ONE canonical dec (sdvae); shared reference

    # ---- generate all 750 images per decoder variant ----
    for name in args.variants.split(","):
        name = name.strip()
        vdir = OUT / name
        (vdir / "cross").mkdir(parents=True, exist_ok=True)
        (vdir / "same").mkdir(parents=True, exist_ok=True)
        dec = build_decoder_variant(name, official, dev)
        manifest = []
        t0 = time.time()
        with torch.inference_mode():
            for i in range(0, len(tasks), args.bs):
                grp = tasks[i:i + args.bs]
                zs = torch.stack([z_gen[t["task_id"]] for t in grp]).to(dev) / sf
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    y = dec(zs).float()
                for j, t in enumerate(grp):
                    sub = "cross" if t["is_cross"] else "same"
                    # task_id contains '/' (style/stem/style) -> flatten to a
                    # safe filename while keeping it reversible in manifest.
                    fn = t["task_id"].replace("/", "__") + ".png"
                    Image.fromarray(to_uint8(y[j])).save(vdir / sub / fn)
                    manifest.append({
                        "task_id": t["task_id"], "file": f"{sub}/{fn}",
                        "is_cross": t["is_cross"],
                        "source_stem": t["source_stem"],
                        "source_style": t["source_style"],
                        "source_style_id": t["source_style_id"],
                        "target_style": t["target_style"],
                        "target_style_id": t["target_style_id"],
                    })
                del zs, y
                torch.cuda.empty_cache()
                if (i // args.bs) % 20 == 0:
                    print(f"  [{name}] {i}/{len(tasks)} {time.time()-t0:.0f}s",
                          flush=True)
        (vdir / "manifest.json").write_text(json.dumps({
            "variant": name, "n": len(manifest),
            "n_cross": sum(m["is_cross"] for m in manifest),
            "n_same": sum(not m["is_cross"] for m in manifest),
            "latent_space": "scaled z' from Oracle1, divided by sf before decoder",
            "scaling_factor": sf,
            "rows": manifest}, indent=2), encoding="utf-8")
        print(f"[gen] {name}: {len(manifest)} PNGs saved "
              f"({time.time()-t0:.0f}s)", flush=True)
        del dec
        torch.cuda.empty_cache()

    (OUT / "README.json").write_text(json.dumps({
        "generator": "Q2a_embed_style + R1 one-step rank_direction (Oracle1)",
        "latents": "oracle_latents_scaled.pt (all 750 tasks, WEAVE scaled space)",
        "decoders": args.variants.split(","),
        "tasks": len(tasks), "cross": sum(t["is_cross"] for t in tasks),
        "same": sum(not t["is_cross"] for t in tasks),
        "image_format": "512x512 RGB PNG, uint8",
        "note": "All decoder variants share IDENTICAL generator latents; only D differs."
    }, indent=2))
    print(f"[gen] DONE -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
