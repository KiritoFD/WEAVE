"""P2 extension: B4 arm (same C3UNet, NO R1 rank step) for B/C/D decoders.

Reuses the exact latent-generation path of gen_q2a_oracle_images.py but swaps
the generator to P1_B4_distill50 and omits rank_direction. Scores under the
same paper_rgb_v1 protocol via s10_sd0_p2.score logic (imported helpers).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path("/root/Workspace/WEAVE_VAE")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "12h"))

from tools.vae_suite.s10_sd0_adapt import OUT, build_models, make_pqc
from tools.vae_suite.decoder_prune import StudentDecoder, build_decoder

QDIR = ROOT / "runs/sd0_s10/q2a_oracle_images"
TASKS = ROOT / "deliver/testset/protocol/tasks_750.json"
B4_CK = ROOT / "deliver/checkpoints/runs/v6/P1_B4_distill50/P1_B4_distill50.pt"
CANON = ROOT / "runs/v6/cache/canonicalizer_stats.pt"
SD0_DIR = "/root/Workspace/xy/DiT/data/pretrained/sd-vae-ft-ema"
P2B4 = OUT / "p2_b4"
P2B4.mkdir(parents=True, exist_ok=True)


def resolve(rel, stem):
    p = ROOT / "deliver/testset" / rel
    if p.exists():
        return p
    c = list((ROOT / "deliver/testset/data/test").rglob(Path(rel).name))
    if c:
        return c[0]
    raise FileNotFoundError(rel)


def load_img(p):
    return Image.open(p).convert("RGB").resize((512, 512), Image.BICUBIC)


def to_tensor(p):
    return torch.from_numpy(np.asarray(load_img(p), np.uint8)).permute(2, 0, 1).float().div_(127.5).sub_(1.0)


def to_uint8(y):
    return (y.clamp(-1, 1).add(1).mul(127.5).round().clamp(0, 255).byte()
            .permute(1, 2, 0).cpu().numpy())


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device

    from diffusers import AutoencoderKL
    from tools.v6.canonicalize import ContentCanonicalizer
    from tools.v6.models_v6 import C3UNet
    spec = json.loads(TASKS.read_text(encoding="utf-8"))
    tasks = spec["tasks"]
    sd0 = AutoencoderKL.from_pretrained(SD0_DIR, torch_dtype=torch.float32).to(dev).eval()
    for p in sd0.parameters():
        p.requires_grad_(False)
    sf = float(sd0.config.scaling_factor)
    canon = ContentCanonicalizer(torch.load(CANON, map_location="cpu",
                                            weights_only=False)).to(dev).eval()
    ck = torch.load(B4_CK, map_location="cpu", weights_only=False)
    gen = C3UNet(base_dim=16, num_styles=5, mode="direct", inject_16=True,
                 style_keys=False).to(dev)
    gen.load_state_dict(ck["model"]); gen.eval()

    a0, sd0b, d, pa = build_models(dev)
    wa = pa["weight"].float().to(dev).view(4, 4)
    ba = pa["bias"].float().to(dev)
    cc = torch.load(OUT / "calibrated_pqc.pt", map_location="cpu", weights_only=False)
    wc = cc["post_quant_conv"]["weight"].float().to(dev).view(4, 4)
    bc = cc["post_quant_conv"]["bias"].float().to(dev)
    decs = {
        "B": StudentDecoder(make_pqc(wa, ba, dev), d).to(dev).eval(),
        "C": StudentDecoder(make_pqc(wc, bc, dev), d).to(dev).eval(),
        "D": lambda z: sd0.decode(z).sample,
    }

    # 150 unique source latents (SD0 space)
    stems = sorted({t["source_stem"] for t in tasks})
    srcp = {t["source_stem"]: resolve(t["source_rel_path"], t["source_stem"])
            for t in tasks}
    z_src = {}
    with torch.inference_mode():
        for i in range(0, len(stems), 8):
            ch = stems[i:i + 8]
            x = torch.stack([to_tensor(srcp[s]) for s in ch]).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = sd0.encode(x).latent_dist.mean.float()
            for j, s in enumerate(ch):
                z_src[s] = z[j].cpu()
    print(f"[P2B4] {len(z_src)} source latents", flush=True)

    # B4 latents for all 750 tasks
    z_gen = {}
    with torch.inference_mode():
        for i in range(0, len(tasks), 8):
            grp = tasks[i:i + 8]
            zs = torch.stack([z_src[t["source_stem"]] for t in grp]).to(dev) * sf
            tg = torch.tensor([t["target_style_id"] for t in grp], device=dev)
            c = canon(zs, mode="c1_gentle", augment=False)
            z0 = gen(c, 0.0, c, tg)         # NO rank_direction for B4
            for j, t in enumerate(grp):
                z_gen[t["task_id"]] = z0[j].cpu()
    torch.save(z_gen, P2B4 / "b4_latents_scaled.pt")
    print(f"[P2B4] {len(z_gen)} B4 latents", flush=True)

    # source round-trip images under B (S10) so DINO-C reference matches
    src_dir = P2B4 / "sources"
    src_dir.mkdir(exist_ok=True)
    with torch.no_grad():
        for i in range(0, len(stems), 8):
            ch = stems[i:i + 8]
            z = torch.stack([z_src[s] for s in ch]).to(dev)
            y = decs["B"](z).float()
            for j, s in enumerate(ch):
                Image.fromarray(to_uint8(y[j])).save(src_dir / f"{s}.png")

    for name, dec in decs.items():
        vdir = P2B4 / name
        (vdir / "cross").mkdir(parents=True, exist_ok=True)
        (vdir / "same").mkdir(parents=True, exist_ok=True)
        manifest = []
        with torch.no_grad():
            for i in range(0, len(tasks), 8):
                grp = tasks[i:i + 8]
                z = torch.stack([z_gen[t["task_id"]] for t in grp]).to(dev) / sf
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    y = dec(z).float()
                for j, t in enumerate(grp):
                    sub = "cross" if t["is_cross"] else "same"
                    fn = t["task_id"].replace("/", "__") + ".png"
                    Image.fromarray(to_uint8(y[j])).save(vdir / sub / fn)
                    manifest.append({
                        "task_id": t["task_id"], "file": f"{sub}/{fn}",
                        "is_cross": t["is_cross"], "source_stem": t["source_stem"],
                        "source_style": t["source_style"],
                        "source_style_id": t["source_style_id"],
                        "target_style": t["target_style"],
                        "target_style_id": t["target_style_id"]})
                del z, y
                torch.cuda.empty_cache()
        (vdir / "manifest.json").write_text(json.dumps({
            "variant": name, "n": len(manifest),
            "n_cross": sum(m["is_cross"] for m in manifest),
            "n_same": sum(not m["is_cross"] for m in manifest),
            "rows": manifest}, indent=2), encoding="utf-8")
        print(f"[P2B4] {name}: {len(manifest)} PNGs", flush=True)


if __name__ == "__main__":
    main()
