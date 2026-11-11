"""P2 reference arm A: A0 Encoder + A0 PQC + old S10, same 750 Q2a+Oracle tasks.

A is NOT a single-variable decoder comparison with B/C/D (it uses a different
encoder and recomputed generator latents), exactly as the plan notes; it is the
performance reference / upper bound.

Pipeline mirrors gen_q2a_oracle_images.py but encodes the 150 source images
with the A0 encoder, then runs Q2a + R1 one-step rank_direction in A0-scaled
latent space, and decodes with old S10 + A0 PQC.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path("/root/Workspace/WEAVE_VAE")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "12h"))

from diffusers import AutoencoderKL
from tools.v6.canonicalize import ContentCanonicalizer
from tools.v6.models_v6 import C3UNet
from tools.v6_next.probe_style import LatentStyleNet
from tools.vae_suite.decoder_prune import StudentDecoder, build_decoder
from tools.vae_suite.s10_sd0_adapt import OUT, build_models, make_pqc

A0_DIR = ROOT / "runs_vae/A0/final_diffusers_vae"
Q2A_CK = ROOT / "deliver/checkpoints/runs/v6/Q2a_embed_style/Q2a_embed_style.pt"
R1_CK = ROOT / "deliver/checkpoints/runs/v6_next/probes/r1_r1.pt"
CANON = ROOT / "runs/v6/cache/canonicalizer_stats.pt"
TASKS = ROOT / "deliver/testset/protocol/tasks_750.json"
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
RMS = 0.25
P2A = OUT / "p2_A"
P2A.mkdir(parents=True, exist_ok=True)


def resolve(rel, stem):
    p = ROOT / "deliver/testset" / rel
    if p.exists():
        return p
    c = list((ROOT / "deliver/testset/data/test").rglob(Path(rel).name))
    if c:
        return c[0]
    raise FileNotFoundError(rel)


def load_pil(p):
    return Image.open(p).convert("RGB").resize((512, 512), Image.BICUBIC)


def to_tensor(p):
    return torch.from_numpy(np.asarray(load_pil(p), np.uint8)).permute(2, 0, 1).float().div_(127.5).sub_(1.0)


def to_uint8(y):
    return (y.clamp(-1, 1).add(1).mul(127.5).round().clamp(0, 255).byte()
            .permute(1, 2, 0).cpu().numpy())


def rank_direction(critic, z, tgt):
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


def main():
    dev = "cuda"
    spec = json.loads(TASKS.read_text(encoding="utf-8"))
    tasks = spec["tasks"]
    a0 = AutoencoderKL.from_pretrained(str(A0_DIR), torch_dtype=torch.float32).to(dev).eval()
    for p in a0.parameters():
        p.requires_grad_(False)
    sf = float(a0.config.scaling_factor)
    canon = ContentCanonicalizer(torch.load(CANON, map_location="cpu",
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

    # old S10 + A0 PQC
    a0b, sd0, d, pa = build_models(dev)
    wa = pa["weight"].float().to(dev).view(4, 4)
    ba = pa["bias"].float().to(dev)
    decA = StudentDecoder(make_pqc(wa, ba, dev), d).to(dev).eval()

    stems = sorted({t["source_stem"] for t in tasks})
    srcp = {t["source_stem"]: resolve(t["source_rel_path"], t["source_stem"])
            for t in tasks}
    # A0-encoded source latents
    z_src = {}
    with torch.inference_mode():
        for i in range(0, len(stems), 8):
            ch = stems[i:i + 8]
            x = torch.stack([to_tensor(srcp[s]) for s in ch]).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = a0.encode(x).latent_dist.mean.float()
            for j, s in enumerate(ch):
                z_src[s] = z[j].cpu()
    print(f"[P2A] {len(z_src)} A0 source latents", flush=True)

    z_gen = {}
    for i in range(0, len(tasks), 8):
        grp = tasks[i:i + 8]
        zs = torch.stack([z_src[t["source_stem"]] for t in grp]).to(dev) * sf
        tg = torch.tensor([t["target_style_id"] for t in grp], device=dev)
        with torch.inference_mode():
            c = canon(zs, mode="c1_gentle", augment=False)
            z0 = gen(c, 0.0, c, tg)
        g0 = rank_direction(critic, z0, tg)
        zgen = z0 + RMS * g0
        for j, t in enumerate(grp):
            z_gen[t["task_id"]] = zgen[j].detach().cpu()
        torch.cuda.empty_cache()
    torch.save(z_gen, P2A / "A_latents_scaled.pt")
    print(f"[P2A] {len(z_gen)} A latents", flush=True)

    # source round-trips (A0 Encoder + A0 PQC + old S10)
    src_dir = P2A / "sources"
    src_dir.mkdir(exist_ok=True)
    with torch.no_grad():
        for i in range(0, len(stems), 8):
            ch = stems[i:i + 8]
            z = torch.stack([z_src[s] for s in ch]).to(dev)
            y = decA(z).float()
            for j, s in enumerate(ch):
                Image.fromarray(to_uint8(y[j])).save(src_dir / f"{s}.png")

    vdir = P2A / "A"
    (vdir / "cross").mkdir(parents=True, exist_ok=True)
    (vdir / "same").mkdir(parents=True, exist_ok=True)
    manifest = []
    with torch.no_grad():
        for i in range(0, len(tasks), 8):
            grp = tasks[i:i + 8]
            z = torch.stack([z_gen[t["task_id"]] for t in grp]).to(dev) / sf
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = decA(z).float()
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
        "variant": "A", "n": len(manifest),
        "n_cross": sum(m["is_cross"] for m in manifest),
        "n_same": sum(not m["is_cross"] for m in manifest),
        "rows": manifest}, indent=2), encoding="utf-8")
    print(f"[P2A] A: {len(manifest)} PNGs", flush=True)


if __name__ == "__main__":
    main()
