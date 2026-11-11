"""P5 layer 3: five-generator Decoder swap on the FROZEN official SD0 Encoder.

Unlike tools/v6_next/vae_swap_full.py (which swaps encoder+decoder PAIRS), here
the encoder is fixed for every arm and ONLY the decoder changes:

    x --E_SD0--> z_src --gen_arm--> z' --D_variant--> y

Variants : sdvae (official) / old_s10 / c0 / c1
Arms     : Q2a+R1 Oracle, P1_B4, P1_B3, ScoreDiff N12, ScoreDiff N24
Tasks    : frozen 750 (600 cross + 150 same), deliver/testset/protocol/tasks_750.json

Metrics  : CSD Target Top-1 + Margin, DINO-S, CLIP-S, Style Affinity,
           DINO-C, LPIPS; per-target breakdown; DINO-S target SELECTIVITY
           (margin, not just a global similarity rise). Paired 95% CI uses
           source-clustered bootstrap THAT PRESERVES DUPLICATE CLUSTER DRAWS.

NOTE: the five generator latents must be produced in the OFFICIAL SD0 latent
space (the same space the decoder was trained on). This script computes them
once per arm and caches them, so all decoder variants see IDENTICAL latents.

Run AFTER P1/P2/P3 produce c0_best.pt / c1_best.pt:
    python -m tools.vae_suite.eval_generator_swap --variants sdvae,old_s10,c0,c1
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL

ROOT = Path("/root/Workspace/WEAVE_VAE")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "12h"))

from tools.vae_suite.decoder_prune import StudentDecoder, build_decoder

SD0 = "/root/Workspace/xy/DiT/data/pretrained/sd-vae-ft-ema"
TEST = ROOT / "deliver" / "testset"
TASKS = TEST / "protocol" / "tasks_750.json"
OUT = ROOT / "runs/sd0_s10"
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
NM = [0.48145466, 0.4578275, 0.40821073]
NS = [0.26862954, 0.26130258, 0.27577711]

ARM_CK = {
    "Oracle1": "deliver/checkpoints/runs/v6/Q2a_embed_style/Q2a_embed_style.pt",
    "B4": "deliver/checkpoints/runs/v6/P1_B4_distill50/P1_B4_distill50.pt",
    "B3": "deliver/checkpoints/runs/v6/P1_B3_distill20/P1_B3_distill20.pt",
    "ScoreDiff_N12": "deliver/checkpoints/runs/v6/score_diff/score_diff_net.pt",
    "ScoreDiff_N24": "deliver/checkpoints/runs/v6/score_diff/score_diff_net.pt",
}
VARIANT_CK = {
    "old_s10": ROOT / "runs/vae_decoder_prune/s10_p5_lowlr/decoder.pt",
    "c0": OUT / "c0_best.pt",
    "c1": OUT / "c1_best.pt",
}


def build_decoder_variant(name, official, dev):
    if name == "sdvae":
        return lambda z: official.decode(z).sample
    f = VARIANT_CK[name]
    if not f.exists():
        raise FileNotFoundError(f"{name}: {f}")
    ck = torch.load(f, map_location="cpu", weights_only=False)
    d = build_decoder([64, 128, 192, 256])
    d.load_state_dict(ck["state_dict"])
    pq = torch.nn.Conv2d(4, 4, 1)
    pq.load_state_dict(ck["post_quant_conv"])
    sd = StudentDecoder(pq.to(dev), d.to(dev)).to(dev).eval()
    return lambda z: sd(z)


def cluster_boot_idx(clusters, n_boot, seed):
    """Whole-cluster resampling WITH replacement, preserving duplicates.

    tools/v6_24h/stats.py uses np.isin(cid, pick) which collapses duplicate
    draws and understates variance; concatenate instead.
    """
    uniq, inv = np.unique(clusters, return_inverse=True)
    by = [np.where(inv == i)[0] for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        yield np.concatenate([by[c] for c in pick])


def paired_ci(a, b, clusters, n_boot=2000, seed=0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    ms = np.array([d[idx].mean() for idx in cluster_boot_idx(clusters, n_boot, seed)])
    lo, hi = float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))
    return {"mean": float(d.mean()), "ci": [lo, hi],
            "excludes_zero": bool(lo > 0 or hi < 0), "n": int(len(d))}


def load_img(p, size=512):
    from PIL import Image
    im = Image.open(p).convert("RGB").resize((size, size), Image.BICUBIC)
    a = np.asarray(im, dtype=np.uint8)
    return torch.from_numpy(a).permute(2, 0, 1).float().div_(127.5).sub_(1.0)


def build_generator_latents(official, arms, tasks, dev, bs=8):
    """Compute z' ONCE per arm in the frozen SD0 space; cache to disk.

    All decoder variants then read the SAME latents (no re-encoding).
    """
    from tools.v6.models_v6 import C3UNet
    from tools.v6.score_diff import edit_with_score_difference
    from tools.v6.models_v6 import ConditionalScoreNet
    from tools.v6_next.probe_style import LatentStyleNet
    from tools.v6.canonicalize import ContentCanonicalizer

    cache = OUT / "gen_latents.pt"
    if cache.exists():
        return torch.load(cache, map_location="cpu", weights_only=False)

    # canonicalizer stats in SD0 space
    stats_f = ROOT / "runs/v6/cache/canonicalizer_stats.pt"
    if not stats_f.exists():
        raise FileNotFoundError("canonicalizer_stats.pt missing; "
                                "build it with tools/v6_next/vae_swap_full.build_canon_stats")
    canon = ContentCanonicalizer(torch.load(stats_f, map_location="cpu",
                                            weights_only=False)).to(dev).eval()
    critic = None
    r1_f = ROOT / ARM_CK.get("R1", "deliver/checkpoints/runs/v6_next/probes/r1_r1.pt")
    if Path(r1_f).exists():
        critic = LatentStyleNet(width=64).to(dev).eval()
        critic.load_state_dict(torch.load(r1_f, map_location="cpu",
                                          weights_only=False)["model"])
        for p in critic.parameters():
            p.requires_grad = False
    sd_ck = torch.load(ROOT / ARM_CK["ScoreDiff_N12"], map_location="cpu",
                       weights_only=False)
    sd = ConditionalScoreNet(base_dim=64, num_styles=5).to(dev).eval()
    sd.load_state_dict(sd_ck["model"])

    out = {}
    src_stems = sorted({t["source_stem"] for t in tasks})
    # resolve source images once
    def resolve(rel, stem):
        p = TEST / rel
        if p.exists():
            return p
        c = list((TEST / "data" / "test").rglob(Path(rel).name))
        if c:
            return c[0]
        raise FileNotFoundError(rel)
    src_path = {t["source_stem"]: resolve(t["source_rel_path"], t["source_stem"])
                for t in tasks}
    for name in arms:
        ck = torch.load(ROOT / ARM_CK[name], map_location="cpu", weights_only=False)
        g = C3UNet(base_dim=16, num_styles=5, mode="direct", inject_16=True,
                   style_keys=False).to(dev)
        g.load_state_dict(ck["model"]); g.eval()
        out[name] = {}
        for i in range(0, len(src_stems), bs):
            ch = src_stems[i:i + bs]
            x = torch.stack([load_img(src_path[s]) for s in ch]).to(dev)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                z = official.encode(x).latent_dist.mean.float()
                c = canon(z, mode="c1_gentle", augment=False)
                # one representative target (per-task latents are computed below)
                out[name].setdefault("_z_src", {})[ch[0]] = z[0].cpu()
        del g
    torch.save(out, cache)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="sdvae,old_s10,c0,c1")
    ap.add_argument("--arms", default="Oracle1,B4,B3,ScoreDiff_N12,ScoreDiff_N24")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device

    print("[P5-L3] This layer requires per-task generator latents in SD0 space.")
    print("        Implemented as a runnable skeleton; full execution is gated on")
    print("        c0_best.pt / c1_best.pt and canonicalizer stats being available.")
    spec = json.loads(TASKS.read_text(encoding="utf-8"))
    tasks = spec["tasks"]
    if args.limit:
        tasks = tasks[:args.limit]
    cross = [t for t in tasks if t.get("is_cross")]
    same = [t for t in tasks if not t.get("is_cross")]
    official = AutoencoderKL.from_pretrained(SD0, torch_dtype=torch.float32).to(dev).eval()
    for p in official.parameters():
        p.requires_grad_(False)
    missing = []
    for v in args.variants.split(","):
        try:
            build_decoder_variant(v.strip(), official, dev)
        except FileNotFoundError:
            missing.append(v.strip())
    report = {
        "status": "skeleton_not_run",
        "n_tasks": len(tasks), "n_cross": len(cross), "n_same": len(same),
        "variants": args.variants.split(","), "arms": args.arms.split(","),
        "missing_checkpoints": missing,
        "reason": ("per-task generator latent construction + 5-metric scoring "
                   "must be executed after P1/P2/P3; this skeleton fixes the "
                   "interface (frozen SD0 Encoder, decoder-only swap, "
                   "source-clustered paired bootstrap)."),
        "todo": ["build per-task z' for all 5 arms in SD0 space",
                 "score CSD Top-1/Margin, DINO-S, CLIP-S, Style Affinity, "
                 "DINO-C, LPIPS for each (arm, variant)",
                 "DINO-S target SELECTIVITY (margin), not just global similarity",
                 "per-target 5-style breakdown",
                 "paired source-clustered bootstrap with duplicate clusters"],
    }
    (OUT / "generator_swap_status.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
