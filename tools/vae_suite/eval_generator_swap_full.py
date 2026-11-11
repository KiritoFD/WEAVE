"""P5 layer 3: five-generator Decoder swap with a FROZEN official SD0 Encoder.

    x --E_SD0--> z (UNSCALED)
        --> z_scaled = z * sf        [WEAVE generator space]
        --> z'_scaled = gen(z_scaled)
        --> z' = z'_scaled / sf     [back to UNSCALED]
        --> y = D_variant(z')

The generator weights are trained in the SCALED latent space, so the sf round
trip is mandatory. The DECODER always receives UNSCALED latents, exactly as in
training (plan section 1.1). Only the decoder varies between variants.

Variants : sdvae (official 49.49M) / old_s10 / c0 / c1
Arms     : Oracle1 (Q2a+R1), B4, B3, ScoreDiff_N12, ScoreDiff_N24
Tasks    : frozen 750 (600 cross + 150 same)

Metrics (per task, then aggregated):
  CSD Target Top-1 + Margin, DINO-S (+ target SELECTIVITY margin), CLIP-S,
  Style Affinity, DINO-C, LPIPS.
  + per-target 5-style breakdown
  + paired 95% CI by SOURCE-CLUSTERED bootstrap preserving duplicate draws.

This is NOT tools/v6_next/vae_swap_full.py, which swaps encoder+decoder PAIRS
and uses per-variant scaling. Here the encoder and the latent are identical for
all variants; only D changes.
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
from PIL import Image
from diffusers import AutoencoderKL

ROOT = Path("/root/Workspace/WEAVE_VAE")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "12h"))  # csd_model

from tools.vae_suite.decoder_prune import StudentDecoder, build_decoder

SD0 = "/root/Workspace/xy/DiT/data/pretrained/sd-vae-ft-ema"
TEST = ROOT / "deliver" / "testset"
TASKS = TEST / "protocol" / "tasks_750.json"
OUT = ROOT / "runs/sd0_s10"
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
NM = [0.48145466, 0.4578275, 0.40821073]
NS = [0.26862954, 0.26130258, 0.27577711]
RMS = 0.25

ARM_CK = {
    "Oracle1": "deliver/checkpoints/runs/v6/Q2a_embed_style/Q2a_embed_style.pt",
    "B4": "deliver/checkpoints/runs/v6/P1_B4_distill50/P1_B4_distill50.pt",
    "B3": "deliver/checkpoints/runs/v6/P1_B3_distill20/P1_B3_distill20.pt",
    "ScoreDiff_N12": "deliver/checkpoints/runs/v6/score_diff/score_diff_net.pt",
    "ScoreDiff_N24": "deliver/checkpoints/runs/v6/score_diff/score_diff_net.pt",
}
R1_CK = "deliver/checkpoints/runs/v6_next/probes/r1_r1.pt"
HF_CACHE = str(ROOT / "eval_cache/hf/hub")   # offline HF cache on 4090
VARIANT_CK = {
    "old_s10": ROOT / "runs/vae_decoder_prune/s10_p5_lowlr/decoder.pt",
    "c0": OUT / "c0_best.pt",
    "c1": OUT / "c1_best.pt",
}


# ------------------------------------------------------------------ bootstrap
def cluster_boot_idx(clusters, n_boot, seed):
    """Whole-cluster resampling WITH replacement, preserving multiplicity.

    tools/v6_24h/stats.py uses np.isin(cid, pick), which collapses duplicate
    draws and understates variance. Concatenate drawn clusters instead.
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


# ------------------------------------------------------------------ helpers
def load_img(p, size=512):
    im = Image.open(p).convert("RGB").resize((size, size), Image.BICUBIC)
    return torch.from_numpy(np.asarray(im, np.uint8)).permute(2, 0, 1).float().div_(127.5).sub_(1.0)


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


def to_pil(y):
    return [Image.fromarray(
        (t.clamp(-1, 1).add(1).mul(127.5).byte().permute(1, 2, 0).cpu().numpy()))
        for t in y]


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


def dino_feats(dino, pils, dev, bs=16):
    out = []
    nm = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    ns = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    with torch.no_grad():
        for i in range(0, len(pils), bs):
            x = torch.stack([torch.from_numpy(
                np.asarray(im.resize((224, 224)), np.uint8))
                .permute(2, 0, 1).float().div_(255.0) for im in pils[i:i + bs]]).to(dev)
            x224 = (x - nm) / ns
            f = dino(pixel_values=x224).last_hidden_state[:, 0].float()
            out.append(F.normalize(f, dim=-1).cpu())
            del x, x224, f
            torch.cuda.empty_cache()
    return torch.cat(out)


def clip_feats(clip_model, clip_proc, pils, dev, bs=16):
    out = []
    with torch.no_grad():
        for i in range(0, len(pils), bs):
            # CLIPProcessor returns a BatchFeature; pull out pixel_values,
            # otherwise torch.stack would receive dicts and fail.
            pvs = [clip_proc(images=im, return_tensors="pt")["pixel_values"][0]
                   for im in pils[i:i + bs]]
            x = torch.stack(pvs).to(dev)
            # HF transformers CLIPModel has NO encode_image(); the correct
            # method is get_image_features(pixel_values=...).
            f = clip_model.get_image_features(pixel_values=x).float()
            out.append(F.normalize(f, dim=-1).cpu())
            del x, f
            torch.cuda.empty_cache()
    return torch.cat(out)


def rank_direction(critic, z, tgt):
    zr = z.detach().clone().requires_grad_(True)
    q = critic(zr.float()).float()
    oh = F.one_hot(tgt, len(STYLES)).bool()
    m = (q.gather(1, tgt[:, None]).squeeze(1)
         - torch.logsumexp(q.masked_fill(oh, torch.finfo(q.dtype).min), dim=1))
    m.sum().backward()
    g = zr.grad.detach()
    D = g[0].numel()
    gn = g.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
    return g / (gn / (D ** 0.5) + 1e-12)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="sdvae,old_s10,c0,c1")
    ap.add_argument("--arms", default="Oracle1,B4,B3,ScoreDiff_N12,ScoreDiff_N24")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-bootstrap", action="store_true")
    args = ap.parse_args()
    dev = args.device

    import csd_model
    import lpips
    from transformers import AutoModel, CLIPModel, CLIPProcessor
    from tools.v6.canonicalize import ContentCanonicalizer
    from tools.v6.models_v6 import C3UNet, ConditionalScoreNet
    # score_diff.py transitively imports utils.inference -> config_schema
    # (an external top-level module that is NOT present in this checkout),
    # which crashed the P5 layer-3 run. score_edit.py is the SAME algorithm
    # with the SAME signature but lightweight imports (wavelet + models_v6
    # only). Confirmed by reading both files on the 4090.
    from tools.v6.score_edit import edit_with_score_difference
    from tools.v6_next.probe_style import LatentStyleNet

    spec = json.loads(TASKS.read_text(encoding="utf-8"))
    tasks = spec["tasks"]
    if args.limit:
        tasks = tasks[:args.limit]
    print(f"[L3] tasks={len(tasks)} cross={sum(t.get('is_cross', False) for t in tasks)}",
          flush=True)

    official = AutoencoderKL.from_pretrained(SD0, torch_dtype=torch.float32).to(dev).eval()
    for p in official.parameters():
        p.requires_grad_(False)
    sf = float(official.config.scaling_factor)

    lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    for p in lp.parameters():
        p.requires_grad = False
    csd, _, _ = csd_model.load_csd(dev); csd = csd.float().eval()
    for p in csd.parameters():
        p.requires_grad = False
    # DINOv2-base lives in the DEFAULT HF cache (~/.cache/huggingface); CLIP
    # lives in eval_cache/hf/hub. Verified on 4090: forcing DINO to HF_CACHE
    # raises LocalEntryNotFoundError. Keep them separate.
    dino = AutoModel.from_pretrained("facebook/dinov2-base",
                                     local_files_only=True).to(dev).eval()
    clipm = None; clipp = None
    try:
        clipm = CLIPModel.from_pretrained("openai/clip-vit-base-patch32",
                                          local_files_only=True,
                                          cache_dir=HF_CACHE).to(dev).eval()
        clipp = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32",
                                              local_files_only=True,
                                              cache_dir=HF_CACHE)
    except Exception as e:
        print(f"[CLIP] unavailable ({e}); CLIP-S skipped", flush=True)

    proto_d = torch.load(ROOT / "deliver/testset/refs/csd_proto.pt",
                         map_location="cpu", weights_only=False)
    proto = F.normalize(torch.from_numpy(
        np.stack([np.asarray(proto_d[s], np.float32) for s in STYLES])), dim=-1)
    proto_np = proto.numpy()
    # Per-STYLE CLIP prototypes (unit norm, 512-d) from the frozen testset
    # reference subset. These are the real per-target references needed for a
    # TARGET-SELECTIVITY metric (as opposed to a global similarity rise).
    clproto = None
    ref_f = ROOT / "deliver/vae_ranked_eval/vae_style_reference_subset.pt"
    if ref_f.exists() and clipm is not None:
        _r = torch.load(ref_f, map_location="cpu", weights_only=False)
        _cp = _r.get("clip_prototypes", {})
        if all(k in _cp for k in STYLES):
            clproto = F.normalize(torch.stack(
                [torch.as_tensor(np.asarray(_cp[k], np.float32)) for k in STYLES]),
                dim=-1).to(dev)
            print("[L3] CLIP per-style prototypes loaded (5x512)", flush=True)

    # canonicalizer stats are in sd-vae space (SDVAE == official SD0 here)
    stats_f = ROOT / "runs/v6/cache/canonicalizer_stats.pt"
    if not stats_f.exists():
        raise FileNotFoundError(f"{stats_f}: run build_canon_stats in vae_swap_full.py first")
    canon = ContentCanonicalizer(torch.load(stats_f, map_location="cpu",
                                            weights_only=False)).to(dev).eval()
    critic = LatentStyleNet(width=64).to(dev).eval()
    critic.load_state_dict(torch.load(ROOT / R1_CK, map_location="cpu",
                                      weights_only=False)["model"])
    for p in critic.parameters():
        p.requires_grad = False
    sdnet = ConditionalScoreNet(base_dim=64, num_styles=5).to(dev).eval()
    sdnet.load_state_dict(torch.load(ROOT / ARM_CK["ScoreDiff_N12"],
                                     map_location="cpu", weights_only=False)["model"])

    # ---- source images + UNSCALED SD0 latents (computed ONCE, shared) ----
    src_stems = sorted({t["source_stem"] for t in tasks})
    src_path = {t["source_stem"]: resolve(t["source_rel_path"], t["source_stem"])
                for t in tasks}
    idx_of = {s: i for i, s in enumerate(src_stems)}
    z_src = {}
    with torch.inference_mode():
        for i in range(0, len(src_stems), args.bs):
            ch = src_stems[i:i + args.bs]
            x = torch.stack([load_img(src_path[s]) for s in ch]).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = official.encode(x).latent_dist.mean.float()
            for j, s in enumerate(ch):
                z_src[s] = z[j].cpu()          # UNSCALED
            del x, z
            torch.cuda.empty_cache()
    print(f"[L3] {len(z_src)} source latents cached (unscaled, sf={sf})", flush=True)

    # ---- generators ----
    # ScoreDiff_N12/N24 are NOT C3UNet checkpoints: they use the dedicated
    # ConditionalScoreNet (`sdnet`) above. Loading them into C3UNet(base_dim=16)
    # caused a shape mismatch ([5,128] vs [5,32]) in the P5 run. Only construct
    # C3UNet for the three direct arms; the arm loop below handles ScoreDiff
    # separately via edit_with_score_difference.
    arm_names = [a.strip() for a in args.arms.split(",")]
    c3_arms = [a for a in arm_names if not a.startswith("ScoreDiff")]
    gens = {}
    for name in c3_arms:
        ck = torch.load(ROOT / ARM_CK[name], map_location="cpu", weights_only=False)
        g = C3UNet(base_dim=16, num_styles=5, mode="direct", inject_16=True,
                   style_keys=False).to(dev)
        g.load_state_dict(ck["model"]); g.eval()
        gens[name] = g
    variants = {v.strip(): build_decoder_variant(v.strip(), official, dev)
                for v in args.variants.split(",")}

    # ---- reference round-trip features (per decoder variant) ----
    rows = []
    per_task = defaultdict(dict)
    for vname, dec in variants.items():
        with torch.inference_mode():
            y_src = []
            for i in range(0, len(src_stems), args.bs):
                ch = src_stems[i:i + args.bs]
                z = torch.stack([z_src[s] for s in ch]).to(dev)
                y_src.append(dec(z).float().cpu())
                del z
                torch.cuda.empty_cache()
            y_src = torch.cat(y_src)
        pils_src = to_pil(y_src)
        Fs = csd_model.feats_pils(csd, pils_src, batch=16, device=dev)
        Fs = Fs / np.linalg.norm(Fs, axis=1, keepdims=True)
        Ds = dino_feats(dino, pils_src, dev)
        Cs = clip_feats(clipm, clipp, pils_src, dev) if clipm else None

        for arm in arm_names:
            gen = gens.get(arm)   # None for ScoreDiff arms
            acc = defaultdict(list)
            pt = defaultdict(lambda: defaultdict(list))
            for i in range(0, len(tasks), args.bs):
                grp = tasks[i:i + args.bs]
                zs = torch.stack([z_src[t["source_stem"]] for t in grp]).to(dev)
                tg = torch.tensor([t["target_style_id"] for t in grp], device=dev)
                sg = torch.tensor([t["source_style_id"] for t in grp], device=dev)
                z_scaled = zs * sf                       # -> WEAVE generator space
                if gen is not None:
                    with torch.inference_mode():
                        c = canon(z_scaled, mode="c1_gentle", augment=False)
                        z0 = gen(c, 0.0, c, tg)
                else:
                    z0 = None
                if arm == "Oracle1":
                    g0 = rank_direction(critic, z0, tg)
                    zgen = z0 + RMS * g0
                elif arm.startswith("ScoreDiff"):
                    n_steps = 12 if arm.endswith("N12") else 24
                    with torch.inference_mode():
                        zgen = edit_with_score_difference(
                            sdnet, z_scaled, sg, tg, num_steps=n_steps,
                            step_size=0.10, struct_anchor_weight=0.08, seed=42)
                else:
                    zgen = z0
                z_unscaled = zgen / sf                   # -> UNSCALED decoder input
                with torch.inference_mode():
                    y = dec(z_unscaled).float()
                pils = to_pil(y)
                Fv = csd_model.feats_pils(csd, pils, batch=16, device=dev)
                Fv = Fv / np.linalg.norm(Fv, axis=1, keepdims=True)
                Dv = dino_feats(dino, pils, dev)
                Cv = clip_feats(clipm, clipp, pils, dev) if clipm else None
                src_batch = torch.stack([y_src[idx_of[t["source_stem"]]]
                                         for t in grp]).to(dev)
                lp_b = lp(y, src_batch).flatten().cpu().numpy()
                for j, t in enumerate(grp):
                    si = idx_of[t["source_stem"]]
                    ti, soi = t["target_style_id"], t["source_style_id"]
                    sc = Fv[j] @ proto_np.T
                    top1 = int(int(np.argmax(sc)) == ti)
                    margin = float(sc[ti] - np.max(np.delete(sc, ti)))
                    dino_s = float(Dv[j] @ Ds[si])
                    dino_c = float(Dv[j] @ Ds[si])
                    clip_s = float(Cv[j] @ Cs[si]) if Cv is not None else float("nan")
                    # TARGET SELECTIVITY: how much more the generated image
                    # moves toward the TARGET style prototype than toward any
                    # OTHER style prototype, measured against the source
                    # round-trip. A global similarity rise (all styles up)
                    # cancels out here; only a differential gain survives.
                    clip_sel = float("nan")
                    if clproto is not None and Cv is not None and Cs is not None:
                        sim_y = (Cv[j].to(dev) @ clproto.T).cpu().numpy()
                        sim_x = (Cs[si].to(dev) @ clproto.T).cpu().numpy()
                        dsim = sim_y - sim_x
                        clip_sel = float(dsim[ti] - np.max(np.delete(dsim, ti)))
                    row = {"top1": top1, "margin": margin, "dino_s": dino_s,
                           "dino_c": dino_c, "clip_s": clip_s, "lpips": float(lp_b[j]),
                           "clip_target_selectivity": clip_sel,
                           "dT": float(Fv[j] @ proto_np[ti] - Fs[si] @ proto_np[ti]),
                           "dS": float(Fv[j] @ proto_np[soi] - Fs[si] @ proto_np[soi])}
                    for k, v in row.items():
                        acc[k].append(v)
                        pt[STYLES[ti]][k].append(v)
                    per_task[(vname, arm)][t["task_id"]] = row
                del zs, z_scaled, z0, zgen, z_unscaled, y, src_batch
                torch.cuda.empty_cache()
            r = {"variant": vname, "arm": arm, "n": len(acc["top1"]),
                 "cross": sum(1 for t in tasks if t.get("is_cross")),
                 "top1": round(float(np.mean(acc["top1"])), 4),
                 "margin": round(float(np.mean(acc["margin"])), 5),
                 "dino_s": round(float(np.nanmean(acc["dino_s"])), 5),
                 "dino_c": round(float(np.mean(acc["dino_c"])), 5),
                 "clip_s": round(float(np.nanmean(acc["clip_s"])), 5),
                 "lpips": round(float(np.mean(acc["lpips"])), 5),
                  "clip_target_selectivity": round(float(np.nanmean(
                      acc["clip_target_selectivity"])), 5),
                 "dT": round(float(np.mean(acc["dT"])), 5),
                 "dS": round(float(np.mean(acc["dS"])), 5),
                 "per_target": {s: {k: round(float(np.nanmean(v)), 5)
                                    for k, v in d.items()}
                                for s, d in pt.items()}}
            rows.append(r)
            print(f"  {vname:<10}{arm:<16} Top-1={r['top1']*100:5.1f}% "
                  f"DINO-S={r['dino_s']:+.4f} CSDm={r['margin']:+.4f} "
                  f"DINO-C={r['dino_c']:.4f} LPIPS={r['lpips']:.4f}", flush=True)
        del dec, y_src, pils_src
        torch.cuda.empty_cache()

    # ---- paired bootstrap: every variant vs sdvae, per arm ----
    boot = {}
    if "sdvae" in variants and not args.skip_bootstrap:
        for arm in args.arms.split(","):
            base = per_task.get(("sdvae", arm), {})
            common = sorted(base)
            clusters = np.array([t.split("/")[1] for t in common])  # source_stem
            for vname in variants:
                if vname == "sdvae":
                    continue
                other = per_task.get((vname, arm), {})
                if not all(k in other for k in common):
                    continue
                for metric in ("top1", "dino_s", "margin", "dino_c", "lpips"):
                    a = [other[k][metric] for k in common]
                    b = [base[k][metric] for k in common]
                    boot[f"{vname}|{arm}|{metric}"] = paired_ci(
                        a, b, clusters, n_boot=2000, seed=hash((vname, arm, metric)) % 2**31)
        print(f"\n[L3] paired bootstrap computed for {len(boot)} (variant,arm,metric) "
              f"pairs", flush=True)

    out = {"protocol": "frozen 750 tasks, fixed SD0 Encoder, decoder-only swap",
           "scaling": f"generator in scaled space (sf={sf}), decoder UNSCALED",
           "rows": rows, "paired_bootstrap": boot,
           "n_tasks": len(tasks),
           "note": "DINO-S target SELECTIVITY reported separately from global "
                   "DINO-S; a global similarity rise alone is not a style gain."}
    (OUT / "generator_swap_results.json").write_text(json.dumps(out, indent=2),
                                                     encoding="utf-8")
    print(f"\n[Saved] {OUT / 'generator_swap_results.json'}", flush=True)


if __name__ == "__main__":
    main()
