"""Re-evaluate the v6 checkpoints under DIFFERENT VAEs, encoder+decoder paired.

For every (arm, variant) pair the whole pipeline uses that variant's own encoder and
decoder, as required: a latent produced by encoder V is only ever decoded by decoder V.

    x --E_V--> z_src --canon--> c --gen_arm--> z0 --(+R1 step)--> z --D_V--> y

Variants
  sdvae  sd-vae-ft-ema (the space the published v6 numbers live in)
  a0     A0 encoder + A0 decoder
  s10    A0 encoder + S10 decoder (10.66M)
  cedit  C_edit encoder + C_edit decoder

Measured latent compatibility (deliver/vae_swap_compat.json):
  cos(a0, sdvae) = +0.992, RMS ratio 0.998  -> A0 is a light fine-tune, same space
  cos(cedit, sdvae) = +0.693, RMS ratio 1.285 -> C_edit is a genuinely new space, so
  its numbers are NOT a decoder swap and must be read separately.

Canonicalizer stats were not delivered; they are recomputed here from the 512px
training pool in sd-vae space and cached, matching ContentCanonicalizer.fit_global_stats.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path("/root/Workspace/WEAVE_VAE")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "12h"))

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from PIL import Image

from tools.v6.canonicalize import STYLES, ContentCanonicalizer
from tools.v6.models_v6 import C3UNet
from tools.v6_next.probe_style import LatentStyleNet

SDVAE = "/root/Workspace/xy/DiT/data/pretrained/sd-vae-ft-ema"
A0 = str(ROOT / "runs_vae/A0/final_diffusers_vae")
S10_CK = ROOT / "runs/vae_decoder_prune/s10_p5_lowlr/decoder.pt"
CED_CK = ROOT / "runs/vae_style_vae/C_edit_best.pt"
TEST = ROOT / "deliver/testset"
TASKS = TEST / "protocol" / "tasks_750.json"
MANIFEST = Path("/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")
CACHE = ROOT / "runs/v6/cache/canonicalizer_stats.pt"
OUT = ROOT / "deliver/vae_swap_full.json"
NM = [0.48145466, 0.4578275, 0.40821073]
NS = [0.26862954, 0.26130258, 0.27577711]

ARMS = {
    "Q2a": "deliver/checkpoints/runs/v6/Q2a_embed_style/Q2a_embed_style.pt",
    "Oracle1": "deliver/checkpoints/runs/v6/Q2a_embed_style/Q2a_embed_style.pt",
    "B4": "deliver/checkpoints/runs/v6/P1_B4_distill50/P1_B4_distill50.pt",
}
R1_CK = "deliver/checkpoints/runs/v6_next/probes/r1_r1.pt"
RMS = 0.25


def load_img(p, size=512):
    im = Image.open(p).convert("RGB").resize((size, size), Image.BICUBIC)
    a = np.asarray(im, dtype=np.uint8)
    return torch.from_numpy(a).permute(2, 0, 1).float().div_(127.5).sub_(1.0)


def csd_feat(csd, x, dev):
    nm = torch.tensor(NM, device=dev).view(1, 3, 1, 1)
    ns = torch.tensor(NS, device=dev).view(1, 3, 1, 1)
    x01 = (x.clamp(-1, 1) + 1) / 2
    x224 = F.interpolate(x01, (224, 224), mode="bicubic", align_corners=False)
    _, _, s = csd((x224 - nm) / ns)
    return F.normalize(s.float(), dim=-1)


def dino_feat(m, x, dev):
    nm = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    ns = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    x01 = (x.clamp(-1, 1) + 1) / 2
    x224 = F.interpolate(x01, (224, 224), mode="bicubic", align_corners=False)
    o = m(pixel_values=(x224 - nm) / ns)
    return F.normalize(o.last_hidden_state[:, 0].float(), dim=-1)


def build_canon_stats(sdvae, dev, cap=1000):
    if CACHE.exists():
        return torch.load(CACHE, map_location="cpu", weights_only=False)
    m = json.loads(MANIFEST.read_text())
    mm = np.memmap(m["bin_path"], dtype=np.uint8, mode="r", shape=tuple(m["shape"]))
    val = set(m["val_indices"])
    by = {s: [] for s in STYLES}   # full per-style pool; fit_global_stats subsamples
    # the WikiArt manifest spells some v6 style names differently
    alias = {"Early_Renaissance": ["Early_Renaissance", "Early Renaissance"],
             "Impressionism": ["Impressionism"],
             "Minimalism": ["Minimalism"],
             "Rococo": ["Rococo"],
             "Ukiyo_e": ["Ukiyo_e", "Ukiyo-e"]}
    sid = {}
    for s in STYLES:
        for cand in alias.get(s, [s]):
            if cand in m["style_to_id"]:
                sid[s] = m["style_to_id"][cand]
                break
        if s not in sid:
            raise KeyError(f"style {s} not in manifest; have {sorted(m['style_to_id'])[:5]}...")
    for r in m["records"]:
        if not r["in_train"] or r["raw_index"] in val:
            continue
        for s in STYLES:
            if r["style_id"] == sid[s]:
                by[s].append(r["raw_index"])
    lat = {}
    with torch.no_grad():
        for s in STYLES:
            chunks = []
            for i in range(0, len(by[s]), 32):
                arr = np.stack([np.ascontiguousarray(mm[r]) for r in by[s][i:i + 32]])
                x = torch.from_numpy(arr).permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0).to(dev)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    z = sdvae.encode(x).latent_dist.mean
                chunks.append((z.float() * sdvae.config.scaling_factor).cpu())
                del x, z
                torch.cuda.empty_cache()
            lat[s] = torch.cat(chunks)
            print(f"  [stats] {s}: {lat[s].shape[0]} latents", flush=True)
    st = ContentCanonicalizer.fit_global_stats(lat, per_style_cap=cap)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(st, CACHE)
    print(f"[stats] saved {CACHE}", flush=True)
    return st


def rank_direction(critic, z, tgt):
    zr = z.detach().clone().requires_grad_(True)
    q = critic(zr.float()).float()
    oh = F.one_hot(tgt, len(STYLES)).bool()
    mg = (q.gather(1, tgt[:, None]).squeeze(1)
          - torch.logsumexp(q.masked_fill(oh, torch.finfo(q.dtype).min), dim=1))
    mg.sum().backward()
    g = zr.grad.detach()
    D = g[0].numel()
    # unit-RMS direction, matching tools/v6_next/amortize_p0.py:
    #   g / (||g|| / sqrt(D))  ==  g * sqrt(D) / ||g||   (RMS == 1)
    gn = g.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
    return g / (gn / D ** 0.5 + 1e-12)


def build_variants(dev):
    from tools.vae_suite.decoder_prune import build_decoder, StudentDecoder
    out = {}
    sd = AutoencoderKL.from_pretrained(SDVAE, torch_dtype=torch.float32).to(dev).eval()
    out["sdvae"] = {"enc": lambda x: sd.encode(x).latent_dist.mean,
                    "dec": lambda z: sd.decode(z).sample,
                    "sf": float(sd.config.scaling_factor)}
    a0 = AutoencoderKL.from_pretrained(A0, torch_dtype=torch.float32).to(dev).eval()
    out["a0"] = {"enc": lambda x: a0.encode(x).latent_dist.mean,
                 "dec": lambda z: a0.decode(z).sample,
                 "sf": float(a0.config.scaling_factor)}
    a0b = AutoencoderKL.from_pretrained(A0, torch_dtype=torch.float32).to(dev).eval()
    ck = torch.load(S10_CK, map_location="cpu", weights_only=False)
    d = build_decoder(ck["meta"]["widths_forward"]); d.load_state_dict(ck["state_dict"])
    s10 = StudentDecoder(a0b.post_quant_conv, d.to(dev)).to(dev).eval()
    out["s10"] = {"enc": lambda x: a0b.encode(x).latent_dist.mean,
                  "dec": lambda z: s10(z), "sf": float(a0b.config.scaling_factor)}
    a0c = AutoencoderKL.from_pretrained(A0, torch_dtype=torch.float32).to(dev).eval()
    ck2 = torch.load(CED_CK, map_location="cpu", weights_only=False)
    a0c.encoder.load_state_dict(ck2["encoder"])
    d2 = build_decoder([64, 128, 192, 256]); d2.load_state_dict(ck2["decoder"])
    ced = StudentDecoder(a0c.post_quant_conv, d2.to(dev)).to(dev).eval()
    out["cedit"] = {"enc": lambda x: a0c.encode(x).latent_dist.mean,
                    "dec": lambda z: ced(z), "sf": float(a0c.config.scaling_factor)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--variants", default="sdvae,a0,s10,cedit")
    ap.add_argument("--arms", default="Q2a,Oracle1,B4")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda"

    spec = json.loads(TASKS.read_text(encoding="utf-8"))
    tasks = spec["tasks"]
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"[Tasks] {len(tasks)} "
          f"(cross={sum(1 for t in tasks if t.get('is_cross'))})", flush=True)

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

    src_paths = {}
    for t in tasks:
        if t["source_stem"] not in src_paths:
            src_paths[t["source_stem"]] = resolve(t["source_rel_path"], t["source_stem"])
    print(f"[Tasks] {len(src_paths)} unique sources", flush=True)

    import lpips
    lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    for p in lp.parameters():
        p.requires_grad = False
    import csd_model
    csd, _, _ = csd_model.load_csd(dev)
    csd = csd.float().eval()
    for p in csd.parameters():
        p.requires_grad = False
    from transformers import AutoModel
    dino = AutoModel.from_pretrained("facebook/dinov2-base").to(dev).eval()
    for p in dino.parameters():
        p.requires_grad = False

    proto_d = torch.load(ROOT / "pseudo_gt_24h/csd_proto.pt", map_location="cpu",
                         weights_only=False)
    proto = torch.from_numpy(
        np.stack([np.asarray(proto_d[s], dtype=np.float32) for s in STYLES])).to(dev)
    proto = F.normalize(proto, dim=-1)

    variants = build_variants(dev)
    sdvae = None
    print("\n[Stats] building canonicalizer stats in sd-vae space ...", flush=True)
    from diffusers import AutoencoderKL as _A
    sdvae = _A.from_pretrained(SDVAE, torch_dtype=torch.float32).to(dev).eval()
    st = build_canon_stats(sdvae, dev)
    canon = ContentCanonicalizer(st).to(dev)
    del sdvae
    torch.cuda.empty_cache()

    critic = LatentStyleNet(width=64).to(dev).eval()
    critic.load_state_dict(torch.load(R1_CK, map_location="cpu",
                                      weights_only=False)["model"])
    for p in critic.parameters():
        p.requires_grad = False

    gens = {}
    for name in args.arms.split(","):
        ck = torch.load(ROOT / ARMS[name], map_location="cpu", weights_only=False)
        g = C3UNet(base_dim=16, num_styles=5, mode="direct", inject_16=True,
                   style_keys=False).to(dev)
        g.load_state_dict(ck["model"]); g.eval()
        gens[name] = g
        print(f"[Gen] {name}: {ck['n_params']} params", flush=True)

    use_v = args.variants.split(",")
    rows = []
    for vi, vname in enumerate(use_v):
        cfg = variants[vname]
        sf = cfg["sf"]
        # source round-trip in THIS variant's space (the dT/dS reference)
        print(f"\n=== variant {vname} (sf={sf:.5f}) ===", flush=True)
        stems = sorted(src_paths)
        with torch.inference_mode():
            src_img, src_z = [], {}
            for i in range(0, len(stems), args.bs):
                ch = stems[i:i + args.bs]
                x = torch.stack([load_img(src_paths[s]) for s in ch]).to(dev)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    z = cfg["enc"](x)
                    y = cfg["dec"](z).float()
                src_img.append(y.cpu())
                for j, s in enumerate(ch):
                    src_z[s] = (z[j].float() * sf).cpu()
                del x, z, y
                torch.cuda.empty_cache()
            src_img = torch.cat(src_img)
        idx = {s: k for k, s in enumerate(stems)}
        # score the round-trip reference once
        pil_src = [Image.fromarray(
            (src_img[idx[s]].clamp(-1, 1).add(1).mul(127.5).byte().permute(1, 2, 0).numpy()))
            for s in stems]
        Fs = csd_model.feats_pils(csd, pil_src, batch=16, device=dev)
        Fs = Fs / np.linalg.norm(Fs, axis=1, keepdims=True)
        Ds, _ = _dino(dino, pil_src, dev)

        for arm in args.arms.split(","):
            gen = gens[arm]
            acc = {"top1": 0, "n": 0, "dT": [], "dS": [], "dino": [], "lp": [],
                   "margin": []}
            for i in range(0, len(tasks), args.bs):
                grp = tasks[i:i + args.bs]
                zs = torch.stack([src_z[t["source_stem"]] for t in grp]).to(dev).float()
                tg = torch.tensor([t["target_style_id"] for t in grp], device=dev)
                with torch.inference_mode():
                    c = canon(zs, mode="c1_gentle", augment=False)
                    z0 = gen(c, 0.0, c, tg)
                # rank_direction needs a live autograd graph, so it must sit OUTSIDE
                # inference_mode (the v6 amortize_p0 script does the same)
                if arm == "Oracle1":
                    g0 = rank_direction(critic, z0, tg)
                    zgen = z0 + RMS * g0
                else:
                    zgen = z0
                with torch.inference_mode():
                    raw = zgen / sf
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        y = cfg["dec"](raw).float()
                pil = [Image.fromarray(
                    (y[j].clamp(-1, 1).add(1).mul(127.5).byte().permute(1, 2, 0).cpu().numpy()))
                    for j in range(len(grp))]
                Fv = csd_model.feats_pils(csd, pil, batch=16, device=dev)
                Fv = Fv / np.linalg.norm(Fv, axis=1, keepdims=True)
                Dv, _ = _dino(dino, pil, dev)
                # batch-level LPIPS: one call per microbatch instead of one per task
                src_batch = torch.stack(
                    [src_img[idx[t["source_stem"]]] for t in grp]).to(dev)
                lp_b = lp(y, src_batch).flatten().detach().cpu().numpy()
                proto_np = proto.cpu().numpy()
                for j, t in enumerate(grp):
                    si = idx[t["source_stem"]]
                    ti, soi = t["target_style_id"], t["source_style_id"]
                    sc = Fv[j] @ proto_np.T
                    acc["top1"] += int(int(np.argmax(sc)) == ti)
                    acc["n"] += 1
                    acc["dT"].append(float(Fv[j] @ proto_np[ti] - Fs[si] @ proto_np[ti]))
                    acc["dS"].append(float(Fv[j] @ proto_np[soi] - Fs[si] @ proto_np[soi]))
                    acc["margin"].append(float(sc[ti] - np.max(np.delete(sc, ti))))
                    acc["dino"].append(float(Dv[j] @ Ds[si]))
                    acc["lp"].append(float(lp_b[j]))
                del zs, z0, zgen, raw, y
                torch.cuda.empty_cache()
            r = {"variant": vname, "arm": arm, "n": acc["n"],
                 "top1": round(acc["top1"] / max(acc["n"], 1), 4),
                 "dT": round(float(np.mean(acc["dT"])), 5),
                 "dS": round(float(np.mean(acc["dS"])), 5),
                 "margin": round(float(np.mean(acc["margin"])), 5),
                 "dino_c": round(float(np.mean(acc["dino"])), 5),
                 "lpips": round(float(np.mean(acc["lp"])), 5)}
            rows.append(r)
            print(f"  {arm:<8} Top-1={r['top1']*100:5.2f}%  dT={r['dT']:+.4f}  "
                  f"dS={r['dS']:+.4f}  DINO-C={r['dino_c']:.4f}  LPIPS={r['lpips']:.4f}",
                  flush=True)

    print("\n" + "=" * 96)
    print("SUMMARY: v6 arms x VAE variants (all encoder+decoder paired)")
    print("=" * 96)
    print(f"  {'variant':<9}{'arm':<9}{'Top-1':>8}{'dT':>9}{'dS':>9}{'margin':>9}"
          f"{'DINO-C':>9}{'LPIPS':>8}")
    for r in rows:
        print(f"  {r['variant']:<9}{r['arm']:<9}{r['top1']*100:7.2f}%{r['dT']:+9.4f}"
              f"{r['dS']:+9.4f}{r['margin']:+9.4f}{r['dino_c']:9.4f}{r['lpips']:8.4f}")
    OUT.write_text(json.dumps({"n_tasks": len(tasks), "rows": rows}, indent=2))
    print(f"\n[Saved] {OUT}")


def _dino(m, pils, dev, batch=16):
    outs = []
    with torch.no_grad():
        for i in range(0, len(pils), batch):
            arr = np.stack([np.asarray(p) for p in pils[i:i + batch]])
            x = torch.from_numpy(arr).permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0).to(dev)
            outs.append(dino_feat(m, x, dev).cpu().numpy())
            del x
            torch.cuda.empty_cache()
    return np.concatenate(outs, 0), None


if __name__ == "__main__":
    main()
