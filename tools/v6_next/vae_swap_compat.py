"""Are the candidate VAEs latent-compatible, and how much do they cost on the eval protocol?

The v6 operator (Q2a / R1) lives in the sd-vae-ft-ema latent space. Swapping the
VAE only makes sense if the replacement's latent distribution is on the same scale;
otherwise the operator is simply out of distribution and any Top-1 collapse says
nothing about the VAE.

Two blocks, both on the 150 protocol test images:

  A. LATENT COMPATIBILITY
     For each encoder: per-channel mean/std, global RMS, and the ratio to sd-vae.
     Also the reconstruction round-trip in the SAME space.

  B. RECONSTRUCTION CONTROL (no operator)
     encode -> decode with each variant, then CSD-keep / LPIPS / DINO-C against the
     original PNG. This isolates each VAE's own fidelity inside the eval protocol,
     which is the number the operator numbers must be compared against.

Variants:
  sdvae  sd-vae-ft-ema encoder + decoder   (what the published v6 numbers used)
  a0     A0 encoder + A0 decoder           (our fine-tuned VAE)
  s10    A0 encoder + S10 decoder          (10.66M decoder, A0 latent space)
  cedit  C_edit encoder + C_edit decoder   (jointly retrained style VAE)
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/root/Workspace/WEAVE_VAE")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "12h"))

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from PIL import Image

SDVAE = "/root/Workspace/xy/DiT/data/pretrained/sd-vae-ft-ema"
A0 = str(ROOT / "runs_vae/A0/final_diffusers_vae")
S10_CK = ROOT / "runs/vae_decoder_prune/s10_p5_lowlr/decoder.pt"
CED_CK = ROOT / "runs/vae_style_vae/C_edit_best.pt"
TESTSET = ROOT / "deliver/testset/data/test"
TASKS = ROOT / "deliver/testset/protocol/tasks_750.json"
OUT = ROOT / "deliver/vae_swap_compat.json"

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
NM = [0.48145466, 0.4578275, 0.40821073]
NS = [0.26862954, 0.26130258, 0.27577711]


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


def build_variants(dev):
    """Each variant owns its own modules.

    NOTE: an earlier version loaded C_edit's encoder into the SAME AutoencoderKL that
    served the a0/s10 variants, which silently made all three share C_edit's encoder
    while keeping A0's decoder. That produced cos=1.000 between a0/s10/cedit and a
    collapsed a0 reconstruction. Separate instances are mandatory.
    """
    from tools.vae_suite.decoder_prune import build_decoder, StudentDecoder

    out = {}

    # 1) sd-vae-ft-ema: the space the published v6 numbers live in
    sd = AutoencoderKL.from_pretrained(SDVAE, torch_dtype=torch.float32).to(dev).eval()
    out["sdvae"] = {
        "enc": lambda x: sd.encode(x).latent_dist.mean,
        "dec": lambda z: sd.decode(z).sample,
        "sf": float(getattr(sd.config, "scaling_factor", 0.18215)),
    }

    # 2) A0 as delivered (untouched encoder + its own decoder)
    a0 = AutoencoderKL.from_pretrained(A0, torch_dtype=torch.float32).to(dev).eval()
    out["a0"] = {
        "enc": lambda x: a0.encode(x).latent_dist.mean,
        "dec": lambda z: a0.decode(z).sample,
        "sf": float(getattr(a0.config, "scaling_factor", 0.18215)),
    }

    # 3) S10 decoder on a SEPARATE A0 instance
    a0b = AutoencoderKL.from_pretrained(A0, torch_dtype=torch.float32).to(dev).eval()
    ck = torch.load(S10_CK, map_location="cpu", weights_only=False)
    d = build_decoder(ck["meta"]["widths_forward"])
    d.load_state_dict(ck["state_dict"])
    s10 = StudentDecoder(a0b.post_quant_conv, d.to(dev)).to(dev).eval()
    out["s10"] = {"enc": lambda x: a0b.encode(x).latent_dist.mean,
                  "dec": lambda z: s10(z),
                  "sf": float(getattr(a0b.config, "scaling_factor", 0.18215))}

    # 4) C_edit: its OWN encoder AND decoder, on a third instance
    a0c = AutoencoderKL.from_pretrained(A0, torch_dtype=torch.float32).to(dev).eval()
    ck2 = torch.load(CED_CK, map_location="cpu", weights_only=False)
    a0c.encoder.load_state_dict(ck2["encoder"])
    d2 = build_decoder([64, 128, 192, 256])
    d2.load_state_dict(ck2["decoder"])
    ced = StudentDecoder(a0c.post_quant_conv, d2.to(dev)).to(dev).eval()
    out["cedit"] = {"enc": lambda x: a0c.encode(x).latent_dist.mean,
                    "dec": lambda z: ced(z),
                    "sf": float(getattr(a0c.config, "scaling_factor", 0.18215))}

    for p_ in out.values():
        for m in ("enc", "dec"):
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda"

    spec = json.loads(TASKS.read_text(encoding="utf-8"))
    seen, items = set(), []
    for t in spec["tasks"]:
        st = t["source_stem"]
        if st in seen:
            continue
        seen.add(st)
        items.append((st, t["source_style"], t["source_rel_path"]))
    if args.limit:
        items = items[: args.limit]
    print(f"[Data] {len(items)} unique protocol sources", flush=True)

    def resolve(rel, stem):
        # protocol paths are data/test/<style>/<file>.jpg; delivered layout matches
        p = ROOT / "deliver/testset" / rel
        if p.exists():
            return p
        for q in TESTSET.rglob(Path(rel).name):
            return q
        cand = list(TESTSET.rglob(f"*{stem}*"))
        if cand:
            return cand[0]
        raise FileNotFoundError(rel)

    paths = []
    for stem, style, rel in items:
        paths.append((stem, style, resolve(rel, stem)))
    print(f"[Data] resolved {len(paths)}/{len(items)} paths", flush=True)

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

    variants = build_variants(dev)

    # ---------------- A. latent compatibility ----------------
    print("\n" + "=" * 92)
    print("A. LATENT COMPATIBILITY (150 protocol sources)")
    print("=" * 92, flush=True)
    stats = {}
    Z = {v: [] for v in variants}
    with torch.no_grad():
        for i in range(0, len(paths), args.bs):
            chunk = paths[i:i + args.bs]
            x = torch.stack([load_img(p) for _, _, p in chunk]).to(dev)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                for v, cfg in variants.items():
                    z = cfg["enc"](x).float()
                    Z[v].append(z.cpu())
            del x
            torch.cuda.empty_cache()
    for v in variants:
        Z[v] = torch.cat(Z[v])
        z = Z[v].float()
        stats[v] = {
            "sf": variants[v]["sf"],
            "rms_raw": float(z.pow(2).mean().sqrt()),
            "rms_scaled": float((z * variants[v]["sf"]).pow(2).mean().sqrt()),
            "mean": float(z.mean()), "std": float(z.std()),
            "ch_mean": [round(float(t), 4) for t in z.mean(dim=(0, 2, 3))],
            "ch_std": [round(float(t), 4) for t in z.std(dim=(0, 2, 3))],
        }
    ref = stats["sdvae"]["rms_scaled"]
    print(f"  {'variant':<10}{'sf':>8}{'RMS raw':>10}{'RMS scaled':>12}"
          f"{'ratio vs sdvae':>16}{'global std':>12}")
    for v, s in stats.items():
        print(f"  {v:<10}{s['sf']:>8.4f}{s['rms_raw']:>10.4f}{s['rms_scaled']:>12.4f}"
              f"{s['rms_scaled']/ref:>16.3f}{s['std']:>12.4f}")
    # cross-space correlation: is A0's latent even aligned with sdvae's?
    print("\n  cosine between latents of the SAME image across encoders:")
    for a in stats:
        row = []
        for b in stats:
            za = Z[a].flatten(1)
            zb = Z[b].flatten(1)
            c = F.cosine_similarity(za, zb, dim=1).mean()
            row.append(f"{float(c):+.3f}")
        print(f"    {a:<8}" + "".join(f"{r:>9}" for r in row))

    # ---------------- B. reconstruction control ----------------
    print("\n" + "=" * 92)
    print("B. RECONSTRUCTION CONTROL (encode -> decode, no operator)")
    print("=" * 92, flush=True)
    recon = {}
    for v, cfg in variants.items():
        l1s, lps, csds, dinos = [], [], [], []
        with torch.no_grad():
            for i in range(0, len(paths), args.bs):
                chunk = paths[i:i + args.bs]
                x = torch.stack([load_img(p) for _, _, p in chunk]).to(dev)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    z = cfg["enc"](x)
                    y = cfg["dec"](z).float()
                l1s.append(float((y - x).abs().mean()))
                lps.append(float(lp(y, x).mean()))
                csds.append(float((csd_feat(csd, y, dev) *
                                   csd_feat(csd, x, dev)).sum(-1).mean()))
                dinos.append(float((dino_feat(dino, y, dev) *
                                    dino_feat(dino, x, dev)).sum(-1).mean()))
                del x, z, y
                torch.cuda.empty_cache()
        recon[v] = {"l1": round(float(np.mean(l1s)), 5),
                    "lpips": round(float(np.mean(lps)), 5),
                    "csd_keep": round(float(np.mean(csds)), 5),
                    "dino_c": round(float(np.mean(dinos)), 5)}
        print(f"  {v:<10} L1={recon[v]['l1']:.5f}  LPIPS={recon[v]['lpips']:.5f}  "
              f"CSD-keep={recon[v]['csd_keep']:.5f}  DINO-C={recon[v]['dino_c']:.5f}",
              flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"n_sources": len(paths), "latent_stats": stats,
                               "reconstruction": recon}, indent=2))
    print(f"\n[Saved] {OUT}")


if __name__ == "__main__":
    main()
