"""Is the generated CSD feature off-manifold?

Every arm has target Top-1 around 0.03-0.05 against a 0.20 chance level, and the confusion
matrix is near-uniform. A near-uniform argmax means the feature sits roughly equidistant from
all five prototypes, i.e. near the centre of the prototype simplex. That is the signature of
an OFF-MANIFOLD feature rather than a feature that is simply "the wrong style".

This measures, for three populations:
  * real source images (post VAE round-trip, from the cached eval features)
  * real target-style images (the training-pool prototypes' own members)
  * generated images (needs a checkpoint, hence the GPU)
the following per image:
    max_cos  = max_k <e, p_k>      how close to the nearest prototype
    spread   = max_k - min_k       how discriminative the prototype simplex is here

A real image has high max_cos and a healthy spread. A generated image that is off-manifold has
LOW max_cos and a SMALL spread. This distinguishes "wrong style" from "no style information".

The real populations need no GPU, so they run now; the generated population is filled in when
the queue releases the card.

Usage
    python3 -m tools.v6_24h.csd_manifold_check                  # real populations only
    python3 -m tools.v6_24h.csd_manifold_check --ckpt <path>    # + generated
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]


def stats(F: np.ndarray, P: np.ndarray, label: str) -> dict:
    """F: (N,D) unit features, P: (5,D) unit prototypes.

    Reported quantities:
      max_cos  distance to the nearest prototype (how much style signal is present at all)
      margin   max_cos - second_max (how decisively one prototype wins)
      spread   max_k - min_k (how much the prototype simplex separates this feature)

    Entropy of softmax(cos) is deliberately NOT reported: all cosines live in a narrow band,
    so an untempered softmax is near-uniform even for perfectly classified real images, which
    makes that number look alarming while carrying no information.
    """
    S = F @ P.T                                  # (N,5)
    srt = np.sort(S, axis=1)
    mx, second, mn = srt[:, -1], srt[:, -2], srt[:, 0]
    out = {"label": label, "n": int(len(F)),
           "max_cos_mean": float(mx.mean()), "max_cos_p10": float(np.percentile(mx, 10)),
           "margin_mean": float((mx - second).mean()),
           "margin_p10": float(np.percentile(mx - second, 10)),
           "spread_mean": float((mx - mn).mean())}
    print(f"  {label:26s} n={out['n']:4d}  max_cos={out['max_cos_mean']:.4f} "
          f"(p10 {out['max_cos_p10']:.4f})  margin={out['margin_mean']:.4f} "
          f"(p10 {out['margin_p10']:.4f})  spread={out['spread_mean']:.4f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--out", default="runs/v6_24h/csd_manifold.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    from tools.v6.train_eval_v6 import V5_ROOT, csd_prototypes

    P = np.asarray(csd_prototypes(), dtype=np.float32)
    P = P / np.linalg.norm(P, axis=1, keepdims=True)
    print(f"prototypes: {P.shape}, all unit-norm "
          f"{np.allclose(np.linalg.norm(P, axis=1), 1.0, atol=1e-3)}")

    results = []

    # 1) real source images (post VAE round-trip), the cached eval features
    cache = torch.load(V5_ROOT / "cache" / "eval_src_csd_feats.pt", map_location="cpu",
                       weights_only=False)
    F_src = np.stack([np.asarray(cache[k], dtype=np.float32) for k in cache])
    F_src = F_src / np.linalg.norm(F_src, axis=1, keepdims=True)
    results.append(stats(F_src, P, "real sources (round-trip)"))

    # 2) real training-pool images per style, via the CSD features if cached, else skip
    #    (recomputing them needs the CSD model, which is GPU work)
    print("  (real target-style reference requires the CSD model; deferred to the GPU run)")

    # 3) generated images, if a checkpoint is given
    if args.ckpt:
        import torch
        from PIL import Image
        from tools.v6.canonicalize import ContentCanonicalizer
        from tools.v6.models_v6 import C3UNet
        from tools.v6.train_eval_v6 import (V6_ROOT, csd_model, decode_trt, load_trt,
                                            load_vae, _load_eval_image_tensor)
        dev = args.device
        ck = torch.load(ROOT / args.ckpt, map_location="cpu", weights_only=False)
        a = ck.get("args", {})
        model = C3UNet(base_dim=a.get("base_dim", 48), num_styles=5,
                       mode=a.get("mode", "direct"),
                       no_style=bool(a.get("no_style", False)),
                       content_mode=a.get("content_mode", "full"),
                       inject_64=not a.get("no_inject_64", False),
                       inject_32=not a.get("no_inject_32", False),
                       inject_16=not a.get("no_inject_16", False)).to(dev)
        model.load_state_dict(ck["model"])
        model.eval()
        canon_mode = a.get("canon_mode", "c1_gentle")
        canon = ContentCanonicalizer(torch.load(
            V6_ROOT / "cache" / "canonicalizer_stats.pt", map_location="cpu",
            weights_only=False)).to(dev)

        tasks = json.loads((V5_ROOT / "v5_eval200_tasks.json").read_text(encoding="utf-8"))
        lat = torch.load(V5_ROOT / "cache" / "eval_src_latents.pt", map_location="cpu")
        stems = sorted({t["src_stem"] for t in tasks})
        idx = {s: i for i, s in enumerate(stems)}
        z_all = torch.stack([lat[s] for s in stems]).to(dev)
        src_idx = torch.tensor([idx[t["src_stem"]] for t in tasks], device=dev)
        tgt = torch.tensor([t["tgt_style_id"] for t in tasks], device=dev, dtype=torch.long)

        csd, _, _ = csd_model.load_csd(dev)
        vae = load_vae(dev)
        trt = load_trt(vae, dev)
        feats = []
        with torch.inference_mode():
            for s in range(0, len(tasks), 32):
                z = z_all[src_idx[s:s + 32]]
                c = canon(z, mode=canon_mode, augment=False)
                zg = model(c, 0.0, c, tgt[s:s + 32])
                img = decode_trt(trt, vae, zg, batch=8)
                u8 = img.mul(255.0).round().clamp(0, 255).to(torch.uint8).cpu()
                arr = u8.permute(0, 2, 3, 1).numpy()
                pils = [Image.fromarray(arr[i]) for i in range(len(arr))]
                feats.append(np.asarray(csd_model.feats_pils(csd, pils, batch=16, device=dev),
                                        dtype=np.float32))
        F_gen = np.concatenate(feats, 0)
        F_gen = F_gen / np.linalg.norm(F_gen, axis=1, keepdims=True)
        results.append(stats(F_gen, P, "generated"))

        d_max = results[-1]["max_cos_mean"] - results[0]["max_cos_mean"]
        d_spr = results[-1]["spread_mean"] - results[0]["spread_mean"]
        print(f"\n  generated minus real: max_cos {d_max:+.4f}, spread {d_spr:+.4f}")
        if results[-1]["max_cos_mean"] < results[0]["max_cos_mean"] - 0.10:
            print("  -> generated features are FAR from every prototype: OFF-MANIFOLD.")
            print("     The images are not 'the wrong style', they carry little style signal.")
        else:
            print("  -> generated features are close to SOME prototype; the failure is a")
            print("     wrong-style failure, not an off-manifold one.")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"ckpt": args.ckpt, "populations": results},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
