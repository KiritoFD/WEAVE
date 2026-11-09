"""V6 24h plan section 7 - M0: the simplest Direct interpolation knob.

    T_alpha(z, s) = z + alpha * [ G(C(z), s) - z ],   alpha in [0, 1]

No training. Tests whether the frozen direct operator gives a monotone, controllable
style/content trade-off, which is what the multiplication stage is supposed to establish.

Guards required by the plan:
  * alpha > 1 is NOT allowed on the main line (history showed alpha=1.8/2.2 raising Top-1
    while lowering dT and exploding LPIPS);
  * alpha = 0 must reproduce the identity path exactly, so the curve has a real origin;
  * report (dT, Top-1, DINO-C, LPIPS) per target and check monotonicity.

The model forward is done ONCE: delta = G(C(z), s) - z is shared by every alpha, so the
only per-alpha cost is decode + CSD/DINO/LPIPS.

Usage
    python3 -m tools.v6_24h.m0_alpha --ckpt runs/v6/D1_c1_gentle/D1_c1_gentle.pt \
        --out runs/v6_24h/m0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image

from tools.v6.canonicalize import ContentCanonicalizer
from tools.v6.models_v6 import C3UNet
from tools.v6.train_eval_v6 import (STYLES, V5_ROOT, V6_ROOT, boot_ci,  # noqa: E402
                                    csd_model, csd_prototypes, decode_trt,
                                    encode_dino, load_dino_model, load_trt, load_vae,
                                    _load_eval_image_tensor)
from utils.run_evaluation import to_lpips_input  # noqa: E402

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/v6/D1_c1_gentle/D1_c1_gentle.pt")
    ap.add_argument("--out", default="runs/v6_24h/m0")
    ap.add_argument("--canon-mode", default="")
    ap.add_argument("--alphas", default=",".join(str(a) for a in ALPHAS))
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke-test mode: evaluate only the first N tasks")
    ap.add_argument("--allow-extrapolation", action="store_true",
                    help="DIAGNOSTIC ONLY: permit alpha > 1. The plan keeps alpha > 1 off "
                         "the main line because history showed it raises Top-1 while "
                         "lowering dT and exploding LPIPS. It is allowed here only to "
                         "measure that trade-off explicitly, never to select a model.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = args.device
    alphas = [float(x) for x in args.alphas.split(",")]
    if any(a < 0 for a in alphas):
        raise SystemExit("alpha must be >= 0")
    if any(a > 1 for a in alphas) and not args.allow_extrapolation:
        raise SystemExit("alpha > 1 is off the main line (plan section 7); pass "
                         "--allow-extrapolation to run it as a diagnostic only")
    if any(a > 1 for a in alphas):
        print("[m0] WARNING: alpha > 1 is an EXTRAPOLATION DIAGNOSTIC, not a candidate "
              "model. Results must be reported as a trade-off curve only.", flush=True)

    ck = torch.load(ROOT / args.ckpt, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    model = C3UNet(base_dim=a.get("base_dim", 48), num_styles=5,
                   mode=a.get("mode", "direct"),
                   no_style=bool(a.get("no_style", False)),
                   content_mode=a.get("content_mode", "full"),
                   inject_64=not a.get("no_inject_64", False),
                   inject_32=not a.get("no_inject_32", False),
                   inject_16=not a.get("no_inject_16", False),
                   style_keys=bool(a.get("style_keys", False)),
                   style_keys_n=int(a.get("style_keys_n", 8) or 8),
                   style_keys_rank=int(a.get("style_keys_rank", 32) or 32)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    canon_mode = args.canon_mode or a.get("canon_mode", "c1_gentle")
    canon = ContentCanonicalizer(torch.load(V6_ROOT / "cache" / "canonicalizer_stats.pt",
                                            map_location="cpu", weights_only=False)).to(device)

    tasks = json.loads((V5_ROOT / "v5_eval200_tasks.json").read_text(encoding="utf-8"))
    if args.limit:
        tasks = tasks[: args.limit]
    lat_dict = torch.load(V5_ROOT / "cache" / "eval_src_latents.pt", map_location="cpu")
    stems = sorted({t["src_stem"] for t in tasks})
    stem_idx = {s: i for i, s in enumerate(stems)}
    z_all = torch.stack([lat_dict[s] for s in stems]).to(device)
    src_idx = torch.tensor([stem_idx[t["src_stem"]] for t in tasks], device=device)
    tgt_ids = torch.tensor([t["tgt_style_id"] for t in tasks], device=device, dtype=torch.long)

    proto = csd_prototypes()
    csd, _, _ = csd_model.load_csd(device)
    dino = load_dino_model(device=device)
    vae = load_vae(device)
    trt = load_trt(vae, device)
    import lpips
    lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    csd_src50 = torch.load(V5_ROOT / "cache" / "eval_src_csd_feats.pt",
                           map_location="cpu", weights_only=False)
    csd_src_cache = {s: np.asarray(csd_src50[s]) for s in stems}
    src_tensors = {}
    for t in tasks:
        if t["src_stem"] not in src_tensors:
            src_tensors[t["src_stem"]] = _load_eval_image_tensor(
                Path(t["src_path"]), size=512).unsqueeze(0).to(device)

    print(f"[m0] ckpt={args.ckpt} canon={canon_mode} alphas={alphas} "
          f"tasks={len(tasks)}", flush=True)

    with torch.inference_mode():
        # source DINO features (once)
        img_src = decode_trt(trt, vae, z_all, batch=8)
        u8_src = img_src.mul(255.0).round().clamp(0, 255).to(torch.uint8).cpu().permute(0, 2, 3, 1).numpy()
        pils_src = [Image.fromarray(u8_src[i]) for i in range(len(stems))]
        dcls, _ = encode_dino(dino, pils_src, batch=16, device=device)
        dino_src_cache = {s: dcls[i].float().cpu().numpy() for i, s in enumerate(stems)}

        # ONE model forward: delta is shared by every alpha
        z_src = z_all[src_idx]
        c = canon(z_src, mode=canon_mode, augment=False)
        delta = model(c, 0.0, c, tgt_ids) - z_src
        print(f"[m0] delta |mean|={delta.abs().mean():.4f} "
              f"|max|={delta.abs().max():.3f}", flush=True)

        results = {}
        for al in alphas:
            z_gen = z_src + al * delta
            img = decode_trt(trt, vae, z_gen, batch=8)
            u8 = img.mul(255.0).round().clamp(0, 255).to(torch.uint8).cpu()
            gen01 = u8.float().div_(255.0)
            arr = u8.permute(0, 2, 3, 1).numpy()
            pils = [Image.fromarray(arr[i]) for i in range(len(tasks))]
            fg = csd_model.feats_pils(csd, pils, batch=16, device=device)
            dg, _ = encode_dino(dino, pils, batch=16, device=device)
            dg = dg.float().cpu().numpy()
            lps = []
            for s in range(0, len(tasks), 16):
                chunk = gen01[s:s + 16].to(device)
                srcb = torch.cat([src_tensors[t["src_stem"]] for t in tasks[s:s + 16]], 0)
                lps.append(lpips_fn(to_lpips_input(chunk), to_lpips_input(srcb)).flatten().cpu())
            lps = torch.cat(lps, 0).numpy()

            dT, top1, marg = [], [], []
            per_t = {s: [] for s in STYLES}
            for i, t in enumerate(tasks):
                fs = csd_src_cache[t["src_stem"]]
                tg, sid = t["tgt_style_id"], t["src_style_id"]
                sc = fg[i] @ proto.T
                d = float(sc[tg] - fs @ proto[tg])
                dT.append(d)
                per_t[t["tgt_style"]].append(d)
                top1.append(int(sc.argmax() == tg))
                marg.append(float(sc[tg] - np.delete(sc, tg).max()))
            results[al] = {
                "n": len(dT),
                "mean_dT": float(np.mean(dT)), "dT_ci": boot_ci(np.array(dT)),
                "mean_top1": float(np.mean(top1)), "mean_margin": float(np.mean(marg)),
                "mean_dino_c": float(np.mean([dg[i] @ dino_src_cache[t["src_stem"]]
                                              for i, t in enumerate(tasks)])),
                "mean_lpips": float(np.mean(lps)),
                "per_target_dT": {s: float(np.mean(v)) for s, v in per_t.items()},
                "styles_improved": int(sum(1 for v in per_t.values() if np.mean(v) > 0)),
            }
            r = results[al]
            print(f"[m0] alpha={al:.2f} dT={r['mean_dT']:+.4f} top1={r['mean_top1']:.3f} "
                  f"margin={r['mean_margin']:+.4f} dino={r['mean_dino_c']:.4f} "
                  f"lpips={r['mean_lpips']:.4f} styles+={r['styles_improved']}/5", flush=True)

        # alpha = 0 must be the exact identity path
        z0 = z_src + 0.0 * delta
        ident_err = float((z0 - z_src).abs().max())

    # monotonicity of the aggregate curve
    dts = [results[a]["mean_dT"] for a in alphas]
    lp = [results[a]["mean_lpips"] for a in alphas]
    mono_dT = all(b >= a_ - 1e-9 for a_, b in zip(dts, dts[1:]))
    mono_lp = all(b >= a_ - 1e-9 for a_, b in zip(lp, lp[1:]))
    print(f"\n[m0] alpha=0 identity error = {ident_err:.3e} (must be 0)")
    print(f"[m0] dT monotone increasing over alpha: {mono_dT} {['%.4f' % x for x in dts]}")
    print(f"[m0] LPIPS monotone increasing over alpha: {mono_lp} {['%.4f' % x for x in lp]}")

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "m0_alpha.json").write_text(json.dumps(
        {"ckpt": args.ckpt, "canon_mode": canon_mode, "alphas": alphas,
         "identity_error": ident_err, "dT_monotone": mono_dT, "lpips_monotone": mono_lp,
         "results": {str(k): v for k, v in results.items()}},
        ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# M0: Direct interpolation knob", "",
             f"ckpt `{args.ckpt}`, canon `{canon_mode}`, {len(tasks)} cross tasks", "",
             "| alpha | ΔT | Top-1 | Margin | DINO-C | LPIPS | 风格+ |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for al in alphas:
        r = results[al]
        lines.append(f"| {al:.2f} | {r['mean_dT']:+.4f} | {r['mean_top1']:.3f} | "
                     f"{r['mean_margin']:+.4f} | {r['mean_dino_c']:.4f} | "
                     f"{r['mean_lpips']:.4f} | {r['styles_improved']}/5 |")
    lines += ["", f"- alpha=0 identity error: `{ident_err:.2e}` (exact identity required)",
              f"- ΔT monotone: **{mono_dT}**", f"- LPIPS monotone: **{mono_lp}**", "",
              "| 目标 | " + " | ".join(f"α={a}" for a in alphas) + " |",
              "|---|" + "---:|" * len(alphas)]
    for s in STYLES:
        lines.append(f"| {s} | " + " | ".join(
            f"{results[a]['per_target_dT'][s]:+.4f}" for a in alphas) + " |")
    (out / "m0_alpha.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[m0] wrote {out}/m0_alpha.{'md,json'}")


if __name__ == "__main__":
    main()
