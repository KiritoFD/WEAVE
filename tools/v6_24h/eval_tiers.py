"""V6 24h plan section 1/8 - evaluate a checkpoint on Heldout400 or Same150.

Val200's source caches only cover its own 50 sources, so this evaluator encodes any source
it needs on the fly (VAE posterior mean, exactly the frozen protocol) and computes the
source CSD / DINO features from the decoded posterior-mean image, matching how
`eval_src_csd_feats.pt` was produced.

Reports the same metrics as Val200 plus a source-clustered CI and the 5x5 confusion matrix,
and stores per-task rows so paired clustered comparisons work.

Same150 is run through the NETWORK with target = source style. Code-level identity is
explicitly NOT used: the plan forbids substituting it for measured self-consistency.

Usage
    python3 -m tools.v6_24h.eval_tiers --ckpt runs/v6/D1_c1_gentle/D1_c1_gentle.pt \
        --tasks runs/v6_24h/heldout400_tasks.json --out runs/v6_24h/eval_heldout400_A0
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
from tools.v6.train_eval_v6 import (STYLES, V6_ROOT, csd_model, csd_prototypes,
                                    decode_trt, encode_dino, load_dino_model, load_trt,
                                    load_vae, _load_eval_image_tensor)
from tools.v6_24h.stats import cluster_ci
from utils.run_evaluation import to_lpips_input


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--canon-mode", default="")
    ap.add_argument("--weights", choices=["raw", "ema"], default="raw")
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke-test mode: evaluate only the first N tasks")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = args.device

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

    spec = json.loads((ROOT / args.tasks).read_text(encoding="utf-8"))
    tasks = spec["tasks"]
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"[tiers] {spec['protocol']['name']}: {len(tasks)} tasks, ckpt={args.ckpt}, "
          f"canon={canon_mode}" + (" [SMOKE]" if args.limit else ""), flush=True)

    proto = csd_prototypes()
    csd, _, _ = csd_model.load_csd(device)
    dino = load_dino_model(device=device)
    vae = load_vae(device)
    trt = load_trt(vae, device)
    import lpips
    lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    # ---- encode every unique source once (posterior mean, frozen protocol) ----
    def src_path(t):
        return t.get("source_rel_path") or t.get("src_path")

    def src_key(t):
        return t.get("source_stem") or t.get("src_stem") or Path(src_path(t)).stem

    uniq = {}
    for t in tasks:
        uniq.setdefault(src_key(t), src_path(t))
    keys = sorted(uniq)
    z_src, csd_src, dino_src, rgb_src = {}, {}, {}, {}
    sf = float(vae.config.scaling_factor)
    # the VAE runs in fp16, so the encoder input must match its dtype
    vae_dtype = next(vae.parameters()).dtype
    print(f"[tiers] vae dtype = {vae_dtype}", flush=True)
    with torch.inference_mode():
        for s in range(0, len(keys), 16):
            chunk = keys[s:s + 16]
            x = torch.stack([_load_eval_image_tensor(Path(uniq[k]), size=512) for k in chunk],
                            0).to(device)
            z = vae.encode(x.to(vae_dtype)).latent_dist.mean * sf
            img = decode_trt(trt, vae, z, batch=8)
            u8 = img.mul(255.0).round().clamp(0, 255).to(torch.uint8).cpu()
            arr = u8.permute(0, 2, 3, 1).numpy()
            pils = [Image.fromarray(arr[i]) for i in range(len(chunk))]
            f = csd_model.feats_pils(csd, pils, batch=16, device=device)
            d, _ = encode_dino(dino, pils, batch=16, device=device)
            d = d.float().cpu().numpy()
            for i, k in enumerate(chunk):
                z_src[k] = z[i:i + 1]
                csd_src[k] = f[i]
                dino_src[k] = d[i]
                rgb_src[k] = torch.from_numpy(arr[i].copy()).permute(2, 0, 1).float() / 255.0
            print(f"[tiers] encoded {min(s + 16, len(keys))}/{len(keys)} sources", flush=True)

        # ---- generate + score ----
        rows = []
        for s in range(0, len(tasks), 8):
            grp = tasks[s:s + 8]
            z = torch.cat([z_src[src_key(t)] for t in grp], 0)
            tg = torch.tensor([STYLES.index(t["target_style"]) for t in grp], device=device)
            c = canon(z, mode=canon_mode, augment=False)
            z_gen = model(c, 0.0, c, tg) if model.mode == "direct" else None
            img = decode_trt(trt, vae, z_gen, batch=8)
            u8 = img.mul(255.0).round().clamp(0, 255).to(torch.uint8).cpu()
            gen01 = u8.float().div_(255.0)
            arr = u8.permute(0, 2, 3, 1).numpy()
            pils = [Image.fromarray(arr[i]) for i in range(len(grp))]
            fg = csd_model.feats_pils(csd, pils, batch=16, device=device)
            dg, _ = encode_dino(dino, pils, batch=16, device=device)
            dg = dg.float().cpu().numpy()
            lp = lpips_fn(to_lpips_input(gen01.to(device)),
                          to_lpips_input(torch.stack([rgb_src[src_key(t)] for t in grp]).to(device))
                          ).flatten().cpu().numpy()
            for i, t in enumerate(grp):
                k = src_key(t)
                tid = STYLES.index(t["target_style"])
                sid = STYLES.index(t["source_style"])
                sc = fg[i] @ proto.T
                fs = csd_src[k]
                rows.append({
                    "task_id": t["task_id"], "src_stem": k,
                    "src_style": t["source_style"], "tgt_style": t["target_style"],
                    "dT": float(sc[tid] - fs @ proto[tid]),
                    "dS": float(sc[sid] - fs @ proto[sid]),
                    "target_margin": float(sc[tid] - np.delete(sc, tid).max()),
                    "target_top1": int(sc.argmax() == tid),
                    "pred_class": int(sc.argmax()),
                    "dino_c": float(dg[i] @ dino_src[k]),
                    "lpips": float(lp[i]),
                })

    dT = np.array([r["dT"] for r in rows])
    dS = np.array([r["dS"] for r in rows])
    cl = [r["src_stem"] for r in rows]
    per_style = {s: float(np.mean([r["dT"] for r in rows if r["tgt_style"] == s]))
                 for s in STYLES}
    conf = np.zeros((5, 5), dtype=int)
    for r in rows:
        conf[STYLES.index(r["tgt_style"]), r["pred_class"]] += 1
    res = {
        "protocol": spec["protocol"], "ckpt": args.ckpt, "canon_mode": canon_mode,
        "n": len(rows),
        "mean_dT": float(dT.mean()), "mean_dS": float(dS.mean()),
        "mean_dM": float((dT - dS).mean()),
        "mean_top1": float(np.mean([r["target_top1"] for r in rows])),
        "mean_margin": float(np.mean([r["target_margin"] for r in rows])),
        "mean_dino_c": float(np.mean([r["dino_c"] for r in rows])),
        "mean_lpips": float(np.mean([r["lpips"] for r in rows])),
        "per_target_dT": per_style,
        "styles_improved": int(sum(1 for v in per_style.values() if v > 0)),
        "dT_cluster_ci": cluster_ci(dT, cl),
        "dS_cluster_ci": cluster_ci(dS, cl),
        "confusion_target_vs_pred": conf.tolist(), "confusion_styles": STYLES,
        "rows": rows,
    }
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    print(f"\n[tiers] {spec['protocol']['name']}  n={res['n']}")
    print(f"  dT={res['mean_dT']:+.4f} "
          f"cluster CI [{res['dT_cluster_ci']['ci'][0]:+.4f}, "
          f"{res['dT_cluster_ci']['ci'][1]:+.4f}]  "
          f"({res['dT_cluster_ci']['n_clusters']} clusters)")
    print(f"  dS={res['mean_dS']:+.4f}  top1={res['mean_top1']:.3f}  "
          f"margin={res['mean_margin']:+.4f}")
    print(f"  DINO-C={res['mean_dino_c']:.4f}  LPIPS={res['mean_lpips']:.4f}  "
          f"styles+={res['styles_improved']}/5")
    print(f"  per target dT: { {k: round(v,4) for k,v in per_style.items()} }")
    print(f"  wrote {out}/eval.json")


if __name__ == "__main__":
    main()
