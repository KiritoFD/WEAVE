"""V6 24h plan section 4 - zero-training EXACT compression of the direct operator.

Two provable equivalences in direct mode, folded into a smaller deployment checkpoint:

1. Stem 12ch -> 4ch.
   The caller is `model(c, 0.0, c, s)`, so `z_in == c` and
   `x0 = cat([z_in, c, z_in-c]) = cat([c, c, 0])`.
   Therefore `Conv2d(12, c1, 3)` on x0 equals `Conv2d(4, c1, 3)` with
   `W4 = W12[:, 0:4] + W12[:, 4:8]` and the same bias.

2. Constant tau folded into the AdaLN biases.
   `tau` is always 0.0 in direct mode, so `t_emb = time_mlp(sin(0))` is a constant vector
   and `cond = s_emb + t_const`. Every consumer is `Linear(W, b)` with `out = W@cond + b`,
   hence `W@(s_emb + t_const) + b = W@s_emb + (b + W@t_const)`.
   Each such Linear gets `b += W @ t_const`, and the time network is dropped.

This is a DEPLOYMENT compression, not a claim that training at 4ch would also be lossless.
Verification is output-for-output on real latents, in fp32 and fp16.

Usage
    python3 -m tools.v6_24h.fold_direct --ckpt runs/v6/D1_c1_gentle/D1_c1_gentle.pt \
        --out runs/v6_24h/folded/A0_folded.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from tools.v6.models_v6 import C3UNet, sinusoidal_embedding

CACHE = Path("/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16")
FILES = ["00_Early_Renaissance.pt", "01_Impressionism.pt", "02_Minimalism.pt",
         "03_Rococo.pt", "04_Ukiyo_e.pt"]

# every Linear whose input is `cond`
COND_LINEARS = ["enc64_1", "enc64_2", "enc32_1", "enc32_2", "mid16_1", "mid16_2",
                "dec32_1", "dec32_2", "dec64_1", "dec64_2"]


def fold_state(src: C3UNet) -> dict:
    """Produce the folded state_dict (exact)."""
    t_const = src.time_mlp(sinusoidal_embedding(
        torch.zeros(1), src.base_dim)).detach()[0]          # (cond_dim,)

    new = {k: v.clone() for k, v in src.state_dict().items()}
    # 1) stem 12 -> 4
    w = src.stem64.weight.detach()
    new["stem64.weight"] = (w[:, 0:4] + w[:, 4:8]).clone()
    # 2) fold t_const into every cond-consuming Linear bias
    folded_biases = {}
    for name in COND_LINEARS:
        key = f"{name}.adaln"
        L = getattr(src, name).adaln
        new[f"{key}.bias"] = (L.bias.detach() + L.weight.detach() @ t_const).clone()
        folded_biases[key] = float((L.weight.detach() @ t_const).abs().max())
    new["out_adaln.bias"] = (src.out_adaln.bias.detach()
                             + src.out_adaln.weight.detach() @ t_const).clone()
    folded_biases["out_adaln"] = float((src.out_adaln.weight.detach() @ t_const).abs().max())
    # 3) drop the time network entirely
    for k in [k for k in new if k.startswith("time_mlp.")]:
        del new[k]
    return new, folded_biases, t_const


def build_folded(src: C3UNet) -> C3UNet:
    dst = C3UNet(base_dim=src.base_dim, num_styles=src.num_styles, mode=src.mode,
                 no_style=src.no_style, content_mode=src.content_mode,
                 inject_64=src.inject_64, inject_32=src.inject_32,
                 inject_16=src.inject_16, folded=True)
    new, _, _ = fold_state(src)
    missing, unexpected = dst.load_state_dict(new, strict=False)
    # time_mlp.* is intentionally absent from the folded model
    missing = [m for m in missing if not m.startswith("time_mlp.")]
    if missing or unexpected:
        raise RuntimeError(f"fold mismatch: missing={missing} unexpected={unexpected}")
    return dst


def real_latents(n_per_style: int = 2) -> torch.Tensor:
    """A few real latents from the training caches, standardized per style."""
    zs = []
    for f in FILES:
        d = torch.load(CACHE / f, map_location="cpu", weights_only=False)
        lat = d["latents"].float()
        zs.append(lat[:n_per_style])
    return torch.cat(zs, 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="runs/v6_24h/folded/folded.pt")
    ap.add_argument("--n-per-style", type=int, default=2)
    args = ap.parse_args()

    ck = torch.load(ROOT / args.ckpt if not Path(args.ckpt).is_absolute() else args.ckpt,
                    map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    src = C3UNet(base_dim=a.get("base_dim", 48), num_styles=5,
                 mode=a.get("mode", "direct"),
                 no_style=bool(a.get("no_style", False)),
                 content_mode=a.get("content_mode", "full"),
                 inject_64=not a.get("no_inject_64", False),
                 inject_32=not a.get("no_inject_32", False),
                 inject_16=not a.get("no_inject_16", False))
    src.load_state_dict(ck["model"])
    src.eval()

    dst = build_folded(src).eval()
    _, folded_bias_max, t_const = fold_state(src)

    n_src = sum(p.numel() for p in src.parameters())
    n_dst = sum(p.numel() for p in dst.parameters())

    # --- output-for-output verification on real latents ---
    z = real_latents(args.n_per_style)                     # (N, 4, 64, 64)
    N = z.shape[0]
    s = torch.arange(N) % 5
    report = {}
    import copy
    for dtype, tag in ((torch.float32, "fp32"), (torch.float16, "fp16")):
        # cast the MODEL too: a half input against float weights is not a valid AMP
        # emulation and raises "Input type (Half) and bias type (float)".
        m_src = copy.deepcopy(src).to(dtype).eval()
        m_dst = copy.deepcopy(dst).to(dtype).eval()
        zs = z.to(dtype)
        with torch.no_grad():
            o_src = m_src(zs, 0.0, zs, s).float()
            o_dst = m_dst(zs, 0.0, zs, s).float()
        d = (o_src - o_dst).abs()
        rel = d.max() / max(o_src.abs().max().item(), 1e-12)
        report[tag] = {"max_abs": float(d.max()), "mean_abs": float(d.mean()),
                       "rel_max": float(rel),
                       "ref_abs_max": float(o_src.abs().max())}
        print(f"[fold] {tag}: max|d|={d.max():.3e} mean|d|={d.mean():.3e} "
              f"rel={rel:.3e} (ref |out|max={o_src.abs().max():.3f})")

    ok_fp32 = report["fp32"]["max_abs"] < 1e-4
    ok_fp16 = report["fp16"]["rel_max"] < 5e-3
    print(f"[fold] params {n_src:,} -> {n_dst:,} "
          f"({(n_dst-n_src)/n_src*100:+.2f}%)")
    print(f"[fold] equivalence: fp32 {'OK' if ok_fp32 else 'FAIL'}, "
          f"fp16 {'OK' if ok_fp16 else 'FAIL'}")
    print(f"[fold] folded bias shift max per layer: "
          f"{max(folded_bias_max.values()):.4e}; |t_const|max={t_const.abs().max():.4e}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": dst.state_dict(),
                "args": {**a, "folded": True},
                "source_ckpt": str(args.ckpt),
                "fold_report": {"params_before": n_src, "params_after": n_dst,
                                "equivalence": report,
                                "fp32_ok": ok_fp32, "fp16_ok": ok_fp16,
                                "folded_bias_max": folded_bias_max,
                                "note": "exact deployment compression for direct mode; "
                                        "NOT a claim that training at 4ch is lossless"}},
               out)
    print(f"[fold] wrote {out}")
    if not (ok_fp32 and ok_fp16):
        sys.exit(1)


if __name__ == "__main__":
    main()
