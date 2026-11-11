"""WEAVE W-next P0: stratified source-style leakage and Style-ID causal audit.

This version fixes two issues from the previous attempt:
  1. source style is parsed from the source_stem prefix, not ``Path(stem).parent``;
  2. source sampling is stratified (8 images per style) so all five styles are covered;
  3. DINO reference/vector dimensions are normalized to 2-D before matmul.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "12h"))

import csd_model  # noqa: E402
from tools.pseudo_gt.matching import encode_dino, load_dino_model  # noqa: E402
from tools.v5.route_c import csd_prototypes, decode_trt, load_trt, load_vae  # noqa: E402
from tools.v6.canonicalize import ContentCanonicalizer  # noqa: E402
from tools.v6.models_v6 import C3UNet  # noqa: E402
from tools.v6.train_eval_v6 import STYLES  # noqa: E402
from tools.v6_next.probe_style import LatentStyleNet  # noqa: E402

REF_FEATURES = ROOT / "eval_protocol/cache/references_v1_features.pt"
SRC_FEATURES = ROOT / "runs/v6_r2/cache/tasks750_source_features.pt"
DMEAN = None
DSTD = None


def to_pil(t01: torch.Tensor) -> Image.Image:
    arr = (t01.detach().cpu().permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def ensure_2d(x: torch.Tensor) -> torch.Tensor:
    return x[None, :] if x.dim() == 1 else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q2a-ckpt", default="runs/v6/Q2a_embed_style/Q2a_embed_style.pt")
    ap.add_argument("--critic-ckpt", default="runs/v6_next/probes/r1_r1.pt")
    ap.add_argument("--per-style", type=int, default=8)
    ap.add_argument("--out-dir", default="runs/v6_w/p0_style_leakage_audit")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = args.device
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[P0-audit] loading models...", flush=True)
    critic_ck = torch.load(ROOT / args.critic_ckpt, map_location="cpu", weights_only=False)
    critic = LatentStyleNet(width=critic_ck.get("width", 64)).to(device).eval()
    critic.load_state_dict(critic_ck["model"])
    for p in critic.parameters():
        p.requires_grad_(False)

    ck = torch.load(ROOT / args.q2a_ckpt, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    model = C3UNet(base_dim=a.get("base_dim", 16), num_styles=5, mode="direct").to(device).eval()
    model.load_state_dict(ck["model"])
    canon = ContentCanonicalizer(torch.load(
        ROOT / "runs/v6/cache/canonicalizer_stats.pt",
        map_location="cpu", weights_only=False)).to(device).eval()

    vae = load_vae(device)
    trt = load_trt(vae, device)
    csd, _, _ = csd_model.load_csd(device)
    dino = load_dino_model(device=device).eval()
    proto = csd_prototypes()

    global DMEAN, DSTD
    DMEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    DSTD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    refs = torch.load(REF_FEATURES, map_location="cpu", weights_only=False)
    ref_dino = {}
    for s in STYLES:
        x = refs["styles"][s]["dino_cls"].float()
        if x.dim() == 1:
            x = x[None, :]
        ref_dino[s] = F.normalize(x, dim=1).to(device)

    src_cache = torch.load(SRC_FEATURES, map_location="cpu", weights_only=False)
    by_style = {s: [] for s in STYLES}
    for k in sorted(src_cache["z_src"].keys()):
        style = k.split("__", 1)[0]
        if style in by_style:
            by_style[style].append(k)
    stems = []
    for s in STYLES:
        stems.extend(by_style[s][: args.per_style])
    print(f"[P0-audit] using {len(stems)} stratified sources "
          f"({args.per_style} per style)", flush=True)

    # ---------------- Part 1: latent representation audit ----------------
    latent_rows = []
    with torch.inference_mode():
        for k in stems:
            z = src_cache["z_src"][k].to(device)
            c = canon(z, mode="c1_gentle", augment=False)
            src_style = k.split("__", 1)[0]
            sid = STYLES.index(src_style)

            def eval_one(q: torch.Tensor) -> dict:
                pred = int(q[0].argmax())
                sorted_q = q[0].sort(descending=True).values
                return {
                    "pred": pred,
                    "pred_style": STYLES[pred],
                    "source_top1": int(pred == sid),
                    "source_margin": float(q[0, sid] - sorted_q[1]),
                }

            latent_rows.append({
                "stem": k,
                "src_style": src_style,
                "raw": eval_one(critic(z.float())),
                "c1_gentle": eval_one(critic(c.float())),
            })

    latent_summary = {
        "n": len(latent_rows),
        "raw_source_top1": float(np.mean([r["raw"]["source_top1"] for r in latent_rows])),
        "c1_source_top1": float(np.mean([r["c1_gentle"]["source_top1"] for r in latent_rows])),
        "raw_source_margin": float(np.mean([r["raw"]["source_margin"] for r in latent_rows])),
        "c1_source_margin": float(np.mean([r["c1_gentle"]["source_margin"] for r in latent_rows])),
        "per_style": {
            s: {
                "n": sum(1 for r in latent_rows if r["src_style"] == s),
                "raw_top1": float(np.mean([r["raw"]["source_top1"] for r in latent_rows if r["src_style"] == s])),
                "c1_top1": float(np.mean([r["c1_gentle"]["source_top1"] for r in latent_rows if r["src_style"] == s])),
            } for s in STYLES
        },
    }

    # ---------------- Part 2: causal Style-ID response ----------------
    gen_rows = []
    dmean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    dstd = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    with torch.inference_mode():
        for k in stems:
            z = src_cache["z_src"][k].to(device)
            c = canon(z, mode="c1_gentle", augment=False)
            src_style = k.split("__", 1)[0]
            sid = STYLES.index(src_style)
            x01 = src_cache["raw_rgb"][k].to(device)
            src224 = F.interpolate(x01[None], size=(224, 224), mode="bicubic", align_corners=False)
            src_dino = F.normalize(
                dino((src224 - dmean) / dstd).last_hidden_state[:, 0].float(), dim=1)

            for tid, tgt in enumerate(STYLES):
                t = torch.tensor([tid], device=device)
                z_gen = model(c, 0.0, c, t)
                img = decode_trt(trt, vae, z_gen, batch=1)
                pils = [to_pil(img[0])]
                f_gen = csd_model.feats_pils(csd, pils, batch=1, device=device)
                f_gen = f_gen / np.linalg.norm(f_gen, axis=1, keepdims=True)
                sc = f_gen[0] @ proto.T
                pred = int(sc.argmax())
                margin = float(sc[tid] - np.max([sc[j] for j in range(5) if j != tid]))
                dg, _ = encode_dino(dino, pils, batch=1, device=device)
                dg = F.normalize(dg.float(), dim=1)
                if dg.dim() == 1:
                    dg = dg[None, :]
                ref_matrix = ref_dino[tgt]
                if ref_matrix.dim() == 1:
                    ref_matrix = ref_matrix[None, :]
                dino_s = float((dg @ ref_matrix.T).max())
                src_vec = F.normalize(src_dino, dim=1)
                dino_c = float((dg @ src_vec.T).sum())
                gen_rows.append({
                    "stem": k,
                    "src_style": src_style,
                    "src_style_id": sid,
                    "target_style": tgt,
                    "target_style_id": tid,
                    "is_cross": tid != sid,
                    "csd_pred": pred,
                    "csd_top1": int(pred == tid),
                    "csd_margin": margin,
                    "csd_dt": float(sc[tid] - src_cache["raw_csd"][k] @ proto[tid]),
                    "dino_s": dino_s,
                    "dino_c": dino_c,
                })

    def agg(rows, key, value):
        out = {}
        for r in rows:
            out.setdefault(r[key], []).append(r[value])
        return {k: float(np.mean(v)) for k, v in out.items()}

    cross = [r for r in gen_rows if r["is_cross"]]
    same = [r for r in gen_rows if not r["is_cross"]]
    causal_summary = {
        "n_sources": len(stems),
        "n_generated": len(gen_rows),
        "cross_n": len(cross),
        "same_n": len(same),
        "cross_top1": float(np.mean([r["csd_top1"] for r in cross])),
        "cross_margin": float(np.mean([r["csd_margin"] for r in cross])),
        "cross_dino_s": float(np.mean([r["dino_s"] for r in cross])),
        "cross_dino_c": float(np.mean([r["dino_c"] for r in cross])),
        "same_top1": float(np.mean([r["csd_top1"] for r in same])),
        "same_dino_c": float(np.mean([r["dino_c"] for r in same])),
        "per_target_top1": agg(cross, "target_style", "csd_top1"),
        "per_target_dino_s": agg(cross, "target_style", "dino_s"),
        # source-style leakage in generated cross-style outputs
        "cross_source_style_still_predicted": float(np.mean([
            int(r["csd_pred"] == r["src_style_id"]) for r in cross])),
    }

    result = {
        "latent_audit": latent_summary,
        "causal_audit": causal_summary,
        "latent_rows": latent_rows,
        "gen_rows": gen_rows,
    }
    (out_dir / "p0_style_leakage_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")

    lines = []
    lines.append("# WEAVE W-next P0：内容条件源风格泄漏与 Style-ID 因果审计\n")
    lines.append(f"- stratified sources: **{len(stems)}**；generated pairs: **{len(gen_rows)}**"
                 f"（{len(cross)} cross / {len(same)} same）")
    lines.append("")
    lines.append("## 1. 潜空间源风格泄漏（R1 判官）\n")
    lines.append("| Latent | Source-style Top-1 | Source-style Margin |")
    lines.append("|---|---:|---:|")
    lines.append(f"| raw posterior latent | {latent_summary['raw_source_top1']*100:.1f}% | "
                 f"{latent_summary['raw_source_margin']:+.4f} |")
    lines.append(f"| c1_gentle canonicalized | {latent_summary['c1_source_top1']*100:.1f}% | "
                 f"{latent_summary['c1_source_margin']:+.4f} |")
    lines.append("")
    lines.append("### 分风格 source-style Top-1\n")
    lines.append("| Style | n | raw | c1_gentle |")
    lines.append("|---|---:|---:|---:|")
    for s in STYLES:
        p = latent_summary["per_style"][s]
        lines.append(f"| {s} | {p['n']} | {p['raw_top1']*100:.1f}% | {p['c1_top1']*100:.1f}% |")
    lines.append("")
    lines.append("## 2. 固定内容、改变 Style-ID 的因果响应（Cross only）\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    for k in ("cross_top1", "cross_margin", "cross_dino_s", "cross_dino_c",
              "same_top1", "same_dino_c", "cross_source_style_still_predicted"):
        lines.append(f"| {k} | {causal_summary[k]:.4f} |")
    lines.append("")
    lines.append("### 分目标 Top-1\n")
    lines.append("| Target | CSD Top-1 | DINO-S |")
    lines.append("|---|---:|---:|")
    for s in STYLES:
        lines.append(f"| {s} | {causal_summary['per_target_top1'].get(s,0)*100:.1f}% | "
                     f"{causal_summary['per_target_dino_s'].get(s,0):.4f} |")
    lines.append("")
    (out_dir / "p0_style_leakage_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[P0-audit] wrote {out_dir/'p0_style_leakage_audit.json'}")


if __name__ == "__main__":
    main()
