"""WEAVE W-next P0: content-condition source-style leakage & Style-ID causal response.

This audits whether the current generator actually uses Style-ID to change style,
or whether the content condition still carries enough source-style information.

Part 1 -- latent representation audit
    Classify raw posterior-mean latents vs c1_gentle canonicalized latents with the
    frozen R1 critic. If c1_gentle retains strong source-style classification, the
    content condition is still leaking source style.

Part 2 -- causal Style-ID response
    For the same source latent, force all five Style-IDs and decode each output.
    Measure CSD target-margin / Top-1, DINO-S, and DINO-C. A useful conditional
    operator should show a strong diagonal response (target style follows the
    requested Style-ID), not a fixed response dominated by source style.

Outputs:
    runs/v6_w/p0_style_leakage_audit.json
    runs/v6_w/p0_style_leakage_audit.md
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
from tools.pseudo_gt.matching import load_dino_model  # noqa: E402
from tools.v5.route_c import csd_prototypes, load_vae, load_trt, decode_trt  # noqa: E402
from tools.v6.canonicalize import ContentCanonicalizer  # noqa: E402
from tools.v6.models_v6 import C3UNet  # noqa: E402
from tools.v6.train_eval_v6 import STYLES  # noqa: E402
from tools.v6_next.probe_style import LatentStyleNet  # noqa: E402

REF_FEATURES = ROOT / "eval_protocol/cache/references_v1_features.pt"
SRC_FEATURES = ROOT / "runs/v6_r2/cache/tasks750_source_features.pt"


def to_pil(t01):
    arr = (t01.detach().cpu().permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q2a-ckpt", default="runs/v6/Q2a_embed_style/Q2a_embed_style.pt")
    ap.add_argument("--critic-ckpt", default="runs/v6_next/probes/r1_r1.pt")
    ap.add_argument("--sources", type=int, default=40)
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

    refs = torch.load(REF_FEATURES, map_location="cpu", weights_only=False)
    ref_dino = {
        s: F.normalize(refs["styles"][s]["dino_cls"].float(), dim=-1).to(device)
        for s in STYLES
    }

    src_cache = torch.load(SRC_FEATURES, map_location="cpu", weights_only=False)
    stems = sorted(src_cache["z_src"].keys())[: args.sources]
    print(f"[P0-audit] using {len(stems)} sources", flush=True)

    # ---------------- Part 1: latent style leakage ----------------
    latent_rows = []
    with torch.inference_mode():
        for k in stems:
            z = src_cache["z_src"][k].to(device)
            c = canon(z, mode="c1_gentle", augment=False)
            q_raw = critic(z.float()).float()
            q_canon = critic(c.float()).float()
            src_style = Path(k).parent.name
            sid = STYLES.index(src_style) if src_style in STYLES else -1

            def row(q):
                pred = int(q[0].argmax())
                sorted_q = q[0].sort(descending=True).values
                return {
                    "pred": pred,
                    "target_top1": int(pred == sid),
                    "target_margin": float(q[0, sid] - sorted_q[1]) if sid >= 0 else float("nan"),
                    "top2_gap": float(sorted_q[0] - sorted_q[1]),
                }

            latent_rows.append({
                "stem": k,
                "src_style": src_style,
                "raw": row(q_raw),
                "c1_gentle": row(q_canon),
            })

    def summarize_latent(field):
        n = len(latent_rows)
        raw_acc = np.mean([r[field]["target_top1"] for r in latent_rows])
        canon_acc = np.mean([r[field]["target_top1"] for r in latent_rows])
        raw_margin = np.nanmean([r[field]["target_margin"] for r in latent_rows])
        canon_margin = np.nanmean([r[field]["target_margin"] for r in latent_rows])
        return {
            "n": n,
            "raw_source_style_top1": float(raw_acc),
            "c1_gentle_source_style_top1": float(canon_acc),
            "raw_source_style_margin": float(raw_margin),
            "c1_gentle_source_style_margin": float(canon_margin),
        }

    latent_summary = {
        "raw": summarize_latent("raw"),
        "c1_gentle": summarize_latent("c1_gentle"),
        "per_style": {},
    }
    for s in STYLES:
        subset = [r for r in latent_rows if r["src_style"] == s]
        latent_summary["per_style"][s] = {
            "n": len(subset),
            "raw_top1": float(np.mean([r["raw"]["target_top1"] for r in subset])),
            "c1_top1": float(np.mean([r["c1_gentle"]["target_top1"] for r in subset])),
            "raw_margin": float(np.nanmean([r["raw"]["target_margin"] for r in subset])),
            "c1_margin": float(np.nanmean([r["c1_gentle"]["target_margin"] for r in subset])),
        }

    # ---------------- Part 2: causal Style-ID response ----------------
    gen_rows = []
    with torch.inference_mode():
        for k in stems:
            z = src_cache["z_src"][k].to(device)
            c = canon(z, mode="c1_gentle", augment=False)
            src_style = Path(k).parent.name
            sid = STYLES.index(src_style) if src_style in STYLES else 0
            x01 = src_cache["raw_rgb"][k].to(device)
            src224 = F.interpolate(x01[None], size=(224, 224), mode="bicubic", align_corners=False)
            src_dino = dcls = F.normalize(
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
                dino_s = float(dg @ ref_dino[tgt].T.max())
                dino_c = float((dg @ src_dino.T).sum())
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
        "per_target_margin": agg(cross, "target_style", "csd_margin"),
        "per_target_dino_s": agg(cross, "target_style", "dino_s"),
        "per_target_dino_c": agg(cross, "target_style", "dino_c"),
        # source-style leakage in generated cross-style outputs: how often the CSD
        # judge still picks the ORIGINAL source style instead of the requested target
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

    # ---------------- Markdown ----------------
    lines = []
    lines.append("# WEAVE W-next P0：内容条件源风格泄漏与 Style-ID 因果审计\n")
    lines.append(f"- Sources: **{len(stems)}**；generated pairs: **{len(gen_rows)}**"
                 f"（含 {len(cross)} cross / {len(same)} same）")
    lines.append("")
    lines.append("## 1. 潜空间源风格泄漏（R1 判官）\n")
    lines.append("| Latent | Source-style Top-1 | Source-style Margin |")
    lines.append("|---|---:|---:|")
    lines.append(f"| raw posterior latent | {latent_summary['raw']['raw_source_style_top1']*100:.1f}% | "
                 f"{latent_summary['raw']['raw_source_style_margin']:+.4f} |")
    lines.append(f"| c1_gentle canonicalized | {latent_summary['c1_gentle']['c1_gentle_source_style_top1']*100:.1f}% | "
                 f"{latent_summary['c1_gentle']['c1_gentle_source_style_margin']:+.4f} |")
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
