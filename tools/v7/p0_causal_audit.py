"""V7 P0: relative-vs-direct 750 evals are produced by r2_eval; this script adds
the B4 Style-ID causal-response audit and the V7_P0_NOTE.md verdict.

Audit (P0.3, docs/v7/V7_IMPLEMENTATION_PLAN.md):
  - 40 stratified sources (8 per style), B4 direct outputs for all 5 Style-IDs;
  - source-style residual ratio (R1 latent classifier + CSD image classifier);
  - latent RMS / CSD feature differences between the five Style-ID outputs;
  - 20 directed routes: per-direction DINO-S / DINO-C;
  - Q2a audit values are NOT reused for B4.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from tools.v7.v7_common import (
    B4_CKPT, R1_CRITIC, ROOT, STYLES, load_c3unet, load_canon, load_critic,
    model_direct, pick_stratified_sources, tasks_sha256, write_json,
)

DEFAULT_REFS = ROOT / "eval_protocol" / "cache" / "references_v1_features.pt"
SRC_CACHE = ROOT / "runs" / "v6_r2" / "cache" / "tasks750_source_features.pt"
P0_DIR = ROOT / "runs" / "v7" / "p0_relative"


def main() -> int:
    ap = argparse.ArgumentParser(description="V7 P0 B4 causal audit")
    ap.add_argument("--ckpt", default=str(B4_CKPT))
    ap.add_argument("--critic", default=str(R1_CRITIC))
    ap.add_argument("--refs", default=str(DEFAULT_REFS))
    ap.add_argument("--src-cache", default=str(SRC_CACHE))
    ap.add_argument("--per-style", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = args.device

    from tools.v5.route_c import decode_trt, load_trt, load_vae, csd_prototypes
    from tools.v6.train_eval_v6 import csd_model, encode_dino, load_dino_model

    t0 = time.time()
    tasks = json.loads((ROOT / "eval_protocol" / "tasks_750.json").read_text(encoding="utf-8"))["tasks"]
    sources = pick_stratified_sources(tasks, args.per_style)
    print(f"[p0audit] {len(sources)} sources: "
          f"{ {s: sum(1 for t in sources if t['source_style'] == s) for s in STYLES} }", flush=True)

    model = load_c3unet(Path(args.ckpt), device)
    canon = load_canon(device)
    critic = load_critic(Path(args.critic), device)
    src_cache = torch.load(args.src_cache, map_location="cpu", weights_only=False)
    ref_data = torch.load(args.refs, map_location="cpu", weights_only=False)
    proto = csd_prototypes()

    vae = load_vae(device)
    trt = load_trt(vae, device)
    csd, _, _ = csd_model.load_csd(device)
    dino = load_dino_model(device=device)

    ref_dino = {
        s: F.normalize(ref_data["styles"][s]["dino_cls"].float(), dim=-1).numpy()
        for s in STYLES
    }  # (30, 384) per style

    audit = {
        "ckpt": str(args.ckpt),
        "tasks_sha256": tasks_sha256(),
        "per_style": args.per_style,
        "n_sources": len(sources),
        "routes": {},
        "per_target": {},
        "overall": {},
    }
    style_delta = {"ckpt": str(args.ckpt), "sources": {}}
    residual_r1, residual_csd, route_rows = [], [], {}
    out_rows = []

    with torch.inference_mode():
        for i, t in enumerate(sources):
            k = t["source_stem"]
            sid = t["source_style_id"]
            z = src_cache["z_src"][k].to(device)
            # C3UNet requires batch(style_ids) == batch(z); expand the single
            # source latent to 5 copies, one per Style-ID.
            z5 = z.expand(5, -1, -1, -1).contiguous()
            z_out = model_direct(model, z5, canon, "c1_gentle",
                                 torch.arange(5, device=device))
            # latent RMS differences between the five Style-ID outputs
            pair_rms = []
            for a in range(5):
                for b in range(a + 1, 5):
                    pair_rms.append(float(
                        (z_out[a] - z_out[b]).flatten().pow(2).mean().sqrt()))
            q = critic(z_out.float()).float()
            pred_latent = q.argmax(dim=1).tolist()
            residual_r1.extend([1.0 if p == sid else 0.0 for p in pred_latent])

            img = decode_trt(trt, vae, z_out, batch=5)
            u8 = img.mul(255).round().clamp(0, 255).to(torch.uint8).cpu()
            arr = u8.permute(0, 2, 3, 1).numpy()
            pils = [Image.fromarray(arr[j]) for j in range(5)]
            fg = csd_model.feats_pils(csd, pils, batch=5, device=device)
            fg = fg / np.linalg.norm(fg, axis=1, keepdims=True)
            dg, _ = encode_dino(dino, pils, batch=5, device=device)
            dg = F.normalize(dg.float(), dim=-1)
            pred_csd = (fg @ proto.T).argmax(axis=1).tolist()
            residual_csd.extend([1.0 if p == sid else 0.0 for p in pred_csd])

            # CSD feature spread between the five Style-ID outputs (style_delta)
            csd_pair = []
            for a in range(5):
                for b in range(a + 1, 5):
                    csd_pair.append(float(np.linalg.norm(fg[a] - fg[b])))

            style_delta["sources"][k] = {
                "source_style": t["source_style"],
                "pairwise_latent_rms_mean": float(np.mean(pair_rms)),
                "pairwise_csd_l2_mean": float(np.mean(csd_pair)),
                "r1_pred_per_style_id": pred_latent,
            }

            d_raw = src_cache["raw_dino"][k]
            for j in range(5):
                tgt_s = STYLES[j]
                dino_s = float(np.max(dg[j].cpu().numpy() @ ref_dino[tgt_s].T))
                dino_c = float(dg[j].cpu().numpy() @ d_raw)
                out_rows.append({
                    "src": k, "src_style": t["source_style"], "tgt_style": tgt_s,
                    "r1_pred": pred_latent[j], "csd_pred": pred_csd[j],
                    "dino_s": dino_s, "dino_c": dino_c,
                })
                route_rows.setdefault((t["source_style"], tgt_s), []).append(
                    (dino_s, dino_c))
            if (i + 1) % 8 == 0:
                print(f"[p0audit] {i+1}/{len(sources)} sources "
                      f"({time.time()-t0:.0f}s)", flush=True)

    for (ss, ts), vals in sorted(route_rows.items()):
        audit["routes"][f"{ss}->{ts}"] = {
            "n": len(vals),
            "dino_s": float(np.mean([v[0] for v in vals])),
            "dino_c": float(np.mean([v[1] for v in vals])),
        }
    for j, s in enumerate(STYLES):
        rows = [r for r in out_rows if r["tgt_style"] == s]
        audit["per_target"][s] = {
            "n": len(rows),
            "dino_s": float(np.mean([r["dino_s"] for r in rows])),
            "dino_c": float(np.mean([r["dino_c"] for r in rows])),
            "source_residual_r1": float(np.mean(
                [1.0 if r["r1_pred"] == STYLES.index(r["src_style"]) else 0.0
                 for r in rows])),
            "source_residual_csd": float(np.mean(
                [1.0 if r["csd_pred"] == STYLES.index(r["src_style"]) else 0.0
                 for r in rows])),
        }
    audit["overall"] = {
        "source_residual_r1": float(np.mean(residual_r1)),
        "source_residual_csd": float(np.mean(residual_csd)),
        "pairwise_latent_rms_mean": float(np.mean(
            [v["pairwise_latent_rms_mean"] for v in style_delta["sources"].values()])),
        "pairwise_csd_l2_mean": float(np.mean(
            [v["pairwise_csd_l2_mean"] for v in style_delta["sources"].values()])),
    }

    write_json(audit, ROOT / "runs" / "v7" / "p0_b4_causal_audit.json")
    write_json(style_delta, ROOT / "runs" / "v7" / "p0_style_delta.json")
    write_p0_note()
    print(f"[p0audit] done in {time.time()-t0:.0f}s", flush=True)
    return 0


def _load_eval(path: Path) -> dict | None:
    f = path / "eval.json"
    if not f.exists():
        return None
    return json.loads(f.read_text(encoding="utf-8"))


GO_THRESHOLDS = {
    "dino_c_min_delta": -0.005,
    "lpips_max_delta": +0.005,
    "dino_s_gain": +0.002,
}


def write_p0_note() -> None:
    arms = {}
    for name in ["p0_q2a_direct", "p0_q2a_relative", "p0_b4_direct", "p0_b4_relative"]:
        e = _load_eval(P0_DIR / name)
        if e:
            arms[name] = e
    lines = ["# V7 P0 Note — relative style parameterization (auto-generated)",
             "",
             f"Generated by tools/v7/p0_causal_audit.py. Thresholds: {GO_THRESHOLDS}",
             ""]
    if not arms:
        lines.append("(750 evals not found; run r2_eval arms first.)")
        (ROOT / "docs" / "v7" / "V7_P0_NOTE.md").write_text("\n".join(lines), encoding="utf-8")
        return

    lines += ["| arm | Top-1 | DINO-S | DINO-C | LPIPS | dT | Same-Top1 | Same-LPIPS |",
              "|---|---|---|---|---|---|---|---|"]
    for name, e in arms.items():
        c, s = e["cross600"], e["same150"]
        lines.append(
            f"| {name} | {c['top1']*100:.1f} | {c['dino_s']:.4f} | {c['dino_c']:.4f} "
            f"| {c['lpips']:.4f} | {c['dT']:+.4f} | {s['same_top1']*100:.1f} "
            f"| {s['same_lpips']:.4f} |")
    verdict = "NO_RELATIVE"
    reasons = []
    for base, rel in [("p0_b4_direct", "p0_b4_relative"),
                      ("p0_q2a_direct", "p0_q2a_relative")]:
        if base in arms and rel in arms:
            d = arms[rel]["cross600"]
            b = arms[base]["cross600"]
            ds = d["dino_s"] - b["dino_s"]
            dc = d["dino_c"] - b["dino_c"]
            dl = d["lpips"] - b["lpips"]
            lines += ["", f"## {rel} vs {base}",
                      f"- ΔDINO-S {ds:+.4f}, ΔDINO-C {dc:+.4f}, ΔLPIPS {dl:+.4f}"]
            if (ds > GO_THRESHOLDS["dino_s_gain"]
                    and dc >= GO_THRESHOLDS["dino_c_min_delta"]
                    and dl <= GO_THRESHOLDS["lpips_max_delta"]):
                verdict = "GO_RELATIVE"
                reasons.append(f"{rel} qualifies")
            elif abs(ds) <= 0.001 and abs(dc) <= 0.001:
                reasons.append(f"{rel}: Style-ID difference too small")
            else:
                reasons.append(f"{rel}: no qualifying gain")
    audit_path = ROOT / "runs" / "v7" / "p0_b4_causal_audit.json"
    if audit_path.exists():
        a = json.loads(audit_path.read_text(encoding="utf-8"))
        lines += ["", "## B4 causal audit summary",
                  f"- source residual (R1 latent cls): {a['overall']['source_residual_r1']*100:.1f}%",
                  f"- source residual (CSD image cls): {a['overall']['source_residual_csd']*100:.1f}%",
                  f"- mean pairwise latent RMS across 5 Style-IDs: {a['overall']['pairwise_latent_rms_mean']:.4f}",
                  f"- mean pairwise CSD L2 across 5 Style-IDs: {a['overall']['pairwise_csd_l2_mean']:.4f}"]
    lines += ["", f"## Verdict: **{verdict}**", ""]
    lines += [f"- {r}" for r in reasons]
    lines += ["", "P3 rule (V7_IMPLEMENTATION_PLAN.md P0.4): "
              "GO_RELATIVE => relative is the default M1/M2 parameterization; "
              "NO_RELATIVE => M2 may still use relative but M1 stays the structural对照 "
              "and no structural benefit may be claimed."]
    (ROOT / "docs" / "v7" / "V7_P0_NOTE.md").write_text("\n".join(lines), encoding="utf-8")
    write_json({"verdict": verdict, "reasons": reasons,
                "thresholds": GO_THRESHOLDS},
               ROOT / "runs" / "v7" / "p0_verdict.json")
    print(f"[p0audit] V7_P0_NOTE.md verdict={verdict}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
