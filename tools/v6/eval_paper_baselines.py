"""WEAVE V6: Re-evaluate Paper Baselines on New Metrics (CSD dT/dS/dM, DINO-C, DINO-S, LPIPS, CLIP-S)
under both 600 Cross-Style and 750 All Protocol.

Evaluates existing generated PNGs stored in results/D5-512/:
  - identity
  - adain
  - wct
  - cut
  - samst
  - samam
  - sdturbo
  - styleid
  - styleshot
  - stytr2
  - aespa
  - weave (published WEAVE)
  - seedream

Outputs:
  - runs/v6/baseline_reval_summary.json
  - docs/v6/BASELINE_REVALUATION_REPORT.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools" / "12h"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "trt"))

import paths
import csd_model
from tools.eval_fast import MetricEngine, Timer, build_protocol
from tools.pseudo_gt.matching import encode_dino, load_dino_model
from utils.run_evaluation import _load_eval_image_tensor, to_lpips_input
from tools.v5.route_c import csd_prototypes, load_vae, load_trt, decode_trt

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
TEST_DIR = REPO_ROOT / "data" / "test"
RESULTS_D5 = REPO_ROOT / "results" / "D5-512"
V6_ROOT = REPO_ROOT / "runs" / "v6"
DOCS_V6 = REPO_ROOT / "docs" / "v6"


def boot_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 0) -> List[float]:
    if len(x) == 0:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def index_source_images() -> Dict[str, Path]:
    """Map stem -> Path for all test images."""
    out = {}
    for s in STYLES:
        sdir = TEST_DIR / s
        if sdir.exists():
            for p in sdir.glob("*.jpg"):
                out[p.stem] = p
                # also without style prefix
                if p.stem.startswith(f"{s}__"):
                    out[p.stem[len(s) + 2:]] = p
                elif p.stem.startswith(f"{s}_"):
                    out[p.stem[len(s) + 1:]] = p
    return out


def parse_filename(p: Path, src_index: Dict[str, Path]) -> dict | None:
    stem = p.stem
    tgt_style = None
    left = None
    for s in STYLES:
        for sep in ["__to__", "_to_"]:
            token = f"{sep}{s}"
            if stem.endswith(token):
                tgt_style = s
                left = stem[:-len(token)]
                break
        if tgt_style:
            break

    if not tgt_style or not left:
        return None

    src_style = None
    core_stem = None
    for s in STYLES:
        for pref in [f"{s}__", f"{s}_"]:
            if left.startswith(pref):
                src_style = s
                core_stem = left[len(pref):]
                while core_stem.startswith(f"{s}__") or core_stem.startswith(f"{s}_"):
                    core_stem = core_stem.split("__", 1)[-1] if "__" in core_stem else core_stem.split("_", 1)[-1]
                break
        if src_style:
            break

    if not src_style or not core_stem:
        return None

    src_path = src_index.get(f"{src_style}__{core_stem}") or src_index.get(core_stem)
    if not src_path:
        for cand_k, cand_p in src_index.items():
            if (cand_k in core_stem or core_stem in cand_k) and cand_p.parent.name == src_style:
                src_path = cand_p
                break

    if not src_path or not src_path.exists():
        return None

    return {
        "file": p.name,
        "path": p,
        "src_style": src_style,
        "tgt_style": tgt_style,
        "src_path": src_path,
        "src_stem": src_path.stem,
        "cross": (src_style != tgt_style),
    }


def evaluate_method_images(
    method_name: str,
    method_dir: Path,
    src_index: Dict[str, Path],
    csd: nn.Module,
    dino: nn.Module,
    lpips_fn: nn.Module,
    clip_model,
    clip_proc,
    proto_csd: np.ndarray,
    ref_dino_cls: Dict[str, torch.Tensor],
    ref_clip_proto: Dict[str, torch.Tensor],
    src_base_cache: Dict[str, dict],
    device: str = "cuda",
    batch_size: int = 32,
) -> dict:
    imgs = sorted(list(method_dir.glob("*.png")) + list(method_dir.glob("*.jpg")))
    parsed = [parse_filename(p, src_index) for p in imgs]
    valid = [r for r in parsed if r is not None]

    if not valid:
        print(f"[{method_name}] WARN: 0 valid pairs found in {method_dir}")
        return {}

    print(f"[{method_name}] Found {len(valid)} valid pairs (total imgs: {len(imgs)})")

    # Evaluate in batches
    rows = []
    t0 = time.time()

    with torch.inference_mode():
        for i in range(0, len(valid), batch_size):
            chunk = valid[i : i + batch_size]
            pils_gen = [Image.open(r["path"]).convert("RGB").resize((512, 512), Image.BICUBIC) for r in chunk]
            pils_src = [Image.open(r["src_path"]).convert("RGB").resize((512, 512), Image.BICUBIC) for r in chunk]

            # 1. CSD features
            csd_gen = csd_model.feats_pils(csd, pils_gen, batch=len(pils_gen), device=device)

            # 2. DINO CLS features
            dino_cls_g, _ = encode_dino(dino, pils_gen, batch=len(pils_gen), device=device)
            dino_cls_g = F.normalize(dino_cls_g.float(), dim=-1)

            # 3. LPIPS Alex
            def to_t(im):
                return torch.from_numpy(np.array(im, dtype=np.float32)).permute(2, 0, 1).to(device) / 127.5 - 1.0

            gt = torch.stack([to_t(im) for im in pils_gen])
            st = torch.stack([to_t(im) for im in pils_src])
            lps = lpips_fn(gt, st).flatten().cpu().tolist()

            # 4. CLIP-S
            clip_inputs = clip_proc(images=pils_gen, return_tensors="pt").to(device)
            clip_feats = clip_model.get_image_features(**clip_inputs)
            if hasattr(clip_feats, "pooler_output") and torch.is_tensor(clip_feats.pooler_output):
                clip_feats = clip_feats.pooler_output
            clip_feats = F.normalize(clip_feats.float(), dim=-1).cpu()

            for j, r in enumerate(chunk):
                sname = r["src_style"]
                tname = r["tgt_style"]
                tid = STYLES.index(tname)
                sid = STYLES.index(sname)

                base_info = src_base_cache[r["src_stem"]]
                cs_b = base_info["csd"]
                cls_b = base_info["dino_cls"]

                cg = csd_gen[j]
                dg = dino_cls_g[j].cpu()

                csd_dT = float(cg @ proto_csd[tid]) - float(cs_b @ proto_csd[tid])
                csd_dS = float(cg @ proto_csd[sid]) - float(cs_b @ proto_csd[sid])
                csd_dM = csd_dT - csd_dS

                dino_c = float(torch.dot(dg, cls_b))
                dino_s = float((dg.to(device) @ ref_dino_cls[tname].T).max().item())

                proto_c = ref_clip_proto[tname]
                clip_s = float(torch.dot(clip_feats[j], proto_c))

                rows.append({
                    "src_style": sname,
                    "tgt_style": tname,
                    "cross": r["cross"],
                    "csd_dT": csd_dT,
                    "csd_dS": csd_dS,
                    "csd_dM": csd_dM,
                    "dino_c": dino_c,
                    "dino_s": dino_s,
                    "clip_s": clip_s,
                    "lpips": float(lps[j]),
                })

    eval_sec = round(time.time() - t0, 1)

    def agg(group_rows: list[dict]) -> dict:
        if not group_rows:
            return {}
        dT = np.array([r["csd_dT"] for r in group_rows])
        dS = np.array([r["csd_dS"] for r in group_rows])
        dM = np.array([r["csd_dM"] for r in group_rows])
        dc = np.array([r["dino_c"] for r in group_rows])
        ds = np.array([r["dino_s"] for r in group_rows])
        cl = np.array([r["clip_s"] for r in group_rows])
        lp = np.array([r["lpips"] for r in group_rows])
        return {
            "n": len(group_rows),
            "CSD_dT": float(dT.mean()),
            "CSD_dT_ci": boot_ci(dT),
            "CSD_dS": float(dS.mean()),
            "CSD_dS_ci": boot_ci(dS),
            "CSD_dM": float(dM.mean()),
            "DINO_C": float(dc.mean()),
            "DINO_S": float(ds.mean()),
            "CLIP_S": float(cl.mean()),
            "LPIPS": float(lp.mean()),
            "frac_dT_positive": float((dT > 0).mean()),
        }

    cross_rows = [r for r in rows if r["cross"]]
    same_rows = [r for r in rows if not r["cross"]]

    per_tgt = {}
    for s in STYLES:
        c_sub = [r["csd_dT"] for r in cross_rows if r["tgt_style"] == s]
        per_tgt[s] = float(np.mean(c_sub)) if c_sub else 0.0

    res = {
        "method": method_name,
        "n_total": len(rows),
        "n_cross": len(cross_rows),
        "n_same": len(same_rows),
        "eval_wall_sec": eval_sec,
        "cross600": agg(cross_rows),
        "same150": agg(same_rows),
        "all750": agg(rows),
        "per_target_cross_CSD_dT": per_tgt,
        "styles_improved": int(sum(1 for v in per_tgt.values() if v > 0)),
        "ukiyo_dT": per_tgt.get("Ukiyo_e", 0.0),
    }
    print(f"[{method_name}] Done in {eval_sec}s: cross600 CSD_dT={res['cross600']['CSD_dT']:+.4f} "
          f"DINO_C={res['cross600']['DINO_C']:.4f} LPIPS={res['cross600']['LPIPS']:.4f} styles+={res['styles_improved']}/5")
    return res


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    V6_ROOT.mkdir(parents=True, exist_ok=True)
    DOCS_V6.mkdir(parents=True, exist_ok=True)

    print("[Baselines] Indexing source images in data/test...", flush=True)
    src_index = index_source_images()
    print(f"[Baselines] Indexed {len(src_index)} source stems.")

    print("[Baselines] Loading CSD, DINOv2, LPIPS, CLIP...", flush=True)
    csd, _, _ = csd_model.load_csd(device)
    proto_csd = csd_prototypes()
    dino = load_dino_model(device=device)

    import lpips
    lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    from transformers import CLIPModel, CLIPProcessor
    clip_cache = str(paths.eval_cache_root("hf") / "hub")
    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32", local_files_only=True, cache_dir=clip_cache
    ).to(device).eval()
    clip_proc = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32", local_files_only=True, cache_dir=clip_cache
    )

    # Precompute reference representations for target styles
    sel_path = REPO_ROOT / "runs" / "p0" / "reference_selection.json"
    sel = json.loads(sel_path.read_text())["selected"]
    ref_dino_cls = {}
    ref_clip_proto = {}
    with torch.inference_mode():
        for s in STYLES:
            ref_imgs = [Image.open(p).convert("RGB").resize((512, 512), Image.BICUBIC) for p in sel[s]["r0"][:30]]
            c_dino, _ = encode_dino(dino, ref_imgs, batch=8, device=device)
            ref_dino_cls[s] = F.normalize(c_dino.float(), dim=-1)

            clip_in = clip_proc(images=ref_imgs, return_tensors="pt").to(device)
            f_clip = clip_model.get_image_features(**clip_in)
            if hasattr(f_clip, "pooler_output") and torch.is_tensor(f_clip.pooler_output):
                f_clip = f_clip.pooler_output
            ref_clip_proto[s] = F.normalize(f_clip.float().mean(dim=0, keepdim=True), dim=-1)[0].cpu()

    # Precompute source base representations (VAE round-trip reference per V4 protocol)
    print("[Baselines] Precomputing source base representations...", flush=True)
    vae = load_vae(device)
    trt = load_trt(vae, device)
    src_base_cache = {}

    stems_sorted = sorted(list({p.stem for p in src_index.values()}))
    with torch.inference_mode():
        for s in range(0, len(stems_sorted), 8):
            chunk = stems_sorted[s : s + 8]
            tensors = torch.stack([_load_eval_image_tensor(src_index[stem], size=512) for stem in chunk]).to(device)
            # VAE encode -> decode to match V4 base_pil
            inp = (tensors * 2.0 - 1.0).half()
            z = (vae.encode(inp).latent_dist.mean * vae.config.scaling_factor).half()
            img_b = decode_trt(trt, vae, z, batch=8)
            u8_b = img_b.mul(255.0).round().clamp(0, 255).to(torch.uint8).cpu().permute(0, 2, 3, 1).numpy()
            pils_b = [Image.fromarray(u8_b[j]) for j in range(len(chunk))]

            csd_b = csd_model.feats_pils(csd, pils_b, batch=len(pils_b), device=device)
            dino_cls_b, _ = encode_dino(dino, pils_b, batch=len(pils_b), device=device)
            dino_cls_b = F.normalize(dino_cls_b.float(), dim=-1).cpu()

            for j, stem in enumerate(chunk):
                src_base_cache[stem] = {
                    "csd": csd_b[j],
                    "dino_cls": dino_cls_b[j],
                }
    del vae, trt
    torch.cuda.empty_cache()
    print(f"[Baselines] Precomputed {len(src_base_cache)} source base features.")

    # All candidate baseline folders in results/D5-512
    method_folders = [
        "identity",
        "adain",
        "wct",
        "cut",
        "samst",
        "samam",
        "sdturbo",
        "styleid",
        "styleshot",
        "stytr2",
        "aespa",
        "weave",
        "seedream",
    ]

    all_summaries = {}
    for m in method_folders:
        mdir = RESULTS_D5 / m
        if not mdir.exists():
            print(f"[Baselines] Skipping missing directory: {mdir}")
            continue
        print(f"\n================ Evaluating Baseline: {m} ================")
        res = evaluate_method_images(
            m, mdir, src_index, csd, dino, lpips_fn, clip_model, clip_proc,
            proto_csd, ref_dino_cls, ref_clip_proto, src_base_cache, device=device,
        )
        if res:
            all_summaries[m] = res

    # Save summary JSON
    out_json = V6_ROOT / "baseline_reval_summary.json"
    out_json.write_text(json.dumps(all_summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[Baselines] Successfully saved {out_json}")

    # Generate Markdown Report
    lines = []
    lines.append("# 论文 Baseline 在新指标体系下的全量重评测报告 (D5-512)\n")
    lines.append("> 评测环境：RTX 3060 (12GB)，严格按照 V4/V5/V6 统一独立新指标协议重新运行：")
    lines.append("> - CSD (ViT-L, 官方 768-d 原型)")
    lines.append("> - DINOv2-small (CLS 内容余弦 DINO-C, 30-ref max 风格相似度 DINO-S)")
    lines.append("> - LPIPS (AlexNet)")
    lines.append("> - CLIP-S (ViT-B/32 风格原型余弦)\n")
    lines.append("## 1. 主判据：600 跨风格任务 (cross600) 对比表\n")
    lines.append("| 方法 / Baseline | 样本数 (cross) | CSD $\\Delta T \\uparrow$ | CSD $\\Delta S$ | CSD $\\Delta M \\uparrow$ | DINO-C $\\uparrow$ | DINO-S $\\uparrow$ | LPIPS $\\downarrow$ | CLIP-S $\\uparrow$ | 5类风格改善数 | Ukiyo-e $\\Delta T$ |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for m, d in all_summaries.items():
        c = d["cross600"]
        lines.append(
            f"| **{m}** | {c['n']} | **{c['CSD_dT']:+.4f}** | {c['CSD_dS']:+.4f} | **{c['CSD_dM']:+.4f}** | "
            f"**{c['DINO_C']:.4f}** | {c['DINO_S']:.4f} | **{c['LPIPS']:.4f}** | {c['CLIP_S']:.4f} | "
            f"{d['styles_improved']}/5 | {d['ukiyo_dT']:+.4f} |"
        )

    lines.append("\n## 2. 独立约束：150 同风格任务 (same150) 身份保持表\n")
    lines.append("| 方法 / Baseline | 样本数 (same) | CSD $\\Delta S$ (漂移) | DINO-C $\\uparrow$ | DINO-S $\\uparrow$ | LPIPS $\\downarrow$ | CLIP-S $\\uparrow$ |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    for m, d in all_summaries.items():
        s = d["same150"]
        if not s:
            continue
        lines.append(
            f"| **{m}** | {s['n']} | **{s['CSD_dS']:+.4f}** | **{s['DINO_C']:.4f}** | "
            f"{s['DINO_S']:.4f} | **{s['LPIPS']:.4f}** | {s['CLIP_S']:.4f} |"
        )

    lines.append("\n## 3. 历史全量：750 全量任务 (all750) 对照表\n")
    lines.append("| 方法 / Baseline | 样本数 (all) | CSD $\\Delta T \\uparrow$ | CSD $\\Delta M \\uparrow$ | DINO-C $\\uparrow$ | DINO-S $\\uparrow$ | LPIPS $\\downarrow$ | CLIP-S $\\uparrow$ |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    for m, d in all_summaries.items():
        a = d["all750"]
        lines.append(
            f"| **{m}** | {a['n']} | **{a['CSD_dT']:+.4f}** | **{a['CSD_dM']:+.4f}** | "
            f"**{a['DINO_C']:.4f}** | {a['DINO_S']:.4f} | **{a['LPIPS']:.4f}** | {a['CLIP_S']:.4f} |"
        )

    out_md = DOCS_V6 / "BASELINE_REVALUATION_REPORT.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Baselines] Successfully wrote {out_md}")


if __name__ == "__main__":
    main()
