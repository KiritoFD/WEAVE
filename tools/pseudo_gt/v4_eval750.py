"""V4 完整 750 评估：历史口径 + 新协议，一份表。

历史口径（与 A0=0.49023 同协议，来自 pgt_render 的 MetricEngine）：
  DINO-S（dino_style_similarity，目标风格，30 refs 取 max，512px，8 步 ODE）
  DINO-C（dino_content_similarity，对源图）、LPIPS-Alex、CLIP-S
新协议（由本脚本从 PNG 现算，基线 = VAE(x) 同编码/解码口径）：
  CSD ΔT / ΔS / ΔM；DINO ΔM（ΔT_dino−ΔS_dino）；Patch-SWD（DINO patch 对目标风格 ref 的 SWD）

统计口径三组：cross（600）/ same（150）/ all（750）。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools" / "12h"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "trt"))

import csd_model  # type: ignore  # noqa: E402
from swd import sliced_wasserstein_distance  # noqa: E402

from tools.eval_fast import _load_eval_image_tensor, build_protocol  # noqa: E402
from tools.pseudo_gt import q0_quality as Q  # noqa: E402
from tools.pseudo_gt.build_assets import STYLES, pgt_root  # noqa: E402
from tools.pseudo_gt.matching import encode_dino  # noqa: E402


def _to_t(im: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(im, dtype=np.float32)).permute(2, 0, 1) / 127.5 - 1


def _boot_ci(x: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    if len(x) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--render-root", default="")
    ap.add_argument("--test-dir", default=str(REPO_ROOT / "data" / "test"))
    ap.add_argument("--max-src", type=int, default=30)
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root = Path(args.render_root) if args.render_root else Path(pgt_root()) / "eval_render"
    out_dir = root / args.cell
    img_dir = out_dir / "images"
    test_images, _req, style_meta = build_protocol(
        Path(args.test_dir), max_src_samples=args.max_src, ref_pool="first")
    names = {sid: style_meta[sid]["name"] for sid in style_meta}
    idx_by_name = {v: k for k, v in names.items()}

    csv_rows = {}
    mpath = out_dir / "metrics.csv"
    if mpath.exists():
        with mpath.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                csv_rows[(r.get("src_image", ""), r.get("tgt_style", ""))] = r
    dino_rows = {}
    dpath = out_dir / "dino_metrics.csv"
    if dpath.exists():
        with dpath.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                dino_rows[(r.get("src_image", ""), r.get("tgt_style", ""))] = r
    print(f"[v4e] metrics.csv rows={len(csv_rows)} dino_metrics.csv rows={len(dino_rows)}", flush=True)

    metric = Q.Metrics()
    protos = Q.build_prototypes(metric, Q.load_latents(), set())
    proto_mat = np.stack([protos[s] for s in STYLES])

    sel = json.loads((REPO_ROOT / "runs" / "p0" / "reference_selection.json").read_text())["selected"]
    ref_cls = {}
    ref_pat = {}
    for style in STYLES:
        refs = [Image.open(p).convert("RGB").resize((args.image_size, args.image_size), Image.BICUBIC)
                for p in sel[style]["r0"][:30]]
        c, p_ = encode_dino(metric.dino, refs, batch=8)
        ref_cls[style] = F.normalize(c.float(), dim=-1)
        ref_pat[style] = F.normalize(p_.float().reshape(-1, p_.shape[-1]), dim=-1)

    def dino_sim(cls: torch.Tensor, style: str) -> float:
        return float((F.normalize(cls, dim=-1) @ ref_cls[style].T).max())

    rows = []
    for sid, (sname, paths) in sorted(test_images.items()):
        for path in paths:
            src_pil = Image.open(path).convert("RGB").resize((args.image_size, args.image_size), Image.BICUBIC)
            z = metric.encode(_load_eval_image_tensor(path, args.image_size).unsqueeze(0).cuda())
            base_pil = metric.decode(z)[0]
            cs_b = csd_model.feats_pils(metric.csd, [base_pil], batch=1)[0]
            cls_b, _ = encode_dino(metric.dino, [base_pil], batch=1)
            for tid, tname in names.items():
                gp = img_dir / f"{sname}_{Path(path).stem}_to_{tname}.png"
                if not gp.exists():
                    continue
                g_pil = Image.open(gp).convert("RGB")
                cs_g = csd_model.feats_pils(metric.csd, [g_pil], batch=1)[0]
                cls_g, pat_g = encode_dino(metric.dino, [g_pil], batch=1)
                cross = tid != sid
                cr = csv_rows.get((Path(path).name, tname), {})
                dr = dino_rows.get((Path(path).name, tname), {})
                try:
                    swd = float(sliced_wasserstein_distance(
                        F.normalize(pat_g.float().reshape(-1, pat_g.shape[-1]), dim=-1),
                        ref_pat[tname], num_projections=64).item())
                except Exception as exc:  # pragma: no cover
                    swd = float("nan")
                    if len(rows) == 0:
                        print(f"[v4e] SWD failed: {exc}", flush=True)
                rows.append({
                    "src": Path(path).name, "src_style": sname, "tgt_style": tname, "cross": cross,
                    "dino_s": float(dr.get("dino_style_similarity", "nan")),
                    "dino_c": float(dr.get("dino_content_similarity", "nan")),
                    "clip_s": float(cr.get("clip_style", "nan")),
                    "lpips": float(cr.get("lpips", "nan")),
                    "csd_dT": float(cs_g @ proto_mat[tid]) - float(cs_b @ proto_mat[tid]),
                    "csd_dS": float(cs_g @ proto_mat[sid]) - float(cs_b @ proto_mat[sid]),
                    "dino_s_tgt": dino_sim(cls_g[0], tname),
                    "dino_s_src": dino_sim(cls_g[0], sname),
                    "dino_s_tgt_base": dino_sim(cls_b[0], tname),
                    "dino_s_src_base": dino_sim(cls_b[0], sname),
                    "swd": swd,
                })

    def agg(sel_rows, key):
        v = np.array([r[key] for r in sel_rows], dtype=float)
        v = v[~np.isnan(v)]
        return float(v.mean()) if len(v) else float("nan")

    groups = {"all750": rows, "cross600": [r for r in rows if r["cross"]],
              "same150": [r for r in rows if not r["cross"]]}
    out = {"cell": args.cell, "n_sources": sum(len(v[1]) for v in test_images.items())}
    for gname, g in groups.items():
        dT = np.array([r["csd_dT"] for r in g]); dS = np.array([r["csd_dS"] for r in g])
        dMt = np.array([r["dino_s_tgt"] - r["dino_s_tgt_base"] for r in g])
        dMs = np.array([r["dino_s_src"] - r["dino_s_src_base"] for r in g])
        out[gname] = {
            "n": len(g),
            "DINO_S": agg(g, "dino_s"), "DINO_C": agg(g, "dino_c"),
            "CLIP_S": agg(g, "clip_s"), "LPIPS": agg(g, "lpips"),
            "CSD_dT": float(dT.mean()), "CSD_dT_ci": _boot_ci(dT),
            "CSD_dS": float(dS.mean()), "CSD_dS_ci": _boot_ci(dS),
            "CSD_dM": float((dT - dS).mean()),
            "DINO_dM": float((dMt - dMs).mean()),
            "PatchSWD": agg(g, "swd"),
            "PatchSWD_sd": float(np.nanstd([r["swd"] for r in g])),
        }
    dest = Path(args.out) if args.out else out_dir / "v4_eval750.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "v4_eval750_rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
