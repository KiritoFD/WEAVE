"""Stiefel NS10 vs Exact FP32 SVD Orthogonality & Low-Rank Outlier Audit on 2,150 WikiArt + Test Sources."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from exp.v21.stiefel_ref_cache import stiefel_polar_quotient
from exp.v32_nodino.cnx_layout import ConvNeXtTinyStage3Extractor
from exp.v36_art_foundation import CACHE_DIR as V36_CACHE_DIR
from src.utils.runtime import REPO, read_json, sanitize_cuda_env, save_json, utc


OUT_JSON = REPO / "runs/paper_audit/stiefel_wikiart_2150.json"


@torch.no_grad()
def _stiefel_ns10(g16: torch.Tensor, w32_t: torch.Tensor, eye3: torch.Tensor, ns_steps: int = 10) -> torch.Tensor:
    B = g16.shape[0]
    fl = torch.bmm(w32_t.unsqueeze(0).expand(B, -1, -1), g16.flatten(2).float())
    fl = fl - fl.mean(dim=-1, keepdim=True)
    x_ns = fl / fl.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    for _ in range(ns_steps):
        x_ns = 0.5 * torch.bmm(eye3 - torch.bmm(x_ns, x_ns.transpose(1, 2)), x_ns)
    return (x_ns * 16.0).reshape(B, -1, 16, 16)


@torch.no_grad()
def run_stiefel_audit(device: str = "cuda") -> dict[str, Any]:
    if OUT_JSON.exists():
        return read_json(OUT_JSON)

    st_ck = torch.load(REPO / "runs/v32_nodino/cache/cnx_stiefel_cache.pt", map_location="cpu", weights_only=False)
    w32 = st_ck["W_stiefel_cnx"].to(device).float()
    w32_t = w32.t().contiguous()
    eye3 = torch.eye(w32.shape[1], device=device).unsqueeze(0) * 3.0
    cnx = ConvNeXtTinyStage3Extractor(device=device).eval()

    tj = read_json(REPO / "eval_protocol/tasks_750.json")
    entries: list[tuple[str, Path]] = [
        (f"test::{s['source_stem']}", REPO / s["source_rel_path"]) for s in tj["sources"]
    ]
    cm_p = V36_CACHE_DIR / "curation_manifest.json"
    if cm_p.exists():
        cm = read_json(cm_p)
        for sp in ("seen_dev", "artist_ood"):
            for r in cm.get(sp, []):
                p = Path(r["path"])
                if p.exists():
                    entries.append((f"{sp}::{r.get('style', '')}/{p.stem}", p))

    eye32 = torch.eye(32, device=device, dtype=torch.float32)
    gram_frob, gram_spec, gram_svd = [], [], []
    cos_q, rmse_q, cond_ratio = [], [], []
    records: list[tuple[str, float, float, float, float]] = []

    for st in range(0, len(entries), 16):
        sl = entries[st:st + 16]
        imgs = []
        for _, path in sl:
            with Image.open(path) as im:
                arr = np.asarray(im.convert("RGB").resize((512, 512), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
            imgs.append(torch.from_numpy(arr).permute(2, 0, 1))
        rgb = torch.stack(imgs, 0).to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            g16_fp16 = cnx.extract_gray_layout(rgb).float()
        q_ns = _stiefel_ns10(g16_fp16, w32_t, eye3, ns_steps=10)
        q_svd = stiefel_polar_quotient(g16_fp16, w32)

        r_ns = q_ns.flatten(2).float() / 16.0
        r_svd = q_svd.flatten(2).float() / 16.0
        fl = torch.bmm(w32_t.unsqueeze(0).expand(len(sl), -1, -1), g16_fp16.flatten(2).float())
        fl = fl - fl.mean(dim=-1, keepdim=True)
        svs = torch.linalg.svdvals(fl)
        cr = (svs[:, 0] / svs[:, -1].clamp_min(1e-8)).cpu().tolist()

        for i in range(len(sl)):
            gn = r_ns[i] @ r_ns[i].t() - eye32
            gs = r_svd[i] @ r_svd[i].t() - eye32
            fn = float(torch.linalg.matrix_norm(gn, ord="fro").item())
            sn = float(torch.linalg.matrix_norm(gn, ord=2).item())
            fs = float(torch.linalg.matrix_norm(gs, ord="fro").item())
            cs = float(F.cosine_similarity(q_ns[i].flatten(), q_svd[i].flatten(), dim=0).item())
            rm = float((q_ns[i] - q_svd[i]).pow(2).mean().sqrt().item())
            gram_frob.append(fn)
            gram_spec.append(sn)
            gram_svd.append(fs)
            cos_q.append(cs)
            rmse_q.append(rm)
            cond_ratio.append(float(cr[i]))
            records.append((sl[i][0], fn, sn, cs, float(cr[i])))

    records.sort(key=lambda x: -x[1])
    arr_fn, arr_sn, arr_cs, arr_rm, arr_cr = map(np.asarray, (gram_frob, gram_spec, cos_q, rmse_q, cond_ratio))
    rep = {
        "utc": utc(),
        "n_total": len(entries),
        "ns10_gram_frob_err": {
            "mean": float(arr_fn.mean()), "p50": float(np.median(arr_fn)),
            "p95": float(np.percentile(arr_fn, 95)), "p99": float(np.percentile(arr_fn, 99)), "max": float(arr_fn.max()),
        },
        "ns10_gram_spectral_err": {
            "mean": float(arr_sn.mean()), "p95": float(np.percentile(arr_sn, 95)), "max": float(arr_sn.max()),
        },
        "svd_gram_frob_err_mean": float(np.mean(gram_svd)),
        "q16_cosine_vs_svd": {"mean": float(arr_cs.mean()), "p05": float(np.percentile(arr_cs, 5)), "min": float(arr_cs.min())},
        "q16_rmse_vs_svd": {"mean": float(arr_rm.mean()), "p95": float(np.percentile(arr_rm, 95)), "max": float(arr_rm.max())},
        "sigma_ratio_s1_over_s32": {"mean": float(arr_cr.mean()), "p95": float(np.percentile(arr_cr, 95)), "max": float(arr_cr.max())},
        "worst_5_outliers": [
            {"stem": s, "gram_frob": round(fn, 6), "gram_spec": round(sn, 6), "cos_vs_svd": round(cs, 6), "cond_ratio": round(cr, 2)}
            for s, fn, sn, cs, cr in records[:5]
        ],
    }
    save_json(OUT_JSON, rep)
    print(f"[Stiefel 2150 DONE] mean_frob={rep['ns10_gram_frob_err']['mean']:.6f} max={rep['ns10_gram_frob_err']['max']:.6f}", flush=True)
    return rep


def main() -> None:
    sanitize_cuda_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    run_stiefel_audit(a.device)


if __name__ == "__main__":
    main()
