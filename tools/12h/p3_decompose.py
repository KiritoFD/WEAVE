"""P3 ablation decomposition + P4 CSD prototype sampling stability.

P3
  For every P0 cell (10 cells: A, C, B_r0..r3, D_r0..r3) recompute the P2 metrics
  against the VAE round-trip baseline (cell A) on the 600 cross-style pairs, and
  report the four-way decomposition
        VAE round-trip / AdaIN-only (B_r0) / ODE-only (C) / full WEAVE (D_r0)
  plus the six reference-scheme variants for reference.

P4
  Rebuild the CSD style prototypes from K in {4,8,16,32,74} randomly drawn
  train-pool images (5 independent draws each), recompute the whole 600-pair
  ΔMargin for C / B_r0 / D_r0 and check whether the *ranking of methods* is
  stable.  Acceptance is ranking stability, not a numeric threshold.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "trt"):
    sys.path.insert(0, str(_p))

import csd_model  # noqa: E402
import eval_fast as ef  # noqa: E402
import paths  # noqa: E402

ALL_CELLS = ["A", "C", "B_r0", "B_r1", "B_r2", "B_r3", "D_r0", "D_r1", "D_r2", "D_r3"]
KEY = {"VAE round-trip": "A", "AdaIN-only (B_r0)": "B_r0", "ODE-only (C)": "C", "full WEAVE (D_r0)": "D_r0"}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render-root", default=str(paths.asset_root() / "eval_assets" / "p2_render"))
    ap.add_argument("--cells", default=",".join(ALL_CELLS))
    ap.add_argument("--base-cell", default="A")
    ap.add_argument("--manifest", default=str(ROOT / "data" / "wikiart_new_packed" / "manifest.json"))
    ap.add_argument("--test-dir", default=str(ROOT / "data" / "test"))
    ap.add_argument("--cache-dir", default=str(paths.eval_cache_root()))
    ap.add_argument("--out", default=str(ROOT / "runs" / "p3" / "p3_p4.json"))
    ap.add_argument("--n", default="300,300,74,300,300")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k-proto", type=int, default=74)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--boot-seed", type=int, default=7)
    ap.add_argument("--n-proj", type=int, default=32)
    ap.add_argument("--swd-quantiles", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--k-grid", default="4,8,16,32,74")
    ap.add_argument("--k-draws", type=int, default=5)
    return ap.parse_args()


def quantile_grid(sorted_vals: np.ndarray, k: int) -> np.ndarray:
    n = sorted_vals.shape[0]
    p = (np.arange(k) + 0.5) / k
    pos = p * (n - 1)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    w = (pos - lo)[:, None]
    return sorted_vals[lo] * (1.0 - w) + sorted_vals[hi] * w


def strat_bootstrap(by_src: dict, n_boot: int, seed: int) -> dict:
    styles = sorted({k[0] for k in by_src})
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        sel = []
        for s in styles:
            keys = [k for k in by_src if k[0] == s]
            idx = rng.integers(0, len(keys), len(keys))
            sel.extend(by_src[keys[i]] for i in idx)
        means[b] = np.concatenate(sel).mean()
    return {"ci95_low": float(np.percentile(means, 2.5)),
            "ci95_high": float(np.percentile(means, 97.5)), "n_boot": n_boot}


def summarise(vals, keys_src, a) -> dict:
    v = np.asarray(vals, float)
    d = {"mean": float(v.mean()), "median": float(np.median(v)),
         "frac_positive": float((v > 0).mean()), "n": int(v.size)}
    d.update(strat_bootstrap(keys_src, a.n_boot, a.boot_seed))
    return d


def main() -> None:
    a = parse_args()
    t_all = time.perf_counter()
    render_root = Path(a.render_root)
    cells = [c.strip() for c in a.cells.split(",") if c.strip()]
    missing = [c for c in cells if not (render_root / c / "images").exists()]
    if missing:
        raise SystemExit(f"[p3] missing renders for {missing}")

    test_images, requests, style_meta = ef.build_protocol(
        Path(a.test_dir), max_src_samples=30, ref_pool="first")
    sids = sorted(style_meta)
    names = [style_meta[s]["name"] for s in sids]
    id2name = {s: style_meta[s]["name"] for s in sids}
    cross = [r for r in requests if r["src_style_id"] != r["tgt_style_id"]]
    print(f"[p3] cells={cells} cross={len(cross)}", flush=True)

    # ---- pool + prototypes ------------------------------------------------
    manifest = json.loads(Path(a.manifest).read_text())
    image_root = Path(manifest["image_root"])
    by_style = defaultdict(list)
    for x in manifest["samples"]:
        if x.get("split") == "train":
            by_style[x["style"]].append(x)
    for k in by_style:
        by_style[k].sort(key=lambda r: int(r["index"]))
    test_stems = {p.stem for sid in style_meta for p in style_meta[sid]["all"]}
    rng = np.random.default_rng(a.seed)
    ns_per = [int(v) for v in a.n.split(",")]
    pool_paths = {}
    for name, n in zip(names, ns_per):
        rows_all = [r for r in by_style[name] if Path(r["path"]).stem not in test_stems]
        if len(rows_all) > n:
            sel = np.sort(rng.choice(len(rows_all), size=n, replace=False))
            rows = [rows_all[int(i)] for i in sel]
        else:
            rows = rows_all
        pool_paths[name] = [image_root / r["path"] for r in rows]

    csd, ck, msg = csd_model.load_csd("cuda")
    print("[p3] CSD:", msg, flush=True)
    pool_feats = {}
    for name in names:
        pool_feats[name] = csd_model.feats_paths(csd, pool_paths[name][: a.k_proto], batch=a.batch)
    proto = {n: (f.mean(0) / np.linalg.norm(f.mean(0))) for n, f in pool_feats.items()}

    R = np.random.default_rng(2024).normal(size=(a.n_proj, 384)).astype(np.float32)
    R /= np.linalg.norm(R, axis=1, keepdims=True)

    def project(tok):
        return np.sort(tok @ R.T, axis=0)

    # ---- DINO engine (references) ----------------------------------------
    ns = argparse.Namespace(
        disable_clip=True, disable_lpips=True, disable_dino=False, test_dir=str(a.test_dir),
        clip_model="openai/clip-vit-base-patch32", clip_local_only=True,
        clip_cache_dir=str(Path(a.cache_dir) / "hf" / "hub"), clip_skip_processor=False, amp=False,
        lpips_net="alex", lpips_chunk=32, dino_model="facebook/dinov2-small",
        dino_cache_dir=str(Path(a.cache_dir) / "hf"), allow_network=False,
        ref_cache="", max_ref_cache=16, dino_max_refs=30, image_size=512,
        metric_batch=a.batch, ref_batch=32, only_clip_style=True,
        clip_style_idt_baseline=0.0, clip_vectors=False,
    )
    timer = ef.Timer(sync_cuda=False)
    engine = ef.MetricEngine(ns, test_images, style_meta, "cuda")
    engine.load(timer)
    engine.prepare(timer, [])
    dino_ref = {id2name[s]: engine.ref_dino_cls[s].float().numpy() for s in sids}

    swd_ref = {}
    for name in names:
        imgs = pool_paths[name][: a.k_proto]
        projs = []
        for s in range(0, len(imgs), a.batch):
            pils = [Image.open(p).convert("RGB") for p in imgs[s:s + a.batch]]
            with torch.inference_mode():
                _, patches = engine._dino_forward(pils)
            pat = F.normalize(patches.float(), dim=-1).cpu().numpy().reshape(-1, patches.shape[-1])
            projs.append(project(pat))
        swd_ref[name] = quantile_grid(np.sort(np.concatenate(projs, 0), axis=0), a.swd_quantiles)
    print("[p3] SWD refs ready", flush=True)

    # ---- features per cell ------------------------------------------------
    def cell_features(cell, keep_swd=True):
        img_dir = render_root / cell / "images"
        csd_f = np.empty((len(requests), 768), np.float32)
        dino_f = np.empty((len(requests), 384), np.float32)
        swd_f = np.empty((len(requests), a.swd_quantiles, a.n_proj), np.float32) if keep_swd else None
        t0 = time.perf_counter()
        for s in range(0, len(requests), a.batch):
            reqs = requests[s:s + a.batch]
            pils = [Image.open(img_dir / r["gen_name"]).convert("RGB") for r in reqs]
            csd_f[s:s + len(reqs)] = csd_model.feats_pils(csd, pils, batch=a.batch)
            with torch.inference_mode():
                cls, patches = engine._dino_forward(pils)
            dino_f[s:s + len(reqs)] = F.normalize(cls.float(), dim=-1).cpu().numpy()
            if keep_swd:
                pat = F.normalize(patches.float(), dim=-1).cpu().numpy()
                for i in range(len(reqs)):
                    swd_f[s + i] = project(pat[i])
        print(f"[p3] {cell} feats {time.perf_counter() - t0:.1f}s", flush=True)
        return {"csd": csd_f, "dino": dino_f, "swd": swd_f}

    feats = {}
    for c in cells:
        keep = c in ("A", "C", "B_r0", "D_r0")
        feats[c] = cell_features(c, keep_swd=keep)

    # ---- index helpers ----------------------------------------------------
    idx_cross = [i for i, r in enumerate(requests) if r["src_style_id"] != r["tgt_style_id"]]
    cross_src = [(requests[i]["src_style"], requests[i]["src_name"]) for i in idx_cross]

    def deltas(cell, protos=None, need_swd=False, need_dino=True):
        pf = protos or proto
        g = feats[cell]
        b = feats[a.base_cell]
        P = np.stack([pf[n] for n in names], 1)          # (768,5)
        n2i = {n: j for j, n in enumerate(names)}
        ti = np.array([n2i[r["tgt_style"]] for r in requests])
        si = np.array([n2i[r["src_style"]] for r in requests])
        cg = g["csd"] @ P
        cb = b["csd"] @ P
        csd_dT = cg[np.arange(len(requests)), ti] - cb[np.arange(len(requests)), ti]
        csd_dS = cg[np.arange(len(requests)), si] - cb[np.arange(len(requests)), si]
        out = {
            "csd_dT": csd_dT[idx_cross], "csd_dS": csd_dS[idx_cross],
            "csd_dM": (csd_dT - csd_dS)[idx_cross],
        }
        if not need_dino:
            return out
        dg = np.array([(dino_ref[r["tgt_style"]] @ g["dino"][i]).max() for i, r in enumerate(requests)])
        db = np.array([(dino_ref[r["tgt_style"]] @ b["dino"][i]).max() for i, r in enumerate(requests)])
        sg = np.array([(dino_ref[r["src_style"]] @ g["dino"][i]).max() for i, r in enumerate(requests)])
        sb = np.array([(dino_ref[r["src_style"]] @ b["dino"][i]).max() for i, r in enumerate(requests)])
        out = {
            "csd_dT": csd_dT[idx_cross], "csd_dS": csd_dS[idx_cross],
            "csd_dM": (csd_dT - csd_dS)[idx_cross],
            "dino_dT": (dg - db)[idx_cross], "dino_dS": (sg - sb)[idx_cross],
            "dino_dM": ((dg - db) - (sg - sb))[idx_cross],
        }
        if need_swd and g["swd"] is not None and b["swd"] is not None:
            sgen = np.array([np.abs(g["swd"][i] - swd_ref[r["tgt_style"]]).mean() for i, r in enumerate(requests)])
            sbase = np.array([np.abs(b["swd"][i] - swd_ref[r["tgt_style"]]).mean() for i, r in enumerate(requests)])
            out["swd_gain"] = (sbase - sgen)[idx_cross]
        return out

    def group(vals):
        d = defaultdict(list)
        for k, v in zip(cross_src, vals):
            d[k].append(v)
        return {k: np.asarray(v, float) for k, v in d.items()}

    # ---- P3 aggregates ----------------------------------------------------
    cell_res, dm_arrays = {}, {}
    for c in cells:
        d = deltas(c, need_swd=(c in ("A", "C", "B_r0", "D_r0")))
        if c in ("A", "C", "B_r0", "D_r0"):
            dm_arrays[c] = d["csd_dM"]
        entry = {k: summarise(v, group(v), a) for k, v in d.items()}
        entry["n"] = len(d["csd_dM"])
        cell_res[c] = entry
        print(f"[p3] {c:6s} CSD dM {entry['csd_dM']['mean']:+.4f} "
              f"[{entry['csd_dM']['ci95_low']:+.4f},{entry['csd_dM']['ci95_high']:+.4f}]  "
              f"DINO dM {entry['dino_dM']['mean']:+.4f}  "
              f"SWD {entry.get('swd_gain', {}).get('mean', float('nan')):+.4f}", flush=True)

    # ---- paired differences (source-image bootstrap) ----------------------
    paired = {}
    for tag, x, y in (("D_r0 - C", "D_r0", "C"), ("D_r0 - B_r0", "D_r0", "B_r0"),
                      ("D_r0 - A", "D_r0", "A"), ("C - A", "C", "A"),
                      ("B_r0 - A", "B_r0", "A"), ("C - B_r0", "C", "B_r0")):
        diff = dm_arrays[x] - dm_arrays[y]
        paired[tag] = summarise(diff, group(diff), a)

    # ---- P4 prototype stability ------------------------------------------
    k_grid = [int(v) for v in a.k_grid.split(",")]
    methods = ["C", "B_r0", "D_r0"]
    ref_means = {m: cell_res[m]["csd_dM"]["mean"] for m in methods}
    ref_rank = sorted(methods, key=lambda m: -ref_means[m])
    p4 = {"reference": {"k": a.k_proto, "means": ref_means, "ranking": ref_rank},
          "grid": {}, "draws": a.k_draws, "metric": "mean CSD ΔMargin on the 600 cross pairs"}
    for K in k_grid:
        draws, ranks = [], []
        for s in range(a.k_draws):
            r2 = np.random.default_rng(1000 + s)
            pr = {}
            for name in names:
                f = pool_feats[name]
                sel = r2.choice(f.shape[0], size=min(K, f.shape[0]), replace=False)
                mu = f[sel].mean(0)
                pr[name] = mu / np.linalg.norm(mu)
            means = {}
            for m in methods:
                dm = deltas(m, protos=pr, need_dino=False)["csd_dM"]
                means[m] = float(np.mean(dm))
            draws.append(means)
            ranks.append(sorted(methods, key=lambda m: -means[m]))
        stable = sum(1 for r in ranks if r == ref_rank) / len(ranks)
        p4["grid"][str(K)] = {
            "mean_over_draws": {m: float(np.mean([d[m] for d in draws])) for m in methods},
            "std_over_draws": {m: float(np.std([d[m] for d in draws])) for m in methods},
            "rankings": ranks, "frac_rank_identical": stable,
        }
        print(f"[p4] K={K:3d} means={ {m: round(v,4) for m,v in p4['grid'][str(K)]['mean_over_draws'].items()} } "
              f"rank stable {stable:.2f}", flush=True)

    # ---- output -----------------------------------------------------------
    out = {
        "cells": cell_res,
        "decomposition": {label: cell_res[c] for label, c in KEY.items()},
        "all_cell_order": ["A", "C", "B_r0", "D_r0", "B_r1", "D_r1", "B_r2", "D_r2", "B_r3", "D_r3"],
        "p4_prototype_stability": p4,
        "paired_csd_dM_differences": paired,
        "definitions": {
            "baseline": "cell A (VAE round-trip); every Δ is computed against it",
            "swd_gain": "swd_base - swd_gen (positive = closer to the target-style token distribution)",
            "p4_acceptance": "ranking of C / B_r0 / D_r0 by mean CSD ΔMargin staying identical "
                             "across prototype draws",
        },
        "timing_sec": time.perf_counter() - t_all,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))

    print("\n===== P3 decomposition (600 cross pairs) =====")
    print(f"{'condition':22s} {'CSD dT':>9s} {'CSD dS':>9s} {'CSD dM':>9s} {'DINO dM':>9s} {'SWD gain':>9s}")
    for label, c in KEY.items():
        e = cell_res[c]
        print(f"{label:22s} {e['csd_dT']['mean']:+9.4f} {e['csd_dS']['mean']:+9.4f} "
              f"{e['csd_dM']['mean']:+9.4f} {e['dino_dM']['mean']:+9.4f} "
              f"{e.get('swd_gain', {}).get('mean', float('nan')):+9.4f}")
    print("\n===== all ten cells (CSD ΔMargin, 95% CI) =====")
    for c in ["A", "C", "B_r0", "D_r0", "B_r1", "D_r1", "B_r2", "D_r2", "B_r3", "D_r3"]:
        e = cell_res[c]["csd_dM"]
        print(f"{c:6s} {e['mean']:+.4f} [{e['ci95_low']:+.4f},{e['ci95_high']:+.4f}] "
              f"frac>0 {e['frac_positive']:.3f}")
    print("\n===== paired CSD ΔMargin differences (source-image bootstrap) =====")
    for tag, e in paired.items():
        print(f"{tag:14s} {e['mean']:+.4f} [{e['ci95_low']:+.4f},{e['ci95_high']:+.4f}]  "
              f"median {e['median']:+.4f}  frac>0 {e['frac_positive']:.3f}")
    print(f"\n[p3] wrote {a.out} wall={time.perf_counter() - t_all:.1f}s")


if __name__ == "__main__":
    main()

