"""P2: 600 cross-style main table.

Metrics (all on the SAME 600 cross-style response pairs, 150 sources x 4 non-self targets):
  * CSD  ΔMargin  (primary)   ΔM = ΔT - ΔS,  ΔT = S_t(G) - S_t(B),  ΔS = S_c(G) - S_c(B)
  * DINO ΔMargin  (control)   same definition, DINOv2-small CLS
  * Patch-SWD     (control)   EMD between DINOv2 patch-token distributions of G and the
                              frozen target-style train pool; reported as a *decrease*
                              relative to the VAE round-trip baseline (positive = closer)

Baseline B = cell A (VAE round-trip, reference independent); G = D_r0 (full WEAVE).
By definition every reported delta is exactly 0 on the baseline row.

S_t: CSD cosine to the style prototype (L2-normalised mean of the first 74 frozen
     train-pool descriptors); DINO: max cosine over the 30 test refs of that style
     (identical to the official dino_style_similarity).
S_c: same score against the *source* style (the "did it stay in its own style" term).

Stats: mean / median / fraction>0; 95% bootstrap CI resampling SOURCE IMAGES,
stratified by source style (30 sources per style, 2000 reps). No success threshold.

No training, no config edits, no reference re-selection against scores.
"""
from __future__ import annotations

import argparse
import csv
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


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render-root", default=str(paths.asset_root() / "eval_assets" / "p2_render"))
    ap.add_argument("--gen-cell", default="D_r0")
    ap.add_argument("--base-cell", default="A")
    ap.add_argument("--manifest", default=str(ROOT / "data" / "wikiart_new_packed" / "manifest.json"))
    ap.add_argument("--test-dir", default=str(ROOT / "data" / "test"))
    ap.add_argument("--cache-dir", default=str(paths.eval_cache_root()))
    ap.add_argument("--p0-root", default=str(ROOT / "runs" / "p0"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "p2" / "p2_cross_style.json"))
    ap.add_argument("--n", default="300,300,74,300,300")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k-proto", type=int, default=74)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--boot-seed", type=int, default=7)
    ap.add_argument("--n-proj", type=int, default=32)
    ap.add_argument("--swd-quantiles", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    return ap.parse_args()


# ---------------------------------------------------------------------------
def quantile_grid(sorted_vals: np.ndarray, k: int) -> np.ndarray:
    """(N, D) sorted along axis 0 -> (k, D) quantile grid at p=(i+0.5)/k."""
    n = sorted_vals.shape[0]
    p = (np.arange(k) + 0.5) / k
    pos = p * (n - 1)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    w = (pos - lo)[:, None]
    return sorted_vals[lo] * (1.0 - w) + sorted_vals[hi] * w


def stats(vals: np.ndarray) -> dict:
    v = np.asarray(vals, dtype=float)
    return {
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "frac_positive": float((v > 0).mean()),
        "n": int(v.size),
    }


def strat_bootstrap(by_src: dict, n_boot: int, seed: int) -> dict:
    """Resample SOURCE IMAGES with replacement, stratified by source style."""
    styles = sorted({k[0] for k in by_src})
    groups = {s: [k for k in by_src if k[0] == s] for s in styles}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        sel = []
        for s in styles:
            keys = groups[s]
            idx = rng.integers(0, len(keys), len(keys))
            sel.extend(by_src[keys[i]] for i in idx)
        means[b] = np.concatenate(sel).mean()
    return {"ci95_low": float(np.percentile(means, 2.5)),
            "ci95_high": float(np.percentile(means, 97.5)),
            "boot_se": float(means.std()), "n_boot": n_boot,
            "resample_unit": "source image", "stratified_by": "source style"}


def summarise(vals, by_src, a) -> dict:
    d = stats(vals)
    d.update(strat_bootstrap(by_src, a.n_boot, a.boot_seed))
    return d


def read_render_csv(cell_dir: Path) -> dict:
    """(src_style, src_image, tgt_style) -> dict of all metric columns."""
    out = {}
    for name in ("dino_metrics.csv", "metrics.csv"):
        p = cell_dir / name
        if not p.exists():
            raise SystemExit(f"[p2] missing {p}")
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                key = (r["src_style"], r["src_image"], r["tgt_style"])
                out.setdefault(key, {}).update(r)
    return out


# ---------------------------------------------------------------------------
def main() -> None:
    a = parse_args()
    t_all = time.perf_counter()
    render_root = Path(a.render_root)
    gen_cell, base_cell = a.gen_cell, a.base_cell

    # ---- protocol (frozen, identical to P0) ------------------------------
    test_images, requests, style_meta = ef.build_protocol(
        Path(a.test_dir), max_src_samples=30, ref_pool="first")
    sids = sorted(style_meta)
    names = [style_meta[s]["name"] for s in sids]
    cross_reqs = [r for r in requests if r["src_style_id"] != r["tgt_style_id"]]
    print(f"[p2] styles={names} cross_pairs={len(cross_reqs)}", flush=True)

    # ---- frozen prototype pool (same construction as P1c AUROC) ---------
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
    pool_rows, pool_artists = {}, {}
    for name, n in zip(names, ns_per):
        rows_all = [r for r in by_style[name] if Path(r["path"]).stem not in test_stems]
        if len(rows_all) > n:
            sel = np.sort(rng.choice(len(rows_all), size=n, replace=False))
            rows = [rows_all[int(i)] for i in sel]
        else:
            rows = rows_all
        pool_rows[name] = rows
        pool_artists[name] = [str(r.get("artist", "")) for r in rows]

    # ---- CSD -------------------------------------------------------------
    csd, ck, msg = csd_model.load_csd("cuda")
    print("[p2] CSD loaded:", msg, flush=True)
    probe = Image.open(sorted(Path(a.test_dir).rglob("*.jpg"))[0]).convert("RGB")
    x = torch.stack([csd_model.PREPROCESS(probe)]).cuda().half()
    with torch.inference_mode():
        for _ in range(5):
            csd(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            csd(x)
        torch.cuda.synchronize()
    csd_ms = round((time.perf_counter() - t0) / 20 * 1000, 2)
    print("[p2] CSD single-image ms:", csd_ms, flush=True)

    # ---- DINO engine (references only; no CLIP / LPIPS) ------------------
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

    # ---- projections for Patch-SWD ---------------------------------------
    R = np.random.default_rng(2024).normal(size=(a.n_proj, 384)).astype(np.float32)
    R /= np.linalg.norm(R, axis=1, keepdims=True)

    def project(sorted_tokens: np.ndarray) -> np.ndarray:
        """(T,384) normalised patch tokens -> (T, n_proj) sorted projections."""
        return np.sort(sorted_tokens @ R.T, axis=0)

    # ---- prototype + SWD reference grids ---------------------------------
    t_proto = time.perf_counter()
    proto_csd, swd_ref = {}, {}
    prot_report = {}
    for name in names:
        imgs = [image_root / r["path"] for r in pool_rows[name][: a.k_proto]]
        csd_feats, projs = [], []
        for s in range(0, len(imgs), a.batch):
            pils = [Image.open(p).convert("RGB") for p in imgs[s:s + a.batch]]
            csd_feats.append(csd_model.feats_pils(csd, pils, batch=a.batch))
            with torch.inference_mode():
                _, patches = engine._dino_forward(pils)
            patches = patches.float().cpu().numpy()
            projs.append(project(patches.reshape(-1, patches.shape[-1])))  # (B*256, n_proj), sorted
        F_all = np.concatenate(csd_feats, 0).astype(np.float32)
        proto = F_all.mean(0)
        proto_csd[name] = proto / np.linalg.norm(proto)
        proj_all = np.sort(np.concatenate(projs, 0), axis=0)   # (N_tokens, n_proj), sorted
        swd_ref[name] = quantile_grid(proj_all, a.swd_quantiles)  # (k, n_proj)
        arts = pool_artists[name][: a.k_proto]
        prot_report[name] = {
            "n_images": len(imgs), "n_unique_artists": len(set(arts)),
            "largest_artist_share": round(max(arts.count(x) for x in set(arts)) / len(arts), 4),
            "swd_ref_tokens": int(proj_all.shape[0]),
        }
        print(f"[p2] proto {name}: {len(imgs)} imgs, {proj_all.shape[0]} patch tokens", flush=True)
    print(f"[p2] prototypes + SWD refs in {time.perf_counter() - t_proto:.1f}s", flush=True)

    # ---- generated features ----------------------------------------------
    def cell_features(cell: str) -> dict:
        img_dir = render_root / cell / "images"
        if not img_dir.exists():
            raise SystemExit(f"[p2] missing render dir {img_dir} (run p2_render.py first)")
        out = {}
        t0 = time.perf_counter()
        for s in range(0, len(requests), a.batch):
            reqs = requests[s:s + a.batch]
            paths_ = [img_dir / r["gen_name"] for r in reqs]
            missing = [p for p in paths_ if not p.exists()]
            if missing:
                raise SystemExit(f"[p2] {len(missing)} images missing in {img_dir}: {missing[:3]}")
            pils = [Image.open(p).convert("RGB") for p in paths_]
            out_csd = csd_model.feats_pils(csd, pils, batch=a.batch)
            with torch.inference_mode():
                cls, patches = engine._dino_forward(pils)
            cls_np = F.normalize(cls.float(), dim=-1).cpu().numpy()
            pat_np = F.normalize(patches.float(), dim=-1).cpu().numpy()
            for i, r in enumerate(reqs):
                out[r["gen_name"]] = {
                    "csd": out_csd[i].astype(np.float32),
                    "dino": cls_np[i].astype(np.float32),
                    "swd": project(pat_np[i]).astype(np.float32),
                }
            if (s // a.batch) % 5 == 0:
                print(f"[p2] {cell} feats {min(s + a.batch, len(requests))}/{len(requests)} "
                      f"({time.perf_counter() - t0:.1f}s)", flush=True)
        print(f"[p2] {cell} feats done in {time.perf_counter() - t0:.1f}s", flush=True)
        return out

    feats = {gen_cell: cell_features(gen_cell), base_cell: cell_features(base_cell)}

    # ---- DINO reference matrices -----------------------------------------
    dino_ref = {style_meta[sid]["name"]: engine.ref_dino_cls[sid].float().numpy()
                for sid in sids}

    # ---- per-pair -------------------------------------------------------
    recs = []
    t_pairs = time.perf_counter()
    for r in cross_reqs:
        g = feats[gen_cell][r["gen_name"]]
        b = feats[base_cell][r["gen_name"]]
        st, sc = r["tgt_style"], r["src_style"]
        # CSD
        ctg, csg = float(g["csd"] @ proto_csd[st]), float(g["csd"] @ proto_csd[sc])
        ctb, csb = float(b["csd"] @ proto_csd[st]), float(b["csd"] @ proto_csd[sc])
        c_dt, c_ds = ctg - ctb, csg - csb
        # DINO (max cosine over the style's 30 test refs)
        d_tg = float((dino_ref[st] @ g["dino"]).max())
        d_sg = float((dino_ref[sc] @ g["dino"]).max())
        d_tb = float((dino_ref[st] @ b["dino"]).max())
        d_sb = float((dino_ref[sc] @ b["dino"]).max())
        d_dt, d_ds = d_tg - d_tb, d_sg - d_sb
        # Patch-SWD (lower = closer to target style); report improvement vs baseline
        s_g = float(np.abs(g["swd"] - swd_ref[st]).mean())
        s_b = float(np.abs(b["swd"] - swd_ref[st]).mean())
        recs.append({
            "src_style": sc, "src_image": r["src_name"], "tgt_style": st,
            "gen_image": r["gen_name"],
            "csd_T_gen": ctg, "csd_T_base": ctb, "csd_S_gen": csg, "csd_S_base": csb,
            "csd_dT": c_dt, "csd_dS": c_ds, "csd_dM": c_dt - c_ds,
            "dino_T_gen": d_tg, "dino_T_base": d_tb, "dino_S_gen": d_sg, "dino_S_base": d_sb,
            "dino_dT": d_dt, "dino_dS": d_ds, "dino_dM": d_dt - d_ds,
            "swd_gen": s_g, "swd_base": s_b, "swd_gain": s_b - s_g,
        })
    print(f"[p2] pair scoring {len(recs)} pairs in {time.perf_counter() - t_pairs:.1f}s", flush=True)

    # ---- aggregation ----------------------------------------------------
    def by_src(key):
        d = defaultdict(list)
        for x in recs:
            d[(x["src_style"], x["src_image"])].append(x[key])
        return {k: np.asarray(v, dtype=float) for k, v in d.items()}

    keys = {
        "csd_dT": "CSD ΔT", "csd_dS": "CSD ΔS", "csd_dM": "CSD ΔMargin",
        "dino_dT": "DINO ΔT", "dino_dS": "DINO ΔS", "dino_dM": "DINO ΔMargin",
        "swd_gain": "Patch-SWD gain (baseline - gen, positive = closer to target)",
    }
    agg = {k: summarise([x[k] for x in recs], by_src(k), a) for k in keys}

    quad = {}
    for tag, dt, ds in (("csd", "csd_dT", "csd_dS"), ("dino", "dino_dT", "dino_dS")):
        cnt = defaultdict(int)
        for x in recs:
            q = ("T+" if x[dt] > 0 else "T-") + "/" + ("S-" if x[ds] < 0 else "S+")
            cnt[q] += 1
        quad[tag] = {k: {"n": v, "pct": round(v / len(recs), 4)} for k, v in sorted(cnt.items())}
        # bootstrap CI for the two "ideal"/"worst" quadrant shares
        for q in ("T+/S-", "T-/S+"):
            bs = {k: np.asarray([1.0 if (("T+" if x[dt] > 0 else "T-") + "/" +
                                         ("S-" if x[ds] < 0 else "S+")) == q else 0.0
                                 for x in recs if (x["src_style"], x["src_image"]) == k])
                  for k in {(x["src_style"], x["src_image"]) for x in recs}}
            ci = strat_bootstrap(bs, a.n_boot, a.boot_seed)
            quad[tag][q]["ci95"] = [ci["ci95_low"], ci["ci95_high"]]

    # ---- 150 same-style table: style change + content preservation only --
    same_reqs = [r for r in requests if r["src_style_id"] == r["tgt_style_id"]]
    cov = {c: read_render_csv(render_root / c) for c in (gen_cell, base_cell)}
    same_recs = []
    for r in same_reqs:
        g = feats[gen_cell][r["gen_name"]]
        b = feats[base_cell][r["gen_name"]]
        nm = r["tgt_style"]
        k3 = (r["src_style"], r["src_name"], r["tgt_style"])
        sg_t = float(g["csd"] @ proto_csd[nm])
        sb_t = float(b["csd"] @ proto_csd[nm])
        dg_t = float((dino_ref[nm] @ g["dino"]).max())
        db_t = float((dino_ref[nm] @ b["dino"]).max())
        same_recs.append({
            "src_style": r["src_style"], "src_image": r["src_name"], "style": nm,
            "csd_style_gen": sg_t, "csd_style_base": sb_t, "csd_dStyle": sg_t - sb_t,
            "dino_style_gen": dg_t, "dino_style_base": db_t, "dino_dStyle": dg_t - db_t,
            "dino_content_gen": float(cov[gen_cell][k3]["dino_content_similarity"]),
            "dino_content_base": float(cov[base_cell][k3]["dino_content_similarity"]),
            "lpips_gen": float(cov[gen_cell][k3]["lpips"]),
            "lpips_base": float(cov[base_cell][k3]["lpips"]),
        })

    def by_src_same(key):
        d = defaultdict(list)
        for x in same_recs:
            d[(x["src_style"], x["src_image"])].append(x[key])
        return {k: np.asarray(v, float) for k, v in d.items()}

    same_keys = {
        "csd_dStyle": "CSD Δoriginal-style similarity",
        "dino_dStyle": "DINO Δoriginal-style similarity",
        "dino_content_gen": "DINO content similarity (gen vs source)",
        "dino_content_base": "DINO content similarity (VAE RT)",
        "lpips_gen": "LPIPS (gen vs source, lower = better)",
        "lpips_base": "LPIPS (VAE RT vs source)",
    }
    same_agg = {k: summarise([x[k] for x in same_recs], by_src_same(k), a) for k in same_keys}

    # ---- 750 (historical compatibility only) ----------------------------
    hist = {}
    for cell in (base_cell, gen_cell):
        cov_c = cov[cell]
        hist[cell] = {
            "n": len(cov_c),
            "dino_style_similarity_mean": float(np.mean([float(v["dino_style_similarity"]) for v in cov_c.values()])),
            "lpips_mean": float(np.mean([float(v["lpips"]) for v in cov_c.values()])),
        }
    p0 = {}
    p0_root = Path(a.p0_root) if a.p0_root else None
    if p0_root and p0_root.exists():
        for cell in (base_cell, gen_cell):
            cell_dir = p0_root / cell
            # 历史 P0 渲染只覆盖既有 cell；新 cell 没有 P0 基线时跳过，不伪造对照
            if not (cell_dir / "dino_metrics.csv").exists() and \
                    not (cell_dir / "metrics.csv").exists():
                print(f"[p2] p0 render missing for {cell}; skip historical P0 comparison")
                continue
            d = read_render_csv(cell_dir)
            p0[cell] = {
                "n": len(d),
                "dino_style_similarity_mean": float(np.mean([float(v["dino_style_similarity"]) for v in d.values()])),
                "lpips_mean": float(np.mean([float(v["lpips"]) for v in d.values()])),
            }

    per_tgt = {}
    for name in names:
        sub = [x for x in recs if x["tgt_style"] == name]
        per_tgt[name] = {k: float(np.mean([x[k] for x in sub])) for k in keys}
    per_src_style = {}
    for name in names:
        sub = [x for x in recs if x["src_style"] == name]
        per_src_style[name] = {k: float(np.mean([x[k] for x in sub])) for k in keys}

    res = {
        "generated_cell": gen_cell, "baseline_cell": base_cell,
        "n_cross_pairs": len(recs),
        "n_sources": len({(x["src_style"], x["src_image"]) for x in recs}),
        "identity_baseline": {
            "row": "VAE round-trip (cell A)",
            "values": {"CSD ΔMargin": 0.0, "DINO ΔMargin": 0.0, "Patch-SWD gain": 0.0},
            "why": "the baseline IS cell A, so every Δ is computed against itself",
        },
        "metrics": agg, "quadrants": quad,
        "same_style_150": {
            "metrics": same_agg,
            "note": "original-style similarity change + content preservation only; "
                    "no margin is reported or mixed into the 600 table",
            "per_pair": same_recs,
        },
        "historical_750": {
            "scope": "all 750 pairs, historical compatibility only (P0 threshold 0.50523 refers to this)",
            "new_render": hist, "p0_render": p0,
        },
        "per_target_style": per_tgt, "per_source_style": per_src_style,
        "definitions": {
            "S_t": "target-style similarity; S_c = same score vs the SOURCE style",
            "csd_score": "cosine(CSD 768-d style descriptor, L2-normalised prototype); "
                         f"prototype = mean of the first {a.k_proto} frozen train-pool descriptors",
            "dino_score": "max cosine over the style's 30 test references "
                          "(the official dino_style_similarity)",
            "dino_S_c_caveat": "with ref_pool='first' the 30 references of a style ARE the 30 test "
                               "sources of that style, so S_c(baseline) = 1.0 by self-match and any "
                               "transformation pushes DINO ΔS negative.  CSD (train-pool prototype, "
                               "no self-match) is therefore the primary line.",
            "patch_swd": f"sliced Wasserstein (L1, {a.n_proj} random directions, "
                          f"{a.swd_quantiles}-point quantile grid) between DINOv2 patch-token "
                          "distributions of the image and the frozen target-style train pool; "
                          "reported as swd_base - swd_gen",
            "quadrant": "sign of (ΔT, ΔS); T+/S- = toward target and away from source",
        },
        "prototype_pool": prot_report,
        "protocol": {
            "pairs": "600 cross-style response pairs (150 sources x 4 non-self targets)",
            "excluded": "150 same-style pairs are reported separately, never mixed in here",
            "tolerance_note": "no success threshold is applied; all numbers are descriptive",
        },
        "provenance_csd": csd_model.provenance(csd, ck, csd_ms),
        "timing_sec": time.perf_counter() - t_all,
        "per_pair": recs,
    }

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))

    # ---- console table ---------------------------------------------------
    print("\n===== P2 600 cross-style main table =====")
    print(f"{'row':42s} {'mean':>9s} {'median':>9s} {'frac>0':>7s} {'95% CI':>22s}")
    print(f"{'VAE round-trip (A, identity)':42s} {0.0:9.4f} {0.0:9.4f} {'def':>7s} "
          f"{'[0, 0]':>22s}")
    for label, k in ((f"{gen_cell} CSD-ΔMargin", "csd_dM"),
                     (f"{gen_cell} DINO-ΔMargin", "dino_dM"),
                     (f"{gen_cell} Patch-SWD gain", "swd_gain")):
        s = agg[k]
        print(f"{label:42s} {s['mean']:9.4f} {s['median']:9.4f} {s['frac_positive']:7.3f} "
              f"[{s['ci95_low']:+.4f}, {s['ci95_high']:+.4f}]")
    print("\n-- decomposition --")
    for k in ("csd_dT", "csd_dS", "csd_dM", "dino_dT", "dino_dS", "dino_dM", "swd_gain"):
        s = agg[k]
        print(f"{keys[k]:58s} {s['mean']:+.4f}  CI [{s['ci95_low']:+.4f}, {s['ci95_high']:+.4f}]  "
              f"med {s['median']:+.4f}  frac>0 {s['frac_positive']:.3f}")
    print("\n-- quadrants --")
    for tag in ("csd", "dino"):
        print(tag, {q: (v["pct"], v["n"]) for q, v in quad[tag].items()})

    print("\n===== 150 same-style table (NO margin) =====")
    print(f"{'metric':46s} {'mean':>9s} {'median':>9s} {'95% CI':>20s}")
    for k, label in same_keys.items():
        s = same_agg[k]
        print(f"{label:46s} {s['mean']:9.4f} {s['median']:9.4f} "
              f"[{s['ci95_low']:+.4f}, {s['ci95_high']:+.4f}]")

    print("\n===== 750 historical compatibility =====")
    for cell in (base_cell, gen_cell):
        print(f"{cell:8s} new dino_s={hist[cell]['dino_style_similarity_mean']:.5f} "
              f"lpips={hist[cell]['lpips_mean']:.5f}  | P0 dino_s="
              f"{p0.get(cell, {}).get('dino_style_similarity_mean', float('nan')):.5f} "
              f"lpips={p0.get(cell, {}).get('lpips_mean', float('nan')):.5f}")
    print(f"\n[p2] wrote {out}  wall={time.perf_counter() - t_all:.1f}s")


if __name__ == "__main__":
    main()

