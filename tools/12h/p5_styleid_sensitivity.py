"""P5 (fixed source vs the five Style-IDs) + CSD low-level perturbation control.

Both parts are CSD-only and cheap (~1 min on 600x images market).

P5
  For every test source image (150, 30 per style) compute the CSD style score
  against all five style prototypes; report the 5x5 mean score matrix, per-style
  top-1 rate and the top1-top2 margin.  No generation involved.

Sensitivity control (replaces the SRQE-based one that was dropped)
  Take each generated cross-style response, apply mild low-level corruptions
  (colour / blur / contrast), recompute the CSD target score, and ask whether
  ΔT gets inflated.  Reported three ways:
      G only corrupted          -> target score rises?
      G and baseline corrupted  -> is the *difference* still stable?
      ΔMargin under corruption  -> does the margin survive?
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
from PIL import Image, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools"):
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
    ap.add_argument("--out", default=str(ROOT / "runs" / "p2" / "p5_styleid_sensitivity.json"))
    ap.add_argument("--n", default="300,300,74,300,300")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k-proto", type=int, default=74)
    ap.add_argument("--batch", type=int, default=32)
    return ap.parse_args()


PERTURB = {
    "colour(+5%bright,+10%sat)": lambda im: ImageEnhance.Color(
        ImageEnhance.Brightness(im).enhance(1.05)).enhance(1.10),
    "blur(sigma=0.5)": lambda im: im.filter(ImageFilter.GaussianBlur(0.5)),
    "contrast(+10%)": lambda im: ImageEnhance.Contrast(im).enhance(1.10),
}


def main() -> None:
    a = parse_args()
    t_all = time.perf_counter()
    render_root = Path(a.render_root)

    test_images, requests, style_meta = ef.build_protocol(
        Path(a.test_dir), max_src_samples=30, ref_pool="first")
    sids = sorted(style_meta)
    names = [style_meta[s]["name"] for s in sids]
    cross = [r for r in requests if r["src_style_id"] != r["tgt_style_id"]]
    print(f"[p5] styles={names} cross={len(cross)}", flush=True)

    # ---- prototype pool (same frozen construction as P1c / P2) -----------
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
    pool = {}
    for name, n in zip(names, ns_per):
        rows_all = [r for r in by_style[name] if Path(r["path"]).stem not in test_stems]
        if len(rows_all) > n:
            sel = np.sort(rng.choice(len(rows_all), size=n, replace=False))
            rows = [rows_all[int(i)] for i in sel]
        else:
            rows = rows_all
        pool[name] = [image_root / r["path"] for r in rows[: a.k_proto]]

    csd, ck, msg = csd_model.load_csd("cuda")
    print("[p5] CSD:", msg, flush=True)
    proto = {}
    for name in names:
        f = csd_model.feats_paths(csd, pool[name], batch=a.batch)
        m = f.mean(0)
        proto[name] = m / np.linalg.norm(m)

    # ---- P5: source vs five style prototypes -----------------------------
    src_paths = sorted({r["src_path"] for r in requests})
    src_feat = csd_model.feats_paths(csd, src_paths, batch=a.batch)
    src_style = {r["src_path"]: r["src_style"] for r in requests}
    S = np.stack([src_feat @ proto[n] for n in names], 1)          # (150, 5)
    rows_p5 = []
    for i, p in enumerate(src_paths):
        st = src_style[p]
        order = np.argsort(-S[i])
        rows_p5.append({
            "src_image": Path(p).name, "src_style": st,
            "scores": {n: float(S[i, j]) for j, n in enumerate(names)},
            "top1": names[order[0]], "top1_correct": bool(names[order[0]] == st),
            "top1_minus_top2": float(S[i, order[0]] - S[i, order[1]]),
        })
    # 5x5: rows = source style, cols = prototype
    mat = {}
    for si, sname in enumerate(names):
        idx = [i for i, p in enumerate(src_paths) if src_style[p] == sname]
        mat[sname] = {n: float(S[idx, j].mean()) for j, n in enumerate(names)}
    diag = {sname: mat[sname][sname] for sname in names}
    offdiag = {sname: float(np.mean([mat[sname][n] for n in names if n != sname])) for sname in names}
    top1 = {sname: float(np.mean([r["top1_correct"] for r in rows_p5 if r["src_style"] == sname]))
            for sname in names}

    # ---- sensitivity control ---------------------------------------------
    base_dir = render_root / a.base_cell / "images"
    gen_dir = render_root / a.gen_cell / "images"
    # baseline image is reference independent: one per source
    uniq_src = {}
    for r in cross:
        uniq_src.setdefault((r["src_style"], r["src_name"]), r)
    base_paths = [base_dir / r["gen_name"] for r in uniq_src.values()]
    base_keys = list(uniq_src.keys())

    out = {"p5": {"score_matrix": mat, "diagonal": diag, "offdiagonal": offdiag,
                  "top1_rate": top1,
                  "overall_top1": float(np.mean([r["top1_correct"] for r in rows_p5])),
                  "per_image": rows_p5},
           "sensitivity": {}}

    base_feat = {k: f for k, f in zip(base_keys, csd_model.feats_paths(csd, base_paths, batch=a.batch))}
    base_pert = {}
    for pname, fn in PERTURB.items():
        ims = [fn(Image.open(p).convert("RGB")) for p in base_paths]
        base_pert[pname] = {k: f for k, f in zip(base_keys, csd_model.feats_pils(csd, ims, batch=a.batch))}

    gen_paths = [gen_dir / r["gen_name"] for r in cross]
    gen_feat = csd_model.feats_paths(csd, gen_paths, batch=a.batch)
    gen_pert = {}
    for pname, fn in PERTURB.items():
        ims = [fn(Image.open(p).convert("RGB")) for p in gen_paths]
        gen_pert[pname] = csd_model.feats_pils(csd, ims, batch=a.batch)

    def by_src(vals):
        d = defaultdict(list)
        for r, v in zip(cross, vals):
            d[(r["src_style"], r["src_name"])].append(v)
        return {k: np.asarray(v, float) for k, v in d.items()}

    def boot(bs_vals, n_boot=2000, seed=7):
        styles = sorted({k[0] for k in bs_vals})
        rng2 = np.random.default_rng(seed)
        reps = []
        for _ in range(n_boot):
            sel = []
            for s in styles:
                keys = [k for k in bs_vals if k[0] == s]
                idx = rng2.integers(0, len(keys), len(keys))
                sel.extend(bs_vals[keys[i]] for i in idx)
            reps.append(np.concatenate(sel).mean())
        return [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]

    def pack(vals, label):
        bs = by_src(vals)
        return {"mean": float(np.mean(vals)), "median": float(np.median(vals)),
                "frac_positive": float((np.asarray(vals) > 0).mean()),
                "ci95": boot(bs), "label": label}

    def s_t(feat, r):
        return float(feat @ proto[r["tgt_style"]])

    def s_c(feat, r):
        return float(feat @ proto[r["src_style"]])

    sens = {}
    dT0 = [s_t(gen_feat[i], r) - s_t(base_feat[(r["src_style"], r["src_name"])], r)
           for i, r in enumerate(cross)]
    dM0 = [dT0[i] - (s_c(gen_feat[i], r) - s_c(base_feat[(r["src_style"], r["src_name"])], r))
           for i, r in enumerate(cross)]
    sens["original"] = {"dT": pack(dT0, "ΔT"), "dM": pack(dM0, "ΔM")}
    for pname in PERTURB:
        dT_g = [s_t(gen_pert[pname][i], r) - s_t(base_feat[(r["src_style"], r["src_name"])], r)
                for i, r in enumerate(cross)]
        dT_b = [s_t(gen_pert[pname][i], r) - s_t(base_pert[pname][(r["src_style"], r["src_name"])], r)
                for i, r in enumerate(cross)]
        dM_b = [dT_b[i] - (s_c(gen_pert[pname][i], r) - s_c(base_pert[pname][(r["src_style"], r["src_name"])], r))
                for i, r in enumerate(cross)]
        sens[pname] = {
            "dT_gen_only": pack(dT_g, "ΔT (only generated corrupted)"),
            "dT_both": pack(dT_b, "ΔT (both sides corrupted)"),
            "dM_both": pack(dM_b, "ΔMargin (both sides corrupted)"),
            "dT_shift_gen_only": float(np.mean(dT_g) - np.mean(dT0)),
            "dT_shift_both": float(np.mean(dT_b) - np.mean(dT0)),
            "dM_shift_both": float(np.mean(dM_b) - np.mean(dM0)),
        }
    # paired per-image shift (how often a perturbation lifts ΔT)
    for pname in PERTURB:
        dT_g = np.asarray([s_t(gen_pert[pname][i], r) - s_t(base_feat[(r["src_style"], r["src_name"])], r)
                           for i, r in enumerate(cross)])
        sens[pname]["frac_pairs_dT_inflated"] = float((dT_g - np.asarray(dT0) > 0).mean())

    out["sensitivity"] = sens
    out["definitions"] = {
        "csd_score": "cosine(CSD 768-d style descriptor, L2-normalised train-pool prototype)",
        "corruptions": list(PERTURB.keys()),
        "interpretation": "a large positive dT_shift_gen_only means CSD responds to low-level "
                          "appearance rather than artistic style",
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))

    print("\n===== P5 source vs five Style-IDs (CSD cosine) =====")
    print("src/proto".ljust(18) + "".join(f"{n[:14]:>15s}" for n in names))
    for sname in names:
        print(f"{sname:18s}" + "".join(f"{mat[sname][n]:15.4f}" for n in names))
    print("diag", {k: round(v, 4) for k, v in diag.items()})
    print("offdiag", {k: round(v, 4) for k, v in offdiag.items()})
    print("top1", {k: round(v, 3) for k, v in top1.items()}, "overall", round(out["p5"]["overall_top1"], 4))

    print("\n===== CSD low-level perturbation control =====")
    for k, v in sens.items():
        if k == "original":
            print(f"{k:28s} dT {v['dT']['mean']:+.4f} [{v['dT']['ci95'][0]:+.4f},{v['dT']['ci95'][1]:+.4f}] "
                  f"dM {v['dM']['mean']:+.4f} [{v['dM']['ci95'][0]:+.4f},{v['dM']['ci95'][1]:+.4f}]")
        else:
            print(f"{k:28s} dT_gen_only {v['dT_gen_only']['mean']:+.4f} ({v['dT_shift_gen_only']:+.4f}) "
                  f"dT_both {v['dT_both']['mean']:+.4f} ({v['dT_shift_both']:+.4f}) "
                  f"dM_both {v['dM_both']['mean']:+.4f} ({v['dM_shift_both']:+.4f}) "
                  f"frac_dT_up {v['frac_pairs_dT_inflated']:.3f}")
    print(f"\n[p5] wrote {a.out} wall={time.perf_counter() - t_all:.1f}s")


if __name__ == "__main__":
    main()

