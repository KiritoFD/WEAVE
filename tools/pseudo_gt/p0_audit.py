"""P0 审计：覆盖率实测 / 匹配通过率 / 缓存一致性 / 反向评分口径。

输出 runs/p2/pgt_p0_audit.json。GPU 只用于匹配（复用已缓存的 DINO 特征）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools" / "12h"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "trt"))

from tools.pseudo_gt import q0_quality as Q  # noqa: E402
from tools.pseudo_gt.build_assets import STYLES, load_manifest, pgt_root  # noqa: E402
from tools.pseudo_gt.masks import E_SPECS, PATCH_SPECS, build_patch_layout  # noqa: E402


def _records(root: Path) -> list:
    mf = root / "manifest.json"
    if not mf.exists():
        return []
    d = json.loads(mf.read_text(encoding="utf-8"))
    return d.get("records", d if isinstance(d, list) else [])


def cache_stats(arm: str) -> dict:
    root = pgt_root() / "pairs" / arm
    recs = _records(root)
    if not recs:
        return {"exists": False}
    ids = [r.get("sample_id") for r in recs]
    shards = sorted(root.glob("shard_*.pt"))
    rows = 0
    for sh in shards:
        import torch

        payload = torch.load(sh, map_location="cpu", weights_only=False)
        rows += int(payload["z0"].shape[0])
    return {
        "exists": True,
        "records": len(recs),
        "unique_sample_ids": len(set(ids)),
        "dup": len(ids) - len(set(ids)),
        "shards": len(shards),
        "rows": rows,
        "specs": sorted({r.get("patch_spec") for r in recs}),
        "methods": sorted({r.get("method") for r in recs}),
        "sources": len({r.get("source_id") for r in recs}),
        "nominal_cov_mean": float(np.mean([r.get("nominal_coverage", 0.0) for r in recs])),
        "effective_cov_mean": float(np.mean([r.get("effective_coverage", 0.0) for r in recs])),
        "effective_equals_nominal": bool(all(
            abs(float(r.get("effective_coverage", 0)) - float(r.get("nominal_coverage", 0)) < 1e-9)
            for r in recs
        )),
    }


def cache_recompute(arm: str, *, n_sample: int, dino: dict, seed: int) -> dict:
    """用缓存里记录的 donor/seed 重算匹配，核对 q 与实际覆盖面积。"""
    recs = _records(pgt_root() / "pairs" / arm)
    if not recs:
        return {"exists": False}
    step = max(1, len(recs) // n_sample)
    picked = recs[::step][:n_sample]
    import torch
    import torch.nn.functional as F

    from tools.pseudo_gt.matching import match_layout

    rows = []
    for r in picked:
        spec, sid, donor = r["patch_spec"], r["source_id"], r["donor_id"]
        s_style, d_style = r["source_style"], r["target_style"]
        si = dino[s_style]["index"].get(sid)
        di = dino[d_style]["index"].get(donor)
        if si is None or di is None:
            continue
        sp = F.normalize(dino[s_style]["patches"][si].float().to("cuda"), dim=-1)
        dp = F.normalize(dino[d_style]["patches"][di].float().to("cuda"), dim=-1)
        rects = build_patch_layout(spec, seed=int(r["seed"]))
        m, q = match_layout(sp, dp, rects)
        if m is None:
            rows.append({"spec": spec, "failed": True})
            continue
        rows.append({
            "spec": spec,
            "failed": False,
            "coverage": float(sum(x.src.area for x in m) / 512 ** 2),
            "nominal": float(sum(x.area for x in rects) / 512 ** 2),
            "patches": len(m),
            "nominal_patches": len(rects),
            "q": float(q),
            "q_recorded": float(r.get("match_score", 0.0)),
        })
    ok = [x for x in rows if not x.get("failed")]
    if not ok:
        return {"exists": True, "n": len(rows), "failed": len(rows)}
    cov = np.array([x["coverage"] for x in ok])
    qd = np.array([abs(x["q"] - x["q_recorded"]) for x in ok])
    return {
        "exists": True,
        "n": len(rows),
        "failed": len(rows) - len(ok),
        "coverage_mean": float(cov.mean()),
        "coverage_median": float(np.median(cov)),
        "nominal_mean": float(np.mean([x["nominal"] for x in ok])),
        "frac_images_with_drop": float(np.mean([x["patches"] < x["nominal_patches"] for x in ok])),
        "drop_frac_mean": float(np.mean([1 - x["patches"] / x["nominal_patches"] for x in ok])),
        "q_recompute_mean": float(np.mean([x["q"] for x in ok])),
        "q_recorded_mean": float(np.mean([x["q_recorded"] for x in ok])),
        "q_abs_diff_max": float(qd.max()),
        "q_abs_diff_mean": float(qd.mean()),
    }


def reverse_algebra() -> dict:
    """ΔT_R = S_s(x_s) − S_s(h) = −ΔS_F；ΔS_R = −ΔT_F；ΔM_R = ΔM_F。"""
    path = Path(Q.paths.project_root()) / "runs/p2/pgt_q0.json"
    if not path.exists():
        return {"exists": False}
    q0 = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for key, rec in q0["configs"].items():
        if not key.startswith("E") or not rec.get("csd_dT"):
            continue
        dT_f, dS_f = rec["csd_dT"], rec["csd_dS"]
        out[key] = {
            "n": rec["n"],
            "dT_R_mean": -dS_f["mean"],
            "dT_R_ci": {"lo": -dS_f["ci"]["hi"], "hi": -dS_f["ci"]["lo"]},
            "dS_R_mean": -dT_f["mean"],
            "dS_R_ci": {"lo": -dT_f["ci"]["hi"], "hi": -dT_f["ci"]["lo"]},
            "dM_R_mean": rec["csd_dM"]["mean"],
            "dM_R_ci": rec["csd_dM"]["ci"],
            "legacy_dT_mean": dT_f["mean"],
            "legacy_dS_mean": dS_f["mean"],
        }
    return {"exists": True, "configs": out}


def band_audit() -> dict:
    path = Path(Q.paths.project_root()) / "runs/p2/pgt_q0.json"
    if not path.exists():
        return {"exists": False}
    q0 = json.loads(path.read_text(encoding="utf-8"))
    keys = ["A_P5", "C_P5", "D_P5", "D_HF_P5", "A_P2", "A_P1", "A_P3"]
    return {
        k: {b: round(q0["configs"][k][f"band_{b}"], 4) for b in ("LL", "LH", "HL", "HH")}
        for k in keys if k in q0["configs"]
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(Q.paths.project_root()) / "runs/p2/pgt_p0_audit.json"))
    ap.add_argument("--cache-sample", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F

    from tools.pseudo_gt.matching import match_layout

    items = load_manifest()
    repo_root = Path(__file__).resolve().parents[2]
    test_stems = {p.stem for p in (repo_root / "data/test").rglob("*.jpg")}
    test_stems |= {p.stem for p in (repo_root / "datasets/distinct5_512/test").rglob("*.jpg")}
    sources = Q.shared_sources(items, test_stems)
    if args.limit:
        sources = sources[: args.limit]
    print(f"[p0] sources={len(sources)}", flush=True)

    lat, dino, retrieval = Q.load_latents(), Q.load_dino(), Q.load_retrieval()
    specs = list(PATCH_SPECS) + list(E_SPECS)
    layouts = {s: build_patch_layout(s, seed=Q.SEED) for s in specs}
    per = {s: [] for s in specs}
    failed = {s: 0 for s in specs}

    for k, it in enumerate(sources):
        tgt = Q.target_style_for(it["style"])
        ret = retrieval[it["style"]]
        cands = ret["by_target"][tgt]["donor_ids"][ret["source_ids"].index(it["id"])]
        cands = [c for c in cands if dino[tgt]["index"].get(c) is not None][:8]
        sp = F.normalize(dino[it["style"]]["patches"][dino[it["style"]]["index"][it["id"]]].to("cuda").float(), dim=-1)
        # 供体只按 P5 选一次（与 Q0 同源，避免 8 候选 × 24 布局的重复匹配）
        donor_id, best_q, sp5 = None, -1.0, None
        for c in cands:
            dp = F.normalize(dino[tgt]["patches"][dino[tgt]["index"][c]].to("cuda").float(), dim=-1)
            ms, q = match_layout(sp, dp, layouts["P5"])
            if ms is not None and ms and q > best_q:
                donor_id, best_q, sp5 = c, q, dp
        if donor_id is None:
            for s in specs:
                failed[s] += 1
            continue
        for s in specs:
            ms, q = match_layout(sp, sp5, layouts[s])
            if ms is None or not ms:
                failed[s] += 1
                continue
            area = sum(m.src.area for m in ms)
            per[s].append({
                "source_id": it["id"],
                "n_patch": len(ms),
                "coverage": area / (512 ** 2),
                "nominal": sum(r.area for r in layouts[s]) / (512 ** 2),
                "q_mean": float(np.mean([m.q for m in ms])) if ms else 0.0,
            })
        if (k + 1) % 40 == 0:
            print(f"  {k+1}/{len(sources)}", flush=True)

    out = {"n_sources": len(sources), "seed": Q.SEED, "specs": {}, "cache": {}}
    for s in specs:
        rows = per[s]
        if not rows:
            out["specs"][s] = {"n": 0, "failed": failed[s]}
            continue
        cov = np.array([r["coverage"] for r in rows])
        out["specs"][s] = {
            "n": len(rows),
            "failed": failed[s],
            "nominal_coverage": float(np.mean([r["nominal"] for r in rows])),
            "coverage_mean": float(cov.mean()),
            "coverage_median": float(np.median(cov)),
            "coverage_min": float(cov.min()),
            "coverage_max": float(cov.max()),
            "patches_mean": float(np.mean([r["n_patch"] for r in rows])),
            "nominal_patches": len(layouts[s]),
            "frac_images_with_drop": float(np.mean([r["n_patch"] < len(layouts[s]) for r in rows])),
            "q_mean": float(np.mean([r["q_mean"] for r in rows])),
        }
    for arm in ("base", "positive", "reverse", "reverse_e6"):
        out["cache"][arm] = cache_stats(arm)
        out["cache"][arm]["recompute"] = cache_recompute(
            arm, n_sample=args.cache_sample, dino=dino, seed=Q.SEED)
    out["reverse_algebra"] = reverse_algebra()
    out["band_audit"] = band_audit()
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2)[:4000], flush=True)


if __name__ == "__main__":
    main()
