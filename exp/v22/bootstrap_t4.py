"""V22 T 矩阵/T4 扫描配对 bootstrap CI：同 task 配对差值 × 源聚类重采样（cross600）。

用法 (4090):
  python -m exp.v22.bootstrap_t4
"""
from __future__ import annotations

import json

import numpy as np

from . import OUT, save_json, utc
from src.eval.metrics import bootstrap_source_ci

PAIRS = [
    ("T4s_wc100", "T0_slf_final_wc11"),
    ("T4s_wc150", "T0_slf_final_wc14"),
    ("T4_wc11", "T0_slf_final_wc11"),
    ("T1_wc11", "T0_slf_final_wc11"),
    ("T2_wc11", "T1_wc11"),
    ("T4_wc11", "T4s_wc100"),
    ("T4Ls_wc100", "T4s_wc100"),  # T4L 训练完成后自动补
]


def load_rows(arm: str) -> dict[str, dict]:
    p = OUT / "measured" / arm / "eval.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text(encoding="utf-8"))
    return {r["task_id"]: r for r in d["rows"] if r.get("is_cross")}


def pair_boot(arm_a: str, arm_b: str) -> dict | None:
    ra, rb = load_rows(arm_a), load_rows(arm_b)
    common = sorted(set(ra) & set(rb))
    if len(common) < 100:
        print(f"[BOOT] SKIP {arm_a} vs {arm_b}: only {len(common)} common cross rows")
        return None
    srcs = np.array([ra[t]["src_stem"] for t in common])

    def vals(rows, key, binarize=None):
        v = [rows[t].get(key) for t in common]
        if binarize:
            tgt = np.array([rows[t]["tgt_style_id"] for t in common])
            pred = np.array([rows[t].get("pred_class", -1) for t in common])
            v = (pred == tgt).astype(float)
        return np.asarray(v, float)

    out: dict[str, dict] = {}
    for key, binarize in (("top1", True), ("dino_s", False), ("dino_c", False), ("lpips", False),
                          ("style_affinity", False), ("clip_s", False)):
        try:
            if binarize:
                va, vb = vals(ra, key, True), vals(rb, key, True)
            else:
                va, vb = vals(ra, key), vals(rb, key)
        except Exception:
            continue
        if np.isnan(va).any() or np.isnan(vb).any():
            continue
        diff = va - vb
        ci = bootstrap_source_ci(diff, srcs, n_boot=4000, seed=7)
        out[key] = {"mean": round(float(va.mean()), 5), "mean_b": round(float(vb.mean()), 5),
                    "delta": round(float(diff.mean()), 5), **ci}
    print(f"[BOOT] {arm_a} vs {arm_b} (n={len(common)}):")
    for k, v in out.items():
        sig = "SIG" if v.get("excludes_zero") else "ns "
        print(f"  {k:16s} Δ={v['delta']:+.5f} CI[{v['ci95'][0]:+.5f},{v['ci95'][1]:+.5f}] {sig}")
    return out


def main():
    res = {}
    for a, b in PAIRS:
        r = pair_boot(a, b)
        if r:
            res[f"{a}_vs_{b}"] = r
    save_json(OUT / "t4_bootstrap_pairs.json", {"updated_utc": utc(), "pairs": res})
    print(f"\nsaved -> {OUT / 't4_bootstrap_pairs.json'}")


if __name__ == "__main__":
    main()
