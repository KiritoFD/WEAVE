"""V11 P3 scoring: can the shared model actually generate each style?

Unconditional samples have no source image, so the transport metrics (dT / dS / DINO-C /
LPIPS) do not apply. What matters instead:

  style_top1   fraction of generated style-j images whose CSD argmax is style j
               -> is the sample recognisable as the intended style at all?
  style_margin v[j] - max_{k != j} v[k]
  csd_swd      distribution distance to that style's frozen reference set
  csd_mmd2     same, kernel MMD with the evaluator's gamma
  vs_real_swd  the same distance computed between the reference set and the real test
               pool of that style, i.e. the achievable floor

The floor matters: if real-vs-real SWD is already ~0.024 then a generated value of
0.026 is close to the ceiling of this metric and must not be read as "bad".
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]


def _spread(cloud: np.ndarray) -> float:
    """Mean pairwise cosine distance inside a cloud (1 - cos). Higher = more diverse."""
    x = cloud / np.clip(np.linalg.norm(cloud, axis=1, keepdims=True), 1e-12, None)
    sim = x @ x.T
    n = len(x)
    if n < 2:
        return 0.0
    iu = np.triu_indices(n, k=1)
    return float(1.0 - sim[iu].mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--gen", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    gen_root = Path(args.gen)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(repo))

    from sbflow.score_cpu import CpuFeatures, _reference_csd, load_u8, mmd2, swd
    man = json.loads((gen_root / "manifest.json").read_text())
    tasks = man["tasks"]

    feat = CpuFeatures(repo, args.threads)
    refs = _reference_csd(repo, feat, repo / "eval_protocol/cache/references_v1_csd_cpu.pt")
    gamma = float(refs["rbf_gamma"])
    proto = np.asarray(feat.proto, dtype=np.float32)

    by_style: Dict[str, List[dict]] = {s: [] for s in STYLES}
    for t in tasks:
        s = t["style"]
        if s not in by_style:
            continue
        a = load_u8(gen_root / t["image"])
        f = feat.features([Image.fromarray(a)])
        csd = np.asarray(f["csd"], dtype=np.float32)[0]
        v = csd @ proto.T
        j = STYLES.index(s)
        other = np.delete(v, j)
        by_style[s].append({"csd": csd, "style_top1": int(v.argmax() == j),
                            "style_margin": float(v[j] - other.max()),
                            "own_score": float(v[j])})

    report = {}
    for s in STYLES:
        rows = by_style[s]
        if not rows:
            continue
        cloud = np.stack([r["csd"] for r in rows], axis=0)
        ref = refs["styles"][s]["csd"]
        report[s] = {
            "n": len(rows),
            "style_top1": float(np.mean([r["style_top1"] for r in rows])),
            "style_margin": float(np.mean([r["style_margin"] for r in rows])),
            "own_score": float(np.mean([r["own_score"] for r in rows])),
            "csd_swd": swd(cloud, ref),
            "csd_mmd2": mmd2(cloud, ref, gamma),
            # Diversity guard: a model that emits one good Rococo 100 times would pass
            # style_top1 while collapsing the distribution. Compare the mean pairwise
            # cosine distance inside the generated cloud against the same statistic for
            # the real reference cloud.
            "spread_gen": _spread(cloud),
            "spread_real": _spread(np.asarray(ref, dtype=np.float32)),
        }
        r = report[s]
        r["spread_ratio"] = (r["spread_gen"] / r["spread_real"]
                             if r["spread_real"] > 0 else float("nan"))

    overall = {
        "n": sum(v["n"] for v in report.values()),
        "style_top1": float(np.mean([v["style_top1"] for v in report.values()])),
        "style_margin": float(np.mean([v["style_margin"] for v in report.values()])),
        "csd_swd": float(np.mean([v["csd_swd"] for v in report.values()])),
        "csd_mmd2": float(np.mean([v["csd_mmd2"] for v in report.values()])),
    }

    out.mkdir(parents=True, exist_ok=True)
    (out / "uncond_scores.json").write_text(
        json.dumps({"protocol": {"gen": str(gen_root), "mmd_gamma": gamma,
                                 "note": "unconditional: no source image, so transport "
                                         "metrics do not apply"},
                    "overall": overall, "by_style": report},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# V11 P3：无条件五风格生成能力", "",
             f"生成目录：`{gen_root}`", "",
             "| 风格 | n | style Top-1 | style Margin | own score | CSD-SWD | CSD-MMD | 多样性比 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in STYLES:
        if s in report:
            r = report[s]
            lines.append(f"| {s} | {r['n']} | {r['style_top1']:.4f} | "
                         f"{r['style_margin']:+.4f} | {r['own_score']:.4f} | "
                         f"{r['csd_swd']:.5f} | {r['csd_mmd2']:.5f} | "
                         f"{r['spread_ratio']:.3f} |")
    lines += ["", f"**overall style Top-1 = {overall['style_top1']:.4f}，"
                  f"mean SWD = {overall['csd_swd']:.5f}**", "",
              "多样性比 = 生成云内平均成对余弦距离 / 真实参考云内同一统计量。",
              "接近 1 表示多样性正常；明显 < 1 说明模式塌缩（即使 style Top-1 很高）。",
              "SWD 地板参考 0.0111（真实 vs 参考集）。", ""]
    (out / "uncond_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
