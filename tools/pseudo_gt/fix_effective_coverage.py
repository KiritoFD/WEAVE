"""修复旧缓存 manifest 的 effective_coverage：原值写的是名义覆盖率。

对 base/positive/reverse 三份 manifest 逐记录重跑 match_layout（与构造时同一 seed/spec），
把实测覆盖率写入新列 `effective_coverage_measured`，并保留 `effective_coverage` 原值、
在顶层标记该列作废。新缓存由 ShardWriter 直接写实测值，不需要本脚本。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.pseudo_gt import q0_quality as Q  # noqa: E402
from tools.pseudo_gt.build_assets import pgt_root  # noqa: E402
from tools.pseudo_gt.matching import match_layout  # noqa: E402
from tools.pseudo_gt.masks import build_patch_layout  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="base,positive,reverse")
    ap.add_argument("--out", default=str(pgt_root() / "pairs" / "coverage_fix_summary.json"))
    args = ap.parse_args()

    dino = Q.load_dino()
    summary = {}
    for arm in args.arms.split(","):
        root = pgt_root() / "pairs" / arm
        mp = root / "manifest.json"
        if not mp.exists():
            summary[arm] = {"exists": False}
            continue
        man = json.loads(mp.read_text(encoding="utf-8"))
        recs = man["records"]
        t0 = time.time()
        meas, qdiff, miss, nominal = [], [], 0, []
        cache: dict[str, torch.Tensor] = {}

        def pat(style: str, img_id: str) -> torch.Tensor | None:
            key = f"{style}|{img_id}"
            if key in cache:
                return cache[key]
            idx = dino[style]["index"].get(img_id)
            if idx is None:
                return None
            v = F.normalize(dino[style]["patches"][idx].to("cuda").float(), dim=-1)
            if len(cache) > 512:
                cache.clear()
            cache[key] = v
            return v

        def pat_any(img_id: str) -> torch.Tensor | None:
            """reverse 臂的 donor 属于另一风格，record 里没有记它，需跨风格查找。"""
            for style in dino:
                v = pat(style, img_id)
                if v is not None:
                    return v
            return None

        for r in recs:
            sp = pat(r["source_style"], r["source_id"])
            dp = pat(r["target_style"], r["donor_id"])
            if dp is None:
                dp = pat_any(r["donor_id"])
            if sp is None or dp is None:
                miss += 1
                continue
            rects = build_patch_layout(r["patch_spec"], seed=int(r["seed"]))
            m, q = match_layout(sp, dp, rects)
            cov = float(sum(x.src.area for x in (m or [])) / 512 ** 2)
            r["effective_coverage_measured"] = cov
            r["effective_coverage_original_value"] = r.get("effective_coverage")
            meas.append(cov)
            nominal.append(float(sum(x.area for x in rects) / 512 ** 2))
            qdiff.append(abs(q - float(r.get("match_score", 0.0))) if m is not None else 0.0)
        summary[arm] = {
            "exists": True, "records": len(recs), "measured_n": len(meas), "missing_patches": miss,
            "coverage_measured_mean": sum(meas) / max(len(meas), 1),
            "coverage_original_mean": sum(float(r.get("effective_coverage", 0)) for r in recs) / max(len(recs), 1),
            "frac_images_with_drop": sum(1 for c, n in zip(meas, nominal) if c < n - 1e-9) / max(len(meas), 1),
            "q_recompute_vs_recorded_maxdiff": max(qdiff) if qdiff else None,
            "seconds": round(time.time() - t0, 1),
        }
        man["effective_coverage_deprecated"] = True
        man["effective_coverage_note"] = (
            "effective_coverage 旧列写的是名义值，已作废；请使用 effective_coverage_measured 与 "
            "effective_coverage_original_value")
        man["effective_coverage_measured_mean"] = summary[arm]["coverage_measured_mean"]
        tmp = mp.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(mp)
        print(f"[coverage-fix] {arm}: {summary[arm]}", flush=True)

    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[coverage-fix] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
