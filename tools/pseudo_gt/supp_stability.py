"""多样本池稳定性：每个配置在 pool1/pool2/pool3 上分别给 CI，并检查符号一致性。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import paths  # noqa: E402
from tools.pseudo_gt.supp_analyze import boot_mean, derive, load  # noqa: E402

KEYS = ["dT_F", "dS_F", "dM_F", "dT_R", "dS_R", "dM_R", "dino_c", "lpips", "seam_ratio"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", action="append", required=True, help="name=records.jsonl，可重复")
    ap.add_argument("--out", default=str(Path(paths.project_root()) / "runs/p2/supp_stability.md"))
    ap.add_argument("--json", default=str(Path(paths.project_root()) / "runs/p2/supp_stability.json"))
    args = ap.parse_args()

    rng = np.random.default_rng(20261106)
    pools = {}
    for item in args.pool:
        name, path = item.split("=", 1)
        by_cfg, _ = load([path])
        pools[name] = by_cfg

    configs = sorted({c for p in pools.values() for c in p})
    table = {}
    for cfg in configs:
        row = {}
        for name, by_cfg in pools.items():
            rows = by_cfg.get(cfg)
            if not rows:
                continue
            cur = {k: boot_mean([r[k] for r in rows.values()],
                                [r["source_style"] for r in rows.values()], rng) for k in KEYS}
            row[name] = {"n": len(rows), **cur}
        if len(row) >= 2:
            agree = {}
            for k in ("dT_F", "dM_F", "dT_R", "dino_c", "lpips", "seam_ratio"):
                signs = [np.sign(v[k]["mean"]) for v in row.values() if not np.isnan(v[k]["mean"])]
                agree[k] = bool(len(set(signs)) == 1)
            spread = {k: float(np.nanmax([v[k]["mean"] for v in row.values()])
                               - np.nanmin([v[k]["mean"] for v in row.values()]))
                      for k in ("dT_F", "dM_F", "dT_R")}
            row["_sign_agree"] = agree
            row["_spread"] = spread
        table[cfg] = row

    Path(args.json).write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    md = ["# 多样本池稳定性（pool1/pool2/pool3）", "",
          "| 配置 | 池 | n | ΔT_F | ΔM_F | ΔT_R | DINO-C | LPIPS | R_seam |",
          "|---|---|---|---|---|---|---|---|---|"]
    for cfg, row in table.items():
        for name, v in row.items():
            if name.startswith("_"):
                continue
            md.append(f"| {cfg} | {name} | {v['n']} | "
                      + " | ".join(f"{v[k]['mean']:.3f} [{v[k]['lo']:.3f},{v[k]['hi']:.3f}]"
                                   for k in ("dT_F", "dM_F", "dT_R", "dino_c", "lpips")) +
                      f" | {v['seam_ratio']['mean']:.2f} |")
        if "_sign_agree" in row:
            md.append(f"| {cfg} | **符号一致** | | " +
                      ", ".join(f"{k}={v}" for k, v in row["_sign_agree"].items()) +
                      f" | spread ΔT={row['_spread']['dT_F']:.3f} |")
    Path(args.out).write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[stability] -> {args.out}")


if __name__ == "__main__":
    main()
