"""P2：FM 监督可达性审计（V4）。

对每条缓存配对 (z0, z1)：
    u      = z1 - z0
    z_reach = IDWT(LL1, LH1, HL1, HH0)     # 只保留三头网络可预测的三个带
比较 z1（真实伪 GT）与 z_reach：
    - CSD 目标风格增益（相对 z0 源图）
    - 内容变化（DINO-C 对 z1 / z0）
    - HH 残余能量（‖HH1-HH0‖/‖u‖、HH 占 u 能量比）
若某方案的风格增益主要落在 HH，则该方案不可能被当前三头网络复现。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools" / "12h"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "trt"))

import csd_model  # type: ignore  # noqa: E402
from wavelet import dwt2_haar, idwt2_haar  # noqa: E402

from tools.pseudo_gt import q0_quality as Q  # noqa: E402
from tools.pseudo_gt.build_assets import STYLES, pgt_root  # noqa: E402
from tools.pseudo_gt.cache import load_shard  # noqa: E402
from tools.pseudo_gt.matching import encode_dino  # noqa: E402


def _mean(xs):
    return float(np.mean(xs)) if xs else float("nan")


def load_records(root: Path):
    man = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    recs = man["records"]
    n_shards = len(sorted(root.glob("shard_*.pt")))
    return recs, n_shards


def pick_indices(n_records: int, n_shards: int, n: int) -> list[tuple[int, int]]:
    """按全局位置均匀抽样，返回 (shard_idx, row_in_shard) 列表。"""
    if n_records == 0:
        return []
    want = min(n, n_records)
    out, last = [], -1
    for g in np.linspace(0, n_records - 1, want).round().astype(int):
        si, row = divmod(int(g), 1000)
        si = min(si, max(n_shards - 1, 0))
        key = si * 1000 + row
        if key == last:
            continue
        last = key
        out.append((si, row))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", required=True,
                    help="逗号分隔 name=pairs子目录，如 A_P5=positive,E7=reverse")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default=str(Path(pgt_root()) / "p2_reach.json"))
    args = ap.parse_args()

    t0 = time.time()
    metric = Q.Metrics()
    lat = Q.load_latents()
    protos = Q.build_prototypes(metric, lat, set())
    proto_ids = list(STYLES)
    proto_mat = np.stack([protos[s] for s in proto_ids])
    print(f"[p2] models ready {time.time()-t0:.0f}s", flush=True)

    report: dict[str, dict] = {}
    all_rows: list[dict] = []
    for spec in args.sets.split(","):
        spec = spec.strip()
        if not spec:
            continue
        name, _, sub = spec.partition("=")
        root = Path(pgt_root()) / "pairs" / sub
        recs, n_shards = load_records(root)
        idxs = pick_indices(len(recs), n_shards, args.n)
        shards: dict[int, dict] = {}
        rows = []
        for si, row in idxs:
            if si not in shards:
                shards[si] = load_shard(root, si)
            sh = shards[si]
            row = min(row, len(sh["meta"]) - 1)
            rec = sh["meta"][row]
            z0 = sh["z0"][row].float()[None]
            z1 = sh["z1"][row].float()[None]
            ll1, lh1, hl1, hh1 = dwt2_haar(z1)
            ll0, lh0, hl0, hh0 = dwt2_haar(z0)
            zr = idwt2_haar(ll1, lh1, hl1, hh0)
            u = z1 - z0
            e = {
                "LL": float(((ll1 - ll0) ** 2).sum()),
                "LH": float(((lh1 - lh0) ** 2).sum()),
                "HL": float(((hl1 - hl0) ** 2).sum()),
                "HH": float(((hh1 - hh0) ** 2).sum()),
            }
            tot = sum(e.values()) + 1e-12
            rows.append({
                "z0": z0, "z1": z1, "zr": zr,
                "source_id": rec["source_id"], "donor_id": rec["donor_id"],
                "target_style": rec["target_style"], "source_style": rec["source_style"],
                "nominal": rec.get("nominal_coverage"),
                "measured": rec.get("effective_coverage_measured", rec.get("effective_coverage")),
                "weighted": rec.get("effective_coverage_weighted", rec.get("effective_coverage")),
                "match_score": rec.get("match_score"),
                "u_norm": float(u.norm()),
                "hh_share": e["HH"] / tot,
                "band_share": {k: v / tot for k, v in e.items()},
                "hh_norm_ratio": float((hh1 - hh0).norm()) / max(float(u.norm()), 1e-9),
            })

        # 分批解码/打分（每批 8 条，24 张图）
        for s in range(0, len(rows), 8):
            chunk = rows[s:s + 8]
            z_cat = torch.cat([r["z0"] for r in chunk] + [r["z1"] for r in chunk] + [r["zr"] for r in chunk], 0)
            pils = metric.decode(z_cat)
            n = len(chunk)
            csd = csd_model.feats_pils(metric.csd, pils, batch=len(pils))
            cls, _ = encode_dino(metric.dino, pils, batch=len(pils))
            for j, r in enumerate(chunk):
                i0, i1, ir = j, n + j, 2 * n + j
                t_idx = proto_ids.index(r["target_style"])
                s_idx = proto_ids.index(r["source_style"])
                T0 = float(csd[i0] @ proto_mat[t_idx])
                T1 = float(csd[i1] @ proto_mat[t_idx])
                TR = float(csd[ir] @ proto_mat[t_idx])
                S0 = float(csd[i0] @ proto_mat[s_idx])
                S1 = float(csd[i1] @ proto_mat[s_idx])
                SR = float(csd[ir] @ proto_mat[s_idx])
                r.update({
                    "csd_dT_pseudo": T1 - T0,
                    "csd_dT_reach": TR - T0,
                    "csd_retention": (TR - T0) / (T1 - T0) if (T1 - T0) > 1e-6 else float("nan"),
                    "csd_dS_pseudo": S1 - S0,
                    "csd_dS_reach": SR - S0,
                    "dino_reach_vs_pseudo": float(cls[ir] @ cls[i1]),
                    "dino_reach_vs_source": float(cls[ir] @ cls[i0]),
                    "dino_pseudo_vs_source": float(cls[i1] @ cls[i0]),
                })
        for r in rows:
            all_rows.append({"set": name, **{k: v for k, v in r.items() if k not in ("z0", "z1", "zr")}})
        report[name] = {
            "root": str(root), "records_total": len(recs), "sampled": len(rows),
            "coverage_nominal_mean": _mean([r["nominal"] for r in rows]),
            "coverage_measured_mean": _mean([r["measured"] for r in rows]),
            "coverage_weighted_mean": _mean([r["weighted"] for r in rows]),
            "frac_measured_ge_022": _mean([1.0 if (r["measured"] or 0) >= 0.22 else 0.0 for r in rows]),
            "frac_measured_lt_018": _mean([1.0 if (r["measured"] or 0) < 0.18 else 0.0 for r in rows]),
            "match_score_mean": _mean([r["match_score"] for r in rows]),
            "csd_dT_pseudo": _mean([r["csd_dT_pseudo"] for r in rows]),
            "csd_dT_reach": _mean([r["csd_dT_reach"] for r in rows]),
            "csd_retention": _mean([r["csd_retention"] for r in rows]),
            "csd_dS_pseudo": _mean([r["csd_dS_pseudo"] for r in rows]),
            "csd_dS_reach": _mean([r["csd_dS_reach"] for r in rows]),
            "hh_share_u": _mean([r["hh_share"] for r in rows]),
            "hh_norm_ratio": _mean([r["hh_norm_ratio"] for r in rows]),
            "band_share": {k: _mean([r["band_share"][k] for r in rows]) for k in ("LL", "LH", "HL", "HH")},
            "dino_reach_vs_pseudo": _mean([r["dino_reach_vs_pseudo"] for r in rows]),
            "dino_reach_vs_source": _mean([r["dino_reach_vs_source"] for r in rows]),
            "dino_pseudo_vs_source": _mean([r["dino_pseudo_vs_source"] for r in rows]),
        }
        print(f"[p2] {name}: {json.dumps(report[name], ensure_ascii=False)}", flush=True)

    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(str(args.out).replace(".json", "_rows.jsonl"), "w", encoding="utf-8") as fh:
        for r in all_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[p2] done {time.time()-t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
