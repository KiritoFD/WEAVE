"""Q0 级质量对照：无注入 / 随机内容匹配 / 同风格注入 / 仓库 destyle_latent。

与 Q0 共用同一批 300 源图、同一 P5 供体与匹配，保证与 A_P5 可直接对比。
输出 schema 与 pgt_q0.json 一致（configs 键为 CTRL_*）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import paths  # noqa: E402
from b2_destyle import destyle_latent  # noqa: E402
from tools.pseudo_gt import construct as C  # noqa: E402
from tools.pseudo_gt import q0_quality as Q  # noqa: E402
from tools.pseudo_gt.masks import Rect, build_patch_layout  # noqa: E402
from tools.pseudo_gt.matching import DEVICE, Match  # noqa: E402

SPEC = "P5"
KEYS = ("CTRL_none", "CTRL_rand", "CTRL_same", "CTRL_destyle")


def _row(plan, method, key, sc, j, z_gen, rects, coverage, seam=None):
    src_it = plan["source"]
    seam_v = seam if seam is not None else C.seam_gradient(
        torch.from_numpy(np.array(sc["gen"][j], dtype=np.float32) / 255.0).permute(2, 0, 1)[None],
        rects,
    )
    base = plan["base"]
    return {
        "source_id": src_it["id"],
        "source_style": src_it["style"],
        "target_style": plan["target_style"],
        "donor_id": plan["donor_id"],
        "q": float(plan["qs"].get(SPEC, 0.0)),
        "csd_dT": sc["csd_T"][j] - base["csd_T"],
        "csd_dS": sc["csd_S"][j] - base["csd_S"],
        "csd_dM": (sc["csd_T"][j] - base["csd_T"]) - (sc["csd_S"][j] - base["csd_S"]),
        "dino_c": sc["dino_c"][j],
        "lpips": sc["lpips"][j],
        "seam": seam_v,
        "bands": C.band_energy_share(plan["z_src"], z_gen),
        "coverage": float(coverage),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(paths.project_root()) / "runs/p2/pgt_q0_controls.json"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=Q.SEED)
    args = ap.parse_args()
    t0 = time.time()

    items = Q.load_manifest()
    test_stems = {p.stem for p in (REPO_ROOT / "data/test").rglob("*.jpg")}
    test_stems |= {p.stem for p in (REPO_ROOT / "datasets/distinct5_512/test").rglob("*.jpg")}
    sources = Q.shared_sources(items, test_stems)
    if args.limit:
        sources = sources[: args.limit]
    print(f"[ctrl] sources={len(sources)}", flush=True)

    lat, dino, retrieval = Q.load_latents(), Q.load_dino(), Q.load_retrieval()
    metric = Q.Metrics()
    protos = Q.build_prototypes(metric, lat, {it["id"] for it in sources})
    rects = build_patch_layout(SPEC, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    id_map = {it["id"]: it for it in items}

    rows = {k: [] for k in KEYS}
    failed = {k: 0 for k in KEYS}
    errs: dict[str, str] = {}

    for k, it in enumerate(sources):
        tgt = Q.target_style_for(it["style"])
        ret = retrieval[it["style"]]
        cands = ret["by_target"][tgt]["donor_ids"][ret["source_ids"].index(it["id"])]
        cands = [c for c in cands if dino[tgt]["index"].get(c) is not None][:8]
        donor_id, matches, qs = Q.pick_donor(it["style"], it["id"], tgt, cands, dino, {SPEC: rects})
        if donor_id is None:
            for key in KEYS:
                failed[key] += 1
            continue
        m_p5 = matches[SPEC]
        src_img = Q._load_rgb(it["path"])
        src_rgb = Q._to_tensor(src_img)[None]
        proto_t = protos[tgt]
        proto_s = protos[it["style"]]
        base = metric.source_baseline(src_img, proto_t, proto_s)
        z_src = lat[it["style"]]["latents"][lat[it["style"]]["index"][it["id"]]][None].float().to(DEVICE)
        z_don = lat[tgt]["latents"][lat[tgt]["index"][donor_id]][None].float().to(DEVICE)
        plan = {"source": it, "target_style": tgt, "donor_id": donor_id, "matches": matches,
                "qs": qs, "z_src": z_src, "z_don": z_don, "base": base}

        # 1) 无注入：pseudo = 源图本身（VAE 往返）
        try:
            z_none = metric.encode(src_rgb)
            sc = metric.score(z_none, [src_img], proto_t, proto_s, base)
            seam_none = C.seam_gradient(
                torch.from_numpy(np.array(sc["gen"][0], dtype=np.float32) / 255.0).permute(2, 0, 1)[None], [])
            rows["CTRL_none"].append(_row(plan, "CTRL", "CTRL_none", sc, 0, z_none, [], 0.0, seam=seam_none))
        except Exception as exc:  # noqa: BLE001
            failed["CTRL_none"] += 1
            errs["CTRL_none"] = f"{type(exc).__name__}: {exc}"

        # 2) 随机内容匹配：目标风格里随机换一个供体，位置沿用 P5 几何（同一坐标）
        try:
            pool_t = [k2 for k2 in dino[tgt]["index"].keys() if k2 != donor_id and k2 in id_map]
            pick = pool_t[int(rng.integers(len(pool_t)))]
            don2 = Q._to_tensor(Q._load_rgb(id_map[pick]["path"]))[None]
            same = [Match(src=r, donor=Rect(x=r.x, y=r.y, w=r.w, h=r.h), q=0.0, mutual_ratio=0.0) for r in rects]
            z_rand = metric.encode(C.construct_A(src_rgb, don2, same))
            sc = metric.score(z_rand, [src_img], proto_t, proto_s, base)
            rows["CTRL_rand"].append(_row(plan, "CTRL", "CTRL_rand", sc, 0, z_rand, [m.src for m in same],
                                          0.25))
        except Exception as exc:  # noqa: BLE001
            failed["CTRL_rand"] += 1
            errs["CTRL_rand"] = f"{type(exc).__name__}: {exc}"

        # 3) 同风格注入：从源风格随机取供体
        try:
            pool_s = [k2 for k2 in dino[it["style"]]["index"].keys() if k2 != it["id"] and k2 in id_map]
            pick = pool_s[int(rng.integers(len(pool_s)))]
            don3 = Q._to_tensor(Q._load_rgb(id_map[pick]["path"]))[None]
            same = [Match(src=r, donor=Rect(x=r.x, y=r.y, w=r.w, h=r.h), q=0.0, mutual_ratio=0.0) for r in rects]
            z_same = metric.encode(C.construct_A(src_rgb, don3, same))
            sc = metric.score(z_same, [src_img], proto_t, proto_s, base)
            rows["CTRL_same"].append(_row(plan, "CTRL", "CTRL_same", sc, 0, z_same, [m.src for m in same],
                                          0.25))
        except Exception as exc:  # noqa: BLE001
            failed["CTRL_same"] += 1
            errs["CTRL_same"] = f"{type(exc).__name__}: {exc}"

        # 4) 仓库已有 destyle_latent：对目标风格供体 latent 去风格
        try:
            gen = torch.Generator(device=DEVICE).manual_seed(int(args.seed + k))
            z_ds = destyle_latent(z_don, generator=gen).float()
            sc = metric.score(z_ds, [src_img], proto_t, proto_s, base)
            rows["CTRL_destyle"].append(_row(plan, "CTRL", "CTRL_destyle", sc, 0, z_ds, [], 1.0, seam=0.0))
        except Exception as exc:  # noqa: BLE001
            failed["CTRL_destyle"] += 1
            errs["CTRL_destyle"] = f"{type(exc).__name__}: {exc}"

        if (k + 1) % 20 == 0:
            print(f"  ctrl {k+1}/{len(sources)}  {time.time()-t0:.1f}s", flush=True)

    out = Q.summarise(rows, failed, len(sources))
    out["errors"] = errs
    out["note"] = "controls share the same 300 sources / P5 donor geometry as A_P5"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: {"n": v.get("n"), "failed": v.get("failed"),
                          "csd_dT": (v.get("csd_dT") or {}).get("mean"),
                          "csd_dM": (v.get("csd_dM") or {}).get("mean"),
                          "dino_c": (v.get("dino_c") or {}).get("mean"),
                          "lpips": (v.get("lpips") or {}).get("mean")}
                      for k, v in out["configs"].items()}, ensure_ascii=False, indent=1))
    print(f"[ctrl] done {time.time()-t0:.1f}s -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
