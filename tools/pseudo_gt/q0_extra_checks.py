"""补齐 gt-build §5/§3.2 的两个缺口：
1) contact sheet：同一源图下 A_P5 / B_P5 / D_P5 各解码 1 张，拼成人工检查图。
2) E 双方向验证：真实目标图是否比伪输入更接近目标风格（CSD_T(real) vs CSD_T(pseudo)）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import paths  # noqa: E402
from tools.pseudo_gt import construct as C  # noqa: E402
from tools.pseudo_gt import q0_quality as Q  # noqa: E402
from tools.pseudo_gt.masks import build_patch_layout  # noqa: E402
import csd_model  # type: ignore  # noqa: E402

SPEC = "P5"
SHEET_CFG = [("src", "A", None), ("A_P5", "A", SPEC), ("B_P5", "B", SPEC), ("D_P5", "D", SPEC)]
N_SHEET_SRC = 3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="E 方向验证的源图上限（0=全部 222）")
    ap.add_argument("--sheet-src", type=int, default=N_SHEET_SRC)
    args = ap.parse_args()
    t0 = time.time()

    items = Q.load_manifest()
    test_stems = {p.stem for p in (REPO_ROOT / "data/test").rglob("*.jpg")}
    test_stems |= {p.stem for p in (REPO_ROOT / "datasets/distinct5_512/test").rglob("*.jpg")}
    sources = Q.shared_sources(items, test_stems)
    lat, dino, retrieval = Q.load_latents(), Q.load_dino(), Q.load_retrieval()
    metric = Q.Metrics()
    protos = Q.build_prototypes(metric, lat, {it["id"] for it in sources})
    rects = build_patch_layout(SPEC, seed=Q.SEED)
    id_map = {it["id"]: it for it in items}

    tiles, labels = [], []
    e_rows = []
    n_done = 0
    for k, it in enumerate(sources):
        tgt = Q.target_style_for(it["style"])
        ret = retrieval[it["style"]]
        cands = ret["by_target"][tgt]["donor_ids"][ret["source_ids"].index(it["id"])]
        cands = [c for c in cands if dino[tgt]["index"].get(c) is not None][:8]
        donor_id, matches, qs = Q.pick_donor(it["style"], it["id"], tgt, cands, dino, {SPEC: rects})
        if donor_id is None:
            continue
        m_p5 = matches[SPEC]
        src_img = Q._load_rgb(it["path"])
        src_rgb = Q._to_tensor(src_img)[None]
        donor_img = Q._load_rgb(id_map[donor_id]["path"])
        donor_rgb = Q._to_tensor(donor_img)[None]
        proto_t = protos[tgt]

        # E 双方向：CSD_T(real target) vs CSD_T(pseudo)
        pseudo_rgb = C.construct_A(src_rgb, donor_rgb, m_p5)
        z_ps = metric.encode(pseudo_rgb)
        gen_img = metric.decode(z_ps)[0]
        f_ps = csd_model.feats_pils(metric.csd, [gen_img], batch=1)[0]
        f_dn = csd_model.feats_pils(metric.csd, [donor_img], batch=1)[0]
        td = float(f_dn @ proto_t) - float(f_ps @ proto_t)
        e_rows.append({"source_id": it["id"], "donor_id": donor_id, "delta_T_real_minus_pseudo": td})
        n_done += 1

        if k < args.sheet_src:
            z_src = lat[it["style"]]["latents"][lat[it["style"]]["index"][it["id"]]][None].float().to(Q.DEVICE)
            z_don = lat[tgt]["latents"][lat[tgt]["index"][donor_id]][None].float().to(Q.DEVICE)
            imgs = {
                "src": src_img,
                "A_P5": metric.decode(metric.encode(C.construct_A(src_rgb, donor_rgb, m_p5)))[0],
                "B_P5": metric.decode(metric.encode(C.construct_B(src_rgb, donor_rgb, m_p5)))[0],
                "D_P5": metric.decode(C.construct_D(z_src, z_don, m_p5))[0],
            }
            for name, img in imgs.items():
                tiles.append(img.resize((256, 256)))
                labels.append(f"{it['style']} | {name}")

        if (k + 1) % 25 == 0:
            print(f"  extra {k+1}/{len(sources)} {time.time()-t0:.1f}s", flush=True)

    # contact sheet
    cols = len(SHEET_CFG)
    rows_n = min(args.sheet_src, max(1, len(tiles) // cols))
    sheet = Image.new("RGB", (256 * cols, 256 * rows_n + 18 * rows_n), (255, 255, 255))
    dr = ImageDraw.Draw(sheet)
    for i, (img, lab) in enumerate(zip(tiles, labels)):
        r, c = divmod(i, cols)
        if r >= rows_n:
            break
        sheet.paste(img, (256 * c, (256 + 18) * r + 18))
        dr.text((256 * c + 4, (256 + 18) * r + 4), lab, fill=(0, 0, 0))
    sheet_p = Path(paths.project_root()) / "runs/p2/pgt_contact_sheet.png"
    sheet.save(sheet_p)

    deltas = np.array([r["delta_T_real_minus_pseudo"] for r in e_rows], dtype=np.float64)
    out = {
        "n": int(deltas.size),
        "mean_delta_T_real_minus_pseudo": float(deltas.mean()),
        "frac_real_closer_to_target": float((deltas > 0).mean()),
        "median": float(np.median(deltas)),
        "note": "CSD_T(real target image) - CSD_T(A_P5 pseudo), same 300 shared sources; E7 construction equals A_P5",
        "rows": e_rows,
        "contact_sheet": str(sheet_p),
        "wall_sec": round(time.time() - t0, 1),
    }
    outp = Path(paths.project_root()) / "runs/p2/pgt_e_direction.json"
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ("rows",)}, ensure_ascii=False), flush=True)
    print(f"[extra] done {time.time()-t0:.1f}s n={n_done}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
