"""Pattern 版训练缓存（P4 T1–T4）：同一遍构造同时写 positive / reverse 两臂。

- 位置只从已通过内容匹配（q≥0.30 且互近邻≥0.15）的 64px 格子里选；
  凑不齐的图直接跳过（不为凑形状接受低置信度 patch）。
- 每张图只做一次 VAE 编码：positive 用它作 z1，reverse 用它作 z0；
  源图 latent 复用已有缓存，不重复编码。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import paths  # noqa: E402
from tools.pseudo_gt import construct as C  # noqa: E402
from tools.pseudo_gt import pattern_masks as PM  # noqa: E402
from tools.pseudo_gt.build_assets import STYLES, _load_rgb, _to_tensor, load_manifest, pgt_root  # noqa: E402
from tools.pseudo_gt.build_train_cache import Encoder, load_latents, load_retrieval  # noqa: E402
from tools.pseudo_gt.cache import SampleRecord, ShardWriter, make_sample_id  # noqa: E402
from tools.pseudo_gt.matching import match_rect  # noqa: E402
from tools.pseudo_gt.masks import Rect  # noqa: E402
from tools.pseudo_gt.supp_eval import texture_saliency  # noqa: E402

REDUCED_TOPK = 4


def cell_qmap(sp, dp, size=64):
    n = PM.cells_of(size)
    qm = np.full((n, n), -2.0)
    mm = np.zeros((n, n))
    for cy in range(n):
        for cx in range(n):
            m = match_rect(sp, dp, Rect(cx * size, cy * size, size, size), min_q=-2.0)
            if m is not None:
                qm[cy, cx] = m.q
                mm[cy, cx] = m.mutual_ratio
    return qm, mm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patterns", required=True, help="逗号分隔，如 R64,F64")
    ap.add_argument("--versions", type=int, default=2)
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--out-root", default="", help="默认 asset/pseudo_gt_24h/pairs")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    pats = [p.strip() for p in args.patterns.split(",") if p.strip()]
    for p in pats:
        if p not in PM.PATTERN_SPECS:
            raise SystemExit(f"未知 pattern {p}")
    root = Path(args.out_root) if args.out_root else pgt_root() / "pairs"
    arms = {}
    for p in pats:
        for arm in ("positive", "reverse"):
            arms[(p, arm)] = ShardWriter(root / f"pattern_{p}_{arm}")

    t0 = time.time()
    all_items = load_manifest()
    items = [it for it in all_items if it["split"] == args.split]
    if args.limit:
        items = items[: args.limit]
    id_map = {it["id"]: it for it in all_items}
    lat, retrieval = load_latents(), load_retrieval()
    dino = {}
    for style in STYLES:
        payload = torch.load(pgt_root() / "dino_224" / f"{style}.pt", map_location="cpu", weights_only=False)
        dino[style] = {"patches": payload["patches"], "cls": F.normalize(payload["cls"].float(), dim=-1),
                       "ids": list(payload["ids"]), "index": {s: i for i, s in enumerate(payload["ids"])}}
    ret_index = {s: {sid: i for i, sid in enumerate(retrieval[s]["source_ids"])} for s in retrieval}
    enc = Encoder()
    done = {k: w.completed_ids() for k, w in arms.items()}
    masks_path = root / "pattern_mask_metrics.json"
    masks = json.loads(masks_path.read_text(encoding="utf-8")) if masks_path.exists() else {}
    mask_f = open(root / "pattern_mask_metrics.jsonl", "a", encoding="utf-8")

    made = {k: 0 for k in arms}
    skipped = 0
    pending: List[dict] = []

    def flush():
        if not pending:
            return
        rgbs = [p["rgb"] for p in pending]
        zs = enc(torch.cat(rgbs, 0))
        for i, p in enumerate(pending):
            zh = zs[i : i + 1]
            src = p["z_src"]
            for (pat, arm), w in arms.items():
                if pat != p["pattern"]:
                    continue
                sid_p = make_sample_id("positive", pat, p["src"]["id"], p["donor"], p["seed"])
                sid_r = make_sample_id("reverse", pat, p["src"]["id"], p["donor"], p["seed"])
                if (arm == "positive" and sid_p in done[(pat, arm)]) or (arm == "reverse" and sid_r in done[(pat, arm)]):
                    continue
                sid = sid_p if arm == "positive" else sid_r
                rec = SampleRecord(
                    sample_id=sid,
                    source_id=p["src"]["id"],
                    donor_id=p["donor"],
                    source_style=p["src"]["style"],
                    target_style=p["src"]["style"] if arm == "reverse" else p["tgt"],
                    method=arm,
                    patch_spec=pat,
                    seed=p["seed"],
                    match_score=p["q_mean"],
                    nominal_coverage=p["coverage"],
                    effective_coverage=p["coverage"],
                    z0_is_original=(arm != "reverse"),
                    z1_is_original=(arm == "reverse"),
                    band_energy=C.band_energy_share(src, zh),
                )
                z0, z1 = (src, zh) if arm == "positive" else (zh, src)
                w.add(z0[0], z1[0], rec)
                made[(pat, arm)] += 1
                masks[sid] = p["mask"]
                mask_f.write(json.dumps({"sample_id": sid, "mask": p["mask"]}, ensure_ascii=False) + "\n")
        mask_f.flush()
        pending.clear()

    for k, src in enumerate(items):
        src_style = src["style"]
        sp = F.normalize(dino[src_style]["patches"][dino[src_style]["index"][src["id"]]].to("cuda").float(), dim=-1)
        for version in range(args.versions):
            seed = 1000 * version + k
            src_idx = STYLES.index(src_style)
            tgt_style = STYLES[(src_idx + 1 + (2 * (k % 2)) + version) % 5]
            cands = []
            if src["id"] in ret_index[src_style]:
                cands = list(retrieval[src_style]["by_target"][tgt_style]["donor_ids"][ret_index[src_style][src["id"]]])
                cands = [c for c in cands if id_map.get(c, {}).get("style") == tgt_style][:REDUCED_TOPK]
            if len(cands) < 2:
                cls_s = dino[src_style]["cls"][dino[src_style]["index"][src["id"]]]
                topv, topi = (dino[tgt_style]["cls"] @ cls_s).topk(min(REDUCED_TOPK, len(dino[tgt_style]["ids"])))
                cands = [dino[tgt_style]["ids"][j] for j in topi.tolist()]
            best = None
            for c in cands:
                di = dino[tgt_style]["index"].get(c)
                if di is None:
                    continue
                dpc = F.normalize(dino[tgt_style]["patches"][di].to("cuda").float(), dim=-1)
                qm, mm = cell_qmap(sp, dpc)
                qual = PM._qualified(qm, mm)
                if int(qual.sum()) < 16:
                    continue
                score = float(np.mean(np.sort(qm.reshape(-1))[-16:]))
                if best is None or score > best[0]:
                    best = (score, c, qm, mm)
            if best is None:
                skipped += 1
                continue
            _, donor_id, qm, mm = best
            dp = F.normalize(dino[tgt_style]["patches"][dino[tgt_style]["index"][donor_id]].to("cuda").float(), dim=-1)
            donor_rgb = _to_tensor(_load_rgb(id_map[donor_id]["path"]))[None]
            src_img = _load_rgb(src["path"])
            rgb0 = _to_tensor(src_img)[None]
            _, sal8 = texture_saliency(src_img)
            z_src = lat[src_style]["latents"][lat[src_style]["index"][src["id"]]][None].float().to("cuda")
            for pat in pats:
                cells = PM.select_cells(
                    pat, qmap=qm, mmap=mm,
                    rng=random.Random(seed * 7919 + sorted(PM.PATTERN_SPECS).index(pat)))
                if cells is None:
                    skipped += 1
                    continue
                rects = PM.rects_from_cells(cells, 64)
                matches = []
                for r in rects:
                    m = match_rect(sp, dp, r, min_q=-2.0)
                    if m is not None:
                        matches.append(m)
                if len(matches) < len(rects):
                    skipped += 1
                    continue
                h = C.construct_A(rgb0, donor_rgb, matches)
                pending.append({
                    "pattern": pat, "src": src, "donor": donor_id, "seed": seed, "tgt": tgt_style,
                    "rgb": h, "z_src": z_src,
                    "q_mean": float(np.mean([m.q for m in matches])),
                    "coverage": float(len(cells) * 64 * 64 / 512 ** 2),
                    "mask": PM.mask_metrics(cells, 64, qmap=qm, saliency=sal8),
                })
            if len(pending) >= 4:
                flush()
        if (k + 1) % 200 == 0:
            print(f"  {k+1}/{len(items)} made={sum(made.values())} skipped={skipped} {time.time()-t0:.0f}s", flush=True)

    flush()
    for w in arms.values():
        w.close()
    mask_f.close()
    masks_path.write_text(json.dumps(masks, ensure_ascii=False), encoding="utf-8")
    print(f"[done] {made} skipped={skipped} {time.time()-t0:.0f}s -> {root}", flush=True)


if __name__ == "__main__":
    main()
