"""CSD 原型作者均衡复核。

1) 复核现有原型（每风格 500 张随机抽样）里的作者分布，重点看 Ukiyo-e 单一作者占比。
2) 构造“作者均衡”原型（每作者至多 ceil(500/n_authors) 张），与现有原型比较余弦相似度。
3) 用同一批源图重建 R64 / A_P5，在两套原型下分别算 ΔT/ΔS，检查结论是否被作者分布改变。
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools" / "12h"))

import csd_model  # noqa: E402
import paths  # noqa: E402
from tools.pseudo_gt import construct as C  # noqa: E402
from tools.pseudo_gt import pattern_masks as PM  # noqa: E402
from tools.pseudo_gt import q0_quality as Q  # noqa: E402
from tools.pseudo_gt import supp_eval as SE  # noqa: E402
from tools.pseudo_gt.build_assets import STYLES, _load_rgb, _to_tensor, load_manifest, pgt_root  # noqa: E402
from tools.pseudo_gt.masks import build_patch_layout  # noqa: E402

NSAMP = 500


def author_of(stem: str) -> str:
    parts = stem.split("__")
    return parts[1] if len(parts) >= 3 else stem


def load_payload(style: str) -> dict:
    si = STYLES.index(style)
    return torch.load(pgt_root() / "orig_mean_fp16" / f"{si:02d}_{style}.pt",
                      map_location="cpu", weights_only=False)


def sample_ids(payload: dict, exclude: set[str], seed: int) -> list[str]:
    ids = [i for i in payload["ids"] if i not in exclude]
    return random.Random(seed).sample(ids, min(NSAMP, len(ids)))


def balanced_ids(payload: dict, exclude: set[str], seed: int, stems: dict) -> list[str]:
    ids = [i for i in payload["ids"] if i not in exclude]
    by_author: dict[str, list[str]] = defaultdict(list)
    for i in ids:
        by_author[author_of(stems.get(i, i))].append(i)
    quota = max(1, math.ceil(NSAMP / max(len(by_author), 1)))
    rng = random.Random(seed)
    out: list[str] = []
    for a in sorted(by_author):
        pool = sorted(by_author[a])
        rng.shuffle(pool)
        out.extend(pool[:quota])
    rng.shuffle(out)
    return out[:NSAMP]


def featurize(metric, payload: dict, ids: list[str]) -> np.ndarray:
    idx = [payload["ids"].index(i) for i in ids]
    feats = []
    for s in range(0, len(idx), 16):
        z = payload["latents"][idx[s:s + 16]].float()
        pils = metric.decode(z)
        feats.append(csd_model.feats_pils(metric.csd, pils, batch=len(pils)))
    return np.concatenate(feats, 0).astype(np.float32).mean(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(paths.project_root()) / "runs/p2/proto_author_check.json"))
    ap.add_argument("--n-sources", type=int, default=40)
    args = ap.parse_args()

    metric = Q.Metrics()
    items = load_manifest()
    stems = {it["id"]: it["stem"] for it in items}
    test_stems = {p.stem for p in (REPO_ROOT / "data/test").rglob("*.jpg")}
    test_stems |= {p.stem for p in (REPO_ROOT / "datasets/distinct5_512/test").rglob("*.jpg")}
    sources = Q.shared_sources(items, test_stems)[: args.n_sources]
    src_ids = {it["id"] for it in sources}

    orig = torch.load(pgt_root() / "csd_proto.pt", map_location="cpu", weights_only=False)
    payloads = {s: load_payload(s) for s in STYLES}

    rows, bal = {}, {}
    for s in STYLES:
        o_ids = sample_ids(payloads[s], src_ids, Q.SEED + 1)
        b_ids = balanced_ids(payloads[s], src_ids, Q.SEED + 1, stems)
        oc, bc = Counter(author_of(stems[i]) for i in o_ids), Counter(author_of(stems[i]) for i in b_ids)
        rows[s] = {
            "n_orig": len(o_ids), "n_bal": len(b_ids),
            "orig_authors": len(oc), "bal_authors": len(bc),
            "orig_top": oc.most_common(5), "bal_top": bc.most_common(5),
            "orig_top_frac": oc.most_common(1)[0][1] / max(len(o_ids), 1),
            "bal_top_frac": bc.most_common(1)[0][1] / max(len(b_ids), 1),
        }
        f = featurize(metric, payloads[s], b_ids)
        bal[s] = f / np.linalg.norm(f)
        cos = float(np.dot(orig[s], bal[s]) / (np.linalg.norm(orig[s]) * np.linalg.norm(bal[s]) + 1e-9))
        rows[s]["cos_orig_balanced"] = cos
        print(f"[author] {s}: n={len(o_ids)} authors={len(oc)} top={oc.most_common(2)} cos={cos:.4f}",
              flush=True)

    ids_proto = list(STYLES)
    p_orig = torch.tensor(np.stack([orig[s] for s in ids_proto]))
    p_bal = torch.tensor(np.stack([bal[s] for s in ids_proto]))

    lat, dino = Q.load_latents(), Q.load_dino()
    ret = Q.load_retrieval()
    id_map = {it["id"]: it for it in items}
    impacts = {"R64": [], "A_P5": []}
    for it in sources:
        s_style, t_style = it["style"], Q.target_style_for(it["style"])
        sp = torch.nn.functional.normalize(
            dino[s_style]["patches"][dino[s_style]["index"][it["id"]].__int__()].to("cuda").float(), dim=-1)
        cands = ret[s_style]["by_target"][t_style]["donor_ids"][ret[s_style]["source_ids"].index(it["id"])]
        donor_id = SE.choose_donor(sp, dino, s_style, t_style, it["id"], cands, Q.SEED)
        if donor_id is None:
            continue
        dp = torch.nn.functional.normalize(
            dino[t_style]["patches"][dino[t_style]["index"][donor_id].__int__()].to("cuda").float(), dim=-1)
        qm, mm = SE.cell_qmap(sp, dp, 64)
        src_img, d_img = _load_rgb(it["path"]), _load_rgb(id_map[donor_id]["path"])
        base = csd_model.feats_pils(metric.csd, [src_img], batch=1)[0]
        for cfg in ("R64", "A_P5"):
            rects = PM.rects_from_cells(PM.select_cells("R64", qmap=qm, mmap=mm, rng=random.Random(7)), 64) \
                if cfg == "R64" else build_patch_layout("P5", seed=Q.SEED)
            if not rects:
                continue
            matches = []
            for _r in rects:
                m = SE.match_rect(sp, dp, _r, min_q=-2.0)
                if m is None:
                    matches = []
                    break
                matches.append(m)
            if len(matches) != len(rects):
                continue
            h = C.construct_A(_to_tensor(src_img)[None], _to_tensor(d_img)[None], matches)
            feat = csd_model.feats_pils(metric.csd, metric.decode(metric.encode(h)), batch=1)[0]
            for tag, proto in (("orig", p_orig), ("bal", p_bal)):
                sc = (torch.tensor(feat) @ proto.T).numpy()
                impacts[cfg].append({
                    "tag": tag, "source_id": it["id"], "source_style": s_style,
                    "target_style": t_style,
                    "dT": float(sc[ids_proto.index(t_style)] - base[ids_proto.index(t_style)]),
                    "dS": float(sc[ids_proto.index(s_style)] - base[ids_proto.index(s_style)])})

    agg = {}
    for cfg, lst in impacts.items():
        for tag in ("orig", "bal"):
            vals = [r for r in lst if r["tag"] == tag]
            agg[f"{cfg}_{tag}"] = {"n": len(vals),
                                   "dT": float(np.mean([r["dT"] for r in vals])) if vals else None,
                                   "dS": float(np.mean([r["dS"] for r in vals])) if vals else None}
        pair = {r["source_id"]: r for r in lst if r["tag"] == "orig"}
        pd = [r["dT"] - pair[r["source_id"]]["dT"] for r in lst if r["tag"] == "bal" and r["source_id"] in pair]
        if pd:
            agg[f"{cfg}_delta_bal_minus_orig"] = {
                "n": len(pd), "mean": float(np.mean(pd)),
                "ci95": [float(np.percentile(pd, 2.5)), float(np.percentile(pd, 97.5))]}

    out = {"prototypes": rows, "impacts": agg}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[author] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
