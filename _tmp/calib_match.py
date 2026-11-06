"""匹配阈值标定：统计 all-pass / ≥75% / ≥50% 规则在不同 min_q 下的通率。"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch
import torch.nn.functional as F

import paths
from tools.pseudo_gt.build_assets import STYLES, load_manifest, pgt_root
from tools.pseudo_gt.masks import E_SPECS, PATCH_SPECS, build_patch_layout
from tools.pseudo_gt.matching import DEVICE, match_rect


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用")
    items = load_manifest()
    dino = {}
    for style in STYLES:
        p = torch.load(pgt_root() / "dino_224" / f"{style}.pt", map_location="cpu", weights_only=False)
        dino[style] = {"ids": p["ids"], "index": {s: i for i, s in enumerate(p["ids"])},
                       "patches": p["patches"]}
        del p
    ret = json.loads((pgt_root() / "retrieval" / "top8.json").read_text(encoding="utf-8"))["styles"]

    specs = list(PATCH_SPECS)
    layouts = {n: build_patch_layout(n, seed=2024) for n in specs}
    styles_per = 3
    import random
    rng = random.Random(2024)
    sources = []
    for style in STYLES:
        pool = [it for it in items if it["style"] == style and it["split"] == "train"]
        pool.sort(key=lambda it: it["id"])
        sources.extend(rng.sample(pool, min(styles_per, len(pool))) if pool else [])
    print(f"[calib] sources {len(sources)}", flush=True)

    rows = []
    for it in sources:
        tgt = STYLES[(STYLES.index(it["style"]) + 1) % 5]
        cands = [c for c in ret[it["style"]]["by_target"][tgt]["donor_ids"][ret[it["style"]]["source_ids"].index(it["id"])]
                 if c in dino[tgt]["index"]][:8]
        if len(cands) < 2:
            continue
        sp = F.normalize(dino[it["style"]]["patches"][dino[it["style"]]["index"][it["id"]]].float(), dim=-1).to(DEVICE)
        for d in cands:
            dp = F.normalize(dino[tgt]["patches"][dino[tgt]["index"][d]].float(), dim=-1).to(DEVICE)
            for name, rects in layouts.items():
                qs, rs = [], []
                for r in rects:
                    m = match_rect(sp, dp, r, min_q=-2.0)
                    if m is None:
                        qs.append(-2.0)
                        rs.append(0.0)
                    else:
                        qs.append(m.q)
                        rs.append(m.mutual_ratio)
                rows.append((it["id"], d, name, qs, rs))
        print(f"[calib] {it['id'][:30]} done", flush=True)

    rules = []
    for qth in [-0.5, 0.25, 0.30, 0.35, 0.40, 0.45]:
        for frac in [1.0, 0.75, 0.5]:
            ok = 0
            for _, _, _, qs, rs in rows:
                good = sum(1 for q, r in zip(qs, rs) if q >= qth and r >= 0.15)
                if good / len(qs) >= frac:
                    ok += 1
            rules.append({"min_q": qth, "frac": frac, "pass": ok, "n": len(rows)})
    print(json.dumps({"rows": len(rows), "rules": rules}, ensure_ascii=False, indent=1))

    # 以 (source, donor) 为单位：16 个 layout（P1–P9 + E1–E7）全部通过
    specs = sorted(layouts)
    pair_rules = []
    for qth in [0.25, 0.30, 0.35, 0.40]:
        for frac in [1.0, 0.75, 0.5]:
            groups = {}
            for sid, d, name, qs, rs in rows:
                groups.setdefault((sid, d), {})[name] = (qs, rs)
            ok_pairs, pairs = 0, 0
            for key, per in groups.items():
                if len(per) != len(specs):
                    continue
                pairs += 1
                if all(
                    sum(1 for q, r in zip(qs, rs) if q >= qth and r >= 0.15) / len(qs) >= frac
                    for qs, rs in per.values()
                ):
                    ok_pairs += 1
            pair_rules.append({"min_q": qth, "frac": frac, "pairs_all_specs_pass": ok_pairs, "pairs": pairs})
    print("pair_rules:", json.dumps(pair_rules, ensure_ascii=False))

    # 每 spec 单独看（frac=0.75, q=0.35）
    by_spec = {}
    for _, _, name, qs, rs in rows:
        good = sum(1 for q, r in zip(qs, rs) if q >= 0.35 and r >= 0.15)
        by_spec.setdefault(name, [0, 0])
        by_spec[name][1] += 1
        if good / len(qs) >= 0.75:
            by_spec[name][0] += 1
    print("by_spec(frac=.75,q=.35):", json.dumps(by_spec, ensure_ascii=False))


if __name__ == "__main__":
    main()
