"""V26 K2-C 本地 CPU 完成版: 题材 (CLIP text 零样本) × 目标画派 Top-1 关联表.

不依赖 GPU/远程: 从 runs/v25/measured/C2nt_sw6_lc0.8_t45/features.pt 读取逐任务
CSD (750,768) 与 CLIP (750,512) 特征, 用 CSD 原型重算 target_top1, 用 CLIP text
编码器对 5 类题材提示打源图题材标签, 关联输出题材偏置表.

用法 (本地):  python -m exp.v26.k2c_local_join
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "runs/v25/measured/C2nt_sw6_lc0.8_t45/features.pt"
OUT_JSON = ROOT / "docs/v26/4090/data/k2c_local_join.json"
STYLES = ("Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e")
SUBJECT_PROMPTS = {
    "person": "a painting featuring one or more people, portraits, figures",
    "landscape": "a landscape painting of scenery, mountains, rivers, trees",
    "architecture": "a painting of buildings, streets, temples, interior rooms",
    "still_life": "a still life painting of flowers, food, or objects on a table",
    "abstract": "an abstract painting of flat planes, lines, or color fields",
}


def main() -> None:
    d = torch.load(FEATURES, map_location="cpu", weights_only=False)
    task_ids, csd, clip = d["task_ids"], d["csd"], d["clip"]
    n = len(task_ids)

    # 目标画派 / 跨画派标记 (从 task_id 解析, 与 tasks_750 同序)
    tgt = [t.split("/")[-1] for t in task_ids]
    src = [t.split("/")[0] for t in task_ids]
    is_cross = [s != t for s, t in zip(src, tgt)]

    # CSD 原型 (与 FeatureStack.csd_prototypes 同源: asset pseudo_gt_24h/csd_proto.pt)
    import numpy as np
    import paths as _paths
    proto_p = _paths.asset_root() / "pseudo_gt_24h" / "csd_proto.pt"
    proto_d = torch.load(proto_p, map_location="cpu", weights_only=False)
    proto = torch.as_tensor(np.stack([proto_d[s] for s in STYLES])).float()
    proto = F.normalize(proto, dim=-1)
    csd_n = F.normalize(csd.float(), dim=-1)
    scores5 = csd_n @ proto.T                                # (750, 5)
    tgt_idx = torch.tensor([STYLES.index(t) for t in tgt])
    top1 = (scores5.argmax(dim=1) == tgt_idx).tolist()

    # CLIP text 零样本题材标签 (本地缓存权重)
    from transformers import CLIPModel, CLIPProcessor
    import paths
    cache = str(paths.eval_cache_root("hf") / "hub")
    clip_m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True,
                                       cache_dir=cache).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True,
                                         cache_dir=cache)
    names = list(SUBJECT_PROMPTS)
    ti = proc(text=[SUBJECT_PROMPTS[k] for k in names], return_tensors="pt", padding=True)
    with torch.no_grad():
        tout = clip_m.get_text_features(**ti)
        tfeat = F.normalize(tout if torch.is_tensor(tout) else tout.pooler_output, dim=-1)
    c_n = F.normalize(clip.float(), dim=-1)
    subj_idx = (c_n @ tfeat.T).argmax(dim=1).tolist()
    subj_counts = {k: subj_idx.count(i) for i, k in enumerate(names)}

    # 题材 × 目标画派 Top-1% 表 (仅 cross600)
    table = {}
    for s_i, sname in enumerate(names):
        for tname in STYLES:
            hit = [t1 for s2, t2, t1, ic in zip(subj_idx, tgt, top1, is_cross)
                   if s2 == s_i and t2 == tname and ic]
            table[f"{sname}|{tname}"] = (round(100.0 * sum(hit) / len(hit), 1) if hit else None,
                                         len(hit))

    out = {
        "task": "V26-K2C local join (subject × target-style CSD Top-1)",
        "features": str(FEATURES).replace("\\", "/").split("runs/")[-1],
        "n_tasks": n, "subject_counts": subj_counts,
        "subject_by_style_top1": {k: v[0] for k, v in table.items()},
        "subject_by_style_n": {k: v[1] for k, v in table.items()},
    }
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[K2C-local] subject_counts = {subj_counts}")
    for sname in names:
        row = [f"{table[f'{sname}|{t}'][0]}%({table[f'{sname}|{t}'][1]})" for t in STYLES]
        print(f"  {sname:14s}: " + " | ".join(row))
    print(f"[K2C-local] saved → {OUT_JSON}")


if __name__ == "__main__":
    main()
