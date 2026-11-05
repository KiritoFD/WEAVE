"""入选方案的完整训练缓存（gt-build §6，13,752 × 2 = 27,504 组/臂）。

- positive：z0 = 源图 latent，z1 = 伪GT（A/B/C/D）
- reverse ：z0 = 被外来 patch 污染的真实目标图，z1 = 真实目标图
- base    ：z0 = 源图，z1 = 真实目标风格图（原 FM 基线，不做任何构造）
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import paths  # noqa: E402
from tools.pseudo_gt import construct as C  # noqa: E402
from tools.pseudo_gt.build_assets import STYLES, STYLE_IDS, _load_rgb, _to_tensor, load_manifest, pgt_root  # noqa: E402
from tools.pseudo_gt.cache import SampleRecord, ShardWriter, make_sample_id  # noqa: E402
from tools.pseudo_gt.matching import DEVICE, assert_cuda, encode_dino, load_dino_model, match_layout  # noqa: E402
from tools.pseudo_gt.masks import E_SPECS, build_patch_layout  # noqa: E402
from utils.inference import download_vae_with_fallback  # noqa: E402


def load_latents() -> Dict[str, dict]:
    out = {}
    for style in STYLES:
        p = pgt_root() / "orig_mean_fp16" / f"{STYLE_IDS[style]:02d}_{style}.pt"
        payload = torch.load(p, map_location="cpu", weights_only=False)
        out[style] = {"latents": payload["latents"], "index": {s: i for i, s in enumerate(payload["ids"])}}
    return out


def load_retrieval() -> dict:
    return json.loads((pgt_root() / "retrieval" / "top8.json").read_text(encoding="utf-8"))["styles"]


class Encoder:
    def __init__(self):
        self.vae = download_vae_with_fallback("stabilityai/sd-vae-ft-ema", device=DEVICE)

    def __call__(self, rgb: torch.Tensor) -> torch.Tensor:
        rgb = rgb.to(DEVICE, torch.float16).clamp(0, 1) * 2 - 1
        rgb = rgb.contiguous(memory_format=torch.channels_last)
        assert_cuda(rgb, "train-cache encode")
        # VAE 权重是普通张量，必须在 inference_mode 下前向；返回值 clone 成普通张量供落盘
        with torch.inference_mode():
            z = self.vae.encode(rgb).latent_dist.mean * self.vae.config.scaling_factor
        return z.clone()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["positive", "reverse", "base"])
    ap.add_argument("--method", default="", help="positive 用：A|B|C|D")
    ap.add_argument("--spec", required=True, help="P2/P5/P7/P9 或 E1–E7")
    ap.add_argument("--versions", type=int, default=2)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    out_root = Path(args.out) if args.out else pgt_root() / "pairs" / args.arm
    writer = ShardWriter(out_root)
    if args.arm == "positive" and args.method not in {"A", "B", "C", "D"}:
        raise SystemExit("positive 需要 --method A|B|C|D")

    t0 = time.time()
    items = [it for it in load_manifest() if it["split"] == "train"]
    id_map = {it["id"]: it for it in items}
    lat, retrieval = load_latents(), load_retrieval()
    dino = {}
    for style in STYLES:
        payload = torch.load(pgt_root() / "dino_224" / f"{style}.pt", map_location="cpu", weights_only=False)
        dino[style] = {
            "patches": payload["patches"],  # fp16 原值；用时转 float 再归一化
            "cls": F.normalize(payload["cls"].float(), dim=-1),
            "ids": list(payload["ids"]),
            "index": {s: i for i, s in enumerate(payload["ids"])},
        }
    # 检索表按 (源风格 → 目标风格) 给出 top-8 候选；不足 2 个时用该风格内
    # 的 CLS 相似度补齐（保证 4 个方向都有候选）。
    ret_index = {s: {sid: i for i, sid in enumerate(retrieval[s]["source_ids"])} for s in retrieval}
    REDUCED_TOPK = 4  # 训练缓存只评估前 4 个候选（Q0 仍用 8）；省 ~一半匹配算力，保守可回滚
    enc = Encoder()
    layouts = {}
    def layout(seed: int):
        # 每图 seed 唯一，缓存只保留最近 8 个，避免 27k 布局常驻内存
        if seed in layouts:
            return layouts[seed]
        if len(layouts) >= 8:
            layouts.pop(next(iter(layouts)))
        layouts[seed] = build_patch_layout(args.spec, seed=seed)
        return layouts[seed]

    feather = args.method == "B" or (args.arm == "reverse" and E_SPECS.get(args.spec, {}).get("feather", True))
    done = writer.completed_ids()
    made, skipped = 0, 0
    # RGB 类构造攒够 8 张再统一 VAE 编码（batch 8 比 batch 1 快 3–4 倍）
    pending: list = []

    def flush() -> None:
        if not pending:
            return
        rgbs, locs = [], []
        for i, p in enumerate(pending):
            if p["rgb0"] is not None:
                rgbs.append(p["rgb0"])
                locs.append((i, 0))
            if p["rgb1"] is not None:
                rgbs.append(p["rgb1"])
                locs.append((i, 1))
        if rgbs:
            for s in range(0, len(rgbs), 8):
                zs = enc(torch.cat(rgbs[s:s + 8], 0))
                for j in range(len(zs)):
                    i, side = locs[s + j]
                    pending[i][f"z{side}"] = zs[j:j + 1]
        for p in pending:
            writer.add(p["z0"][0], p["z1"][0], p["rec"])
        pending.clear()

    for k, src in enumerate(items):
        src_style = src["style"]
        for version in range(args.versions):
            seed = 1000 * version + k
            # 4 个跨风格方向均衡：偶数位用 +1/+2，奇数位用 +3/+4。
            src_idx = STYLES.index(src_style)
            tgt_style = STYLES[(src_idx + 1 + (2 * (k % 2)) + version) % 5]
            cands = list(retrieval[src_style]["by_target"][tgt_style]["donor_ids"][ret_index[src_style][src["id"]]])
            cands = [c for c in cands if id_map.get(c, {}).get("style") == tgt_style][:REDUCED_TOPK]
            if len(cands) < 2:
                cls_s = dino[src_style]["cls"][dino[src_style]["index"][src["id"]]]
                topv, topi = (dino[tgt_style]["cls"] @ cls_s).topk(min(REDUCED_TOPK, len(dino[tgt_style]["ids"])))
                cands = [dino[tgt_style]["ids"][j] for j in topi.tolist()]

            src_pat = F.normalize(dino[src_style]["patches"][dino[src_style]["index"][src["id"]]].to(DEVICE).float(), dim=-1)
            donor_id, m, best_q = None, None, -1.0
            for cand in cands:
                dm = F.normalize(dino[tgt_style]["patches"][dino[tgt_style]["index"][cand]].to(DEVICE).float(), dim=-1)
                mm, q = match_layout(src_pat, dm, layout(seed))
                if mm is not None and q > best_q:
                    donor_id, m, best_q = cand, mm, q
            if donor_id is None:
                skipped += 1
                continue

            sid = make_sample_id(args.method if args.arm == "positive" else args.arm,
                                 args.spec, src["id"], donor_id, seed)
            if sid in done:
                skipped += 1
                continue
            z_src = lat[src_style]["latents"][lat[src_style]["index"][src["id"]]][None].float().to(DEVICE)
            z_don = lat[tgt_style]["latents"][lat[tgt_style]["index"][donor_id]][None].float().to(DEVICE)
            rgb0 = rgb1 = None
            if args.arm == "reverse":
                # 反向：循环项 = 真实目标 y；候选 = 异风格污染源 x。
                # z1 是真实 y（目标风格 = 循环项风格），z0 是贴入 x patch 后的 y。
                rgb0 = C.construct_E(_to_tensor(_load_rgb(src["path"]))[None],
                                     _to_tensor(_load_rgb(id_map[donor_id]["path"]))[None], m, feather=feather)
                zb = z_src
                target_style = src_style
            elif args.arm == "base":
                za, zb = z_src, z_don
                target_style = tgt_style
            else:
                za = z_src
                target_style = tgt_style
                if args.method == "A":
                    rgb1 = C.construct_A(_to_tensor(_load_rgb(src["path"]))[None],
                                         _to_tensor(_load_rgb(id_map[donor_id]["path"]))[None], m)
                elif args.method == "B":
                    rgb1 = C.construct_B(_to_tensor(_load_rgb(src["path"]))[None],
                                         _to_tensor(_load_rgb(id_map[donor_id]["path"]))[None], m)
                elif args.method == "C":
                    zb = C.construct_C(z_src, z_don, m)
                else:
                    zb = C.construct_D(z_src, z_don, m)
            rects = layout(seed)
            cov = float(sum(r.area for r in rects) / 512 ** 2)
            rec = SampleRecord(
                sample_id=sid,
                source_id=src["id"],
                donor_id=donor_id,
                source_style=src_style,
                target_style=target_style,
                method=args.method if args.arm == "positive" else args.arm,
                patch_spec=args.spec,
                seed=seed,
                match_score=float(best_q),
                nominal_coverage=cov,
                effective_coverage=cov,
                z0_is_original=(args.arm != "reverse"),
                z1_is_original=(args.arm in {"reverse", "base"}),
            )
            pending.append({"rgb0": rgb0, "rgb1": rgb1,
                            "z0": None if rgb0 is not None else za,
                            "z1": None if rgb1 is not None else zb,
                            "rec": rec})
            made += 1
            if len(pending) >= 8:
                flush()
        if (k + 1) % 200 == 0:
            print(f"  {args.arm} {k+1}/{len(items)}  made={made} skipped={skipped}  {time.time()-t0:.0f}s", flush=True)
    flush()
    writer.close()
    print(f"[done] {args.arm}: {made} 组, skipped {skipped}  {time.time()-t0:.0f}s -> {out_root}", flush=True)


if __name__ == "__main__":
    main()
