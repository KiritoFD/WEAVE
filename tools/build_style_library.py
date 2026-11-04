"""P0：用 512 训练原图建五类风格特征库，并报告类内/类间距离。

用法::

    python3 tools/build_style_library.py \
        --data-root data/train --out runs/v2/swd/library_5styles_512.pt \
        --image-size 512 --batch 8 --max-per-style 1000

产物 schema ``swd-style-library-v1``：
  styles / files / local（每风格 [N,P,14] 三尺度）/ global（[N,86]）
  local_mean/std、global_mean/std（白化参数）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v2.features import (  # noqa: E402
    LOCAL_DIM,
    PATCHES_PER_LEVEL,
    FeatureStats,
    encode_style,
    fit_whitener,
    global_descriptor,
    local_descriptors,
)


def _list_images(root: Path, styles: list[str]) -> list[list[Path]]:
    out = []
    for name in styles:
        files = sorted(
            p for p in (root / name).rglob("*")
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        )
        out.append(files)
    return out


def _load_batch(paths: list[Path], size: int) -> torch.Tensor:
    import numpy as np

    ims = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize((size, size), Image.BICUBIC)
        ims.append(torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 255.0)
    return torch.stack(ims)


def _packed_rows(bin_path: Path, index_path: Path, styles: list[str], size: int) -> list[list[int]]:
    import numpy as np

    meta = json.loads(index_path.read_text(encoding="utf-8"))
    n, h, w, c = int(meta["n"]), int(meta["height"]), int(meta["width"]), int(meta["channels"])
    if (h, w) != (size, size):
        raise ValueError(f"packed images are {h}x{w}, requested image-size={size}; no silent resize")
    mm = np.memmap(bin_path, dtype=np.uint8, mode="r", shape=(n, h, w, c))
    records = meta["records"]
    out: list[list[int]] = []
    for name in styles:
        rows = [i for i, r in enumerate(records) if r["style"] == name]
        if not rows:
            raise ValueError(f"style {name!r} not found in {index_path}")
        out.append(rows)
    _MOUNT["memmap"] = mm  # 避免 memmap 被 GC
    return out


_MOUNT: dict[str, object] = {}


def _load_packed(rows: list[int]) -> torch.Tensor:
    import numpy as np

    mm = _MOUNT["memmap"]
    arr = np.asarray(mm[rows])  # [B,H,W,3] uint8
    return torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0


def _subsample(desc: list[torch.Tensor], gen: torch.Generator) -> list[torch.Tensor]:
    """每个尺度只留固定数量的 patch，控制缓存体积。"""
    out = []
    for d in desc:
        b, c, h, w = d.shape
        total = h * w
        pick = torch.randperm(total, generator=gen)[:PATCHES_PER_LEVEL].to(d.device)
        idx = (torch.arange(b, device=d.device)[:, None] * total + pick[None, :]).reshape(-1)
        flat = d.permute(0, 2, 3, 1).reshape(b * total, c)
        out.append(flat.index_select(0, idx).reshape(b, PATCHES_PER_LEVEL, c))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/train")
    ap.add_argument("--out", default="runs/v2/swd/library_5styles_512.pt")
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-per-style", type=int, default=1000)
    ap.add_argument("--styles", default="Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e")
    ap.add_argument("--packed-bin", default="data/train_d5/images_hwc_uint8.bin")
    ap.add_argument("--packed-index", default="data/train_d5/images_index.json")
    ap.add_argument("--use-packed", action="store_true",
                    help="从 NHWC uint8 packed bin 读取（与 512 训练原图完全同源）")
    ap.add_argument("--encoder-ckpt", default="",
                    help="冻结的轻量 Style Encoder；给出则整图描述子追加其 128-d 嵌入")
    args = ap.parse_args()

    root = Path(args.data_root)
    styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    if args.use_packed:
        bin_path, idx_path = Path(args.packed_bin), Path(args.packed_index)
        records = json.loads(idx_path.read_text(encoding="utf-8"))["records"]
        items = _packed_rows(bin_path, idx_path, styles, args.image_size)
        names_of = [[records[i]["file"] for i in rows] for rows in items]
        load_batch = _load_packed
        print(f"[library] source=packed {bin_path} image_size={args.image_size}", flush=True)
    else:
        files = _list_images(root, styles)
        items = files
        names_of = [[str(p.relative_to(root)) for p in f] for f in files]
        load_batch = lambda chunk: _load_batch(chunk, args.image_size)  # noqa: E731
        print(f"[library] source=dir {root}", flush=True)
    print(f"[library] styles={styles}", flush=True)
    print(f"[library] counts={[len(f) for f in items]} total={sum(len(f) for f in items)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = None
    encoder_size = 128
    if args.encoder_ckpt:
        from v2.style_encoder import StyleEncoder

        payload = torch.load(args.encoder_ckpt, map_location="cpu", weights_only=False)
        encoder = StyleEncoder(int(payload["num_styles"]), int(payload["embed_dim"]))
        encoder.load_state_dict(payload["state_dict"])
        encoder.to(device).eval().requires_grad_(False)
        encoder_size = int(payload.get("size", 128))
        print(f"[library] encoder={args.encoder_ckpt} val_acc={payload.get('val_acc'):.4f}", flush=True)
    gen = torch.Generator().manual_seed(777)
    per_style_local: list[list[torch.Tensor]] = []
    per_style_global: list[torch.Tensor] = []
    kept_names: list[list[str]] = []
    t0 = time.time()
    for si, (name, paths) in enumerate(zip(styles, items)):
        if len(paths) > args.max_per_style:
            pick = torch.randperm(len(paths), generator=gen)[: args.max_per_style].sort().values.tolist()
            paths = [paths[i] for i in pick]
            names_of[si] = [names_of[si][i] for i in pick]
        loc_parts: list[list[torch.Tensor]] = [[] for _ in range(3)]
        glob_parts: list[torch.Tensor] = []
        for s in range(0, len(paths), args.batch):
            chunk = paths[s: s + args.batch]
            x = load_batch(chunk).to(device)
            with torch.no_grad():
                desc = _subsample(local_descriptors(x), gen)
                glob = global_descriptor(x)
                if encoder is not None:
                    glob = torch.cat([glob, encode_style(encoder, x, size=encoder_size)], dim=1)
            for k in range(3):
                loc_parts[k].append(desc[k].float().cpu())
            glob_parts.append(glob.float().cpu())
        per_style_local.append([torch.cat(p, dim=0) for p in loc_parts])
        per_style_global.append(torch.cat(glob_parts, dim=0))
        kept_names.append(names_of[si])
        print(f"[library] {name}: n={len(paths)} elapsed={time.time() - t0:.1f}s", flush=True)

    # 白化参数（只用真实库统计）
    all_local = [torch.cat([per_style_local[s][k] for s in range(len(styles))], dim=0) for k in range(3)]
    local_mean = torch.stack([t.reshape(-1, t.shape[-1]).mean(0) for t in all_local])
    local_std = torch.stack([t.reshape(-1, t.shape[-1]).std(0).clamp_min(1e-3) for t in all_local])
    all_glob = torch.cat(per_style_global, dim=0)
    global_mean = all_glob.mean(0)
    global_std = all_glob.std(0).clamp_min(1e-3)

    labels_global = torch.cat(
        [torch.full((per_style_global[s].shape[0],), s, dtype=torch.long) for s in range(len(styles))]
    )
    labels_local = torch.cat(
        [torch.full((per_style_local[s][0].shape[0] * per_style_local[s][0].shape[1],), s, dtype=torch.long)
         for s in range(len(styles))]
    )
    local_whiten = []
    for k in range(3):
        z = (all_local[k].reshape(-1, LOCAL_DIM).float() - local_mean[k]) / local_std[k]
        local_whiten.append(fit_whitener(z, labels_local, len(styles)))
    global_whiten = fit_whitener((all_glob - global_mean) / global_std, labels_global, len(styles))
    local_whiten = torch.stack(local_whiten)
    stats = FeatureStats(local_mean, local_std, local_whiten, global_mean, global_std, global_whiten)

    # 类内/类间距离（白化 + 每图 patch 均值）
    def image_vectors(style_idx: int) -> torch.Tensor:
        vecs = []
        for k in range(3):
            d = per_style_local[style_idx][k].float().reshape(len(per_style_local[style_idx][k]), -1, LOCAL_DIM)
            z = (d - stats.local_mean[k].view(1, 1, -1)) / stats.local_std[k].view(1, 1, -1)
            vecs.append(torch.einsum("bpc,dc->bpd", z, stats.local_whiten[k]).mean(1))
        g = (per_style_global[style_idx].float() - stats.global_mean) / stats.global_std
        vecs.append(g @ stats.global_whiten.transpose(0, 1))
        return torch.cat(vecs, dim=1)

    vecs = [image_vectors(i) for i in range(len(styles))]
    cent = torch.stack([v.mean(0) for v in vecs])
    report: dict[str, object] = {"styles": styles, "counts": [len(v) for v in vecs], "image_size": args.image_size}
    # 类内半径 = 样本到本类质心的平均距离（与类间质心距离同尺度、可直接比较）
    intra = []
    intra_style = {}
    for i, name in enumerate(styles):
        sub = vecs[i]
        val = float((sub - cent[i]).norm(dim=1).mean().item())
        intra_style[name] = val
        intra.append(val)
    # 最近质心可分性：每类 800 建质心 / 200 留出
    correct = 0
    total_eval = 0
    centroids_train = []
    for i in range(len(styles)):
        v = vecs[i]
        cut = min(800, v.shape[0] - 1)
        centroids_train.append(v[:cut].mean(0))
        held = v[cut:]
        if held.shape[0] == 0:
            continue
        d = torch.cdist(held, torch.stack(centroids_train))
        correct += int((d.argmin(dim=1) == i).sum().item())
        total_eval += held.shape[0]
    report_nn_acc = correct / max(total_eval, 1)
    inter = []
    inter_pairs = {}
    for i in range(len(styles)):
        for j in range(i + 1, len(styles)):
            val = float((cent[i] - cent[j]).norm().item())
            inter.append(val)
            inter_pairs[f"{styles[i]}|{styles[j]}"] = val
    report["intra_mean"] = sum(intra) / len(intra)
    report["inter_mean"] = sum(inter) / len(inter)
    report["ratio_inter_over_intra"] = report["inter_mean"] / report["intra_mean"]
    report["intra_by_style"] = intra_style
    report["inter_pairs"] = inter_pairs
    report["nearest_centroid_eval_acc"] = report_nn_acc

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "swd-style-library-v1",
            "styles": styles,
            "files": kept_names,
            "image_size": args.image_size,
            "patches_per_level": PATCHES_PER_LEVEL,
            "local": per_style_local,
            "global": per_style_global,
            "local_mean": local_mean,
            "local_std": local_std,
            "local_whiten": local_whiten,
            "global_mean": global_mean,
            "global_std": global_std,
            "global_whiten": global_whiten,
            "encoder_ckpt": args.encoder_ckpt,
            "encoder_size": encoder_size,
            "report": report,
        },
        out,
    )
    (out.parent / "library_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False), flush=True)
    print(f"[library] saved {out} ({out.stat().st_size / 1e6:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
