"""在真实 512 作品上训练并冻结轻量 Style Encoder（Style-ID + 监督对比）。

产物：``runs/v2/swd/style_encoder_512.pt``（含 embed_dim / num_styles / 权重），
供 ``tools/build_style_library.py --encoder-ckpt`` 追加到整图描述子。

* 只用于特征库；不参与任何对抗训练；
* 目标不是分类 SOTA，而是让五类在嵌入空间可分（类内 << 类间）。
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v2.style_encoder import StyleEncoder, supcon_loss  # noqa: E402


def _load_packed(bin_path: Path, index_path: Path):
    meta = json.loads(index_path.read_text(encoding="utf-8"))
    n, h, w, c = int(meta["n"]), int(meta["height"]), int(meta["width"]), int(meta["channels"])
    mm = np.memmap(bin_path, dtype=np.uint8, mode="r", shape=(n, h, w, c))
    styles = [r["style"] for r in meta["records"]]
    return mm, styles


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed-bin", default="data/train_d5/images_hwc_uint8.bin")
    ap.add_argument("--packed-index", default="data/train_d5/images_index.json")
    ap.add_argument("--out", default="runs/v2/swd/style_encoder_512.pt")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--contrast-weight", type=float, default=0.2)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mm, styles = _load_packed(Path(args.packed_bin), Path(args.packed_index))
    names = sorted(set(styles))
    label = torch.tensor([names.index(s) for s in styles], dtype=torch.long)
    num_styles = len(names)
    print(f"[encoder] n={len(styles)} styles={names}", flush=True)

    # 每类 800 训练 / 200 留出（仅分类准确率诊断，不改变全量特征）
    g = torch.Generator().manual_seed(9)
    train_idx, val_idx = [], []
    for c in range(num_styles):
        idx = torch.nonzero(label == c).flatten()
        perm = torch.randperm(idx.numel(), generator=g)
        idx = idx[perm]
        train_idx.append(idx[:800])
        val_idx.append(idx[800:])
    train_idx = torch.cat(train_idx)
    val_idx = torch.cat(val_idx)

    model = StyleEncoder(num_styles=num_styles).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    size = args.size

    def batch_tensor(idx: torch.Tensor, *, flip: bool) -> torch.Tensor:
        arr = np.asarray(mm[idx.numpy()])
        x = torch.from_numpy(arr).permute(0, 3, 1, 2).float().div_(255.0)
        x = F.interpolate(x, size=(size, size), mode="area")
        if flip:
            x = torch.flip(x, dims=[3])
        return x.to(device)

    def accuracy() -> float:
        model.eval()
        correct = 0
        with torch.no_grad():
            for s in range(0, val_idx.numel(), 64):
                chunk = val_idx[s: s + 64]
                _, logits = model(batch_tensor(chunk, flip=False))
                correct += int((logits.argmax(dim=1) == label[chunk].to(device)).sum().item())
        model.train()
        return correct / max(val_idx.numel(), 1)

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(train_idx.numel(), generator=g)
        total = 0.0
        steps = 0
        for s in range(0, train_idx.numel(), args.batch):
            chunk = train_idx[perm[s: s + args.batch]]
            x = batch_tensor(chunk, flip=True)
            y = label[chunk].to(device)
            emb, logits = model(x)
            loss = F.cross_entropy(logits, y) + args.contrast_weight * supcon_loss(emb, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item())
            steps += 1
        print(f"[encoder] epoch {epoch}/{args.epochs} loss={total / max(steps, 1):.4f} "
              f"val_acc={accuracy():.4f} elapsed={time.time() - t0:.1f}s", flush=True)

    acc = accuracy()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": "swd-style-encoder-v1",
        "num_styles": num_styles,
        "styles": names,
        "size": size,
        "embed_dim": 128,
        "val_acc": acc,
        "state_dict": model.state_dict(),
    }, out)
    print(f"[encoder] saved {out} val_acc={acc:.4f} ({out.stat().st_size / 1e6:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
