"""临时诊断：确认 Style Encoder 是否真的在学。"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from v2.style_encoder import StyleEncoder, supcon_loss  # noqa: E402

meta = json.loads(Path("data/train_d5/images_index.json").read_text())
mm = np.memmap("data/train_d5/images_hwc_uint8.bin", dtype=np.uint8, mode="r",
               shape=(meta["n"], meta["height"], meta["width"], meta["channels"]))
label = torch.tensor([sorted({r["style"] for r in meta["records"]}).index(r["style"]) for r in meta["records"]])
dev = "cuda"
g = torch.Generator().manual_seed(0)
idx = torch.randperm(5000, generator=g)[:64]
arr = np.asarray(mm[idx.numpy()])
print("raw img stats", arr.mean(), arr.std(), "unique rows", len({tuple(r[0, 0]) for r in arr[:8]}))
x = torch.from_numpy(arr).permute(0, 3, 1, 2).float().div_(255.0)
x = F.interpolate(x, size=(128, 128), mode="area").to(dev)
print("x stats", float(x.mean()), float(x.std()))
y = label[idx].to(dev)

model = StyleEncoder(5).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for step in range(8):
    emb, logits = model(x)
    loss = F.cross_entropy(logits, y) + 0.5 * supcon_loss(emb, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
    opt.step()
    print(f"step {step} loss={float(loss):.4f} ce={float(F.cross_entropy(logits, y)):.4f} "
          f"logit_std={float(logits.std()):.4f} grad_sum={gn:.3f} emb_std={float(emb.std()):.5f}")
