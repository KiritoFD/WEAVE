"""P1 R1: a light latent style discriminator, and the gate that decides whether P2 may use it.

The point is NOT to build a stronger classifier to flatter the score. It is to find a style
representation that (a) separates the hard targets on REAL images and (b) has usable gradients into
the generator. The plan is explicit that a critic which only pleases itself must not be added to the
generator, so this module reports both the classification gate and a gradient probe.

Design
  input   SD-VAE latent (4,64,64), the same space the Direct operator edits
  model   multi-scale CNN: 64 -> 32 -> 16, ~0.4M parameters
  loss    class-balanced cross entropy + class-balanced supervised contrastive
  split   GROUPED BY PAINTER so the same painter never appears in both train and val; where the
          painter cannot be parsed from the filename the image becomes its own group, which is the
          strictest possible split

Gate (plan section 4)
  balanced top-1 >= 0.75 on the real-image val split
  every target recall >= 0.50
  must beat the existing DWT-moment representation on ER / Rococo / Ukiyo

Usage
    python3 -m tools.v6_next.probe_style --tag r1
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CACHE = Path("/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16")
FILES = ["00_Early_Renaissance.pt", "01_Impressionism.pt", "02_Minimalism.pt",
         "03_Rococo.pt", "04_Ukiyo_e.pt"]
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
OUT = ROOT / "runs/v6_next/probes"


def painter_of(path: str) -> str:
    """Painter token from 'Style__painter_title.jpg', else the whole stem.

    Minimalism and Ukiyo filenames do not follow the pattern ('Minimalism__14_r-71.jpg',
    'Ukiyo_e__beauty_mitate-...'), where the first token is a number or a common noun. In those
    cases the image becomes its own group, which cannot leak across the split.
    """
    stem = Path(path).stem
    if "__" not in stem:
        return stem
    rest = stem.split("__", 1)[1]
    tok = re.split(r"[_]", rest)[0]
    if len(tok) < 5 or tok.isdigit():
        return stem
    return tok


class LatentStyleNet(nn.Module):
    """Multi-scale CNN on (4,64,64) latents. ~0.4M parameters at width 64."""

    def __init__(self, num_styles: int = 5, width: int = 64):
        super().__init__()
        w = width
        self.stem = nn.Sequential(
            nn.Conv2d(4, w, 3, padding=1), nn.GroupNorm(8, w), nn.SiLU())
        self.b64 = nn.Sequential(
            nn.Conv2d(w, w, 3, padding=1), nn.GroupNorm(8, w), nn.SiLU())
        self.down32 = nn.Conv2d(w, w * 2, 3, stride=2, padding=1)
        self.b32 = nn.Sequential(
            nn.Conv2d(w * 2, w * 2, 3, padding=1), nn.GroupNorm(8, w * 2), nn.SiLU())
        self.down16 = nn.Conv2d(w * 2, w * 3, 3, stride=2, padding=1)
        self.b16 = nn.Sequential(
            nn.Conv2d(w * 3, w * 3, 3, padding=1), nn.GroupNorm(8, w * 3), nn.SiLU())
        self.head = nn.Linear(w * 3, num_styles)

    def embed(self, z: torch.Tensor) -> torch.Tensor:
        h = self.stem(z)
        h = self.b64(h)
        h = self.b32(self.down32(h))
        h = self.b16(self.down16(h))
        return F.adaptive_avg_pool2d(h, 1).flatten(1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(z))


def supcon(feats: torch.Tensor, labels: torch.Tensor, T: float = 0.1) -> torch.Tensor:
    """Class-balanced supervised contrastive loss (one positive set per class)."""
    f = F.normalize(feats, dim=1)
    sim = f @ f.t() / T
    n = f.shape[0]
    eye = torch.eye(n, device=f.device, dtype=torch.bool)
    pos = (labels[:, None] == labels[None, :]) & ~eye
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    exp = torch.exp(sim) * (~eye)
    log_prob = sim - torch.log(exp.sum(1, keepdim=True) + 1e-12)
    pos_cnt = pos.sum(1)
    valid = pos_cnt > 0
    if valid.sum() == 0:
        return torch.zeros((), device=f.device)
    loss = -(log_prob * pos).sum(1)[valid] / pos_cnt[valid]
    return loss.mean()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="r1")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--supcon-w", type=float, default=0.5)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- load latents with painter groups ----
    Z, Y, G = [], [], []
    for sid, f in enumerate(FILES):
        d = torch.load(CACHE / f, map_location="cpu", weights_only=False)
        lat = d["latents"]
        ids = d.get("files") or d.get("ids") or [f"img{i}" for i in range(len(lat))]
        Z.append(lat.half())
        Y.append(torch.full((len(lat),), sid, dtype=torch.long))
        G += [painter_of(x) for x in ids]
        print(f"[r1] {STYLES[sid]:20s} n={len(lat):5d} groups={len(set(G))}", flush=True)
    Z = torch.cat(Z, 0)
    Y = torch.cat(Y, 0)
    groups = np.array(G)
    print(f"[r1] total {len(Z)} images, {len(set(G))} painter groups", flush=True)

    # ---- grouped split, STRATIFIED PER STYLE ----
    # A global painter split is badly unbalanced here: Minimalism and Ukiyo filenames often lack a
    # parseable painter so each image becomes its own group, and those singleton groups dominated
    # the global 20%, leaving Rococo with only 47 validation images out of 2519. Splitting painters
    # within each style keeps every style represented while still never sharing a painter across
    # the two sides.
    rng = np.random.default_rng(args.seed)
    is_val = np.zeros(len(groups), dtype=bool)
    for s in range(5):
        idx_s = np.where(Y.numpy() == s)[0]
        g_s = sorted({groups[i] for i in idx_s})
        rng.shuffle(g_s)
        n_v = max(1, int(round(len(g_s) * args.val_frac)))
        vg = set(g_s[:n_v])
        for i in idx_s:
            if groups[i] in vg:
                is_val[i] = True
    tr_idx = torch.from_numpy(np.where(~is_val)[0]).long()
    va_idx = torch.from_numpy(np.where(is_val)[0]).long()
    print(f"[r1] split: train {len(tr_idx)}, val {len(va_idx)} — stratified per style, "
          f"no painter shared", flush=True)
    for s in range(5):
        ntr = int((Y[tr_idx] == s).sum()); nva = int((Y[va_idx] == s).sum())
        print(f"       {STYLES[s]:20s} train {ntr:5d}  val {nva:4d}", flush=True)
    if int((Y[va_idx] == 3).sum()) < 100:
        print("[r1] WARNING: Rococo validation set is small; recall estimate will be noisy",
              flush=True)

    model = LatentStyleNet(width=args.width).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[r1] parameters: {n_par:,}", flush=True)

    # class-balanced CE weights
    cnt = torch.bincount(Y[tr_idx], minlength=5).float()
    w_ce = (cnt.sum() / (5 * cnt.clamp_min(1))).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    Zg = Z.to(dev)
    Yg = Y.to(dev)
    tr_g = tr_idx.to(dev)
    va_g = va_idx.to(dev)
    gtr = torch.Generator(device="cpu").manual_seed(args.seed)

    def evaluate(idx):
        model.eval()
        preds, labels, embs = [], [], []
        with torch.inference_mode():
            for s in range(0, len(idx), 256):
                b = idx[s:s + 256]
                z = Zg[b].float()
                e = model.embed(z)
                preds.append(model.head(e).argmax(1).cpu())
                labels.append(Yg[b].cpu())
                embs.append(e.cpu())
        p = torch.cat(preds); l = torch.cat(labels)
        rec = [(float(((p == s) & (l == s)).sum()) / max(1, int((l == s).sum())))
               for s in range(5)]
        conf = torch.zeros(5, 5, dtype=torch.long)
        for t, q in zip(l.tolist(), p.tolist()):
            conf[t, q] += 1
        return {"balanced_top1": float(np.mean(rec)), "per_class_recall": rec,
                "confusion": conf.tolist(), "n": int(len(l)),
                "embeddings": torch.cat(embs)}

    for step in range(1, args.steps + 1):
        model.train()
        b = tr_idx[torch.randint(0, len(tr_idx), (args.batch,), generator=gtr)]
        z = Zg[b].float()
        y = Yg[b]
        e = model.embed(z)
        logits = model.head(e)
        loss = F.cross_entropy(logits, y, weight=w_ce)
        if args.supcon_w > 0:
            loss = loss + args.supcon_w * supcon(e, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0 or step == 1:
            r = evaluate(va_g)
            print(f"[r1] step {step:5d} loss={float(loss):.4f} "
                  f"val_bal_top1={r['balanced_top1']:.3f} "
                  f"recall={[round(x,3) for x in r['per_class_recall']]}", flush=True)

    rv = evaluate(va_g)
    rt = evaluate(tr_g)
    print(f"\n[r1] FINAL  train bal_top1={rt['balanced_top1']:.3f}  "
          f"val bal_top1={rv['balanced_top1']:.3f}")
    print(f"  val per-class recall: "
          + "  ".join(f"{s[:9]}={r:.3f}" for s, r in zip(STYLES, rv['per_class_recall'])))
    print("  val confusion (rows=true, cols=pred):")
    print("    " + "".join(f"{s[:8]:>10s}" for s in STYLES))
    for i, s in enumerate(STYLES):
        print(f"  {s[:16]:16s}" + "".join(f"{c:10d}" for c in rv["confusion"][i]))

    gate_top1 = rv["balanced_top1"] >= 0.75
    gate_recall = all(r >= 0.50 for r in rv["per_class_recall"])
    hard = {s: rv["per_class_recall"][STYLES.index(s)] for s in
            ("Early_Renaissance", "Rococo", "Ukiyo_e")}
    print(f"\n  GATE balanced_top1>=0.75 : {gate_top1} ({rv['balanced_top1']:.3f})")
    print(f"  GATE every recall>=0.50  : {gate_recall} "
          f"(min {min(rv['per_class_recall']):.3f})")
    print(f"  hard-target recall       : "
          + "  ".join(f"{k[:9]}={v:.3f}" for k, v in hard.items()))

    res = {"tag": args.tag, "params": n_par, "width": args.width, "steps": args.steps,
           "n_train": int(len(tr_idx)), "n_val": int(len(va_idx)),
           "n_groups": int(len(set(G))), "val": {k: v for k, v in rv.items()
                                               if k != "embeddings"},
           "train_balanced_top1": rt["balanced_top1"],
           "gate_top1": gate_top1, "gate_recall": gate_recall,
           "hard_target_recall": hard, "styles": STYLES}
    (OUT / f"r1_{args.tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    torch.save({"model": model.state_dict(), "width": args.width, "tag": args.tag},
               OUT / f"r1_{args.tag}.pt")
    print(f"\nwrote {OUT}/r1_{args.tag}.json")


if __name__ == "__main__":
    main()
