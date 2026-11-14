"""V13 P1: local stroke encoder E_s with content-stratified contrastive learning.

Input per token: concat(H2_ds 9ch, LL2_ds 3ch, edge 1ch) on a 32x32 grid of a
128-crop from a real painting. Output: D=64 embedding per token (spatial).

Supervision (plan §3 / user directive):
- positives: same style, DIFFERENT painting (cross-painting is mandatory)
- hard negatives: different style, CONTENT-MATCHED (nearest content descriptor
  across styles) — prevents subject/composition shortcuts
- VICReg variance+covariance regularization prevents collapse
- colour ablation check: the model must NOT rely on chroma (grayscale inputs
  keep style separability — evaluated at gate time)

Run: python -m exp.v13.stroke_encoder --updates 4000
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from exp.v11b.bands import dwt2, sobel_mag

REPO = Path(__file__).resolve().parents[2]
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
V11B = REPO / "exp/v11b"


def build_tokens(ds, indices, device, max_paintings=None):
    """RealWaveletDataset -> per-painting (13,32,32) token maps + labels.

    H2 (9,128,128) and LL2 (3,128,128) are average-pooled 4x to 32x32;
    edge is the Sobel magnitude of LL2 (1,32,32).
    Returns (X (N,13,32,32) fp16 cpu, y (N,) long, painting_idx per token not
    needed — tokens of one painting share the row).
    """
    from exp.v11b.dataset_real import RealWaveletDataset  # noqa: F401
    X, y = [], []
    take = indices if max_paintings is None else indices[:max_paintings]
    for i in take:
        crop, sid = ds[i]
        c = crop.float().to(device)
        d = dwt2(c[None])
        h2 = Fn.adaptive_avg_pool2d(d["H2"], 32)
        ll2 = Fn.adaptive_avg_pool2d(d["LL2"], 32)
        e = Fn.adaptive_avg_pool2d(sobel_mag(d["LL2"].mean(1, keepdim=True)), 32)
        x = torch.cat([h2, ll2, e], 1)[0]                # (13,32,32)
        X.append(x.half().cpu())
        y.append(sid)
    return torch.stack(X), torch.tensor(y, dtype=torch.long)


class StrokeEncoder(nn.Module):
    """13ch 32x32 -> (64,32,32) spatial embedding + pooled painting vector."""

    def __init__(self, d: int = 64, in_ch: int = 13):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, d, 3, padding=1), nn.SiLU(),
            nn.Conv2d(d, d, 3, padding=1), nn.SiLU(),
            nn.Conv2d(d, d, 3, padding=1), nn.SiLU())
        self.head = nn.Conv2d(d, d, 1)

    def forward(self, x):
        h = self.net(x)
        return self.head(h)                              # (B,D,32,32)


def vicreg_terms(z: torch.Tensor, var_target: float = 1.0):
    """z (B,D): variance + off-diagonal covariance penalties (VICReg-lite)."""
    zc = z - z.mean(0)
    std = zc.std(0)
    var_loss = Fn.relu(var_target - std).mean()
    n, d = zc.shape
    cov = (zc.T @ zc) / (n - 1)
    off = cov - torch.diag(torch.diag(cov))
    cov_loss = off.square().sum() / d
    return var_loss, cov_loss


def info_nce(emb, sid, pid, content_key, tau=0.2, content_match_w=0.5):
    """Contrastive loss on a token batch (from multiple paintings).

    positives: same style AND different painting (cross-painting mandatory);
    hard negatives: nearest-content cross-style token (content matching keeps
    the encoder from using subject/composition shortcuts).
    emb (N,D) normalized; sid/pid (N,); content_key (N,13).
    """
    N = emb.shape[0]
    sim = emb @ emb.T / tau
    same = sid[:, None] == sid[None, :]
    diff_paint = pid[:, None] != pid[None, :]
    pos_mask = same & diff_paint
    if not pos_mask.any():
        return torch.zeros((), device=emb.device), {}
    # vectorized positive loss (same style, different painting)
    pos_logp = torch.log_softmax(sim, dim=1)
    pos_loss = -(pos_mask.float() * pos_logp).sum(1)
    pos_cnt = pos_mask.sum(1).clamp_min(1)
    loss_pos = (pos_loss / pos_cnt).mean()
    # vectorized hard negative: nearest-content cross-style token
    cn = Fn.normalize(content_key, dim=1)
    d_content = torch.cdist(cn, cn)
    d_cross = d_content.clone()
    d_cross[sid[:, None] == sid[None, :]] = float("inf")
    j_hard = d_cross.argmin(dim=1)
    hard_logp = torch.log_softmax(sim, dim=1)
    loss_hard = -(hard_logp[torch.arange(N), j_hard]).mean()
    loss_hard = loss_hard if torch.isfinite(loss_hard) else \
        torch.zeros((), device=emb.device)
    total = loss_pos + content_match_w * loss_hard
    return total, {"pos": loss_pos.item(), "hard": loss_hard.item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", type=Path,
                    default=REPO / "runs/v13/stroke_encoder")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    device = a.device
    a.out.mkdir(parents=True, exist_ok=True)

    from exp.v11b.dataset_real import RealWaveletDataset
    ds = RealWaveletDataset("train", crop=128)
    g = torch.Generator().manual_seed(0)
    tr_idx = torch.randperm(len(ds), generator=g)[:1200].tolist()
    print("[p1] caching tokens for 1200 train paintings...", flush=True)
    X_tr, y_tr = build_tokens(ds, tr_idx, device)
    print(f"[p1] tokens: {tuple(X_tr.shape)}", flush=True)

    model = StrokeEncoder(d=a.d).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.updates, pct_start=0.05)

    n = X_tr.shape[0]
    N_PAINT, TOK = 16, 32
    t0 = time.time()
    for step in range(1, a.updates + 1):
        pidx = torch.randperm(n)[:N_PAINT]
        grids = X_tr[pidx].to(device).float()          # (16,13,32,32)
        sid_g = y_tr[pidx].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=device == "cuda"):
            out = model(grids)                          # (16,64,32,32)
        out = out.reshape(N_PAINT, a.d, 1024)
        sel = torch.randperm(1024)[:TOK]
        emb = out[:, :, sel].transpose(1, 2).reshape(N_PAINT * TOK, a.d)
        emb = Fn.normalize(emb.float(), dim=1)
        key = grids.reshape(N_PAINT, 13, 1024)[:, :, sel] \
            .transpose(1, 2).reshape(N_PAINT * TOK, 13)
        sid_tok = sid_g.repeat_interleave(TOK)
        pid_tok = pidx.to(device).repeat_interleave(TOK)
        loss, parts = info_nce(emb, sid_tok, pid_tok, key, tau=0.2)
        var_l, cov_l = vicreg_terms(emb)
        loss = loss + 0.5 * var_l + 0.05 * cov_l
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == 1:
            print(f"[p1] step {step}/{a.updates} loss={loss.item():.4f} "
                  f"pos={parts.get('pos', 0):.3f} hard={parts.get('hard', 0):.3f} "
                  f"var={var_l.item():.3f} {time.time()-t0:.0f}s", flush=True)

    torch.save({"model": model.state_dict(), "d": a.d,
                "args": vars(a)}, a.out / "es_encoder.pt")
    print("saved ->", a.out / "es_encoder.pt", flush=True)


if __name__ == "__main__":
    main()
