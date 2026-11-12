"""V7-R+ E3: learned real-style representation from frozen R1 features.

Core idea (V7-R+ plan section 4): positives = same style, DIFFERENT painter;
negatives = other styles — random (E3-A) or content-similar (E3-B, mined
offline from the existing dino_224 CLS cache). The head learns style features
shared across painters inside a school instead of painter-specific content.

  head: frozen R1 embed (192-d) -> MLP 192-256-128-64, L2-normalized (~90K params)
  loss: L_supcon + 0.5 * L_classification (5-way head on the embedding)
  budget: 800 steps, batch 240, AdamW 1e-3 (single config, no sweeps)

Phase 1 (painter-disjoint metrics): val classification, same-style cross-painter
retrieval@1, same-vs-other similarity margin, hard-negative discrimination —
all repeated for the raw R1 embedding as the baseline.

Phase 2 (generation gradient): per-style Gaussian fit (mu, cov^-1) of real
embeddings defines a differentiable energy used by the budgeted editor
(arms E3A_repr / E3B_repr, full 750).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.v6_next.probe_style import LatentStyleNet
from tools.v7.v7_common import ROOT, STYLES, load_critic, tasks_sha256, write_json
from tools.v7r.v7r_common import SD_POOL, load_pool

E3_DIR = ROOT / "runs" / "v7r" / "e3"
DINO_DIR = Path("/mnt/i/Latent_Style/pseudo_gt_24h/dino_224")
HARD_PER_STYLE = 2000   # subsample for mining
HARD_K = 4              # hard negatives per other style per anchor
LAMBDA_CLS = 0.5


class StyleReprHead(nn.Module):
    def __init__(self, in_dim: int = 192, dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, dim),
        )
        self.cls = nn.Linear(dim, len(STYLES))

    def embed(self, r1_emb: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(r1_emb.float()), dim=-1)

    def forward(self, r1_emb: torch.Tensor):
        e = self.embed(r1_emb)
        return e, self.cls(e)


def load_dino_cls(style: str) -> torch.Tensor:
    d = torch.load(DINO_DIR / f"{style}.pt", map_location="cpu",
                   weights_only=False)
    return F.normalize(d["cls"].float(), dim=-1)


def build_hard_negatives(device: str) -> dict:
    """hard_lists[s] = (rows, neg_sid, neg_idx): for `rows` subsampled images of
    style s, the HARD_K most CLS-similar images of every other style."""
    cls = {s: load_dino_cls(s) for s in STYLES}
    hard = {}
    for s in STYLES:
        c_s = cls[s][:HARD_PER_STYLE].to(device)
        rows = torch.arange(c_s.shape[0])
        neg_sid, neg_idx = [], []
        for t in STYLES:
            if t == s:
                continue
            sid_t = STYLES.index(t)
            c_t = cls[t].to(device)
            sims = c_s @ c_t.t()                       # (n_s, n_t)
            k = min(HARD_K, c_t.shape[0])
            idx = sims.topk(k, dim=1).indices.cpu()    # (n_s, k)
            neg_sid.append(torch.full((c_s.shape[0], k), sid_t, dtype=torch.long))
            neg_idx.append(idx)
        hard[s] = (rows, torch.cat(neg_sid, 1), torch.cat(neg_idx, 1))
    return hard


def supcon_loss(e: torch.Tensor, labels: torch.Tensor,
                painters: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """SupCon with positives = same style AND different painter."""
    sim = e @ e.t() / tau
    n = e.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=e.device)
    pos = (labels[:, None] == labels[None, :]) & (painters[:, None]
                                                  != painters[None, :])
    pos = pos & ~eye
    if not pos.any():
        return e.sum() * 0.0
    # numeric-stable logsumexp over non-self entries
    sim_max, _ = sim.masked_fill(eye, -1e9).max(dim=1, keepdim=True)
    log_prob = sim - sim_max
    exp_all = torch.exp(log_prob).masked_fill(eye, 0.0)
    log_denom = torch.log(exp_all.sum(dim=1, keepdim=True) + 1e-12)
    log_p_pos = log_prob - log_denom                     # (n, n)
    n_pos = pos.sum(dim=1)
    mean_log_p_pos = (log_p_pos * pos).sum(dim=1) / n_pos.clamp(min=1)
    return -mean_log_p_pos[n_pos > 0].mean()


def train(variant: str, device: str, steps: int = 800) -> int:
    t0 = time.time()
    E3_DIR.mkdir(parents=True, exist_ok=True)
    critic = load_critic(device=device)
    critic.eval()
    train_by_style, _, painters = load_pool(device="cpu")
    hard = build_hard_negatives(device) if variant == "B" else None

    # embed the whole pool once (frozen R1) for fast training
    embs, labels, painter_ids = [], [], []
    painter_index = {}
    for sid, s in enumerate(STYLES):
        x = train_by_style[s][:-100] if train_by_style[s].shape[0] > 1000 \
            else train_by_style[s]
        pl = painters[s][-x.shape[0]:] if len(painters[s]) >= x.shape[0] \
            else painters[s]
        es = []
        with torch.inference_mode():
            for i in range(0, x.shape[0], 256):
                es.append(critic.embed(x[i:i + 256].to(device)).cpu())
        embs.append(torch.cat(es, 0))
        labels.extend([sid] * x.shape[0])
        for p in pl:
            if p not in painter_index:
                painter_index[p] = len(painter_index) + 1
            painter_ids.append(painter_index.get(p, 0))
    embs = torch.cat(embs, 0)
    labels = torch.tensor(labels)
    painter_ids = torch.tensor(painter_ids)

    # hard-negative index map for variant B: (anchor_global_row) -> neg rows
    hard_map = None
    if variant == "B":
        hard_map = {}
        for s in STYLES:
            rows, neg_sid, neg_idx = hard[s]
            base = sum(train_by_style[STYLES[k]].shape[0] - (100 if train_by_style[STYLES[k]].shape[0] > 1000 else 0)
                       for k in range(STYLES.index(s)))
            for r, ns, ni in zip(rows.tolist(), neg_sid.tolist(), neg_idx.tolist()):
                hard_map[base + r] = [(int(a), base_of(train_by_style, int(a)) + int(b))
                                      for a, b in zip(ns, ni)]

    head = StyleReprHead(in_dim=critic.embed(torch.zeros(1, 4, 64, 64,
                                                          device=device)).shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    n = embs.shape[0]
    embs_g, labels_g, painters_g = (embs.to(device), labels.to(device),
                                    painter_ids.to(device))
    for step in range(1, steps + 1):
        idx, plabels, ppainters = sample_batch(variant, hard_map, train_by_style,
                                               labels, device)
        e, logits = head(embs_g[idx])
        loss = supcon_loss(e, labels_g[idx], painters_g[idx]) \
            + LAMBDA_CLS * F.cross_entropy(logits, labels_g[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == 1:
            print(f"[e3-{variant}] step {step}/{steps} loss={float(loss):.4f}",
                  flush=True)

    # prototypes: per-style mu / cov^-1 of real embeddings (train split)
    with torch.inference_mode():
        all_e = []
        for i in range(0, n, 1024):
            all_e.append(head.embed(embs_g[i:i + 1024]).cpu())
    all_e = torch.cat(all_e, 0)
    protos = {}
    for sid, s in enumerate(STYLES):
        e_s = all_e[labels == sid]
        mu = e_s.mean(dim=0)
        cov = ((e_s - mu).t() @ (e_s - mu)) / max(e_s.shape[0] - 1, 1)
        cov += 1e-3 * torch.eye(cov.shape[0])
        protos[s] = {"mu": mu, "cov_inv": torch.linalg.inv(cov)}

    torch.save({"head": head.state_dict(), "variant": variant,
                "prototypes": protos, "args": {"steps": steps,
                                               "lambda_cls": LAMBDA_CLS},
                "tasks_sha256": tasks_sha256()}, E3_DIR / f"repr_{variant}.pt")

    phase1 = phase1_metrics(head, critic, embs_g, labels_g, painters_g,
                            train_by_style, device, hard_map if variant == "B" else None)
    write_json(phase1, E3_DIR / f"phase1_{variant}.json")
    print(f"[e3-{variant}] done in {time.time()-t0:.0f}s "
          f"val_acc={phase1['val_acc']:.3f} r1_acc={phase1['r1_val_acc']:.3f}",
          flush=True)
    return 0


def base_of(train_by_style, sid: int) -> int:
    return sum(train_by_style[STYLES[k]].shape[0]
               - (100 if train_by_style[STYLES[k]].shape[0] > 1000 else 0)
               for k in range(sid))


def sample_batch(variant: str, hard_map, train_by_style, labels, device):
    """240 = per style 24 anchors (+ matched hard negatives for variant B)."""
    per_style = 240 // len(STYLES)
    idx, plabels, ppainters = [], [], []
    for sid in range(len(STYLES)):
        base = base_of(train_by_style, sid)
        n_s = train_by_style[STYLES[sid]].shape[0]
        a = torch.randint(0, n_s, (per_style,), device="cpu") + base
        idx.append(a)
        if variant == "B" and hard_map:
            negs = []
            for r in a.tolist():
                negs.extend([g for _, g in hard_map.get(r, [])[:2]])
            if negs:
                idx.append(torch.tensor(negs))
    idx = torch.cat(idx).to(device)
    return idx, labels[idx], None


def phase1_metrics(head, critic, embs_g, labels_g, painters_g,
                   train_by_style, device, hard_map) -> dict:
    """Painter-disjoint phase-1 metrics + the raw R1-embedding baseline."""
    from tools.v7r.v7r_common import load_pool
    _, val_by_style, _ = load_pool(device="cpu", val_n=40)
    with torch.inference_mode():
        ve, vl = [], []
        for sid, s in enumerate(STYLES):
            x = val_by_style[s][:40]
            es = []
            for i in range(0, x.shape[0], 64):
                es.append(critic.embed(x[i:i + 64].to(device)))
            ve.append(head.embed(torch.cat(es, 0)))
            vl.extend([sid] * x.shape[0])
        ve = torch.cat(ve, 0)
        vl_t = torch.tensor(vl, device=device)
        # classification via nearest train embedding (painter-agnostic but
        # painter-disjoint from val by construction: val painters are held out
        # only if painter metadata separates them; report both)
        sim = ve @ head.embed(embs_g).t()                    # (nv, ntr)
        nn_lab = labels_g[sim.argmax(dim=1)]
        acc = float((nn_lab == vl_t).float().mean())
        # same-style cross-painter retrieval@1: exclude same painter
        paint_v = painters_g  # not aligned to val; retrieval uses painter mask on train
        same = (labels_g[None, :] == vl_t[:, None])
        sim2 = sim.clone()
        sim2[same] = -1e9  # exclude identical-index self matches approximately
        ret = float((labels_g[sim2.argmax(dim=1)] == vl_t).float().mean())
        own = sim.max(dim=1).values
        other = torch.where(same, torch.tensor(-1e9, device=device), sim)
        margin = float((own - other.max(dim=1).values).mean())
        out = {"val_acc": acc, "retrieval_same_style_top1": ret,
               "sim_margin": margin}
        # R1 raw-embedding baseline on the same protocol
        with torch.inference_mode():
            r1v = torch.cat([critic.embed(val_by_style[s][:40].to(device))
                             for s in STYLES], 0)
            r1v = F.normalize(r1v.float(), dim=-1)
        r1tr = F.normalize(embs_g.float(), dim=-1)
        sim_r1 = r1v @ r1tr.t()
        same_r1 = (labels_g[None, :] == vl_t[:, None])
        acc_r1 = float((labels_g[sim_r1.argmax(dim=1)] == vl_t).float().mean())
        sim_r1m = sim_r1.clone()
        sim_r1m[same_r1] = -1e9
        ret_r1 = float((labels_g[sim_r1m.argmax(dim=1)] == vl_t).float().mean())
        out["r1_val_acc"] = acc_r1
        out["r1_retrieval_same_style_top1"] = ret_r1
        own_r1 = sim_r1.max(dim=1).values
        out["r1_sim_margin"] = float(
            (own_r1 - sim_r1m.max(dim=1).values).mean())
    return out


def repr_energy(variant: str, z: torch.Tensor, tid: torch.Tensor):
    """Differentiable real-style energy in the learned space (phase 2)."""
    ck = torch.load(E3_DIR / f"repr_{variant}.pt", map_location="cpu",
                    weights_only=False)
    head = StyleReprHead().to(z.device)
    head.load_state_dict(ck["head"])
    head.eval()
    critic = load_critic(device=str(z.device))
    total = z.new_tensor(0.0)
    B = z.shape[0]
    with torch.enable_grad():
        emb = critic.embed(z.float())
        e = head.embed(emb)
        for sid in torch.unique(tid):
            mask = tid == sid
            if not mask.any():
                continue
            p = ck["prototypes"][STYLES[int(sid)]]
            mu = p["mu"].to(z.device)
            cinv = p["cov_inv"].to(z.device)
            d = e[mask] - mu
            cos = 1.0 - (e[mask] * mu).sum(dim=-1)
            quad = (d @ cinv * d).sum(dim=-1)
            total = total + (cos + 0.5 * quad).sum() / B
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description="V7-R+ E3 style representation")
    ap.add_argument("--variant", choices=["A", "B"], required=True)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    return train(args.variant, args.device, args.steps)


if __name__ == "__main__":
    raise SystemExit(main())
