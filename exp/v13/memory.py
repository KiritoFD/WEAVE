"""V13 P1 gate: per-style stroke memories from E_s + Sim check.

Everything at PAINTING level (mean of the 1024 token embeddings per painting):
  M_s = kmeans(32) over train-painting mean embeddings, per style
  Sim = cosine(painting embedding, each style's nearest prototype)
Gate: token/painting accuracy + correct-vs-wrong margin on holdout.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .stroke_encoder import StrokeEncoder, build_tokens

REPO = Path(__file__).resolve().parents[2]
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
N_PROTO = 32


def paint_embed(model, ds, indices, device):
    """-> (n_paint, D) painting-level mean embeddings + (n_paint,) labels."""
    X, y = build_tokens(ds, indices, device)                 # (n,13,32,32), (n,)
    embs = []
    with torch.no_grad():
        for i in range(0, X.shape[0], 64):
            xb = X[i:i + 64].to(device).float()
            e = model(xb)                                    # (b,D,32,32)
            embs.append(e.mean(dim=(2, 3)).half().cpu())     # (b,D)
    return torch.cat(embs), y


def kmeans(x: torch.Tensor, k: int, iters: int = 25, seed: int = 0):
    g = torch.Generator(device=x.device).manual_seed(seed)
    idx = torch.randperm(x.shape[0], generator=g, device=x.device)[:k]
    c = x[idx].clone()
    for _ in range(iters):
        d = torch.cdist(x, c)
        a = d.argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                c[j] = x[m].mean(0)
    return c, a


def main():
    device = "cuda"
    enc_dir = REPO / "runs/v13/stroke_encoder"
    model = StrokeEncoder(d=64).to(device)
    ck = torch.load(enc_dir / "es_encoder.pt", map_location="cpu",
                    weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()

    from exp.v11b.dataset_real import RealWaveletDataset
    ds = RealWaveletDataset("train", crop=128)
    g = torch.Generator().manual_seed(1)
    tr_idx = torch.randperm(len(ds), generator=g)[:800].tolist()

    print("[mem] embedding train paintings...", flush=True)
    E_tr, y_tr = paint_embed(model, ds, tr_idx, device)
    M = {}
    for si in range(5):
        m = y_tr == si
        E_s = E_tr[m].float().to(device)
        M[si], _ = kmeans(E_s, N_PROTO, seed=si)
    print("[mem] prototypes built", flush=True)

    # ---- Sim gate on holdout ----
    ds_h = RealWaveletDataset("holdout", crop=128)
    g2 = torch.Generator().manual_seed(2)
    ho_idx = torch.randperm(len(ds_h), generator=g2)[:200].tolist()
    E_ho, y_ho = paint_embed(model, ds_h, ho_idx, device)
    E_ho = E_ho.float().to(device)

    sims = torch.zeros(E_ho.shape[0], 5, device=device)
    for si in range(5):
        pn = torch.nn.functional.normalize(M[si], dim=1)
        e = torch.nn.functional.normalize(E_ho, dim=1)
        sims[:, si] = (e @ pn.T).max(dim=1).values
    pred = sims.argmax(1).cpu()
    acc = (pred == y_ho).float().mean().item()

    # correct-vs-wrong margin
    true_idx = y_ho.to(device)
    correct_sim = torch.gather(sims, 1, true_idx[:, None]).squeeze(1)
    wrong_sim = sims.scatter(1, true_idx[:, None], float("-inf")).max(1).values
    margin = (correct_sim - wrong_sim).mean().item()

    # per-style breakdown
    per_style = {}
    for si in range(5):
        m = y_ho == si
        if m.any():
            per_style[STYLES[si]] = round((pred[m] == si).float().mean().item(), 3)

    report = {"painting_acc": round(acc, 4),
              "correct_sim": round(correct_sim.mean().item(), 4),
              "wrong_sim": round(wrong_sim.mean().item(), 4),
              "margin": round(margin.item(), 4) if hasattr(margin, 'item') else round(margin, 4),
              "per_style_acc": per_style,
              "gate_pass": bool(acc >= 0.5 and margin > 0)}
    print(json.dumps(report, indent=1), flush=True)

    torch.save({"protos": {STYLES[si]: M[si].cpu() for si in range(5)}},
               REPO / "runs/v13/stroke_memory.pt")
    (REPO / "runs/v13/p1_gate.json").write_text(json.dumps(report, indent=1))
    print("saved memory + gate report", flush=True)


if __name__ == "__main__":
    main()
