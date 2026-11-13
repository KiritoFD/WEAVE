"""Closed-form content-conditioned patch style KDE (V10-A/B/C), plan §4.

  p_sigma(v|q,t) ∝ sum_j exp( -||v-v_j||^2/(2 sigma^2) - ||q-q_j||^2/(2 tau^2) )
  s_t(v|q)       = ( sum_j w~_j v_j - v ) / sigma^2

Bank-space conventions (built by exp.v10.bank):
  Q  : standardized content descriptors (Dq=84)
  Vf : standardized joint style features (34)            -> variant B
  Vr : standardized residual v = f - h_psi(q) (34)       -> variant C
  Vf_color (14) / Vf_texture (20)                        -> shortcut audit views

Memory design (6GB budget):
- full-bank distances under no_grad in 1024-query chunks (fp32 accumulate);
- the differentiable path gathers only the top-K (4096) bank rows, so the
  autograd graph never holds an (nq x nbank) tensor;
- the style gradient flows through the v path only; q is stop-gradient
  (pre-registered: conditioning selects WHERE to pull, the pull acts on v).
"""
from __future__ import annotations

import torch

SIGMA = 1.0
TAU = 2.0
TOPK = 4096
QUERY_CHUNK = 1024

# variant -> (space, view, conditional)
VARIANTS = {
    "A": ("res", "joint", False),        # unconditional KDE, residual space
    "A_f": ("f", "joint", False),        # unconditional KDE, raw f space
    "B": ("f", "joint", True),           # content-conditional, raw f
    "C": ("res", "joint", True),         # content-conditional + residual
    "B_color": ("f", "color", True),
    "B_texture": ("f", "texture", True),
}

_SPACE_VIEW_KEY = {
    ("f", "joint"): "f",
    ("res", "joint"): "res",
    ("f", "color"): "f_color",
    ("f", "texture"): "f_texture",
}


def _d2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(n,D) x (m,D) squared distances, fp32 accumulate."""
    a2 = (a * a).sum(1, keepdim=True)
    b2 = (b * b).sum(1)
    ab = a @ b.T
    return (a2 + b2 - 2 * ab).clamp_min(0)


class StyleBank:
    def __init__(self, path, device="cuda"):
        raw = torch.load(path, map_location="cpu", weights_only=False)
        self.styles = raw["styles"]
        self.device = device
        self.stats = raw.get("stats", {})              # standardization tensors (cpu)
        self.Q = {s: raw["Q"][s].to(device) for s in self.styles}
        self.V = {v: {s: raw["V"][v][s].to(device) for s in self.styles}
                  for v in ("f", "res", "f_color", "f_texture")}
        self.meta = raw.get("meta", {})

    def logp_sum(self, q: torch.Tensor, v: torch.Tensor, style_idx: int,
                 cond: bool, space: str = "res", view: str = "joint") -> torch.Tensor:
        """Differentiable truncated log-density sum for (nq, Dv) patches.

        q: standardized content descriptors (stop-grad inside);
        v: query descriptors in bank space, on the autograd path.
        """
        s = self.styles[style_idx]
        Vb = self.V[_SPACE_VIEW_KEY[(space, view)]][s].float()
        Qb = self.Q[s].float() if cond else None
        n = v.shape[0]
        k = min(TOPK, Vb.shape[0])
        # 1) no_grad full-bank pass: pick top-K neighbours per query
        with torch.no_grad():
            idx_parts = []
            for i in range(0, n, QUERY_CHUNK):
                dv = _d2(v[i:i + QUERY_CHUNK].detach().float(), Vb)
                logits = -dv / (2 * SIGMA ** 2)
                if cond:
                    dq = _d2(q[i:i + QUERY_CHUNK].detach().float(), Qb)
                    logits = logits - dq / (2 * TAU ** 2)
                idx_parts.append(torch.topk(logits, k, dim=1).indices)
            idx = torch.cat(idx_parts) if len(idx_parts) > 1 else idx_parts[0]
        # 2) differentiable restricted evaluation on the gathered rows
        outs = []
        for c0 in range(0, n, QUERY_CHUNK):
            c1 = min(n, c0 + QUERY_CHUNK)
            Vtop = Vb[idx[c0:c1]]                        # (c,K,Dv)
            d2 = (v[c0:c1].unsqueeze(1) - Vtop).square().sum(-1)
            outs.append(torch.logsumexp(-d2 / (2 * SIGMA ** 2), dim=1))
        return torch.cat(outs).sum()

    @torch.no_grad()
    def logp_per_patch(self, q: torch.Tensor, v: torch.Tensor, style_idx: int,
                       cond: bool, space: str = "res", view: str = "joint") -> torch.Tensor:
        """Detached per-patch truncated log-density (nq,) — monotonicity checks."""
        s = self.styles[style_idx]
        Vb = self.V[_SPACE_VIEW_KEY[(space, view)]][s].float()
        Qb = self.Q[s].float() if cond else None
        n = v.shape[0]
        k = min(TOPK, Vb.shape[0])
        outs = []
        for i in range(0, n, QUERY_CHUNK):
            dv = _d2(v[i:i + QUERY_CHUNK].float(), Vb)
            logits = -dv / (2 * SIGMA ** 2)
            if cond:
                dq = _d2(q[i:i + QUERY_CHUNK].float(), Qb)
                logits = logits - dq / (2 * TAU ** 2)
            outs.append(torch.logsumexp(logits, dim=1))
        return torch.cat(outs)

    def direction(self, z: torch.Tensor, style_idx: int, variant: str,
                  feat_fn, samples_per_call: int = 4) -> torch.Tensor:
        """Per-sample-normalized grad_z of truncated logp for latents (B,4,64,64).

        feat_fn(z_leaf) -> {"q": (B*64,84) standardized content (detached ok),
                            "v": (B*64,Dv) bank-space descriptors on-path}.
        Supplied by exp.v10.bank.make_feat_fn so kde.py stays feature-agnostic.
        """
        space, view, cond = VARIANTS[variant]
        b = z.shape[0]
        g = torch.zeros_like(z)
        for i0 in range(0, b, samples_per_call):
            zi = z[i0:i0 + samples_per_call].detach().clone().requires_grad_(True)
            feats = feat_fn(zi)
            L = self.logp_sum(feats["q"], feats["v"], style_idx, cond, space, view)
            g[i0:i0 + samples_per_call] = torch.autograd.grad(L, zi)[0]
        flat = g.flatten(1)
        nrm = flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return (flat / nrm).view_as(z)
