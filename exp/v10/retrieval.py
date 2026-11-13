"""V10.1 retrieval: content-conditioned local stroke targets (stop-gradient).

Modes (pre-registered):
  M0  mean      : style-mean stroke vector mu_t (unconditional control)
  M1  top1      : the single best-matching bank patch's v
  M2  topk      : softmax-weighted mean over Top-8 matches (tau_s=0.1)
  M3  soft      : softmax-weighted mean over the whole bank (tau=2, ~KDE mean)

Similarity: cosine on standardized content k. Targets are DETACHED
(StyleGallery-style stop-gradient objective); the gradient flows through the
query's stroke encoding only.
"""
from __future__ import annotations

import torch

MODES = ("M0", "M1", "M2", "M3", "M1v", "M2v")
TAU_SOFT = 2.0
TAU_TOPK = 0.1
TOPK = 8
QUERY_CHUNK = 1024


class LocalBank:
    def __init__(self, path, device="cuda"):
        raw = torch.load(path, map_location="cpu", weights_only=False)
        self.styles = raw["styles"]
        self.scales = raw["scales"]
        self.device = device

        def _key(k):
            s, sc = k.rsplit("|", 1)
            return (s, int(sc))

        self.Q = {_key(k): t.to(device) for k, t in raw["Q"].items()}
        self.V = {_key(k): t.to(device) for k, t in raw["V"].items()}
        self.mu = {_key(k): t.to(device) for k, t in raw["mu"].items()}
        self.stats = {int(k): {kk: vv.to(device) for kk, vv in v.items()}
                      for k, v in raw["stats"].items()}

    def feats(self, z: torch.Tensor, scale: int):
        """Latent (B,4,64,64) -> (patches (N,4,H,W), k (N,Dk), v (N,51)).

        Gradient path: k and v both on-path; targets are detached at the
        retrieval step. Standardization constants are buffers.
        """
        from .features import q_feats
        from .featv2 import v_feats
        st = self.stats[scale]
        p = patchify_scale(z, scale)
        k = (q_feats(p) - st["q_mean"]) / st["q_std"]
        v = (v_feats(p) - st["v_mean"]) / st["v_std"]
        return p, k, v

    @torch.no_grad()
    def retrieve(self, k: torch.Tensor, style_idx: int, scale: int, mode: str,
                 v_query: torch.Tensor | None = None):
        """-> detached target (N,Dv) and per-query best similarity (N,).

        sim_key 'q' (default) retrieves by content similarity; 'v' retrieves by
        current stroke similarity (post-hoc snap-to-manifold diagnostic).
        """
        sim_key = "v" if mode.endswith("v") else "q"
        base_mode = mode[:-1] if mode.endswith("v") else mode
        s = self.styles[style_idx]
        Qb = self.Q[(s, scale)].float()
        Vb = self.V[(s, scale)].float()
        if mode == "M0":
            return self.mu[(s, scale)].float().expand(k.shape[0], -1), \
                torch.zeros(k.shape[0], device=k.device)
        if sim_key == "v":
            query = v_query
            ref = Vb
        else:
            query = k
            ref = Qb
        qn = torch.nn.functional.normalize(query.float(), dim=1)
        rn = torch.nn.functional.normalize(ref, dim=1)
        sims = torch.empty(qn.shape[0], rn.shape[0], device=self.device)
        for i in range(0, qn.shape[0], QUERY_CHUNK):
            sims[i:i + QUERY_CHUNK] = qn[i:i + QUERY_CHUNK] @ rn.T
        if base_mode == "M1":
            j = sims.argmax(1)
            return Vb[j], sims.max(1).values
        if base_mode == "M2":
            vals, idx = sims.topk(min(TOPK, Vb.shape[0]), dim=1)
            w = torch.softmax(vals / TAU_TOPK, dim=1)
            return (w.unsqueeze(-1) * Vb[idx]).sum(1), vals.max(1).values
        if base_mode == "M3":
            w = torch.softmax(sims / TAU_SOFT, dim=1)
            return (w @ Vb), sims.max(1).values
        raise ValueError(mode)


def patchify_scale(z: torch.Tensor, scale: int) -> torch.Tensor:
    if scale == 8:
        from .features import patchify
        return patchify(z)
    if scale == 16:
        from .lmbank import patchify16
        return patchify16(z)
    raise ValueError(scale)


def lm_style_loss(bank: LocalBank, z: torch.Tensor, tid: torch.Tensor,
                  mode: str, w_fine: float = 0.5):
    """Differentiable multi-scale distance to the retrieved (detached) targets.

    Returns (loss (B,), dist_per_patch dict {scale: (B,patches)}).
    """
    b = z.shape[0]
    dev = z.device
    total = torch.zeros(b, device=dev)
    per_patch = {}
    for scale, w in ((8, w_fine), (16, 1.0 - w_fine)):
        p, k, v = bank.feats(z, scale)
        n_patch = (64 // scale) ** 2        # patches per sample: 64 (s8), 16 (s16)
        dist_rows = []
        for i in range(b):
            tgt, _sim = bank.retrieve(k[i * n_patch:(i + 1) * n_patch],
                                      int(tid[i]), scale, mode,
                                      v_query=v[i * n_patch:(i + 1) * n_patch])
            d = (v[i * n_patch:(i + 1) * n_patch] - tgt.detach()).square().sum(1)
            dist_rows.append(d)
        dp = torch.stack(dist_rows)                     # (B,n_patch)
        per_patch[scale] = dp
        total = total + w * dp.mean(1)
    return total, per_patch


def lm_direction(bank: LocalBank, z: torch.Tensor, tid: torch.Tensor,
                 mode: str, samples_per_call: int = 2):
    """Per-sample unit-normalized -grad_z of the multi-scale target distance."""
    b = z.shape[0]
    g = torch.zeros_like(z)
    with torch.enable_grad():
        for i0 in range(0, b, samples_per_call):
            zi = z[i0:i0 + samples_per_call].detach().clone().requires_grad_(True)
            L, _ = lm_style_loss(bank, zi, tid[i0:i0 + samples_per_call], mode)
            g[i0:i0 + samples_per_call] = torch.autograd.grad(L.sum(), zi)[0]
    flat = g.flatten(1)
    nrm = flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return (flat / nrm).view_as(z)
