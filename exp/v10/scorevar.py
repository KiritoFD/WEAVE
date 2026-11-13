"""V10.2 variant style scores (plan: docs/v10_2/WEAVE_V10_2_CONDITIONAL_SCORE_750.md).

All variants operate on the v10 fine-scale KDE bank (kde_bank.pt, standardized
f-space, 34-dim) unless stated. Pre-registered hyperparams: sigma=1.0, tau=2.0.

  A1  target KDE            : exact v10 baseline (reproduction)
  A2  wrong-style KDE       : score against bank of style (t+1)%5
  A3  mixture KDE           : bank = concat of all five styles
  A4  q-shuffle             : kernel weights use within-sample shuffled q
  A5  no-v-weight           : mean-shift with q-only weights (NOT a density
                              gradient — v does not enter the weights; kept as
                              the pre-registered "state selection" control)
  A6  multi-scale joint     : 0.5*L_s8(f34 kde_bank) + 0.5*L_s16(lm_bank v51)
  A7  relations-only s8     : lm_bank s8 (v51) single scale (interpretability)

  B-* difference scores     : L = L_target - beta*L_mix (or +0.3*diff), plus the
                              absolute-density acceptance constraint at optimization.
"""
from __future__ import annotations

import torch

SIGMA = 1.0
TAU = 2.0
QUERY_CHUNK = 1024


class VariantScores:
    def __init__(self, runs_dir, device="cuda"):
        raw = torch.load(runs_dir / "bank/kde_bank.pt", map_location="cpu",
                         weights_only=False)
        self.styles = raw["styles"]
        self.device = device
        # fine-scale (f34) per-style banks, on GPU, detached
        self.Q = {s: raw["Q"][s].to(device).float() for s in self.styles}
        self.V = {s: raw["V"]["f"][s].to(device).float() for s in self.styles}
        self.Q_mix = torch.cat([self.Q[s] for s in self.styles])
        self.V_mix = torch.cat([self.V[s] for s in self.styles])
        # coarse scale (16x16, v51) from lm_bank for A6
        lmb = torch.load(runs_dir / "bank/lm_bank.pt", map_location="cpu",
                         weights_only=False)
        self.Q16 = {s: lmb["Q"][f"{s}|16"].to(device).float() for s in self.styles}
        self.V16 = {s: lmb["V"][f"{s}|16"].to(device).float() for s in self.styles}
        self.st16 = {k: {kk: vv.to(device) for kk, vv in v.items()}
                     for k, v in lmb["stats"].items()}[16]

    # ------------------------------------------------------------------
    def _logp_rows(self, kq, vq, Qb, Vb, use_v_weight=True, shuffle_q=False):
        """Truncated-free full log-density per query row (sum over bank), with
        optional within-batch q shuffling. Differentiable through vq (and through
        kq unless shuffled)."""
        n = vq.shape[0]
        if shuffle_q:
            perm = torch.randperm(n, device=vq.device)
            kq = kq[perm]
        outs = []
        for i in range(0, n, QUERY_CHUNK):
            dv = torch.cdist(vq[i:i + QUERY_CHUNK].float(), Vb).square()
            logits = -dv / (2 * SIGMA ** 2)
            if use_v_weight:
                dq = torch.cdist(kq[i:i + QUERY_CHUNK].float(), Qb).square()
                logits = logits - dq / (2 * TAU ** 2)
            outs.append(torch.logsumexp(logits, dim=1))
        return torch.cat(outs)

    def logp_sum(self, z_feats, style_idx: int, variant: str, beta: float = 1.0):
        """Differentiable scalar for autograd; z_feats = {"q","f"} on-path."""
        q, v = z_feats["q"], z_feats["f"]
        s = self.styles[style_idx]
        if variant == "A1":
            return self._logp_rows(q, v, self.Q[s], self.V[s]).sum()
        if variant == "A2":
            t2 = (style_idx + 1) % 5
            s2 = self.styles[t2]
            return self._logp_rows(q, v, self.Q[s2], self.V[s2]).sum()
        if variant == "A3":
            return self._logp_rows(q, v, self.Q_mix, self.V_mix).sum()
        if variant == "A4":
            return self._logp_rows(q, v, self.Q[s], self.V[s],
                                   shuffle_q=True).sum()
        if variant == "A6":
            st = self.st16
            from .features import patchify  # noqa: F401  (feats built by caller)
            raise NotImplementedError("A6 handled via logp_s16 below")
        if variant.startswith("B"):
            Lt = self._logp_rows(q, v, self.Q[s], self.V[s])
            Lm = self._logp_rows(q, v, self.Q_mix, self.V_mix)
            if variant == "B-T":
                return Lt.sum()
            if variant == "B-TM0.5":
                return (Lt - 0.5 * Lm).sum()
            if variant == "B-TM1.0":
                return (Lt - Lm).sum()
            if variant == "B-T+D":
                return (Lt + 0.3 * (Lt - Lm)).sum()
        raise ValueError(variant)

    def logp_s16_sum(self, z16_feats):
        """Coarse-scale KDE (lm_bank 16x16, v51) — used by A6's second term."""
        k, v = z16_feats["k"], z16_feats["v"]
        s = z16_feats["style"]
        return self._logp_rows(k, v, self.Q16[s], self.V16[s]).sum()

    def mean_shift_q_only(self, z_feats, style_idx: int):
        """A5: pull v toward the q-weighted bank mean; weights ignore v.
        Returns (N,Dv) field (NOT normalized by sigma — direction gets
        per-sample normalized by the caller anyway)."""
        q, v = z_feats["q"], z_feats["f"]
        s = self.styles[style_idx]
        n = v.shape[0]
        shifts = torch.empty_like(v)
        for i in range(0, n, QUERY_CHUNK):
            dq = torch.cdist(q[i:i + QUERY_CHUNK].float(), self.Q[s]).square()
            w = torch.softmax(-dq / (2 * TAU ** 2), dim=1)
            shifts[i:i + QUERY_CHUNK] = w @ self.V[s] - v[i:i + QUERY_CHUNK]
        return shifts

    def diff_logp_target(self, z_feats, style_idx: int):
        """Absolute target density (acceptance constraint for B variants)."""
        s = self.styles[style_idx]
        return self._logp_rows(z_feats["q"], z_feats["f"],
                               self.Q[s], self.V[s])
