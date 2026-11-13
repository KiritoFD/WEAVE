"""Differentiable local style energy over patch descriptors.

E_{t,i}(z) = -tau * log mean_{p in NN4(f_i(z), P_t)} exp(-||f_i-p||^2/tau)
L_loc(z,t) = mean_i [ E_{t,i} + beta * softplus(mu + E_{t,i} - E_{~t,i}) ]

Prototypes are detached; distances keep gradient w.r.t. z.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .patch_bank import patch_feats, STYLES

TAU = 0.5
BETA = 1.0
MU = 0.0
TOP4 = 4


class LocalStyleEnergy:
    def __init__(self, bank_path: str, device: str = "cuda"):
        bank = torch.load(bank_path, map_location="cpu", weights_only=False)
        self.mean = bank["mean"].to(device)
        self.std = bank["std"].to(device)
        # protos: (5, K, D) unit-normalised space handled by caller feats
        self.protos = {s: bank["protos"][s].to(device).float()
                       for s in bank["styles"]}
        self.style_idx = {s: i for i, s in enumerate(STYLES)}
        self.device = device

    def feats(self, z: torch.Tensor) -> torch.Tensor:
        """(B,4,64,64) -> (B*64, D) normalised."""
        return (patch_feats(z) - self.mean) / self.std

    def _energy_one(self, f: torch.Tensor, proto: torch.Tensor) -> torch.Tensor:
        """f: (N,D) -> per-patch softmin energy (N,)."""
        # chunked cdist to bound memory: N can be B*64
        out = torch.empty(f.shape[0], device=f.device)
        for s in range(0, f.shape[0], 4096):
            d = torch.cdist(f[s:s + 4096], proto) ** 2      # (n,K)
            v, _ = d.topk(TOP4, dim=1, largest=False)
            out[s:s + 4096] = -TAU * torch.logsumexp(-v / TAU, dim=1) \
                + TAU * torch.log(torch.tensor(TOP4, device=f.device))
        return out

    def energies(self, z: torch.Tensor, tid: torch.Tensor):
        """Return E_t and E_negmean per sample: (B,64) each."""
        b = z.shape[0]
        f = self.feats(z.reshape(b, 4, 64, 64) if z.dim() == 5 else z)
        n_patch = 64
        E_t = torch.empty(b, n_patch, device=self.device)
        E_n = torch.empty(b, n_patch, device=self.device)
        for i in range(b):
            t = int(tid[i])
            pt = self.protos[STYLES[t]]
            et = self._energy_one(f[i * n_patch:(i + 1) * n_patch], pt)
            en = torch.stack([self._energy_one(
                f[i * n_patch:(i + 1) * n_patch], self.protos[STYLES[j]])
                for j in range(5) if j != t]).mean(0)
            E_t[i], E_n[i] = et, en
        return E_t, E_n

    def loss(self, z: torch.Tensor, tid: torch.Tensor) -> torch.Tensor:
        """Scalar L_loc for the batch."""
        E_t, E_n = self.energies(z, tid)
        per = E_t + BETA * F.softplus(torch.tensor(MU, device=z.device) + E_t - E_n)
        return per.mean()

    @torch.no_grad()
    def patch_scores(self, z: torch.Tensor, tid: torch.Tensor):
        """Per-patch style gain estimate (B,64): E_neg - E_t (higher = more style room)."""
        E_t, E_n = self.energies(z, tid)
        return (E_n - E_t)
