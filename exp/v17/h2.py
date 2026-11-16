"""P3: Signed RGB-H2 family H0-H7.

Correct signed-gate formulation (plan §2.4): modify H2 coefficients, then
idwt2(LL2, H2', H1) — NEVER RGB + gamma*H2(RGB).  Real-painting per-style/
direction/edge-group H2 RMS statistics (H1) are computed from the training
pool; the analytic arm (H2arm) applies clamp(target_rms/source_rms-1) signed
gains per position/direction; learned arms H3 (gain-only) / H4 (gain+bias)
are zero-initialised heads supervised by real-H2 restoration of synthetic
five-style spectral corruption — NOT real cross-style pairs, so Cross600
independence carries the generalisation claim.  H5/H6/H7 are mask/label/
identity controls evaluated independently on the full 750.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import OUT, STYLES, save
from exp.v11b.bands import dwt2, idwt2, sobel_mag


@torch.no_grad()
def h1_stats(pool_rgb_dir: Path | None = None, pool_split=None,
             device='cuda', dec=None):
    """Per-style x direction x edge-group H2 RMS statistics + quantiles.

    pool_split: v16 pool_samples result (latents); decoded via S10 on the fly.
    """
    from src.utils import load_s10_decoder, decode_s10
    if dec is None:
        dec = load_s10_decoder(device)
    stats = {s: {'n': 0, 'sum': torch.zeros(9, device=device),
                 'sumsq': torch.zeros(9, device=device),
                 'edge_sum': torch.zeros(9, device=device),
                 'flat_sum': torch.zeros(9, device=device)} for s in range(5)}
    store = {}
    if pool_split is None:
        from .common import read
        pool_split = torch.load(OUT / 'pairs' / 'teacher_pairs_T0.pt',
                                map_location='cpu', weights_only=False) \
            if False else None
    if pool_split is None:
        raise ValueError('need pool_split (latents+sid) for H1 statistics')
    ids = pool_split['train']['ids'] if 'train' in pool_split else \
        list(pool_split['train']['ids'])
    z = pool_split['train']['z'] if 'train' in pool_split else \
        pool_split['train']['z']
    sid = pool_split['train']['sid']
    for st in range(0, len(z), 8):
        rgb = decode_s10(dec, z[st:st + 8].to(device), batch=8).to(device).clamp(0, 1)
        sids = sid[st:st + 8].to(device)
        bands = dwt2(rgb)
        h2 = bands['H2']                              # (B,9,128,128)
        edge = sobel_mag(bands['LL2'])                # (B,1,128,128)
        emax = edge.amax((1, 2, 3), keepdim=True).clamp_min(1e-8)
        edge_m = (edge / emax > .25).float()
        for s in range(5):
            m = sids == s
            if not m.any():
                continue
            h = h2[m]
            e = edge_m[m].expand_as(h)
            per_dir = h.square().mean((2, 3))         # (B,3)
            stats[s]['n'] += int(m.sum())
            stats[s]['sum'] += per_dir.sum(0)
            stats[s]['sumsq'] += per_dir.square().sum(0)
            stats[s]['edge_sum'] += (h.square() * e).mean((2, 3)).sum(0)
            stats[s]['flat_sum'] += (h.square() * (1 - e))                 .mean((2, 3)).sum(0)
    out = {}
    for s in range(5):
        n = max(stats[s]['n'], 1)
        mean = stats[s]['sum'] / n
        var = stats[s]['sumsq'] / n - mean.square()
        out[STYLES[s]] = {'n': stats[s]['n'],
                          'h2_rms_per_direction': [float(x) for x in
                                                   mean.sqrt()],
                          'h2_rms_std': [float(x) for x in var.sqrt()],
                          'edge_rms': [float(x) for x in
                                       (stats[s]['edge_sum'] / n).sqrt()],
                          'flat_rms': [float(x) for x in
                                       (stats[s]['flat_sum'] / n).sqrt()]}
    save(OUT / 'h2' / 'h1_stats.json', out)
    return out


def signed_gain(source_rgb: torch.Tensor, tid: torch.Tensor,
                target_rms: torch.Tensor, gmax=.5):
    """Analytic signed gain (P3-H2arm): per-direction
    gain = clamp(target_rms/source_rms - 1, -gmax, +gmax), tiled across the
    3 colour channels of each direction (H2 has 9 = 3 dirs x 3 colours),
    applied inside the H2 band, then idwt2.  Returns the edited RGB."""
    bands = dwt2(source_rgb)
    h2 = bands['H2']                                    # (B,9,H,W)
    b, c9 = h2.shape[0], h2.shape[1]
    src_rms = h2.view(b, 3, 3, h2.shape[2], h2.shape[3]) \
        .square().mean((2, 3, 4)).sqrt().clamp_min(1e-8)   # (B,3) per direction
    gain = (target_rms.to(source_rgb.device)[tid] / src_rms - 1.)
    gain = gain.clamp(-gmax, gmax)                       # (B,3)
    gain9 = gain.repeat_interleave(3, dim=1).view(-1, 9, 1, 1)
    h2p = h2 * (1 + gain9)
    return idwt2(bands['LL2'], h2p, bands['H1']).clamp(0, 1)


class H2Head(nn.Module):
    """Learned per-position signed gain (H3) / gain+bias (H4) on H2 coeffs.

    Input: H2 coefficients (9 ch) + normalised edge map + target-label onehot.
    Zero-initialised output => exact identity (gamma=0, beta=0) at start.
    gamma bounded to [-.5,.5] via tanh; beta trust-scaled by band RMS.
    """

    def __init__(self, bias=True, width=64):
        super().__init__()
        self.bias = bias
        ch_in = 9 + 1 + 5
        self.net = nn.Sequential(
            nn.Conv2d(ch_in, width, 3, padding=1), nn.SiLU(True),
            nn.Conv2d(width, width, 3, padding=1), nn.SiLU(True),
            nn.Conv2d(width, 18 if bias else 9, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h2: torch.Tensor, edge: torch.Tensor, tid: torch.Tensor,
                beta_trust: torch.Tensor | None = None):
        onehot = F.one_hot(tid, 5).float()[:, :, None, None] \
            .expand(-1, 5, h2.shape[2], h2.shape[3])
        if beta_trust is None:
            beta_trust = torch.ones(h2.shape[0], device=h2.device)
        x = torch.cat([h2, edge, onehot], 1)
        o = self.net(x)
        gamma = .5 * torch.tanh(o[:, :9])
        out = h2 * (1 + gamma)
        if self.bias:
            beta = o[:, 9:18] * 0  # trust-scaled below by the caller
            if beta_trust is not None:
                beta = o[:, 9:18] * beta_trust.view(-1, 1, 1, 1)
            out = out + beta
        return out, gamma


def corruption_batch(rgb: torch.Tensor, tid: torch.Tensor,
                     stats: dict, device='cuda', gen=None):
    """Synthesise five-style spectral corruption: resample H2 band energies to
    a WRONG style's statistics -> H2_corrupt; ground truth stays H2_real."""
    bands = dwt2(rgb)
    h2 = bands['H2']
    rms = h2.square().mean((2, 3)).sqrt().clamp_min(1e-8)
    wrong = (tid + torch.randint(1, 5, (len(tid),),
                                 generator=gen).to(tid.device)) % 5
    tgt = torch.stack([
        torch.tensor([stats[STYLES[int(w)]]['h2_rms_per_direction'][d]
                      for d in range(3)], device=device)
        for w in wrong.tolist()])
    scale = (tgt / rms).clamp(.25, 4.).repeat_interleave(3, dim=1) \
        .view(-1, 9, 1, 1)
    h2_corrupt = h2 * scale
    return h2, h2_corrupt, bands
