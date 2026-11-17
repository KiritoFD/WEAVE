"""G2 Query: multi-scale spatial-relational GCL (zero params).

Tokens: per-cell structural descriptors from the Haar LL pyramid (cell value +
x/y gradient magnitude), channel-normalized per cell.
  16x16 grid (256 tokens): full relation matrix A = softmax(q q^T / tau)
  32x32 grid (1024 tokens): windowed relations (5x5 neighbourhoods)
Loss: mean KL( A(z_src) || A(z) ) per token row.

tau is calibrated on TRAIN paintings (plan §3-G2): largest tau in a fixed sweep
such that geometric perturbation moves relations >= 3x as much as appearance
perturbation. No Test750 statistics enter the calibration.
"""
from __future__ import annotations

import torch
import torch.nn.functional as Fn

TAU_SWEEP = (0.25, 0.5, 1.0, 2.0)
WINDOW = 5


def _tokens16(z: torch.Tensor) -> torch.Tensor:
    """(B,4,64,64) -> (B,256,12) structural tokens on the 16x16 grid."""
    ll16 = Fn.avg_pool2d(z, 4)                                  # (B,4,16,16)
    gx = (ll16[..., :, 1:] - ll16[..., :, :-1])                 # (B,4,16,15)
    gy = (ll16[..., 1:, :] - ll16[..., :-1, :])
    gx = Fn.pad(gx, (0, 1))
    gy = Fn.pad(gy, (0, 0, 0, 1))
    mag_x = gx.square().mean(1, keepdim=True).sqrt()            # (B,1,16,16)
    mag_y = gy.square().mean(1, keepdim=True).sqrt()
    tok = torch.cat([ll16, mag_x.repeat(1, 4, 1, 1), mag_y.repeat(1, 4, 1, 1)],
                    1)                                          # (B,12,16,16)
    return tok.reshape(tok.shape[0], tok.shape[1], -1).transpose(1, 2)  # (B,256,12)


def _tokens32(z: torch.Tensor) -> torch.Tensor:
    """(B,4,64,64) -> (B,1024,12) structural tokens on the 32x32 grid."""
    ll32 = Fn.avg_pool2d(z, 2)                                  # (B,4,32,32)
    gx = Fn.pad(ll32[..., :, 1:] - ll32[..., :, :-1], (0, 1))
    gy = Fn.pad(ll32[..., 1:, :] - ll32[..., :-1, :], (0, 0, 0, 1))
    mag_x = gx.square().mean(1, keepdim=True).sqrt()
    mag_y = gy.square().mean(1, keepdim=True).sqrt()
    tok = torch.cat([ll32, mag_x.repeat(1, 4, 1, 1), mag_y.repeat(1, 4, 1, 1)],
                    1)                                          # (B,12,32,32)
    b, c = tok.shape[0], tok.shape[1]
    return tok.reshape(b, c, -1).transpose(1, 2)                # (B,1024,12)


def _relations(tok: torch.Tensor, tau: float) -> torch.Tensor:
    """(B,N,D) -> (B,N,N) row-softmax similarity with temperature tau."""
    qn = Fn.normalize(tok, dim=2)
    sims = qn @ qn.transpose(1, 2) / tau
    return torch.softmax(sims, dim=-1)


def _window_relations(tok: torch.Tensor, tau: float, win: int = WINDOW):
    """(B,N,D) -> (B, n_windows, win*win) row-softmax within windows."""
    b, n, d = tok.shape
    side = int(n ** 0.5)
    qn = Fn.normalize(tok, dim=2).reshape(b, side, side, d)
    rows = []
    step = win // 2
    for ty in range(0, side - win + 1, step):
        for tx in range(0, side - win + 1, step):
            patch = qn[:, ty:ty + win, tx:tx + win, :]        # (B,win,win,D)
            patch = patch.reshape(b, win * win, d)
            sims = patch @ patch.transpose(1, 2) / tau
            rows.append(torch.softmax(sims, dim=-1))
    return torch.stack(rows, 1)                                # (B,nw,win*win)


def _kl_rows(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """KL(p || q) per row, both row-softmax distributions."""
    return (p * (p.add(1e-10).log() - q.add(1e-10).log())).sum(-1)


def relation_gcl_loss(z: torch.Tensor, z_src: torch.Tensor, tau: float,
                      use32: bool = True) -> torch.Tensor:
    """Per-sample multi-scale relational GCL -> (B,).

    16x16 full matrix + 32x32 windowed relations, source row-distributions as
    the (detached) reference, candidate rows keep the gradient to z.
    """
    t16z, t16s = _tokens16(z), _tokens16(z_src.detach())
    a16z, a16s = _relations(t16z, tau), _relations(t16s, tau)
    loss = _kl_rows(a16s, a16z).mean(1)
    if use32:
        w32z, w32s = _window_relations(_tokens32(z), tau), \
            _window_relations(_tokens32(z_src.detach()), tau)
        loss = loss + _kl_rows(w32s, w32z).mean(dim=(1, 2))
    return loss


def g2o_selfsim_loss(z: torch.Tensor, z_src: torch.Tensor) -> torch.Tensor:
    """G2O control: the frozen V5-era self-similarity constraint, mirrored from
    tools/legacy_root/b1_style_operator.py::_content_selfsim."""
    from tools.legacy_root.features import patch_self_similarity

    def pool16(t):  # b1_style_operator._pool_for_selfsim with max_tokens=256
        return Fn.adaptive_avg_pool2d(t.float(), (16, 16))
    sa = patch_self_similarity(pool16(z))
    sb = patch_self_similarity(pool16(z_src.detach()))
    return Fn.l1_loss(sa, sb, reduction="none").mean(dim=tuple(
        range(1, sa.dim())))
