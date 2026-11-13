"""V6 real style-distribution supervision, self-contained for exp/v8.

Copied semantics from tools/v6/train_eval_v6.py::compute_style_distribution_loss
(in-domain multi-scale Haar moment + covariance + patch-SWD). The frozen
per-style targets come from runs/v6/cache/indomain_style_targets_v2_k1024.pt
(training-pool latents only — same provenance as V6, no eval leakage).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from wavelet import dwt2_haar

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
TARGETS_CACHE = "runs/v6/cache/indomain_style_targets_v2_k1024.pt"


def load_style_targets(device="cuda"):
    from pathlib import Path
    p = Path(__file__).resolve().parents[2] / TARGETS_CACHE
    return torch.load(p, map_location="cpu", weights_only=False)


def style_distribution_loss(z_pred: torch.Tensor, tgt_ids: torch.Tensor,
                            style_targets: dict) -> torch.Tensor:
    """Moment + covariance + patch-SWD style loss vs real painting targets."""
    dev = z_pred.device
    ll, lh, hl, hh = dwt2_haar(z_pred.float())
    ll2, _, _, _ = dwt2_haar(ll)
    R_ll = style_targets["R_ll"].to(dev)
    R_hf = style_targets["R_hf"].to(dev)
    B = z_pred.shape[0]
    total = z_pred.new_tensor(0.0)
    for sid, s in enumerate(STYLES):
        mask = tgt_ids == sid
        if not mask.any():
            continue
        st = style_targets["styles"][s]
        ll_s, lh_s, hl_s, hh_s = ll[mask], lh[mask], hl[mask], hh[mask]
        ll2_s = ll2[mask]
        l_mom = (
            F.mse_loss(ll_s.mean(dim=(2, 3)), st["ll_mu"].to(dev).unsqueeze(0).expand(ll_s.shape[0], -1))
            + F.mse_loss(ll_s.std(dim=(2, 3)), st["ll_sd"].to(dev).unsqueeze(0).expand(ll_s.shape[0], -1))
            + F.mse_loss(lh_s.std(dim=(2, 3)), st["lh_sd"].to(dev).unsqueeze(0).expand(lh_s.shape[0], -1))
            + F.mse_loss(hl_s.std(dim=(2, 3)), st["hl_sd"].to(dev).unsqueeze(0).expand(hl_s.shape[0], -1))
            + F.mse_loss(hh_s.std(dim=(2, 3)), st["hh_sd"].to(dev).unsqueeze(0).expand(hh_s.shape[0], -1))
            + F.mse_loss(ll2_s.std(dim=(2, 3)), st["ll2_sd"].to(dev).unsqueeze(0).expand(ll2_s.shape[0], -1))
        )
        flat = ll_s.reshape(ll_s.shape[0], 4, -1)
        flat = flat - flat.mean(dim=2, keepdim=True)
        cov = torch.bmm(flat, flat.transpose(1, 2)) / max(flat.shape[2] - 1, 1)
        l_cov = F.mse_loss(cov.mean(dim=0), st["ll_cov"].to(dev))
        pat_ll = F.unfold(ll_s, kernel_size=4, stride=4).permute(0, 2, 1).reshape(-1, 64)
        proj_ll, _ = torch.sort(pat_ll @ R_ll.t(), dim=0)
        q_idx = torch.linspace(0, proj_ll.shape[0] - 1, 128, device=dev).long()
        l_swd_ll = F.l1_loss(proj_ll[q_idx], st["qgrid_ll"].to(dev))
        hf_cat = torch.cat([lh_s, hl_s, hh_s], dim=1)
        pat_hf = F.unfold(hf_cat, kernel_size=4, stride=4).permute(0, 2, 1).reshape(-1, 192)
        proj_hf, _ = torch.sort(pat_hf @ R_hf.t(), dim=0)
        q_idx_h = torch.linspace(0, proj_hf.shape[0] - 1, 128, device=dev).long()
        l_swd_hf = F.l1_loss(proj_hf[q_idx_h], st["qgrid_hf"].to(dev))
        sub = l_mom + 0.5 * l_cov + 0.5 * (l_swd_ll + l_swd_hf)
        total = total + sub * (mask.sum().float() / B)
    return total
