"""Style-Bridge: synthetic cross-style editing pairs with REAL endpoints.

z_tilde = T_r(z): appearance perturbation toward style r using V6 in-domain
subband moments (ported from tools/v6/train_eval_v6.make_coherent_real_target —
that module is not importable on Windows due to unix-only deps).
strength in {0, 0.5, 1.0}.
Endpoint z_1 = z (the real painting). Bridge path on [tau0, 1]:

    z_tau0 = (1-tau0)*eps + tau0*z_tilde
    z_t    = (1-t)/(1-tau0) * z_tau0 + (t-tau0)/(1-tau0) * z
    v*     = (z - z_tau0) / (1-tau0)          # constant along the path

Degenerate check: z_tilde == z  =>  v* == z - eps  (original FM velocity).
"""
from __future__ import annotations

import torch
import torch.nn.functional as Fn

from tools.legacy_root.wavelet import dwt2_haar, idwt2_haar

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo",
          "Ukiyo_e"]


def load_style_targets(path="runs/v6/cache/indomain_style_targets_v2_k64.pt",
                       device="cpu"):
    return torch.load(path, map_location=device, weights_only=False)


def _coherent_target(z_real, style_ids, style_targets, beta):
    """Port of tools/v6/train_eval_v6.make_coherent_real_target.
    Aligns subband palette/energy of z_real toward each row's style_id
    prototype while preserving spatial brushwork. beta=0 -> identity."""
    if beta <= 0.0:
        return z_real
    ll, lh, hl, hh = dwt2_haar(z_real.float())
    ll_out = torch.empty_like(ll)
    lh_out = torch.empty_like(lh)
    hl_out = torch.empty_like(hl)
    hh_out = torch.empty_like(hh)
    for sid, s in enumerate(STYLES):
        mask = (style_ids == sid)
        if not mask.any():
            continue
        st = style_targets["styles"][s]
        dev = z_real.device
        ll_sub = ll[mask]
        m_sub = ll_sub.mean(dim=(2, 3), keepdim=True)
        sd_sub = ll_sub.std(dim=(2, 3), keepdim=True).clamp(min=1e-4)
        ll_norm = (ll_sub - m_sub) / sd_sub
        ll_out[mask] = (1 - beta) * ll_sub + beta * (
            ll_norm * st["ll_sd"].to(dev).view(1, 4, 1, 1)
            + st["ll_mu"].to(dev).view(1, 4, 1, 1))
        for b_sub, b_out, mk, sk in (
                (lh[mask], lh_out, "lh_mu", "lh_sd"),
                (hl[mask], hl_out, "hl_mu", "hl_sd"),
                (hh[mask], hh_out, "hh_mu", "hh_sd")):
            mb = b_sub.mean(dim=(2, 3), keepdim=True)
            sb = b_sub.std(dim=(2, 3), keepdim=True).clamp(min=1e-4)
            bn = (b_sub - mb) / sb
            b_out[mask] = (1 - beta) * b_sub + beta * (
                bn * st[sk].to(dev).view(1, 4, 1, 1)
                + st[mk].to(dev).view(1, 4, 1, 1))
    return idwt2_haar(ll_out, lh_out, hl_out, hh_out).to(z_real.dtype)


def perturb_appearance(z, target_ids, strength, style_targets):
    """z: (B,4,64,64); target_ids: (B,) foreign style; strength scalar or (B,)."""
    dev = z.device
    if isinstance(strength, (int, float)):
        strength = torch.full((z.shape[0],), float(strength), device=dev)
    else:
        strength = strength.to(dev)
    z_tilde = z.clone()
    for sid in range(5):
        m = (target_ids == sid)
        if not m.any():
            continue
        for u in torch.unique(strength[m]):
            mm = m & (strength == u)
            z_tilde[mm] = _coherent_target(
                z[mm].float(),
                torch.full((int(mm.sum()),), sid, device=dev,
                           dtype=torch.long),
                style_targets, float(u))
    return z_tilde.to(z.dtype)


def sample_bridge(z_real, style_id, style_targets, device="cuda", gen=None,
                  p_drop=0.15):
    """One bridge item per row. Returns dict of tensors."""
    B = z_real.shape[0]
    dev = z_real.device
    eps = torch.randn_like(z_real)
    r = (style_id + 1 + torch.randint(0, 4, (B,), device=dev,
                                      generator=gen)) % 5
    strength = torch.tensor([0.0, 0.5, 1.0], device=dev)[
        torch.randint(0, 3, (B,), device=dev, generator=gen)]
    tau0_t = torch.tensor([0.4, 0.7], device=dev)[
        torch.randint(0, 2, (B,), device=dev, generator=gen)]
    z_tilde = perturb_appearance(z_real.float(), r, strength,
                                 style_targets)
    t0 = tau0_t.view(-1, 1, 1, 1)
    z_tau0 = (1 - t0) * eps + t0 * z_tilde
    t = tau0_t + (1 - tau0_t) * torch.rand(B, device=dev, generator=gen)
    span = (1 - tau0_t).clamp_min(1e-6)
    a = ((1 - t) / span).view(-1, 1, 1, 1)
    b = ((t - tau0_t) / span).view(-1, 1, 1, 1)
    z_t = a * z_tau0 + b * z_real
    v_star = (z_real - z_tau0) / span.view(-1, 1, 1, 1)
    drop = torch.rand(B, device=dev) < p_drop
    return {"z_t": z_t, "v_star": v_star, "sid": style_id, "tau": t,
            "drop": drop, "z_tau0": z_tau0, "z_tilde": z_tilde,
            "tau0": tau0_t, "strength": strength}


def bridge_loss(model, batch):
    """MSE(v_theta(z_t, sid, t), v*) with unconditional style drop."""
    dev = batch["z_t"].device
    has_c = "content" in batch and batch["content"] is not None
    kw = {"content": batch["content"]} if has_c else {}
    with torch.autocast(dev.type, dtype=torch.bfloat16,
                        enabled=dev.type == "cuda"):
        v = model(batch["z_t"], batch["sid"], batch["tau"],
                  unconditional=False, **kw).float()
    if batch["drop"].any():
        idx = batch["drop"].nonzero(as_tuple=True)[0]
        kw2 = {"content": batch["content"][idx]} if has_c else {}
        with torch.autocast(dev.type, dtype=torch.bfloat16,
                            enabled=dev.type == "cuda"):
            v[idx] = model(batch["z_t"][idx], batch["sid"][idx],
                           batch["tau"][idx], unconditional=True,
                           **kw2).float()
    return Fn.mse_loss(v, batch["v_star"])
