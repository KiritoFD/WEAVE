"""Losses for the style-friendly VAE (experiments A / B / C).

A: reconstruction only
       L = L1(x, xhat) + lam_p * LPIPS512 + beta * KL
B: + style organisation, applied ONLY to the 8-dim coordinate s(z)
       L += lam_s * (CE(StyleHead(s), y) + SupCon(s, y) + crop consistency)
C: + editability through the statistics swap
       L += lam_e * ||s(z~) - sg(s(z_b))||_1 + lam_c * ||P(c(z~)) - sg(P(c(z_a)))||_1

KL convention is explicit: diffusers' AutoencoderKL uses sum-over-dims then
mean-over-batch, with a small weight (~1e-6 for SD). Both reductions are returned
so the choice is auditable rather than inherited silently.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import style_content, style_swap


def kl_divergence(post):
    """Return (kl_sum_per_sample_mean, kl_elementwise_mean).

    kl_sum: 0.5*sum(exp(logvar)+mean^2-1-logvar) per sample, averaged over batch
            -- this is the diffusers convention.
    kl_elem: the same quantity averaged over every element (smaller by 4*64*64).
    """
    mean = post.mean
    logvar = post.logvar
    per = 0.5 * (logvar.exp() + mean.pow(2) - 1.0 - logvar)
    return per.sum(dim=[1, 2, 3]).mean(), per.mean()


def supcon(s, labels, temperature=0.1):
    """Supervised contrastive loss on the 8-dim style coordinate."""
    z = F.normalize(s, dim=-1)
    sim = z @ z.t() / temperature
    n = s.shape[0]
    eye = torch.eye(n, device=s.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, float("-inf"))
    pos_mask = (labels[:, None] == labels[None, :]) & ~eye
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    # mask BEFORE summing: log_prob is -inf off-positive and (-inf)*0 == nan
    log_prob = torch.where(pos_mask, log_prob, torch.zeros_like(log_prob))
    pos_cnt = pos_mask.sum(1)
    valid = pos_cnt > 0
    if valid.sum() == 0:
        return s.new_zeros(())
    loss = -(log_prob.sum(1)[valid] / pos_cnt[valid])
    return loss.mean()


def crop_consistency(s1, s2):
    """Same artwork, two crops -> the style coordinate should not move."""
    return F.l1_loss(s1, s2.detach())


def lowres_content(c, size=8):
    """Coarse content layout: average-pool the normalised content to 8x8."""
    return F.adaptive_avg_pool2d(c, size)


def edit_loss(model, x_a, x_b, lam_c=0.5, detach_targets=True, use_ckpt=False):
    """Latent swap + re-encode cycle consistency (experiment C).

    z_ab = sigma(z_b) * c(z_a) + mu(z_b)   -> y_ab = D(z_ab) -> z~ = E(y_ab)
    The output should carry b's style coordinate while keeping a's coarse layout.

    This alone does NOT guarantee a real artistic style change (the model could
    satisfy it with invisible perturbations), so it must be judged by the
    independent image-domain metrics, never by this loss value.
    """
    post_a = model.encode(x_a, use_ckpt=use_ckpt)
    post_b = model.encode(x_b, use_ckpt=use_ckpt)
    z_a = post_a.mean
    z_b = post_b.mean
    z_ab = style_swap(z_a, z_b)
    y_ab = model.decode(z_ab)
    post_ab = model.encode(y_ab, use_ckpt=use_ckpt)
    z_t = post_ab.mean

    s_t, c_t, _, _ = style_content(z_t)
    s_b, c_a, _, _ = style_content(z_b)
    s_a, _, _, _ = style_content(z_a)

    tgt_s = s_b.detach() if detach_targets else s_b
    tgt_c = lowres_content(c_a).detach() if detach_targets else lowres_content(c_a)
    l_style = F.l1_loss(s_t, tgt_s)
    l_content = F.l1_loss(lowres_content(c_t), tgt_c)
    return l_style + lam_c * l_content, {
        "edit_style": float(l_style.detach()),
        "edit_content": float(l_content.detach()),
        "swap_s_shift": float((s_t - s_a).abs().mean().detach()),
    }


class StyleVAELoss(nn.Module):
    """Bundles the reconstruction term and the optional B/C terms."""

    def __init__(self, lpips_fn=None, lam_p=0.5, beta=1e-6, kl_reduction="sum"):
        super().__init__()
        self.lpips_fn = lpips_fn
        self.lam_p = lam_p
        self.beta = beta
        self.kl_reduction = kl_reduction

    def forward(self, x, y, post, lpips_size=512):
        l1 = (y.float() - x).abs().mean()
        parts = {"l1": float(l1.detach())}
        total = l1
        if self.lpips_fn is not None:
            if lpips_size and lpips_size != y.shape[-1]:
                ys = F.interpolate(y.float(), lpips_size, mode="bilinear",
                                   align_corners=False)
                xs = F.interpolate(x, lpips_size, mode="bilinear",
                                   align_corners=False)
            else:
                ys, xs = y.float(), x
            l_p = self.lpips_fn(ys, xs).mean()
            total = total + self.lam_p * l_p
            parts["lpips"] = float(l_p.detach())
        kl_sum, kl_elem = kl_divergence(post)
        kl = kl_sum if self.kl_reduction == "sum" else kl_elem
        total = total + self.beta * kl
        parts["kl_sum"] = float(kl_sum.detach())
        parts["kl_elem"] = float(kl_elem.detach())
        return total, parts
