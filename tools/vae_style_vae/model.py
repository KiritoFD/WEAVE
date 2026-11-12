"""Style-friendly VAE: A0 Encoder + S10 Decoder with an explicit statistics split.

The latent is 4x64x64. We split it exactly (no information loss) into

    mu(z)    = z.mean((2,3))            (B,4)
    logs(z)  = log(z.std((2,3)) + eps)  (B,4)
    s(z)     = [mu, logs]               (B,8)   <- candidate STYLE coordinate
    c(z)     = (z - mu) / (sigma + eps) (B,4,64,64) <- candidate CONTENT coordinate
    z        = sigma * c + mu           (exact reconstruction)

This is the AdaIN channel-statistics split, chosen because it is *directly
editable*: swapping style between two latents is one line, and the decoder keeps
its original 4x64x64 interface.

The 8-dim capacity is a hypothesis under test, not an assumption. If B succeeds
and C fails, the documented fallback is z=(z_c, s) with a 16/32-dim style code
and zero-init FiLM.
"""
import torch
import torch.nn as nn
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

EPS = 1e-5


def style_content(z, eps=EPS):
    """Split a latent into (s, c, mu, sigma). s is (B,8), c is (B,4,64,64)."""
    mu = z.mean(dim=(2, 3))                       # (B,4)
    sigma = z.std(dim=(2, 3), unbiased=False)     # (B,4)
    logs = torch.log(sigma + eps)
    s = torch.cat([mu, logs], dim=1)              # (B,8)
    c = (z - mu[:, :, None, None]) / (sigma[:, :, None, None] + eps)
    return s, c, mu, sigma


def recombine(c, mu, sigma, eps=EPS):
    """z = sigma * c + mu. Inverse of style_content up to the eps."""
    return sigma[:, :, None, None] * c + mu[:, :, None, None]


def style_swap(z_a, z_b, eps=EPS):
    """z_{a->b}: keep a's normalised content, take b's statistics."""
    _, c_a, _, _ = style_content(z_a, eps)
    _, _, mu_b, sig_b = style_content(z_b, eps)
    return recombine(c_a, mu_b, sig_b, eps)


class StyleVAE(nn.Module):
    """Frozen-or-trainable A0 encoder + S10 decoder, exposing the s/c split.

    post_quant_conv belongs to the decoder side and is applied inside forward(),
    matching AutoencoderKL.decode()'s contract.
    """

    def __init__(self, vae, decoder, post_quant_conv=None):
        super().__init__()
        self.encoder = vae.encoder
        self.quant_conv = vae.quant_conv
        self.post_quant_conv = post_quant_conv if post_quant_conv is not None \
            else vae.post_quant_conv
        self.decoder = decoder
        self._frozen_pqc = post_quant_conv is None

    def encode(self, x, use_ckpt=False):
        """Return a DiagonalGaussianDistribution (posterior over the latent).

        diffusers' Encoder has NO gradient-checkpointing support (verified: no
        enable_gradient_checkpointing / _set_gradient_checkpointing), so when the
        encoder trains at 512px we wrap its whole forward manually. This trades
        one extra encoder forward in the backward pass for not storing the full
        activation chain, which is what pushes bs=4 over 22 GB.
        """
        if use_ckpt and self.training:
            from torch.utils.checkpoint import checkpoint
            h = checkpoint(self.encoder, x, use_reentrant=False)
        else:
            h = self.encoder(x)
        return DiagonalGaussianDistribution(self.quant_conv(h))

    def decode(self, z):
        return self.decoder(self.post_quant_conv(z))

    def forward(self, x, sample=True):
        post = self.encode(x)
        z = post.sample() if sample else post.mean
        return self.decode(z), z, post


def freeze_encoder(module):
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_encoder(model, mode="full"):
    """mode: 'frozen' | 'partial' | 'full', applied to a StyleVAE.

    partial = last two down_blocks + mid_block + output convs + quant_conv, i.e.
    the layers that decide the latent's global statistics. Measured earlier as
    30.7M trainable params vs 34.2M for the full encoder.
    """
    enc = model.encoder
    freeze_encoder(enc)
    freeze_encoder(model.quant_conv)
    if mode == "frozen":
        return []
    if mode == "partial":
        targets = [enc.down_blocks[2], enc.down_blocks[3], enc.mid_block,
                   enc.conv_norm_out, enc.conv_out, model.quant_conv]
        names = ["down_blocks.2", "down_blocks.3", "mid_block",
                 "conv_norm_out", "conv_out", "quant_conv"]
    else:
        targets = [enc, model.quant_conv]
        names = ["encoder", "quant_conv"]
    for m in targets:
        for p in m.parameters():
            p.requires_grad = True
    return names


class StyleHead(nn.Module):
    """Tiny linear head on the 8-dim style coordinate only.

    Deliberately trivial: if style cannot be read out of s(z) with a linear
    probe, that is the finding, not something to paper over with capacity.
    """

    def __init__(self, n_classes, s_dim=8):
        super().__init__()
        self.fc = nn.Linear(s_dim, n_classes)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, s):
        return self.fc(s)
