"""Shared, direction- and style-pair-conditioned latent DiT.

No style encoder / classifier / adversarial loss lives here: style is the target
data distribution. The network only needs to know WHICH pair it is transporting.

Conditioning modes
------------------
`none`        legacy two-domain behaviour: c = time(tau) + direction. No style info.
`decomposed`  c = time(tau) + W[e_a ; e_b] + direction, with five shared style
              embeddings. This is the default: it knows both endpoints and shares
              knowledge across the ten unordered pairs.
`pair`        c = time(tau) + pair_embed[pair_id] + direction. Ablation (exp 09)
              that gives each of the ten pairs its own embedding.

Output heads
------------
`heads_out=1`  one shared output for both drift directions.
`heads_out=2`  separate forward / reverse output projections (exp 07).
`two_head="score_current"`  two heads predicting (v, s) instead of (b+, b-) (exp 10).
"""
import math

import torch
import torch.nn.functional as F
from torch import nn


def time_embed(t, dim):
    half = dim // 2
    freq = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32)
                     / max(half - 1, 1))
    u = 2 * math.pi * t.float().reshape(-1, 1) * freq.reshape(1, -1)
    x = torch.cat([u.sin(), u.cos()], dim=-1)
    return F.pad(x, (0, dim - x.shape[1]))


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class AdaBlock(nn.Module):
    def __init__(self, width, heads, local_conv=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)
        # exp 38: cheap local detail injection on the 8x8 token grid (no token growth).
        self.local_conv = local_conv
        if local_conv:
            self.dw = nn.Conv2d(width, width, 3, padding=1, groups=width)
            nn.init.zeros_(self.dw.weight)
            nn.init.zeros_(self.dw.bias)
            self.side = nn.Parameter(torch.zeros(1))

    def forward(self, x, c, side=None):
        sa, sca, ga, sm, scm, gm = self.ada(c).chunk(6, dim=-1)
        z = modulate(self.norm1(x), sa, sca)
        z = self.attn(z, z, z, need_weights=False)[0]
        x = x + ga.unsqueeze(1) * z
        if self.local_conv:
            B, N, C = x.shape
            g = int(round(N ** 0.5))
            if g * g == N:
                h = x.transpose(1, 2).reshape(B, C, g, g)
                x = x + torch.tanh(self.side) * self.dw(h).reshape(B, C, N).transpose(1, 2)
        x = x + gm.unsqueeze(1) * self.mlp(modulate(self.norm2(x), sm, scm))
        return x


class BidirDiT(nn.Module):
    def __init__(self, channels=32, image_size=16, patch=2, width=256, depth=6, heads=4,
                 num_styles=5, style_cond="none", heads_out=1, two_head="drift",
                 local_conv=False, pair_rank=0):
        super().__init__()
        if image_size % patch != 0:
            raise ValueError("image size must divide patch")
        if style_cond not in ("none", "decomposed", "pair"):
            raise ValueError(f"unknown style_cond {style_cond}")
        if heads_out not in (1, 2):
            raise ValueError("heads_out must be 1 or 2")
        if two_head not in ("drift", "score_current"):
            raise ValueError(two_head)

        self.channels, self.image_size, self.patch = channels, image_size, patch
        self.num_styles, self.style_cond = num_styles, style_cond
        self.heads_out, self.two_head = heads_out, two_head
        self.nt = (image_size // patch) ** 2
        self.width = width

        self.proj = nn.Conv2d(channels, width, kernel_size=patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, self.nt, width))
        nn.init.normal_(self.pos, std=0.02)
        self.time = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.dir = nn.Embedding(2, width)

        if style_cond == "decomposed":
            self.style_emb = nn.Embedding(num_styles, width)
            nn.init.normal_(self.style_emb.weight, std=0.02)
            self.style_mix = nn.Linear(2 * width, width)
            # Start with zero style influence so training begins as the pair-agnostic model.
            nn.init.zeros_(self.style_mix.weight)
            nn.init.zeros_(self.style_mix.bias)
        elif style_cond == "pair":
            n_pairs = num_styles * (num_styles - 1) // 2
            self.pair_emb = nn.Embedding(n_pairs, width)
            nn.init.zeros_(self.pair_emb.weight)

        self.blocks = nn.ModuleList([AdaBlock(width, heads, local_conv) for _ in range(depth)])
        self.fin_norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.fin_ada = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))
        self.fin_out = nn.ModuleList(
            [nn.Linear(width, channels * patch * patch) for _ in range(heads_out)])
        for lin in self.fin_out:
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

        # exp 37: small pair-specific low-rank adapter on the shared trunk.
        self.pair_rank = pair_rank
        if pair_rank > 0:
            n_pairs = num_styles * (num_styles - 1) // 2
            self.lora_down = nn.Embedding(n_pairs, pair_rank)
            self.lora_up = nn.Linear(pair_rank, width, bias=False)
            nn.init.normal_(self.lora_down.weight, std=0.02)
            nn.init.zeros_(self.lora_up.weight)

    # ---- conditioning ------------------------------------------------------ #
    def _pair_id(self, a, b):
        """Canonical unordered pair index for style ids a < b."""
        lo = torch.minimum(a, b)
        hi = torch.maximum(a, b)
        n = self.num_styles
        # number of pairs before row `lo`: lo*(2n-lo-1)/2
        base = lo * (2 * n - lo - 1) // 2
        return base + (hi - lo - 1)

    def _cond(self, tau, direction, style_a, style_b):
        c = self.time(time_embed(tau, self.width)) + self.dir(direction)
        if self.style_cond == "decomposed":
            ea = self.style_emb(style_a)
            eb = self.style_emb(style_b)
            c = c + self.style_mix(torch.cat([ea, eb], dim=-1))
        elif self.style_cond == "pair":
            c = c + self.pair_emb(self._pair_id(style_a, style_b))
        return c

    # ---- forward ----------------------------------------------------------- #
    def forward(self, x, tau, direction, style_a=None, style_b=None, return_both=False):
        B, C, H, W = x.shape
        if C != self.channels or H != self.image_size or W != self.image_size:
            raise ValueError(f"Expected B,{self.channels},{self.image_size},{self.image_size}, got {tuple(x.shape)}")
        tau = tau.flatten()
        direction = direction.flatten().long()

        if self.style_cond == "none":
            c = self.time(time_embed(tau, self.width)) + self.dir(direction)
        else:
            if style_a is None or style_b is None:
                raise ValueError("style_a / style_b are required for a style-conditioned model")
            c = self._cond(tau, direction, style_a.flatten().long(), style_b.flatten().long())

        h = self.proj(x).flatten(2).transpose(1, 2) + self.pos
        if self.pair_rank > 0 and self.style_cond != "none":
            pid = self._pair_id(style_a.flatten().long(), style_b.flatten().long())
            h = h + self.lora_up(self.lora_down(pid)).unsqueeze(1)
        for block in self.blocks:
            h = block(h, c)

        shift, scale = self.fin_ada(c).chunk(2, dim=-1)
        h = modulate(self.fin_norm(h), shift, scale)
        outs = [lin(h) for lin in self.fin_out]
        p = self.patch
        side = self.image_size // p

        def unpatch(o):
            return (o.reshape(B, side, side, p, p, C)
                    .permute(0, 5, 1, 3, 2, 4).reshape(B, C, H, W))

        if return_both:
            return [unpatch(o) for o in outs]
        if self.heads_out == 2:
            # outs[0] = forward drift head, outs[1] = reverse drift head
            sel = (direction.view(-1, 1, 1) == 1)
            o = torch.where(sel, outs[0], outs[1])
            return unpatch(o)
        return unpatch(outs[0])
