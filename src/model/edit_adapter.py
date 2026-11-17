"""Small zero-output adapter for distilling a bounded latent edit operator."""
import torch
from torch import nn
import torch.nn.functional as F


class EditResidualAdapter(nn.Module):
    def __init__(self, width=64, blocks=3):
        super().__init__()
        self.width, self.blocks = width, blocks
        self.input = nn.Conv2d(12, width, 3, padding=1)
        self.style = nn.Embedding(5, width)
        self.body = nn.ModuleList([
            nn.ModuleDict({
                'norm': nn.GroupNorm(8, width),
                'first': nn.Conv2d(width, width, 3, padding=1),
                'second': nn.Conv2d(width, width, 3, padding=1),
                'film': nn.Linear(width, width * 2),
            }) for _ in range(blocks)
        ])
        self.output = nn.Conv2d(width, 4, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, base, content, target):
        h = self.input(torch.cat([base, content, base - content], 1))
        cond = self.style(target)
        for layer in self.body:
            scale, shift = layer['film'](cond).chunk(2, -1)
            v = layer['norm'](layer['first'](F.silu(h)))
            v = v * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
            h = h + layer['second'](F.silu(v))
        return self.output(F.silu(h))
