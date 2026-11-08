"""Five-style data plumbing: cache loading, shared statistics, pair schedule, samplers.

Design rules from docs/bridgev10.md
-----------------------------------
* All five domains share ONE global mean/std. Per-style standardization is
  forbidden because it would erase real cross-domain statistical differences.
* EDM endpoint statistics are kept per style, and the ten pair moments follow
  analytically with zero cross covariance for independent endpoints.
* Every source of randomness (data index, bridge noise, time sampling, OT,
  initialization) gets its own generator so arms differ only in the intended knob.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .preconditioning import StyleStats


STYLE_NAMES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]


# --------------------------------------------------------------------------- #
@dataclass
class StyleData:
    """Per-style latent pool (already globally standardized) plus raw thumbnails."""

    styles: List[str]
    z: List[torch.Tensor]          # each (n_i, C, H, W) float32 on device, standardized
    thumb: List[torch.Tensor]      # each (n_i, 3, 32, 32) float32 in 0..1
    stats: StyleStats
    counts: List[int]

    @property
    def num_styles(self) -> int:
        return len(self.styles)

    def pairs(self) -> List[Tuple[int, int]]:
        return list(combinations(range(self.num_styles), 2))


def load_caches(paths: Sequence[Path], device: str, style_order: Optional[Sequence[str]] = None) -> StyleData:
    """Load per-style DCAE caches, apply ONE shared global standardization."""
    caches = []
    for p in paths:
        d = torch.load(p, map_location="cpu", weights_only=False)
        for k in ("latent", "thumb", "ae", "scaling_factor"):
            if k not in d:
                raise KeyError(f"{p} missing '{k}'")
        caches.append(d)

    a0 = caches[0]
    for d, p in zip(caches[1:], paths[1:]):
        if d["ae"] != a0["ae"]:
            raise ValueError(f"AE mismatch: {p} uses {d['ae']} vs {a0['ae']}")
        if d["scaling_factor"] != a0["scaling_factor"]:
            raise ValueError(f"scaling_factor mismatch in {p}")
        if tuple(d["latent"].shape[1:]) != tuple(a0["latent"].shape[1:]):
            raise ValueError(f"latent shape mismatch in {p}")

    if style_order is None:
        styles = [d.get("style") or Path(p).stem for d, p in zip(caches, paths)]
    else:
        styles = list(style_order)
        by_name = {(d.get("style") or Path(p).stem): d for d, p in zip(caches, paths)}
        caches = [by_name[s] for s in styles]

    raw = [d["latent"].float() for d in caches]
    both = torch.cat(raw, dim=0)
    gmean = both.mean((0, 2, 3))
    gstd = both.var((0, 2, 3), unbiased=False).clamp_min(1e-8).sqrt()
    del both

    z = [((r - gmean[None, :, None, None]) / gstd[None, :, None, None]).to(device) for r in raw]
    thumb = [(d["thumb"].float().to(device) / 255.0) for d in caches]
    del raw

    stats = StyleStats.from_state_dict({
        "global_mean": gmean,
        "global_std": gstd,
        "style_mean": torch.stack([zi.mean((0, 2, 3)).cpu() for zi in z]),
        "style_var": torch.stack([zi.var((0, 2, 3), unbiased=False).clamp_min(1e-10).cpu() for zi in z]),
    })
    return StyleData(styles=styles, z=z, thumb=thumb, stats=stats,
                     counts=[int(t.shape[0]) for t in z])


# --------------------------------------------------------------------------- #
# pair schedule
# --------------------------------------------------------------------------- #
def pair_schedule(pairs: Sequence[Tuple[int, int]], total_steps: int,
                  gen: torch.Generator) -> List[Tuple[int, int]]:
    """Balanced schedule: every consecutive len(pairs) steps cover all pairs once,
    in an independently shuffled order per cycle."""
    out: List[Tuple[int, int]] = []
    pairs = list(pairs)
    while len(out) < total_steps:
        perm = torch.randperm(len(pairs), generator=gen).tolist()
        out.extend(pairs[i] for i in perm)
    return out[:total_steps]


# --------------------------------------------------------------------------- #
# samplers
# --------------------------------------------------------------------------- #
class PermutationSampler:
    """Balanced sampling without replacement; reshuffles when exhausted.

    Avoids the ~38% duplicate positions of with-replacement sampling when
    batch is close to the pool size (docs/bridgev10.md 1.4).
    """

    def __init__(self, n: int, gen: torch.Generator, device: str = "cpu"):
        if n <= 0:
            raise ValueError("empty pool")
        self.n = n
        self.gen = gen
        self.device = device
        self.perm = torch.randperm(n, generator=gen)
        self.pos = 0

    def draw(self, k: int) -> torch.Tensor:
        out = []
        while k > 0:
            if self.pos >= self.n:
                self.perm = torch.randperm(self.n, generator=self.gen)
                self.pos = 0
            take = min(k, self.n - self.pos)
            out.append(self.perm[self.pos:self.pos + take])
            self.pos += take
            k -= take
        return torch.cat(out)


class WithReplacementSampler:
    def __init__(self, n: int, gen: torch.Generator):
        self.n, self.gen = n, gen

    def draw(self, k: int) -> torch.Tensor:
        return torch.randint(0, self.n, (k,), generator=self.gen)


def make_sampler(kind: str, n: int, gen: torch.Generator) -> object:
    if kind == "perm":
        return PermutationSampler(n, gen)
    if kind == "replacement":
        return WithReplacementSampler(n, gen)
    raise ValueError(f"unknown sampler {kind}")


# --------------------------------------------------------------------------- #
# reproducibility record
# --------------------------------------------------------------------------- #
def record_experiment(experiment_id: str, cfg: dict, extra: dict) -> dict:
    import hashlib

    payload = json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return {
        "experiment_id": experiment_id,
        "config_sha256": hashlib.sha256(payload).hexdigest(),
        "config": cfg,
        **extra,
    }
