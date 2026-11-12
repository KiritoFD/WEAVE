"""WEAVE-VAE V2: 75% Natural + 25% Style-Balanced Sampler.

Implements:
p_train(x) = 0.75 * p_natural(x) + 0.25 * p_balanced(x)
- 75% sampled uniformly from raw dataset (preserving natural frequency).
- 25% sampled uniformly over styles then uniformly over artworks in that style (boosting long-tail coverage).
"""
from __future__ import annotations

import collections
import random
from typing import Iterator, List

import numpy as np
from torch.utils.data import Sampler


class HybridStyleSampler(Sampler[int]):
    """Generates sample indices according to 75% natural + 25% style-balanced distribution."""

    def __init__(
        self,
        dataset,
        p_natural: float = 0.75,
        total_samples: int = 0,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.p_natural = p_natural
        self.num_samples = total_samples if total_samples > 0 else len(dataset)
        self.seed = seed

        # Build natural indices and style lookup
        self.all_indices = list(range(len(dataset)))
        self.style_to_indices = collections.defaultdict(list)

        for i, idx in enumerate(dataset.indices):
            rec = dataset.records_by_idx[idx]
            self.style_to_indices[rec["style_id"]].append(i)

        self.styles = sorted(list(self.style_to_indices.keys()))
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        for _ in range(self.num_samples):
            if self.rng.random() < self.p_natural:
                # 75% Natural
                yield self.rng.choice(self.all_indices)
            else:
                # 25% Balanced Style
                st = self.rng.choice(self.styles)
                yield self.rng.choice(self.style_to_indices[st])
