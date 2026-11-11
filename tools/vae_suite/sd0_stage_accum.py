"""One-shot training-state accumulator shared by all SD0-S10 stages.

The metrics jsonl written every 250 updates contains only the periodic VALID
metrics. Gate/R needs training-side aggregates (mean response loss, response
cosine, amplitude ratio) over the WHOLE stage, so we accumulate them in a
side-channel file that does not disturb the running trainer.
"""
from __future__ import annotations

import json
from collections import defaultdict


class StageAccum:
    def __init__(self):
        self.sums = defaultdict(float)
        self.count = defaultdict(int)
        self.first = None
        self.last = None

    def update(self, acc: dict):
        if self.first is None:
            self.first = {k: float(v) for k, v in acc.items()
                          if isinstance(v, (int, float))}
        for k, v in acc.items():
            if isinstance(v, (int, float)):
                self.sums[k] += float(v)
                self.count[k] += 1
        self.last = {k: float(v) for k, v in acc.items()
                     if isinstance(v, (int, float))}

    def summary(self):
        mean = {k: self.sums[k] / max(self.count[k], 1) for k in self.sums}
        return {"first": self.first, "last": self.last, "mean": mean,
                "n_updates": max(self.count.values()) if self.count else 0}
