"""Style-Rank data pipeline: RAM-resident images + full-coverage epoch sampler.

Design (per spec):
  * Every epoch traverses the whole training pool exactly once (shuffle, no replacement).
  * Microbatch = 3 styles x 2 images (different artists) = 6 images.
  * All 6 images act as anchors: partner = same-style image, negative = content-closest
    among the other 4 images (precomputed DINO cosine, no online FAISS).
  * Style queues are refilled only after being exhausted; odd tails are carried over
    and recorded, never silently dropped.
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


# ---------------------------------------------------------------- RAM dataset
class RamStyleDataset(Dataset):
    """Keeps the selected images in a contiguous uint8 RAM array.

    __getitem__ returns {'image': float32 CHW in [-1,1], 'pos': local index}.
    """

    def __init__(self, manifest_path, raw_indices, image_size=512, preload=True,
                 style_ids=None):
        man = json.loads(Path(manifest_path).read_text())
        self.mmap = np.memmap(man["bin_path"], dtype=np.uint8, mode="r",
                              shape=tuple(man["shape"]))
        self.raw = list(raw_indices)
        self.size = image_size
        self.style_ids = None if style_ids is None else np.asarray(style_ids, dtype=np.int64)
        self.ram = None
        if preload:
            n = len(self.raw)
            self.ram = np.empty((n, image_size, image_size, 3), dtype=np.uint8)
            for i, ri in enumerate(self.raw):
                self.ram[i] = self.mmap[ri]
                if (i + 1) % 10000 == 0:
                    print(f"    [RAM] {i+1}/{n}", flush=True)
            print(f"[RAM] {n} images resident "
                  f"({self.ram.nbytes/1024**3:.1f} GB)", flush=True)

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, i):
        arr = self.ram[i] if self.ram is not None else self.mmap[self.raw[i]]
        t = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).float()
        t = t.div_(127.5).sub_(1.0)
        sid = -1 if self.style_ids is None else int(self.style_ids[i])
        return {"image": t, "pos": i, "style_id": sid}


# ---------------------------------------------------------------- sampler
def _interleave_by_artist(items, rng):
    """Order (pos, artist) so consecutive entries come from different artists."""
    by_art = defaultdict(list)
    for pos, art in items:
        by_art[art].append(pos)
    arts = list(by_art.keys())
    rng.shuffle(arts)
    for a in arts:
        rng.shuffle(by_art[a])
    out = []
    while True:
        progressed = False
        for a in arts:
            if by_art[a]:
                out.append(by_art[a].pop())
                progressed = True
        if not progressed:
            break
    return out


class EpochCoverageSampler(Sampler):
    """Yields n_styles*2 local positions per microbatch; one full pass per epoch."""

    def __init__(self, style_to_items, seed=42, n_styles=3):
        self.style_to_items = {s: list(v) for s, v in style_to_items.items()}
        self.styles = sorted(self.style_to_items.keys())
        self.seed = seed
        self.n_styles = n_styles
        self.epoch = 0
        self.n_slots = sum(math.ceil(len(v) / 2) for v in self.style_to_items.values())
        self.n_micro = math.ceil(self.n_slots / n_styles)
        self.n_images = sum(len(v) for v in self.style_to_items.values())
        print(f"[EpochSampler] {len(self.styles)} styles | {self.n_images} images | "
              f"{self.n_slots} pair-slots | {self.n_micro} microbatches/epoch "
              f"({n_styles} styles x 2 = {n_styles*2} images)", flush=True)

    def __len__(self):
        return self.n_micro

    def _build_epoch(self):
        rng = random.Random(self.seed + self.epoch * 7919)
        queues = {s: _interleave_by_artist(items, rng) for s, items in self.style_to_items.items()}

        slots = []
        for s in self.styles:
            k = math.ceil(len(queues[s]) / 2)
            slots += [(s, j) for j in range(k)]
        rng.shuffle(slots)

        # pad to a multiple of n_styles so no slot is left unconsumed
        pad = (-len(slots)) % self.n_styles
        tail_fill = 0
        if pad:
            extra = rng.sample(slots, pad)
            slots += extra
            tail_fill = pad

        batches, odd_tail = [], 0
        i = 0
        L = len(slots)
        while i < L:
            grp = [slots[i]]
            i += 1
            while len(grp) < self.n_styles and i < L:
                have = {g[0] for g in grp}
                j = i
                while j < L and slots[j][0] in have:
                    j += 1
                if j >= L:
                    break
                slots[i], slots[j] = slots[j], slots[i]
                grp.append(slots[i])
                i += 1
            if len(grp) < self.n_styles:
                break
            pos6 = []
            for s, jj in grp:
                q = queues[s]
                a = q[2 * jj]
                if 2 * jj + 1 < len(q):
                    b = q[2 * jj + 1]
                else:
                    b = a                      # odd tail: single image repeated
                    odd_tail += 1
                pos6 += [a, b]
            batches.append(pos6)
        self.last_epoch_stats = {"odd_tail_repeats": odd_tail, "tail_fill_slots": tail_fill,
                                 "batches": len(batches)}
        return batches

    def __iter__(self):
        while True:
            self.epoch += 1
            for b in self._build_epoch():
                yield b


# ---------------------------------------------------------------- negative queue
class NegQueue:
    """FIFO queue of detached style embeddings used as cross-style negatives.

    Stores (style embedding, static DINO content feature, dataset position, style).
    Negatives are content-matched (|c_an - c_ap| < tol), drawn from distinct styles,
    and prefer the most recent `recent` slots; older slots are used only to fill the
    quota, never at the cost of the content-match constraint.
    """

    def __init__(self, size, k, device, tol=0.05, recent=1536):
        self.size = size
        self.k = k
        self.tol = tol
        self.recent = recent
        self.device = device
        self.qu = torch.zeros(size, k, device=device)
        self.qd = torch.zeros(size, k, device=device)
        self.qs = torch.full((size,), -1, dtype=torch.long, device=device)
        self.qr = torch.full((size,), -1, dtype=torch.long, device=device)
        self.ptr = 0
        self.n = 0

    @torch.no_grad()
    def push(self, u, dino, style, raw_pos):
        B = u.shape[0]
        for i in range(B):
            p = self.ptr
            self.qu[p] = u[i].detach()
            self.qd[p] = dino[i]
            self.qs[p] = style[i]
            self.qr[p] = raw_pos[i]
            self.ptr = (self.ptr + 1) % self.size
            self.n = min(self.n + 1, self.size)

    def _recency(self):
        """0 = newest slot, increasing with age (in pushes)."""
        idx = torch.arange(self.n, device=self.device)
        return (self.ptr - 1 - idx) % self.size

    def pick(self, u, dino, style, K=4):
        """Returns (s_ap, s_an, valid, match_frac).

        u:(B,k) dino:(B,k) style:(B,). s_an:(B,K), valid:(B,K)
        """
        B = u.shape[0]
        dev = u.device
        ar = torch.arange(B, device=dev)
        partner = ar ^ 1
        s_ap = (u * u[partner]).sum(-1)                       # (B,)
        c_ap = (dino * dino[partner]).sum(-1)                 # (B,)

        if self.n == 0:
            return s_ap, s_ap.detach().unsqueeze(1).expand(B, K), \
                   torch.zeros(B, K, dtype=torch.bool, device=dev), 0.0

        qu = self.qu[:self.n]
        qd = self.qd[:self.n]
        qs = self.qs[:self.n]
        age = self._recency()                                  # (n,)

        c_an = dino @ qd.t()                                   # (B,n)
        gap = (c_an - c_ap.unsqueeze(1)).abs()                 # (B,n)

        ok_content = gap < self.tol
        ok_style = qs.unsqueeze(0) != style.unsqueeze(1)       # (B,n)
        is_recent = (age < self.recent).unsqueeze(0)           # (1,n)

        # soft recency preference: all recent candidates rank before any stale one
        score = gap + 10.0 * (~is_recent).float()
        mask = ok_content & ok_style

        chosen = []
        chosen_styles = torch.full((B, K), -1, dtype=torch.long, device=dev)
        valid = torch.zeros(B, K, dtype=torch.bool, device=dev)
        cur = mask.clone()
        for j in range(K):
            sc = score.masked_fill(~cur, float("inf"))
            best = sc.argmin(1)
            ok = torch.isfinite(sc[ar, best])
            valid[:, j] = ok
            chosen.append(best)
            st = qs[best]
            chosen_styles[:, j] = st
            cur &= (qs.unsqueeze(0) != st.unsqueeze(1))        # distinct styles
            cur &= (torch.arange(self.n, device=dev).unsqueeze(0) != best.unsqueeze(1))

        idx = torch.stack(chosen, 1)                           # (B,K)
        s_an = torch.einsum("bk,bjk->bj", u, qu[idx])          # (B,K)

        n_ok = int(valid.sum())
        match = float((gap[ar.repeat_interleave(K).reshape(B, K),
                          idx][valid] < self.tol).float().mean()) if n_ok else 0.0
        return s_ap, s_an, valid, match

    @torch.no_grad()
    def sample_slots(self, n_sample=32):
        """Random filled slots for staleness monitoring: (slot_idx, raw_pos)."""
        if self.n == 0:
            return None, None
        idx = torch.randint(0, self.n, (min(n_sample, self.n),), device=self.device)
        return idx, self.qr[idx]


def pick_negatives(dino_b):
    """In-batch variant (used when microbatch holds >= 2 styles)."""
    B = dino_b.shape[0]
    dev = dino_b.device
    S = dino_b @ dino_b.t()                                # (B,B)
    ar = torch.arange(B, device=dev)
    partner = ar ^ 1
    c_ap = S[ar, partner]                                  # (B,)

    valid = torch.ones(B, B, dtype=torch.bool, device=dev)
    valid[ar, ar] = False
    valid[ar, partner] = False
    gaps = (S - c_ap.unsqueeze(1)).abs()
    gaps = gaps.masked_fill(~valid, float("inf"))
    neg = gaps.argmin(1)
    c_an = S[ar, neg]
    return partner, neg, c_ap, c_an, float((gaps[ar, neg] < 0.05).float().mean())


def verify_epoch_coverage(style_to_items, seed=42, n_styles=3, verbose=True):
    """Simulate one epoch and report exposure counts."""
    smp = EpochCoverageSampler(style_to_items, seed, n_styles=n_styles)
    smp.epoch = 0
    batches = smp._build_epoch()
    counts = np.zeros(sum(len(v) for v in style_to_items.values()), dtype=np.int64)
    for b in batches:
        for p in b:
            counts[p] += 1
    n = len(counts)
    res = {
        "images": int(n),
        "microbatches": len(batches),
        "updates": len(batches) // 8,
        "covered": int((counts >= 1).sum()),
        "coverage_pct": round(float((counts >= 1).mean()) * 100, 3),
        "count_min": int(counts.min()),
        "count_max": int(counts.max()),
        "count_mean": round(float(counts.mean()), 3),
        "n_zero": int((counts == 0).sum()),
        "n_gt1": int((counts > 1).sum()),
        "odd_tail_repeats": smp.last_epoch_stats["odd_tail_repeats"],
        "tail_fill_slots": smp.last_epoch_stats.get("tail_fill_slots", 0),
    }
    if verbose:
        print(json.dumps(res, indent=2), flush=True)
    return res, counts
