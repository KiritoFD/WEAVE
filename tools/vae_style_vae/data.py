"""Data for style-friendly VAE training: real RGB, verified split, no latent cache.

The old 78,661 latent cache is INVALID here because the encoder is being updated.
Images are read from the 512px memmap and encoded on the fly.

Split is the same manifest-filtered pool verified earlier:
    train = {r : r["in_train"] and r["raw_index"] not in val_indices}  -> 78,661
    val   = the frozen 2,048
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

MANIFEST = Path("/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")


def load_split(manifest_path=MANIFEST):
    m = json.loads(Path(manifest_path).read_text())
    val = set(m["val_indices"])
    recs = m["records"]
    train = [r for r in recs if r["in_train"] and r["raw_index"] not in val]
    valrec = [r for r in recs if r["raw_index"] in val]
    return m, train, valrec


class StyleImageDS(Dataset):
    """Serves (image, style_id, artist_id, raw_index) for the training pool."""

    def __init__(self, records, manifest, indices=None):
        self.m = manifest
        self.mmap = np.memmap(manifest["bin_path"], dtype=np.uint8, mode="r",
                              shape=tuple(manifest["shape"]))
        self.recs = records
        self.indices = list(range(len(records))) if indices is None else indices
        self.by_style = defaultdict(list)
        self.by_artist = defaultdict(list)
        for pos in self.indices:
            r = self.recs[pos]
            self.by_style[r["style_id"]].append(pos)
            self.by_artist[r.get("artist_id", -1)].append(pos)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        pos = self.indices[i]
        r = self.recs[pos]
        arr = np.ascontiguousarray(self.mmap[r["raw_index"]])
        x = torch.from_numpy(arr).permute(2, 0, 1).float().div_(127.5).sub_(1.0)
        return {"x": x, "style_id": int(r["style_id"]),
                "artist_id": int(r.get("artist_id", -1)),
                "raw_index": int(r["raw_index"]), "pos": int(pos)}


class BalancedStyleBatchSampler(torch.utils.data.Sampler):
    """Yields batches of (n_styles x per_style) dataset indices.

    Why: experiment B needs same-style/different-artist positives and experiment C
    needs two different styles in the same batch. Putting both INSIDE the batch
    avoids extra encoder passes, which is what OOMed at bs=4 with three encodes.

    Within a style group we prefer distinct artists so the contrastive positives
    are not the same artwork (or the same hand) twice.
    """

    def __init__(self, ds, n_styles=2, per_style=2, seed=0):
        self.ds = ds
        self.n_styles = n_styles
        self.per_style = per_style
        self.rng = np.random.default_rng(seed)
        self.by_style_artist = defaultdict(lambda: defaultdict(list))
        for i in range(len(ds)):
            r = ds.recs[ds.indices[i]]
            self.by_style_artist[r["style_id"]][r.get("artist_id", -1)].append(i)
        self.styles = sorted(s for s, a in self.by_style_artist.items()
                             if sum(len(v) for v in a.values()) >= per_style)
        self._len = (len(ds) // (n_styles * per_style)) * n_styles

    def __len__(self):
        return self._len

    def __iter__(self):
        for _ in range(self._len):
            yield self._one_batch()

    def _one_batch(self):
        picked = self.rng.choice(len(self.styles), self.n_styles, replace=False)
        out = []
        for si in picked:
            st = self.styles[int(si)]
            arts = [a for a, v in self.by_style_artist[st].items() if v]
            self.rng.shuffle(arts)
            chosen = []
            for a in arts:
                if len(chosen) >= self.per_style:
                    break
                lst = self.by_style_artist[st][a]
                chosen.append(int(lst[self.rng.integers(len(lst))]))
            while len(chosen) < self.per_style:      # tiny styles fallback
                lst = self.by_style_artist[st][arts[0]]
                chosen.append(int(lst[self.rng.integers(len(lst))]))
            out += chosen
        return out


class PairSampler:
    """Builds the supervision groups experiment B/C need.

    For a batch of anchors it returns:
      pos_idx : same style, DIFFERENT artist (so the pair is not trivially the
                same artwork) - pushes all works of a style together.
      neg_idx : different style - the standard SupCon negatives.

    Artist disjointness matters: same-artist pairs share palette and handling, so
    pulling them together would let the model satisfy the loss without learning
    anything about style.
    """

    def __init__(self, ds, rng_seed=0):
        self.ds = ds
        self.rng = np.random.default_rng(rng_seed)
        self.by_style_artist = defaultdict(lambda: defaultdict(list))
        for i in range(len(ds)):
            r = ds.recs[ds.indices[i]]
            self.by_style_artist[r["style_id"]][r.get("artist_id", -1)].append(i)
        self.styles = [s for s in self.by_style_artist
                       if len(self.by_style_artist[s]) >= 2]
        self.style_list = list(self.by_style_artist)

    def _same_style_other_artist(self, style_id, artist_id):
        arts = self.by_style_artist.get(style_id, {})
        cands = [a for a in arts if a != artist_id and arts[a]]
        if not cands:
            return None
        a = cands[self.rng.integers(len(cands))]
        lst = arts[a]
        return int(lst[self.rng.integers(len(lst))])

    def _other_style(self, style_id):
        others = [s for s in self.style_list if s != style_id]
        if not others:
            return None
        s = others[self.rng.integers(len(others))]
        lst = list(self.by_style_artist[s].values())
        flat = lst[self.rng.integers(len(lst))]
        return int(flat[self.rng.integers(len(flat))])

    def build(self, metas):
        """metas: list of dataset __getitem__ dicts. Returns pos/neg index lists."""
        pos, neg = [], []
        for m in metas:
            p = self._same_style_other_artist(m["style_id"], m["artist_id"])
            n = self._other_style(m["style_id"])
            pos.append(p if p is not None else m["pos"])
            neg.append(n if n is not None else m["pos"])
        return pos, neg


def collate(batch):
    return {
        "x": torch.stack([b["x"] for b in batch]),
        "style_id": torch.tensor([b["style_id"] for b in batch], dtype=torch.long),
        "artist_id": torch.tensor([b["artist_id"] for b in batch], dtype=torch.long),
        "pos": [b["pos"] for b in batch],
        "raw_index": torch.tensor([b["raw_index"] for b in batch], dtype=torch.long),
    }
