"""Data layer for the frozen-latent Local Style Reader experiment.

Everything is cached, so NO VAE forward is ever needed:

  latent_train   (78661, 16384) fp16  A0 posterior mean   [verified aligned]
  latent_holdout (2048, 16384)  fp16
  dino_train     (78661, 768)   fp16  offline content features (sampling only)
  neg_candidates (78661, 64)          content-matched, different-style indices

All four share the SAME row ordering (checked element-wise via raw_index).
Labels come from the WikiArt manifest.

Painter-disjoint protocol
  A fixed set of 250 painters is held out: none of their training-pool images
  are used to train the reader. Validation reports BOTH the full 2048 holdout
  (reference) and the strict subset whose painter is in the held-out set.
  Unseen schools (31) never provide positives, exactly as in art_vae.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.art_vae.data import (LOGICAL_BATCH, NEG_K, N_SUPERVISED, OUT as ART_OUT,
                                TrainPool)

LAT_DIR = ROOT / "runs" / "vae_decoder_prune" / "full"
MANIFEST = Path("/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")
HELD_PAINTERS = 250
HELD_SEED = 0


class FrozenLatents:
    """Memmapped frozen latents + labels + offline DINO/negative indices."""

    def __init__(self, manifest_path: Path = MANIFEST):
        self.pool = TrainPool(manifest_path)
        p = self.pool
        split = json.loads((LAT_DIR / "split.json").read_text(encoding="utf-8"))
        assert split["n_train"] == p.n, "latent/train-pool size mismatch"

        # alignment is NON-NEGOTIABLE: a silent row shift destroys hard negatives
        raw = np.load(LAT_DIR / "raw_index_train.npy")
        assert np.array_equal(raw, p.raw_index), "latent row order != TrainPool"
        raw_h = np.load(LAT_DIR / "raw_index_holdout.npy")
        val = set(p.manifest["val_indices"])
        assert set(raw_h.tolist()) == val, "holdout raw_index != val set"

        self.z_train = np.memmap(split["train_bin"], dtype=np.float16, mode="r",
                                 shape=(split["n_train"], 16384))
        self.z_hold = np.memmap(split["holdout_bin"], dtype=np.float16, mode="r",
                                shape=(split["n_holdout"], 16384))
        self.raw_hold = raw_h
        self.hold_records = [r for r in p.manifest["records"]
                             if r["raw_index"] in val]
        assert len(self.hold_records) == len(raw_h)
        # holdout records are in manifest order, latent rows are in raw_h order;
        # build the mapping explicitly rather than assuming they coincide
        pos_of_raw = {int(r): i for i, r in enumerate(raw_h.tolist())}
        self.hold_style = np.zeros(len(raw_h), np.int64)
        self.hold_artist = np.zeros(len(raw_h), np.int64)
        self.hold_genre = np.zeros(len(raw_h), np.int64)
        for r in self.hold_records:
            i = pos_of_raw[r["raw_index"]]
            self.hold_style[i] = r["style_id"]
            self.hold_artist[i] = r["artist_id"]
            self.hold_genre[i] = r["genre_id"]
        self.hold_supervised = np.isin(self.hold_style, p.supervised_styles)
        self.hold_unseen = np.isin(self.hold_style, p.unseen_styles)

        self.dino = torch.load(ART_OUT / "dino_train.pt", map_location="cpu",
                               weights_only=False).float()
        assert self.dino.shape[0] == p.n
        self.neg = np.load(ART_OUT / "neg_candidates.npy", mmap_mode="r")

        # ---- painter-disjoint split ----
        all_art = np.unique(p.artist)
        hold_art_set = set(self.hold_artist.tolist())
        cand = np.array([a for a in all_art if a in hold_art_set])
        rng = np.random.default_rng(HELD_SEED)
        self.held_painters = rng.choice(cand, min(HELD_PAINTERS, len(cand)),
                                        replace=False)
        self.train_excluded = np.isin(p.artist, self.held_painters)
        self.train_allowed = ~self.train_excluded
        # strict validation subset: holdout images of held-out painters
        self.hold_painter_disjoint = np.isin(self.hold_artist, self.held_painters)
        # supervised-school images eligible for training
        self.sup_train_idx = np.where(self.train_allowed & p.supervised)[0]
        self.sup_by_style_artist = defaultdict(lambda: defaultdict(list))
        self.sup_by_style = defaultdict(list)
        for i in self.sup_train_idx:
            s, a = int(p.style[i]), int(p.artist[i])
            self.sup_by_style[s].append(i)
            self.sup_by_style_artist[s][a].append(i)

    def describe(self):
        p = self.pool
        print(f"[Frozen] train latents {self.z_train.shape} fp16 | "
              f"holdout {self.z_hold.shape}", flush=True)
        print(f"  supervised schools {len(p.supervised_styles)} | "
              f"unseen {len(p.unseen_styles)} | painters {len(set(p.artist.tolist()))}",
              flush=True)
        print(f"  painter-heldout: {len(self.held_painters)} painters -> "
              f"excluded {int(self.train_excluded.sum())} train imgs "
              f"({self.train_excluded.mean()*100:.1f}%)", flush=True)
        print(f"  eligible supervised train imgs: {len(self.sup_train_idx)}", flush=True)
        print(f"  strict painter-disjoint holdout imgs: "
              f"{int(self.hold_painter_disjoint.sum())} / {len(self.hold_artist)}",
              flush=True)
        print(f"  holdout unseen-school imgs: {int(self.hold_unseen.sum())}", flush=True)
        return self

    def latent(self, idx, hold=False) -> torch.Tensor:
        """(K,4,64,64) float tensor from the frozen cache."""
        src = self.z_hold if hold else self.z_train
        a = np.asarray(src[np.asarray(idx, np.int64)], dtype=np.float32)
        return torch.from_numpy(a).view(-1, 4, 64, 64)


class SupConBatchSampler:
    """Logical batch of 64 cached latent indices.

    32 anchors: 4 supervised schools x 8 images, painter-disjoint where possible
    32 hard negatives: one DINO-content-matched DIFFERENT-style image per anchor
    A few anchors get a second view of the SAME latent (different deterministic
    crop in readout space is not needed here; we use exact-duplicate views with
    a small Gaussian-free jitter=0, i.e. same latent, weight via view id).

    Unseen-school images are allowed as negatives but NEVER as positives.
    """

    def __init__(self, data: FrozenLatents, seed: int = 0,
                 batches_per_epoch: int = 1000, n_view_pairs: int = 4):
        self.d = data
        self.rng = np.random.default_rng(seed)
        self.n_view_pairs = n_view_pairs
        self.batches = batches_per_epoch
        self.cursor = defaultdict(int)
        self.order = {}
        self._reshuffle()

    def _reshuffle(self):
        for s in self.d.pool.supervised_styles:
            lst = np.array(self.d.sup_by_style[s])
            self.rng.shuffle(lst)
            self.order[s] = lst.tolist()
            self.cursor[s] = 0

    def _take(self, s, k):
        out = []
        for _ in range(k):
            if self.cursor[s] >= len(self.order[s]):
                lst = np.array(self.d.sup_by_style[s])
                self.rng.shuffle(lst)
                self.order[s] = lst.tolist()
                self.cursor[s] = 0
            out.append(int(self.order[s][self.cursor[s]]))
            self.cursor[s] += 1
        return out

    def __iter__(self):
        p = self.d.pool
        for _ in range(self.batches):
            styles = self.rng.choice(len(p.supervised_styles), 4, replace=False)
            anchors = []
            for si in styles:
                anchors += self._take(p.supervised_styles[int(si)], 8)
            self.rng.shuffle(anchors)
            batch = list(anchors)                     # 32 anchors
            for k, a in enumerate(anchors):
                cand = self.d.neg[a][:16]
                cand = cand[cand >= 0]
                q = int(cand[self.rng.integers(len(cand))]) if len(cand) else a
                batch.append(q)                       # 32 hard negatives
            if self.n_view_pairs:
                picks = self.rng.choice(32, self.n_view_pairs, replace=False)
                for k in picks:
                    batch[32 + int(k)] = anchors[int(k)]   # same latent, 2nd view
            yield batch

    def __len__(self):
        return self.batches


def batch_meta(data: FrozenLatents, idxs):
    p = data.pool
    idx = np.asarray(idxs, np.int64)
    return {
        "style": torch.tensor(p.style[idx], dtype=torch.long),
        "artist": torch.tensor(p.artist[idx], dtype=torch.long),
        "view": torch.tensor(idx, dtype=torch.long),
        "supervised": torch.tensor(p.supervised[idx], dtype=torch.bool),
        "idx": idx,
    }


if __name__ == "__main__":
    FrozenLatents().describe()
