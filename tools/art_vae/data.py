"""Data layer for the Art-VAE encoder.

Owns the single authoritative training-pool ordering. Every cached artifact
(DINO features, content-matched negative index) is indexed by the position in
this list, so alignment is verifiable and cannot silently drift.

    train_pool = [r for r in manifest.records
                  if r["in_train"] and r["raw_index"] not in val_indices]  # 78,661

Stage A/B/C all read RGB from the 512px memmap and encode on the fly: no stale
latent cache is ever used.

Modes (CLI):
    python -m tools.art_vae.data --prepare-dino
    python -m tools.art_vae.data --prepare-neg
    python -m tools.art_vae.data --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MANIFEST = Path("/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")
OUT = ROOT / "runs" / "art_vae"
OUT.mkdir(parents=True, exist_ok=True)

DINO_MEAN = [0.485, 0.456, 0.406]
DINO_STD = [0.229, 0.224, 0.225]
N_SUPERVISED = 20          # top-20 schools participate in style supervision
UNSEEN_MIN_IMAGES = 200    # schools with >= this many images are held out
NEG_K = 64                 # content-matched negative candidates per image
LOGICAL_BATCH = 64


# --------------------------------------------------------------------------- pool
class TrainPool:
    """Authoritative, ordered training pool + label arrays + style grouping."""

    def __init__(self, manifest_path: Path = MANIFEST):
        m = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        self.manifest = m
        self.mmap = np.memmap(m["bin_path"], dtype=np.uint8, mode="r",
                              shape=tuple(m["shape"]))
        val = set(m["val_indices"])
        self.records = [r for r in m["records"]
                        if r["in_train"] and r["raw_index"] not in val]
        self.n = len(self.records)
        self.raw_index = np.array([r["raw_index"] for r in self.records], np.int64)
        self.style = np.array([r["style_id"] for r in self.records], np.int64)
        self.artist = np.array([r["artist_id"] for r in self.records], np.int64)
        self.genre = np.array([r["genre_id"] for r in self.records], np.int64)
        self.id2name = {v: k for k, v in m["style_to_id"].items()}

        counts = defaultdict(int)
        for s in self.style:
            counts[s] += 1
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        self.supervised_styles = [s for s, _ in ranked[:N_SUPERVISED]]
        self.unseen_styles = [s for s, c in ranked[N_SUPERVISED:]
                              if c >= UNSEEN_MIN_IMAGES]
        self.supervised = np.isin(self.style, self.supervised_styles)
        self.by_sup_style_artist = defaultdict(lambda: defaultdict(list))
        self.by_sup_style = defaultdict(list)
        for i in range(self.n):
            if self.supervised[i]:
                self.by_sup_style[self.style[i]].append(i)
                self.by_sup_style_artist[self.style[i]][self.artist[i]].append(i)

    def describe(self):
        print(f"[Pool] n={self.n} styles={len(set(self.style.tolist()))} "
              f"artists={len(set(self.artist.tolist()))}", flush=True)
        print(f"  supervised schools ({len(self.supervised_styles)}): "
              + ", ".join(f"{self.id2name[s]}({int((self.style==s).sum())})"
                          for s in self.supervised_styles[:6]) + " ...", flush=True)
        print(f"  UNSEEN schools held out from supervision: "
              f"{len(self.unseen_styles)} "
              f"(min {UNSEEN_MIN_IMAGES} imgs each)", flush=True)
        return self


# --------------------------------------------------------------------------- dino
def prepare_dino(dev="cuda", batch=32, limit=0, out_name="dino_train.pt"):
    """Offline DINOv2-base CLS features for the WHOLE train pool, in pool order."""
    from transformers import AutoModel
    pool = TrainPool().describe()
    f = OUT / out_name
    if f.exists():
        d = torch.load(f, map_location="cpu", weights_only=False)
        if d.shape[0] == pool.n and bool(torch.isfinite(d).all()):
            print(f"[DINO] cache valid: {tuple(d.shape)}", flush=True)
            return f
    dm = AutoModel.from_pretrained("facebook/dinov2-base",
                                   local_files_only=True).to(dev).eval()
    nm = torch.tensor(DINO_MEAN, device=dev).view(1, 3, 1, 1)
    ns = torch.tensor(DINO_STD, device=dev).view(1, 3, 1, 1)
    out = []
    import time
    t0 = time.time()
    n_work = min(pool.n, limit) if limit else pool.n
    with torch.no_grad():
        for i in range(0, n_work, batch):
            chunk = pool.raw_index[i:i + batch]
            arr = np.stack([np.ascontiguousarray(pool.mmap[int(r)]) for r in chunk])
            x = torch.from_numpy(arr).permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0).to(dev)
            x01 = (x.clamp(-1, 1) + 1) / 2
            x224 = F.interpolate(x01, (224, 224), mode="bicubic", align_corners=False)
            o = dm(pixel_values=(x224 - nm) / ns)
            out.append(F.normalize(o.last_hidden_state[:, 0].float(), dim=-1).cpu())
            if (i // batch + 1) % 200 == 0:
                el = time.time() - t0
                print(f"  {min(i+batch, n_work)}/{n_work} {el:.0f}s", flush=True)
            del x, x01, x224, o
            torch.cuda.empty_cache()
    D = torch.cat(out).half()
    torch.save(D, f)
    (OUT / "dino_meta.json").write_text(json.dumps({
        "n": D.shape[0], "dim": 768, "dtype": "fp16",
        "source": "DINOv2-base CLS, L2-normalised, pool order",
        "manifest": str(MANIFEST)}, indent=2), encoding="utf-8")
    print(f"[DINO] saved {f} {tuple(D.shape)} in {(time.time()-t0)/60:.1f} min",
          flush=True)
    return f


def prepare_neg(dev="cuda", chunk=512):
    """Content-matched negative candidates: DINO top-K with a DIFFERENT style.

    DINO is used only here, offline. It never enters a training gradient.
    """
    pool = TrainPool().describe()
    dino_f = OUT / "dino_train.pt"
    if not dino_f.exists():
        raise FileNotFoundError("run --prepare-dino first")
    D = torch.load(dino_f, map_location="cpu", weights_only=False).float().to(dev)
    assert D.shape[0] == pool.n
    style = torch.tensor(pool.style, device=dev)
    neg = np.full((pool.n, NEG_K), -1, np.int32)
    t0 = __import__("time").time()
    for i in range(0, pool.n, chunk):
        s = D[i:i + chunk] @ D.t()                       # (c, N) fp32
        s[style[i:i + chunk, None] == style[None, :]] = float("-inf")
        k = min(NEG_K, s.shape[1])
        idx = s.topk(k, dim=1).indices.cpu().numpy()
        neg[i:i + chunk, :k] = idx
        if (i // chunk + 1) % 40 == 0:
            el = __import__("time").time() - t0
            print(f"  {min(i+chunk, pool.n)}/{pool.n} {el:.0f}s", flush=True)
        del s
        torch.cuda.empty_cache()
    np.save(OUT / "neg_candidates.npy", neg)
    # integrity: every row has >=1 valid candidate, and all are different-style
    bad = int((neg[:, 0] < 0).sum())
    same = 0
    for j in range(0, pool.n, 5000):
        sub = neg[j:j + 5000, :8]
        same += int((pool.style[sub] == pool.style[j:j + 5000, None]).sum())
    print(f"[NEG] saved, rows with no candidate={bad}, same-style among top8={same}",
          flush=True)
    assert bad == 0 and same == 0, "negative index integrity failed"
    return OUT / "neg_candidates.npy"


# --------------------------------------------------------------------------- dataset
class ArtVAEDataset:
    """Real-time RGB with DETERMINISTIC views (required by Gradient Cache)."""

    def __init__(self, pool: TrainPool):
        self.p = pool

    def __len__(self):
        return self.p.n

    def _image(self, pos: int, view_flag: int):
        arr = np.ascontiguousarray(self.p.mmap[int(self.p.raw_index[pos])])
        im = Image.fromarray(arr)
        if view_flag:   # second view of the SAME work: modest central crop
            im = im.crop((32, 32, 480, 480)).resize((512, 512), Image.BICUBIC)
        a = np.asarray(im, dtype=np.uint8)
        return torch.from_numpy(a).permute(2, 0, 1).float().div_(127.5).sub_(1.0)

    def get(self, pos: int, view_flag: int = 0):
        return {
            "x": self._image(pos, view_flag),
            "style_id": int(self.p.style[pos]),
            "artist_id": int(self.p.artist[pos]),
            "genre_id": int(self.p.genre[pos]),
            "supervised": bool(self.p.supervised[pos]),
            "pos": int(pos),
        }


class StyleContrastBatchSampler:
    """Yields logical batches of 64:

      32 anchors  : 4 supervised schools x 8 images, painter-disjoint where possible
      32 negatives: one content-matched, DIFFERENT-style image per anchor

    A few anchors (default 4) get a second view of the SAME work instead of a
    negative, producing same-work positive pairs at weight W_SAME_VIEW.

    Anchor sampling is epoch-shuffled WITHOUT replacement, so every supervised
    image is seen once per epoch (reconstruction coverage). Negatives come from
    the offline DINO index and may include unseen schools -- but unseen-vs-unseen
    positive pairs are disabled in the SupCon weight matrix.
    """

    def __init__(self, pool: TrainPool, seed: int = 0, n_view_pairs: int = 4,
                 batches_per_epoch: int | None = None):
        self.p = pool
        self.rng = np.random.default_rng(seed)
        self.n_view_pairs = n_view_pairs
        self.cursor = defaultdict(int)
        self.order = {}
        self._reshuffle()
        self.batches_per_epoch = batches_per_epoch or (pool.n // LOGICAL_BATCH)

    def _reshuffle(self):
        for s in self.p.supervised_styles:
            lst = np.array(self.p.by_sup_style[s])
            self.rng.shuffle(lst)
            self.order[s] = lst.tolist()
            self.cursor[s] = 0

    def _take(self, s, k):
        out = []
        for _ in range(k):
            if self.cursor[s] >= len(self.order[s]):
                lst = np.array(self.p.by_sup_style[s])
                self.rng.shuffle(lst)
                self.order[s] = lst.tolist()
                self.cursor[s] = 0
            out.append(int(self.order[s][self.cursor[s]]))
            self.cursor[s] += 1
        return out

    def __iter__(self):
        neg_path = OUT / "neg_candidates.npy"
        neg = np.load(neg_path, mmap_mode="r") if neg_path.exists() else None
        if neg is None:
            print("[WARN] neg_candidates.npy missing -> random cross-style "
                  "negatives (content matching DISABLED). Run --prepare-neg "
                  "before the real run.", flush=True)
        sup_positions = np.where(self.p.supervised)[0]
        for _ in range(self.batches_per_epoch):
            styles = self.rng.choice(len(self.p.supervised_styles), 4, replace=False)
            anchors = []
            for si in styles:
                anchors += self._take(self.p.supervised_styles[int(si)], 8)
            self.rng.shuffle(anchors)
            batch = [(p, 0) for p in anchors]
            # one content-matched negative per anchor
            for k, p in enumerate(anchors):
                if neg is not None:
                    cand = neg[p][:16]
                    cand = cand[cand >= 0]
                    q = int(cand[self.rng.integers(len(cand))]) if len(cand) else p
                else:
                    for _ in range(20):
                        q = int(sup_positions[self.rng.integers(len(sup_positions))])
                        if self.p.style[q] != self.p.style[p]:
                            break
                batch.append((q, 0))
            # replace a few negatives with a second view of the same work
            if self.n_view_pairs:
                picks = self.rng.choice(len(anchors), self.n_view_pairs, replace=False)
                for k in picks:
                    batch[32 + int(k)] = (anchors[int(k)], 1)
            yield batch

    def __len__(self):
        return self.batches_per_epoch


def collate(batch_items, ds: ArtVAEDataset):
    recs = [ds.get(p, v) for p, v in batch_items]
    return {
        "x": torch.stack([r["x"] for r in recs]),
        "style": torch.tensor([r["style_id"] for r in recs], dtype=torch.long),
        "artist": torch.tensor([r["artist_id"] for r in recs], dtype=torch.long),
        "supervised": torch.tensor([r["supervised"] for r in recs], dtype=torch.bool),
        "view": torch.tensor([r["pos"] for r in recs], dtype=torch.long),
        "pos": [r["pos"] for r in recs],
    }


# --------------------------------------------------------------------------- test
def test_data_alignment():
    print("\n=== 5. Data alignment ===", flush=True)
    pool = TrainPool().describe()
    assert pool.n == 78661, f"unexpected train pool size {pool.n}"
    assert len(set(pool.raw_index.tolist())) == pool.n, "duplicate raw_index"
    assert not (set(pool.raw_index.tolist()) &
                set(pool.manifest["val_indices"])), "val leakage"
    print("  [PASS] 78,661 unique train images, no val leakage", flush=True)
    assert len(pool.supervised_styles) == N_SUPERVISED
    assert len(pool.unseen_styles) >= 10
    print(f"  [PASS] {len(pool.supervised_styles)} supervised / "
          f"{len(pool.unseen_styles)} unseen schools", flush=True)

    ds = ArtVAEDataset(pool)
    x0 = ds.get(0, 0)["x"]; x1 = ds.get(0, 1)["x"]
    assert tuple(x0.shape) == (3, 512, 512)
    assert float((x0 - x1 ).abs().mean()) > 0.01, "second view must differ"
    print("  [PASS] deterministic two-view crop works", flush=True)

    dino_f = OUT / "dino_train.pt"
    if dino_f.exists():
        D = torch.load(dino_f, map_location="cpu", weights_only=False)
        assert D.shape[0] == pool.n and D.shape[1] == 768
        assert bool(torch.isfinite(D).all())
        print(f"  [PASS] DINO cache aligned {tuple(D.shape)}", flush=True)
    else:
        print("  [SKIP] DINO cache not built yet", flush=True)

    neg_f = OUT / "neg_candidates.npy"
    if neg_f.exists():
        neg = np.load(neg_f, mmap_mode="r")
        assert neg.shape[0] == pool.n
        print(f"  [PASS] negative index aligned {tuple(neg.shape)}", flush=True)
    else:
        print("  [SKIP] negative index not built yet", flush=True)

    if neg_f.exists():
        s = StyleContrastBatchSampler(pool, seed=0, batches_per_epoch=3)
        b = next(iter(s))
        assert len(b) == LOGICAL_BATCH
        c = collate(b, ds)
        st = c["style"].numpy()
        sup = c["supervised"].numpy()
        print(f"  [PASS] logical batch {len(b)}: supervised={sup.sum()}, "
              f"distinct styles={len(set(st.tolist()))}", flush=True)
        from tools.art_vae.losses import supcon_relation_weights
        W = supcon_relation_weights(c["style"], c["artist"], c["view"],
                                    supervised=c["supervised"])
        sup_t = torch.as_tensor(sup, dtype=torch.bool)
        unsup_pos = int((W * (~sup_t[:, None] & ~sup_t[None, :]).float()).sum())
        print(f"  [PASS] unseen-school positive pairs = {unsup_pos}", flush=True)
        assert unsup_pos == 0, "unseen schools received implicit supervision"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare-dino", action="store_true")
    ap.add_argument("--prepare-neg", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-name", default="dino_train.pt")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.prepare_dino:
        prepare_dino(args.device, limit=args.limit, out_name=args.out_name)
    if args.prepare_neg:
        prepare_neg(args.device)
    if args.selftest or not (args.prepare_dino or args.prepare_neg):
        test_data_alignment()
    print("[data] done", flush=True)


if __name__ == "__main__":
    main()
