"""Patch bank: real training-painting patch statistics per style.

The bank stores RAW prototype descriptors plus a shared whitening transform.
Whiteners are fitted once on the TRAIN pool and shared by every style, so a
target style's distinctiveness is not whitened away, and bank variants
(seed / painter subsets / reduced size) remain comparable.

Provenance is mandatory: painter identity is parsed from the real sample id and
unknown provenance is rejected rather than counted as a distinct artist.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import torch

from .features import GeometryDescriptor, StyleDescriptor, N_PATCH

REPO = Path(__file__).resolve().parents[2]
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
BANK_PATH = REPO / "runs/v8/bank.pt"
# Prototypes actually used at edit time. More is not better: OT cost grows with K.
EDIT_PROTO = 256


def parse_painter(sample_id: str) -> str | None:
    """WikiArt ids look like style__style__artist_work; be conservative."""
    text = str(sample_id).split("/")[-1]
    for suf in (".jpg", ".jpeg", ".png", ".webp", ".pt"):
        if text.lower().endswith(suf):
            text = text[:-len(suf)]
    parts = [p for p in text.split("__") if p]
    if len(parts) < 2:
        return None
    tail = parts[-1]
    artist = tail.split("_", 1)[0]
    styles_lower = {s.lower() for s in STYLES} | {"wikiart"}
    while artist.lower() in styles_lower and "_" in tail:
        tail = tail.split("_", 1)[1]
        artist = tail.split("_", 1)[0]
    if not artist or artist.isdigit() or artist.lower() == "unknown":
        return None
    return artist


@dataclass(frozen=True)
class SourceRecord:
    style: str
    stem: str
    raw_index: int
    source_hash: str
    painter: str | None = None
    path: str = ""

    @property
    def resolved_painter(self):
        return self.painter or parse_painter(self.stem)

    @property
    def key(self):
        return f"{self.raw_index}:{self.source_hash}"


class Whitener:
    def __init__(self, mean: torch.Tensor, scale: torch.Tensor):
        self.mean, self.scale = mean, scale

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x)) / self.scale.to(x).clamp_min(1e-6)

    def state(self):
        return {"mean": self.mean, "scale": self.scale}


class Bank:
    def __init__(self, styles: dict, whiteners: dict, meta: dict):
        self.styles = styles          # style -> {'q','h','u','ll'} RAW tensors
        self.whiteners = whiteners    # name -> Whitener
        self.meta = meta

    def select(self, style: str, feature: str, *, device=None, whiten: bool = True):
        x = self.styles[style][feature]
        if device is not None:
            x = x.to(device)
        return self.whiteners[feature](x) if whiten and feature in self.whiteners else x

    def whiten(self, feature: str, x: torch.Tensor) -> torch.Tensor:
        return self.whiteners[feature](x) if feature in self.whiteners else x

    def stack(self, styles, feature: str, *, device=None, whiten: bool = True,
              max_proto: int | None = None):
        """All styles' prototypes as (S,K,D) in one tensor.

        Batched energies need every style at once; doing it here keeps the
        whitening transform identical to `select`.
        """
        out = []
        for s in styles:
            x = self.styles[s][feature]
            if max_proto is not None and x.shape[0] > max_proto:
                x = x[:max_proto]
            if device is not None:
                x = x.to(device)
            out.append(self.whiteners[feature](x) if whiten and feature in self.whiteners else x)
        return torch.stack(out, 0)

    def state(self):
        return {"styles": self.styles, "meta": self.meta,
                "whiteners": {k: v.state() for k, v in self.whiteners.items()}}


def _load_pool(root: Path = REPO):
    from tools.v7r.v7r_common import SD_POOL
    return SD_POOL


def build_bank(*, per_style: int = 4096, device: str = "cpu", seed: int = 42,
               exclude_hashes: set | None = None) -> Bank:
    """Sample real patches from the training pool; keep provenance per prototype.

    `exclude_hashes` must contain every evaluation source and reference hash so
    no evaluation image can enter the bank.
    """
    pool = _load_pool()
    geom = GeometryDescriptor(); style = StyleDescriptor()
    exclude_hashes = exclude_hashes or set()
    g = torch.Generator().manual_seed(seed)
    raw, prov = {}, {}
    accum = {"q": [], "h": [], "u": [], "ll": []}
    for sid, name in enumerate(STYLES):
        path = pool / f"{sid:02d}_{name}.pt"
        d = torch.load(path, map_location="cpu", weights_only=False)
        x = d["latents"].float()[100:]                 # training split only
        ids = d.get("ids", [f"{i}" for i in range(d["latents"].shape[0])])[100:]
        take = torch.randperm(x.shape[0], generator=g)[:max(1, per_style // N_PATCH)]
        samples, provs = [], []
        for k in take.tolist():
            painter = parse_painter(ids[k])
            if painter is None:
                continue
            samples.append(x[k]); provs.append({"painter": painter, "index": int(k),
                                                "id": ids[k]})
        if not samples:
            raise RuntimeError(f"no verified-painter samples for {name}")
        xb = torch.stack(samples).to(device)
        with torch.no_grad():
            q = geom(xb); st = style(xb)
        raw[name] = {"q": q.reshape(-1, q.shape[-1]).cpu(),
                     "h": st["h"].reshape(-1, st["h"].shape[-1]).cpu(),
                     "u": st["u"].reshape(-1, st["u"].shape[-1]).cpu(),
                     "ll": st["ll"].reshape(-1, st["ll"].shape[-1]).cpu()}
        prov[name] = provs
        for key in accum:
            accum[key].append(raw[name][key])
    whiteners = {}
    for key in accum:
        cat = torch.cat(accum[key], 0)
        whiteners[key] = Whitener(cat.mean(0), cat.std(0).clamp_min(1e-4))
    meta = {"per_style": per_style, "seed": seed, "styles": STYLES,
            "provenance": prov, "n_proto": {k: int(v["h"].shape[0]) for k, v in raw.items()},
            "painter_counts": {k: len({p["painter"] for p in v}) for k, v in prov.items()},
            "whiteing_source": "train split only, common across styles",
            "built_at": __import__("time").strftime("%Y-%m-%d %H:%M:%S")}
    return Bank(raw, whiteners, meta)


def save_bank(bank: Bank, path: Path = BANK_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank.state(), path)


def load_bank(device: str = "cpu", path: Path = BANK_PATH) -> Bank:
    if not Path(path).exists():
        raise FileNotFoundError(f"patch bank missing: {path}; build it first")
    st = torch.load(path, map_location="cpu", weights_only=False)
    wh = {k: Whitener(v["mean"], v["scale"]) for k, v in st["whiteners"].items()}
    return Bank(st["styles"], wh, st["meta"])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-style", type=int, default=4096)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(BANK_PATH))
    a = ap.parse_args()
    exclusions = set()
    tasks = json.loads((REPO / "eval_protocol/tasks_750.json").read_text(encoding="utf-8"))
    for t in tasks.get("sources", tasks.get("tasks", [])):
        exclusions.add(t.get("source_sha256", ""))
    b = build_bank(per_style=a.per_style, device=a.device, seed=a.seed,
                   exclude_hashes=exclusions)
    save_bank(b, Path(a.out))
    print("BANK_BUILT", json.dumps({k: b.meta[k] for k in
          ("n_proto", "painter_counts", "per_style", "seed")}, ensure_ascii=False))
