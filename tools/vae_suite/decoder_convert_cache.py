"""Convert the EXISTING correct caches into the format decoder_train.py needs.

Source (already correct, built with ds[p]["image"]):
  runs_vae/v2_diagnosis/p1_cache/train_z_20000.pt   (12,991 x 16384 fp32)
  runs_vae/v2_diagnosis/p1_cache/train_meta_20000.pt (style/artist/genre)
  runs_vae/v2_diagnosis/p1_cache/val_z.pt            (2,048 x 16384 fp32)

The train cache did not save its image positions, but the selection was
deterministic (numpy default_rng(21), per-style cap). We replay it and VERIFY by
comparing the recovered style ids against the saved meta. No GPU, no encoding.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/Workspace/WEAVE_VAE")

import numpy as np
import torch

from tools.vae_suite.dataset import RawMemmapWikiArtDataset

SRC = Path("/root/Workspace/WEAVE_VAE/runs_vae/v2_diagnosis/p1_cache")
MANIFEST = Path("/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")
OUT = Path("/root/Workspace/WEAVE_VAE/runs/vae_decoder_prune")
OUT.mkdir(parents=True, exist_ok=True)
N_TRAIN = 20000
DIM = 16384


def main():
    # ---- replay the deterministic selection ----
    ds = RawMemmapWikiArtDataset(MANIFEST, is_training=True)
    by_style = {}
    for pos, raw in enumerate(ds.indices):
        rec = ds.records_by_idx[raw]
        by_style.setdefault(rec["style_id"], []).append(pos)

    rng = np.random.default_rng(21)
    per = max(1, N_TRAIN // max(len(by_style), 1))
    sel, sel_style = [], []
    for s, lst in by_style.items():
        pick = lst if len(lst) <= per else rng.choice(lst, per, replace=False)
        for p in pick:
            sel.append(int(p))
            sel_style.append(s)
    sel = sel[:N_TRAIN]
    sel_style = sel_style[:N_TRAIN]
    print(f"[Replay] recovered {len(sel)} positions, {len(by_style)} styles", flush=True)

    meta = torch.load(SRC / "train_meta_20000.pt", weights_only=False)
    saved_style = meta["style"].numpy()
    n = len(saved_style)
    assert n == len(sel), f"count mismatch: saved {n} vs replay {len(sel)}"
    match = (np.array(sel_style) == saved_style).mean()
    print(f"[Verify] style-id agreement with saved meta = {match*100:.2f}%", flush=True)
    if match < 1.0:
        bad = np.where(np.array(sel_style) != saved_style)[0][:5]
        print(f"  mismatch at rows {bad.tolist()} "
              f"(replay {[sel_style[i] for i in bad]} vs saved {saved_style[bad].tolist()})",
              flush=True)
        print("  ABORT: cannot trust the position mapping", flush=True)
        return

    # ---- write fp16 memmap + raw_index map ----
    Z = torch.load(SRC / "train_z_20000.pt", weights_only=False).float()
    if Z.dim() == 4:
        Z = Z.flatten(1)
    assert Z.shape == (n, DIM), f"unexpected latent shape {tuple(Z.shape)}"
    print(f"[Train] latents {tuple(Z.shape)}", flush=True)

    binp = OUT / "latent_fp16.bin"
    dst = np.memmap(binp, dtype=np.float16, mode="w+", shape=(n, DIM))
    for i in range(0, n, 2048):
        dst[i:i + 2048] = Z[i:i + 2048].numpy().astype(np.float16)
    dst.flush()
    del dst, Z

    raw_index = np.array([ds.indices[p] for p in sel], dtype=np.int32)
    np.save(OUT / "raw_index.npy", raw_index)

    # ---- holdout: use the val cache (independent, never trained on) ----
    ds_val = RawMemmapWikiArtDataset(MANIFEST, is_training=False)
    Zv = torch.load(SRC / "val_z.pt", weights_only=False).float()
    if Zv.dim() == 4:
        Zv = Zv.flatten(1)
    nv = Zv.shape[0]
    binv = OUT / "latent_holdout_fp16.bin"
    dv = np.memmap(binv, dtype=np.float16, mode="w+", shape=(nv, DIM))
    for i in range(0, nv, 2048):
        dv[i:i + 2048] = Zv[i:i + 2048].numpy().astype(np.float16)
    dv.flush()
    del dv, Zv
    hold_raw = np.array([ds_val.indices[i] for i in range(nv)], dtype=np.int32)
    np.save(OUT / "holdout_raw_index.npy", hold_raw)

    (OUT / "split.json").write_text(json.dumps({
        "n_train": n, "n_holdout": nv, "dim": DIM,
        "train_bin": str(binp), "holdout_bin": str(binv),
        "source": "p1_cache/train_z_20000.pt (12,991) + val_z.pt (2,048)",
        "style_agreement": float(match),
        "n_styles": len(by_style),
    }, indent=2), encoding="utf-8")
    print(f"[Split] train={n} holdout={nv}", flush=True)
    print(f"[Saved] {binp} ({binp.stat().st_size/1024**2:.0f} MB), {binv}", flush=True)


if __name__ == "__main__":
    main()
