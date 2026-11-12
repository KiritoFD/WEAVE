"""P1: cache A0 latents for the PROPER WikiArt train pool (data-isolation verified).

The repo's RawMemmapWikiArtDataset(is_training=True) filters ONLY by val_indices
and ignores the `in_train` flag, so it returns 100,207 = 78,661 real train images
+ 21,546 leaked TEST images. This script filters explicitly:

    train_pool = {r : r["in_train"] is True and r["raw_index"] not in val_indices}

Verified counts (measured 2026-11-09):
    total records 102,255 | in_train 78,661 | in_train=False 23,594
    val_indices 2,048 (all inside the in_train=False portion)
    => proper train pool 78,661, no leakage

Integrity checks before writing split.json:
  * latent row count == train image count
  * no all-zero rows, no NaN/Inf, per shard
  * re-encode N random samples and compare against the cached row (cos ~1.0)
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/Workspace/WEAVE_VAE")

import numpy as np
import torch
from diffusers import AutoencoderKL

from tools.vae_suite.dataset import RawMemmapWikiArtDataset

A0_DIR = "/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae"
MANIFEST = Path("/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")
OUT = Path("/root/Workspace/WEAVE_VAE/runs/vae_decoder_prune")
FULL = OUT / "full"
FULL.mkdir(parents=True, exist_ok=True)
P1C = Path("/root/Workspace/WEAVE_VAE/runs_vae/v2_diagnosis/p1_cache")
DIM = 16384


def main():
    m = json.loads(MANIFEST.read_text())
    recs = m["records"]
    val = set(m["val_indices"])
    train = [r for r in recs if r["in_train"] and r["raw_index"] not in val]
    train_raw = np.array([r["raw_index"] for r in train], dtype=np.int64)
    n = len(train_raw)
    print(f"[Split] records={len(recs)} in_train={sum(r['in_train'] for r in recs)} "
          f"val={len(val)} -> PROPER train pool = {n}", flush=True)
    assert n == 78661, f"unexpected train pool size {n}"
    assert len(set(train_raw.tolist())) == n, "duplicate raw_index in train pool"
    assert not (set(train_raw.tolist()) & val), "val images leaked into train pool"
    print("[Split] OK: no duplicates, no val overlap", flush=True)

    # image reader that respects the manifest directly (no dataset-class filter)
    mmap = np.memmap(m["bin_path"], dtype=np.uint8, mode="r", shape=tuple(m["shape"]))

    binp = FULL / "latent_train_fp16.bin"
    dst = np.memmap(binp, dtype=np.float16, mode="w+", shape=(n, DIM))
    vae = AutoencoderKL.from_pretrained(A0_DIR, torch_dtype=torch.bfloat16).to("cuda").eval()
    for p in vae.parameters():
        p.requires_grad = False

    t0 = time.time()
    BS = 32
    shard_stats = []
    with torch.no_grad():
        for i in range(0, n, BS):
            chunk = train_raw[i:i + BS]
            arr = np.stack([np.ascontiguousarray(mmap[int(r)]) for r in chunk])
            x = torch.from_numpy(arr).permute(0, 3, 1, 2).float().cuda()
            x = x.div_(127.5).sub_(1.0)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                mu = vae.encode(x).latent_dist.mean
            mu = mu.float().flatten(1)
            # per-shard integrity
            bad_zero = int((mu.abs().sum(1) == 0).sum())
            bad_num = int((~torch.isfinite(mu)).sum())
            if bad_zero or bad_num:
                shard_stats.append({"at": i, "zero": bad_zero, "nonfinite": bad_num})
            dst[i:i + len(chunk)] = mu.cpu().numpy().astype(np.float16)
            if (i // BS + 1) % 200 == 0:
                el = time.time() - t0
                done = i + len(chunk)
                print(f"  {done}/{n}  {el:.0f}s  eta {el/done*(n-done)/60:.1f} min",
                      flush=True)
    dst.flush()
    del dst
    print(f"[Train] {binp.name} {binp.stat().st_size/1024**3:.2f} GB in "
          f"{(time.time()-t0)/60:.1f} min", flush=True)
    print(f"[Shards] problem shards: {len(shard_stats)}", flush=True)
    if shard_stats:
        print(f"  {shard_stats[:5]}", flush=True)

    np.save(FULL / "raw_index_train.npy", train_raw)

    # ---- holdout: frozen 2,048 val set, reused from the already-verified cache ----
    Zv = torch.load(P1C / "val_z.pt", weights_only=False).float()
    if Zv.dim() == 4:
        Zv = Zv.flatten(1)
    ds_val = RawMemmapWikiArtDataset(MANIFEST, is_training=False)
    nv = Zv.shape[0]
    assert nv == len(ds_val) == len(val), f"val mismatch {nv}/{len(ds_val)}/{len(val)}"
    val_raw = np.array(ds_val.indices, dtype=np.int64)
    assert set(val_raw.tolist()) == val, "val cache does not match val_indices"
    binv = FULL / "latent_holdout_fp16.bin"
    dv = np.memmap(binv, dtype=np.float16, mode="w+", shape=(nv, DIM))
    for i in range(0, nv, 2048):
        dv[i:i + 2048] = Zv[i:i + 2048].numpy().astype(np.float16)
    dv.flush()
    del dv, Zv
    np.save(FULL / "raw_index_holdout.npy", val_raw)
    print(f"[Holdout] {nv} latents (frozen val set), matches val_indices", flush=True)

    # ---- re-encode verification ----
    lat = np.memmap(binv if False else binp, dtype=np.float16, mode="r", shape=(n, DIM))
    rng = np.random.default_rng(0)
    probe = rng.choice(n, 8, replace=False)
    cos_list = []
    with torch.no_grad():
        for j in probe:
            r = int(train_raw[j])
            a = np.ascontiguousarray(mmap[r])
            x = torch.from_numpy(a).permute(2, 0, 1).float().div_(127.5).sub_(1.0)
            x = x.unsqueeze(0).cuda()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                mu = vae.encode(x).latent_dist.mean.float().flatten(1)[0].cpu()
            cached = torch.from_numpy(lat[int(j)].astype(np.float32))
            cos_list.append(float(torch.nn.functional.cosine_similarity(
                cached, mu, dim=0)))
    print(f"[Verify] re-encode cos(cached, fresh): "
          f"min={min(cos_list):.5f} mean={np.mean(cos_list):.5f} "
          f"(must be > 0.999)", flush=True)
    assert min(cos_list) > 0.999, "latent/image correspondence mismatch"
    del lat

    (FULL / "split.json").write_text(json.dumps({
        "n_train": int(n), "n_holdout": int(nv), "dim": DIM,
        "train_bin": str(binp), "holdout_bin": str(binv),
        "source": "A0 posterior mean; manifest-filtered proper train pool "
                  "(in_train=True and raw_index not in val_indices)",
        "n_styles": "all", "leak_check": "passed",
        "reencode_cos_min": round(min(cos_list), 6),
    }, indent=2), encoding="utf-8")
    print(f"[Done] {FULL}", flush=True)


if __name__ == "__main__":
    main()
