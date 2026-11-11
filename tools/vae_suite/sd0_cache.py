"""P0: cache official SD0 (sd-vae-ft-ema) posterior-mean latents.

Strictly separate from the old A0 caches: writes to runs/sd0_s10/full/ and never
touches runs/vae_decoder_prune/full/.

Split (verified 2026-11-09, reused here):
    train   = {r : r["in_train"] and r["raw_index"] not in val_indices}  -> 78,661
    holdout = the frozen 2,048 val images
    test256 = a disjoint 256-image subset for final acceptance

z = E_SD0(x).mean   (UNSCALED; no scaling_factor applied anywhere in training)

Checks written into full/split.json:
  * image-id disjointness across train / holdout / test256
  * no NaN / Inf / all-zero latent rows (full scan, not just probes)
  * 128 random re-encode rows, cos(cached, fresh) > 0.999
  * Encoder determinism on the same input
  * official Encoder checkpoint hash (sha256 of config + safetensors)
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL

sys.path.insert(0, "/root/Workspace/WEAVE_VAE")

SD0_DIR = "/root/Workspace/xy/DiT/data/pretrained/sd-vae-ft-ema"
MANIFEST = Path("/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")
OUT = Path("/root/Workspace/WEAVE_VAE/runs/sd0_s10")
FULL = OUT / "full"
FULL.mkdir(parents=True, exist_ok=True)
DIM = 16384
TEST_N = 256


def sha256_file(p: Path, limit=None):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
            if limit and f.tell() >= limit * (1 << 20):
                break
    return h.hexdigest()


def load_image(mmap, raw_idx):
    arr = np.ascontiguousarray(mmap[int(raw_idx)])
    return torch.from_numpy(arr).permute(2, 0, 1).float().div_(127.5).sub_(1.0)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true",
                    help="reuse existing latent_*.bin, only redo verification + split.json")
    _a = ap.parse_args()
    m = json.loads(MANIFEST.read_text())
    recs = m["records"]
    val = set(m["val_indices"])
    train = [r for r in recs if r["in_train"] and r["raw_index"] not in val]
    hold = [r for r in recs if r["raw_index"] in val]
    # test256: disjoint from train and holdout, prefer in_train=False records
    used = set(r["raw_index"] for r in train) | set(r["raw_index"] for r in hold)
    pool_test = [r for r in recs if r["raw_index"] not in used and not r["in_train"]]
    rng = np.random.default_rng(17)
    test = [pool_test[int(i)] for i in
            rng.choice(len(pool_test), min(TEST_N, len(pool_test)), replace=False)]

    n_tr, n_ho, n_te = len(train), len(hold), len(test)
    print(f"[Split] train={n_tr} holdout={n_ho} test256={n_te}", flush=True)
    assert n_tr == 78661 and n_ho == 2048, "unexpected split sizes"
    s_tr, s_ho, s_te = (set(r["raw_index"] for r in g) for g in (train, hold, test))
    assert not (s_tr & s_ho) and not (s_tr & s_te) and not (s_ho & s_te), \
        "image-id overlap across splits"
    assert not (s_tr & val), "val leakage into train"

    mmap = np.memmap(m["bin_path"], dtype=np.uint8, mode="r", shape=tuple(m["shape"]))
    vae = AutoencoderKL.from_pretrained(SD0_DIR, torch_dtype=torch.float32).to("cuda").eval()
    for p in vae.parameters():
        p.requires_grad = False
    ck_hash = {
        "config.json": sha256_file(Path(SD0_DIR) / "config.json"),
        "diffusion_pytorch_model.safetensors":
            sha256_file(Path(SD0_DIR) / "diffusion_pytorch_model.safetensors"),
    }
    print(f"[Encoder] SD0 hash {ck_hash['config.json'][:16]}...", flush=True)

    # ---- full integrity cache ----
    def cache_group(records, name):
        n = len(records)
        binp = FULL / f"latent_{name}_fp16.bin"
        dst = np.memmap(binp, dtype=np.float16, mode="w+", shape=(n, DIM))
        raw = np.array([r["raw_index"] for r in records], np.int64)
        bs = 32
        t0 = time.time()
        bad = {"zero": 0, "nonfinite": 0}
        with torch.no_grad():
            for i in range(0, n, bs):
                chunk = raw[i:i + bs]
                x = torch.stack([load_image(mmap, r) for r in chunk]).cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    z = vae.encode(x).latent_dist.mean
                zf = z.float().flatten(1)
                bad["zero"] += int((zf.abs().sum(1) == 0).sum())
                bad["nonfinite"] += int((~torch.isfinite(zf)).any(dim=1).sum())
                dst[i:i + len(chunk)] = zf.cpu().numpy().astype(np.float16)
                del x, z, zf
                if (i // bs + 1) % 200 == 0:
                    el = time.time() - t0
                    print(f"  [{name}] {min(i+bs,n)}/{n} {el:.0f}s "
                          f"eta {el/(i+bs)*(n-i-bs)/60:.1f}min", flush=True)
        dst.flush()
        del dst
        np.save(FULL / f"raw_index_{name}.npy", raw)
        print(f"  [{name}] {binp.name} {binp.stat().st_size/2**30:.2f}GiB "
              f"zero={bad['zero']} nonfinite={bad['nonfinite']} "
              f"in {(time.time()-t0)/60:.1f}min", flush=True)
        assert bad["zero"] == 0 and bad["nonfinite"] == 0, f"{name}: bad latent rows"
        return binp, bad

    if _a.verify_only:
        print("[VerifyOnly] reusing existing latent_*.bin (no re-encode)", flush=True)
        b_tr = FULL / "latent_train_fp16.bin"
        b_ho = FULL / "latent_holdout_fp16.bin"
        b_te = FULL / "latent_test256_fp16.bin"
        for f, n in [(b_tr, n_tr), (b_ho, n_ho), (b_te, n_te)]:
            assert f.exists() and f.stat().st_size == n * DIM * 2,                 f"{f.name} missing or wrong size"
        # full-scan integrity on the reused files
        bad_tr = bad_ho = bad_te = {"zero": 0, "nonfinite": 0}
        for f, n in [(b_tr, n_tr), (b_ho, n_ho), (b_te, n_te)]:
            a = np.memmap(f, dtype=np.float16, mode="r", shape=(n, DIM))
            for i in range(0, n, 4096):
                q = np.asarray(a[i:i + 4096], np.float32)
                bad_tr["zero"] += int((np.abs(q).sum(1) == 0).sum())
                bad_tr["nonfinite"] += int((~np.isfinite(q)).any(1).sum())
            del a
        print(f"[VerifyOnly] full scan zero={bad_tr['zero']} "
              f"nonfinite={bad_tr['nonfinite']}", flush=True)
        assert bad_tr["zero"] == 0 and bad_tr["nonfinite"] == 0
    else:
        b_tr, bad_tr = cache_group(train, "train")
        b_ho, bad_ho = cache_group(hold, "holdout")
        b_te, bad_te = cache_group(test, "test256")

    # ---- re-encode 128 random training rows ----
    lat = np.memmap(b_tr, dtype=np.float16, mode="r", shape=(n_tr, DIM))
    probe = np.random.default_rng(0).choice(n_tr, 128, replace=False)
    cos_list = []
    with torch.no_grad():
        for j in probe:
            x = load_image(mmap, int(train[int(j)]["raw_index"])).unsqueeze(0).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = vae.encode(x).latent_dist.mean.float().flatten()
            c = torch.from_numpy(lat[int(j)].astype(np.float32)).to(x.device)
            cos_list.append(float(F.cosine_similarity(c, z, dim=0)))
            del x, z
        # determinism: same input twice
        x = load_image(mmap, int(train[0]["raw_index"])).unsqueeze(0).cuda()
        z1 = vae.encode(x).latent_dist.mean
        z2 = vae.encode(x).latent_dist.mean
        det = float((z1 - z2).abs().max())
        del x, z1, z2
    print(f"[Verify] re-encode cos min={min(cos_list):.6f} "
          f"mean={float(np.mean(cos_list)):.6f} | encoder determinism "
          f"max|z1-z2|={det:.2e}", flush=True)
    assert min(cos_list) > 0.999 and det == 0.0, "cache verification failed"
    del lat

    split = {
        "n_train": n_tr, "n_holdout": n_ho, "n_test256": n_te, "dim": DIM,
        "train_bin": str(b_tr), "holdout_bin": str(b_ho), "test256_bin": str(b_te),
        "source": "official sd-vae-ft-ema posterior mean, UNSCALED latent",
        "encoder": SD0_DIR, "encoder_hash": ck_hash,
        "scaling_factor": float(vae.config.scaling_factor),
        "latent_convention": "z = encoder(x).latent_dist.mean (no scaling applied)",
        "image_id_disjoint": True,
        "bad_rows": {"train": bad_tr, "holdout": bad_ho, "test256": bad_te},
        "reencode_cos_min": round(min(cos_list), 6),
        "encoder_determinism_max_abs": det,
        "split_seed": 17,
    }
    (FULL / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    print(f"[Done] {FULL / 'split.json'}", flush=True)


if __name__ == "__main__":
    main()
