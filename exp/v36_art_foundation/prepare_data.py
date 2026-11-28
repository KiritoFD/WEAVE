"""Stage P0: Whole-WikiArt Deduplication, Strict Core-5 Holdout Isolation, and Memmap Cache Builder.

Constructs:
  - Seen-Train (all 131 non-holdout movements, ~83-85k unique artworks, no per-style cap)
  - Artist-OOD (~1,000 artworks from disjoint artists inside the 131 movements)
  - Seen-Dev (~1,000 artworks across the 131 movements for stage diagnostics & P3 calibration)
  - Movement-OOD (5 held-out movements, default Core-5, strictly excluded from ALL training/PCA/norm)
Fits W_stiefel_cnx (384x32), P_Y/P_C/u_y, luma_proj, and 1561-D stat normalization ONLY on Seen-Train.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import math
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 250_000_000

from exp.v18 import tasks_and_sources
from exp.v20.slf_flow import coarse_y16_lcn
from exp.v32_nodino.cnx_layout import ConvNeXtTinyStage3Extractor, extract_cnx_q32
from exp.v34_frontier.stage_a_startpoint import _py_unit_vec, decompose_py_luma
from exp.v35_cleave_one.prepare_data import _encode_rgb_scaled, _find_sg_root, _load_sd_vae_encoder
from exp.v35_cleave_one.style_features import ConvNeXtExplicitAndPatchBackbone
from exp.v36_art_foundation import (
    CACHE_DIR,
    HOLDOUT_PRESETS,
    ROOT,
    SEED_V36,
    find_wikiart_data_root,
)
from exp.v36_art_foundation.proxy import destylize_proxy_v36
import paths
from src.utils.protocol import canonical_pid
from src.utils.runtime import bf16_autocast, read_json, sanitize_cuda_env, save_json, seed_all, utc


GENERIC_TITLES = frozenset({
    "untitled",
    "not identified",
    "unknown title",
    "not detected",
    "portrait",
    "self-portrait",
    "landscape",
    "still life",
    "nude",
    "study",
    "sketch",
    "composition",
    "abstract",
    "flowers",
    "woman",
    "man",
})


def _slug_title(artist: str, title: str) -> str:
    a = artist.casefold().strip().replace(" ", "-")
    t = title.casefold().strip().replace(" ", "-")
    return f"{a}_{t}"


def _thumb16_vec(im: Image.Image) -> np.ndarray:
    sm = im.resize((16, 16), Image.Resampling.BILINEAR)
    return (np.asarray(sm, dtype=np.float32).reshape(-1) / 255.0)


def _load_banned_signatures(sg_root: Path | None) -> tuple[set[str], set[str], np.ndarray]:
    tasks = read_json(ROOT / "eval_protocol/tasks_750.json")["tasks"]
    refs_d = read_json(ROOT / "eval_protocol/references_v1.json")["references"]
    banned_pids: set[str] = {canonical_pid(t["source_stem"]) for t in tasks}
    banned_pids |= {canonical_pid(r["stem"]) for rows in refs_d.values() for r in rows}
    banned_slugs: set[str] = set()
    for p in banned_pids:
        banned_slugs.add(p)

    thumbs = []
    for sub in sorted((ROOT / "data/test").rglob("*.*")):
        if sub.suffix.lower() in (".jpg", ".jpeg", ".png"):
            banned_pids.add(canonical_pid(sub.stem))
            try:
                with Image.open(sub) as im:
                    thumbs.append(_thumb16_vec(im.convert("RGB")))
            except Exception:
                pass
    if sg_root is not None and sg_root.exists():
        for sub in sorted(sg_root.rglob("*.*")):
            if sub.suffix.lower() in (".jpg", ".jpeg", ".png"):
                try:
                    with Image.open(sub) as im:
                        thumbs.append(_thumb16_vec(im.convert("RGB")))
                except Exception:
                    pass
    thumb_mat = np.stack(thumbs, axis=0) if thumbs else np.zeros((0, 768), dtype=np.float32)
    return banned_pids, banned_slugs, thumb_mat


def curate_wikiart_v36(
    wikiart_root: Path,
    sg_root: Path | None,
    *,
    holdout_preset: str = "core5",
    min_px: int = 256,
    n_artist_ood: int = 1000,
    n_seen_dev: int = 1000,
    seed: int = SEED_V36,
) -> dict:
    """Whole-WikiArt deduplication FIRST, then strict movement & artist isolation."""
    manifest_p = CACHE_DIR / "curation_manifest.json"
    if manifest_p.exists():
        return read_json(manifest_p)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    rng = np.random.default_rng(seed)

    holdout_styles = set(HOLDOUT_PRESETS[holdout_preset])
    banned_pids, _, banned_thumbs = _load_banned_signatures(sg_root)

    train_dir = wikiart_root / "train/train"
    test_dir = wikiart_root / "test/test"
    train_files = set(os.listdir(train_dir)) if train_dir.exists() else set()
    test_files = set(os.listdir(test_dir)) if test_dir.exists() else set()

    # Step 1: Load ALL valid entries across the entire WikiArt corpus
    raw_rows = []
    with (wikiart_root / "all_data_info.csv").open("r", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            fn = (r.get("new_filename") or "").strip()
            if fn in train_files:
                fpath = train_dir / fn
            elif fn in test_files:
                fpath = test_dir / fn
            else:
                continue
            style = (r.get("style") or "").strip()
            if not style or style == "UNKNOWN":
                continue
            try:
                px = float(r.get("pixelsx") or 0)
                py = float(r.get("pixelsy") or 0)
                sz = int(float(r.get("size_bytes") or 0))
            except Exception:
                px, py, sz = 0.0, 0.0, 0
            if min(px, py) < min_px:
                continue
            artist = (r.get("artist") or "UNKNOWN").strip()
            title = (r.get("title") or "").strip()
            raw_rows.append({
                "filename": fn,
                "path": str(fpath),
                "style": style,
                "artist": artist,
                "title": title,
                "px": int(px),
                "py": int(py),
                "size_bytes": sz,
                "in_train_flag": fn in train_files,
            })

    # Step 2: Whole-WikiArt deduplication BEFORE movement separation!
    # 2a) Build forbidden exact/title keys from holdout styles + tasks_750 / references_v1
    forbidden_exact: set[tuple[str, int, int, int]] = set()
    forbidden_at: set[tuple[str, str]] = set()
    for r in raw_rows:
        ak = r["artist"].casefold()
        tk = r["title"].casefold()
        is_holdout = r["style"] in holdout_styles
        slug = canonical_pid(f"{r['artist']}_{r['title']}") if r["title"] else ""
        is_banned = (canonical_pid(Path(r["filename"]).stem) in banned_pids) or (slug in banned_pids)
        if is_holdout or is_banned:
            forbidden_exact.add((ak, r["px"], r["py"], r["size_bytes"]))
            if tk and tk not in GENERIC_TITLES:
                forbidden_at.add((ak, tk))

    # 2b) Keep unique non-holdout works & clean holdout works separately
    seen_exact: set[tuple[str, int, int, int]] = set()
    seen_non_generic_at: set[tuple[str, str]] = set()
    seen_pool: list[dict] = []
    holdout_pool: list[dict] = []

    n_purged_cross = 0
    n_purged_dup = 0
    for r in raw_rows:
        ak = r["artist"].casefold()
        tk = r["title"].casefold()
        ek = (ak, r["px"], r["py"], r["size_bytes"])
        atk = (ak, tk) if (tk and tk not in GENERIC_TITLES) else None

        if r["style"] in holdout_styles:
            holdout_pool.append(r)
            continue

        if ek in forbidden_exact or (atk is not None and atk in forbidden_at):
            n_purged_cross += 1
            continue
        if ek in seen_exact or (atk is not None and atk in seen_non_generic_at):
            n_purged_dup += 1
            continue
        seen_exact.add(ek)
        if atk is not None:
            seen_non_generic_at.add(atk)
        seen_pool.append(r)

    # Step 3: Partition 131 seen movements into Seen-Train / Artist-OOD (~1,000) / Seen-Dev (~1,000)
    by_artist: dict[str, list[dict]] = {}
    for r in seen_pool:
        by_artist.setdefault(r["artist"].casefold(), []).append(r)

    # Hold out artists deterministically (~1.5% of seen-movement artists -> ~1,000 works, zero train overlap)
    def _ahash(s: str) -> int:
        return int(hashlib.md5(s.encode("utf-8", "ignore")).hexdigest()[:8], 16) % 1000

    cand_artists = [a for a in sorted(by_artist.keys()) if a != "unknown" and len(by_artist[a]) <= 120]
    cand_artists.sort(key=lambda a: (_ahash(a), a))
    artist_ood_set: set[str] = set()
    artist_ood_rows: list[dict] = []
    for a in cand_artists:
        if len(artist_ood_rows) >= n_artist_ood:
            break
        artist_ood_set.add(a)
        artist_ood_rows.extend(by_artist[a])
    if len(artist_ood_rows) > n_artist_ood:
        idx = rng.choice(len(artist_ood_rows), size=n_artist_ood, replace=False)
        artist_ood_rows = [artist_ood_rows[i] for i in sorted(idx.tolist())]

    remaining_seen = [r for r in seen_pool if r["artist"].casefold() not in artist_ood_set]
    # Stratified sample for Seen-Dev (~1,000 works across the 131 movements)
    by_style_rem: dict[str, list[int]] = {}
    for idx, r in enumerate(remaining_seen):
        by_style_rem.setdefault(r["style"], []).append(idx)
    dev_idx_set: set[int] = set()
    for st, idxs in sorted(by_style_rem.items()):
        if len(idxs) >= 4:
            take = max(1, int(round(len(idxs) * (n_seen_dev / float(len(remaining_seen))))))
            take = min(take, len(idxs) // 3)
            picks = rng.choice(idxs, size=take, replace=False)
            dev_idx_set.update(int(x) for x in picks)
    if len(dev_idx_set) > n_seen_dev:
        dev_list = sorted(dev_idx_set)
        keep_d = rng.choice(dev_list, size=n_seen_dev, replace=False)
        dev_idx_set = {int(x) for x in keep_d}
    elif len(dev_idx_set) < n_seen_dev:
        avail = [i for i in range(len(remaining_seen)) if i not in dev_idx_set]
        extra = rng.choice(avail, size=min(n_seen_dev - len(dev_idx_set), len(avail)), replace=False)
        dev_idx_set.update(int(x) for x in extra)

    seen_dev = [remaining_seen[i] for i in sorted(dev_idx_set)]
    seen_train = [remaining_seen[i] for i in range(len(remaining_seen)) if i not in dev_idx_set]

    # Assert strict disjointness invariants
    tr_artists = {r["artist"].casefold() for r in seen_train}
    ao_artists = {r["artist"].casefold() for r in artist_ood_rows}
    tr_styles = {r["style"] for r in seen_train}
    assert not (tr_artists & ao_artists), "Artist leakage between Seen-Train and Artist-OOD!"
    assert not (tr_styles & holdout_styles), "Holdout movement leaked into Seen-Train!"
    assert not ({r["style"] for r in seen_dev} & holdout_styles), "Holdout movement leaked into Seen-Dev!"

    # Select balanced Movement-OOD evaluation subset (up to 100/style = 500 across the 5 held-out movements)
    rng.shuffle(holdout_pool)
    mv_counts: dict[str, int] = {}
    mv_ood = []
    for r in holdout_pool:
        st = r["style"]
        if mv_counts.get(st, 0) < 100:
            mv_ood.append(r)
            mv_counts[st] = mv_counts.get(st, 0) + 1

    manifest = {
        "created_utc": utc(),
        "seed": seed,
        "wikiart_root": str(wikiart_root),
        "holdout_preset": holdout_preset,
        "holdout_movements": sorted(holdout_styles),
        "n_raw_valid": len(raw_rows),
        "n_purged_cross_boundary": n_purged_cross,
        "n_purged_intra_dup": n_purged_dup,
        "n_seen_train": len(seen_train),
        "n_seen_movements": len(tr_styles),
        "n_seen_dev": len(seen_dev),
        "n_artist_ood": len(artist_ood_rows),
        "n_movement_ood_eval": len(mv_ood),
        "seen_train": seen_train,
        "seen_dev": seen_dev,
        "artist_ood": artist_ood_rows,
        "movement_ood": mv_ood,
    }
    save_json(manifest_p, manifest)
    print(
        f"[V36 Curate] seen_train={len(seen_train)} ({len(tr_styles)} movements), "
        f"seen_dev={len(seen_dev)}, artist_ood={len(artist_ood_rows)}, "
        f"purged_cross={n_purged_cross}, purged_dup={n_purged_dup}",
        flush=True,
    )
    return manifest


def _load_uint8_batch(
    paths_list: list[str],
    workers: int = 8,
    pool: ThreadPoolExecutor | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode & resize 512x512 RGB uint8 arrays cleanly without OpenMP/TF.to_tensor GIL overhead."""
    if not paths_list:
        return np.zeros((0, 512, 512, 3), dtype=np.uint8), np.zeros((0,), dtype=bool)
    def _one(p: str) -> tuple[np.ndarray, bool]:
        try:
            with Image.open(p) as im:
                try:
                    im.draft("RGB", (768, 768))
                except Exception:
                    pass
                im = im.convert("RGB")
                w, h = im.size
                if w != h:
                    side = min(w, h)
                    im = im.crop(((w - side) // 2, (h - side) // 2, (w + side) // 2, (h + side) // 2))
                im512 = im.resize((512, 512), Image.Resampling.BILINEAR)
                return np.asarray(im512, dtype=np.uint8), True
        except Exception:
            return np.zeros((512, 512, 3), dtype=np.uint8), False

    if pool is not None:
        res = list(pool.map(_one, paths_list))
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            res = list(ex.map(_one, paths_list))
    u8 = np.stack([r[0] for r in res], axis=0)
    ok = np.array([r[1] for r in res], dtype=bool)
    return u8, ok


def _load_resized_batch(paths_list: list[str], workers: int = 8, pool: ThreadPoolExecutor | None = None) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """Decode & resize 512x512 RGB tensors (using JPEG draft fast-scale) plus cropped 256x256 views."""
    u8, ok = _load_uint8_batch(paths_list, workers=workers, pool=pool)
    x512 = torch.from_numpy(u8).permute(0, 3, 1, 2).contiguous().float().mul_(1.0 / 255.0)
    cw = int(round(512 * 0.78))
    off = (512 - cw) // 2
    xcrop = F.interpolate(x512[:, :, off:off + cw, (off // 2):(off // 2) + cw], size=(256, 256), mode="bilinear", align_corners=False)
    thumbs = F.interpolate(x512, size=(16, 16), mode="area").permute(0, 2, 3, 1).reshape(len(paths_list), 768).numpy()
    return x512, xcrop, thumbs, ok


@torch.no_grad()
def _extract_split_to_memmap(
    name: str,
    records: list[dict],
    vae,
    cnx_gray: ConvNeXtTinyStage3Extractor,
    cnx_all: ConvNeXtExplicitAndPatchBackbone,
    w_stiefel: torch.Tensor,
    w_l: torch.Tensor,
    b_l: torch.Tensor,
    *,
    with_proxies: bool,
    batch: int = 16,
    device: str = "cuda",
) -> dict[str, str | int]:
    """Resumably stream latents, style stats, Stiefel Q32/Y16, and optional dual proxies into compact SSD memmaps."""
    N = len(records)
    mm_dir = CACHE_DIR / "memmap"
    mm_dir.mkdir(parents=True, exist_ok=True)
    prog_p = CACHE_DIR / f"{name}_progress.json"

    paths_rel = {
        "z_gt": f"runs/v36_art_foundation/cache/memmap/{name}_z_gt.npy",
        "q32": f"runs/v36_art_foundation/cache/memmap/{name}_q32.npy",
        "y16": f"runs/v36_art_foundation/cache/memmap/{name}_y16.npy",
        "stat": f"runs/v36_art_foundation/cache/memmap/{name}_stat.npy",
        "stat_crop": f"runs/v36_art_foundation/cache/memmap/{name}_stat_crop.npy",
        "ok": f"runs/v36_art_foundation/cache/memmap/{name}_ok.npy",
        "thumb16": f"runs/v36_art_foundation/cache/memmap/{name}_thumb16.npy",
    }
    if with_proxies:
        paths_rel["q32_p0"] = f"runs/v36_art_foundation/cache/memmap/{name}_q32_p0.npy"
        paths_rel["q32_p1"] = f"runs/v36_art_foundation/cache/memmap/{name}_q32_p1.npy"
        paths_rel["y16_p0"] = f"runs/v36_art_foundation/cache/memmap/{name}_y16_p0.npy"
        paths_rel["y16_p1"] = f"runs/v36_art_foundation/cache/memmap/{name}_y16_p1.npy"

    if prog_p.exists():
        st_d = read_json(prog_p)
        if st_d.get("done") and int(st_d.get("n", 0)) == N:
            return {"n": N, "paths": paths_rel}
        start_idx = int(st_d.get("next_idx", 0))
        mode = "r+"
    else:
        start_idx = 0
        mode = "w+"

    mm_z = np.lib.format.open_memmap(ROOT / paths_rel["z_gt"], mode=mode, dtype=np.float16, shape=(N, 4, 64, 64))
    mm_q = np.lib.format.open_memmap(ROOT / paths_rel["q32"], mode=mode, dtype=np.float16, shape=(N, 32, 16, 16))
    mm_y = np.lib.format.open_memmap(ROOT / paths_rel["y16"], mode=mode, dtype=np.float16, shape=(N, 1, 16, 16))
    mm_st = np.lib.format.open_memmap(ROOT / paths_rel["stat"], mode=mode, dtype=np.float16, shape=(N, 1561))
    mm_sc = np.lib.format.open_memmap(ROOT / paths_rel["stat_crop"], mode=mode, dtype=np.float16, shape=(N, 1561))
    mm_ok = np.lib.format.open_memmap(ROOT / paths_rel["ok"], mode=mode, dtype=np.uint8, shape=(N,))
    mm_th = np.lib.format.open_memmap(ROOT / paths_rel["thumb16"], mode=mode, dtype=np.float16, shape=(N, 768))

    if with_proxies:
        mm_qp0 = np.lib.format.open_memmap(ROOT / paths_rel["q32_p0"], mode=mode, dtype=np.float16, shape=(N, 32, 16, 16))
        mm_qp1 = np.lib.format.open_memmap(ROOT / paths_rel["q32_p1"], mode=mode, dtype=np.float16, shape=(N, 32, 16, 16))
        mm_yp0 = np.lib.format.open_memmap(ROOT / paths_rel["y16_p0"], mode=mode, dtype=np.float16, shape=(N, 1, 16, 16))
        mm_yp1 = np.lib.format.open_memmap(ROOT / paths_rel["y16_p1"], mode=mode, dtype=np.float16, shape=(N, 1, 16, 16))

    gen0 = torch.Generator(device=device).manual_seed(SEED_V36 + 101)
    gen1 = torch.Generator(device=device).manual_seed(SEED_V36 + 202)
    t0 = time.perf_counter()
    cw_crop = int(round(512 * 0.78))
    off_crop = (512 - cw_crop) // 2
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    # Overlap CPU JPEG uint8 decoding up to 16 batches ahead with GPU crop/encode/stats via continuous producer queue
    ranges = [(s, min(s + batch, N)) for s in range(start_idx, N, batch)]
    with ThreadPoolExecutor(max_workers=8) as decode_pool, torch.inference_mode():
        if not ranges:
            save_json(prog_p, {"done": True, "n": N, "next_idx": N, "utc": utc()})
            return {"n": N, "paths": paths_rel}
        q_buf: queue.Queue = queue.Queue(maxsize=16)
        stop_ev = threading.Event()

        def _producer() -> None:
            for bi, (s, e) in enumerate(ranges):
                if stop_ev.is_set():
                    break
                item = (bi, s, e, _load_uint8_batch([records[i]["path"] for i in range(s, e)], 8, decode_pool))
                q_buf.put(item)
            q_buf.put(None)

        prod_th = threading.Thread(target=_producer, daemon=True)
        prod_th.start()
        try:
            while True:
                pkt = q_buf.get()
                if pkt is None:
                    break
                bi, s, e, (u8, ok) = pkt

                x_d = torch.from_numpy(u8).to(device, non_blocking=True).permute(0, 3, 1, 2).float().mul_(1.0 / 255.0)
                xc_d = F.interpolate(
                    x_d[:, :, off_crop:off_crop + cw_crop, (off_crop // 2):(off_crop // 2) + cw_crop],
                    size=(256, 256), mode="bilinear", align_corners=False,
                )
                thumbs_gpu = F.interpolate(x_d, size=(16, 16), mode="area").permute(0, 2, 3, 1).reshape(e - s, 768).half()

                z = _encode_rgb_scaled(vae, x_d)
                with bf16_autocast(device):
                    st = cnx_all.extract_stat_1561(x_d)
                    stc = cnx_all.extract_stat_1561(xc_d)
                    if with_proxies:
                        x256 = F.interpolate(x_d, size=(256, 256), mode="bilinear", align_corners=False)
                        xp0 = destylize_proxy_v36(x256, gen=gen0)
                        xp1 = destylize_proxy_v36(x256, gen=gen1)
                        cg_cat = cnx_gray.extract_gray_layout(torch.cat([x256, xp0, xp1], dim=0))
                        cg, cg0, cg1 = cg_cat.chunk(3, dim=0)
                    else:
                        cg = cnx_gray.extract_gray_layout(x_d)

                q32 = extract_cnx_q32(cg.float(), w_stiefel)
                y16 = coarse_y16_lcn(((z.float() * w_l).sum(1, keepdim=True) + b_l).clamp(0.0, 1.0))

                mm_z[s:e] = z.cpu().numpy()
                mm_q[s:e] = q32.cpu().half().numpy()
                mm_y[s:e] = y16.cpu().half().numpy()
                mm_st[s:e] = st.float().cpu().half().numpy()
                mm_sc[s:e] = stc.float().cpu().half().numpy()
                mm_ok[s:e] = ok.astype(np.uint8)
                mm_th[s:e] = thumbs_gpu.cpu().numpy()

                if with_proxies:
                    mm_qp0[s:e] = extract_cnx_q32(cg0.float(), w_stiefel).cpu().half().numpy()
                    mm_qp1[s:e] = extract_cnx_q32(cg1.float(), w_stiefel).cpu().half().numpy()
                    y0_256 = 0.299 * xp0[:, 0:1] + 0.587 * xp0[:, 1:2] + 0.114 * xp0[:, 2:3]
                    y1_256 = 0.299 * xp1[:, 0:1] + 0.587 * xp1[:, 1:2] + 0.114 * xp1[:, 2:3]
                    mm_yp0[s:e] = coarse_y16_lcn(F.interpolate(y0_256, (64, 64), mode="area")).cpu().half().numpy()
                    mm_yp1[s:e] = coarse_y16_lcn(F.interpolate(y1_256, (64, 64), mode="area")).cpu().half().numpy()

                if (bi + 1) % 25 == 0 or e == N:
                    save_json(prog_p, {"done": e == N, "n": N, "next_idx": e, "utc": utc()})
                    dt = time.perf_counter() - t0
                    rate = max(1e-3, (e - start_idx) / dt)
                    print(f"[V36 Cache {name}] {e}/{N} ({100.0 * e / N:.1f}%) | {rate:.1f} img/s | ETA {(N - e) / rate / 60.0:.1f}m", flush=True)
        finally:
            stop_ev.set()
            prod_th.join(timeout=5.0)

        for arr in (mm_z, mm_q, mm_y, mm_st, mm_sc, mm_ok, mm_th):
            arr.flush()
        if with_proxies:
            for arr in (mm_qp0, mm_qp1, mm_yp0, mm_yp1):
                arr.flush()

    save_json(prog_p, {"done": True, "n": N, "next_idx": N, "utc": utc()})
    return {"n": N, "paths": paths_rel}


@torch.no_grad()
def _fit_seen_train_priors(
    train_records: list[dict],
    vae,
    cnx_gray: ConvNeXtTinyStage3Extractor,
    cnx_all: ConvNeXtExplicitAndPatchBackbone,
    *,
    n_fit: int = 4096,
    batch: int = 16,
    device: str = "cuda",
) -> dict[str, torch.Tensor]:
    """Fit Stiefel Q32 PCA (384x32), P_Y/P_C/u_y, luma_proj, and initial 1561-D stats strictly on Seen-Train."""
    priors_p = CACHE_DIR / "v36_seen_train_priors.pt"
    if priors_p.exists():
        return torch.load(priors_p, map_location="cpu", weights_only=False)

    idx_fit = np.linspace(0, len(train_records) - 1, min(n_fit, len(train_records)), dtype=np.int64)
    cg_list, z_list, y_list, st_list = [], [], [], []
    with ThreadPoolExecutor(max_workers=16) as pool, torch.inference_mode():
        for s in range(0, len(idx_fit), batch):
            sub = idx_fit[s:min(s + batch, len(idx_fit))]
            x512, _, _, ok = _load_resized_batch([train_records[int(i)]["path"] for i in sub], 16, pool)
            if not ok.any():
                continue
            xd = x512[torch.from_numpy(ok)].to(device)
            z = _encode_rgb_scaled(vae, xd)
            with bf16_autocast(device):
                cg = cnx_gray.extract_gray_layout(xd)
                st = cnx_all.extract_stat_1561(xd)
            cg_list.append(cg.cpu().half())
            st_list.append(st.float().cpu())
            if len(z_list) * batch < 1536:
                y64 = F.interpolate(0.299 * xd[:, 0:1] + 0.587 * xd[:, 1:2] + 0.114 * xd[:, 2:3], (64, 64), mode="area")
                z_list.append(z.float().cpu())
                y_list.append(y64.float().cpu())

    X_gray = torch.cat(cg_list, dim=0).float().permute(0, 2, 3, 1).reshape(-1, 384)
    X_gray = X_gray - X_gray.mean(0, keepdim=True)
    # 384x384 Gram eigendecomposition on GPU matches SVD right-singular vectors cleanly with <1MB workspace!
    cov = (X_gray.T @ X_gray).to(device)
    _, eigvecs = torch.linalg.eigh(cov)
    w_stiefel = eigvecs[:, -32:].flip(dims=(1,)).contiguous()
    del X_gray, cov, eigvecs, cg_list
    torch.cuda.empty_cache()

    st_all = torch.cat(st_list, dim=0)
    stat_mean = st_all.mean(dim=0, keepdim=True)
    stat_std = st_all.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-2)

    z_sub = torch.cat(z_list, dim=0).to(device)
    y64_sub = torch.cat(y_list, dim=0).to(device)
    X_l = torch.cat([z_sub.permute(0, 2, 3, 1).reshape(-1, 4), torch.ones(z_sub.shape[0] * 4096, 1, device=device)], dim=1)
    sol = torch.linalg.lstsq(X_l, y64_sub.permute(0, 2, 3, 1).reshape(-1, 1)).solution
    w_l = sol[:4, 0].view(1, 4, 1, 1).contiguous()
    b_l = sol[4, 0].view(1).contiguous()

    u_y = F.normalize(w_l.view(4).cpu(), dim=0)
    p_y = torch.outer(u_y, u_y)
    p_c = torch.eye(4) - p_y
    u_y = _py_unit_vec(p_y)

    priors = {
        "updated_utc": utc(),
        "fit_n_samples": int(st_all.shape[0]),
        "W_stiefel_cnx": w_stiefel.cpu(),
        "stat_mean": stat_mean.cpu(),
        "stat_std": stat_std.cpu(),
        "luma_w": w_l.cpu(),
        "luma_b": b_l.cpu(),
        "P_Y": p_y.cpu(),
        "P_C": p_c.cpu(),
        "u_y": u_y.cpu(),
    }
    torch.save(priors, priors_p)
    print(f"[V36 Priors] Fitted Stiefel Q32, P_Y/P_C, and 1561-D stats strictly on {st_all.shape[0]} Seen-Train works", flush=True)
    return priors


@torch.no_grad()
def build_v36_caches(
    *,
    holdout_preset: str = "core5",
    min_px: int = 256,
    n_artist_ood: int = 1000,
    n_seen_dev: int = 1000,
    batch: int = 16,
    device: str = "cuda",
) -> dict:
    ready_p = CACHE_DIR / "v36_caches_ready.json"
    if ready_p.exists():
        return read_json(ready_p)

    sanitize_cuda_env()
    seed_all(SEED_V36)

    wikiart_root = find_wikiart_data_root()
    sg_root = _find_sg_root()
    manifest = curate_wikiart_v36(
        wikiart_root,
        sg_root,
        holdout_preset=holdout_preset,
        min_px=min_px,
        n_artist_ood=n_artist_ood,
        n_seen_dev=n_seen_dev,
    )

    vae = _load_sd_vae_encoder(device)
    cnx_gray = ConvNeXtTinyStage3Extractor(device=device)
    cnx_all = ConvNeXtExplicitAndPatchBackbone(device=device)

    # 1. Fit W_stiefel_cnx (384x32), P_Y/P_C, luma_proj, and initial stats on 4,096 Seen-Train works FIRST
    priors = _fit_seen_train_priors(manifest["seen_train"], vae, cnx_gray, cnx_all, n_fit=4096, batch=batch, device=device)
    w_stiefel = priors["W_stiefel_cnx"].to(device).float()
    w_l = priors["luma_w"].to(device).float()
    b_l = priors["luma_b"].to(device).float()

    # 2. Stream ALL Seen-Train works directly into compact (~6.7GB) Q32/Y16/Z/Stat/Proxy memmaps
    tr_info = _extract_split_to_memmap(
        "seen_train", manifest["seen_train"], vae, cnx_gray, cnx_all,
        w_stiefel, w_l, b_l, with_proxies=True, batch=batch, device=device,
    )
    N_tr = int(tr_info["n"])

    # 3. Build fast Seen-Train pairing tables + update global stat_mean/std over ALL Seen-Train valid rows
    meta_pt = CACHE_DIR / "seen_train_index.pt"
    if not meta_pt.exists():
        mm_ok = np.array(np.lib.format.open_memmap(ROOT / tr_info["paths"]["ok"], mode="r", dtype=np.uint8, shape=(N_tr,)))
        _, _, banned_thumbs = _load_banned_signatures(sg_root)
        if banned_thumbs.shape[0] > 0:
            mm_th = np.lib.format.open_memmap(ROOT / tr_info["paths"]["thumb16"], mode="r", dtype=np.float16, shape=(N_tr, 768))
            bt = torch.from_numpy(banned_thumbs).to(device).float()
            n_thumb_purged = 0
            for s in range(0, N_tr, 2048):
                e = min(s + 2048, N_tr)
                tv = torch.from_numpy(np.array(mm_th[s:e])).to(device).float()
                d_min = torch.cdist(tv, bt, p=1.0).min(dim=1).values / 768.0
                hit = (d_min < 0.032).cpu().numpy()
                if hit.any():
                    mm_ok[s:e][hit] = 0
                    n_thumb_purged += int(hit.sum())
            print(f"[V36 Dedup] Purged {n_thumb_purged} visual thumbnail collisions vs test750/refs/SG", flush=True)
        mm_st = np.array(np.lib.format.open_memmap(ROOT / tr_info["paths"]["stat"], mode="r", dtype=np.float16, shape=(N_tr, 1561))).astype(np.float32)
        valid_mask = mm_ok > 0
        st_valid = torch.from_numpy(mm_st[valid_mask])
        priors["stat_mean"] = st_valid.mean(dim=0, keepdim=True)
        priors["stat_std"] = st_valid.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-2)
        priors["fit_n_samples"] = int(valid_mask.sum())
        torch.save(priors, CACHE_DIR / "v36_seen_train_priors.pt")
        st_norm = (mm_st - priors["stat_mean"].numpy()) / priors["stat_std"].numpy()
        st_norm = st_norm / np.linalg.norm(st_norm, axis=1, keepdims=True).clip(min=1e-4)

        styles = [r["style"] for r in manifest["seen_train"]]
        style_cnt: dict[str, int] = {}
        for i, st in enumerate(styles):
            if mm_ok[i] > 0:
                style_cnt[st] = style_cnt.get(st, 0) + 1

        by_as: dict[tuple[str, str], list[int]] = {}
        by_a: dict[str, list[int]] = {}
        for i, r in enumerate(manifest["seen_train"]):
            if mm_ok[i] == 0:
                continue
            ak = r["artist"].casefold()
            if ak == "unknown":
                continue
            by_as.setdefault((ak, r["style"]), []).append(i)
            by_a.setdefault(ak, []).append(i)

        artist_nbrs = np.full((N_tr, 8), -1, dtype=np.int32)
        for ak, idxs in by_a.items():
            if len(idxs) < 2:
                continue
            for i in idxs:
                pool = by_as.get((ak, styles[i]), [])
                if len(pool) < 2:
                    pool = idxs
                others = [j for j in pool if j != i]
                if len(others) <= 8:
                    artist_nbrs[i, :len(others)] = others
                else:
                    sims = st_norm[others] @ st_norm[i]
                    top_pos = np.argpartition(sims, -min(16, len(others)))[-min(16, len(others)):]
                    cands = [others[p] for p in top_pos if sims[p] < 0.97] or [others[p] for p in top_pos]
                    cands = sorted(cands, key=lambda j: float(st_norm[j] @ st_norm[i]), reverse=True)[:8]
                    artist_nbrs[i, :len(cands)] = cands

        p_unif = valid_mask.astype(np.float64)
        p_unif /= p_unif.sum()
        inv_sqrt = np.zeros(N_tr, dtype=np.float64)
        for i, st in enumerate(styles):
            if valid_mask[i]:
                inv_sqrt[i] = 1.0 / math.sqrt(float(max(1, style_cnt[st])))
        avg_w = inv_sqrt[valid_mask].mean()
        inv_sqrt = np.minimum(inv_sqrt, avg_w * 8.0)
        p_sqrt = inv_sqrt / inv_sqrt.sum()
        sample_probs = 0.80 * p_unif + 0.20 * p_sqrt
        sample_probs /= sample_probs.sum()

        torch.save({
            "n_seen": N_tr,
            "valid_idx": torch.from_numpy(np.flatnonzero(valid_mask).astype(np.int64)),
            "artist_nbrs": torch.from_numpy(artist_nbrs),
            "sample_probs": torch.from_numpy(sample_probs),
            "style_counts": style_cnt,
            "q32_rel": tr_info["paths"]["q32"],
            "q32_p0_rel": tr_info["paths"]["q32_p0"],
            "q32_p1_rel": tr_info["paths"]["q32_p1"],
            "y16_rel": tr_info["paths"]["y16"],
            "y16_p0_rel": tr_info["paths"]["y16_p0"],
            "y16_p1_rel": tr_info["paths"]["y16_p1"],
            "z_gt_rel": tr_info["paths"]["z_gt"],
            "stat_rel": tr_info["paths"]["stat"],
            "stat_crop_rel": tr_info["paths"]["stat_crop"],
        }, meta_pt)

    # 5. Precompute lightweight eval splits (Seen-Dev, Artist-OOD, Movement-OOD, Core5 test750+refs150)
    dev_info = _extract_split_to_memmap(
        "seen_dev", manifest["seen_dev"], vae, cnx_gray, cnx_all,
        w_stiefel, w_l, b_l, with_proxies=False, batch=batch, device=device,
    )
    art_info = _extract_split_to_memmap(
        "artist_ood", manifest["artist_ood"], vae, cnx_gray, cnx_all,
        w_stiefel, w_l, b_l, with_proxies=False, batch=batch, device=device,
    )
    mv_info = _extract_split_to_memmap(
        "movement_ood", manifest["movement_ood"], vae, cnx_gray, cnx_all,
        w_stiefel, w_l, b_l, with_proxies=False, batch=batch, device=device,
    )

    # Precompute test750 Q32/Y16 + refs150 z_ref & 1561-D stats + SG-25 using V36's own Seen-Train Stiefel Q32!
    eval_assets_p = CACHE_DIR / "v36_eval_assets.pt"
    if not eval_assets_p.exists():
        _, z_src_750, _ = tasks_and_sources()
        src_paths = [str(ROOT / t["source_rel_path"]) for t in read_json(ROOT / "eval_protocol/tasks_750.json")["tasks"]]
        # Deduplicate 150 unique sources (order preserved across 750 = 5 targets per source or similar)
        unique_src_paths = sorted(set(src_paths))
        src_map = {p: i for i, p in enumerate(unique_src_paths)}
        xs = torch.cat([_load_resized_batch([unique_src_paths[i] for i in range(s, min(s + 32, len(unique_src_paths)))])[0]
                        for s in range(0, len(unique_src_paths), 32)], dim=0)
        q_u, y_u = [], []
        for s in range(0, xs.shape[0], 32):
            xb = xs[s:s + 32].to(device)
            cg = cnx_gray.extract_gray_layout(xb)
            q_u.append(extract_cnx_q32(cg.float(), w_stiefel).half().cpu())
        q_unique = torch.cat(q_u, dim=0)
        q32_750 = torch.stack([q_unique[src_map[p]] for p in src_paths], dim=0)
        y16_750 = coarse_y16_lcn(((z_src_750.to(device).float() * w_l).sum(1, keepdim=True) + b_l).clamp(0.0, 1.0)).half().cpu()

        refs_d = read_json(ROOT / "eval_protocol/references_v1.json")["references"]
        ref_recs = [r for st_k in sorted(refs_d.keys()) for r in refs_d[st_k]]
        xr_all = torch.cat([_load_resized_batch([str(ROOT / r["rel_path"]) for r in ref_recs[s:min(s + 32, len(ref_recs))]])[0]
                            for s in range(0, len(ref_recs), 32)], dim=0)
        z_refs, st_refs = [], []
        for s in range(0, xr_all.shape[0], 32):
            xb = xr_all[s:s + 32].to(device)
            z_refs.append(_encode_rgb_scaled(vae, xb).cpu())
            with bf16_autocast(device):
                st_refs.append(cnx_all.extract_stat_1561(xb).float().half().cpu())
        refs_pack = {
            "records": ref_recs,
            "z_ref": torch.cat(z_refs, dim=0),
            "stat_1561": torch.cat(st_refs, dim=0),
        }

        sg_pack = None
        if sg_root is not None and (sg_root / "content").exists() and (sg_root / "style").exists():
            c_files = sorted([p for p in (sg_root / "content").rglob("*.*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")])[:25]
            s_files = []
            for sdir in sorted([d for d in (sg_root / "style").iterdir() if d.is_dir()]):
                imgs = sorted([p for p in sdir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
                if imgs:
                    s_files.append(imgs[0])
            if not s_files:
                s_files = sorted([p for p in (sg_root / "style").rglob("*.*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
            s_files = s_files[:25]
            if c_files and s_files:
                xc_sg = _load_resized_batch([str(p) for p in c_files])[0].to(device)
                xs_sg = _load_resized_batch([str(p) for p in s_files])[0].to(device)
                sg_zc = torch.cat([_encode_rgb_scaled(vae, xc_sg[i:i + 16]).cpu() for i in range(0, len(c_files), 16)], dim=0)
                sg_zr = torch.cat([_encode_rgb_scaled(vae, xs_sg[i:i + 16]).cpu() for i in range(0, len(s_files), 16)], dim=0)
                sg_q = extract_cnx_q32(cnx_gray.extract_gray_layout(xc_sg).float(), w_stiefel).half().cpu()
                sg_y16 = coarse_y16_lcn(((sg_zc.to(device).float() * w_l).sum(1, keepdim=True) + b_l).clamp(0.0, 1.0)).half().cpu()
                with bf16_autocast(device):
                    sg_st = cnx_all.extract_stat_1561(xs_sg).float().half().cpu()
                sg_pack = {
                    "c_paths": [str(p) for p in c_files],
                    "s_paths": [str(p) for p in s_files],
                    "c_z": sg_zc,
                    "c_q32": sg_q,
                    "c_y16": sg_y16,
                    "s_z": sg_zr,
                    "s_stat": sg_st,
                }

        torch.save({
            "q32_750": q32_750,
            "y16_750": y16_750,
            "refs150": refs_pack,
            "sg25": sg_pack,
        }, eval_assets_p)

    del vae, cnx_gray, cnx_all
    gc.collect()
    torch.cuda.empty_cache()

    ready = {
        "utc": utc(),
        "holdout_preset": holdout_preset,
        "holdout_movements": manifest["holdout_movements"],
        "n_seen_train": N_tr,
        "n_seen_dev": int(dev_info["n"]),
        "n_artist_ood": int(art_info["n"]),
        "n_movement_ood": int(mv_info["n"]),
        "dev_info": dev_info,
        "art_info": art_info,
        "mv_info": mv_info,
    }
    save_json(ready_p, ready)
    print(f"[V36 P0 Complete] Ready: {ready_p}", flush=True)
    return ready


def main():
    import math as _math  # noqa: F401
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout-preset", default="core5", choices=list(HOLDOUT_PRESETS.keys()))
    ap.add_argument("--min-px", type=int, default=256)
    ap.add_argument("--n-artist-ood", type=int, default=1000)
    ap.add_argument("--n-seen-dev", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    build_v36_caches(
        holdout_preset=a.holdout_preset,
        min_px=a.min_px,
        n_artist_ood=a.n_artist_ood,
        n_seen_dev=a.n_seen_dev,
        batch=a.batch,
        device=a.device,
    )


if __name__ == "__main__":
    main()
