"""ArtFID (FID_style + LPIPS content) for WEAVE 750-task image sets.

Protocol adaptation of tools/legacy_utils/artfid_metric.py +
targetwise_artfid_summary.py to the tasks_750 protocol:

  * gen images grouped by TARGET style (~150 imgs/style incl. same-style
    tasks); FID_style = Fréchet distance vs real paintings of that style
    (datasets/distinct5_512/train/<style>, all 1000 imgs/style).
  * content distance = mean LPIPS-Alex between each generated image and
    its task source image (data/test/... from tasks_750.json).
  * per-style art_fid_s = (1 + FID_s) * (1 + content_s); arm score =
    mean over the 5 target styles (mean_of_target_means).

Naming modes:
  hash   : "{i:04d}_{hash}.png" where i = index into tasks_750.json
           (V11B out_*, Hist2Style, StyleGallery images)
  taskid : "{task_id with '/'->'__'}.png"  (V15 png, V10/V9 eval/images)
           matched via ASCII-folded stems for mojibake task_ids.

Outputs (incremental, resume-safe):
  runs/artfid/artfid_results.json   per-arm metrics
  runs/artfid/feats_<arm>.npz       cached gen features per style
  runs/artfid/ref_stats.npz         per-style ref (mu, sigma)

Usage:
  WEAVE_EVAL_CACHE_ROOT=/path/to/eval_cache python tools/artfid_750.py \
      [--device cuda] [--arms V15_W_dim96_conv_base.t00 ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# --- mmap torch.load: avoids commit-charge AV on memory-tight hosts ------
_orig_load = torch.load


def _patched_load(f, *args, **kwargs):
    use_mmap = isinstance(f, (str, Path)) and kwargs.get("mmap") is None
    if use_mmap:
        kwargs["mmap"] = True
    try:
        return _orig_load(f, *args, **kwargs)
    except RuntimeError as exc:
        if use_mmap and "mmap" in str(exc).lower():
            kwargs.pop("mmap", None)
            return _orig_load(f, *args, **kwargs)
        raise


torch.load = _patched_load
# -------------------------------------------------------------------------

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
TASKS = json.loads((REPO / "eval_protocol/tasks_750.json").read_text())["tasks"]
OUT = REPO / "runs/artfid"
REF_DIR = REPO / "datasets/distinct5_512/train"
ARTFID_CKPT = Path(os.environ.get("WEAVE_ARTFID_CKPT", "") or (REPO.parent / "eval_cache/artfid/art_inception.pth"))


def ascii_key(s: str) -> str:
    return re.sub(r"[^\x00-\x7f]", "",
                  unicodedata.normalize("NFC", s))


def task_png_name(tid: str) -> str:
    return tid.replace("/", "__") + ".png"


_src_cache: dict[str, Path | None] = {}


def resolve_src(task: dict) -> Path | None:
    """source_rel_path may carry mojibake (joaquín etc.); fall back to
    NFC/NFD normalization then ASCII-folded directory scan."""
    rel = task["source_rel_path"]
    if rel in _src_cache:
        return _src_cache[rel]
    cands = [REPO / rel]
    nfc = unicodedata.normalize("NFC", rel)
    nfd = unicodedata.normalize("NFD", rel)
    if nfc != rel:
        cands.append(REPO / nfc)
    if nfd != rel:
        cands.append(REPO / nfd)
    for c in cands:
        if c.is_file():
            _src_cache[rel] = c
            return c
    # best fuzzy match in the parent dir (mojibake accents differ beyond
    # ASCII folding, e.g. 'joaquín' vs 'joaquã­n')
    import difflib
    p = REPO / rel
    parent, stem = p.parent, p.stem
    if parent.is_dir():
        want = ascii_key(stem)
        best, best_r = None, 0.0
        for f in parent.glob("*.jpg"):
            r = difflib.SequenceMatcher(
                None, want, ascii_key(f.stem)).ratio()
            if r > best_r:
                best, best_r = f, r
        if best is not None and best_r > 0.80:
            _src_cache[rel] = best
            return best
    _src_cache[rel] = None
    return None


def hash_png_name(i: int, tid: str) -> str:
    return f"{i:04d}_" + hashlib.sha256(tid.encode()).hexdigest()[:10] + ".png"


def build_index(dirs: list[Path], mode: str) -> dict[int, Path]:
    """task index -> image path."""
    idx: dict[int, Path] = {}
    if mode == "hash":
        for d in dirs:
            for p in d.glob("*.png"):
                m = re.match(r"^(\d{4})_", p.name)
                if m:
                    idx.setdefault(int(m.group(1)), p)
    elif mode == "d5":
        # docs/baseline/D5-512/jpg/<method>/*.jpg — naming schemes differ
        # per method ('__to__', '_to_', extra prefixes). Match by
        # containment: filename must contain the task's source stem
        # (task_id part[1]) followed later by the target style.
        stems = [t["task_id"].split("/") for t in TASKS]
        for d in dirs:
            files = list(d.glob("*.jpg"))
            for i, parts in enumerate(stems):
                if len(parts) != 3 or i in idx:
                    continue
                stem, tgt = parts[1], parts[2]
                for f in files:
                    n = f.name
                    pos = n.find(stem)
                    if pos < 0:
                        pos = ascii_key(n).find(ascii_key(stem))
                        if pos < 0:
                            continue
                        tail = ascii_key(n)[pos + len(ascii_key(stem)):]
                    else:
                        tail = n[pos + len(stem):]
                    if tgt in tail:
                        idx[i] = f
                        break
    else:  # taskid
        for d in dirs:
            nmap = {ascii_key(p.stem): p for p in d.glob("*.png")}
            for i, t in enumerate(TASKS):
                p = nmap.get(ascii_key(task_png_name(t["task_id"])[:-4]))
                if p is not None:
                    idx.setdefault(i, p)
    return idx


def arm_specs() -> list[tuple[str, list[Path], str]]:
    specs = []
    # V15 Group A (only arm with returned PNGs)
    for lev in ["t0_0.0", "t0_0.4", "t0_0.7"]:
        specs.append((f"V15_W_dim96_conv_base.{lev[-3:]}",
                      [REPO / f"runs/v15/eval/png/W_dim96_conv_base/{lev}"],
                      "taskid"))
    # reproduced baselines — key points, computed early
    specs.append(("Baseline.Hist2Style",
                  [REPO / "docs/baseline/Hist2Style/images"], "hash"))
    specs.append(("Baseline.StyleGallery",
                  [REPO / "docs/baseline/StyleGallery/images"], "hash"))
    # V11-B arms (hashed names)
    for d in sorted((REPO / "runs/v11b").glob("out_*/")):
        specs.append((f"V11B.{d.name[4:]}", [d], "hash"))
    # V10 arms
    for d in sorted((REPO / "runs/v10_local_4070/all").glob("*/eval/images")):
        specs.append((f"V10.{d.parent.parent.name}", [d], "taskid"))
    # WEAVE LD family (dev+test split, same taskid naming)
    for arm in ["LD0", "LD1", "LD2", "LD3", "LD4"]:
        specs.append((f"WEAVE.{arm}",
                      [REPO / f"runs/v9_local_4070/{s}/{arm}/eval/images"
                       for s in ("dev", "test")], "taskid"))
    # V8 arms with locally available eval images (F2 family only;
    # other V8 outputs live on the 3060)
    for arm in ["F2.e", "F2.s1", "F2.s4"]:
        specs.append((f"V8.{arm}",
                      [REPO / f"runs/v8/{arm}/eval/images"], "taskid"))
    # published-baseline 750 outputs archived under docs/baseline/D5-512
    d5 = REPO / "docs/baseline/D5-512/jpg"
    # stylealigned/zstar ran a DIFFERENT source set (only ~20/750 stems
    # overlap) — excluded; seedream 720/750, styleshot 745/750 partial.
    D5_SKIP = {"stylealigned", "zstar"}
    if d5.is_dir():
        for m in sorted(d5.iterdir()):
            if m.is_dir() and m.name not in D5_SKIP:
                specs.append((f"Baseline.D5.{m.name}", [m], "d5"))
    return specs


from concurrent.futures import ThreadPoolExecutor  # noqa: E402

_POOL = ThreadPoolExecutor(max_workers=10)


def _load_many(paths, workers=8):
    """Parallel PIL decode+resize (releases GIL); preserves order."""
    from tools.legacy_utils.artfid_metric import load_image_tensor
    return list(_POOL.map(load_image_tensor, paths))


def collect_feats_threaded(paths, *, model, batch_size, device,
                           workers=8):
    """Decode ALL images upfront in parallel (~150/style ≈ 150MB), then
    stream batches to GPU — GPU never waits on PIL."""
    import torch
    feats = []
    bs = max(1, int(batch_size))
    # decode in slabs of 256 (~256MB) to bound RAM on big ref sets
    for d0 in range(0, len(paths), 256):
        tensors = _load_many(paths[d0:d0 + 256])
        for s in range(0, len(tensors), bs):
            imgs = torch.stack(tensors[s:s + bs]).to(
                device=device, dtype=torch.float32)
            with torch.no_grad():
                out = model(imgs)
            feats.append(out.detach().cpu().numpy())
            del imgs
        del tensors
    if not feats:
        return np.empty((0, 2048), dtype=np.float32)
    return np.concatenate(feats, axis=0)


def content_dists_threaded(gen_paths, src_paths, *, loss_fn, batch_size,
                           device, workers=8):
    """Threaded-decode variant of compute_artfid_content_distances_from_paths."""
    import torch
    dists = []
    bs = max(1, int(batch_size))
    # decode slabs of 128 pairs upfront (parallel), then forward batches
    for d0 in range(0, len(gen_paths), 128):
        gc, sc = gen_paths[d0:d0 + 128], src_paths[d0:d0 + 128]
        xs = _load_many(gc)
        ys = _load_many(sc)
        for s in range(0, len(xs), bs):
            x = torch.stack(xs[s:s + bs])
            y = torch.stack(ys[s:s + bs])
            try:
                with torch.no_grad():
                    cur = loss_fn(
                        (x.to(device=device, dtype=torch.float32) * 2 - 1),
                        (y.to(device=device, dtype=torch.float32) * 2 - 1))
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                torch.cuda.empty_cache()
                with torch.no_grad():
                    cur = loss_fn((x * 2 - 1), (y * 2 - 1))
            dists.append(cur.detach().view(-1).cpu().numpy())
            del x, y
        del xs, ys
    return np.concatenate(dists) if dists else None


def ref_paths(style: str, limit: int = 0) -> list[Path]:
    fs = sorted((REF_DIR / style).glob("*.jpg"))
    return fs[:limit] if limit else fs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--ref-per-style", type=int, default=0,
                    help="0 = all 1000 refs/style")
    ap.add_argument("--arms", nargs="*", default=None)
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"

    from tools.legacy_utils.artfid_metric import (
        ArtFIDFeatureExtractor, compute_artfid_content_distance_from_paths,
        compute_artfid_fid_from_features, frechet_distance,
        load_artfid_lpips, load_image_tensor,
    )

    feat_model = ArtFIDFeatureExtractor(ARTFID_CKPT, device=device)
    feat_model.eval()
    lpips_fn = load_artfid_lpips(device=device)

    # ---- per-style reference stats (cached) ------------------------------
    stats_path = OUT / "ref_stats.npz"
    ref_stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if stats_path.is_file():
        z = np.load(stats_path)
        for s in STYLES:
            ref_stats[s] = (z[f"{s}__mu"], z[f"{s}__sigma"])
    else:
        for s in STYLES:
            paths = [str(p) for p in ref_paths(s, a.ref_per_style)]
            t0 = time.time()
            feats = collect_feats_threaded(
                paths, model=feat_model, batch_size=a.batch, device=device)
            mu = feats.mean(0)
            sigma = np.cov(feats, rowvar=False)
            ref_stats[s] = (mu, sigma)
            print(f"ref {s}: {feats.shape[0]} imgs "
                  f"({time.time()-t0:.0f}s)", flush=True)
        np.savez(stats_path, **{f"{s}__mu": ref_stats[s][0]
                                for s in STYLES},
                 **{f"{s}__sigma": ref_stats[s][1] for s in STYLES})

    results_path = OUT / "artfid_results.json"
    results = json.loads(results_path.read_text()) \
        if results_path.is_file() else {}

    specs = arm_specs()
    if a.arms:
        keep = set(a.arms)
        specs = [s for s in specs if s[0] in keep]

    for name, dirs, mode in specs:
        if name in results and results[name].get("status") == "done":
            print(f"skip {name} (done)", flush=True)
            continue
        t0 = time.time()
        idx = build_index(dirs, mode)
        missing = [i for i in range(len(TASKS)) if i not in idx]
        if missing:
            print(f"{name}: {len(missing)} images missing "
                  f"(first idx {missing[:3]})", flush=True)
        # group gen paths by target style, keep task order for LPIPS
        by_style: dict[str, list[tuple[int, Path]]] = {s: [] for s in STYLES}
        for i, p in sorted(idx.items()):
            by_style[TASKS[i]["target_style"]].append((i, p))

        # FID per style (gen feats cached per arm)
        cache = OUT / f"feats_{re.sub(r'[^A-Za-z0-9_.-]', '_', name)}.npz"
        gen_feats: dict[str, np.ndarray] = {}
        if cache.is_file():
            z = np.load(cache)
            for s in STYLES:
                gen_feats[s] = z[s]
        else:
            for s in STYLES:
                paths = [str(p) for _, p in by_style[s]]
                if len(paths) < 2:
                    continue
                gen_feats[s] = collect_feats_threaded(
                    paths, model=feat_model, batch_size=a.batch,
                    device=device)
            np.savez(cache, **gen_feats)

        fid_s, content_s = {}, {}
        all_pairs: list[tuple[str, str]] = []
        n_src_missing = 0
        for s in STYLES:
            gf = gen_feats.get(s)
            if gf is None or gf.shape[0] < 2:
                continue
            fid_s[s] = compute_artfid_fid_from_features(
                gf, ref_stats=ref_stats[s])
            pairs = []
            for i, p in by_style[s]:
                sp = resolve_src(TASKS[i])
                if sp is None:
                    n_src_missing += 1
                    continue
                pairs.append((str(p), str(sp)))
            all_pairs += [(s, g, src) for g, src in pairs]
        # content distance in one pass, grouped per style afterwards
        for s in STYLES:
            ps = [(g, src) for ss, g, src in all_pairs if ss == s]
            if not ps:
                continue
            d = content_dists_threaded(
                [g for g, _ in ps], [src for _, src in ps],
                loss_fn=lpips_fn, batch_size=a.batch, device=device)
            content_s[s] = (float(np.mean(d))
                            if d is not None and d.size else None)

        art_fid = {s: (1.0 + fid_s[s]) * (1.0 + content_s[s])
                   for s in fid_s if content_s.get(s) is not None}
        score = float(np.mean(list(art_fid.values()))) if art_fid else None
        results[name] = {
            "status": "done",
            "n_images": len(idx),
            "n_missing": len(missing) + n_src_missing,
            "fid_style": {k: float(v) for k, v in fid_s.items()},
            "content_lpips": {k: (None if v is None else float(v))
                              for k, v in content_s.items()},
            "art_fid_per_style": {k: float(v) for k, v in art_fid.items()},
            "art_fid_mean_of_target_means": score,
            "seconds": round(time.time() - t0, 1),
        }
        results_path.write_text(json.dumps(results, indent=1))
        print(f"{name}: art_fid={score} "
              f"({results[name]['seconds']}s)", flush=True)

    print("ARTFID_DONE")


if __name__ == "__main__":
    main()
