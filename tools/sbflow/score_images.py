"""Consumer process: read generated images from disk and compute style metrics.

Runs independently of `sbflow.generate`, so a generation pass on the GPU is never
blocked by metric work. Image decoding is done on a CPU thread pool and prefetched
ahead of the feature extractor, which is what makes this side cheap.

Input  : a directory written by `sbflow.generate` (per-checkpoint `manifest.json`).
Output : one `style_scores.json` per checkpoint with the same schema as the legacy
         `eval_test600.score` (`protocol` / `aggregate` / `rows`), so
         `compare_pairs.py`, `per_direction_table.py` and `make_table.py` keep working.

Metrics
  csd_dT / csd_dS        style-space shift toward the target / away from the source
  target_margin          target CSD score minus the best competing style
  target_top1            fraction whose argmax CSD style is the target
  csd_mmd2_unbiased      distribution match against the frozen reference pool
  lpips_to_input         severe-destruction diagnostic only (not an objective)
  dino_c                 DINOv2 CLS cosine with the source (content preservation)

Usage
    python -m sbflow.score_images --repo-root . \
        --gen runs/blog_sbflow/gen/Mid8M_b16 \
        --out runs/blog_sbflow/scores/Mid8M_b16 --batch 16
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
KEYS = ("csd_dT", "csd_dS", "target_margin", "target_top1", "lpips_to_input", "dino_c")


# --------------------------------------------------------------------------- #
# CPU-side image loading
# --------------------------------------------------------------------------- #
def load_rgb(path: Path, size: Optional[int] = None) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        if size is not None and im.size != (size, size):
            im = im.resize((size, size), Image.Resampling.BICUBIC)
        return np.asarray(im, dtype=np.uint8).copy()


def prefetch(paths: Sequence[Path], workers: int, size: Optional[int] = None) -> List[np.ndarray]:
    if workers <= 0:
        return [load_rgb(p, size) for p in paths]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda p: load_rgb(p, size), paths))


def u8_to_float(u8: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(u8.astype(np.float32)).permute(2, 0, 1).to(device) / 127.5 - 1


def mmd2(gen: np.ndarray, real: np.ndarray, gamma: float) -> float:
    x = torch.as_tensor(gen, dtype=torch.float32)
    y = torch.as_tensor(real, dtype=torch.float32)
    n, m = len(x), len(y)
    if n < 2 or m < 2:
        return float("nan")

    def k(a, b):
        return torch.exp(-gamma * torch.cdist(a, b).square())

    xx, yy = k(x, x), k(y, y)
    return float((xx.sum() - xx.diagonal().sum()) / (n * (n - 1))
                 + (yy.sum() - yy.diagonal().sum()) / (m * (m - 1))
                 - 2 * k(x, y).mean())


def swd(gen: np.ndarray, real: np.ndarray, n_proj: int = 128, seed: int = 0) -> float:
    g = torch.as_tensor(gen, dtype=torch.float32)
    r = torch.as_tensor(real, dtype=torch.float32)
    if len(g) < 2 or len(r) < 2:
        return float("nan")
    proj = torch.nn.functional.normalize(
        torch.randn(g.shape[1], n_proj, generator=torch.Generator().manual_seed(seed)), dim=0)
    pg = torch.sort(g @ proj, dim=0).values.numpy()
    pr = torch.sort(r @ proj, dim=0).values.numpy()
    q = np.linspace(0.0, 1.0, 256)
    xg = np.linspace(0.0, 1.0, pg.shape[0])
    xr = np.linspace(0.0, 1.0, pr.shape[0])
    return float(np.mean([np.abs(np.interp(q, xg, pg[:, p]) - np.interp(q, xr, pr[:, p])).mean()
                          for p in range(n_proj)]))


# --------------------------------------------------------------------------- #
class ImageScorer:
    def __init__(self, repo: Path, device: str):
        self.repo = repo
        self.device = device
        sys.path.insert(0, str(repo))
        from tools.evaluation_v2.features import EvaluatorModels, build_reference_features_cache

        t0 = time.perf_counter()
        self.ev = EvaluatorModels(device)
        self.refs = build_reference_features_cache(self.ev)
        self.gamma = float(self.refs["rbf_gamma"])
        proto = self.ev.proto_csd
        self.proto = (proto.detach().float().cpu().numpy() if torch.is_tensor(proto)
                      else np.asarray(proto, dtype=np.float32))
        self.load_sec = time.perf_counter() - t0
        self.timing: Dict[str, float] = {}

    def _rows_for(self, images: Sequence[np.ndarray], tasks: Sequence[dict],
                  sources: Sequence[np.ndarray], variant: str) -> List[dict]:
        pils = [Image.fromarray(a) for a in images]
        feats = self.ev.extract_image_features(pils)
        src_feats = self.ev.extract_image_features([Image.fromarray(a) for a in sources])
        csd = feats["csd"].numpy()
        src_csd = src_feats["csd"].numpy()
        values = csd @ self.proto.T
        src_values = src_csd @ self.proto.T
        dino_c = torch.nn.functional.cosine_similarity(
            feats["dino_cls"].float(), src_feats["dino_cls"].float(), dim=1).numpy()

        rows = []
        for j, t in enumerate(tasks):
            tid = STYLES.index(t["target_style"])
            sid = STYLES.index(t["source_style"])
            other = np.delete(values[j], tid)
            a = u8_to_float(images[j], self.device)
            b = u8_to_float(sources[j], self.device)
            with torch.inference_mode():
                lp = float(self.ev.lpips(a[None], b[None]).mean().item())
            rows.append({
                "task_id": t["task_id"], "source": t["source_style"], "target": t["target_style"],
                "variant": variant,
                "csd_dT": float(values[j, tid] - src_values[j, tid]),
                "csd_dS": float(values[j, sid] - src_values[j, sid]),
                "target_margin": float(values[j, tid] - other.max()),
                "target_top1": int(values[j].argmax() == tid),
                "lpips_to_input": lp,
                "dino_c": float(dino_c[j]),
                "csd_embedding": csd[j].tolist(),
            })
        return rows

    def aggregate(self, rows: List[dict], src_style: str, tgt_style: str) -> dict:
        agg = {}
        for tag in ("generated", "dcae_noop"):
            one = [r for r in rows if r["variant"] == tag]
            if not one:
                continue
            agg[tag] = {}
            for style in (src_style, tgt_style):
                vals = [v for v in one if v["target"] == style]
                if not vals:
                    continue
                feat = np.asarray([v["csd_embedding"] for v in vals], dtype=np.float32)
                real = self.refs["styles"][style]["csd"]
                agg[tag][style] = {k: float(np.mean([v[k] for v in vals])) for k in KEYS}
                agg[tag][style]["csd_mmd2_unbiased"] = mmd2(feat, real, self.gamma)
                agg[tag][style]["csd_swd"] = swd(feat, real)
                agg[tag][style]["n"] = len(vals)
            agg[tag]["cross_overall"] = {k: float(np.mean([v[k] for v in one])) for k in KEYS}
            feat = np.asarray([v["csd_embedding"] for v in one], dtype=np.float32)
            real = self.refs["styles"][tgt_style]["csd"]
            agg[tag]["cross_overall"]["csd_mmd2_unbiased"] = mmd2(feat, real, self.gamma)
            agg[tag]["cross_overall"]["csd_swd"] = swd(feat, real)
        return agg


# --------------------------------------------------------------------------- #
def score_checkpoint(scorer: ImageScorer, tag_dir: Path, out_dir: Path, batch: int,
                     workers: int, with_noop: bool) -> dict:
    t_start = time.perf_counter()
    man = json.loads((tag_dir / "manifest.json").read_text(encoding="utf-8"))
    info, tasks = man["info"], man["tasks"]
    src_style, tgt_style = info["source_style"], info["target_style"]

    # CPU prefetch of every image we need, in parallel with nothing else: cheap.
    t0 = time.perf_counter()
    gen_paths = [tag_dir / t["image"] for t in tasks]
    src_paths = [scorer.repo / t["source_rel_path"] for t in tasks]
    gen_imgs = prefetch(gen_paths, workers)
    gen_size = gen_imgs[0].shape[0]
    src_imgs = prefetch(src_paths, workers, size=gen_size)
    if with_noop:
        noop_paths = [tag_dir.parent / "_dcae_noop" / Path(t["image"]).name for t in tasks]
        noop_imgs = prefetch(noop_paths, workers) if all(p.exists() for p in noop_paths) else None
    else:
        noop_imgs = None
    scorer.timing["io_load"] = scorer.timing.get("io_load", 0.0) + (time.perf_counter() - t0)

    rows: List[dict] = []
    t0 = time.perf_counter()
    for start in range(0, len(tasks), batch):
        sl = slice(start, start + batch)
        rows.extend(scorer._rows_for(gen_imgs[sl], tasks[sl], src_imgs[sl], "generated"))
    scorer.timing["metrics_generated"] = scorer.timing.get("metrics_generated", 0.0) + (time.perf_counter() - t0)

    if noop_imgs is not None:
        t0 = time.perf_counter()
        for start in range(0, len(tasks), batch):
            sl = slice(start, start + batch)
            rows.extend(scorer._rows_for(noop_imgs[sl], tasks[sl], src_imgs[sl], "dcae_noop"))
        scorer.timing["metrics_noop"] = scorer.timing.get("metrics_noop", 0.0) + (time.perf_counter() - t0)

    agg = scorer.aggregate(rows, src_style, tgt_style)
    wall = time.perf_counter() - t_start
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": {
            "source": "image-consumer path (reads PNGs written by sbflow.generate)",
            "n_tasks": len(tasks), "source_style": src_style, "target_style": tgt_style,
            "network_nfe": info["network_nfe"], "solver": info["solver"],
            "solver_steps": info["solver_steps"], "mmd_gamma": scorer.gamma,
            "checkpoint": info["checkpoint"], "weights": info["weights"],
            "score_wall_sec": round(wall, 2),
        },
        "aggregate": agg,
        "rows": rows,
    }
    (out_dir / "style_scores.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
    g = agg["generated"]["cross_overall"]
    print(f"  [{info['tag']}] top1={g['target_top1']:.4f} margin={g['target_margin']:+.4f} "
          f"dT={g['csd_dT']:+.4f} mmd={g['csd_mmd2_unbiased']:.4f} lpips={g['lpips_to_input']:.4f} "
          f"dino_c={g['dino_c']:.4f} | {wall:.1f}s", flush=True)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--gen", required=True, help="directory written by sbflow.generate")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8, help="CPU threads for image decoding")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-noop", action="store_true", help="skip the DCAE no-op baseline")
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    gen_root = Path(args.gen)
    out_root = Path(args.out)
    tags = sorted(p.parent for p in gen_root.glob("*/manifest.json"))
    if not tags:
        raise SystemExit(f"no per-checkpoint manifest.json under {gen_root}")

    t_all = time.perf_counter()
    scorer = ImageScorer(repo, args.device)
    print(f"[score] {len(tags)} checkpoint(s), batch={args.batch} workers={args.workers} "
          f"device={args.device} model_load={scorer.load_sec:.1f}s", flush=True)

    for tag_dir in tags:
        score_checkpoint(scorer, tag_dir, out_root / tag_dir.name, args.batch,
                         args.workers, not args.no_noop)

    total = time.perf_counter() - t_all
    print(f"[score] {len(tags)} checkpoint(s) in {total:.1f}s "
          f"(cached: { {k: round(v, 1) for k, v in scorer.timing.items()} })", flush=True)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps({
        "model_load_sec": round(scorer.load_sec, 2),
        "timing": {k: round(v, 2) for k, v in scorer.timing.items()},
        "total_sec": round(total, 2), "n_checkpoints": len(tags),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
