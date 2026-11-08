"""Pure-CPU metric scorer for five-style generation output.

No CUDA calls anywhere: this process can run while the GPU is busy generating or
training. It is intentionally slow; run it in the background or as a catch-up pass.

Metrics (same definitions as the GPU evaluator, so numbers stay comparable):
  csd_dT / csd_dS        target / source CSD style-probability shift
  target_margin          target CSD score minus the best competing style
  target_top1            argmax CSD style equals the target
  csd_mmd2_unbiased      distribution match against the frozen reference pool
  csd_swd                sliced Wasserstein on the same features
  lpips_to_input         destruction guard only
  dino_c                 DINOv2 CLS cosine with the source (content preservation)

Grouping is ALWAYS by `target_style`, never by source.

Usage
    python -m sbflow.score_cpu --repo-root . \
        --gen runs/blog_sbflow/gen5/<model> --out runs/blog_sbflow/scores5/<model> \
        --batch 8 --threads 8
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
import torch.nn.functional as F
from PIL import Image

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
KEYS = ("csd_dT", "csd_dS", "target_margin", "target_top1", "lpips_to_input", "dino_c")


# --------------------------------------------------------------------------- #
# CPU feature models
# --------------------------------------------------------------------------- #
class CpuFeatures:
    """CSD (ViT-L style head), DINOv2-small CLS, CLIP ViT-B/32 and LPIPS, all on CPU."""

    def __init__(self, repo: Path, threads: int = 8):
        torch.set_num_threads(max(1, threads))
        sys.path.insert(0, str(repo))
        sys.path.insert(0, str(repo / "tools" / "12h"))
        sys.path.insert(0, str(repo / "tools" / "trt"))

        import csd_model
        import paths
        from tools.v5.route_c import csd_prototypes

        t0 = time.perf_counter()
        self.csd, _, _ = csd_model.load_csd("cpu")
        self.csd.eval()
        for p in self.csd.parameters():
            p.requires_grad_(False)
        self.preprocess = csd_model.PREPROCESS

        from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

        hf = str(paths.eval_cache_root("hf") / "hub")
        self.dino_proc = AutoImageProcessor.from_pretrained(
            "facebook/dinov2-small", local_files_only=True, cache_dir=hf)
        self.dino = AutoModel.from_pretrained(
            "facebook/dinov2-small", local_files_only=True, cache_dir=hf).eval()
        self.clip_proc = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32", local_files_only=True, cache_dir=hf)
        self.clip = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32", local_files_only=True, cache_dir=hf).eval()
        for m in (self.dino, self.clip):
            for p in m.parameters():
                p.requires_grad_(False)

        import lpips
        self.lpips = lpips.LPIPS(net="alex", verbose=False).eval()
        for p in self.lpips.parameters():
            p.requires_grad_(False)

        self.proto = np.asarray(csd_prototypes(), dtype=np.float32)
        self.load_sec = time.perf_counter() - t0
        self.timing: Dict[str, float] = {}

    @torch.inference_mode()
    def _csd(self, pils: Sequence[Image.Image]) -> np.ndarray:
        x = torch.stack([self.preprocess(im) for im in pils]).float()
        _, _, sty = self.csd(x)
        return F.normalize(sty.float(), dim=-1).numpy()

    @torch.inference_mode()
    def _dino(self, pils: Sequence[Image.Image]) -> np.ndarray:
        inp = self.dino_proc(images=list(pils), return_tensors="pt")
        out = self.dino(**inp).last_hidden_state[:, 0]
        return F.normalize(out.float(), dim=-1).numpy()

    @torch.inference_mode()
    def _lpips(self, a: torch.Tensor, b: torch.Tensor) -> float:
        return float(self.lpips(a, b).mean().item())

    def features(self, pils: Sequence[Image.Image]) -> dict:
        t0 = time.perf_counter()
        csd = self._csd(pils)
        self.timing["csd"] = self.timing.get("csd", 0.0) + (time.perf_counter() - t0)
        t0 = time.perf_counter()
        dino = self._dino(pils)
        self.timing["dino"] = self.timing.get("dino", 0.0) + (time.perf_counter() - t0)
        return {"csd": csd, "dino": dino}


# --------------------------------------------------------------------------- #
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
    proj = F.normalize(torch.randn(g.shape[1], n_proj, generator=torch.Generator().manual_seed(seed)),
                       dim=0)
    pg = torch.sort(g @ proj, dim=0).values.numpy()
    pr = torch.sort(r @ proj, dim=0).values.numpy()
    q = np.linspace(0.0, 1.0, 256)
    xg = np.linspace(0.0, 1.0, pg.shape[0])
    xr = np.linspace(0.0, 1.0, pr.shape[0])
    return float(np.mean([np.abs(np.interp(q, xg, pg[:, p]) - np.interp(q, xr, pr[:, p])).mean()
                          for p in range(n_proj)]))


def load_u8(path: Path, size: Optional[int] = None) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        if size and im.size != (size, size):
            im = im.resize((size, size), Image.Resampling.BICUBIC)
        return np.asarray(im, dtype=np.uint8).copy()


def u8_tensor(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(a.astype(np.float32)).permute(2, 0, 1) / 127.5 - 1


# --------------------------------------------------------------------------- #
def aggregate(rows: List[dict], refs: dict, gamma: float) -> dict:
    """Per directed pair, per target style, and overall. Grouping by target_style."""
    out: Dict[str, dict] = {}

    def block(sub: List[dict]) -> dict:
        res = {k: float(np.mean([r[k] for r in sub])) for k in KEYS}
        tgt = sub[0]["target"]
        feat = np.asarray([r["csd_embedding"] for r in sub], dtype=np.float32)
        real = refs["styles"][tgt]["csd"]
        res["csd_mmd2_unbiased"] = mmd2(feat, real, gamma)
        res["csd_swd"] = swd(feat, real)
        res["n"] = len(sub)
        return res

    gen = [r for r in rows if r["variant"] == "generated"]
    out["overall"] = block(gen)
    out["by_direction"] = {}
    for d in sorted({r["direction"] for r in gen}):
        out["by_direction"][d] = block([r for r in gen if r["direction"] == d])
    out["by_target"] = {}
    for s in sorted({r["target"] for r in gen}):
        out["by_target"][s] = block([r for r in gen if r["target"] == s])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--gen", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--io-workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    gen_root = Path(args.gen)
    out_root = Path(args.out)
    man = json.loads((gen_root / "manifest.json").read_text(encoding="utf-8"))
    info, tasks = man["info"], man["tasks"]
    if args.limit:
        tasks = tasks[: args.limit]

    sys.path.insert(0, str(repo))
    from tools.evaluation_v2.features import build_reference_features_cache

    feat = CpuFeatures(repo, args.threads)
    # Reference features only need CSD; reuse the CPU extractor rather than loading GPU models.
    import paths
    cache = repo / "eval_protocol" / "cache" / "references_v1_csd_cpu.pt"
    refs = _reference_csd(repo, feat, cache)
    gamma = float(refs["rbf_gamma"])
    print(f"[score-cpu] models loaded in {feat.load_sec:.1f}s, tasks={len(tasks)}, "
          f"threads={args.threads}", flush=True)

    t0 = time.perf_counter()
    pool = ThreadPoolExecutor(max_workers=args.io_workers)
    gen_imgs = list(pool.map(lambda t: load_u8(gen_root / t["image"]), tasks))
    size = gen_imgs[0].shape[0]
    src_imgs = list(pool.map(lambda t: load_u8(repo / t["source_rel_path"], size), tasks))
    pool.shutdown(wait=True)
    feat.timing["io"] = time.perf_counter() - t0
    print(f"[score-cpu] loaded {2 * len(tasks)} images in {feat.timing['io']:.1f}s", flush=True)

    rows: List[dict] = []
    for s in range(0, len(tasks), args.batch):
        g = tasks[s:s + args.batch]
        gp, sp = gen_imgs[s:s + args.batch], src_imgs[s:s + args.batch]
        fg = feat.features([Image.fromarray(a) for a in gp])
        fs = feat.features([Image.fromarray(a) for a in sp])
        vg = fg["csd"] @ feat.proto.T
        vs = fs["csd"] @ feat.proto.T
        dc = F.cosine_similarity(torch.as_tensor(fg["dino"]), torch.as_tensor(fs["dino"]), dim=1).numpy()
        for j, t in enumerate(g):
            tid, sid = STYLES.index(t["target_style"]), STYLES.index(t["source_style"])
            other = np.delete(vg[j], tid)
            rows.append({
                "task_id": t["task_id"],
                "direction": f"{t['source_style']}->{t['target_style']}",
                "source": t["source_style"], "target": t["target_style"],
                "variant": "generated",
                "csd_dT": float(vg[j, tid] - vs[j, tid]),
                "csd_dS": float(vg[j, sid] - vs[j, sid]),
                "target_margin": float(vg[j, tid] - other.max()),
                "target_top1": int(vg[j].argmax() == tid),
                "lpips_to_input": feat._lpips(u8_tensor(gp[j])[None], u8_tensor(sp[j])[None]),
                "dino_c": float(dc[j]),
                "csd_embedding": fg["csd"][j].tolist(),
            })
        if s % (args.batch * 4) == 0:
            print(f"[score-cpu] {s + len(g)}/{len(tasks)}", flush=True)

    agg = aggregate(rows, refs, gamma)
    wall = time.perf_counter() - t0
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "style_scores.json").write_text(json.dumps({
        "protocol": {"source": "pure CPU scorer", "subset": info.get("subset"),
                     "n_tasks": len(tasks), "network_nfe": info.get("network_nfe"),
                     "mmd_gamma": gamma, "checkpoint": info.get("checkpoint"),
                     "score_wall_sec": round(wall, 1),
                     "timing": {k: round(v, 1) for k, v in feat.timing.items()}},
        "aggregate": agg, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    o = agg["overall"]
    print(f"[score-cpu] overall top1={o['target_top1']:.4f} margin={o['target_margin']:+.4f} "
          f"dT={o['csd_dT']:+.4f} dS={o['csd_dS']:+.4f} mmd={o['csd_mmd2_unbiased']:.4f} "
          f"swd={o['csd_swd']:.4f} dino_c={o['dino_c']:.4f} | {wall:.1f}s", flush=True)


def _reference_csd(repo: Path, feat: CpuFeatures, cache: Path) -> dict:
    """Build (and cache) the 30-per-style reference CSD features on CPU."""
    if cache.exists():
        return torch.load(cache, map_location="cpu", weights_only=False)
    raw = json.loads((repo / "eval_protocol" / "references_v1.json").read_text(encoding="utf-8"))
    entries = raw["references"]
    styles: Dict[str, dict] = {}
    for s in STYLES:
        items = entries[s]
        rels = [it["rel_path"] if isinstance(it, dict) else it for it in items]
        pils = [Image.open(repo / p).convert("RGB") for p in rels]
        styles[s] = {"csd": torch.from_numpy(feat._csd(pils)).float()}
        for im in pils:
            im.close()
    refs = {"styles": styles,
            "rbf_gamma": float(raw.get("rbf_gamma", 1.0 / 768.0))}
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(refs, cache)
    return refs


if __name__ == "__main__":
    main()
