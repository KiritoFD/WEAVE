"""P1 (evaluation only): style-discrimination AUROC on real artworks + ΔM(VAE)=0 check.

No training, no config change, no reference re-selection against test scores.
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "trt"):
    sys.path.insert(0, str(_p))
import eval_fast as ef
import trt_vae_decoder
from utils.inference import encode_image, load_vae

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default=str(ROOT / "data" / "wikiart_new_packed" / "manifest.json"))
ap.add_argument("--test-dir", default=str(ROOT / "data" / "test"))
ap.add_argument("--out", default=str(ROOT / "runs" / "p1" / "auroc_dm.json"))
ap.add_argument("--n", default="300,300,74,300,300")
ap.add_argument("--image-size", type=int, default=512)
ap.add_argument("--cache-dir", default="/mnt/i/Latent_Style/eval_cache")
ap.add_argument("--engine-dir", default=str(ROOT / "runs" / "trt" / "engines"))
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)

# ---------------- features ----------------
ns = argparse.Namespace(
    disable_clip=False, disable_lpips=True, disable_dino=False, test_dir=str(a.test_dir),
    clip_model="openai/clip-vit-base-patch32", clip_local_only=True,
    clip_cache_dir=str(Path(a.cache_dir) / "hf" / "hub"), clip_skip_processor=False, amp=False,
    lpips_net="alex", lpips_chunk=32, dino_model="facebook/dinov2-small",
    dino_cache_dir=str(Path(a.cache_dir) / "hf"), allow_network=False,
    ref_cache="", max_ref_cache=16, dino_max_refs=30, image_size=a.image_size,
    metric_batch=32, ref_batch=32, only_clip_style=True,
    clip_style_idt_baseline=0.639920825263, clip_vectors=False,
)
timer = ef.Timer(sync_cuda=False)
test_images, _req, style_meta = ef.build_protocol(Path(a.test_dir), max_src_samples=30, ref_pool="first")
engine = ef.MetricEngine(ns, test_images, style_meta, "cuda")
engine.load(timer)


def dino_feats(paths):
    cls_all, stat_all = [], []
    for s in range(0, len(paths), 32):
        ims = [Image.open(p).convert("RGB") for p in paths[s:s + 32]]
        with torch.inference_mode():
            cls, patches = engine._dino_forward(ims)
        patches = patches.float()
        stat_all.append(torch.cat([patches.mean(1), patches.std(1)], dim=-1))
        cls_all.append(cls.float())
    return (F.normalize(torch.cat(cls_all, 0), dim=-1).cpu().numpy(),
            F.normalize(torch.cat(stat_all, 0), dim=-1).cpu().numpy())


def clip_feats(paths):
    out = []
    for s in range(0, len(paths), 32):
        batch = torch.stack([ef._load_eval_image_tensor(p, size=a.image_size) for p in paths[s:s + 32]]).cuda()
        with torch.inference_mode():
            out.append(engine._clip_images(batch).cpu())
    return F.normalize(torch.cat(out, 0), dim=-1).numpy()


manifest = json.loads(Path(a.manifest).read_text())
image_root = Path(manifest["image_root"])
by_style = defaultdict(list)
for x in manifest["samples"]:
    if x.get("split") == "train":
        by_style[x["style"]].append(x)
for k in by_style:
    by_style[k].sort(key=lambda r: int(r["index"]))

test_stems = {p.stem for sid in style_meta for p in style_meta[sid]["all"]}
sids = sorted(style_meta)
ns_per = [int(v) for v in a.n.split(",")]
names = [style_meta[s]["name"] for s in sids]
print("styles:", names, flush=True)

pool = {}
for name, n in zip(names, ns_per):
    rows = [r for r in by_style[name] if Path(r["path"]).stem not in test_stems][:n]
    pool[name] = rows
    print(f"{name}: usable {len(rows)} / {len(by_style[name])}", flush=True)

paths_by = {k: [image_root / r["path"] for r in v] for k, v in pool.items()}
feat = {}
for k, ps in paths_by.items():
    cls, stat = dino_feats(ps)
    feat[k] = {"dino": cls, "stat": stat, "clip": clip_feats(ps)}
    print("feat done", k, cls.shape, flush=True)


def auroc(pos, neg):
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), dtype=float)
    ranks[order] = np.arange(1, len(allv) + 1, dtype=float)
    npos, nneg = len(pos), len(neg)
    return float((ranks[:npos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def pair_sims(fa, fb, idx_a, idx_b, same):
    if same:
        i, j = np.triu_indices(len(idx_a), 1)
        return np.einsum("ij,ij->i", fa[idx_a][i], fa[idx_a][j])
    return np.einsum("ij,ij->i", fa[idx_a], fb[idx_b])


res = {"per_style": {}, "n_per_style": {}}
for name in names:
    n = len(paths_by[name])
    others = [k for k in names if k != name]
    res["n_per_style"][name] = n
    f = feat[name]
    cell = {}
    for key in ("dino", "stat"):
        fa = f[key]
        # positives: all within-style pairs
        pi, pj = np.triu_indices(n, 1)
        pos = np.einsum("ij,ij->i", fa[pi], fa[pj])
        # negatives: cross-style pairs, sampled to the same count
        ni, nj = [], []
        while len(ni) < len(pos):
            o = others[int(rng.integers(len(others)))]
            a_i = int(rng.integers(n))
            b_j = int(rng.integers(len(paths_by[o])))
            ni.append(a_i)
            nj.append((others.index(o), b_j))
        neg = np.array([float(fa[i] @ feat[others[o]][key][j]) for i, (o, j) in zip(ni, nj)])
        # content proxy = CLIP image feature
        cpos = np.einsum("ij,ij->i", f["clip"][pi], f["clip"][pj])
        cneg = np.array([float(f["clip"][i] @ feat[others[o]]["clip"][j]) for i, (o, j) in zip(ni, nj)])
        hard = cneg >= np.quantile(cneg, 0.75)
        k = int(hard.sum())
        hard_auroc = None
        if k > 10:
            sel = rng.choice(len(pos), size=k, replace=False)
            hard_auroc = auroc(pos[sel], neg[hard])
        cell[key] = {
            "auroc": auroc(pos, neg),
            "auroc_content_hard_top25": hard_auroc,
            "n_pos": int(len(pos)), "n_neg": int(len(neg)),
            "content_pos_mean": float(cpos.mean()), "content_neg_mean": float(cneg.mean()),
            "content_pos_q25": float(np.quantile(cpos, 0.25)),
            "content_hard_thr": float(np.quantile(cneg, 0.75)),
        }
    res["per_style"][name] = cell
    print(name, {k: round(v["auroc"], 4) for k, v in cell.items()},
          "hard:", {k: (round(v["auroc_content_hard_top25"], 4) if v["auroc_content_hard_top25"] else None) for k, v in cell.items()},
          flush=True)

# ---------------- ΔM on VAE round-trip ----------------
scale = 0.18215
vae = load_vae("cuda", model_id="ema", cache_dir=str(Path(a.cache_dir) / "hf"), compile_decoder=False).to(torch.float32)
trt = trt_vae_decoder.TrtVaeDecoder(a.engine_dir, vae, precision="fp16", device="cuda")
vae_scale = float(getattr(vae.config, "scaling_factor", scale))

# prototypes: equal-size (min class), train pool, test-overlap removed (already), independent of test scores
proto = {}
for name in names:
    n = len(paths_by[name])
    proto[name] = {
        "dino": F.normalize(torch.from_numpy(feat[name]["dino"][:n].mean(0, keepdims=True)), dim=-1)[0].numpy(),
        "stat": F.normalize(torch.from_numpy(feat[name]["stat"][:n].mean(0, keepdims=True)), dim=-1)[0].numpy(),
    }

src_paths = sorted({r["src_path"] for r in _req})
x0 = torch.stack([ef._load_eval_image_tensor(p, size=a.image_size) for p in src_paths]).cuda()
with torch.inference_mode():
    lats = []
    for s in range(0, x0.shape[0], 8):
        lats.append(encode_image(vae, x0[s:s + 8], "cuda"))
    lat = torch.cat(lats, 0) * (scale / vae_scale)
    dec = []
    for s in range(0, lat.shape[0], 8):
        ch = lat[s:s + 8]
        n = ch.shape[0]
        if n < 8:
            ch = torch.cat([ch, ch[-1:].expand(8 - n, -1, -1, -1)], 0)
        dec.append(trt.decode(ch)[:n])
gen = torch.cat(dec, 0)
u8 = gen.clamp(0, 1).mul(255).round().to(torch.uint8)
png_dir = Path(a.out).parent / "vae_roundtrip"
png_dir.mkdir(parents=True, exist_ok=True)
png_paths = []
for i in range(u8.shape[0]):
    p = png_dir / f"{Path(src_paths[i]).stem}.png"
    Image.fromarray(u8[i].permute(1, 2, 0).cpu().numpy()).save(p)
    png_paths.append(p)

f_in = {}
# in-memory (decoded tensor) vs on-disk PNG (quantised) — two independent f() paths
cls_t, stat_t = [], []
for s in range(0, u8.shape[0], 32):
    with torch.inference_mode():
        c, p_ = engine._dino_forward([Image.fromarray(u8[i].permute(1, 2, 0).cpu().numpy()) for i in range(s, min(s + 32, u8.shape[0]))])
    p_ = p_.float()
    stat_t.append(torch.cat([p_.mean(1), p_.std(1)], dim=-1))
    cls_t.append(c.float())
f_in["dino"] = F.normalize(torch.cat(cls_t, 0), dim=-1).cpu().numpy()
f_in["stat"] = F.normalize(torch.cat(stat_t, 0), dim=-1).cpu().numpy()
cls_d, stat_d = dino_feats(png_paths)
f_png = {"dino": cls_d, "stat": stat_d}

src_styles = [style_meta[s]["name"] for s in sorted(style_meta)]
style_of_src = {}
for r in _req:
    style_of_src[r["src_path"]] = r["src_style"]

dm = {}
for key in ("dino", "stat"):
    vals = []
    per_style = defaultdict(list)
    for i, p in enumerate(src_paths):
        s = style_of_src[p]
        for t in names:
            if t == s:
                continue
            def M(f):
                return float(f[i] @ proto[t][key]) - float(f[i] @ proto[s][key])
            d = M(f_in[key]) - M(f_png[key])
            vals.append(abs(d))
            per_style[s].append(abs(d))
    dm[key] = {"max_abs": float(np.max(vals)), "mean_abs": float(np.mean(vals)),
               "per_src_style_mean": {k: float(np.mean(v)) for k, v in per_style.items()}}
    print("ΔM(V) ", key, dm[key]["max_abs"], dm[key]["mean_abs"], flush=True)

# ΔM(V) is identically 0 when both sides use the same f(); the only real source of
# non-zero is the quantisation gap between the in-memory tensor and the on-disk PNG.

res["delta_m_vae"] = dm
res["protocol"] = {"features": ["dinov2-small CLS", "dinov2-small patch mean/std (CSD substitute)"],
                   "content_proxy": "CLIP ViT-B/32 image embedding", "test_overlap_removed": True,
                   "source_images": len(src_paths), "n_per_style": res["n_per_style"]}
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps(res, indent=1))
print(json.dumps({k: res[k] for k in ("delta_m_vae",)}, indent=1))

