"""P1c (evaluation only): style-discrimination AUROC with the official CSD descriptor.

Main line  : CSD (ViT-L, 768-d style head)  -- per style, plus macro/micro
Control    : DINOv2-small CLS (384-d)  and  DINOv2-small patch mean/std (auxiliary)
Variants   : all_pairs (any two real works)  /  artist_disjoint (same painter removed)
Confound   : content-hard top-25% negatives selected with CLIP ViT-B/32 image features
Also       : ΔM(VAE round-trip) == 0 verification with CSD / DINO / patch-stat features

No training, no config edits, no reference re-selection against scores.
"""
import argparse
import json
import sys
import time
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
from tools.legacy_utils.inference import encode_image, load_vae
import csd_model

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default=str(ROOT / "data" / "wikiart_new_packed" / "manifest.json"))
ap.add_argument("--test-dir", default=str(ROOT / "data" / "test"))
ap.add_argument("--out", default=str(ROOT / "runs" / "p1" / "auroc_csd_dm.json"))
ap.add_argument("--n", default="300,300,74,300,300")
ap.add_argument("--image-size", type=int, default=512)
ap.add_argument("--cache-dir", default="/mnt/i/Latent_Style/eval_cache")
ap.add_argument("--engine-dir", default=str(ROOT / "runs" / "trt" / "engines"))
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)

# ---------------------------------------------------------------- CSD
t_csd = time.time()
csd, ck, csd_msg = csd_model.load_csd("cuda")
print("CSD loaded in", round(time.time() - t_csd, 1), "s:", csd_msg, flush=True)

# single-image forward timing (warmup 5 -> 20 reps)
_probe = Image.open(sorted(Path(a.test_dir).rglob("*.jpg"))[0]).convert("RGB")
_x = torch.stack([csd_model.PREPROCESS(_probe)]).cuda().half()
with torch.inference_mode():
    for _ in range(5):
        csd(_x)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20):
        csd(_x)
    torch.cuda.synchronize()
csd_ms = round((time.time() - t0) / 20 * 1000, 2)
print("CSD single-image forward ms:", csd_ms, flush=True)

# ---------------------------------------------------------------- protocol / features
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


def dino_feats(paths_):
    cls_all, stat_all = [], []
    for s in range(0, len(paths_), 32):
        ims = [Image.open(p).convert("RGB") for p in paths_[s:s + 32]]
        with torch.inference_mode():
            cls, patches = engine._dino_forward(ims)
        patches = patches.float()
        stat_all.append(torch.cat([patches.mean(1), patches.std(1)], dim=-1))
        cls_all.append(cls.float())
    return (F.normalize(torch.cat(cls_all, 0), dim=-1).cpu().numpy(),
            F.normalize(torch.cat(stat_all, 0), dim=-1).cpu().numpy())


def clip_feats(paths_):
    out = []
    for s in range(0, len(paths_), 32):
        batch = torch.stack([ef._load_eval_image_tensor(p, size=a.image_size) for p in paths_[s:s + 32]]).cuda()
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

pool, artists = {}, {}
for name, n in zip(names, ns_per):
    rows_all = [r for r in by_style[name] if Path(r["path"]).stem not in test_stems]
    if len(rows_all) > n:
        sel = np.sort(rng.choice(len(rows_all), size=n, replace=False))
        rows = [rows_all[int(i)] for i in sel]
    else:
        rows = rows_all
    pool[name] = rows
    artists[name] = [str(r.get("artist", "")) for r in rows]
    print(f"{name}: usable {len(rows)} / {len(by_style[name])}", flush=True)

paths_by = {k: [image_root / r["path"] for r in v] for k, v in pool.items()}
feat = {}
for k, ps in paths_by.items():
    cls, stat = dino_feats(ps)
    csd_f = csd_model.feats_paths(csd, ps)
    feat[k] = {"csd": csd_f, "dino": cls, "stat": stat, "clip": clip_feats(ps)}
    print("feat done", k, "csd", csd_f.shape, flush=True)


def auroc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return None
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), dtype=float)
    ranks[order] = np.arange(1, len(allv) + 1, dtype=float)
    npos, nneg = len(pos), len(neg)
    return float((ranks[:npos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


FEATS = ("csd", "dino", "stat")


def r4(v):
    return None if v is None else round(v, 4)


def build_pairs(name, artist_disjoint):
    n = len(paths_by[name])
    others = [k for k in names if k != name]
    arts = artists[name]
    oarts = {k: artists[k] for k in others}

    pi, pj = np.triu_indices(n, 1)
    if artist_disjoint:
        keep = np.array([arts[i] != arts[j] for i, j in zip(pi, pj)], dtype=bool)
        dropped = int((~keep).sum())
        pi, pj = pi[keep], pj[keep]
    else:
        dropped = 0
    n_pos = len(pi)

    ni, nj, no = [], [], []
    guard = 0
    while len(ni) < n_pos and guard < 50 * max(n_pos, 1):
        guard += 1
        o = others[int(rng.integers(len(others)))]
        i = int(rng.integers(n))
        j = int(rng.integers(len(paths_by[o])))
        if artist_disjoint and arts[i] == oarts[o][j]:
            continue
        ni.append(i)
        nj.append(j)
        no.append(o)
    return dict(pi=pi, pj=pj, ni=np.array(ni), nj=np.array(nj),
                others=no, n_pos=n_pos, n_dropped=dropped, others_list=others)


def scores(name, p, key):
    f = feat[name][key]
    oth = p["others"]
    pos = np.einsum("ij,ij->i", f[p["pi"]], f[p["pj"]])
    neg = np.array([float(f[i] @ feat[o][key][j]) for i, o, j in zip(p["ni"], oth, p["nj"])])
    return pos, neg


res = {"per_style": {}, "n_per_style": {}, "pool_construction": {}}
pooled = {v: {k: {"pos": [], "neg": []} for k in FEATS} for v in ("all", "artist")}

for name in names:
    res["n_per_style"][name] = len(paths_by[name])
    arts_ct = {x: artists[name].count(x) for x in set(artists[name])}
    res["pool_construction"][name] = {
        "n_images": len(paths_by[name]),
        "n_unique_artists": len(arts_ct),
        "largest_artist": max(arts_ct, key=arts_ct.get),
        "largest_artist_share": round(max(arts_ct.values()) / len(paths_by[name]), 4),
        "source": "wikiart_new_packed train split, test-directory overlap removed, "
                  "seeded random sample (numpy seed 42)",
    }
    cell = {}
    for variant, ad in (("all", False), ("artist", True)):
        p = build_pairs(name, ad)
        vcell = {}
        for key in FEATS:
            pos, neg = scores(name, p, key)
            cpos, cneg = scores(name, p, "clip")
            hard = cneg >= np.quantile(cneg, 0.75)
            k = int(hard.sum())
            hard_auroc = None
            if k > 10:
                sel = rng.choice(len(pos), size=k, replace=False)
                hard_auroc = auroc(pos[sel], neg[hard])
            vcell[key] = {
                "auroc": auroc(pos, neg),
                "auroc_content_hard_top25": hard_auroc,
                "n_pos": int(len(pos)), "n_neg": int(len(neg)),
                "n_pos_dropped_same_artist": int(p["n_dropped"]),
                "content_pos_mean": float(cpos.mean()), "content_neg_mean": float(cneg.mean()),
            }
            pooled[variant][key]["pos"].append(pos)
            pooled[variant][key]["neg"].append(neg)
        cell[variant] = vcell
    res["per_style"][name] = cell
    print(name, "n=", len(paths_by[name]),
          "ALL", {k: r4(v["auroc"]) for k, v in cell["all"].items()},
          "ARTIST", {k: r4(v["auroc"]) for k, v in cell["artist"].items()},
          flush=True)

summary = {}
for variant in ("all", "artist"):
    per = res["per_style"]
    summary[variant] = {
        "macro": {k: float(np.nanmean([per[n][variant][k]["auroc"] if per[n][variant][k]["auroc"] is not None else np.nan for n in names])) for k in FEATS},
        "macro_content_hard": {k: float(np.nanmean([per[n][variant][k]["auroc_content_hard_top25"] if per[n][variant][k]["auroc_content_hard_top25"] is not None else np.nan for n in names])) for k in FEATS},
        "micro": {k: auroc(np.concatenate(pooled[variant][k]["pos"]) if pooled[variant][k]["pos"] else np.array([]),
                           np.concatenate(pooled[variant][k]["neg"]) if pooled[variant][k]["neg"] else np.array([])) for k in FEATS},
    }
res["summary"] = summary
print(json.dumps(summary, indent=1), flush=True)

# ---------------------------------------------------------------- ΔM(V) round-trip
scale = 0.18215
vae = load_vae("cuda", model_id="ema", cache_dir=str(Path(a.cache_dir) / "hf"), compile_decoder=False).to(torch.float32)
trt = trt_vae_decoder.TrtVaeDecoder(a.engine_dir, vae, precision="fp16", device="cuda")
vae_scale = float(getattr(vae.config, "scaling_factor", scale))

K_PROTO = min(res["n_per_style"].values())
proto = {name: {k: F.normalize(torch.from_numpy(feat[name][k][:K_PROTO].mean(0, keepdims=True)), dim=-1)[0].numpy()
                for k in FEATS} for name in names}

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

pils_mem = [Image.fromarray(u8[i].permute(1, 2, 0).cpu().numpy()) for i in range(u8.shape[0])]
f_mem = {"csd": csd_model.feats_pils(csd, pils_mem)}
cls_t, stat_t = [], []
for s in range(0, len(pils_mem), 32):
    with torch.inference_mode():
        c, p_ = engine._dino_forward(pils_mem[s:s + 32])
    p_ = p_.float()
    stat_t.append(torch.cat([p_.mean(1), p_.std(1)], dim=-1))
    cls_t.append(c.float())
f_mem["dino"] = F.normalize(torch.cat(cls_t, 0), dim=-1).cpu().numpy()
f_mem["stat"] = F.normalize(torch.cat(stat_t, 0), dim=-1).cpu().numpy()

f_png = {"csd": csd_model.feats_paths(csd, png_paths)}
f_png["dino"], f_png["stat"] = dino_feats(png_paths)

style_of_src = {r["src_path"]: r["src_style"] for r in _req}
dm = {}
for key in FEATS:
    vals, per_style = [], defaultdict(list)
    for i, p in enumerate(src_paths):
        s = style_of_src[p]
        for t in names:
            if t == s:
                continue
            marg = lambda f: float(f[i] @ proto[t][key]) - float(f[i] @ proto[s][key])
            d = abs(marg(f_mem[key]) - marg(f_png[key]))
            vals.append(d)
            per_style[s].append(d)
    dm[key] = {"max_abs": float(np.max(vals)), "mean_abs": float(np.mean(vals)),
               "n_compared": len(vals),
               "per_src_style_mean": {k: float(np.mean(v)) for k, v in per_style.items()}}
    print("ΔM(V)", key, dm[key]["max_abs"], dm[key]["mean_abs"], flush=True)

res["delta_m_vae"] = dm
res["delta_m_setup"] = {
    "tolerance": 0.0,
    "why_zero": "both sides decode the same u8 buffer; the only difference is in-memory tensor vs PNG "
                "file bytes, which are byte-identical after round(u8) quantisation, so any non-zero "
                "value would indicate a preprocessing inconsistency between the two f() paths",
    "prototype_images_per_style": K_PROTO,
    "prototype_construction": f"first {K_PROTO} images of the frozen AUROC pool per style "
                              "(seeded random sample over the train split, test overlap removed; "
                              "no use of any eval score)",
    "reference_pool": "same frozen 5-class train pool used for the AUROC (300/300/74/300/300, "
                      "equal-size truncated to 74 for the prototypes)",
    "margin_definition": "M = cos(f, proto_target) - cos(f, proto_source_style)",
    "n_source_images": len(src_paths),
    "n_target_styles_per_source": len(names) - 1,
}
res["provenance"] = csd_model.provenance(csd, ck, csd_ms)
res["provenance"]["dino_comparison"] = {
    "model": "facebook/dinov2-small", "feature": "CLS, 384-d, L2-normalised",
    "input": "engine preprocessing at 512px (WEAVE eval pipeline)",
}
res["protocol"] = {
    "positives": "real artwork pairs inside one style class",
    "negatives": "real artwork pairs across style classes, sampled to match the positive count",
    "artist_disjoint": "positives and negatives restricted to different painter names",
    "content_hard": "negatives whose CLIP ViT-B/32 image cosine sits in the top 25% (content-confusion control)",
    "patched_stat_descriptor": "DINOv2 patch mean/std - auxiliary only, not a CSD substitute",
    "test_overlap_removed": True,
}

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps(res, indent=1))
print("wrote", a.out, flush=True)

