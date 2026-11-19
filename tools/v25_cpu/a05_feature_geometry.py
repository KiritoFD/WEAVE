"""A05 (Slot A, V25): 128D/795D style-feature effective-dimension diagnosis.

Task: docs/19_pull_together/agent2.md section 5 A05. Answers, with PCA /
correlation / distance geometry on CPU (numpy/scipy/sklearn):

1. 795D Pi_parallel fiber (Oklab27 + DINO768 moments, exp/v19/fiber_duality.py):
   PCA spectrum, effective rank (participation ratio / entropy dim), per-style
   Fisher ratio & silhouette, 5-class centroid cosine matrix, PCA-k -> 5-way
   accuracy curve ("how many dims carry the 5-style discrimination").
2. Block structure: Oklab27 vs DINO768 (per-block effective dims, block-wise
   separability, CCA + ridge R^2 cross-prediction = internal redundancy).
3. One-Shot reference-side geometry: SG-5 refs + references_v1 (150 wikiart
   refs) against train-proxy style centroids (label-free conditioning check).
4. 128D u-space (BLOCKED locally: fiber_state/u caches live on the 4090) --
   consumed from an exported npz when available (--u-npz); export command is
   printed and the --export-u mode implements it (CUDA-gated, remote only).
5. RGB ConvNeXt v16 pooled features (BLOCKED locally: weights absent) --
   consumed from an exported npz (--cnx-npz); --export-cnx implements it.

Populations (all local, see docs/v25/audits/a05_feature_geometry.md):
  P1_eval150 : the 150 unique test750 source images; primary variant decodes
               the cached z_src latents through S10 (production eval path),
               secondary variant reads the original JPEGs (proxy gap measured
               pairwise on the same 150).
  P2_train600: seeded 120/style sample of datasets/distinct5_512/train
               (StyleGallery train pool = the population v18 draws train_4400
               from), minus test750-source/reference stems (banned set
               replicated from exp.v18.clean_balanced_pool).
  P3_oneshot : the exact SG_5_STEMS (exp/v21/stiefel_ref_cache.py) + 150
               references_v1 wikiart reference images.

Protocol fidelity notes are recorded in the output JSON meta.protocol_notes.
No GPU/CUDA is used in analysis mode; OMP/MKL threads are pinned to 2.

Run:  python -m tools.v25_cpu.a05_feature_geometry [--u-npz P] [--cnx-npz P]
Exports (remote GPU queues only, refused on CPU-only machines):
  ssh 4090 "cd /root/Workspace/WEAVE && OMP_NUM_THREADS=2 python -m tools.v25_cpu.a05_feature_geometry --export-u > runs/v25/cache/a05_export_u.log"
  ssh 3060 "cd <3060 WEAVE> && OMP_NUM_THREADS=2 python -m tools.v25_cpu.a05_feature_geometry --export-cnx > runs/v25/cache/a05_export_cnx.log"
"""
from __future__ import annotations

import os

# Pin CPU thread budgets BEFORE numpy/torch import (agent2.md section 11).
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import argparse
import json
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEED = 20261120
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
DINO_NAME = "facebook/dinov2-small"
N_PER_STYLE = 120          # P2 sample size per style
OUT_JSON = ROOT / "docs/v25/audits/a05_feature_geometry.json"

# Block layout of the 795D fiber: [Oklab27 | DINO patch-token mean 384 | std 384]
OKLAB_DIM, DINO_MEAN_SL, DINO_STD_SL = 27, slice(27, 27 + 384), slice(27 + 384, 27 + 768)


# --------------------------------------------------------------------------
# shared small utils
# --------------------------------------------------------------------------
def canonical_pid(name: str) -> str:
    """Replicate exp.v18.canonical_pid (kept local to avoid heavy import chains)."""
    return unicodedata.normalize("NFC", Path(str(name).replace("\\", "/")).stem.split("__")[-1]).casefold()


def save_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def resize512(im) -> Image.Image:
    return im.convert("RGB").resize((512, 512), Image.BILINEAR)


def pil_to_rgb01(im):
    import torch

    arr = np.asarray(resize512(im), dtype=np.uint8)
    return torch.from_numpy(arr.copy()).permute(2, 0, 1).float() / 255.0


def rgb01_to_pil(rgb01):
    """uint8-quantize a (3,H,W)/(B,3,H,W) rgb01 tensor exactly like production."""
    import torch

    t = rgb01.detach().cpu().float() * 255.0
    t = t.round().clamp(0, 255).byte()
    arr = t.permute(1, 2, 0).numpy() if t.dim() == 3 else t.permute(0, 2, 3, 1).numpy()
    return [Image.fromarray(a) for a in arr]


# --------------------------------------------------------------------------
# feature extraction (exact v19 protocol, CPU)
# --------------------------------------------------------------------------
def load_dino_cpu():
    """Load facebook/dinov2-small from the local HF snapshot (paths.eval_cache_root)."""
    import paths
    from transformers import AutoModel

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    repo = Path(paths.eval_cache_root("hf")) / "hub" / "models--facebook--dinov2-small" / "snapshots"
    revs = [p for p in repo.iterdir() if p.is_dir()]
    if not revs:
        raise FileNotFoundError(f"no local dinov2-small snapshot under {repo}")
    model = AutoModel.from_pretrained(str(revs[0])).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.float()


def dino_forward_tokens(model, pil_imgs, batch: int = 8):
    """Replicate tools.legacy_utils DINO_TRANSFORM + encode_dino token protocol (CPU fp32).

    Returns (patch_list, cls_list): patches L2-normalized per token, exactly as
    exp.v19.fiber_duality.extract_dino_rgb_moments_768d consumes them.
    """
    import torch
    import torch.nn.functional as F
    from tools.legacy_utils.compute_dino_metrics import DINO_TRANSFORM

    patches, clss = [], []
    with torch.inference_mode():
        for i in range(0, len(pil_imgs), batch):
            x = torch.stack([DINO_TRANSFORM(im.convert("RGB")) for im in pil_imgs[i : i + batch]])
            hidden = model(x).last_hidden_state
            n = hidden.shape[1]
            if n == 261:      # register variant: 1 CLS + 4 register + 256 patch
                cls, pat = hidden[:, 0], hidden[:, 5:]
            elif n == 257:
                cls, pat = hidden[:, 0], hidden[:, 1:]
            else:
                raise RuntimeError(f"unexpected DINO token count {n}")
            patches.append(F.normalize(pat.float(), dim=-1))
            clss.append(F.normalize(cls.float(), dim=-1))
    return torch.cat(patches, 0), torch.cat(clss, 0)


def fiber795_from_rgb01(model, rgb01) -> np.ndarray:
    """rgb01 (B,3,512,512) -> (B,795) = [extract_color_fiber_27d | dino768].

    extract_color_fiber_27d is imported from exp.v19.fiber_duality verbatim so
    the Oklab27 block is bit-identical to production; the DINO768 block
    replicates extract_dino_rgb_moments_768d with a CPU fp32 forward.
    """
    import torch
    from exp.v19.fiber_duality import extract_color_fiber_27d

    col27 = extract_color_fiber_27d(rgb01)
    pat, _ = dino_forward_tokens(model, rgb01_to_pil(rgb01))
    mu, sd = pat.mean(dim=1), pat.std(dim=1, unbiased=False)
    return torch.cat([col27, torch.cat([mu, sd], dim=-1)], dim=-1).numpy()


def fiber_from_paths(model, paths, batch: int = 8) -> np.ndarray:
    import torch

    outs = []
    for i in range(0, len(paths), batch):
        rgb = torch.stack([pil_to_rgb01(Image.open(p)) for p in paths[i : i + batch]])
        outs.append(fiber795_from_rgb01(model, rgb))
    return np.concatenate(outs, 0)


# --------------------------------------------------------------------------
# populations
# --------------------------------------------------------------------------
def load_tasks_and_refs():
    tasks = json.loads((ROOT / "eval_protocol/tasks_750.json").read_text(encoding="utf-8-sig"))["tasks"]
    refs = json.loads((ROOT / "eval_protocol/references_v1.json").read_text(encoding="utf-8-sig"))["references"]
    return tasks, refs


def banned_pids(tasks, refs) -> set:
    banned = {canonical_pid(t["source_stem"]) for t in tasks}
    for rows in refs.values():
        banned |= {canonical_pid(r["stem"]) for r in rows}
    return banned


def p1_eval_sources(tasks):
    """150 unique test750 sources: stem -> (first task idx, sid, jpg path)."""
    seen = {}
    for i, t in enumerate(tasks):
        if t["source_stem"] not in seen:
            seen[t["source_stem"]] = (i, int(t["source_style_id"]), ROOT / t["source_rel_path"])
    return seen


def p2_train_sample(tasks, refs, n_per_style: int, rng):
    """Seeded sample of the COMBINED v18 pool population (canonical-pid dedup).

    tools/pseudo_gt/build_assets.py enumerate_images() builds the latent pool
    from datasets/wikiart_new_512 (manifest order, wikiart ids first) PLUS
    datasets/distinct5_512/train; exp.v18.clean_balanced_pool then dedups by
    canonical_pid and samples 880/style train + 60/style holdout. This proxy
    mirrors that combined population (distribution level; the exact 880/style
    membership needs the remote pool manifest).
    """
    banned = banned_pids(tasks, refs)
    manifest = [json.loads(l) for l in (ROOT / "weights_local/wikiart_new_512/manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    out, skipped, seen = [], 0, set()
    by_style = {s: [] for s in STYLES}
    for row in manifest:  # wikiart first (matches build_assets enumeration order)
        pid = canonical_pid(row["filename"])
        seen.add(pid)
        if pid not in banned:
            by_style[row["style"]].append(ROOT / "datasets/wikiart_new_512" / row["filename"])
        else:
            skipped += 1
    for sid, style in enumerate(STYLES):
        d = ROOT / "datasets/distinct5_512/train" / style
        extra = [f for f in sorted(d.glob("*.jpg")) if canonical_pid(f.stem) not in seen]
        for f in extra:
            seen.add(canonical_pid(f.stem))
            if canonical_pid(f.stem) not in banned:
                by_style[style].append(f)
            else:
                skipped += 1
    for sid, style in enumerate(STYLES):
        files = sorted(by_style[style])
        pick = rng.permutation(len(files))[:n_per_style]
        for j in sorted(pick.tolist()):
            out.append((sid, files[j]))
    return out, skipped


def sg5_files():
    from exp.v21.stiefel_ref_cache import SG_5_STEMS

    out = []
    for s, stem in enumerate(SG_5_STEMS):
        d = ROOT / "datasets/distinct5_512/train" / STYLES[s]
        hit = [f for f in d.glob("*.jpg") if canonical_pid(f.stem) == canonical_pid(stem)]
        if not hit:
            raise FileNotFoundError(f"SG-5 stem not found locally: {stem}")
        out.append((s, hit[0]))
    return out


# --------------------------------------------------------------------------
# numpy analysis helpers
# --------------------------------------------------------------------------
def standardize(X, mean=None, std=None, clamp: float = 8.0):
    mean = X.mean(0) if mean is None else mean
    std = np.clip(X.std(0), 1e-4, None) if std is None else std
    return np.clip((X - mean) / std, -clamp, clamp), mean, std


def pca_report(X, tag: str):
    w_all = np.linalg.eigvalsh(((X - X.mean(0)).T @ (X - X.mean(0))) / max(len(X) - 1, 1))
    w = np.clip(w_all[::-1], 0.0, None)
    rank = int((w > w.max() * 1e-10).sum()) if w.max() > 0 else 0
    tot = float(w.sum())
    cum = np.cumsum(w) / max(tot, 1e-12)

    def k_at(q: float) -> int:
        return int(np.searchsorted(cum, q) + 1)

    p = w / max(tot, 1e-12)
    p = p[p > 0]
    ent = float(np.exp(-(p * np.log(p)).sum()))
    return {
        "tag": tag,
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "numeric_rank": min(rank, len(X) - 1),
        "eig_top12": [round(float(v), 6) for v in w[:12]],
        "cum_variance": {
            "k90": k_at(0.90), "k95": k_at(0.95), "k99": k_at(0.99),
            "var_top8": round(float(cum[min(7, len(cum) - 1)]), 4),
            "var_top16": round(float(cum[min(15, len(cum) - 1)]), 4),
            "var_top32": round(float(cum[min(31, len(cum) - 1)]), 4),
        },
        "participation_ratio": round(float(tot ** 2 / max((w ** 2).sum(), 1e-12)), 2),
        "entropy_effective_dim": round(ent, 2),
    }


def fisher_ratio(Xs, y) -> float:
    mu = Xs.mean(0)
    sb = sw = 0.0
    for c in np.unique(y):
        Xc = Xs[y == c]
        mc = Xc.mean(0)
        sb += len(Xc) * float(((mc - mu) ** 2).sum())
        sw += float(((Xc - mc) ** 2).sum())
    return round(sb / max(sw, 1e-12), 4)


def fstat_significant_dims(Xs, y, alpha: float = 0.01) -> dict:
    from scipy.stats import f as fdist
    from sklearn.feature_selection import f_classif

    F, pval = f_classif(Xs, y)
    # Bonferroni-style guard: require p < alpha/d and F well above its null quantile
    d = Xs.shape[1]
    thresh = float(fdist.ppf(1 - alpha / max(d, 1), len(STYLES) - 1, max(len(Xs) - len(STYLES), 1)))
    sig = (pval < alpha) & (F > thresh)
    return {
        "n_significant_dims": int(sig.sum()),
        "frac_significant": round(float(sig.mean()), 4),
        "f_threshold": round(thresh, 2),
        "f_top10": [round(float(v), 1) for v in np.sort(F)[::-1][:10]],
    }


def silhouette(Xs, y) -> float:
    from sklearn.metrics import silhouette_score

    return round(float(silhouette_score(Xs, y, metric="cosine")), 4)


def _closest_pair(C):
    """Most-similar pair (max off-diagonal cosine) and most-opposite pair (min)."""
    most_sim = (-9.0, "", "")
    most_opp = (9.0, "", "")
    for i in range(len(STYLES)):
        for j in range(i + 1, len(STYLES)):
            v = float(C[i, j])
            if v > most_sim[0]:
                most_sim = (round(v, 4), STYLES[i], STYLES[j])
            if v < most_opp[0]:
                most_opp = (round(v, 4), STYLES[i], STYLES[j])
    return {
        "most_similar": {"cos": most_sim[0], "pair": [most_sim[1], most_sim[2]]},
        "most_opposite": {"cos": most_opp[0], "pair": [most_opp[1], most_opp[2]]},
    }


def centroid_cosine_matrix(Xs, y):
    cents = np.stack([Xs[y == c].mean(0) for c in range(len(STYLES))])
    cents = cents / np.clip(np.linalg.norm(cents, axis=1, keepdims=True), 1e-12, None)
    C = cents @ cents.T
    off = C[~np.eye(len(STYLES), dtype=bool)]
    return {
        "matrix_5x5": [[round(float(v), 4) for v in row] for row in C],
        "offdiag_mean": round(float(off.mean()), 4),
        "offdiag_max": round(float(off.max()), 4),
        "offdiag_min": round(float(off.min()), 4),
        "closest_pair": _closest_pair(C),
        "style_order": STYLES,
    }


def logistic_acc_curve(Xs, y, ks, seed: int):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = list(skf.split(Xs, y))
    out = {}
    for k in list(ks) + ["full"]:
        accs = []
        for tr, te in folds:
            if k == "full":
                Xtr, Xte = Xs[tr], Xs[te]
            else:
                kk = min(int(k), Xs.shape[1], len(tr) - 1)
                pca = PCA(n_components=kk)
                Xtr = pca.fit_transform(Xs[tr])
                Xte = pca.transform(Xs[te])
            clf = LogisticRegression(max_iter=1000, C=1.0)
            clf.fit(Xtr, y[tr])
            accs.append(float((clf.predict(Xte) == y[te]).mean()))
        out[str(k)] = round(float(np.mean(accs)), 4)
    return out


def cca_and_ridge(A, B, n_comp: int = 5, alpha: float = 10.0, seed: int = SEED):
    from sklearn.cross_decomposition import CCA
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    # out-of-fold canonical correlations (in-sample CCA with dim(A) << dim(B) hits 1.0 by overfit)
    oof_cors = [[] for _ in range(n_comp)]
    r2_ab, r2_ba = [], []
    for tr, te in kf.split(A):
        cca = CCA(n_components=n_comp, max_iter=300, tol=1e-6)
        ta_tr, tb_tr = cca.fit_transform(A[tr], B[tr])
        ta_te, tb_te = cca.transform(A[te], B[te])
        for i in range(n_comp):
            oof_cors[i].append(float(np.corrcoef(ta_te[:, i], tb_te[:, i])[0, 1]))
        r2_ab.append(float(Ridge(alpha=alpha).fit(A[tr], B[tr]).score(A[te], B[te])))
        r2_ba.append(float(Ridge(alpha=alpha).fit(B[tr], A[tr]).score(B[te], A[te])))
    return {
        "cca_cv_canonical_correlations": [round(float(np.mean(c)), 4) for c in oof_cors],
        "ridge_cv_r2_A_to_B": round(float(np.mean(r2_ab)), 4),
        "ridge_cv_r2_B_to_A": round(float(np.mean(r2_ba)), 4),
        "note": "A=first arg, B=second arg; canonical correlations are out-of-fold (5-fold CV); ridge alpha=10; in-sample CCA with dim(A)<<dim(B) saturates at 1.0 by overfit and is not reported",
    }


def style_geometry(X_raw, y, tag: str, ks=(2, 4, 8, 16, 32, 64, 128, 256), with_curve: bool = True):
    Xs, _, _ = standardize(X_raw)
    rep = {
        "tag": tag,
        "n": int(len(X_raw)),
        "dim": int(X_raw.shape[1]),
        "fisher_ratio": fisher_ratio(Xs, y),
        "silhouette_cosine": silhouette(Xs, y),
        "centroid_cosine": centroid_cosine_matrix(Xs, y),
        "fstat": fstat_significant_dims(Xs, y),
    }
    if with_curve:
        rep["logistic_acc_by_pca_k"] = logistic_acc_curve(Xs, y, ks, SEED)
    return rep


# --------------------------------------------------------------------------
# analysis driver
# --------------------------------------------------------------------------
def run_analysis(args) -> dict:
    import torch

    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    t0 = time.time()
    tasks, refs = load_tasks_and_refs()
    srcs = p1_eval_sources(tasks)
    assert len(srcs) == 150, f"expected 150 unique sources, got {len(srcs)}"

    model = load_dino_cpu()
    out: dict = {"meta": {}, "p1_eval150": {}, "proxy_gap": {}, "p2_train_proxy": {},
                 "p3_oneshot": {}, "cross_population": {}, "u128": None, "convnext": None}

    # ---------------- P2 train proxy (600 x 795) ----------------
    p2_rows, skipped_banned = p2_train_sample(tasks, refs, N_PER_STYLE, rng)
    fib_p2 = fiber_from_paths(model, [p for _, p in p2_rows])
    y_p2 = np.array([s for s, _ in p2_rows])
    pids_p2 = np.array([canonical_pid(p.stem) for _, p in p2_rows])
    fib_p2_mean, fib_p2_std = fib_p2.mean(0), np.clip(fib_p2.std(0), 1e-4, None)
    print(f"[A05] P2 train-proxy fiber {fib_p2.shape} ({time.time()-t0:.0f}s)", flush=True)

    t1 = time.time()

    out["p2_train_proxy"] = {
        "population": f"combined v18 pool population (wikiart_new_512 + distinct5_512/train, canonical-pid dedup, wikiart first) seeded sample {N_PER_STYLE}/style (seed {SEED}); {skipped_banned} banned test750-source/reference files excluded",
        "full_795d": style_geometry(fib_p2, y_p2, "train_proxy_795d"),
        "oklab27_block": style_geometry(fib_p2[:, :OKLAB_DIM], y_p2, "train_proxy_oklab27"),
        "dino768_block": style_geometry(fib_p2[:, OKLAB_DIM:], y_p2, "train_proxy_dino768"),
        "pca_full795": pca_report(fib_p2, "train_proxy_795d"),
        "pca_oklab27": pca_report(fib_p2[:, :OKLAB_DIM], "train_proxy_oklab27"),
        "pca_dino768": pca_report(fib_p2[:, OKLAB_DIM:], "train_proxy_dino768"),
        "internal_redundancy_oklab_vs_dino": cca_and_ridge(fib_p2[:, :OKLAB_DIM], fib_p2[:, OKLAB_DIM:]),
    }
    print(f"[A05] P2 geometry analysis done ({time.time()-t1:.0f}s)", flush=True)

    t1 = time.time()

    # ---------------- P1 eval150 (original RGB + production S10 path) ----------------
    rows = sorted(srcs.items(), key=lambda kv: kv[1][0])
    stems = [k for k, _ in rows]
    sids = np.array([v[1] for _, v in rows])
    jpgs = [v[2] for _, v in rows]
    fib_orig = fiber_from_paths(model, jpgs)
    out["p1_eval150"] = {
        "population": "150 unique test750 sources (data/test JPEGs resized 512 bilinear)",
        "original_rgb": style_geometry(fib_orig, sids, "eval150_original_rgb", with_curve=False),
    }
    print(f"[A05] P1 original-RGB fiber done ({time.time()-t0:.0f}s)", flush=True)

    if not args.skip_s10_decode:
        from src.utils import load_s10_decoder, decode_s10

        n2c = torch.load(ROOT / "runs/v9_local_4070/cache/n2c_z.pt", map_location="cpu", weights_only=False)
        first_rows = [v[0] for _, v in rows]
        z = n2c["z_src"][first_rows].float()
        s10 = load_s10_decoder("cpu").float()
        fib_dec_chunks, cls_dec_chunks = [], []
        for i in range(0, len(z), 8):
            rgb_dec = decode_s10(s10, z[i : i + 8], batch=4, dtype=torch.float32)
            fib_dec_chunks.append(fiber795_from_rgb01(model, rgb_dec))
            _, cl_d = dino_forward_tokens(model, rgb01_to_pil(rgb_dec))
            cls_dec_chunks.append(cl_d.numpy())
        fib_dec = np.concatenate(fib_dec_chunks, 0)
        cls_dec = np.concatenate(cls_dec_chunks, 0)
        _, cls_orig = dino_forward_tokens(model, [resize512(p) for p in [Image.open(p) for p in jpgs]])
        cls_orig = cls_orig.numpy()
        del s10
        print(f"[A05] P1 S10 decode+fiber done ({time.time()-t1:.0f}s)", flush=True)

        t1 = time.time()

        out["p1_eval150"]["s10_decoded_z_src"] = style_geometry(fib_dec, sids, "eval150_s10_decoded", with_curve=False)

        # proxy gap: same 150 images, original RGB vs production S10(z_src) path
        def _cos(A, B):
            A = A - A.mean(0, keepdims=True)
            B = B - B.mean(0, keepdims=True)
            an = np.clip(np.linalg.norm(A, axis=1), 1e-12, None)
            bn = np.clip(np.linalg.norm(B, axis=1), 1e-12, None)
            return (A * B).sum(1) / (an * bn)

        g_full = _cos(fib_orig, fib_dec)
        g_ok = _cos(fib_orig[:, :OKLAB_DIM], fib_dec[:, :OKLAB_DIM])
        g_dino = _cos(fib_orig[:, OKLAB_DIM:], fib_dec[:, OKLAB_DIM:])
        cc = (cls_orig * cls_dec).sum(1)
        Cc_o = np.array(centroid_cosine_matrix(standardize(fib_orig)[0], sids)["matrix_5x5"])
        Cc_d = np.array(centroid_cosine_matrix(standardize(fib_dec)[0], sids)["matrix_5x5"])
        out["proxy_gap"] = {
            "note": "original data/test JPEG vs production path decode_s10(z_src); same 150 images",
            "fiber795_cosine_after_centering": {"mean": round(float(g_full.mean()), 4), "p50": round(float(np.median(g_full)), 4), "min": round(float(g_full.min()), 4)},
            "oklab27_cosine_mean": round(float(g_ok.mean()), 4),
            "dino768_cosine_mean": round(float(g_dino.mean()), 4),
            "dino_cls_cosine_orig_vs_decoded": {"mean": round(float(cc.mean()), 4), "min": round(float(cc.min()), 4)},
            "centroid_cos_matrix_max_abs_delta": round(float(np.max(np.abs(Cc_o - Cc_d))), 4),
            "fisher_ratio_orig_vs_decoded": [out["p1_eval150"]["original_rgb"]["fisher_ratio"], out["p1_eval150"]["s10_decoded_z_src"]["fisher_ratio"]],
        }

        # fp32-vs-fp16 decode numeric gap (production decodes fp16 on GPU)
        if not args.skip_fp16_check:
            s10h = load_s10_decoder("cpu")  # ships as .half()
            z3 = z[:1]
            print(f"[A05] fp16 probe start ({time.time()-t1:.0f}s)", flush=True)
            d16 = decode_s10(s10h, z3, batch=1, dtype=torch.float16)
            print(f"[A05] fp16 decode done ({time.time()-t1:.0f}s)", flush=True)
            d32 = decode_s10(s10h.float(), z3, batch=1, dtype=torch.float32)
            diff = (d16.float() - d32).abs() * 255.0
            f16 = fiber795_from_rgb01(model, d16.float())
            f32 = fiber795_from_rgb01(model, d32)
            cos_pair = float(f16[0] @ f32[0] / max(np.linalg.norm(f16[0]) * np.linalg.norm(f32[0]), 1e-12))
            out["proxy_gap"]["decode_fp16_vs_fp32"] = {
                "rgb_mae_255": round(float(diff.mean()), 3),
                "rgb_maxdiff_255": round(float(diff.max()), 3),
                "fiber795_cosine_1latent_raw": round(cos_pair, 6),
                "note": "1 latent; CPU fp16 decode is ~100x slower than fp32 on 2 threads, so the 150-image variant decodes in fp32 (diff measured here is negligible at analysis resolution)",
            }
            del s10h
            print(f"[A05] fp16-vs-fp32 probe done ({time.time()-t1:.0f}s)", flush=True)
        print(f"[A05] P1 S10-decoded fiber + proxy gap done ({time.time()-t0:.0f}s)", flush=True)

    # ---------------- P3 One-Shot reference side ----------------
    sg5 = sg5_files()
    fib_sg5 = fiber_from_paths(model, [p for _, p in sg5])
    ref_rows = [(sid, ROOT / r["rel_path"]) for sid, style in enumerate(STYLES) for r in refs[style]]
    fib_ref = fiber_from_paths(model, [p for _, p in ref_rows])
    y_ref = np.array([s for s, _ in ref_rows])
    print(f"[A05] P3 oneshot fibers done ({time.time()-t0:.0f}s)", flush=True)

    # standardization + centroids strictly from P2 train proxy (label-free side)
    def _std_by_p2(F):
        return np.clip((F - fib_p2_mean) / fib_p2_std, -8, 8)

    Xs_p2 = _std_by_p2(fib_p2)
    cents = np.stack([Xs_p2[y_p2 == c].mean(0) for c in range(5)])
    cents = cents / np.clip(np.linalg.norm(cents, axis=1, keepdims=True), 1e-12, None)

    def _centric(F, y):
        Z = _std_by_p2(F)
        Zn = Z / np.clip(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12, None)
        C = Zn @ cents.T
        own = C[np.arange(len(y)), y]
        other = np.where(np.eye(5, dtype=bool)[y], -np.inf, C).max(1)
        pred = C.argmax(1)
        return {
            "nearest_centroid_is_own_style_acc": round(float((pred == y).mean()), 4),
            "own_centroid_cosine_mean": round(float(own.mean()), 4),
            "margin_own_minus_best_other_mean": round(float((own - other).mean()), 4),
            "margin_p10": round(float(np.percentile(own - other, 10)), 4),
        }

    Z_sg5 = _std_by_p2(fib_sg5)
    Z_sg5n = Z_sg5 / np.clip(np.linalg.norm(Z_sg5, axis=1, keepdims=True), 1e-12, None)
    out["p3_oneshot"] = {
        "sg5_exact5_standardized_by_p2": _centric(fib_sg5, np.arange(5)),
        "sg5_centroid_cosine_5x5": [[round(float(v), 4) for v in row] for row in (Z_sg5n @ cents.T)],
        "sg5_style_order": STYLES,
        "references_v1_150_standardized_by_p2": _centric(fib_ref, y_ref),
        "note": "standardization and centroids come from the P2 train proxy; this is the label-free conditioning-side geometry the One-Shot u_glob module consumes (795D standardized fiber)",
    }

    # cross-population PCA transfer (train proxy basis applied to eval side)
    from sklearn.decomposition import PCA

    k95 = out["p2_train_proxy"]["pca_full795"]["cum_variance"]["k95"]
    pca = PCA(n_components=min(k95, fib_p2.shape[1])).fit(Xs_p2)
    var_cap = float(pca.transform(_std_by_p2(fib_orig)).var(0).sum() / _std_by_p2(fib_orig).var(0).sum())
    out["cross_population"] = {
        "p2_pca_k95_components_explained_variance_on_p1": round(var_cap, 4),
        "note": "PCA fitted on standardized P2 (train proxy), applied to standardized P1 (eval side); 1.0 = no distribution shift",
    }

    # ---------------- optional npz consumers ----------------
    if args.u_npz:
        out["u128"] = analyze_u_npz(Path(args.u_npz))
    if args.cnx_npz:
        out["convnext"] = analyze_cnx_npz(Path(args.cnx_npz), fib_p2, y_p2, pids_p2)

    out["meta"] = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": SEED,
        "wall_sec": round(time.time() - t0, 1),
        "style_order": STYLES,
        "p2_pids": pids_p2.tolist(),
        "p1_source_stems": stems,
        "protocol_notes": [
            "Oklab27 block: exp.v19.fiber_duality.extract_color_fiber_27d imported verbatim (identical math to production)",
            "DINO768 block: DINO_TRANSFORM(Resize224/BICUBIC+CenterCrop224) + per-token L2-normalized patch moments (mean|std), replicating exp.v19.fiber_duality.extract_dino_rgb_moments_768d; local snapshot is the 257-token (no-register) dinov2-small",
            "DINO forward runs CPU fp32; production uses CUDA fp16 autocast (fp16-vs-fp32 impact measured in proxy_gap.decode_fp16_vs_fp32 when not skipped)",
            "S10-decoded variant uses deliver s10_p5_lowlr decoder, CPU fp32, on runs/v9_local_4070/cache/n2c_z.pt z_src rows (first occurrence per unique source); production decodes fp16 on GPU",
            "P2 is a proxy for the exact train_4400 (pool membership lives in the remote pseudo_gt pool); 128D u-space and ConvNeXt analyses require remote exports (u128/convnext sections stay null until --u-npz/--cnx-npz provided)",
            "DATA-03 boundary honored: analysis only, no feature branch is removed or changed",
        ],
    }
    return out


# --------------------------------------------------------------------------
# npz consumers (128D u-space / ConvNeXt pooled)
# --------------------------------------------------------------------------
def analyze_u_npz(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    u_tr, sid_tr = d["u_tr"].astype(np.float64), d["sid_tr"].astype(int)
    rep: dict = {"source": path.name, "n_train": int(len(u_tr)), "dim": int(u_tr.shape[1])}
    rep["pca"] = pca_report(u_tr, "u128_train")
    Xs_tr = standardize(u_tr)[0]
    rep["fisher_ratio_train"] = fisher_ratio(Xs_tr, sid_tr)
    rep["centroid_cosine_train"] = centroid_cosine_matrix(Xs_tr, sid_tr)
    step = max(1, len(u_tr) // 1500)
    rep["silhouette_cosine_train"] = silhouette(Xs_tr[::step], sid_tr[::step])
    rep["logistic_acc_by_pca_k"] = logistic_acc_curve(Xs_tr, sid_tr, (2, 4, 8, 16, 32, 64, 128), SEED)
    if "u_ho" in d and "sid_ho" in d:
        u_ho, sid_ho = d["u_ho"].astype(np.float64), d["sid_ho"].astype(int)
        Xs_ho = standardize(u_ho)[0]
        rep["fisher_ratio_holdout300"] = fisher_ratio(Xs_ho, sid_ho)
        rep["centroid_cosine_holdout300"] = centroid_cosine_matrix(Xs_ho, sid_ho)
        rep["holdout_silhouette_cosine"] = silhouette(Xs_ho, sid_ho)
    if "tr_fib" in d:
        fib_tr = d["tr_fib"].astype(np.float64)
        rep["fisher_ratio_fiber795_train"] = fisher_ratio(standardize(fib_tr)[0], sid_tr)
        rep["pca_fiber795_train"] = pca_report(fib_tr, "fiber795_train_exact4400")
    if "protos" in d:
        p = d["protos"].astype(np.float64)
        p = p / np.clip(np.linalg.norm(p, axis=1, keepdims=True), 1e-12, None)
        rep["prototype_cosine_5x5"] = [[round(float(v), 4) for v in row] for row in (p @ p.T)]
        rep["prototype_style_order"] = STYLES
    return rep


def analyze_cnx_npz(path: Path, fib_p2: np.ndarray, y_p2: np.ndarray, pids_p2: np.ndarray) -> dict:
    d = np.load(path, allow_pickle=True)
    C, sid_c = d["cnx_pooled"].astype(np.float64), d["sids"].astype(int)
    pid_c = np.array([str(x) for x in d["pids"]])
    rep: dict = {"source": path.name, "n": int(len(C)), "dim": int(C.shape[1]),
                 "fisher_ratio_cnx": fisher_ratio(standardize(C)[0], sid_c),
                 "pca_cnx": pca_report(C, "cnx_pooled")}
    pos = {p: i for i, p in enumerate(pid_c)}
    idx_c, idx_f = [], []
    for i, p in enumerate(pids_p2):
        if p in pos:
            idx_f.append(i)
            idx_c.append(pos[p])
    rep["matched_n"] = len(idx_f)
    if len(idx_f) >= 200:
        Fs = standardize(fib_p2[idx_f])[0]
        Cs = standardize(C[idx_c])[0]
        rep["redundancy_full795_vs_cnx"] = cca_and_ridge(Fs, Cs)
        rep["logistic_acc_matched"] = {
            "fiber795_only": logistic_acc_curve(Fs, y_p2[idx_f], ["full"], SEED)["full"],
            "cnx_only": logistic_acc_curve(Cs, y_p2[idx_f], ["full"], SEED)["full"],
            "concat_795_plus_cnx": logistic_acc_curve(np.hstack([Fs, Cs]), y_p2[idx_f], ["full"], SEED)["full"],
        }
    else:
        rep["redundancy_full795_vs_cnx"] = "BLOCKED: matched overlap < 200 (export must carry distinct5 train pids)"
    return rep


# --------------------------------------------------------------------------
# remote export modes (CUDA-gated; never runnable on this CPU slot)
# --------------------------------------------------------------------------
def _require_cuda(mode: str) -> str:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"--{mode} requires CUDA (4090/3060 remote queue only); "
            "the local Slot-A analysis must not use this flag"
        )
    return "cuda"


def export_u(args) -> None:
    """4090 export: u vectors + raw 795D fiber matrices + labels for the exact 128D analysis."""
    import numpy as np
    import torch

    device = _require_cuda("export-u")
    torch.set_num_threads(2)
    from exp.v19.fiber_duality import StyleFiberEncoder, build_or_load_v19_foundation

    pool = args.pool or str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16")
    bundle = build_or_load_v19_foundation(pool, device=device)
    fnd = bundle["fnd"]
    enc = StyleFiberEncoder(in_dim=fnd["tr_fib"].shape[1], dim=128).to(device).eval()
    enc.load_state_dict(fnd["fiber_state"])
    with torch.no_grad():
        u_tr = enc.encode(fnd["tr_fib"].to(device).float()).cpu().numpy()
        u_ho = enc.encode(fnd["ho_fib"].to(device).float()).cpu().numpy()
        u_750 = enc.encode(fnd["test750_fib"].to(device).float()).cpu().numpy()
    tr, ho = bundle["v18_cache"]["train"], bundle["v18_cache"]["holdout"]
    payload = {
        "u_tr": u_tr.astype(np.float16), "sid_tr": tr["sid"].numpy().astype(np.int64),
        "ids_tr": np.array([str(x) for x in tr["ids"]]),
        "u_ho": u_ho.astype(np.float16), "sid_ho": ho["sid"].numpy().astype(np.int64),
        "u_750": u_750.astype(np.float16),
        "tr_fib": fnd["tr_fib"].numpy().astype(np.float16),
        "ho_fib": fnd["ho_fib"].numpy().astype(np.float16),
        "protos": enc.class_barycenters().cpu().numpy().astype(np.float32),
    }
    sg5_cache = ROOT / "runs/v21/cache/v21_stiefel_ref_cache.pt"
    if sg5_cache.exists():
        c21 = torch.load(sg5_cache, map_location="cpu", weights_only=False)
        payload["u_sg5"] = c21["sg5"]["u"].numpy().astype(np.float16)
        payload["sg5_stems"] = np.array(list(c21["sg5"]["stems"]))
    out = Path(args.out or "runs/v25/cache/a05_u_export.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)
    print(f"[A05 export-u] wrote {out} ({out.stat().st_size / 1e6:.1f} MB) keys={sorted(payload)}", flush=True)


def export_cnx(args) -> None:
    """3060/4090 export: global-pooled RGB ConvNeXt v16 (192-D) for train_4400 (+ optional test750)."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    device = _require_cuda("export-cnx")
    torch.set_num_threads(2)
    from exp.v20.submode_patchswd import load_convnext_tiny_offline
    from exp.v19.fiber_duality import build_or_load_v19_foundation
    from src.utils import decode_s10, load_s10_decoder
    from exp.v18 import tasks_and_sources

    pool = args.pool or str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16")
    bundle = build_or_load_v19_foundation(pool, device=device)
    tr = bundle["v18_cache"]["train"]
    dec = load_s10_decoder(device)
    cnx = load_convnext_tiny_offline(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def _pooled(z_all):
        outs = []
        for st in range(0, len(z_all), 32):
            rgb = decode_s10(dec, z_all[st : st + 32].to(device), batch=16, output_cpu=False)
            x = (F.interpolate(rgb.float(), size=(256, 256), mode="bilinear", align_corners=False) - mean) / std
            v16 = F.avg_pool2d(cnx[:4](x), 2, 2)
            outs.append(v16.mean(dim=(2, 3)).cpu().float())
        return torch.cat(outs, 0).numpy().astype(np.float16)

    payload = {
        "cnx_pooled": _pooled(tr["z"]),
        "sids": tr["sid"].numpy().astype(np.int64),
        "pids": np.array([str(x) for x in tr["ids"]]),
    }
    n2c_path = ROOT / "runs/v9_local_4070/cache/n2c_z.pt"
    if n2c_path.exists():
        _, z_src_750, _ = tasks_and_sources()
        payload["cnx_750"] = _pooled(z_src_750.float())
    sg5_cache = ROOT / "runs/v21/cache/v21_stiefel_ref_cache.pt"
    if sg5_cache.exists():
        c21 = torch.load(sg5_cache, map_location="cpu", weights_only=False)
        payload["sg5_stems"] = np.array(list(c21["sg5"]["stems"]))
    out = Path(args.out or "runs/v25/cache/a05_cnx_export.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)
    print(f"[A05 export-cnx] wrote {out} ({out.stat().st_size / 1e6:.1f} MB)", flush=True)


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="A05: 128D/795D style feature effective-dimension diagnosis (CPU)")
    ap.add_argument("--u-npz", default="", help="path to a05_u_export.npz (enables the 128D analysis)")
    ap.add_argument("--cnx-npz", default="", help="path to a05_cnx_export.npz (enables the ConvNeXt redundancy analysis)")
    ap.add_argument("--skip-s10-decode", action="store_true", help="skip the production-path variant (fast fallback)")
    ap.add_argument("--skip-fp16-check", action="store_true", help="skip the fp16/fp32 decode diff probe (saves ~3 min)")
    ap.add_argument("--out", default="", help="override output JSON path")
    ap.add_argument("--export-u", action="store_true", help="4090 only: dump u vectors + fiber matrices npz")
    ap.add_argument("--export-cnx", action="store_true", help="3060/4090 only: dump pooled ConvNeXt v16 npz")
    ap.add_argument("--pool", default="", help="pool dir for export modes (default ../pseudo_gt_24h/orig_mean_fp16)")
    args = ap.parse_args()

    if args.export_u:
        export_u(args)
        return 0
    if args.export_cnx:
        export_cnx(args)
        return 0

    os.environ["CUDA_VISIBLE_DEVICES"] = ""   # analysis mode is strictly CPU-only
    result = run_analysis(args)
    tgt = Path(args.out) if args.out else OUT_JSON
    save_json(tgt, result)

    p2 = result["p2_train_proxy"]
    print("\n=== A05 feature geometry summary ===")
    for blk in ("full_795d", "oklab27_block", "dino768_block"):
        g = p2[blk]
        cp = g["centroid_cosine"]["closest_pair"]
        print(f"P2 {blk:14s} Fisher={g['fisher_ratio']:8.3f} Sil(cos)={g['silhouette_cosine']:6.3f} "
              f"sigDims={g['fstat']['n_significant_dims']:3d}/{g['dim']} "
              f"mostSim=({cp['most_similar']['pair'][0][:12]},{cp['most_similar']['pair'][1][:12]}) cos={cp['most_similar']['cos']} "
              f"mostOpp=({cp['most_opposite']['pair'][0][:12]},{cp['most_opposite']['pair'][1][:12]}) cos={cp['most_opposite']['cos']}")
        if "logistic_acc_by_pca_k" in g:
            print(f"    acc_by_k={g['logistic_acc_by_pca_k']}")
    pca = p2["pca_full795"]
    print(f"P2 795D PCA: PR={pca['participation_ratio']} entropyDim={pca['entropy_effective_dim']} "
          f"k90/95/99={pca['cum_variance']['k90']}/{pca['cum_variance']['k95']}/{pca['cum_variance']['k99']}")
    print(f"redundancy oklab<->dino: {p2['internal_redundancy_oklab_vs_dino']}")
    if result["proxy_gap"]:
        pg = result["proxy_gap"]
        print(f"proxy gap: fiber cos={pg['fiber795_cosine_after_centering']['mean']} "
              f"cls cos={pg['dino_cls_cosine_orig_vs_decoded']['mean']} "
              f"centroid max|d|={pg['centroid_cos_matrix_max_abs_delta']}")
    p3 = result["p3_oneshot"]
    print(f"P3 SG-5 own-acc={p3['sg5_exact5_standardized_by_p2']['nearest_centroid_is_own_style_acc']} "
          f"margin={p3['sg5_exact5_standardized_by_p2']['margin_own_minus_best_other_mean']} | "
          f"refs150 own-acc={p3['references_v1_150_standardized_by_p2']['nearest_centroid_is_own_style_acc']}")
    print(f"cross-pop: P2 PCA-k95 explains {result['cross_population']['p2_pca_k95_components_explained_variance_on_p1']} of P1 variance")
    if result["u128"]:
        print(f"u128 PCA: PR={result['u128']['pca']['participation_ratio']} k95={result['u128']['pca']['cum_variance']['k95']} fisher={result['u128']['fisher_ratio_train']}")
    else:
        print("u128: BLOCKED locally -> run --export-u on 4090 (command in report)")
    if not result["convnext"]:
        print("convnext: BLOCKED locally -> run --export-cnx on 3060/4090 (command in report)")
    print(f"[A05] wrote {tgt} ({result['meta']['wall_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
