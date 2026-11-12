"""Fast CPU+RAM style-geometry evaluation via true multi-process parallelism.

Key CPU infa optimizations (vs naive single-process torch.set_num_threads(64)):
  * many small-thread workers instead of one fat process: CPU conv scales poorly
    past ~16 threads (measured 0.43 -> 0.58 img/s from 16 -> 64 threads).
  * torch.jit.optimize_for_inference: MKLDNN weight pre-packing + conv fusion.
    freeze() alone does NOT repack weights -> this was the missing speedup.
  * channels_last memory format for blocked oneDNN conv kernels.
  * fork context + parent pre-loads the traced module -> workers inherit it
    (copy-on-write), no per-worker import of diffusers/transformers/accelerate.
  * sched_setaffinity pins each worker to a contiguous core range (2-socket
    EPYC NUMA), avoiding cross-socket traffic.

Parent: builds traced encoder once per checkpoint, shards images, gathers latents,
computes all metrics. Workers: CPU only, never touch CUDA, never load the decoder.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

OUT = Path("/root/Workspace/WEAVE_VAE/runs/vae_style_rank20")
MANIFEST = Path("/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")
A0_DIR = "/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae"
TMP = OUT / "cpu_traced"
TMP.mkdir(parents=True, exist_ok=True)

_G = {}


# ---------------------------------------------------------------- worker (CPU only)
def _worker(task):
    tag, shard, start, end, bs, threads, cores = task
    try:
        if cores:
            os.sched_setaffinity(0, cores)
        torch.set_num_threads(threads)
        torch.backends.mkldnn.enabled = True
        enc = _G["enc"]
        reader = _G["reader"]
        raws = _G["raws"]
        out = []
        with torch.inference_mode():
            for i in range(start, end, bs):
                idx = list(range(i, min(i + bs, end)))
                arr = np.stack([np.ascontiguousarray(reader[raws[j]]) for j in idx])
                x = torch.from_numpy(arr).permute(0, 3, 1, 2).float()
                x = x.div_(127.5).sub_(1.0).contiguous(memory_format=torch.channels_last)
                out.append(enc(x).numpy().astype(np.float16))
        return shard, np.concatenate(out)
    except Exception as e:
        return shard, f"ERROR: {type(e).__name__}: {e}"


# ---------------------------------------------------------------- traced encoder
class EncoderOnly(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.encoder = vae.encoder
        self.quant_conv = vae.quant_conv

    def forward(self, x):
        h = self.quant_conv(self.encoder(x))
        return h[:, :4].flatten(1).float()


def build_encoder(tag, ck):
    """Build an eager encoder-only module in the PARENT (no forward, no OpenMP pool).

    Workers inherit it via fork copy-on-write; no jit.trace (VAE downsampling has
    data-dependent python bools that break tracing), no per-worker imports.
    """
    from diffusers import AutoencoderKL          # lazy: heavy import, parent-only
    t0 = time.time()
    vae = AutoencoderKL.from_pretrained(A0_DIR, torch_dtype=torch.float32)
    if ck is not None:
        vae.load_state_dict(ck["model"])
    m = EncoderOnly(vae).eval().to(memory_format=torch.channels_last)
    print(f"[Build] {tag}: encoder-only, channels_last, {time.time()-t0:.1f}s", flush=True)
    return m


# ---------------------------------------------------------------- metrics
def auroc(feats, pos, neg, n_boot=200, seed=42, cap=1000):
    from sklearn.metrics import roc_auc_score  # lazy: parent-only
    n = min(len(pos), len(neg), cap)
    if n < 10:
        return None
    sp = np.array(pos[:n]); sn = np.array(neg[:n])
    a1 = (feats[sp[:, 0]] * feats[sp[:, 1]]).sum(1)
    a0 = (feats[sn[:, 0]] * feats[sn[:, 1]]).sum(1)
    lbs = np.concatenate([np.ones(n), np.zeros(n)])
    try:
        auc = float(roc_auc_score(lbs, np.concatenate([a1, a0])) * 100)
    except ValueError:
        return None
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        pi = rng.integers(0, n, n); ni = rng.integers(0, n, n)
        try:
            boot.append(roc_auc_score(lbs, np.concatenate([a1[pi], a0[ni]])) * 100)
        except ValueError:
            pass
    lo, hi = np.percentile(boot, [2.5, 97.5]) if boot else (auc, auc)
    return {"auroc": round(auc, 2), "lo": round(float(lo), 2),
            "hi": round(float(hi), 2), "n": int(n)}


def build_pairs_sub(y, a, g, seed=42):
    rng = np.random.default_rng(seed)
    N = len(y)
    cross, diff, hard = [], [], []
    for i in range(N):
        sc = np.where(y == y[i])[0]; sc = sc[sc != i]
        if len(sc) and a[i] >= 0:
            ca = sc[a[sc] != a[i]]
            if len(ca):
                cross.append((i, int(rng.choice(ca))))
        dc = np.where(y != y[i])[0]
        if len(dc):
            diff.append((i, int(rng.choice(dc))))
            if g[i] >= 0:
                hc = dc[g[dc] == g[i]]
                if len(hc):
                    hard.append((i, int(rng.choice(hc))))
    return cross, diff, hard


def build_cm_sub(dino, y, a, g, cap=1000, seed=42):
    rng = np.random.default_rng(seed)
    pos, neg, gaps = [], [], []
    for i in rng.permutation(len(y)):
        if len(pos) >= cap:
            break
        pc = np.where(y == y[i])[0]; pc = pc[pc != i]
        if a[i] >= 0:
            pc = pc[a[pc] != a[i]]
        if len(pc) == 0:
            continue
        nc = np.where((y != y[i]) & (g == g[i]))[0]
        if len(nc) < 2:
            nc = np.where(y != y[i])[0]
        if len(nc) < 2:
            continue
        if len(pc) > 16: pc = rng.choice(pc, 16, replace=False)
        if len(nc) > 64: nc = rng.choice(nc, 64, replace=False)
        sp = dino[pc] @ dino[i]; sn = dino[nc] @ dino[i]
        gap = np.abs(sp[:, None] - sn[None, :])
        pi, ni = np.unravel_index(np.argmin(gap), gap.shape)
        pos.append((int(i), int(pc[pi]))); neg.append((int(i), int(nc[ni])))
        gaps.append(float(gap[pi, ni]))
    return pos, neg, np.array(gaps)


def knn_map(feats, styles, ks=(1, 5, 10), n_q=200, seed=42):
    N = len(feats)
    sim = feats @ feats.T
    np.fill_diagonal(sim, -np.inf)
    order = np.argsort(-sim, axis=1)
    st = np.asarray(styles)
    knn = {f"knn{k}": round(float((st[order[:, :k]] == st[:, None]).mean()) * 100, 2)
           for k in ks}
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N); q, gg = idx[:n_q], idx[n_q:]
    gs = st[gg]; aps = []
    for qi in q:
        sims = feats[gg] @ feats[qi]
        top = np.argsort(-sims)[:10]
        rel = (gs[top] == st[qi]).astype(float)
        if rel.sum() == 0:
            aps.append(0.0); continue
        prec = np.cumsum(rel) / (np.arange(len(rel)) + 1)
        aps.append(float((prec * rel).sum() / rel.sum()))
    return knn, round(float(np.mean(aps) * 100), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="A0,ckpt_u00200,ckpt_u00500,ckpt_u01000")
    ap.add_argument("--model-paths", default="",
                    help="override: comma list name=/abs/path/to/ckpt.pt")
    ap.add_argument("--ckpt-dir", default=str(OUT / "checkpoints"))
    ap.add_argument("--n-images", type=int, default=2400)
    ap.add_argument("--shards", type=int, default=32)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--self-test", type=int, default=0)
    ap.add_argument("--out", default=str(OUT / "cpu_eval_metrics.json"))
    args = ap.parse_args()

    torch.set_num_threads(1)
    proj = torch.load(OUT / "projection.pt", weights_only=False)
    P = proj["P"].float()
    proto = torch.load(OUT / "prototypes.pt", weights_only=False).float()
    m_default = F.linear(proj["mu_z"].float(), P).squeeze(0)
    ev = torch.load(OUT / "eval_pairs.pt", weights_only=False)

    n = min(args.n_images, len(ev["raw"]))
    if args.self_test:
        n = min(args.self_test, n)
    raws = ev["raw"][:n]
    y, a, g = ev["y"][:n], ev["a"][:n], ev["g"][:n]

    # Rebuild pairs ON THE SUBSET (truncating the 2400-image pair list would make
    # the number of surviving pairs decay quadratically and blow up the CI).
    dino = ev["dino"][:n]
    dino = dino / (np.linalg.norm(dino, axis=1, keepdims=True) + 1e-8)
    cm_pos, cm_neg, cm_gap = build_cm_sub(dino, y, a, g, cap=1000)
    cross, diff, hard = build_pairs_sub(y, a, g)
    print(f"[EvalSet] {n} images | cm={len(cm_pos)} cross={len(cross)} hard={len(hard)} "
          f"| cm_gap_mean={cm_gap.mean():.4f} frac<0.05={float((cm_gap<0.05).mean()):.2f}",
          flush=True)

    man = json.loads(MANIFEST.read_text())
    _G["reader"] = np.memmap(man["bin_path"], dtype=np.uint8, mode="r",
                             shape=tuple(man["shape"]))
    _G["raws"] = raws

    # EPYC 7542: 2 sockets x 32 physical cores = 64 physical (logical 0..63),
    # SMT siblings are 64..127. Use ONLY physical cores, never SMT siblings,
    # and keep each worker inside one socket (contiguous physical core range).
    ncpu = os.cpu_count() or 64
    nphys = 64
    nsh = min(args.shards, max(1, n // args.batch))
    cores_per = max(1, args.threads)
    core_sets = []
    for s in range(nsh):
        lo = (s * cores_per) % nphys
        hi = min(nphys, lo + cores_per)
        core_sets.append(set(range(lo, hi)) if hi > lo else None)
    print(f"[CPU] {ncpu} logical / {nphys} physical cores | {nsh} workers x "
          f"{args.threads} threads | pinned PHYSICAL {cores_per} cores/worker "
          f"| fork+COW | batch={args.batch}", flush=True)

    results = {}
    path_map = {}
    if args.model_paths:
        for kv in args.model_paths.split(","):
            if "=" in kv:
                nm, pth = kv.split("=", 1)
                path_map[nm.strip()] = Path(pth.strip())
        print(f"[ModelPaths] {path_map}", flush=True)
    ctx = mp.get_context("fork")
    for tag in args.models.split(","):
        tag = tag.strip()
        if not tag:
            continue
        t0 = time.time()
        if tag == "A0":
            ck, m_c = None, m_default.clone()
        else:
            p = path_map.get(tag) or (Path(args.ckpt_dir) / f"{tag}.pt")
            if not p.exists():
                print(f"[{tag}] missing {p}, skip", flush=True)
                continue
            ck = torch.load(p, map_location="cpu", weights_only=False)
            m_c = ck.get("m_center", m_default).float()
            if m_c.dim() == 2:
                m_c = m_c.squeeze(0)

        # Parent builds once; fork gives every worker the module via COW.
        _G["enc"] = build_encoder(tag, ck)

        bounds = np.linspace(0, n, nsh + 1).astype(int)
        tasks = [(tag, s, int(bounds[s]), int(bounds[s + 1]),
                  args.batch, args.threads, core_sets[s]) for s in range(nsh)
                 if bounds[s + 1] > bounds[s]]
        t1 = time.time()
        chunks = {}
        with cf.ProcessPoolExecutor(max_workers=nsh, mp_context=ctx) as ex:
            for shard, arr in ex.map(_worker, tasks):
                if isinstance(arr, str):
                    print(f"[{tag}] shard {shard}: {arr}", flush=True)
                else:
                    chunks[shard] = arr
        if len(chunks) != len(tasks):
            print(f"[{tag}] {len(chunks)}/{len(tasks)} shards ok, skip", flush=True)
            continue
        Z = np.concatenate([chunks[s] for s in sorted(chunks)]).astype(np.float32)
        dt = time.time() - t1
        print(f"[{tag}] encoded {len(Z)} imgs in {dt:.0f}s ({len(Z)/dt:.2f} img/s)", flush=True)

        if args.self_test:
            results[tag] = {"encoded": int(len(Z)), "sec": round(dt, 1),
                            "img_per_sec": round(len(Z) / dt, 2),
                            "n_workers": nsh, "threads": args.threads}
            continue

        h = F.linear(torch.from_numpy(Z), P)
        hc = h - m_c.unsqueeze(0)
        u = F.normalize(hc, dim=-1, eps=1e-6).numpy()
        r_cm = auroc(u, cm_pos, cm_neg)
        r_cross = auroc(u, cross, diff)
        r_hard = auroc(u, cross, hard)
        knn, mpv = knn_map(u, y)

        # ---- validation-set Rank margin, CE accuracy, common-mode ratio, variance ----
        sp = np.array([u[a] @ u[b] for a, b in cm_pos])
        sn = np.array([u[a] @ u[b] for a, b in cm_neg])
        ce_logits = hc.numpy() @ proto.numpy().T / 0.15
        ce_acc = float((ce_logits.argmax(1) == y).mean() * 100)
        h_mean_norm = float(h.mean(0).norm())
        h_norm_mean = float(h.norm(dim=1).mean())
        c_ratio = h_mean_norm / max(h_norm_mean, 1e-12)
        z_std = float(Z.std(axis=0).mean())

        results[tag] = {
            "n": int(n), "m_center_norm": round(float(m_c.norm()), 3),
            "content_matched": r_cm, "cross_artist": r_cross,
            "same_genre_hard": r_hard, "knn": knn, "map10": mpv,
            "val_rank_margin": round(float(sp.mean() - sn.mean()), 5),
            "val_sim_pos": round(float(sp.mean()), 5),
            "val_sim_neg": round(float(sn.mean()), 5),
            "val_ce_acc_pct": round(ce_acc, 2),
            "common_mode_ratio": round(c_ratio, 4),
            "latent_std_mean": round(z_std, 4),
            "encode_sec": round(dt, 1), "img_per_sec": round(len(Z) / dt, 2),
            "total_sec": round(time.time() - t0, 1),
        }
        print(f"[{tag}] CM={r_cm['auroc'] if r_cm else None} "
              f"[{r_cm['lo'] if r_cm else ''},{r_cm['hi'] if r_cm else ''}] "
              f"Cross={r_cross['auroc'] if r_cross else None} "
              f"Hard={r_hard['auroc'] if r_hard else None} "
              f"kNN1={knn['knn1']} mAP={mpv} | "
              f"valMarg={sp.mean()-sn.mean():+.4f} CEacc={ce_acc:.1f}% "
              f"c_rat={c_ratio:.3f} z_std={z_std:.3f}", flush=True)
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")

    if not args.self_test:
        print("\n" + "=" * 92)
        print("CPU PARALLEL EVAL SUMMARY (Content-Matched AUROC is primary)")
        print("=" * 92)
        print(f"{'model':<18}{'CM':>8}{'95% CI':>16}{'Cross':>8}{'Hard':>8}{'kNN@1':>8}{'mAP':>8}")
        print("-" * 92)
        for k, v in results.items():
            cm = v["content_matched"]
            ci = "[{},{}]".format(cm["lo"], cm["hi"]) if cm else ""
            print(f"{k:<18}{cm['auroc']:>8.2f}{ci:>16}"
                  f"{v['cross_artist']['auroc']:>8.2f}"
                  f"{v['same_genre_hard']['auroc']:>8.2f}"
                  f"{v['knn']['knn1']:>8.2f}{v['map10']:>8.2f}")
        print("=" * 92)
    print(f"[Saved] {args.out} | self_test={args.self_test}", flush=True)


if __name__ == "__main__":
    main()
