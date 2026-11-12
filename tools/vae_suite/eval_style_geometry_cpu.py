"""CPU+RAM-only style geometry evaluation (no GPU usage, safe to run alongside training).

For each checkpoint, encodes the held-out eval images on CPU, projects with the
frozen P and the checkpoint's saved EMA center m_t, and reports:
    Content-Matched AUROC (primary), Cross-Artist AUROC, Same-Genre Hard AUROC,
    kNN@1/5/10, mAP@10, plus center/variance diagnostics.

Only the encoder is loaded (decoder stripped) -> smallest RAM & fastest.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/root/Workspace/WEAVE_VAE")

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from sklearn.metrics import roc_auc_score

OUT = Path("/root/Workspace/WEAVE_VAE/runs/vae_style_rank20")
MANIFEST = Path("/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")
A0_DIR = "/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae"


def auroc(feats, pos, neg, n_boot=200, seed=42, cap=1000):
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


class ImageReader:
    def __init__(self, raw_list):
        man = json.loads(MANIFEST.read_text())
        self.mmap = np.memmap(man["bin_path"], dtype=np.uint8, mode="r",
                              shape=tuple(man["shape"]))
        self.raw = raw_list

    def get(self, idxs):
        arr = np.stack([np.ascontiguousarray(self.mmap[self.raw[i]]) for i in idxs])
        x = torch.from_numpy(arr).permute(0, 3, 1, 2).float()
        return x.div_(127.5).sub_(1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--n-images", type=int, default=800)
    ap.add_argument("--models", default="A0,ckpt_u00200,ckpt_u00500,ckpt_u01000")
    ap.add_argument("--out", default=str(OUT / "cpu_eval_metrics.json"))
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(2)
    print(f"[CPU] threads={args.threads} batch={args.batch} n_images={args.n_images}", flush=True)

    proj = torch.load(OUT / "projection.pt", weights_only=False)
    P = proj["P"].float()                     # (768, 16384)
    mu_z = proj["mu_z"].float()
    m_default = F.linear(mu_z, P).squeeze(0)

    ev = torch.load(OUT / "eval_pairs.pt", weights_only=False)
    n_all = len(ev["raw"])
    n = min(args.n_images, n_all)
    reader = ImageReader(ev["raw"][:n])
    y, a, g = ev["y"][:n], ev["a"][:n], ev["g"][:n]

    def sub(pairs):
        return [(i, j) for i, j in pairs if i < n and j < n]

    cm_pos, cm_neg = sub(ev["cm_pos"]), sub(ev["cm_neg"])
    cross, diff, hard = sub(ev["cross"]), sub(ev["diff"]), sub(ev["hard"])
    print(f"[EvalSet] {n}/{n_all} images | cm={len(cm_pos)} cross={len(cross)} "
          f"hard={len(hard)}", flush=True)

    results = {}
    for tag in args.models.split(","):
        tag = tag.strip()
        if not tag:
            continue
        t0 = time.time()
        if tag == "A0":
            ck = None
            path = A0_DIR
            m_c = m_default.clone()
        else:
            path = OUT / "checkpoints" / f"{tag}.pt"
            if not path.exists():
                print(f"[{tag}] missing {path}, skip", flush=True)
                continue
            ck = torch.load(path, map_location="cpu", weights_only=False)
            m_c = ck.get("m_center", m_default).float()
            if m_c.dim() == 2:
                m_c = m_c.squeeze(0)

        print(f"\n--- {tag} (m_center norm={m_c.norm().item():.2f}) ---", flush=True)
        vae = AutoencoderKL.from_pretrained(A0_DIR, torch_dtype=torch.float32)
        if ck is not None:
            vae.load_state_dict(ck["model"])
        vae.eval()
        del vae.decoder
        del vae.post_quant_conv

        us = []
        zs_rms = []
        with torch.no_grad():
            for i in range(0, n, args.batch):
                x = reader.get(list(range(i, min(i + args.batch, n))))
                h = vae.encoder(x)
                h = vae.quant_conv(h)
                mu = h[:, :4].float().flatten(1)
                zs_rms.append(float(mu.pow(2).mean().sqrt()))
                hp = F.linear(mu, P)
                us.append(F.normalize(hp - m_c.unsqueeze(0), dim=-1, eps=1e-6).numpy())
                if (i // args.batch + 1) % 50 == 0:
                    el = time.time() - t0
                    done = i + args.batch
                    print(f"  encoded {done}/{n} in {el:.0f}s "
                          f"(eta {el/done*(n-done)/60:.1f} min)", flush=True)
        u = np.concatenate(us)
        del vae

        r_cm = auroc(u, cm_pos, cm_neg)
        r_cross = auroc(u, cross, diff)
        r_hard = auroc(u, cross, hard)
        knn, mp = knn_map(u, y)
        c_norm = float(m_c.norm())
        h_norm = float(np.mean(zs_rms))
        res = {
            "n": int(n),
            "m_center_norm": round(c_norm, 3),
            "latent_rms": round(h_norm, 4),
            "content_matched": r_cm, "cross_artist": r_cross,
            "same_genre_hard": r_hard, "knn": knn, "map10": mp,
            "elapsed_sec": round(time.time() - t0, 1),
        }
        results[tag] = res
        print(f"[{tag}] CM={r_cm['auroc'] if r_cm else None} "
              f"[{r_cm['lo'] if r_cm else ''},{r_cm['hi'] if r_cm else ''}] "
              f"Cross={r_cross['auroc'] if r_cross else None} "
              f"Hard={r_hard['auroc'] if r_hard else None} "
              f"kNN1={knn['knn1']} mAP={mp} | {res['elapsed_sec']}s", flush=True)

        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n" + "=" * 92)
    print("CPU EVAL SUMMARY (Content-Matched AUROC is primary)")
    print("=" * 92)
    print(f"{'model':<18}{'CM':>8}{'95% CI':>16}{'Cross':>8}{'Hard':>8}{'kNN@1':>8}{'mAP':>8}")
    print("-" * 92)
    for k, v in results.items():
        cm = v["content_matched"]
        ci = "[{},{}]".format(cm["lo"], cm["hi"]) if cm else ""
        print(f"{k:<18}{cm['auroc']:>8.2f}{ci:>16}"
              f"{v['cross_artist']['auroc']:>8.2f}{v['same_genre_hard']['auroc']:>8.2f}"
              f"{v['knn']['knn1']:>8.2f}{v['map10']:>8.2f}")
    print("=" * 92)
    print(f"[Saved] {args.out}")


if __name__ == "__main__":
    main()
