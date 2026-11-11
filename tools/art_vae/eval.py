"""Acceptance evaluation for the Art-VAE encoder.

    python -m tools.art_vae.eval --ckpt runs/art_vae/B_best.pt --tag B

Five reported blocks (all on the FROZEN 2,048-image validation split, never
used for training):

  1. Cross-Artist Content-Matched AUROC  -- exclude painter/content shortcuts.
     Positives: same style, DIFFERENT painter.
     Negatives: different style, DINO-content-matched hard negatives.
  2. Few-shot linear probe / retrieval   -- r(z) vs the full flattened latent.
     A simple readout must be at least as learnable as a complex one.
  3. Unseen-school retrieval            -- schools held out of style supervision.
  4. Reconstruction                     -- PSNR / LPIPS / SSIM / HF-band error.
  5. Posterior probe                    -- freeze the VAE, train a small linear
     style operator for the same budget on A0 vs the new encoder, compare the
     learning curve. Necessary (not sufficient) evidence of editability.

All style judgements use the frozen latent readout r(z); CSD/DINO enter only as
EVALUATION metrics, never as training gradients.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import lpips
from diffusers import AutoencoderKL
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "12h"))

from tools.art_vae.data import (DINO_MEAN, DINO_STD, MANIFEST, OUT, TrainPool)
from tools.art_vae.style_readout import style_readout, D_OUT

NM = [0.48145466, 0.4578275, 0.40821073]
NS = [0.26862954, 0.26130258, 0.27577711]


def load_variant(name, ckpt_path, dev):
    from tools.vae_suite.decoder_prune import build_decoder, StudentDecoder
    a0 = AutoencoderKL.from_pretrained(
        str(ROOT / "runs_vae" / "A0" / "final_diffusers_vae"),
        torch_dtype=torch.float32).to(dev).eval()
    for p in a0.parameters():
        p.requires_grad = False
    if name == "a0":
        return a0, lambda x: a0.encode(x).latent_dist.mean, \
            lambda z: a0.decode(z).sample
    if name == "s10":
        ck = torch.load(ROOT / "runs/vae_decoder_prune/s10_p5_lowlr/decoder.pt",
                        map_location="cpu", weights_only=False)
        d = build_decoder(ck["meta"]["widths_forward"])
        d.load_state_dict(ck["state_dict"])
        sd = StudentDecoder(a0.post_quant_conv, d.to(dev)).to(dev).eval()
        return a0, lambda x: a0.encode(x).latent_dist.mean, lambda z: sd(z)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a0.encoder.load_state_dict(ck["encoder"])
    a0.quant_conv.load_state_dict(ck["quant_conv"])
    d = build_decoder([64, 128, 192, 256])
    d.load_state_dict(ck["decoder"])
    sd = StudentDecoder(a0.post_quant_conv, d.to(dev)).to(dev).eval()
    return a0, lambda x: DiagonalGaussianDistribution(
        a0.quant_conv(a0.encoder(x))).mean, lambda z: sd(z)


def _img(pool, raw, view=0):
    from PIL import Image
    arr = np.ascontiguousarray(pool.mmap[int(raw)])
    im = Image.fromarray(arr)
    if view:
        im = im.crop((32, 32, 480, 480)).resize((512, 512), Image.BICUBIC)
    return torch.from_numpy(np.asarray(im, np.uint8)).permute(2, 0, 1).float().div_(127.5).sub_(1.0)


@torch.no_grad()
def val_pool_records(pool):
    val = set(pool.manifest["val_indices"])
    return [r for r in pool.manifest["records"] if r["raw_index"] in val]


@torch.no_grad()
def encode_all(enc, recs, pool, dev, bs=8):
    Z, R = [], []
    for i in range(0, len(recs), bs):
        x = torch.stack([_img(pool, r["raw_index"]) for r in recs[i:i + bs]]).to(dev)
        z = enc(x).float()
        Z.append(z)
        R.append(style_readout(z))
        del x
        torch.cuda.empty_cache()
    return torch.cat(Z), torch.cat(R)


@torch.no_grad()
def dino_features(recs, pool, dev, bs=32):
    from transformers import AutoModel
    m = AutoModel.from_pretrained("facebook/dinov2-base",
                                  local_files_only=True).to(dev).eval()
    nm = torch.tensor(DINO_MEAN, device=dev).view(1, 3, 1, 1)
    ns = torch.tensor(DINO_STD, device=dev).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(recs), bs):
        x = torch.stack([_img(pool, r["raw_index"]) for r in recs[i:i + bs]]).to(dev)
        x01 = (x.clamp(-1, 1) + 1) / 2
        x224 = F.interpolate(x01, (224, 224), mode="bicubic", align_corners=False)
        f = F.normalize(m(pixel_values=(x224 - nm) / ns).last_hidden_state[:, 0].float(), dim=-1)
        out.append(f)
        del x, x01, x224
        torch.cuda.empty_cache()
    return torch.cat(out)


def auroc_from_scores(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # rank-based AUROC, O(n log n), avoids the NxM bool matrix
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv)
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    avg = sums / cnt
    ranks = avg[inv]
    n1, n0 = len(pos), len(neg)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def eval_auroc(R, style, artist, dino, hard_k=16, seed=0):
    """Cross-artist positives vs content-matched hard negatives."""
    rng = np.random.default_rng(seed)
    sim_z = (R @ R.t()).cpu().numpy()
    sim_d = (dino @ dino.t()).cpu().numpy()
    n = len(style)
    pos_scores, neg_scores = [], []
    for i in range(n):
        same_style_diff_artist = np.where(
            (style == style[i]) & (artist != artist[i]))[0]
        diff_style = np.where(style != style[i])[0]
        if len(same_style_diff_artist) == 0 or len(diff_style) == 0:
            continue
        # hard negatives: the most content-similar DIFFERENT-style images
        cand = diff_style[np.argsort(-sim_d[i, diff_style])[:hard_k]]
        for j in same_style_diff_artist:
            pos_scores.append(sim_z[i, j])
        for j in cand:
            neg_scores.append(sim_z[i, j])
    return auroc_from_scores(np.array(pos_scores), np.array(neg_scores)), \
        len(pos_scores), len(neg_scores)


def retrieval_acc(R, style, train_idx, test_idx, k=1):
    """Nearest-centroid / 1-NN retrieval accuracy with few-shot prototypes."""
    by = defaultdict(list)
    for i in train_idx:
        by[style[i]].append(i)
    if len(by) < 2:
        return float("nan")
    proto = {s: R[v].mean(0) for s, v in by.items()}
    proto = {s: F.normalize(p, dim=0) for s, p in proto.items()}
    keys = list(proto)
    P = torch.stack([proto[s] for s in keys])
    correct = tot = 0
    for i in test_idx:
        if style[i] not in proto:
            continue
        pred = keys[int((P @ R[i]).argmax())]
        correct += int(pred == style[i]); tot += 1
    return correct / max(tot, 1)


def few_shot_curve(R, style, n_shots=(1, 2, 5, 10, 20), seed=0, steps=200):
    """Learning curve of a LINEAR style operator on the frozen readout."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(style)
    idx_of = {s: np.where(style == s)[0] for s in uniq}
    out = {}
    for k in n_shots:
        tr, te = [], []
        for s in uniq:
            v = rng.permutation(idx_of[s])
            tr += v[:k].tolist(); te += v[k:k + 20].tolist()
        if len(tr) < 2 or len(te) == 0:
            continue
        X = R[tr]; y = torch.tensor([style[i] for i in tr], device=R.device)
        ymap = {s: j for j, s in enumerate(uniq)}
        yj = torch.tensor([ymap[style[i]] for i in tr], device=R.device)
        tej = np.array([ymap[style[i]] for i in te])
        clf = torch.nn.Linear(R.shape[1], len(uniq)).to(R.device)
        opt = torch.optim.AdamW(clf.parameters(), lr=1e-2, weight_decay=1e-4)
        curve = []
        for st in range(steps):
            loss = F.cross_entropy(clf(X), yj)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            if (st + 1) % 20 == 0:
                with torch.no_grad():
                    acc = float((clf(R[te]).argmax(1).cpu().numpy() == tej).mean())
                curve.append(round(acc, 4))
        out[str(k)] = {"final": round(float(curve[-1]), 4) if curve else None,
                       "curve": curve}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--tag", default="new")
    ap.add_argument("--baselines", default="a0,s10")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="cap val images (smoke)")
    args = ap.parse_args()
    dev = args.device
    pool = TrainPool().describe()
    recs = val_pool_records(pool)
    if args.limit:
        recs = recs[:args.limit]
    style = np.array([r["style_id"] for r in recs], np.int64)
    artist = np.array([r["artist_id"] for r in recs], np.int64)
    print(f"[Eval] val images={len(recs)} styles={len(set(style.tolist()))}", flush=True)

    print("[Eval] computing DINO content features for hard negatives...", flush=True)
    dino = dino_features(recs, pool, dev)
    print(f"  dino {tuple(dino.shape)}", flush=True)

    variants = {}
    for b in args.baselines.split(","):
        variants[b] = load_variant(b.strip(), "", dev)
    if args.ckpt:
        variants[args.tag] = load_variant("ckpt", args.ckpt, dev)

    results = {}
    for name, (_, enc, dec) in variants.items():
        print(f"\n=== variant {name} ===", flush=True)
        Z, R = encode_all(enc, recs, pool, dev)
        auroc, np_, nn_ = eval_auroc(R, style, artist, dino)
        print(f"  cross-artist content-matched AUROC = {auroc:.4f} "
              f"(pos pairs {np_}, hard neg {nn_})", flush=True)

        rng = np.random.default_rng(0)
        uniq = np.unique(style)
        tr, te = [], []
        for s in uniq:
            v = rng.permutation(np.where(style == s)[0])
            tr += v[:5].tolist(); te += v[5:15].tolist()
        ret = retrieval_acc(R, style, tr, te)
        ret_flat = retrieval_acc(Z.flatten(1) /
                                 (Z.flatten(1).norm(dim=1, keepdim=True) + 1e-8),
                                 style, tr, te)
        print(f"  5-shot retrieval: r(z)={ret*100:.1f}%  "
              f"flat-latent={ret_flat*100:.1f}%", flush=True)

        unseen = np.array([s for s in pool.unseen_styles if s in set(style.tolist())])
        mask = np.isin(style, unseen)
        if mask.sum() > 20:
            ui = np.where(mask)[0]
            rng.shuffle(ui)
            cut = max(2, len(ui) // 2)
            uret = retrieval_acc(R, style, ui[:cut], ui[cut:])
            print(f"  UNSEEN-school retrieval ({len(unseen)} schools, n={mask.sum()}) "
                  f"= {uret*100:.1f}%", flush=True)
        else:
            uret = float("nan")
            print(f"  UNSEEN-school retrieval: n/a (only {mask.sum()} val images)", flush=True)

        curve = few_shot_curve(R, style, n_shots=(1, 5, 10), steps=120)
        print(f"  linear-probe final acc: "
              + ", ".join(f"{k}-shot={v['final']}" for k, v in curve.items()), flush=True)

        # ---- reconstruction on a subset ----
        lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
        for p in lp.parameters():
            p.requires_grad = False
        import csd_model
        csd, _, _ = csd_model.load_csd(dev); csd = csd.float().eval()
        nm = torch.tensor(NM, device=dev).view(1, 3, 1, 1)
        ns = torch.tensor(NS, device=dev).view(1, 3, 1, 1)
        sub = recs[:128]
        l1s, lps, sss, pss, hfs, csds = [], [], [], [], [], []
        with torch.no_grad():
            for i in range(0, len(sub), 4):
                x = torch.stack([_img(pool, r["raw_index"]) for r in sub[i:i + 4]]).to(dev)
                y = dec(enc(x)).float()
                l1s.append(float((y - x).abs().mean()))
                lps.append(float(lp(y.clamp(-1, 1), x.clamp(-1, 1)).mean()))
                mse = float(F.mse_loss((y.clamp(-1, 1) + 1) / 2, (x + 1) / 2))
                pss.append(10 * np.log10(1 / max(mse, 1e-10)))
                x01 = (x + 1) / 2; y01 = (y.clamp(-1, 1) + 1) / 2
                mx = F.avg_pool2d(x01, 11, 1, 5); my = F.avg_pool2d(y01, 11, 1, 5)
                mxx, myy, mxy = mx * mx, my * my, mx * my
                sx = F.avg_pool2d(x01 * x01, 11, 1, 5) - mxx
                sy = F.avg_pool2d(y01 * y01, 11, 1, 5) - myy
                sxy = F.avg_pool2d(x01 * y01, 11, 1, 5) - mxy
                sss.append(float((((2 * mxy + 1e-4) * (2 * sxy + 9e-4)) /
                                  ((mxx + myy + 1e-4) * (sx + sy + 9e-4))).mean()))
                a, b = x01[:, :, 0::2, 0::2], x01[:, :, 1::2, 0::2]
                c, d = x01[:, :, 0::2, 1::2], x01[:, :, 1::2, 1::2]
                ay, by = y01[:, :, 0::2, 0::2], y01[:, :, 1::2, 0::2]
                cy, dy = y01[:, :, 0::2, 1::2], y01[:, :, 1::2, 1::2]
                hx, hy = (a - b + c - d) * 0.5, (ay - by + cy - dy) * 0.5
                hfs.append(float((hx - hy).abs().mean()))
                f = []
                for t in (y, x):
                    t01 = (t.clamp(-1, 1) + 1) / 2
                    t224 = F.interpolate(t01, (224, 224), mode="bicubic", align_corners=False)
                    _, _, s = csd((t224 - nm) / ns)
                    f.append(F.normalize(s.float(), dim=-1))
                csds.append(float((f[0] * f[1]).sum(-1).mean()))
                del x, y
                torch.cuda.empty_cache()
        rec = {"l1": round(float(np.mean(l1s)), 5),
               "lpips": round(float(np.mean(lps)), 5),
               "psnr": round(float(np.mean(pss)), 2),
               "ssim": round(float(np.mean(sss)), 4),
               "hf_l1": round(float(np.mean(hfs)), 5),
               "csd_keep": round(float(np.mean(csds)), 5)}
        print(f"  recon: {json.dumps(rec)}", flush=True)

        results[name] = {"auroc": round(auroc, 4), "n_pos_pairs": np_,
                         "n_hard_neg": nn_, "retrieval_5shot": round(ret, 4),
                         "retrieval_flat_latent": round(ret_flat, 4),
                         "unseen_retrieval": None if np.isnan(uret) else round(uret, 4),
                         "few_shot_linear_probe": curve, "recon": rec}

    outp = OUT / f"eval_{args.tag}.json"
    outp.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + "=" * 96)
    print("ACCEPTANCE SUMMARY")
    print("=" * 96)
    print(f"{'variant':<10}{'AUROC':>9}{'retr':>8}{'unseen':>9}"
          f"{'LPIPS':>9}{'PSNR':>8}{'SSIM':>8}{'CSD':>8}")
    for n, r in results.items():
        u = r["unseen_retrieval"]
        print(f"{n:<10}{r['auroc']:>9.4f}{r['retrieval_5shot']*100:>7.1f}%"
              f"{(u*100 if u is not None else float('nan')):>8.1f}%"
              f"{r['recon']['lpips']:>9.5f}{r['recon']['psnr']:>8.2f}"
              f"{r['recon']['ssim']:>8.4f}{r['recon']['csd_keep']:>8.4f}")
    print(f"\n[Saved] {outp}", flush=True)


if __name__ == "__main__":
    main()
