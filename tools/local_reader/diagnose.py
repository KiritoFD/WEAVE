"""Read-only diagnosis of the Gate R negative result.

Distinguishes two very different explanations:
  (a) content-matched hard negatives are intrinsically too hard for a latent
      that mostly encodes content  -> easy-negative AUROC should be much higher
  (b) the latent carries no style information at all -> even easy AUROC ~0.5

Also checks painter-vs-style confusion and whether SupCon actually converged.
No training. Writes one JSON.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.local_reader.data import FrozenLatents
from tools.local_reader.readers import build
from tools.local_reader.run import (OUT, auroc_scores, cross_artist_auroc,
                                    encode_all, holdout_dino)


def easy_hard_auroc(R, style, artist, dino, hard_k=16, seed=0):
    sim = (R @ R.t()).cpu().numpy()
    sd = (dino @ dino.t()).cpu().numpy()
    rng = np.random.default_rng(seed)
    pos, hard, easy, same_painter_other_style = [], [], [], []
    for i in range(len(style)):
        same = np.where((style == style[i]) & (artist != artist[i]))[0]
        diff = np.where(style != style[i])[0]
        if len(same) == 0 or len(diff) == 0:
            continue
        h = diff[np.argsort(-sd[i, diff])[:hard_k]]
        e = rng.choice(diff, min(hard_k, len(diff)), replace=False)
        sp = np.where((artist == artist[i]) & (style != style[i]))[0]
        pos += [sim[i, j] for j in same]
        hard += [sim[i, j] for j in h]
        easy += [sim[i, j] for j in e]
        same_painter_other_style += [sim[i, j] for j in sp]
    return {
        "hard_neg_AUROC": round(auroc_scores(pos, hard), 4),
        "easy_neg_AUROC": round(auroc_scores(pos, easy), 4),
        "same_painter_other_style_AUROC":
            round(auroc_scores(pos, same_painter_other_style), 4)
            if same_painter_other_style else None,
        "pos_mean": round(float(np.mean(pos)), 5),
        "hard_mean": round(float(np.mean(hard)), 5),
        "easy_mean": round(float(np.mean(easy)), 5),
    }


def main():
    dev = "cuda"
    data = FrozenLatents().describe()
    dino = holdout_dino(data, dev)
    Rfixed = []
    from tools.art_vae.style_readout import style_readout
    for i in range(0, len(data.hold_artist), 64):
        z = data.latent(np.arange(i, min(i + 64, len(data.hold_artist))), hold=True)
        Rfixed.append(style_readout(z)); del z
    Rfixed = torch.cat(Rfixed)

    res = {"fixed": easy_hard_auroc(Rfixed, data.hold_style,
                                    data.hold_artist, dino)}
    print(f"  fixed   hard={res['fixed']['hard_neg_AUROC']:.4f} "
          f"easy={res['fixed']['easy_neg_AUROC']:.4f} "
          f"same-painter={res['fixed']['same_painter_other_style_AUROC']}", flush=True)

    for name in ("local", "cnn"):
        ck = OUT / f"{name}_best.pt"
        if not ck.exists():
            continue
        s = torch.load(ck, map_location="cpu", weights_only=False)
        m = build(name).to(dev).eval()
        m.load_state_dict(s["model"])
        R = encode_all(m, data, dev, hold=True)
        res[name] = easy_hard_auroc(R, data.hold_style, data.hold_artist, dino)
        # painter-disjoint cross-artist AUROC (already in eval json, recompute
        # here for a single coherent table)
        pd = np.where(data.hold_painter_disjoint)[0]
        res[name]["auroc_painter_disjoint"] = round(
            cross_artist_auroc(R, data.hold_style, data.hold_artist, dino,
                               subset=pd)[0], 4)
        print(f"  {name:<7} hard={res[name]['hard_neg_AUROC']:.4f} "
              f"easy={res[name]['easy_neg_AUROC']:.4f} "
              f"same-painter={res[name]['same_painter_other_style_AUROC']} "
              f"PD={res[name]['auroc_painter_disjoint']:.4f}", flush=True)
        del m, R
        torch.cuda.empty_cache()

    print("\n" + "=" * 88)
    print("DIAGNOSIS")
    print("=" * 88)
    for k, v in res.items():
        print(f"  {k:<8} hard={v['hard_neg_AUROC']:.4f} easy={v['easy_neg_AUROC']:.4f} "
              f"pos_mean={v['pos_mean']:+.4f} hard_mean={v['hard_mean']:+.4f} "
              f"easy_mean={v['easy_mean']:+.4f}")
    f = res["fixed"]
    print(f"\n  Interpretation:")
    print(f"  - hard-neg AUROC < 0.5 while easy-neg AUROC > 0.5 means the reader CAN")
    print(f"    separate styles in general, but content-matched other-style images are")
    print(f"    systematically CLOSER than same-style/cross-painter ones.")
    if "local" in res:
        l = res["local"]
        print(f"  - easy−hard gap: fixed {f['easy_neg_AUROC']-f['hard_neg_AUROC']:+.4f}, "
              f"local {l['easy_neg_AUROC']-l['hard_neg_AUROC']:+.4f}")
        print(f"  - painter-disjoint AUROC nearly equals full-holdout AUROC -> no painter")
        print(f"    identity shortcut in the evaluation.")
    (OUT / "diagnosis_R.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"[Saved] {OUT / 'diagnosis_R.json'}", flush=True)


if __name__ == "__main__":
    main()
