"""Self-tests for tools.local_reader. Run BEFORE the real experiment."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.local_reader.readers import D_OUT, build, n_params
from tools.local_reader.run import (auroc_scores, cluster_boot_idx,
                                    paired_cluster_ci)


def _ok(cond, label):
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}", flush=True)
    if not cond:
        raise AssertionError(label)


def test_readers():
    print("\n=== 1. readers ===", flush=True)
    z = torch.randn(8, 4, 64, 64)
    for name in ("local", "cnn"):
        m = build(name)
        y = m(z)
        _ok(tuple(y.shape) == (8, D_OUT), f"{name} output (8,{D_OUT})")
        _ok(float((y.norm(dim=1) - 1).abs().max()) < 1e-5, f"{name} row-L2 normalised")
        n = n_params(m)
        _ok(10_000 <= n <= 100_000, f"{name} params={n:,} in [10K,100K]")
        y.pow(2).mean().backward()
        gs = [p.grad for p in m.parameters() if p.grad is not None]
        _ok(all(torch.isfinite(g).all() for g in gs), f"{name} gradients finite")
    # permutation invariance of the local reader: shuffling patches must not
    # change the pooled representation (it is a Deep Sets mean)
    m = build("local").eval()
    U = torch.nn.functional.unfold(z[:2], 3).transpose(1, 2)
    h = m.norm(m.phi(U))
    with torch.no_grad():
        a = m(z[:2])
    perm = torch.randperm(U.shape[1])
    Up = U[:, perm]
    hp = m.norm(m.phi(Up))
    with torch.no_grad():
        out_perm = torch.nn.functional.normalize(m.rho(hp.mean(1)), dim=1)
    _ok(float((a - out_perm).abs().max()) < 1e-5,
        "local reader is permutation-invariant over patches")


def test_bootstrap():
    print("\n=== 2. paired cluster bootstrap ===", flush=True)
    clusters = np.array(["a"] * 4 + ["b"] * 4 + ["c"] * 4)
    idx = next(cluster_boot_idx(clusters, 1, 0))
    _ok(len(idx) == len(clusters), "resample has exactly n rows (multiplicity kept)")
    # duplicates preserved: a cluster drawn twice contributes twice
    a = np.ones(12); b = np.zeros(12)
    r = paired_cluster_ci(a, b, clusters, n_boot=500, seed=0)
    _ok(abs(r["mean"] - 1.0) < 1e-9, "paired mean correct")
    _ok(r["ci"][0] > 0 and r["excludes_zero"], "CI excludes zero for constant gain")
    # known AUROC: pos all greater than neg -> 1.0; identical -> 0.5
    _ok(abs(auroc_scores(np.array([2., 3.]), np.array([0., 1.])) - 1.0) < 1e-9,
        "AUROC=1 for perfectly separated scores")
    _ok(abs(auroc_scores(np.array([1., 1.]), np.array([1., 1.])) - 0.5) < 1e-9,
        "AUROC=0.5 for tied scores")


def test_data():
    print("\n=== 3. frozen-latent data layer ===", flush=True)
    from tools.local_reader.data import FrozenLatents, SupConBatchSampler, batch_meta
    d = FrozenLatents().describe()
    p = d.pool
    _ok(d.z_train.shape == (78661, 16384), "train latent shape")
    _ok(d.z_hold.shape == (2048, 16384), "holdout latent shape")
    _ok(d.dino.shape == (78661, 768), "DINO shape")
    _ok(d.neg.shape == (78661, 64), "negative index shape")
    # painter-disjoint: no allowed training image belongs to a held-out painter
    leak = int(np.isin(p.artist[d.train_allowed], d.held_painters).sum())
    _ok(leak == 0, f"no held-out painter in training pool (leak={leak})")
    _ok(int(d.hold_painter_disjoint.sum()) >= 500,
        f"strict val subset has {int(d.hold_painter_disjoint.sum())} imgs")
    # sampler: logical batch, unseen schools never form positives
    s = SupConBatchSampler(d, seed=0, batches_per_epoch=3)
    b = next(iter(s))
    _ok(len(b) == 64, f"logical batch = {len(b)}")
    meta = batch_meta(d, b)
    from tools.art_vae.losses import supcon_relation_weights
    W = supcon_relation_weights(meta["style"], meta["artist"], meta["view"],
                                supervised=meta["supervised"])
    sup = meta["supervised"].numpy()
    unsup = (~sup).astype(float)
    unsup_pos = float((W * (unsup[:, None] * unsup[None, :])).sum())
    _ok(unsup_pos == 0, f"unseen-school positive weight = {unsup_pos}")
    z = d.latent(meta["idx"])
    _ok(tuple(z.shape) == (64, 4, 64, 64) and bool(torch.isfinite(z).all()),
        "cached latent batch loads and is finite")


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[selftest] device={dev}", flush=True)
    test_readers()
    test_bootstrap()
    test_data()
    print("\n[selftest] ALL TESTS PASSED", flush=True)


if __name__ == "__main__":
    main()
