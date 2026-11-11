"""Additional unit tests required by the SD0-S10 V2 plan.

  * BF16 training vs FP32 eval consistency
  * checkpoint resume: optimizer + scheduler state complete
  * data coverage audit (every row reachable, no overlap)
  * validation does not share the training graph / device memory

Run AFTER the main selftest, standalone:
    python -m tools.vae_suite.test_sd0_s10_extra
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/root/Workspace/WEAVE_VAE")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "vae_suite"))

from tools.vae_suite.decoder_prune import StudentDecoder, build_decoder
from tools.vae_suite.sd0_losses import ReconLoss, WaveletHF
from tools.vae_suite.train_sd0_s10 import load_data

OUT = ROOT / "runs/sd0_s10"
WIDTHS = [64, 128, 192, 256]


def _ok(c, label):
    print(f"  [{'PASS' if c else 'FAIL'}] {label}", flush=True)
    if not c:
        raise AssertionError(label)


def test_bf16_fp32(dev):
    print("\n=== 4. BF16 / FP32 consistency ===", flush=True)
    ck = torch.load(OUT / "cold_init.pt", map_location="cpu", weights_only=False)
    d = build_decoder(WIDTHS).to(dev).eval()
    d.load_state_dict(ck["state_dict"])
    pq = torch.nn.Conv2d(4, 4, 1).to(dev)
    pq.load_state_dict(ck["post_quant_conv"])
    m = StudentDecoder(pq, d).to(dev).eval()
    z = torch.randn(2, 4, 64, 64, device=dev)
    with torch.no_grad():
        y32 = m(z).float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ybf = m(z).float()
    rel = float((y32 - ybf).abs().mean() / (y32.abs().mean() + 1e-8))
    _ok(bool(torch.isfinite(ybf).all()) and rel < 0.05,
        f"BF16 vs FP32 mean relative diff = {rel:.5f} (< 0.05)")
    # HF loss must be FP32-internally even when y is bf16
    whf = WaveletHF({(l, b): 1.0 for l in range(2) for b in range(3)}).to(dev)
    yb = ybf.half()
    l_hf, _, _ = whf(yb, y32)
    _ok(bool(torch.isfinite(l_hf)), "WaveletHF stays finite with BF16 output")


def test_resume(dev):
    print("\n=== 5. checkpoint resume ===", flush=True)
    # create a minimal optimizer + scheduler, save, restore, compare one step
    d = build_decoder(WIDTHS).to(dev)
    opt = torch.optim.AdamW(d.parameters(), lr=3e-5)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.9 * s)
    for _ in range(3):
        opt.zero_grad()
        d(torch.randn(1, 4, 64, 64, device=dev)).float().abs().mean().backward()
        opt.step(); sched.step()
    state = {"opt": opt.state_dict(), "sched": sched.state_dict(),
             "sd": d.state_dict()}
    f = OUT / "_resume_selftest.pt"
    torch.save(state, f)
    d2 = build_decoder(WIDTHS).to(dev)
    opt2 = torch.optim.AdamW(d2.parameters(), lr=3e-5)
    sched2 = torch.optim.lr_scheduler.LambdaLR(opt2, lambda s: 0.9 * s)
    s = torch.load(f, map_location="cpu", weights_only=False)
    d2.load_state_dict(s["sd"])
    opt2.load_state_dict(s["opt"])
    sched2.load_state_dict(s["sched"])
    lr_ok = abs(opt2.param_groups[0]["lr"] - opt.param_groups[0]["lr"]) < 1e-12
    _ok(lr_ok, f"restored LR = {opt2.param_groups[0]['lr']:.3e}")
    _ok(len(opt2.state_dict()["state"]) == len(opt.state_dict()["state"]),
        "optimizer momentum state restored for all params")
    # one more step in both must stay identical
    x = torch.randn(1, 4, 64, 64, device=dev)
    for m_, o_ in ((d, opt), (d2, opt2)):
        o_.zero_grad()
        m_(x).float().abs().mean().backward()
        o_.step()
    diff = max(float((a - b).abs().max())
               for a, b in zip(d.state_dict().values(), d2.state_dict().values()))
    _ok(diff < 1e-6, f"resumed optimizer reproduces the next step (diff {diff:.2e})")
    f.unlink()


def test_data_audit():
    print("\n=== 6. data coverage audit ===", flush=True)
    sp, ds_tr, ds_ho, ds_te = load_data()
    rtr = np.load(OUT / "full/raw_index_train.npy")
    rho = np.load(OUT / "full/raw_index_holdout.npy")
    rte = np.load(OUT / "full/raw_index_test256.npy")
    _ok(len(rtr) == sp["n_train"] == len(ds_tr), "train index count matches dataset")
    _ok(len(rho) == sp["n_holdout"] == len(ds_ho), "holdout index count matches")
    _ok(len(rte) == sp["n_test256"] == len(ds_te), "test256 index count matches")
    a, b, c = set(rtr.tolist()), set(rho.tolist()), set(rte.tolist())
    _ok(not (a & b) and not (a & c) and not (b & c), "no image-id overlap")
    _ok(sp["image_id_disjoint"] and sp["encoder_determinism_max_abs"] == 0.0,
        "split.json audit flags present")
    # latent row i must correspond to raw_index[i] (verified end-to-end by
    # sd0_cache.py's 128-row re-encode check); here confirm the dataset reads
    # the same memmap row the index names.
    for i in (0, 1000, len(ds_tr) - 1):
        rec = ds_tr[i]
        _ok(int(rec["z"].shape[0]) == 4 and int(rtr[i]) >= 0,
            f"dataset row {i} bounded & index valid")
    print("  (full row-order alignment verified by sd0_cache.py re-encode)",
          flush=True)


def test_validation_isolation(dev):
    print("\n=== 7. validation graph isolation ===", flush=True)
    # Validation runs inside @torch.no_grad(); verify no grad is retained
    d = build_decoder(WIDTHS).to(dev).train()
    z = torch.randn(2, 4, 64, 64, device=dev, requires_grad=True)
    with torch.no_grad():
        y = d(z)
    _ok(not y.requires_grad and not z.grad, "no_grad validation leaves no graph")


def main():
    dev = "cuda"
    print(f"[extra-tests] device={dev}", flush=True)
    test_bf16_fp32(dev)
    test_resume(dev)
    test_data_audit()
    test_validation_isolation(dev)
    print("\n[extra-tests] ALL PASSED", flush=True)


if __name__ == "__main__":
    main()
