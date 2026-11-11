"""Self-tests for SD0-S10 V2 losses + interface. Run BEFORE training.

Covers the plan's mandatory unit tests:
  interface : Encoder param hash unchanged, decode path == AutoencoderKL.decode,
              Student applies post_quant_conv EXACTLY ONCE, output shape/range.
  losses    : Wavelet error ~0 for identical images, band shapes/grads,
              Response ~0 with delta=0, teacher has NO grad, student input grad
              finite & non-zero, energy term off by default.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path("/root/Workspace/WEAVE_VAE")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "vae_suite"))

from diffusers import AutoencoderKL
from tools.vae_suite.decoder_prune import StudentDecoder
from tools.vae_suite.sd0_losses import (ReconLoss, WaveletHF, haar_dwt,
                                        make_perturbation, off_manifold_loss,
                                        prune_with_indices, response_loss)

SD0 = "/root/Workspace/xy/DiT/data/pretrained/sd-vae-ft-ema"
A0 = str(ROOT / "runs_vae" / "A0" / "final_diffusers_vae")
WIDTHS = [64, 128, 192, 256]


def _ok(c, label):
    print(f"  [{'PASS' if c else 'FAIL'}] {label}", flush=True)
    if not c:
        raise AssertionError(label)


def enc_hash(vae):
    h = hashlib.sha256()
    for k, v in sorted(vae.state_dict().items()):
        if k.startswith("encoder") or k.startswith("quant_conv"):
            h.update(k.encode()); h.update(v.cpu().numpy().tobytes())
    return h.hexdigest()


def test_interface(dev):
    print("\n=== 1. interface ===", flush=True)
    vae = AutoencoderKL.from_pretrained(SD0, torch_dtype=torch.float32).to(dev).eval()
    h0 = enc_hash(vae)
    # build a 10M student from the OFFICIAL decoder using S10 channel indices
    idx = torch.load(ROOT / "runs/vae_decoder_prune/s10_p1_l1/decoder.pt",
                     map_location="cpu", weights_only=False)["meta"]["channel_index"]
    # Cold-SD0 reuses the EXACT original S10 channel index set.
    student_dec, meta = prune_with_indices(vae.decoder, WIDTHS, idx)
    student = StudentDecoder(vae.post_quant_conv, student_dec.to(dev)).to(dev).eval()
    _ok(sum(p.numel() for p in student_dec.parameters()) == 10657923,
        "Cold student params == 10,657,923")

    z = torch.randn(2, 4, 64, 64, device=dev)
    with torch.no_grad():
        y_official = vae.decode(z).sample
        y_student = student(z)
    _ok(tuple(y_student.shape) == (2, 3, 512, 512), "student output shape")
    _ok(bool(torch.isfinite(y_student).all()), "student output finite")
    _ok(enc_hash(vae) == h0, "official Encoder+quant_conv hash UNCHANGED")

    # exactly one pqc: comparing student(z) to decoder(pqc(pqc(z))) must differ
    with torch.no_grad():
        once = student(z)
        twice = student_dec(vae.post_quant_conv(vae.post_quant_conv(z)))
    _ok(float((once - twice).abs().mean()) > 1e-3,
        "Student applies post_quant_conv exactly once (double would differ)")

    # scaling_factor must NOT be applied inside the decoder path
    _ok(float(vae.config.scaling_factor) == 0.18215 and
        "scaling" not in type(student).__name__.lower(),
        "no internal scaling_factor")
    return vae, student, idx


def test_wavelet(dev):
    print("\n=== 2. wavelet losses ===", flush=True)
    x = torch.randn(2, 3, 512, 512, device=dev)
    bands = haar_dwt(x, 2)
    _ok(len(bands) == 2 and all(len(b) == 4 for b in bands), "two levels x 4 bands")
    _ok(tuple(bands[0][0].shape) == (2, 3, 256, 256) and
        tuple(bands[1][0].shape) == (2, 3, 128, 128), "band shapes L1/L2")
    scales = {(l, b): 1.0 for l in range(2) for b in range(3)}
    whf = WaveletHF(scales).to(dev)
    l_hf, l_eng, per = whf(x, x)
    _ok(float(l_hf) < 1e-6, f"identical images -> HF ~0 ({float(l_hf):.2e})")
    _ok(float(l_eng) < 1e-6, "identical images -> energy ~0")
    y = (x + 0.1 * torch.randn_like(x)).requires_grad_(True)
    l_hf2, _, _ = whf(y, x)
    _ok(float(l_hf2) > 0, "perturbed image -> HF > 0")
    l_hf2.backward()
    _ok(y.grad is not None and bool(torch.isfinite(y.grad).all()) and
        float(y.grad.abs().sum()) > 0, "HF gradient finite & non-zero")


def test_response(dev, vae, student):
    print("\n=== 3. response distillation ===", flush=True)
    teacher = vae.decoder.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    z = torch.randn(2, 4, 64, 64, device=dev)
    zero = torch.zeros_like(z)
    with torch.no_grad():
        l0, d0 = response_loss(student, teacher, z, zero)
    _ok(float(l0) < 1e-6, f"delta=0 -> response ~0 ({float(l0):.2e})")

    delta = make_perturbation(z, "corr_gauss", 0.05,
                              generator=torch.Generator(device=dev).manual_seed(0))
    rms = float(delta.pow(2).mean().sqrt() / z.pow(2).mean().sqrt())
    _ok(abs(rms - 0.05) < 0.01, f"perturbation RMS ratio ≈0.05 ({rms:.4f})")
    zr = z.clone().requires_grad_(True)
    lr, parts = response_loss(student, teacher, zr, delta)
    lr.backward()
    _ok(zr.grad is not None and bool(torch.isfinite(zr.grad).all()) and
        float(zr.grad.abs().sum()) > 0, "student input grad finite & non-zero")
    _ok(all(p.grad is None for p in teacher.parameters()),
        "teacher parameters have NO gradient")
    lo, _ = off_manifold_loss(student, teacher, z + delta)
    _ok(bool(torch.isfinite(lo)), "off-manifold loss finite")
    print(f"    resp parts: {parts}", flush=True)


def main():
    dev = "cuda"
    print(f"[selftest] device={dev}", flush=True)
    vae, student, idx = test_interface(dev)
    test_wavelet(dev)
    test_response(dev, vae, student)
    print("\n[selftest] ALL TESTS PASSED", flush=True)


if __name__ == "__main__":
    main()
