"""Compact standalone deployment bundle builder & zero-bloat loader for CLEAVE V35.

Extracts only required inference weights/tensors (~400 MB) out of multi-GB training caches
(ccf_pool_cache.pt / cnx_stiefel_cache.pt), verifies SHA-256 hashes, and enforces true
E-only deployment without instantiating P-Core weights unless requested.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import torch

from exp.deploy_v35 import BUNDLE_DEFAULT_DIR, P8_LOCKED_CFG
from src.utils.runtime import REPO as ROOT, read_json, save_json, utc


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _locate_convnext_pth() -> Path:
    import paths
    fname = "convnext_tiny-983f1562.pth"
    for cand in (
        Path(torch.hub.get_dir()) / "checkpoints" / fname,
        paths.eval_cache_root("torch") / "hub/checkpoints" / fname,
        paths.weights_root(fname),
    ):
        if cand.exists() and cand.stat().st_size > 100_000_000:
            return cand
    raise FileNotFoundError(f"Offline {fname} not found in torch hub / eval_cache / weights")


def _extract_tiny_tensors() -> dict[str, Any]:
    tiny_p = ROOT / P8_LOCKED_CFG["tiny_tensors"]
    if tiny_p.exists():
        d = torch.load(tiny_p, map_location="cpu", weights_only=False)
        return {"W_stiefel_cnx": d["W_stiefel_cnx"].float(), "luma_proj": d["luma_proj"]}
    st = torch.load(ROOT / "runs/v32_nodino/cache/cnx_stiefel_cache.pt", map_location="cpu", weights_only=False)
    v18 = torch.load(ROOT / "runs/v18/cache/ccf_pool_cache.pt", map_location="cpu", weights_only=False)
    return {"W_stiefel_cnx": st["W_stiefel_cnx"].float(), "luma_proj": v18["luma_proj"]}


def build_deploy_bundle(bundle_dir: Path | str = BUNDLE_DEFAULT_DIR, *, overwrite: bool = True) -> dict[str, Any]:
    """Assemble self-contained V35 inference bundle with SHA-256 manifest."""
    out = Path(bundle_dir)
    if out.exists() and overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    copies = {
        "pcore.pt": ROOT / P8_LOCKED_CFG["pcore_ckpt"],
        "ecore.pt": ROOT / P8_LOCKED_CFG["ecore_ckpt"],
        "a0_encoder.pt": ROOT / P8_LOCKED_CFG["a0_ckpt"],
        "s10b_decoder.pt": ROOT / P8_LOCKED_CFG["s10b_ckpt"],
        "k8_bank.pt": ROOT / P8_LOCKED_CFG["k8_bank"],
        "a3_priors.pt": ROOT / P8_LOCKED_CFG["a3_priors"],
        "convnext_tiny.pth": _locate_convnext_pth(),
    }
    assets: dict[str, Any] = {}
    for dst_name, src_path in copies.items():
        if not src_path.exists():
            raise FileNotFoundError(f"Missing required bundle source asset: {src_path}")
        dst_path = out / dst_name
        shutil.copy2(src_path, dst_path)
        assets[dst_name] = {
            "source_rel": src_path.relative_to(ROOT).as_posix() if src_path.is_relative_to(ROOT) else src_path.name,
            "bytes": dst_path.stat().st_size,
            "sha256": file_sha256(dst_path),
        }

    tiny = _extract_tiny_tensors()
    tiny_out = out / "tiny_tensors.pt"
    torch.save(tiny, tiny_out)
    assets["tiny_tensors.pt"] = {
        "source_rel": "extracted(W_stiefel_cnx, luma_proj)",
        "bytes": tiny_out.stat().st_size,
        "sha256": file_sha256(tiny_out),
    }

    manifest = {
        "version": "cleave_v35_final",
        "created_utc": utc(),
        "schedule": {
            "ws_mul": P8_LOCKED_CFG["ws_mul"],
            "wc_mul": P8_LOCKED_CFG["wc_mul"],
            "dtau0": P8_LOCKED_CFG["dtau0"],
            "steps": P8_LOCKED_CFG["steps"],
            "alpha_y": P8_LOCKED_CFG["alpha_y"],
            "alpha_c": P8_LOCKED_CFG["alpha_c"],
            "scaling_factor": P8_LOCKED_CFG["scaling_factor"],
        },
        "total_bytes": sum(v["bytes"] for v in assets.values()),
        "assets": assets,
    }
    save_json(out / "manifest.json", manifest)
    return manifest


def verify_bundle_manifest(bundle_dir: Path | str = BUNDLE_DEFAULT_DIR) -> dict[str, Any]:
    out = Path(bundle_dir)
    manifest = read_json(out / "manifest.json")
    for fname, meta in manifest["assets"].items():
        fp = out / fname
        if not fp.exists():
            raise FileNotFoundError(f"Bundle asset missing: {fp}")
        if fp.stat().st_size != meta["bytes"]:
            raise ValueError(f"Size mismatch for {fname}: {fp.stat().st_size} != {meta['bytes']}")
        got_hash = file_sha256(fp)
        if got_hash != meta["sha256"]:
            raise ValueError(f"SHA256 mismatch for {fname}: {got_hash} != {meta['sha256']}")
    return manifest


def load_pcore_from_file(pcore_path: Path, device: str = "cuda", *, dtype: torch.dtype = torch.float32):
    from exp.v29_biglittle.pcore_model import PCoreSLFUNet
    sv = torch.load(pcore_path, map_location="cpu", weights_only=False)
    cfg = sv.get("cfg") or {"base_dim": 208, "u_dim": 417, "cond_dim": 384}
    model = PCoreSLFUNet(
        base_dim=int(cfg["base_dim"]),
        u_dim=int(cfg["u_dim"]),
        cond_dim=int(cfg["cond_dim"]),
        use_gamma=True,
        normalize_u=True,
    )
    model.load_state_dict(sv["ema"], strict=True)
    return model.to(device=device, dtype=dtype).eval().requires_grad_(False), cfg


def load_ecore_from_file(ecore_path: Path, device: str = "cuda", *, dtype: torch.dtype = torch.float32):
    from exp.v28_unified.student_models import build_student_model
    ec_sv = torch.load(ecore_path, map_location="cpu", weights_only=False)
    raw_stats = ec_sv["raw_stats"].to(device=device, dtype=torch.float32)
    priors_ec = {k: v.to(device=device, dtype=torch.float32) for k, v in ec_sv["priors"].items()}
    ecore = build_student_model(ec_sv.get("arch", "unet"), raw_stats.shape[-1], device=device)
    ecore.load_state_dict(ec_sv["ema"], strict=True)
    ecore = ecore.to(device=device, dtype=dtype).eval().requires_grad_(False)
    return ecore, raw_stats, priors_ec


def load_a0_encoder_from_file(a0_path: Path, device: str = "cuda", *, dtype: torch.dtype = torch.bfloat16):
    from diffusers import AutoencoderKL
    from exp.v25.r14_e2e_benchmark import _SD_VAE_CFG
    vae = AutoencoderKL.from_config(_SD_VAE_CFG)
    ck = torch.load(a0_path, map_location="cpu", weights_only=False)
    vae.load_state_dict(ck["state_dict"], strict=False)
    return vae.to(device=device, dtype=dtype).eval().requires_grad_(False)


def load_s10b_from_file(s10_path: Path, device: str = "cuda", *, dtype: torch.dtype = torch.float16):
    import torch.nn as nn
    from src.utils.vae import StudentDecoder, _build_s10_decoder
    ck = torch.load(s10_path, map_location="cpu", weights_only=False)
    pqc = nn.Conv2d(4, 4, 1)
    pqc.load_state_dict(ck["post_quant_conv"])
    dec = _build_s10_decoder(list(ck["meta"]["widths_forward"]))
    dec.load_state_dict(ck["state_dict"])
    return StudentDecoder(pqc, dec).to(device=device, dtype=dtype).eval().requires_grad_(False)


def load_convnext_s3_from_file(cnx_path: Path, device: str = "cuda", *, dtype: torch.dtype = torch.float32):
    import torchvision.models as tvm
    from exp.v32_nodino.cnx_layout import ConvNeXtTinyStage3Extractor
    net = tvm.convnext_tiny(weights=None)
    net.load_state_dict(torch.load(cnx_path, map_location="cpu", weights_only=True))
    ext = ConvNeXtTinyStage3Extractor.__new__(ConvNeXtTinyStage3Extractor)
    torch.nn.Module.__init__(ext)
    ext.stage3 = net.features[:6].to(device=device, dtype=dtype).eval()
    for p in ext.stage3.parameters():
        p.requires_grad_(False)
    ext.register_buffer("im_mean", torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype).view(1, 3, 1, 1))
    ext.register_buffer("im_std", torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype).view(1, 3, 1, 1))
    return ext


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-dir", default=str(BUNDLE_DEFAULT_DIR))
    args = ap.parse_args()
    m = build_deploy_bundle(args.bundle_dir, overwrite=True)
    verify_bundle_manifest(args.bundle_dir)
    print(f"BUNDLE_OK {args.bundle_dir} bytes={m['total_bytes']}")
