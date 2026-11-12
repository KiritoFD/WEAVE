"""WEAVE inference: checkpoint in, raw latents and images out.

Deliberately separate from evaluation:
  * this module NEVER computes DINO/CLIP/CSD/LPIPS and never writes PNG;
  * it stores raw latents plus uint8 RGB so metrics can be recomputed later
    without re-running the GPU.

One CLI, one model registry. Any current or future variant is registered in
`MODELS` and dispatched by name, so adding a variant never changes the runner.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import torch

REPO = Path(__file__).resolve().parents[2]

# Fixed batch for the compiled TensorRT decoder. A short final batch is padded
# so the engine is used for every call (no silent PyTorch fallback).
DEFAULT_BATCH = 8


@dataclass
class InferSpec:
    """Everything a model variant needs, resolved from CLI arguments."""
    ckpt: Path | None = None
    kind: str = "direct"
    canon_mode: str = "c1_gentle"
    steps: int = 4
    step_rms: float = 0.08
    rho: float = 0.16
    energy: str = ""
    seed: int = 42
    extra: dict = field(default_factory=dict)


class Generator(Protocol):
    """A callable that maps source latents to output latents."""
    def __call__(self, z_src: torch.Tensor, sid: torch.Tensor,
                 tid: torch.Tensor, tasks: list[dict]) -> dict: ...


# --------------------------------------------------------------------------
# model registry
# --------------------------------------------------------------------------
MODELS: dict[str, Callable[[InferSpec, str], Generator]] = {}


def register(name: str):
    def deco(fn):
        MODELS[name] = fn
        return fn
    return deco


def build(spec: InferSpec, device: str = "cuda") -> Generator:
    if not MODELS:
        from . import variants  # noqa: F401  (registers every variant)
    if spec.kind not in MODELS:
        raise KeyError(f"unknown model kind {spec.kind!r}; registered: {sorted(MODELS)}")
    return MODELS[spec.kind](spec, device)


# --------------------------------------------------------------------------
# shared pieces (deployment plumbing lives in src.utils)
# --------------------------------------------------------------------------
def load_vae(device: str):
    from src.utils import load_vae as _load
    return _load(device)


def load_trt(vae, device: str):
    from src.utils import load_trt as _load
    return _load(vae, device)


def decode_padded(trt, vae, z: torch.Tensor, batch: int = DEFAULT_BATCH) -> torch.Tensor:
    from src.utils import decode
    return decode(trt, vae, z, batch)


def canonicalizer(device: str):
    from src.model import canonicalizer as _canon
    return _canon(device)


def load_c3unet(path: Path, device: str, style_keys: bool = False):
    """Load a C3UNet checkpoint. The network class comes from src.model."""
    from src.model import C3UNet
    ck = torch.load(path, map_location="cpu", weights_only=False)
    args = ck.get("args", {})
    model = C3UNet(
        base_dim=args.get("base_dim", 16), num_styles=5,
        mode=args.get("mode", "direct"),
        inject_64=not args.get("no_inject_64", False),
        inject_32=not args.get("no_inject_32", False),
        inject_16=not args.get("no_inject_16", False),
        style_keys=bool(style_keys or args.get("style_keys", False)),
    ).to(device).eval()
    model.load_state_dict(ck["model"])
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ck


def load_critic(device: str, path: Path | None = None):
    from src.model import LatentStyleNet
    path = path or REPO / "runs/v6_next/probes/r1_r1.pt"
    ck = torch.load(path, map_location="cpu", weights_only=False)
    critic = LatentStyleNet(width=ck.get("width", 64)).to(device).eval()
    critic.load_state_dict(ck["model"])
    for p in critic.parameters():
        p.requires_grad_(False)
    return critic
