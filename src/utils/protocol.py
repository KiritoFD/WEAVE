"""Protocol helpers: Tasks750 loader, clean pool disjointness, and collision-free wrong-label shifts."""
from __future__ import annotations

from pathlib import Path
import unicodedata

import torch

from .runtime import REPO, read_json

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
EVAL_SEED = 20261185


def canonical_pid(name: str) -> str:
    return unicodedata.normalize(
        "NFC", Path(str(name).replace("\\", "/")).stem.split("__")[-1]
    ).casefold()


def load_tasks_750() -> list[dict]:
    return read_json(REPO / "eval_protocol/tasks_750.json")["tasks"]


def pick_wrong_cross_tid(sid: torch.Tensor, tid: torch.Tensor, num_styles: int = 5) -> torch.Tensor:
    """Deterministic counterfactual wrong target label: guarantees wrong != tid AND wrong != sid on cross tasks."""
    same = sid == tid
    cand = (tid + 1) % num_styles
    cand = torch.where(cand == sid, (cand + 1) % num_styles, cand)
    return torch.where(same, tid, cand)


def load_vgg16_features_offline(device: str = "cuda"):
    """Load ImageNet VGG-16 features preferring local cached checkpoints before torchvision hub."""
    import paths
    import torchvision.models as tvm

    fname = "vgg16-397923af.pth"
    candidates = [
        Path(torch.hub.get_dir()) / "checkpoints" / fname,
        paths.eval_cache_root("torch") / "hub/checkpoints" / fname,
        paths.weights_root(fname),
        paths.project_root() / "_tmp" / fname,
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 500_000_000:
            net = tvm.vgg16(weights=None)
            net.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))
            feat = net.features.to(device).eval()
            for param in feat.parameters():
                param.requires_grad_(False)
            return feat

    feat = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1).features.to(device).eval()
    for param in feat.parameters():
        param.requires_grad_(False)
    return feat
