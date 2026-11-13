"""Windows/local fallback for the r2_eval reference interface used by FeatureStack.

``tools.v6_next.r2_eval`` transitively imports ``tools.v5.route_c`` which pulls in
``tools.eval_fast`` -> ``resource`` (Unix-only). On machines without that chain we
re-export the exact same functions from their original modules so feature
extraction is bit-identical:

- csd_model      -> tools/12h/csd_model.py  (official CSD ViT-L)
- load_dino_model / encode_dino -> tools/pseudo_gt/matching.py (DINOv2-small)
- csd_prototypes -> same csd_proto.pt asset route_c loads
- to_lpips_input -> utils/run_evaluation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / "tools" / "12h")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import csd_model  # noqa: E402  (tools/12h/csd_model.py)
from tools.pseudo_gt.matching import encode_dino, load_dino_model  # noqa: E402
from tools.legacy_utils.run_evaluation import to_lpips_input  # noqa: E402

import paths  # noqa: E402


def csd_prototypes() -> np.ndarray:
    import paths as _paths
    STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
    p = _paths.asset_root() / "pseudo_gt_24h" / "csd_proto.pt"
    d = torch.load(p, map_location="cpu", weights_only=False)
    return np.stack([d[s] for s in STYLES])


__all__ = ["csd_model", "encode_dino", "load_dino_model", "csd_prototypes",
           "to_lpips_input"]
