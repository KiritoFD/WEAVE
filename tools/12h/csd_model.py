"""Official CSD (ViT-L) loader + feature extraction.

Code:    https://github.com/learn2phoenix/CSD @ 3a9df32605b869eceb704897839be80977a9f1ea
Weights: tomg-group-umd/CSD-ViT-L @ 5bc26a6fb0487f3f00a2a7313135103a005b1b67 (HF mirror)
"""
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image

import paths
import clip
from clip.model import VisionTransformer, convert_weights

REVISION = "5bc26a6fb0487f3f00a2a7313135103a005b1b67"
REPO_ID = "tomg-group-umd/CSD-ViT-L"
CSD_CODE_COMMIT = "3a9df32605b869eceb704897839be80977a9f1ea"

PREPROCESS = transforms.Compose([
    transforms.Resize(size=224, interpolation=TF.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    ),
])


class CSD_CLIP(nn.Module):
    """Same structure as CSD/model.py::CSD_CLIP. The CLIP ViT-L/14 backbone is
    instantiated directly (identical architecture); all of its weights come from the
    CSD checkpoint, so no separate CLIP download is needed."""

    def __init__(self, name="vit_large", content_proj_head="default"):
        super().__init__()
        self.content_proj_head = content_proj_head
        if name == "vit_large":
            self.backbone = VisionTransformer(
                input_resolution=224, patch_size=14, width=1024, layers=24, heads=16,
                output_dim=768,
            )
            self.embedding_dim = 1024
        else:
            raise Exception("This model is not implemented")
        convert_weights(self.backbone)
        self.last_layer_style = copy.deepcopy(self.backbone.proj)
        self.last_layer_content = copy.deepcopy(self.backbone.proj)
        self.backbone.proj = None

    @property
    def dtype(self):
        return self.backbone.conv1.weight.dtype

    def forward(self, x):
        feature = self.backbone(x)
        style_output = nn.functional.normalize(feature @ self.last_layer_style, dim=1, p=2)
        content_output = nn.functional.normalize(feature @ self.last_layer_content, dim=1, p=2)
        return feature, content_output, style_output


def weights_path():
    return paths.hf_cache_root("models--tomg-group-umd--CSD-ViT-L", "snapshots", REVISION, "pytorch_model.bin")


def load_csd(device="cuda"):
    model = CSD_CLIP("vit_large", "default")
    ck = torch.load(str(weights_path()), map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"] if "model_state_dict" in ck else ck
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    msg = model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()
    return model, ck, msg


def feats_pils(model, pils, batch=32, device="cuda"):
    """style head output (768-d, L2-normalised) for a list of PIL images."""
    out = []
    with torch.inference_mode():
        for s in range(0, len(pils), batch):
            x = torch.stack([PREPROCESS(im) for im in pils[s:s + batch]])
            x = x.to(device=device, dtype=torch.float16)
            with torch.cuda.amp.autocast():
                _, _, sty = model(x)
            out.append(sty.float().cpu())
    return torch.cat(out, 0).numpy()


def feats_paths(model, paths_, batch=32, device="cuda"):
    return feats_pils(model, [Image.open(p).convert("RGB") for p in paths_], batch, device)


def provenance(model, ck, timing_ms):
    import clip as clip_pkg
    ckpt_iter = ck.get("iter") if isinstance(ck, dict) else None
    return {
        "descriptor": "CSD (Contrastive Style Descriptor), official checkpoint",
        "code_repo": "https://github.com/learn2phoenix/CSD",
        "code_commit": CSD_CODE_COMMIT,
        "weights_repo": REPO_ID,
        "weights_revision": REVISION,
        "weights_file": "pytorch_model.bin",
        "weights_bytes": weights_path().stat().st_size,
        "weights_path_rel_cache_root": str(weights_path().relative_to(paths.hf_cache_root())),
        "hf_endpoint_used": "https://hf-mirror.com",
        "hf_cache_root": str(paths.hf_cache_root()),
        "clip_package": f"clip-anytorch {getattr(clip_pkg, '__version__', '2.6.0')}",
        "checkpoint_iter": ckpt_iter,
        "backbone": "CLIP ViT-L/14 visual encoder (224px, patch 14, width 1024, 24 layers, 16 heads)",
        "params": int(sum(p.numel() for p in model.parameters())),
        "dtype": str(model.dtype),
        "preprocess": "Resize(shorter side 224, bicubic) -> CenterCrop(224) -> ToTensor -> CLIP mean/std",
        "input_resolution": 224,
        "patch_grid": "16x16 (+1 CLS) = 257 tokens",
        "feature_dim": 768,
        "feature_extraction": "backbone CLS (1024-d) projected by last_layer_style -> L2 normalised (768-d)",
        "feature_norm": "L2",
        "similarity": "cosine (= inner product of L2-normalised descriptors)",
        "single_image_forward_ms": timing_ms,
        "upstream_caveat": "official README (2024-10-03) flags a known discrepancy of the uploaded "
                           "weights vs the numbers reported in the paper",
    }


if __name__ == "__main__":
    t0 = time.time()
    model, ck, msg = load_csd()
    print("load msg:", msg)
    print("load seconds:", round(time.time() - t0, 2))
    print(json.dumps(provenance(model, ck, None), indent=1))

