"""Next-gen Route A/C: pluggable frozen visual feature extractors.

All output-level style/content supervision (Route A) and the reference style
encoder (Route C) need features from a frozen pretrained visual network
(VGG / DINOv2 / ...). To keep the backbone swappable, every extractor
implements the same minimal interface:

    extractor.forward_layers(x, layers) -> dict[layer_name, Tensor]

with ``x`` an image batch in [0, 1] (B, 3, H, W). All extractors are frozen
(``requires_grad_(False)``, ``eval()``) but remain differentiable w.r.t. their
*input* so gradients flow back into the decoded image.

Available backends:
  - ``tiny_conv``: deterministic randomly-initialised tiny CNN. No download,
    CPU-friendly. This is the DEFAULT so smoke tests run without GPU/data.
    It is NOT semantically meaningful — real experiments must switch to
    ``vgg16`` / ``dinov2``.
  - ``vgg16`` / ``vgg19``: torchvision VGG feature stack (classic Gatys-style
    layers relu1_2, relu2_2, relu3_3, relu4_3, ...). Requires torchvision
    weights (server).
  - ``dinov2``: DINOv2 ViT via torch.hub (server). Exposes intermediate
    patch-token layers plus the CLS token.
"""
from __future__ import annotations

from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F
from torch import nn


class FrozenFeatureExtractor(nn.Module):
    """Interface for pluggable frozen feature extractors."""

    name: str = "abstract"

    def __init__(self) -> None:
        super().__init__()
        self.eval()

    def available_layers(self) -> List[str]:
        raise NotImplementedError

    def forward_layers(self, x: torch.Tensor, layers: Iterable[str]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, layers: Iterable[str] | None = None) -> Dict[str, torch.Tensor]:
        return self.forward_layers(x, list(layers) if layers is not None else self.available_layers())

    def freeze_(self) -> "FrozenFeatureExtractor":
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)
        return self


class TinyConvExtractor(FrozenFeatureExtractor):
    """Deterministic tiny CNN extractor for offline/CPU smoke tests.

    Fixed seed => identical weights across runs/machines. Features are random
    projections, useful ONLY for plumbing tests (forward/backward/shape).
    """

    name = "tiny_conv"
    LAYERS = ("conv1", "conv2", "conv3", "conv4")

    def __init__(self, in_channels: int = 3, width: int = 32, seed: int = 1234) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        ch = [in_channels, width, width, width * 2, width * 2]
        blocks = []
        for i in range(4):
            conv = nn.Conv2d(ch[i], ch[i + 1], kernel_size=3, padding=1, stride=1 if i < 2 else 2)
            with torch.no_grad():
                w = torch.empty(conv.weight.shape)
                w.normal_(0.0, 0.05, generator=g)
                conv.weight.copy_(w)
                if conv.bias is not None:
                    conv.bias.zero_()
            blocks.append(nn.Sequential(conv, nn.ReLU(inplace=True)))
        self.blocks = nn.ModuleList(blocks)
        self.freeze_()

    def available_layers(self) -> List[str]:
        return list(self.LAYERS)

    def forward_layers(self, x: torch.Tensor, layers: Iterable[str]) -> Dict[str, torch.Tensor]:
        wanted = set(layers)
        out: Dict[str, torch.Tensor] = {}
        h = x
        for name, blk in zip(self.LAYERS, self.blocks):
            h = blk(h)
            if name in wanted:
                out[name] = h
        missing = wanted - set(out)
        if missing:
            raise KeyError(f"TinyConvExtractor has no layers {sorted(missing)}; available {self.LAYERS}")
        return out


_VGG_LAYER_INDEX = {
    # torchvision vgg16 feature indices (with ReLU after conv)
    "relu1_2": 3, "relu2_2": 8, "relu3_3": 15, "relu4_3": 22, "relu5_3": 31,
}


class VGGExtractor(FrozenFeatureExtractor):
    """torchvision VGG feature extractor (Gatys-style perceptual layers)."""

    name = "vgg16"

    def __init__(self, arch: str = "vgg16", pretrained: bool = True) -> None:
        super().__init__()
        from torchvision import models  # lazy: server-only dependency

        weights = "DEFAULT" if pretrained else None
        if arch == "vgg19":
            net = models.vgg19(weights=weights).features
            self.name = "vgg19"
            self._index = {
                "relu1_2": 3, "relu2_2": 8, "relu3_4": 17, "relu4_4": 26, "relu5_4": 35,
            }
        else:
            net = models.vgg16(weights=weights).features
            self._index = dict(_VGG_LAYER_INDEX)
        self.features = net
        # ImageNet normalisation buffers
        self.register_buffer("_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.freeze_()

    def available_layers(self) -> List[str]:
        return list(self._index)

    def forward_layers(self, x: torch.Tensor, layers: Iterable[str]) -> Dict[str, torch.Tensor]:
        wanted = set(layers)
        unknown = wanted - set(self._index)
        if unknown:
            raise KeyError(f"VGGExtractor has no layers {sorted(unknown)}; available {list(self._index)}")
        h = (x.clamp(0.0, 1.0) - self._mean) / self._std
        max_idx = max(self._index[name] for name in wanted)
        out: Dict[str, torch.Tensor] = {}
        for i, layer in enumerate(self.features):
            h = layer(h)
            for name, idx in self._index.items():
                if i == idx and name in wanted:
                    out[name] = h
            if i >= max_idx:
                break
        return out


class DINOv2Extractor(FrozenFeatureExtractor):
    """DINOv2 ViT extractor, loading from a local HF directory when provided.

    Layers are intermediate transformer block indices as strings
    (``"block3"``, ``"block6"``, ...) returning patch tokens reshaped to
    (B, C, H/14, W/14), plus ``"cls"`` returning the CLS token (B, C).

    ``local_dir`` (preferred) loads ``transformers.Dinov2Model.from_pretrained``
    with ``local_files_only=True`` so the server never touches the network;
    otherwise falls back to ``torch.hub``.
    """

    name = "dinov2"

    def __init__(self, arch: str = "dinov2_vits14", pretrained: bool = True, local_dir: str = "") -> None:
        super().__init__()
        self.arch = arch
        self.local_dir = str(local_dir or "").strip()
        self._hf = bool(self.local_dir)
        if self._hf:
            from transformers import Dinov2Model  # lazy: server-only dependency

            self.model = Dinov2Model.from_pretrained(self.local_dir, local_files_only=True)
            cfg = self.model.config
            self.embed_dim = int(getattr(cfg, "hidden_size", 768))
            self.patch = int(getattr(cfg, "patch_size", 14))
            n_blocks = int(getattr(cfg, "num_hidden_layers", 12))
        else:
            self.model = torch.hub.load("facebookresearch/dinov2", arch, pretrained=pretrained)
            self.embed_dim = int(getattr(self.model, "embed_dim", 384))
            self.patch = int(getattr(getattr(self.model, "patch_embed", None), "patch_size", (14, 14))[0]) \
                if hasattr(getattr(self.model, "patch_embed", None), "patch_size") else 14
            n_blocks = len(getattr(self.model, "blocks", []))
        self._layers = [f"block{i}" for i in range(n_blocks)] + ["cls"]
        self.register_buffer("_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.freeze_()

    def available_layers(self) -> List[str]:
        return list(self._layers)

    def _hf_forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        kwargs = {"output_hidden_states": True, "return_dict": True}
        try:
            out = self.model(h, interpolate_pos_encoding=True, **kwargs)
        except TypeError:
            out = self.model(h, **kwargs)
        hs = out.hidden_states
        if hs is None:
            raise RuntimeError("Dinov2Model did not return hidden states")
        out_map: Dict[str, torch.Tensor] = {}
        for name in self._layers:
            if name == "cls":
                out_map[name] = hs[-1][:, 0]
                continue
            idx = int(name[5:]) + 1  # hidden_states[0] is the embedding output
            idx = min(idx, len(hs) - 1)
            tokens = hs[idx]
            patch_tokens = tokens[:, 1:]  # drop CLS
            n = patch_tokens.shape[1]
            side = int(round(n ** 0.5))
            if side * side != n:
                raise RuntimeError(f"DINO patch token count {n} is not a perfect square")
            b, _, c = patch_tokens.shape
            out_map[name] = patch_tokens.transpose(1, 2).reshape(b, c, side, side).contiguous()
        return out_map

    def forward_layers(self, x: torch.Tensor, layers: Iterable[str]) -> Dict[str, torch.Tensor]:
        wanted = set(layers)
        unknown = wanted - set(self._layers)
        if unknown:
            raise KeyError(f"DINOv2Extractor has no layers {sorted(unknown)}; available {self._layers}")
        h = (x.clamp(0.0, 1.0) - self._mean) / self._std
        if self._hf:
            out = self._hf_forward(h)
            return {name: out[name] for name in wanted}
        block_ids = sorted(int(name[5:]) for name in wanted if name.startswith("block"))
        out: Dict[str, torch.Tensor] = {}
        feats = self.model.get_intermediate_layers(h, n=block_ids, reshape=True)
        for bid, feat in zip(block_ids, feats):
            out[f"block{bid}"] = feat
        if "cls" in wanted:
            # Cheap CLS approximation from the deepest requested block's patch mean.
            tokens = self.model.get_intermediate_layers(h, n=[max(block_ids) if block_ids else len(self._layers) - 2], reshape=False)
            out["cls"] = tokens[0][:, 0]
        return out


def build_feature_extractor(
    name: str,
    *,
    device: torch.device | str = "cpu",
    pretrained: bool = True,
    local_dir: str = "",
) -> FrozenFeatureExtractor:
    """Factory: build a frozen extractor by name.

    ``tiny_conv`` works everywhere; ``vgg16``/``vgg19``/``dinov2`` require
    their respective weights (server side). ``local_dir`` (dinov2 only) loads
    a local HF snapshot offline instead of torch.hub.
    """
    key = str(name).strip().lower()
    if key in ("tiny", "tiny_conv", "random"):
        ext = TinyConvExtractor()
    elif key in ("vgg", "vgg16"):
        ext = VGGExtractor("vgg16", pretrained=pretrained)
    elif key == "vgg19":
        ext = VGGExtractor("vgg19", pretrained=pretrained)
    elif key.startswith("dinov2"):
        ext = DINOv2Extractor(
            arch=key if key != "dinov2" else "dinov2_vits14",
            pretrained=pretrained, local_dir=local_dir,
        )
    else:
        raise ValueError(f"Unknown feature extractor {name!r}; expected tiny_conv|vgg16|vgg19|dinov2")
    return ext.to(device).freeze_()


# ---------------------------------------------------------------------------
# Shared differentiable helpers used by style/content losses.
# ---------------------------------------------------------------------------

def extract_patch_descriptors(feat: torch.Tensor, patch: int, stride: int) -> torch.Tensor:
    """Extract flattened local patch descriptors from a feature map.

    Args:
        feat: (B, C, H, W) feature map.
    Returns:
        (B, N, C*patch*patch) descriptors, N = number of patches. Gradient flows
        back into ``feat``.
    """
    B, C, H, W = feat.shape
    if H < patch or W < patch:
        # Fall back to global descriptor when the map is smaller than a patch.
        return feat.flatten(2).mean(dim=2, keepdim=False).unsqueeze(1)  # (B,1,C)
    unf = F.unfold(feat, kernel_size=patch, stride=stride)  # (B, C*p*p, N)
    return unf.transpose(1, 2)  # (B, N, C*p*p)


def extract_patch_descriptors_chunked(
    feat: torch.Tensor, patch: int, stride: int, chunk_rows: int = 0,
) -> torch.Tensor:
    """Memory-capped variant of :func:`extract_patch_descriptors`.

    Processes the feature map in horizontal bands of ``chunk_rows`` patch rows,
    so the peak ``unfold`` buffer scales with the band height instead of the
    full patch count. Slicing keeps the graph differentiable w.r.t. ``feat``.
    """
    if chunk_rows <= 0:
        return extract_patch_descriptors(feat, patch, stride)
    B, C, H, W = feat.shape
    if H < patch or W < patch:
        return extract_patch_descriptors(feat, patch, stride)
    n_rows = (H - patch) // stride + 1
    parts: List[torch.Tensor] = []
    r = 0
    while r < n_rows:
        r1 = min(n_rows, r + chunk_rows)
        lo = r * stride
        hi = (r1 - 1) * stride + patch
        sub = feat[:, :, lo:hi, :]
        unf = F.unfold(sub, kernel_size=patch, stride=stride)  # (B, C*p*p, rows_chunk * Wcols)
        parts.append(unf.transpose(1, 2))
        r = r1
    return torch.cat(parts, dim=1)


def sobel_edges(x: torch.Tensor) -> torch.Tensor:
    """Differentiable Sobel edge magnitude for an image/feature map.

    Args:
        x: (B, C, H, W); edges computed per channel then averaged.
    Returns:
        (B, 1, H, W) edge magnitude in [0, ~1].
    """
    B, C, H, W = x.shape
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=x.device, dtype=x.dtype)
    ky = kx.t().contiguous()
    weight = torch.stack([kx, ky]).unsqueeze(1)  # (2,1,3,3)
    weight = weight.repeat(C, 1, 1, 1)  # depthwise
    g = F.conv2d(x, weight, padding=1, groups=C)
    gx, gy = g[:, 0::2], g[:, 1::2]
    mag = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-8)
    return mag.mean(dim=1, keepdim=True)


def patch_self_similarity(feat: torch.Tensor) -> torch.Tensor:
    """Normalised patch self-similarity (Gram) matrix of a feature map.

    Used by the DINO patch self-similarity content loss: compare the spatial
    correlation *structure* of output vs content features, not their values.

    Args:
        feat: (B, C, H, W)
    Returns:
        (B, N, N) cosine-similarity matrices over spatial positions.
    """
    B, C, H, W = feat.shape
    f = feat.flatten(2).transpose(1, 2)  # (B, N, C)
    f = F.normalize(f, p=2, dim=-1)
    return torch.bmm(f, f.transpose(1, 2))  # (B, N, N)
