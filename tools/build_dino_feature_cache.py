"""Build the offline frozen-feature cache used by infra patch P1.

For every packed latent in the training pool this script runs the frozen
decoder + feature extractor ONCE and stores the resulting per-layer feature
maps, so training steps no longer pay for the two ``no_grad`` teacher passes on
style/content images.

Layout produced under ``--out``::

    index.json                  # styles, offsets, layers, shape, dtype
    f<res>_<layer>.npy          # [rows, tokens, channels], float16

Row for style ``s``, item ``i``, flip ``f``: ``offset[s] + i * stride + f``
(``stride`` is 2 with ``--flip_rows``, else 1). ``--flip_rows`` must match
``data.allow_hflip`` of the run that will consume the cache.

Usage (server side, after the packed latent cache exists)::

    python tools/build_dino_feature_cache.py \
        --config config.json --out /mnt/i/Latent_Style/dino_feats_f384 \
        --decoder taesd --extractor dinov2 \
        --extractor_local_dir /mnt/i/Latent_Style/weights/dinov2-base \
        --style_layers block6,block9 --content_layers block3,block6 \
        --selfsim_layer block9 --feature_res 384 --batch_size 16 --flip_rows
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config_schema import load_experiment_config  # noqa: E402
from features import build_feature_extractor  # noqa: E402
from output_supervision import build_latent_decoder  # noqa: E402
from utils.dataset import AdaCUTLatentDataset  # noqa: E402

logger = logging.getLogger("build_dino_feature_cache")

_NUMPY_DTYPES = {"float16": np.float16, "float32": np.float32}


def _resolve_data_root(cfg, override: str) -> str:
    if override:
        return override
    data_cfg = getattr(cfg, "data", None) if cfg is not None else None
    for key in ("data_root", "data_dir", "train_data_dir"):
        value = getattr(data_cfg, key, "") if data_cfg is not None else ""
        if value:
            return str(value)
    return "data"


def _resolve_latent_cache_dir(cfg, override: str) -> str:
    if override:
        return override
    data_cfg = getattr(cfg, "data", None) if cfg is not None else None
    return str(getattr(data_cfg, "latent_cache_dir", "") or "")


def _split_csv(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(_REPO_ROOT / "config.json"), help="Experiment config used to resolve defaults.")
    parser.add_argument("--out", required=True, help="Output directory for the feature cache.")
    parser.add_argument("--data_root", default="", help="Latent data root (defaults to config.data.data_root).")
    parser.add_argument("--latent_cache_dir", default="", help="Packed latent cache dir (defaults to config.data.latent_cache_dir).")
    parser.add_argument("--style_subdirs", default="", help="Comma separated style subdirs (defaults to config.data.style_subdirs).")
    parser.add_argument("--decoder", default="", help="Latent decoder: taesd|sd15|tiny (defaults to config.bridge.output_supervision_decoder).")
    parser.add_argument("--extractor", default="dinov2", help="Feature extractor name.")
    parser.add_argument("--extractor_local_dir", default="", help="Offline HF snapshot for dinov2.")
    parser.add_argument("--style_layers", default="", help="Layer names for the style (SWD) branch.")
    parser.add_argument("--content_layers", default="", help="Perceptual layer names for the content branch.")
    parser.add_argument("--selfsim_layer", default="", help="Self-similarity layer name.")
    parser.add_argument("--feature_res", type=int, default=0, help="Square feature resolution; 0 = use config value.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp_dtype", default="bf16", choices=["bf16", "fp16", "none"],
                        help="Autocast dtype used while caching (match the training run).")
    parser.add_argument("--cache_dtype", default="float16", choices=sorted(_NUMPY_DTYPES))
    parser.add_argument("--flip_rows", action=argparse.BooleanOptionalAction, default=None,
                        help="Store a horizontally flipped variant per item (match data.allow_hflip).")
    parser.add_argument("--limit_per_style", type=int, default=0, help="Debug: only cache the first N items per style.")
    parser.add_argument("--overwrite", action="store_true", help="Rewrite an existing cache directory.")
    return parser.parse_args(argv)


def _autocast_ctx(device: torch.device, amp_dtype: str):
    if device.type != "cuda" or amp_dtype == "none":
        return torch.autocast("cpu", enabled=False)
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    return torch.amp.autocast("cuda", enabled=True, dtype=dtype)


class _LayerWriter:
    """Thin wrapper around one ``np.memmap`` per layer."""

    def __init__(self, out_dir: Path, *, layer_files: dict[str, str], shape: tuple[int, int, int], dtype: np.dtype) -> None:
        self._maps = {
            name: np.memmap(out_dir / rel, dtype=dtype, mode="w+", shape=shape)
            for name, rel in layer_files.items()
        }

    def write(self, rows: list[int], feats: dict[str, torch.Tensor]) -> None:
        for name, feat in feats.items():
            array = feat.detach().to("cpu", dtype=torch.float32).numpy()
            array = array.transpose(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
            self._maps[name][rows] = array.reshape(len(rows), -1, array.shape[-1]).astype(self._maps[name].dtype)

    def flush(self) -> None:
        for mm in self._maps.values():
            mm.flush()


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = _parse_args(argv)

    cfg = None
    if args.config and Path(args.config).is_file():
        cfg = load_experiment_config(args.config)
    else:
        logger.warning("Config %s not found; falling back to CLI/default values.", args.config)

    bridge = getattr(cfg, "bridge", None) if cfg is not None else None
    data_cfg = getattr(cfg, "data", None) if cfg is not None else None

    style_subdirs = _split_csv(args.style_subdirs) or list(getattr(data_cfg, "style_subdirs", []) or [])
    if not style_subdirs:
        raise SystemExit("No style subdirs: pass --style_subdirs or a valid --config")

    feature_res = int(args.feature_res or getattr(bridge, "output_supervision_feature_res", 0) or 0)
    if feature_res <= 0:
        raise SystemExit("feature_res must be > 0; the cache is only valid when training downsamples to a fixed resolution")

    style_layers = _split_csv(args.style_layers) or _split_csv(str(getattr(bridge, "output_supervision_style_layers", "")))
    content_layers = _split_csv(args.content_layers) or _split_csv(str(getattr(bridge, "output_supervision_content_layers", "")))
    selfsim_layer = str(args.selfsim_layer or getattr(bridge, "output_supervision_selfsim_layer", "")).strip()
    layers = list(dict.fromkeys(style_layers + content_layers + [selfsim_layer]))
    if not layers:
        raise SystemExit("No layers requested")

    decoder_name = str(args.decoder or getattr(bridge, "output_supervision_decoder", "tiny"))
    extractor_dir = str(args.extractor_local_dir or getattr(bridge, "output_supervision_extractor_local_dir", "") or "")

    allow_hflip = bool(getattr(data_cfg, "allow_hflip", True)) if args.flip_rows is None else bool(args.flip_rows)
    if data_cfg is not None and allow_hflip != bool(getattr(data_cfg, "allow_hflip", True)):
        logger.warning(
            "flip_rows=%s does not match data.allow_hflip=%s; flipped samples will fall back to the online path.",
            allow_hflip, getattr(data_cfg, "allow_hflip", True),
        )

    data_root = _resolve_data_root(cfg, args.data_root)
    latent_cache_dir = _resolve_latent_cache_dir(cfg, args.latent_cache_dir)

    dataset = AdaCUTLatentDataset(
        data_root=data_root,
        style_subdirs=style_subdirs,
        allow_hflip=allow_hflip,
        limit_per_style=int(args.limit_per_style),
        latent_cache_mode="packed" if latent_cache_dir else "off",
        latent_cache_dir=latent_cache_dir,
    )
    counts = {style_subdirs[sid]: int(stack.shape[0]) for sid, stack in dataset.style_tensors.items()}

    out_dir = Path(args.out)
    if out_dir.is_dir() and any(out_dir.glob("f*_*.npy")):
        if not args.overwrite:
            raise SystemExit(f"{out_dir} already contains a cache; pass --overwrite to rebuild")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    decoder = build_latent_decoder(decoder_name, device=device)
    extractor = build_feature_extractor(args.extractor, device=device, local_dir=extractor_dir)

    stride = 2 if allow_hflip else 1
    offsets: dict[str, int] = {}
    total = 0
    for name in style_subdirs:
        offsets[name] = total
        total += counts.get(name, 0) * stride

    if total <= 0:
        raise SystemExit("Empty latent pool; nothing to cache")

    numpy_dtype = _NUMPY_DTYPES[args.cache_dtype]
    layer_files = {name: f"f{feature_res}_{name.replace('/', '_')}.npy" for name in layers}

    def _feats(latents: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            img = decoder.decode_image(latents)
            if img.shape[-1] != feature_res:
                img = F.interpolate(img, size=(feature_res, feature_res), mode="bilinear", align_corners=False)
        with torch.no_grad(), _autocast_ctx(device, args.amp_dtype):
            return extractor.forward_layers(img.float(), layers)

    # Probe one batch to learn (tokens, channels, side) before allocating memmaps.
    first_style = style_subdirs[0]
    probe_latents = dataset.style_tensors[0][: min(max(1, int(args.batch_size)), counts[first_style])]
    probe_feats = _feats(probe_latents.to(device))
    tokens = int(probe_feats[layers[0]].shape[-2] * probe_feats[layers[0]].shape[-1])
    side = int(probe_feats[layers[0]].shape[-1])
    channels = int(probe_feats[layers[0]].shape[1])
    for name, feat in probe_feats.items():
        if int(feat.shape[1]) != channels or int(feat.shape[-2] * feat.shape[-1]) != tokens:
            raise SystemExit(f"Layer {name} has shape {tuple(feat.shape)}; all cached layers must share tokens/channels")

    writer = _LayerWriter(out_dir, layer_files=layer_files, shape=(total, tokens, channels), dtype=numpy_dtype)
    logger.info("Cache plan: %d rows, shape=(%d, %d, %d), layers=%s", total, tokens, channels, layers)

    batch_size = max(1, int(args.batch_size))
    written = 0
    for sid, style_name in enumerate(style_subdirs):
        stack = dataset.style_tensors[sid]
        n = int(stack.shape[0])
        offset = offsets[style_name]
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            latents = stack[start:end].to(device)
            feats = _feats(latents)
            rows = [offset + (start + i) * stride for i in range(end - start)]
            writer.write(rows, feats)
            written += end - start
            if allow_hflip:
                flipped = torch.flip(latents, dims=[-1])
                feats_flip = _feats(flipped)
                writer.write([row + 1 for row in rows], feats_flip)
            if written % (batch_size * 10) == 0 or written == total:
                logger.info("cached %d/%d items", written, total)
        logger.info("style %s done (%d items)", style_name, n)

    writer.flush()
    index = {
        "schema": 1,
        "feature_res": feature_res,
        "tokens": tokens,
        "channels": channels,
        "side": side,
        "dtype": args.cache_dtype,
        "flip_rows": bool(allow_hflip),
        "stride": stride,
        "layers": layer_files,
        "style_subdirs": style_subdirs,
        "style_counts": counts,
        "style_offsets": offsets,
        "decoder": decoder_name,
        "extractor": str(args.extractor),
        "extractor_local_dir": extractor_dir,
        "amp_dtype": str(args.amp_dtype),
        "data_root": str(data_root),
        "latent_cache_dir": str(latent_cache_dir),
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote cache index: %s", out_dir / "index.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
