"""Shared infrastructure: deployment plumbing, S10 decoding, and frozen referees."""
from .vae import (DEFAULT_BATCH, S10_DECODER, StudentDecoder, decode,
                  decode_s10, load_s10_decoder, load_trt, load_vae,
                  s10_scaling_factor)
from .features import FeatureStack, STYLES, reference_bank
from .raw_io import RawWriter, export_raw_pngs, save_png_batch_fast
from .runtime import (auto_decode_batch, auto_infer_batch, auto_referee_batch,
                      auto_vram_budget_gb, bf16_autocast, read_json,
                      run_isolated_python, sanitize_cuda_env, save_json,
                      seed_all, sha256_file, tier_micro_accum, utc,
                      vram_total_gb)
from .protocol import (EVAL_SEED, canonical_pid, load_tasks_750,
                       load_vgg16_features_offline, pick_wrong_cross_tid)
from .pool import clean_balanced_pool
from .structure import (extract_color_fiber_27d, extract_whitened_dino_layout,
                        latent_to_structure_64, local_contrast_norm,
                        rgb_to_gray, rgb_to_oklab, rgb_to_structure_64,
                        y64_to_structure_4ch)

__all__ = [
    "DEFAULT_BATCH", "EVAL_SEED", "FeatureStack", "RawWriter", "S10_DECODER",
    "STYLES", "StudentDecoder", "auto_decode_batch", "auto_infer_batch",
    "auto_referee_batch", "auto_vram_budget_gb", "bf16_autocast", "canonical_pid",
    "clean_balanced_pool", "decode", "decode_s10", "export_raw_pngs",
    "extract_color_fiber_27d", "extract_whitened_dino_layout",
    "latent_to_structure_64", "load_s10_decoder", "load_tasks_750",
    "load_trt", "load_vae", "load_vgg16_features_offline",
    "local_contrast_norm", "pick_wrong_cross_tid", "read_json",
    "reference_bank", "rgb_to_gray", "rgb_to_oklab", "rgb_to_structure_64",
    "run_isolated_python", "sanitize_cuda_env", "save_json",
    "save_png_batch_fast", "seed_all", "s10_scaling_factor",
    "sha256_file", "tier_micro_accum", "utc", "vram_total_gb",
    "y64_to_structure_4ch",
]
