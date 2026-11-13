"""Canonical evaluation protocol — the only source of comparable DINO-S numbers.

Provenance (verified 2026-11-05, re-run reproducing 0.49023):
- docs/12h/PROGRESS.md:5       protocol definition (750 all-pairs, 512 px, TRT 512 decoder, batch 8)
- docs/12h/PROGRESS.md:7       anchor numbers (dino_s .49023 / dino_c .87962 / clip_s .70463 / lpips .26203)
- runs/12h/anchors/base_ep10_750_512/eval_fast_meta.json   settings actually used by the anchor run
- runs/12h/anchors/base_ep10_750_512/dino_summary.json     numbers the anchor produced
- tools/eval_fast.py:718       AdaIN scale hard-coded to 2.0
- tools/eval_fast.py:719-721   split forced to all-pairs

Anything produced by another image size (256/384), another backbone
(CLIP-L/14, dinov2-base/large), another decoder (torch instead of TRT) or
another batch is NOT comparable to the anchor and must not be published.
"""

# --- canonical constants -------------------------------------------------
ENTRY_SCRIPT = "tools/eval_fast.py"
PROTOCOL = "paper"
IMAGE_SIZE = 512
DINO_MODEL = "facebook/dinov2-small"
CLIP_MODEL = "openai/clip-vit-base-patch32"
ODE_STEPS = 8
STEP_SIZE = 1.0
STYLE_STRENGTH = 1.0
ADAIN_SCALE = 2.0
EVAL_BATCH = 8
METRIC_BATCH = 32
DECODE_BATCH = 8
WORKERS = 4
DECODER = "trt"
DECODER_ENGINE = "vae_decoder_512_b8_fp16"
REFERENCE_POLICY = "first_image_of_target_style_test_dir"
REF_POOL = "first"
SEED = 0
VAE_MODEL = "ema"
VAE_DTYPE = "fp32"
GEN_DTYPE = "fp32"
ENCODE_BATCH = 1
LPIPS_NET = "alex"
MAX_SRC_SAMPLES = 30
MAX_REF_CACHE = 16
DINO_MAX_REFS = 30
SPLIT = "all-pairs"

# Anchor produced by exp/dino_s_break/brk_a_ll03_10ep/epoch_0010.pt on the 750 split.
ANCHOR = {
    "dino_s": 0.4902256206522385,
    "dino_c": 0.8796243784427643,
    "clip_s": 0.7046319619019826,
    "lpips": 0.2620316799134016,
}
ANCHOR_DINO_S = ANCHOR["dino_s"]
ANCHOR_TOL = 0.005


def canonical_meta() -> dict:
    """The subset of eval_fast metadata that defines comparability."""
    return {
        "protocol": PROTOCOL,
        "image_size": IMAGE_SIZE,
        "clip_model": CLIP_MODEL,
        "dino_model": DINO_MODEL,
        "split": SPLIT,
        "steps": ODE_STEPS,
        "adain": ADAIN_SCALE,
        "eval_batch": EVAL_BATCH,
        "metric_batch": METRIC_BATCH,
        "decode_batch": DECODE_BATCH,
        "decoder": DECODER,
        "reference_policy": REFERENCE_POLICY,
        "seed": SEED,
    }


def is_canonical_meta(meta: dict) -> bool:
    """True when an eval_fast_meta.json describes the canonical protocol."""
    want = canonical_meta()
    for key, value in want.items():
        if meta.get(key) != value:
            return False
    return True


def anchor_reproduced(dino_s: float, tol: float = ANCHOR_TOL) -> bool:
    return abs(float(dino_s) - ANCHOR_DINO_S) <= tol


def canonical_eval_argv(checkpoint: str, output: str, *, test_dir: str = "",
                        max_src_samples: int = -1, batch: int = EVAL_BATCH,
                        cache_dir: str = "/mnt/i/Latent_Style/eval_cache",
                        engine_dir: str = "/mnt/i/Latent_Style/WEAVE/runs/trt/engines",
                        backbone: str = "base_ep10_512") -> list:
    """CLI flags for the canonical chain; use with ENTRY_SCRIPT.

    ``max_src_samples`` / ``batch`` may only be narrowed for reduced validation
    sets (e.g. the 75-pair screen); the encoder and preprocessing stay identical.
    """
    argv = [
        "--checkpoint", checkpoint,
        "--output", output,
        "--protocol", PROTOCOL,
        "--backbone", backbone,
        "--image-size", str(IMAGE_SIZE),
        "--batch", str(batch),
        "--metric-batch", str(METRIC_BATCH),
        "--decode-batch", str(DECODE_BATCH),
        "--num-steps", str(ODE_STEPS),
        "--step-size", str(STEP_SIZE),
        "--style-strength", str(STYLE_STRENGTH),
        "--ref-pool", REF_POOL,
        "--reference-policy", REFERENCE_POLICY,
        "--max-src-samples", str(MAX_SRC_SAMPLES if max_src_samples < 0 else max_src_samples),
        "--max-ref-cache", str(MAX_REF_CACHE),
        "--dino-max-refs", str(DINO_MAX_REFS),
        "--seed", str(SEED),
        "--vae-model", VAE_MODEL,
        "--vae-dtype", VAE_DTYPE,
        "--gen-dtype", GEN_DTYPE,
        "--encode-batch", str(ENCODE_BATCH),
        "--decoder", DECODER,
        "--engine-dir", engine_dir,
        "--dino-model", DINO_MODEL,
        "--clip-model", CLIP_MODEL,
        "--cache-dir", cache_dir,
        "--lpips-net", LPIPS_NET,
        "--workers", str(WORKERS),
    ]
    if test_dir:
        argv += ["--test-dir", test_dir]
    return argv
