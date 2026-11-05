"""Guard: one canonical eval protocol, no competing DINO-S entry points."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import eval_protocol as ep  # noqa: E402

SKIP_DIRS = ("archives/", "archive/", "tests/", "EXPERIMENT_ARCHAEOLOGY/", "docs/")


def _iter_py():
    for p in ROOT.rglob("*.py"):
        rel = p.relative_to(ROOT).as_posix()
        if rel.startswith(SKIP_DIRS):
            continue
        if "__pycache__" in rel:
            continue
        yield rel, p


def test_canonical_constants():
    assert ep.IMAGE_SIZE == 512
    assert ep.DINO_MODEL == "facebook/dinov2-small"
    assert ep.CLIP_MODEL == "openai/clip-vit-base-patch32"
    assert ep.ODE_STEPS == 8
    assert ep.STEP_SIZE == 1.0
    assert ep.ADAIN_SCALE == 2.0
    assert ep.DECODER == "trt"
    assert ep.DECODER_ENGINE == "vae_decoder_512_b8_fp16"
    assert ep.REFERENCE_POLICY == "first_image_of_target_style_test_dir"
    assert ep.EVAL_BATCH == 8
    assert ep.SPLIT == "all-pairs"
    assert abs(ep.ANCHOR_DINO_S - 0.49023) < 1e-5


def test_eval_fast_pins_canonical_values():
    src = (ROOT / "tools" / "eval_fast.py").read_text(encoding="utf-8")
    assert "args.adain_scale = 2.0" in src
    assert 'args.split = "all-pairs"' in src
    assert re.search(r'"--image-size", type=int, default=512', src)
    assert 'default="facebook/dinov2-small"' in src


def test_no_competing_image_size_defaults():
    offenders = []
    for rel, p in _iter_py():
        txt = p.read_text(encoding="utf-8", errors="ignore")
        for m in re.finditer(r'"--image-size"\s*,\s*type=int\s*,\s*default=(\d+)', txt):
            if int(m.group(1)) != ep.IMAGE_SIZE:
                offenders.append((rel, m.group(1)))
    assert offenders == [], f"non-canonical --image-size defaults: {offenders}"


def test_no_competing_dino_backbone_defaults():
    offenders = []
    for rel, p in _iter_py():
        txt = p.read_text(encoding="utf-8", errors="ignore")
        for m in re.finditer(r'"--dino-model"\s*,\s*default="([^"]+)"', txt):
            if m.group(1) != ep.DINO_MODEL:
                offenders.append((rel, m.group(1)))
    assert offenders == [], f"non-canonical --dino-model defaults: {offenders}"


def test_only_one_clip_backbone_in_eval_entries():
    offenders = []
    for rel, p in _iter_py():
        if 'add_argument("--checkpoint"' not in p.read_text(encoding="utf-8", errors="ignore"):
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        for m in re.finditer(r'"--clip-model"\s*,\s*default="([^"]*)"', txt):
            if m.group(1) and m.group(1) != ep.CLIP_MODEL:
                offenders.append((rel, m.group(1)))
    assert offenders == [], f"non-canonical --clip-model defaults: {offenders}"
