"""Build minimal code.zip (<50MB): code + configs + checkpoint only.
Does not delete any project directories; only excludes them from the archive.
"""
from __future__ import annotations

import json
import re
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGE = ROOT / "_code_pack"
ZIP = ROOT / "code.zip"

CORE = [
    "model.py",
    "run.py",
    "trainer.py",
    "flow.py",
    "blocks.py",
    "style.py",
    "wavelet.py",
    "config_schema.py",
    "internal_dynamics.py",
    "style_families.py",
    "__init__.py",
    "__main__.py",
    "config.json",
    "inference.json",
    "requirements.txt",
    "pytest.ini",
]

ORIENTED = [
    "hf_oriented_internal_early_stop.json",
    "hf_oriented_nohh.json",
    "hf_oriented_internal_probe.json",
]

ABS_RE = re.compile(r"(?:[A-Za-z]:[\\/]|/mnt/)[^\s\"']+")


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_py_tree(src: Path, dst: Path) -> None:
    for path in src.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(src)
        copy_file(path, dst / rel)


def sanitize_json(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    cleaned = (
        text.replace("G:\\GitHub\\Latent_Style\\WEAVE\\", "")
        .replace("G:/GitHub/Latent_Style/WEAVE/", "")
        .replace("G:\\GitHub\\Latent_Style\\", "")
        .replace("G:/GitHub/Latent_Style/", "")
        .replace("I:/Github/Latent_Style/", "")
        .replace("/mnt/i/Github/Latent_Style/", "")
    )
    cleaned = ABS_RE.sub(lambda m: Path(m.group(0).replace("\\", "/")).name, cleaned)
    if cleaned != text:
        path.write_text(cleaned, encoding="utf-8")


def main() -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir()

    for name in CORE:
        src = ROOT / name
        if src.exists():
            copy_file(src, STAGE / name)

    for name in ORIENTED:
        src = ROOT / "experiments" / "architecture" / name
        if src.exists():
            copy_file(src, STAGE / "configs" / name)

    # Package layout: configs/ inherits package-root config.json
    nohh = STAGE / "configs" / "hf_oriented_nohh.json"
    if nohh.exists():
        obj = json.loads(nohh.read_text(encoding="utf-8"))
        obj["_base"] = "../config.json"
        nohh.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")

    if (ROOT / "utils").is_dir():
        copy_py_tree(ROOT / "utils", STAGE / "utils")
    if (ROOT / "tests").is_dir():
        copy_py_tree(ROOT / "tests", STAGE / "tests")

    ckpt = ROOT / "runs" / "submission" / "canonical_oriented_epoch4" / "epoch_0004.pt"
    if ckpt.exists():
        copy_file(ckpt, STAGE / "checkpoints" / "weave_epoch_0004.pt")

    # Lightweight docs/tools (no datasets, no ONNX binaries)
    extras = [
        (ROOT / "tools" / "prepare_submission_data.py", STAGE / "tools" / "prepare_submission_data.py"),
        (ROOT / "docs" / "model_probe" / "generation_only_timing_summary.json", STAGE / "docs" / "generation_only_timing_summary.json"),
        (ROOT / "docs" / "onnx_export_and_trt.md", STAGE / "docs" / "onnx_export_and_trt.md"),
        (ROOT / "data" / "README.md", STAGE / "data" / "README.md"),
        (ROOT / "requirements-dev.txt", STAGE / "requirements-dev.txt"),
    ]
    for src, dst in extras:
        if src.exists():
            copy_file(src, dst)

    for path in STAGE.rglob("*.json"):
        sanitize_json(path)

    cfg_path = STAGE / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        mtm = cfg.setdefault("_main_table_metrics", {})
        mtm["checkpoint"] = "checkpoints/weave_epoch_0004.pt"
        mtm["eval_protocol"] = (
            "base config stores endpoint_adain_scale=1.0; "
            "Table-1 eval uses inference.json override endpoint_adain_scale=2.0 "
            "(beta is solver-only, not in training loss)"
        )
        mtm["eval_endpoint_adain_scale"] = 2.0
        mtm["base_endpoint_adain_scale"] = 1.0
        cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    (STAGE / "README.md").write_text(
        """# WEAVE code package (anonymous)

Minimal archive: source, configs, selected **D5** checkpoint, and lightweight docs.
**Not** included: WikiArt data, packed latents, ONNX/TRT binaries, paper TeX/PDF,
or collection-specific P2A/R5 checkpoints.

## Contents
- Core modules (`model.py`, `run.py`, `trainer.py`, ...)
- `utils/`, `tests/`
- `config.json`, `inference.json`, `configs/`
- `checkpoints/weave_epoch_0004.pt` (D5 only)
- `tools/prepare_submission_data.py`
- `docs/generation_only_timing_summary.json` (73.0 s / 750 ledger)
- `docs/onnx_export_and_trt.md` (how to rebuild ONNX/TRT; graphs not shipped)
- `data/README.md` (expected local data layout only)
- `requirements.txt`, optional `requirements-dev.txt` (pytest)

## Relative paths
Run from this directory. Configs expect local:
- `data/train` — D5 latent cache
- `data/test` — D5 eval images

See `data/README.md`. Prepare data yourself.

## Train / eval (D5 Table 1)
```bash
pip install -r requirements.txt
# optional: pip install -r requirements-dev.txt && python -m pytest -q
python run.py --config configs/hf_oriented_internal_early_stop.json
python -m utils.run_evaluation \\
  --checkpoint checkpoints/weave_epoch_0004.pt \\
  --test_dir data/test \\
  --output eval_outputs/rerun \\
  --config config.json \\
  --config_override inference.json \\
  --num_steps 8
```

`endpoint_adain_scale` / β scales the analytic fiber update **at inference only**
(does not enter the training loss). Base config stores 1.0; main-table eval uses
`inference.json` override **2.0**.

R5/P2A main-table rows need collection-specific checkpoints (not in this zip).
Using the D5 checkpoint on R5 images is a zero-shot diagnostic only.

ONNX/TRT 73 s timing: rebuild graphs per `docs/onnx_export_and_trt.md` (not shipped).
""",
        encoding="utf-8",
    )

    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(STAGE.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=str(Path("WEAVE") / path.relative_to(STAGE)))

    unpacked = sum(p.stat().st_size for p in STAGE.rglob("*") if p.is_file())
    zsize = ZIP.stat().st_size
    print(f"unpacked: {unpacked/1024/1024:.2f} MB, {sum(1 for _ in STAGE.rglob('*') if _.is_file())} files")
    print(f"zip:      {zsize/1024/1024:.2f} MB -> {ZIP}")
    if zsize > 50 * 1024 * 1024:
        raise SystemExit("code.zip exceeds 50MB")
    print("OK under 50MB")


if __name__ == "__main__":
    main()
