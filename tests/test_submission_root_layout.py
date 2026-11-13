"""Layout guards after the 2026-11-14 repository reorganization.

Contract:
- Legacy paper-era flat modules live in tools/legacy_root/ and stay importable
  (same-directory imports + `python -m weave` shim).
- The v4-era utils/ package lives in tools/legacy_utils/ and stays importable
  as tools.legacy_utils.*.
- Active code: src/ (facade) + exp/ packages; pinned model classes remain in
  tools/v6*, tools/v7 (see tools/README.md).
- Canonical config.json / inference.json stay at the repo root.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")


def _walk_strings(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)
    elif isinstance(value, str):
        yield value


def test_legacy_modules_resolve_from_project_root() -> None:
    code = """
import json
from pathlib import Path
import tools.legacy_root.config_schema
import tools.legacy_root.model
import tools.legacy_root.trainer
import tools.legacy_utils.dataset

print(json.dumps({
    'config_schema': str(Path(tools.legacy_root.config_schema.__file__).resolve()),
    'model': str(Path(tools.legacy_root.model.__file__).resolve()),
    'trainer': str(Path(tools.legacy_root.trainer.__file__).resolve()),
    'dataset': str(Path(tools.legacy_utils.dataset.__file__).resolve()),
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    paths = {name: Path(path) for name, path in json.loads(result.stdout).items()}
    assert paths["config_schema"].parent == PROJECT_ROOT / "tools" / "legacy_root"
    assert paths["model"].parent == PROJECT_ROOT / "tools" / "legacy_root"
    assert paths["trainer"].parent == PROJECT_ROOT / "tools" / "legacy_root"
    assert paths["dataset"].parent == PROJECT_ROOT / "tools" / "legacy_utils"


def test_weave_shim_entry_point() -> None:
    """The root __init__/__main__ shims re-export from tools.legacy_root.

    Note: `python -m weave` only ever worked from the repo's PARENT directory
    (the package name is the directory name). What must keep working everywhere
    is the shim's import surface itself.
    """
    code = """
from tools.legacy_root.model import WEAVE, build_model_from_config
from tools.legacy_root.run import main

assert WEAVE is not None
assert callable(build_model_from_config)
assert callable(main)
print('ok')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_legacy_training_entry_point_has_help() -> None:
    result = subprocess.run(
        [sys.executable, "tools/legacy_root/run.py", "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout


def test_pinned_tools_subpackages_import() -> None:
    code = """
import tools.v6.models_v6
import tools.v7.models_v7
import tools.v6_next.probe_style
import tools.v7r.v7r_common
print('ok')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_s10_contract_import_surface() -> None:
    code = """
from src.utils import load_s10_decoder, decode_s10
from src.utils.vae import S10_DECODER

assert S10_DECODER.name == 'decoder.pt'
print('ok')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_canonical_configs_are_portable_and_complete() -> None:
    sys.path.insert(0, str(PROJECT_ROOT))
    from tools.legacy_root.config_schema import (
        load_config,
        load_experiment_config,
        merge_config_dicts,
        resolve_full_eval_section,
        resolve_inference_section,
    )
    from tools.legacy_root.model import build_model_from_config, count_parameters

    config_path = PROJECT_ROOT / "config.json"
    inference_path = PROJECT_ROOT / "inference.json"
    config = load_experiment_config(config_path)
    inference = load_config(inference_path)
    merged = merge_config_dicts(config.to_dict(), inference)
    inference_settings = resolve_inference_section(merged)
    eval_settings = resolve_full_eval_section(merged)

    assert count_parameters(build_model_from_config(config.model, bridge_cfg=config.bridge)) == 873_680
    assert inference_settings == {"num_steps": 8, "step_size": 1.0, "style_strength": 1.0}
    assert eval_settings["num_steps"] == 8
    assert eval_settings["batch_size"] == 16
    assert eval_settings["max_ref_compare"] == 16
    assert eval_settings["max_ref_cache"] == 16
    assert eval_settings["target_chunk_size"] == 1
    assert eval_settings["save_generated_images"] is True
    assert merged["model"]["endpoint_adain_scale"] == 2.0
    assert config.training.freeze_mode == "none"
    assert config.training.resume_checkpoint == ""
    assert merged["checkpoint"]["resume_checkpoint"] == ""
    assert config.model.output_appearance_alignment_mode == "none"
    assert eval_settings["postprocess_mode"] == "none"
    assert eval_settings["latent_postprocess_mode"] == "none"
    assert eval_settings["allow_metric_postprocess"] is False

    for path in (config_path, inference_path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        bad_paths = [
            value for value in _walk_strings(payload)
            if WINDOWS_ABSOLUTE_PATH.match(value) or value.startswith(("/mnt/", "\\\\"))
        ]
        assert not bad_paths, f"absolute paths in {path.name}: {bad_paths}"
        assert all(not value.startswith("$index:") for value in _walk_strings(payload))

    assert not (PROJECT_ROOT / "dataset_index.json").exists()


def test_submission_launchers_do_not_use_src_entry_points() -> None:
    launcher = (PROJECT_ROOT / "scripts" / "run_submission_repro.ps1").read_text(encoding="utf-8")
    evaluator = (PROJECT_ROOT / "scripts" / "batch_eval_all.py").read_text(encoding="utf-8")

    assert "src\\run.py" not in launcher
    assert "src\\default_config.json" not in launcher
    assert ' / "src" / "utils" ' not in evaluator


def test_epoch_selection_parser() -> None:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from batch_eval_all import parse_epoch_selection

    assert parse_epoch_selection("6") == {6}
    assert parse_epoch_selection("1,3-5,8") == {1, 3, 4, 5, 8}
