"""Portable, CPU-only configuration and frozen run manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STAGES = [("S0", 3600), ("S1", 3600), ("E0", 5400), ("E1", 5400),
          ("E2", 5400), ("P0", 5400), ("P1", 5400), ("P2", 5400),
          ("R0", 7200), ("I0", 3600), ("F0", 7200)]


def defaults() -> dict:
    return {
        "schema": 1, "seed": 20263800, "stage_budgets": dict(STAGES),
        "paths": {"output_root": "runs/stage1_recipe", "cache_root": "runs/v36_art_foundation/cache",
                  "e_initial": "runs/v37_editing/checkpoints/initial4500.pt",
                  "p_initial": None,
                  "p_candidates": ["runs/v36_art_foundation/checkpoints/v36_main_p1_final.pt",
                                   "runs/v36_art_foundation/checkpoints/v36_main_p2_final.pt"]},
        "training": {
            "train_budget_s": 4500, "min_updates": 600, "save_reserve_s": 60,
            "e": {"warmup_steps": 100, "ema_decays": [.995, .999], "batch_a": 6,
                  "batch_b": 4, "batch_c": 4, "grad_clip": 1.5, "micro": 1,
                  "checkpointing": True, "save_every": 250, "monitor_every": 250},
            "p": {"warmup_steps": 100, "ema_decays": [.999, .9995], "batch_size": 96,
                  "grad_clip": 1., "micro": 4, "checkpointing": True,
                  "content_dropout": .30, "style_dropout": .15, "y16_dropout": .40,
                  "task_ratios": None, "save_every": 250, "monitor_every": 250}},
        "arms": {
            "E0": {"a_q32": "target", "a_weight": 1., "b_style_weight": 1., "lr": 1e-4},
            "E1": {"a_q32": "proxy", "a_weight": 1., "b_style_weight": 1., "lr": 1e-4},
            "E2": {"a_q32": "proxy", "a_weight": 1., "b_style_weight": 1.25, "lr": 1e-4},
            "E3": {"a_q32": "proxy", "a_weight": .75, "b_style_weight": 1., "lr": 1e-4},
            "P0": {"lr_p": 1e-4, "lr_style": 1.5e-4, "q_noise": .05},
            "P1": {"lr_p": 5e-5, "lr_style": 7.5e-5, "q_noise": .05},
            "P2": {"lr_from": "seen_winner_P0_P1", "q_noise": .025},
            "P3": {"lr_from": "seen_winner_P0_P1", "style_lr_ratio": 2., "q_noise": .05}},
        "evaluation": {"seen_n": 300, "taus": [.35, .50, .65, .80],
                       "e_lambdas": [.40, .60, .80, 1.], "content_anchors": [.60, .65, .70, .75],
                       "steps": 6, "wj": 3.8, "wc": 1.6, "alpha_y": 1., "alpha_c": 1.,
                       "batch": 4, "bootstrap": 2000, "minimum_valid_bootstrap": 1000,
                       "s_gain": .01, "top1_gain_pp": 5., "s_noninferiority": -.005,
                       "same_noninferiority_pp": -2., "ood_2way_noninferiority_pp": -2.,
                       "formal_max_candidates": 1},
        "profile": {"micro_p": [4, 8, 16], "micro_e": [1, 2, 4],
                    "checkpointing": [False, True], "peak_gib_limit": 10.,
                    "warmup": 2, "steps": 5, "safety_factor": .90},
        "gpu": {"required_name": None, "idle_memory_mib_max": 768, "idle_util_max": 5},
        "s0": {"lpips_repair_spec": None, "lpips_device": "cpu"},
        "r0": {"mode": "auto", "seed_offset": 1},
        "prohibited": ["closed_set_retraining", "muon", "encoder_distillation", "cfg_distillation",
                       "qwen_inference", "F5", "architecture_search"],
    }


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def resolve_root(value: str | Path, root: Path = ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(root) / path


def read(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def digest(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def validate(config: dict) -> None:
    assert sum(config["stage_budgets"].values()) == 16 * 3600, "Stage budgets must sum to 16h"
    assert config["evaluation"]["steps"] == 6, "No low-NFE search in stage one"
    assert config["training"]["p"]["batch_size"] == 96
    assert [config["training"]["e"][k] for k in ("batch_a", "batch_b", "batch_c")] == [6, 4, 4]
    assert config["arms"]["E0"]["a_q32"] == "target"
    assert config["arms"]["E1"]["a_q32"] == config["arms"]["E2"]["a_q32"] == "proxy"
    assert config["evaluation"]["formal_max_candidates"] >= 1
    assert config["training"]["min_updates"] >= 600
    for family in ("e", "p"):
        assert config["training"][family]["warmup_steps"] == 100
    for key, value in config["paths"].items():
        if isinstance(value, str) and not value.strip():
            raise ValueError(f"Path cannot be empty: {key}")


def audit_root(root: Path, config: dict) -> dict:
    """Read-only asset inventory that is safe while another worker owns the GPU."""
    root = Path(root).resolve()
    at = lambda value: resolve_root(value, root)
    cache = at(config["paths"]["cache_root"])
    required = [at(config["paths"]["e_initial"]),
                *[at(p) for p in config["paths"]["p_candidates"]],
                cache / "seen_train_index.pt", cache / "v36_seen_train_priors.pt",
                cache / "v36_caches_ready.json", cache / "curation_manifest.json",
                cache / "v36_eval_assets.pt", root / "deliver/checkpoints/sd0_vae/config.json",
                root / "deliver/checkpoints/sd0_vae/diffusion_pytorch_model.safetensors",
                root.parent / "eval_cache/torch/hub/checkpoints/vgg16-397923af.pth",
                root.parent / "eval_cache/torch/hub/checkpoints/convnext_tiny-983f1562.pth",
                root / "deliver/checkpoints/runs/vae_decoder_prune/s10_p5_lowlr/decoder.pt"]
    required.extend(cache / "memmap" / name for name in (
        "seen_train_z_gt.npy", "seen_train_q32.npy", "seen_train_stat.npy",
        "seen_train_stat_crop.npy", "seen_dev_z_gt.npy", "seen_dev_q32.npy",
        "seen_dev_y16.npy", "artist_ood_z_gt.npy", "movement_ood_z_gt.npy"))
    missing = [str(p) for p in required if not p.is_file()]
    return {"root": str(root), "required_count": len(required), "missing": missing,
            "ready": not missing, "gpu_touched": False,
            "note": "Path inventory only; checkpoint headers and Python environment still require remote CPU verification."}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--write", type=Path, help="Write configuration only; never launch a worker")
    a = p.parse_args()
    c = defaults()
    validate(c)
    if a.write:
        if a.write.exists():
            raise FileExistsError(a.write)
        write(a.write, c)
    else:
        print(json.dumps(c, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
