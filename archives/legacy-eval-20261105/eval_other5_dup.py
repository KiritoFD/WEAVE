"""Evaluate a WEAVE checkpoint on the 750-pair Other5 board.

The registered / full-FT checkpoints are num_styles=10 (D5 rows 0-4 + Other5 rows
5-9). For a clean, exact 750-pair (Other5 x Other5) eval we reshape them into a
num_styles=5 eval checkpoint where Other5 memory occupies rows 0-4, then evaluate
with style_subdirs=[5 Other5] (exactly 5x5x30 = 750 pairs, no wasted D5 targets).

Usage:
  python scripts/eval_other5.py --checkpoint <ckpt> --out <out_dir> --num_steps 8
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import os
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
OTHER5_TEST = Path(r"G:\GitHub\Latent_Style\Dataset\other5_512\test")
OTHER5_STYLES = [
    "Abstract_Expressionism", "Art_Nouveau_Modern", "Cubism",
    "Expressionism", "Symbolism",
]
D5_CKPT = ROOT / "archive" / "runs_submission" / "canonical_oriented_epoch4" / "epoch_0004.pt"
D5_CONFIG = ROOT / "archive" / "exp" / "other5_reg" / "d5_config.json"
CONFIG_OVERRIDE = {"model": {"num_styles": 5}}
CONFIG_OVERRIDE_PATH = ROOT / "archive" / "exp" / "other5_reg" / "eval_config_override.json"

sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from config_schema import ExperimentConfig, ModelConfig, BridgeConfig  # noqa: E402
from model import build_model_from_config  # noqa: E402


def run(cmd):
    print("+", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, cwd=str(ROOT))


def load_config(num_styles: int):
    raw = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    mc = ModelConfig.from_mapping(raw["model"])
    bc = BridgeConfig.from_mapping(raw.get("bridge", {}))
    mc.num_styles = num_styles
    mc.target_latent_hf_subband_fusion_enabled = True
    return ExperimentConfig(model=mc, bridge=bc)


def reshape_to_ns5(reg_ckpt: Path, out_dir: Path) -> Path:
    """Extract Other5 memory (rows 5-9) into a num_styles=5 eval checkpoint."""
    reg = torch.load(reg_ckpt, map_location="cpu", weights_only=False)
    sd = reg["model_state_dict"]
    if int(sd["style_conditioner.style_memory"].shape[0]) == 5:
        return reg_ckpt  # already num_styles=5
    exp = load_config(5)
    model = build_model_from_config(exp.model, bridge_cfg=exp.bridge)
    d5 = torch.load(D5_CKPT, map_location="cpu", weights_only=False)["model_state_dict"]
    new_sd = {k: v for k, v in d5.items() if k != "style_conditioner.style_memory"}
    new_sd["style_conditioner.style_memory"] = sd["style_conditioner.style_memory"][5:10].clone()
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print(f"[reshape] missing={len(missing)} unexpected={len(unexpected)}")
    out = out_dir / "eval_ns5.pt"
    # Embed the D5 config so run_evaluation.py builds the exact architecture
    # (subband route + objective_mode=flow_matching) instead of an empty config.
    d5_cfg = json.loads(D5_CONFIG.read_text(encoding="utf-8"))
    torch.save({"model_state_dict": model.state_dict(), "num_styles": 5,
                "other5_styles": OTHER5_STYLES, "config": d5_cfg}, out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num_steps", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=4)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    CONFIG_OVERRIDE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_OVERRIDE_PATH.write_text(json.dumps(CONFIG_OVERRIDE), encoding="utf-8")

    ckpt = reshape_to_ns5(Path(args.checkpoint), out)

    # 1) Generate the exact 750-pair Other5 x Other5 board.
    run([
        sys.executable, "utils/run_evaluation.py",
        "--checkpoint", str(ckpt),
        "--config", str(D5_CONFIG),
        "--config_override", str(CONFIG_OVERRIDE_PATH),
        "--style_subdirs", ",".join(OTHER5_STYLES),
        "--test_dir", str(OTHER5_TEST),
        "--output", str(out / "board"),
        "--num_steps", str(args.num_steps),
        "--batch_size", str(args.batch_size),
        "--clip_allow_network",
    ])

    # 2) compute_dino_metrics reads metrics.csv + test_dir refs -> DINO-S/DINO-C.
    run([
        sys.executable, "utils/compute_dino_metrics.py",
        "--eval_dir", str(out / "board"),
        "--test_dir", str(OTHER5_TEST),
        "--allow_network",
        "--output_json", str(out / "dino_summary.json"),
    ])
    summary = json.loads((out / "dino_summary.json").read_text(encoding="utf-8"))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
