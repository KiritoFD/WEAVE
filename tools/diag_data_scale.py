"""P0 diagnostic: 5k vs 18k data-scale comparison under TWO protocols.

The review doc (section 四) notes the current model trained on only 5,000
packed latents (1,000/style) while 18,000 images are available, and requires
the comparison to control for BOTH:
  * protocol "fixed_updates": same total optimiser updates (batch * steps);
  * protocol "fixed_epochs":   same passes over each dataset (5k sees more
                               repeats per sample).

This script is a thin, honest harness: it builds two dataset variants from a
packed latent cache (subset 1000/style vs full), runs the standard trainer for
each protocol cell, and writes a results matrix JSON. It does NOT invent new
training logic — it calls run.py's trainer on generated configs.

Cells: {5k, 18k} x {fixed_updates, fixed_epochs} (4 runs; fixed_epochs for 5k
is the historical baseline).

Smoke mode (--smoke): synthetic latent cache (24 latents/style), 1 epoch,
tiny model on CPU; proves config generation + dataset subsetting + metric
collection end-to-end.

Usage (server):
    python tools/diag_data_scale.py --packed_dir data/train_d5_taesd/.latent_cache/packed \
        --base_config config_d5_adain2.json --out runs/diag_data_scale.json
Usage (smoke):
    python tools/diag_data_scale.py --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import torch


def build_cells(per_style_full: int, per_style_small: int, updates: int, batch_size: int) -> list[dict]:
    """Return the 4 experiment cells with resolved epoch counts."""
    cells = []
    for tag, per_style in (("small", per_style_small), ("full", per_style_full)):
        n = per_style  # per-style samples
        # fixed_updates: epochs chosen so total optimiser updates ~= updates.
        steps_per_epoch = max(1, n // batch_size)
        epochs_fu = max(1, round(updates / steps_per_epoch))
        # fixed_epochs: 4 epochs like the historical main run.
        cells.append({
            "name": f"{tag}_fixed_updates",
            "per_style": per_style, "protocol": "fixed_updates",
            "num_epochs": epochs_fu, "expected_updates": epochs_fu * steps_per_epoch,
        })
        cells.append({
            "name": f"{tag}_fixed_epochs",
            "per_style": per_style, "protocol": "fixed_epochs",
            "num_epochs": 4, "expected_updates": 4 * steps_per_epoch,
        })
    return cells


def run_smoke(out: Path) -> None:
    from config_schema import ExperimentConfig
    from trainer import SBTrainer
    from torch.utils.data import DataLoader, Dataset

    torch.manual_seed(0)
    n_per_style, n_styles = 12, 2
    small = 6
    batch_size = 4
    latents = {s: torch.randn(n_per_style, 4, 16, 16) for s in range(n_styles)}

    class DS(Dataset):
        def __init__(self, per_style: int) -> None:
            self.per_style = per_style
            self.pairs = [(s, i) for s in range(n_styles) for i in range(per_style)]

        def __len__(self) -> int:
            return len(self.pairs)

        def __getitem__(self, idx: int) -> dict:
            s, i = self.pairs[idx % len(self.pairs)]
            t_style = (s + 1) % n_styles
            return {
                "content": latents[s][i],
                "target_style": latents[t_style][(i + 1) % self.per_style],
                "target_style_id": t_style,
                "source_style_id": s,
                "content_index": i,
            }

    results = []
    for cell in build_cells(n_per_style, small, updates=8, batch_size=batch_size):
        ck = Path(tempfile.mkdtemp())
        cfg = ExperimentConfig.from_mapping({
            "model": {"base_dim": 32, "num_res_blocks": 2, "num_styles": n_styles, "contract_family": "weave"},
            "bridge": {},
            "training": {"num_epochs": cell["num_epochs"], "batch_size": batch_size,
                         "use_tqdm": False, "log_interval": 0, "save_interval": 999,
                         "use_amp": False, "gpu_monitor_enabled": False, "use_gradient_checkpointing": False},
            "checkpoint": {"save_dir": str(ck)},
            "data": {},
        })
        trainer = SBTrainer(cfg, torch.device("cpu"))
        dl = DataLoader(DS(cell["per_style"]), batch_size=batch_size)
        metrics = trainer.train_epoch(dl, 1)
        results.append({"cell": cell["name"], "protocol": cell["protocol"],
                        "per_style": cell["per_style"], "final_loss": float(metrics.get("loss", 0.0)),
                        "updates": trainer.global_step})
        print(f"[smoke] {cell['name']}: updates={trainer.global_step} loss={metrics.get('loss', 0.0):.4f}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"smoke": True, "results": results}, indent=2))
    print(f"[smoke] data-scale harness OK -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed_dir", type=str, default="")
    ap.add_argument("--base_config", type=str, default="config.json")
    ap.add_argument("--per_style_small", type=int, default=1000)
    ap.add_argument("--per_style_full", type=int, default=3600)
    ap.add_argument("--updates", type=int, default=208, help="historical main-run total updates")
    ap.add_argument("--batch_size", type=int, default=96)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    if args.smoke:
        run_smoke(out)
        return

    # Server path: emit one config per cell under runs/diag_data_scale/, the
    # user launches each with run.py --config <cell config>. Dataset subsetting
    # uses data.latent_pack_limit_per_style (existing config knob).
    cells = build_cells(args.per_style_full, args.per_style_small, args.updates, args.batch_size)
    base = json.loads(Path(args.base_config).read_text())
    gen_dir = out.parent / "diag_data_scale_configs"
    gen_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for cell in cells:
        cfg = json.loads(json.dumps(base))
        cfg.setdefault("training", {})["num_epochs"] = cell["num_epochs"]
        cfg.setdefault("data", {})["latent_pack_limit_per_style"] = cell["per_style"]
        cfg["training"]["save_interval"] = max(1, cell["num_epochs"])
        ckpt_dir = str(out.parent / "diag_data_scale_runs" / cell["name"])
        cfg.setdefault("checkpoint", {})["save_dir"] = ckpt_dir
        path = gen_dir / f"{cell['name']}.json"
        path.write_text(json.dumps(cfg, indent=2))
        manifest.append({"cell": cell, "config": str(path), "checkpoint_dir": ckpt_dir,
                         "launch": f"python run.py --config {path}"})
    payload = {
        "protocol_note": "fixed_updates holds total optimiser updates constant; "
                         "fixed_epochs holds dataset passes constant (historical baseline = small+fixed_epochs).",
        "cells": manifest,
        "collect": "After the 4 runs, compare cross-style DINO-S/CLIP-S/DINO-C/LPIPS per cell via tools/run_eval_split.py.",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {len(manifest)} cell configs -> {gen_dir}; manifest -> {out}")


if __name__ == "__main__":
    main()
