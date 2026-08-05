"""Create WEAVE R5 training config from the canonical D5 config.json.

Changes:
  - data.data_root: "data/train" -> "data/train_r5"
  - data.style_subdirs: D5 styles -> R5 styles
  - data.pairing_cache_path: clear (no pre-computed pairing for R5)
  - training.batch_size: 96 -> 8 (RTX 3060 12GB safe)
  - training.full_eval_each_epoch: true (evaluate every epoch)
  - training.test_image_dir: point to R5 test set
  - checkpoint.save_dir: runs/r5/weave_r5
  - ablation: R5 info
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
src = ROOT / "config.json"
dst = ROOT / "config_r5.json"

cfg = json.loads(src.read_text(encoding="utf-8"))

# Data: switch to R5
cfg["data"]["data_root"] = "data/train_r5"
cfg["data"]["style_subdirs"] = ["Cubism", "Expressionism", "Pop_Art", "Romanticism", "Symbolism"]
cfg["data"]["pairing_cache_path"] = ""  # no pre-computed pairing for R5
# latent_cache_mode and latent_cache_dir stay the same ("packed" / ".latent_cache/packed")

# Training: reduce batch_size for RTX 3060 12GB
cfg["training"]["batch_size"] = 8
cfg["training"]["num_epochs"] = 15  # keep 15
cfg["training"]["full_eval_each_epoch"] = True
cfg["training"]["test_image_dir"] = "datasets/r5/test"  # R5 test set relative to WEAVE root

# Checkpoint: new save dir for R5
cfg["checkpoint"]["save_dir"] = "runs/r5/weave_r5"

# Ablation metadata
cfg["ablation"] = {
    "name": "weave_r5_15ep",
    "axis": "r5_retrain",
    "stage": "r5_baselines",
    "notes": "WEAVE retrained on R5 (5 styles x 1000 train images), 15 epochs, batch_size=8."
}

dst.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"R5 config written: {dst}")
print(f"  data_root: {cfg['data']['data_root']}")
print(f"  style_subdirs: {cfg['data']['style_subdirs']}")
print(f"  batch_size: {cfg['training']['batch_size']}")
print(f"  num_epochs: {cfg['training']['num_epochs']}")
print(f"  test_image_dir: {cfg['training']['test_image_dir']}")
print(f"  checkpoint_dir: {cfg['checkpoint']['save_dir']}")
