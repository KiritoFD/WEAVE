"""CPU-only wiring check: config merge + PairedLatentDataset sampling (no GPU, no training)."""
from __future__ import annotations

import sys

sys.path.insert(0, "/mnt/i/Latent_Style/WEAVE")

from config_schema import load_experiment_config  # noqa: E402
from paths import resolve_dataset_path  # noqa: E402
from utils.dataset import PairedLatentDataset  # noqa: E402

CASES = [
    ("configs/pgt/S0_smoke.json", None),
    ("configs/pgt/S0_base.json", "base"),
    ("configs/pgt/S1_positive.json", "positive"),
    ("configs/pgt/S2_reverse.json", "reverse"),
]

ROOT = "/mnt/i/Latent_Style/WEAVE"

for cfg_rel, arm in CASES:
    cfg = load_experiment_config(f"{ROOT}/{cfg_rel}")
    d = cfg.data
    print(f"[cfg] {cfg_rel}: epochs={cfg.training.num_epochs} b={cfg.training.batch_size} "
          f"cap={cfg.training.stop_after_global_steps} es={cfg.training.internal_early_stop_enabled} "
          f"min_ep={cfg.training.internal_early_stop_min_epoch} "
          f"gate_thr={cfg.training.internal_early_stop_gate_delta_threshold} "
          f"ratio_thr={cfg.training.internal_early_stop_shared_ratio_drop_threshold} "
          f"passthrough={cfg.bridge.paired_latent_passthrough} "
          f"dir={d.paired_latent_cache_dir} styles={len(d.style_subdirs)}")
    assert cfg.bridge.paired_latent_passthrough, "passthrough must be on"
    assert d.paired_latent_cache_dir, "cache dir must be set"
    if arm is None:
        continue
    try:
        ds = PairedLatentDataset(cache_dir=str(resolve_dataset_path(d.paired_latent_cache_dir)),
                                 style_names=list(d.style_subdirs))
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[ds] {arm}: SKIP (not finished) {type(exc).__name__}: {exc}")
        continue
    print(f"[ds] {arm}: n={len(ds)}")
    for idx in (0, len(ds) // 2, len(ds) - 1):
        it = ds[idx]
        z0, z1 = it["content"], it["target_style"]
        assert tuple(z0.shape) == (4, 64, 64), z0.shape
        assert tuple(z1.shape) == (4, 64, 64), z1.shape
        same = float((z0 - z1).abs().mean())
        print(f"[ds] {arm}[{idx}] z0={tuple(z0.shape)} z1={tuple(z1.shape)} "
              f"dtype={z0.dtype}/{z1.dtype} |z0-z1|={same:.4f} "
              f"tgt_id={int(it['target_style_id'])} src_id={int(it['source_style_id'])} "
              f"pair={it['pair_sample_id']}")
print("WIRE_OK")
