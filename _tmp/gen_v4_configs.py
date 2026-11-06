"""生成 V4 四条训练臂配置（统一协议：b256 / 2000 步余弦 / 无桥噪声 / 无 AdaIN / τ 均匀）。"""
import json
from pathlib import Path

ROOT = Path(r"M:\WEAVE")
OUT = ROOT / "configs" / "v4"
OUT.mkdir(parents=True, exist_ok=True)

ARMS = [
    ("S1_A_P5", "positive", 0.22, "A_P5 4×128 25% Hard（正向：源→拼贴伪GT，标签=供体风格）", 600),
    ("S2_E7", "reverse", 0.22, "E7 4×128 25% Hard（反向：拼贴输入→真实图，标签=原图风格）", 600),
    ("S3_E3", "reverse_e3", 0.22, "E3 4×128 25% Feather（与 S2 同一基础配对集）", 600),
    ("S4a_E1", "reverse_e1", 0.22, "E1 16×64 25% Feather（S4 的 64px 对照；E8 因 HH 不可达被否）", 600),
]

for name, cache, mincov, note, stop_steps in ARMS:
    cfg = {
        "_comment": f"WEAVE V4 {name}: effective batch 256 (32×8), 2000-step cosine (num_epochs=20 → 2042 global steps), ckpt 200/400/600/1000/1500/2000.",
        "_base": "../../config.json",
        "model": {
            "target_latent_hf_subband_fusion_enabled": True,
            "target_latent_hf_head_fusion_gate_init": 0.05,
        },
        "bridge": {
            "paired_latent_passthrough": True,
            "bridge_sigma": 0.0,
            "t_sampling_mode": "uniform",
        },
        "data": {
            "train_sample_limit": 0,
            "paired_latent_cache_dir": f"@asset/pseudo_gt_24h/pairs/{cache}",
            "paired_min_coverage": mincov,
            "content_style_sampling_weights": [1.0, 1.0, 1.0, 1.0, 1.0],
            "target_style_sampling_weights": [1.0, 1.0, 1.0, 1.0, 1.0],
        },
        "training": {
            "batch_size": 32,
            "accumulation_steps": 8,
            "num_epochs": 25,
            "save_interval": 1,
            "save_step_milestones": [200, 400, 600, 1000, 1500, 2000],
            "stop_after_global_steps": stop_steps,
            "full_eval_each_epoch": False,
            "internal_probe_enabled": True,
            "internal_probe_every": 1,
            "internal_probe_batch_size": 4,
            "internal_probe_fixed_t": 0.5,
            "internal_early_stop_enabled": True,
            "internal_early_stop_min_epoch": 3,
            "internal_early_stop_gate_delta_threshold": 0.0,
            "internal_early_stop_shared_ratio_drop_threshold": 0.65,
            "internal_early_stop_min_steps": 0,
        },
        "checkpoint": {"save_dir": f"runs/v4/{name}"},
        "ablation": {"name": name, "axis": "pseudo_gt_pairing_v4", "stage": "v4_round", "notes": note},
    }
    (OUT / f"{name}.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", OUT / f"{name}.json")
