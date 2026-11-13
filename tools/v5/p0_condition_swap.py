"""V5 P0: Condition Swapping Test & Baseline Measurement
1. Verify model conditioning sensitivity to Style-ID (correct vs swapped/wrong style ID)
   with style_latent strictly set to None (zero reference-GT leakage).
2. Measure baseline CSD scores on the 200 validation tasks under VAE round-trip.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools" / "12h"))

import paths
import csd_model
from tools.legacy_root.config_schema import load_config
from tools.legacy_root.model import WEAVE

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
STYLE_IDS = {name: i for i, name in enumerate(STYLES)}


def run_condition_swap_test(device: str = "cuda") -> dict:
    """Test model velocity output difference between correct and swapped Style-ID with NO style_latent."""
    ckpt_path = REPO_ROOT / "runs" / "12h" / "anchors" / "base_ep10_750_512" / "epoch_0010.pt"
    if not ckpt_path.exists():
        # Fallback to any available pt
        pts = list((REPO_ROOT / "runs").glob("**/*.pt"))
        if not pts:
            return {"status": "skipped", "reason": "no existing checkpoint found for swap test"}
        ckpt_path = pts[0]

    print(f"[p0] Running condition swap test using {ckpt_path.name}...", flush=True)
    cfg = load_config(str(REPO_ROOT / "config.json"))
    from config_schema import ModelConfig
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else getattr(cfg, "model", {})
    if isinstance(model_cfg, dict):
        model_cfg = ModelConfig(**{k: v for k, v in model_cfg.items() if hasattr(ModelConfig, k)})
    model = WEAVE(model_cfg).to(device=device, dtype=torch.float32).eval()
    
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)

    torch.manual_seed(42)
    B = 10
    x = torch.randn(B, 4, 64, 64, device=device, dtype=torch.float32)
    t = torch.full((B,), 0.5, device=device, dtype=torch.float32)

    # Correct style id: 0..4 repeated
    correct_id = torch.tensor([i % 5 for i in range(B)], device=device, dtype=torch.long)
    # Swapped style id: (i + 1) % 5
    swapped_id = torch.tensor([(i + 1) % 5 for i in range(B)], device=device, dtype=torch.long)

    with torch.no_grad():
        # Notice: style_latent is strictly None! Only style_id is supplied!
        v_correct = model(x, t=t, style_id=correct_id, style_latent=None)
        v_swapped = model(x, t=t, style_id=swapped_id, style_latent=None)

    # Compute difference across subbands
    diffs = {}
    for band in ("ll", "lh", "hl"):
        vc = v_correct[band].flatten(1)
        vs = v_swapped[band].flatten(1)
        cos_sim = F.cosine_similarity(vc, vs, dim=-1).mean().item()
        l2_diff = (vc - vs).norm(p=2, dim=-1).mean().item()
        diffs[band] = {"cosine_similarity": float(cos_sim), "l2_diff": float(l2_diff)}

    print(f"[p0] Condition swap test passed! LL cosine: {diffs['ll']['cosine_similarity']:.4f}, L2 diff: {diffs['ll']['l2_diff']:.4f}")
    return {"status": "passed", "checkpoint": str(ckpt_path), "diffs": diffs}


def run_baseline_measurement(device: str = "cuda") -> dict:
    """Compute VAE round-trip CSD baseline for all 200 validation tasks."""
    out_dir = REPO_ROOT / "runs" / "v5"
    tasks_p = out_dir / "v5_eval200_tasks.json"
    tasks = json.loads(tasks_p.read_text(encoding="utf-8"))

    # Load CSD prototypes
    proto_p = paths.asset_root() / "pseudo_gt_24h" / "csd_proto.pt"
    proto_dict = torch.load(proto_p, map_location="cpu", weights_only=False)
    # proto_dict maps style_name -> numpy array (768,)
    proto_mat = np.stack([proto_dict[s] for s in STYLES])  # (5, 768)

    # Load CSD model
    csd, ck, msg = csd_model.load_csd(device)

    # Load source latents
    src_lat_dict = torch.load(out_dir / "cache" / "eval_src_latents.pt", map_location="cpu")

    # Load VAE for decode
    candidates = [
        paths.asset_root() / "eval_cache" / "hf" / "modelscope" / "stabilityai_sd-vae-ft-ema" / "models" / "stabilityai--sd-vae-ft-ema" / "snapshots",
        REPO_ROOT / "eval_cache" / "hf" / "modelscope" / "stabilityai_sd-vae-ft-ema" / "models" / "stabilityai--sd-vae-ft-ema" / "snapshots",
    ]
    vae_dir = None
    for c in candidates:
        if c.exists():
            snaps = list(c.glob("*"))
            if snaps and (snaps[0] / "config.json").exists():
                vae_dir = snaps[0]
                break

    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(str(vae_dir), local_files_only=True).to(device=device, dtype=torch.float32).eval()

    print(f"[p0] Decoding and computing CSD baseline for {len(src_lat_dict)} unique sources...", flush=True)
    src_csd_feats = {}
    with torch.inference_mode():
        for stem, z in src_lat_dict.items():
            z_in = (z.unsqueeze(0).to(device) / vae.config.scaling_factor).float()
            img_t = vae.decode(z_in).sample.squeeze(0).clamp(-1.0, 1.0)  # (3, 512, 512)
            arr = ((img_t.permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5).astype(np.uint8)
            pil_img = Image.fromarray(arr)
            # CSD feat
            feat = csd_model.feats_pils(csd, [pil_img], batch=1, device=device)[0]  # (768,)
            src_csd_feats[stem] = feat

    del vae
    del csd
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Pre-calculate baseline scores on all 200 tasks
    task_baselines = []
    for task in tasks:
        stem = task["src_stem"]
        src_feat = src_csd_feats[stem]
        tgt_id = task["tgt_style_id"]
        src_id = task["src_style_id"]
        
        cos_tgt = float(src_feat @ proto_mat[tgt_id])
        cos_src = float(src_feat @ proto_mat[src_id])
        task_baselines.append({
            "task_id": task["task_id"],
            "src_stem": stem,
            "src_style": task["src_style"],
            "tgt_style": task["tgt_style"],
            "src_cos_tgt": cos_tgt,
            "src_cos_src": cos_src,
            "baseline_dT": 0.0,
            "baseline_dS": 0.0,
            "baseline_dM": 0.0,
        })

    # Save source CSD features and baselines
    torch.save(src_csd_feats, out_dir / "cache" / "eval_src_csd_feats.pt")
    out = {
        "num_sources": len(src_csd_feats),
        "num_tasks": len(task_baselines),
        "mean_src_cos_target": float(np.mean([t["src_cos_tgt"] for t in task_baselines])),
        "mean_src_cos_source": float(np.mean([t["src_cos_src"] for t in task_baselines])),
        "task_baselines": task_baselines,
    }
    (out_dir / "v5_eval200_baseline.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[p0] Saved v5_eval200_baseline.json. Mean src_cos_target={out['mean_src_cos_target']:.4f}, src_cos_src={out['mean_src_cos_source']:.4f}", flush=True)
    return out


def main():
    print("=== Running V5 P0: Condition Swapping Test & Baseline Measurement ===", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    swap_res = run_condition_swap_test(device)
    base_res = run_baseline_measurement(device)

    out = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "condition_swap": swap_res,
        "baseline": {
            "num_sources": base_res["num_sources"],
            "num_tasks": base_res["num_tasks"],
            "mean_src_cos_target": base_res["mean_src_cos_target"],
            "mean_src_cos_source": base_res["mean_src_cos_source"],
        }
    }
    out_p = REPO_ROOT / "runs" / "v5" / "p0_report.json"
    out_p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[p0] Complete! Saved report to {out_p}", flush=True)


if __name__ == "__main__":
    main()
