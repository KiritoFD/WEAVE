"""过拟合检验: 训练配对 vs 未见配对 vs 全新内容 (固定 t=0.5, 固定 noise).

三组都用同一条监督管线:
  endpoint = objective._build_training_target(content, target, target)
  u        = dwt(endpoint - content)
  x_t      = (1-t)*content + t*endpoint - sigma*sqrt(t(1-t))*noise   (subtractive)
  v        = model(x_t, t, style_id, style_latent=target)

noise 由固定 seed 生成并按 (group, index) 复用, 因此不同 ckpt 之间逐样本可比.
每个 batch 只包含单一 style_id (模型的条件是标量 style_id).

用法:
  python3 tools/12h/probe_overfit.py --config configs/12h/A1.json \
      --checkpoints runs/12h/A1/step_000200.pt runs/12h/A1/step_002000.pt \
      --device cpu --out runs/12h/probe_overfit/A1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import diagnose_saturation as ds  # noqa: E402
from tools.legacy_root.config_schema import load_experiment_config  # noqa: E402
from tools.legacy_root.model import build_model_from_config  # noqa: E402
from tools.legacy_utils.dataset import AdaCUTLatentDataset  # noqa: E402

SUBBANDS = ("ll", "lh", "hl")
FIXED_T = 0.5
NOISE_SEED = 1234


def build_dataset(cfg, device, train_sample_limit: int):
    data_cfg = cfg.data
    train_cfg = cfg.training
    return AdaCUTLatentDataset(
        data_root=data_cfg.data_root,
        style_subdirs=data_cfg.style_subdirs,
        allow_hflip=bool(data_cfg.allow_hflip),
        identity_ratio=data_cfg.identity_ratio,
        batch_size_hint=int(train_cfg.batch_size),
        balance_target_styles_per_batch=bool(data_cfg.balance_target_styles_per_batch),
        preload_to_gpu=False,
        preload_max_vram_gb=float(data_cfg.preload_max_vram_gb),
        preload_reserve_ratio=float(data_cfg.preload_reserve_ratio),
        virtual_length_multiplier=float(data_cfg.virtual_length_multiplier),
        limit_per_style=int(getattr(data_cfg, "packed_limit_per_style", 0) or 0),
        train_sample_limit=int(train_sample_limit),
        pairing_mode="fixed",
        pairing_seed=int(getattr(data_cfg, "pairing_seed", 0) or 0),
        identity_prob=float(getattr(data_cfg, "identity_prob", 0.0) or 0.0),
        content_style_sampling_weights=data_cfg.content_style_sampling_weights,
        target_style_sampling_weights=data_cfg.target_style_sampling_weights,
        pairing_cache_path="",
        pairing_cache_topk=int(data_cfg.pairing_cache_topk),
        pairing_cache_active_topk=0,
        pairing_cache_sample_mode=str(data_cfg.pairing_cache_sample_mode),
        pairing_cache_rank_schedule="fixed",
        pairing_cache_min_topk=int(data_cfg.pairing_cache_min_topk),
        pairing_cache_curriculum_epochs=0,
        pairing_cache_rank_power=float(data_cfg.pairing_cache_rank_power),
        pairing_cache_explore_prob=0.0,
        pairing_cache_explore_topk=0,
        pairing_cache_dual_target_mix=0.0,
        pairing_cache_dual_target_topk=0,
        pairing_cache_aux_target_topk=0,
        pairing_cache_cross_only=bool(data_cfg.pairing_cache_cross_only),
        latent_cache_mode="off",
        latent_cache_dir="",
        style_caption_path=str(getattr(data_cfg, "style_caption_path", "")),
        reference_dino_cache_dir="",
        device=str(device),
    )


def metrics(v, u) -> dict:
    """Per-row cosine / ratio / MSE; ratios use median so a few near-zero-u rows
    cannot dominate (the earlier run's 1e9 nmse came from exactly that)."""
    vv = torch.cat([v[k].flatten(1) for k in SUBBANDS], dim=1)
    uu = torch.cat([u[k].flatten(1) for k in SUBBANDS], dim=1)
    un = uu.norm(dim=1)
    vn = vv.norm(dim=1)
    cos_rows = F.cosine_similarity(vv, uu, dim=1, eps=1e-8)
    ratio_rows = vn / un.clamp_min(1e-8)
    mse_rows = (vv - uu).pow(2).mean(dim=1)
    out = {
        "cos": float(cos_rows.mean()),
        "cos_median": float(cos_rows.median()),
        "ratio": float(ratio_rows.median()),
        "ratio_mean": float(ratio_rows.mean()),
        "mse": float(F.mse_loss(vv, uu)),
        "n_u_near_zero": int((un < 1e-4).sum()),
        "u_norm_mean": float(un.mean()),
        "v_norm_mean": float(vn.mean()),
    }
    per = {}
    for k in SUBBANDS:
        a, b = v[k].flatten(1), u[k].flatten(1)
        bn = b.norm(dim=1)
        per[k] = {
            "cos": float(F.cosine_similarity(a, b, dim=1, eps=1e-8).mean()),
            "ratio": float((a.norm(dim=1) / bn.clamp_min(1e-8)).median()),
            "mse": float(F.mse_loss(a, b)),
            "u_norm_mean": float(bn.mean()),
        }
    out["per_subband"] = per
    return out


def run_group(model, objective, items, style_ids, device):
    """Forward one group; batches are per style_id so conditioning is exact."""
    acc_v = {k: [] for k in SUBBANDS}
    acc_u = {k: [] for k in SUBBANDS}
    g = torch.Generator().manual_seed(NOISE_SEED)
    sigma = float(getattr(objective, "bridge_sigma", 0.0) or 0.0)
    with torch.no_grad():
        for sid in sorted(set(style_ids)):
            idxs = [i for i, s in enumerate(style_ids) if s == sid]
            if not idxs:
                continue
            content = torch.stack([items[i]["content"] for i in idxs]).to(device)
            target = torch.stack([items[i]["target_style"] for i in idxs]).to(device)
            endpoint = objective._build_training_target(content, target, target)
            c_sub = ds._subbands(content)
            e_sub = ds._subbands(endpoint)
            u = {k: e_sub[k] - c_sub[k] for k in SUBBANDS}
            t = FIXED_T
            x_t = (1.0 - t) * content + t * endpoint
            if sigma > 0.0:
                noise = torch.randn(content.shape, generator=g, dtype=content.dtype)
                x_t = x_t - noise.to(device) * sigma * (t * (1.0 - t)) ** 0.5
            v = model(x_t, t=t, style_id=int(sid), style_latent=target)
            for k in SUBBANDS:
                acc_v[k].append(v[k])
                acc_u[k].append(u[k])
    merged_v = {k: torch.cat(acc_v[k], dim=0) for k in SUBBANDS}
    merged_u = {k: torch.cat(acc_u[k], dim=0) for k in SUBBANDS}
    return metrics(merged_v, merged_u)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device(args.device)
    cfg = load_experiment_config(args.config)
    model = build_model_from_config(cfg.model, bridge_cfg=cfg.bridge).to(device)
    from flow import FlowMatchingObjective
    objective = FlowMatchingObjective(cfg)

    n_train = int(getattr(cfg.data, "train_sample_limit", 0) or 0) or 32
    ds_train = build_dataset(cfg, device, n_train)
    ds_full = build_dataset(cfg, device, 0)
    ds_train.set_epoch(0)
    ds_full.set_epoch(0)

    train_items = [ds_train[i] for i in range(len(ds_train))]
    train_sids = [int(it["target_style_id"]) for it in train_items]

    # 未见配对: 同 content / 同 target style, 换一张目标图 (全量池 +1)
    unseen_items, unseen_sids = [], []
    for it in train_items:
        sid = int(it["target_style_id"])
        pool = ds_full.style_tensors[sid]
        tidx = (int(it["target_cache_index"]) + 1) % pool.shape[0]
        unseen_items.append({"content": it["content"], "target_style": pool[tidx]})
        unseen_sids.append(sid)

    # 全新内容: 每个风格取训练子集之外的前 8 张, 目标是下一风格的图
    fresh_items, fresh_sids = [], []
    caps = ds_train._train_limit_per_style or [0] * len(ds_train.style_subdirs)
    for sid in range(len(ds_train.style_subdirs)):
        pool = ds_full.style_tensors[sid]
        start = int(caps[sid]) if sid < len(caps) else 0
        tsid = (sid + 1) % len(ds_train.style_subdirs)
        tpool = ds_full.style_tensors[tsid]
        for j in range(start, min(start + 8, pool.shape[0])):
            fresh_items.append({"content": pool[j], "target_style": tpool[j % tpool.shape[0]]})
            fresh_sids.append(tsid)
    fresh_items = fresh_items[: n_train]
    fresh_sids = fresh_sids[: n_train]

    groups = (
        ("train_pairs", train_items, train_sids),
        ("unseen_pairs", unseen_items, unseen_sids),
        ("fresh_content", fresh_items, fresh_sids),
    )

    payload = {
        "config": args.config,
        "train_sample_limit": n_train,
        "t": FIXED_T,
        "noise_seed": NOISE_SEED,
        "bridge_sigma": float(getattr(objective, "bridge_sigma", 0.0) or 0.0),
        "checkpoints": {},
    }

    for ckpt in args.checkpoints:
        ds._load_checkpoint(model, ckpt, device)
        model.eval()
        rows = {name: {"n": len(items), **run_group(model, objective, items, sids, device)}
                for name, items, sids in groups}
        rows["train_vs_unseen"] = {
            "cos_gap": rows["train_pairs"]["cos"] - rows["unseen_pairs"]["cos"],
            "mse_gap": rows["unseen_pairs"]["mse"] - rows["train_pairs"]["mse"],
        }
        payload["checkpoints"][ckpt] = rows

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps(payload, indent=2, ensure_ascii=False)[:2000], flush=True)
    print(f"[probe_overfit] wrote {out}")


if __name__ == "__main__":
    main()

