"""P4 FM 探针：伪配对留出集 FM 误差 / 正确 vs 错误 Style-ID / 跨 Pattern 泛化。

对每个训练臂的 ckpt，在留出集配对缓存上以固定 t=0.5 计算 FM 谱损失（不更新参数），
并按目标 Style-ID（正确 / 偏移 1 / 偏移 2）分别统计，用于回答：
  - 模型是否真的把“伪配对端点”当成可回归的目标（留出集 FM 误差）；
  - Style-ID 是否为必要条件（错误 ID 时误差上升多少）；
  - 只在训练时见过的一种 Mask（R64）上学到的能力，能否迁移到另一种（F64）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import paths  # noqa: E402
from flow import FlowMatchingObjective  # noqa: E402
from config_schema import ExperimentConfig  # noqa: E402
from utils.inference import LGTInference  # noqa: E402
from utils.dataset import PairedLatentDataset  # noqa: E402
from tools.pseudo_gt.build_assets import STYLES  # noqa: E402

T_FIXED = 0.5


@torch.inference_mode()
def probe_cache(model, obj, cache_dir: str, sid_mode: str, batch: int, device: str) -> dict:
    ds = PairedLatentDataset(str(cache_dir))
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)
    sums = {"ll": 0.0, "lh": 0.0, "hl": 0.0, "hh": 0.0, "total": 0.0}
    n = 0
    n_style = len(STYLES)
    for b in dl:
        content = b["content"].to(device=device, dtype=torch.float32)
        target = b["target_style"].to(device=device, dtype=torch.float32)
        sid = b["target_style_id"].to(device=device, dtype=torch.long)
        if sid_mode != "correct":
            shift = int(sid_mode)
            sid = (sid + shift) % n_style
        t = torch.full((content.shape[0],), T_FIXED, device=device, dtype=torch.float32)
        out = obj.compute_probe_losses(
            model, content=content, target_style=target, target_style_id=sid,
            conditioning={"target_style_latent": target}, t=t, noise=None,
        )
        wl = {"ll": obj.w_ll, "lh": obj.w_lh, "hl": obj.w_hl, "hh": obj.w_hh}
        bs = content.shape[0]
        for k in ("ll", "lh", "hl", "hh"):
            sums[k] += float(out[f"loss_fm_spectral_{k}_raw"].item()) * bs
        sums["total"] += float(
            (wl["ll"] * out["loss_fm_spectral_ll_raw"]
             + wl["lh"] * out["loss_fm_spectral_lh_raw"]
             + wl["hl"] * out["loss_fm_spectral_hl_raw"]).item()
        ) * bs
        n += bs
    return {k: v / max(n, 1) for k, v in sums.items()} | {"n": n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--probe-cache", action="append", required=True,
                    help="name=dir，可重复，如 R64_pos=<dir>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--config-override", default=str(REPO_ROOT / "inference.json"))
    args = ap.parse_args()

    if not torch.cuda.is_available() or args.device != "cuda":
        raise SystemExit("[fm_probe] CUDA required")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ExperimentConfig.from_mapping(ckpt.get("config", {}) or {})
    model = LGTInference(
        model_path=args.checkpoint, device=args.device, num_steps=1,
        config_override_path=args.config_override,
    ).model.eval()
    obj = FlowMatchingObjective(cfg)

    rows = []
    for item in args.probe_cache:
        name, cache_dir = item.split("=", 1)
        for sid_mode in ("correct", "1", "2"):
            m = probe_cache(model, obj, cache_dir, sid_mode, args.batch, args.device)
            rows.append({"arm": args.arm, "checkpoint": str(args.checkpoint), "probe": name,
                         "cache_dir": cache_dir, "sid_mode": sid_mode, **m})
            print(f"[fm_probe] {args.arm} {name} sid={sid_mode} total={m['total']:.5f} n={m['n']}",
                  flush=True)
    Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[fm_probe] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
