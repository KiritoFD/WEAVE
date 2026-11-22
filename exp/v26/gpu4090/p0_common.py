r"""P0 公共设施: 三条防呆红线 (用户 2026-11-22 指令) + z0⊥ 色度对齐锚.

红线 1 教师质量自动熔断: 任何新教师缓存生成后先 150 任务解码直测;
        Top-1 < 50% 或五画派单类 < 20% → RuntimeError 停机.
红线 2 严禁标签泄漏: 1-Shot 推理入口强制断言无 target_class_id 贯穿全流程.
红线 3 双门控记录: 结果 JSON 强制并排 Cross600 (overall + 5-style min) 与
        SG-25 OOD (delta_dino_s / style_affinity / swap_dist / dino_c).

z0⊥ 色度对齐热启动 (免标签锚点候选 b): 1D 亮度 P_Y 锁源, 3D 色度 P_C 矩对齐参考
(latent 逐通道 mean/std 匹配, ch0 亮度通道保源), τ0=0 直接从 z0 全程积分.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from exp.v26 import OUT, save_json, utc

STYLES = ("Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e")


# ------------------------------------------------- 红线 2 ----

def assert_label_free(ctx: dict) -> None:
    """1-Shot 推理入口断言: 上下文中不得出现任何目标画派 id / one-hot / 原型索引."""
    banned = ("target_class_id", "target_style_id", "sid", "class_id", "proto_idx", "tid")
    for k, v in ctx.items():
        kl = k.lower()
        if any(b in kl for b in banned):
            raise RuntimeError(f"[红线2] 标签泄漏嫌疑: 推理上下文含 '{k}'={v!r} — 1-Shot 协议禁止")
    print("[红线2] label-free assertion passed (no target class id in ctx)", flush=True)


# ------------------------------------------------- 红线 1 ----

@torch.no_grad()
def teacher_fuse_gate(dec, stack, lat_by_style: dict[int, list[torch.Tensor]],
                      n_per_style: int = 30, device: str = "cuda",
                      gate_top1: float = 0.50, gate_min_style: float = 0.20,
                      tag: str = "") -> dict:
    """教师质量自动熔断 (红线 1): 逐目标画派抽样直测 CSD Top-1.

    lat_by_style: {target_style_id: [scaled latents]} — 教师对跨画派内容的输出.
    Top-1 < gate_top1 或 单类 < gate_min_style → RuntimeError (禁止进入训练).
    """
    from src.utils import decode_s10
    lat_all, want_all, style_all = [], [], []
    for t, lats in sorted(lat_by_style.items()):
        step = max(1, len(lats) // n_per_style)
        pick = lats[::step][:n_per_style]
        lat_all += [x for x in pick]
        want_all += [t] * len(pick)
        style_all += [t] * len(pick)
    pred_all = []
    for st in range(0, len(lat_all), 24):
        chunk = torch.stack(lat_all[st:st + 24]).to(device).float()
        rgb = decode_s10(dec, chunk, batch=24).clamp(0.0, 1.0)
        feats = stack.extract([_pil(x) for x in rgb], batch=24)
        cs = feats["csd"] @ stack.csd_prototypes.T
        pred_all += cs.argmax(1).tolist()
    per_style = {STYLES[t]: round(sum(1 for p, s in zip(pred_all, style_all)
                                      if s == t and p == t)
                                  / max(1, sum(1 for s in style_all if s == t)), 4)
                 for t in range(5)}
    top1 = round(sum(int(p == w) for p, w in zip(pred_all, want_all)) / len(pred_all), 4)
    res = {"tag": tag, "n_pairs": len(pred_all), "top1": top1, "per_style": per_style,
           "gate": {"top1": gate_top1, "min_style": gate_min_style},
           "updated_utc": utc()}
    ok = top1 >= gate_top1 and min(per_style.values()) >= gate_min_style
    res["gate_pass"] = bool(ok)
    save_json(OUT / "gpu4090" / f"p0_teacher_fuse_{tag or 'gate'}.json", res)
    if not ok:
        raise RuntimeError(
            f"[红线1] 教师熔断: 直测 Top-1={top1} < {gate_top1} 或 单类最低 "
            f"{min(per_style.values())} < {gate_min_style} — 禁止进入训练. per_style={per_style}")
    print(f"[红线1] teacher gate PASS: Top-1={top1} per_style={per_style}", flush=True)
    return res


# ------------------------------------------------- 红线 3 ----

def dual_gate_record(arm: str, cross600: dict, sg25: dict | None,
                     extra: dict | None = None, out_name: str | None = None) -> dict:
    """双门控记录 (红线 3): Cross600 (overall + 5-style min) 与 SG-25 并排落盘."""
    five = cross600.get("top1_per_target_style") or cross600.get("top1_5style") or {}
    rec = {
        "arm": arm, "updated_utc": utc(),
        "cross600": {"top1": cross600.get("csd_top1") or cross600.get("top1"),
                     "dino_c": cross600.get("dino_c"), "dino_s": cross600.get("dino_s"),
                     "lpips": cross600.get("lpips"),
                     "top1_per_target_style": five,
                     "top1_min_style": min(five.values()) if five else None},
        "sg25_ood": None if sg25 is None else {
            "delta_dino_s": sg25.get("delta_dino_s"),
            "style_affinity": (sg25.get("rows") or [{}])[0].get("style_affinity"),
            "swap_dist": sg25.get("swap_dist"),
            "dino_c": (sg25.get("rows") or [{}])[0].get("dino_c")},
        **(extra or {}),
    }
    save_json(OUT / "gpu4090" / (out_name or f"p0_dualgate_{arm}.json"), rec)
    print(f"[红线3] dual-gate {arm}: cross600={rec['cross600']['top1']} "
          f"min_style={rec['cross600']['top1_min_style']} sg25={rec['sg25_ood']}", flush=True)
    return rec


# ------------------------------------------------- z0⊥ 锚 ----

def chroma_align_anchor(z_src: torch.Tensor, z_ref: torch.Tensor,
                        luma_lock: float = 1.0) -> torch.Tensor:
    """z0⊥ 色度对齐热启动 (免标签): ch0 亮度通道保源, ch1..3 色度通道矩对齐参考.

    z_src/z_ref: (B, 4, 64, 64) scaled latents. 返回 z0 (与 z_src 同形状).
    """
    z0 = z_src.clone()
    z0[:, 0] = luma_lock * z_src[:, 0]  # P_Y 锁源
    for c in range(1, 4):
        ms, ss = z_src[:, c].mean(), z_src[:, c].std().clamp_min(1e-4)
        mr, sr = z_ref[:, c].mean(), z_ref[:, c].std().clamp_min(1e-4)
        z0[:, c] = ((z_src[:, c] - ms) / ss) * sr + mr
    return z0


def _pil(x: torch.Tensor):
    from PIL import Image as PILImage
    return PILImage.fromarray((x.permute(1, 2, 0).cpu().numpy() * 255).round().clip(0, 255).astype("uint8"))
