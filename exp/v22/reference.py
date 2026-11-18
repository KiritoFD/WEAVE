"""V22-P2 参考图编码与风格编码器诊断（3060 可独立运行）。

诊断 1（§3.4-Q1）: 同风格参考对的编码距离是否显著小于跨风格距离
  —— 直接在 v19_foundation 的 tr/ho fiber 缓存上计算，无需 GPU 重推理。
诊断 2（§3.4-Q2 / §8.4）: 固定内容图、更换 5 张参考图后输出是否真正跟随参考
  —— 需要 V21 prism_attn_final + stiefel cache（轻量 50 任务×5 参考）。
诊断 3（P2-2）: 任意 RGB→latent Encoder 可用性盘点（仅记录，不做重推理）。
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from . import OUT, ROOT, sanitize_cuda_env, save_json, utc
from exp.v19.fiber_duality import build_or_load_v19_foundation


def diag_fiber_separation(fnd: dict, device: str = "cuda") -> dict:
    """同风格内 vs 跨风格 fiber 距离（128-D StyleFiberEncoder 输出）。"""
    res = {}
    for split, fib_key, sid_key in (
        ("train", "tr_fib", None), ("holdout", "ho_fib", None),
    ):
        fib = fnd[fib_key].float().to(device)
        sid = (fnd["v18_cache"]["train"]["sid"] if split == "train" else fnd["v18_cache"]["holdout"]["sid"]).long().to(device)
        from exp.v19.fiber_duality import StyleFiberEncoder
        enc = StyleFiberEncoder(in_dim=fib.shape[1], dim=128).to(device).eval()
        enc.load_state_dict(fnd["fiber_state"])
        with torch.no_grad():
            u = enc.encode(fib)
        same_d, cross_d = [], []
        g = torch.Generator(device="cpu").manual_seed(20261202)
        for _ in range(4000):
            i, j = torch.randint(0, len(u), (2,), generator=g).tolist()
            d = float(F.pairwise_distance(u[i:i+1].cpu(), u[j:j+1].cpu()))
            (same_d if int(sid[i]) == int(sid[j]) else cross_d).append(d)
        res[split] = {
            "same_style_mean_dist": round(sum(same_d) / max(1, len(same_d)), 4),
            "cross_style_mean_dist": round(sum(cross_d) / max(1, len(cross_d)), 4),
            "separation_ratio": round((sum(cross_d) / max(1, len(cross_d))) / max(1e-6, sum(same_d) / max(1, len(same_d))), 4),
            "n_same": len(same_d), "n_cross": len(cross_d),
        }
        del enc
        torch.cuda.empty_cache()
    verdict = all(r["separation_ratio"] > 1.10 for r in res.values())
    res["verdict_same_lt_cross"] = bool(verdict)
    return res


@torch.no_grad()
def diag_reference_swap_sensitivity(bundle: dict, device: str = "cuda", n_tasks: int = 10) -> dict:
    """固定内容图，换 5 张不同参考（SG-5），输出 DINO/色彩统计变化幅度。"""
    from exp.v18 import tasks_and_sources
    from exp.v21.prism_attn_flow import PRISMAttnUNet, integrate_prism_attn
    from exp.v21.stiefel_ref_cache import build_or_load_v21_cache
    from src.utils import FeatureStack, decode_s10, load_s10_decoder
    from PIL import Image

    pack = build_or_load_v21_cache(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16", device=device)
    c21, fnd = pack["c21"], pack["fnd"]
    tasks, z_src_750, _ = tasks_and_sources()

    # 5 个目标各取 2 个 cross 任务（内容固定、参考轮换）
    sel = []
    for s in range(5):
        cands = [i for i, t in enumerate(tasks) if t["source_style_id"] != t["target_style_id"] and t["target_style_id"] == s]
        sel.extend(cands[:n_tasks // 5])
    sel_t = torch.tensor(sorted(sel))
    tid = torch.tensor([tasks[i]["target_style_id"] for i in sel], device=device)

    model = PRISMAttnUNet(base_dim=144).to(device).eval()
    ck = torch.load(OUT.parent / "v21/attn_flow/prism_attn_final.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    dec = load_s10_decoder(device)
    stack = FeatureStack(device)

    y16_750 = c21["test750"]["y16"]
    y32_750 = c21["test750"]["y32"]
    q32_750 = c21["test750"]["q32"]
    mid_dsd = float(c21["mid_dsd"])
    eps = torch.randn(z_src_750.shape, generator=torch.Generator().manual_seed(20261203), device=device)

    out_stats = []
    for ref_i in range(5):  # 5 张 SG 参考（与目标 style 无关地轮换，检验是否盲目跟随）
        sg = c21["sg5"]
        u_r = sg["u"][ref_i].to(device).float().expand(len(sel_t), -1)
        k_r = sg["q32"][ref_i].float().flatten(2).transpose(1, 2).to(device).expand(len(sel_t), -1, -1)
        v_r = sg["vref"][ref_i].to(device).float().expand(len(sel_t), -1, -1)
        dsd_r = sg["dsd"][ref_i].to(device).float().expand(len(sel_t))
        z_gen = integrate_prism_attn(
            model, z_src_750[sel_t].to(device).float(), eps[sel_t],
            y16_750[sel_t].to(device).float(), y32_750[sel_t].to(device).float(),
            q32_750[sel_t].to(device).float(), u_r, k_r, v_r, dsd_r, mid_dsd, steps=10,
        )
        rgb = decode_s10(dec, z_gen, batch=10)
        pils = [Image.fromarray((x.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype("uint8")) for x in rgb]
        fo = stack.extract(pils, batch=10)
        dino_f = F.normalize(torch.as_tensor(fo["dino"], device=device, dtype=torch.float32), dim=-1)
        out_stats.append({"ref_idx": ref_i, "dino_feat_mean": dino_f.mean(0).cpu(),
                          "dino_feat_std_mean": float(dino_f.std(0).mean())})

    # 参考更换后的输出特征变化幅度
    diffs = []
    for i in range(len(out_stats)):
        for j in range(i + 1, len(out_stats)):
            d = float(F.pairwise_distance(out_stats[i]["dino_feat_mean"], out_stats[j]["dino_feat_mean"]))
            diffs.append(d)
    res = {
        "n_tasks": len(sel_t),
        "n_refs": 5,
        "pairwise_output_feat_dist": [round(d, 4) for d in diffs],
        "mean_pairwise_dist": round(sum(diffs) / max(1, len(diffs)), 4),
        "note": "mean_pairwise_dist 接近 0 => 输出几乎不随参考变化（忽略参考）; 过大 => 参考内容泄漏。需结合目检解释。",
    }
    return res


def diag_rgb_encoder_availability() -> dict:
    import os
    candidates = [
        "deliver/checkpoints/runs/sd0_s10/sd0_s10_best_overall.pt",
        "deliver/checkpoints/runs/sd0_s10/sd0_s10_best_response.pt",
        "deliver/checkpoints/runs/a0/a0_encoder_only.pt",
        "deliver/checkpoints/runs/vae_style_vae/C_edit_best.pt",
    ]
    found = {}
    for rel in candidates:
        p = ROOT / rel
        found[rel] = {"exists": p.exists(), "size_mb": round(p.stat().st_size / 1e6, 1) if p.exists() else None}
    return {
        "candidates": found,
        "status": "unverified：任一 encoder 能否以 0.18215 scaling 编码 SD1.5 latent 域需实际双向验证后再宣称 RGB→RGB 端到端",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-swap", action="store_true", help="跳过需要 V21 ckpt 的参考轮换诊断")
    args = ap.parse_args()

    sanitize_cuda_env()
    res = {"updated_utc": utc()}

    bundle = build_or_load_v19_foundation(args.pool, device=args.device)
    print("[V22 RefDiag] 1/3 fiber same-vs-cross separation...", flush=True)
    res["fiber_separation"] = diag_fiber_separation(bundle["fnd"], device=args.device)
    print(f"  train sep={res['fiber_separation']['train']['separation_ratio']}, "
          f"holdout sep={res['fiber_separation']['holdout']['separation_ratio']}, "
          f"verdict={res['fiber_separation']['verdict_same_lt_cross']}", flush=True)

    res["rgb_encoder"] = diag_rgb_encoder_availability()

    if not args.skip_swap:
        try:
            print("[V22 RefDiag] 2/3 reference swap sensitivity (V21, 50 gen)...", flush=True)
            res["reference_swap"] = diag_reference_swap_sensitivity(bundle, device=args.device)
            print(f"  mean pairwise dist={res['reference_swap']['mean_pairwise_dist']}", flush=True)
        except Exception as e:  # V21 ckpt 缺失时降级
            res["reference_swap"] = {"error": str(e)}

    save_json(OUT / "p2_reference_diagnostics.json", res)
    print(f"[V22 RefDiag] Saved -> {OUT / 'p2_reference_diagnostics.json'}", flush=True)


if __name__ == "__main__":
    main()
