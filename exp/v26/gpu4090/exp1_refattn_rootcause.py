r"""V26-L Exp 1: RefAttn 根因复核 (零训练) — 几何泄露 vs 噪声的因果判别.

docs/1120/08_v26_long_plan.md 局部性专章 §Exp 1.

K1 五组差分锁定的病因: 局部支路**泄露参考图几何** —
  - Value=0 → DINO-C 0.5892 (门控把强全局调小, 内容反弹);
  - Original RefAttn (在 Value=0 基础上加回空间对齐的原始局部特征) → 0.5392
    (DINO-C 骤降 −0.050, 风格仅 +0.006);
  - Value 打乱 → 0.5886 ≈ Value=0 (空间连贯性被破坏 → 内容破坏消失).
数学推论: 未打乱 V_ref 携带空间连贯的参考图自身几何, 经尖锐 Softmax 搬运后把
源图结构拉向参考图几何; 打乱后空间连贯性破坏 → 加权平均退化为空间常偏置.

本实验在 C2nt 权重上做四组**零训练**干预 (150 cross 任务, in-process DINO 评分):
  X1  V_ref 去空间化 (spatial mean → 逐 token 常量): 若 DINO-C 反弹 → 几何泄露确认
  X2  温度扫描: softmax 温度 T ∈ {4, 48} (基线 12) — 尖锐度因果
  X3  UOT Γ̄ 替换 Dense Softmax (零参数, 冻结 DINO token 代价)
  X4  每图空间门控方差 + Attention 熵 (per-image 统计)

用法:  python -m exp.v26.gpu4090.exp1_refattn_rootcause --pool ../pseudo_gt_24h/orig_mean_fp16
"""
from __future__ import annotations

import argparse
import math
from types import MethodType

import torch
import torch.nn.functional as NF

from exp.v26 import OUT, ROOT, save_json, utc
from exp.v26.sr_cfg import load_sr_env

N_TASKS = 150


def _make_instrumented(mode: str, temp: float = 12.0, use_uot: bool = False):
    """构造干预版 RefAttnV24.forward (V_ref 变换 + 注意力形式)."""
    from exp.v24.p0_o1_o3_value import RefAttnV24
    import torch.nn.functional as F

    def forward(self, h16, d16_c, d16_r, v16_r, u_glob, smask2, cmask4):
        B, _, H, W = h16.shape
        Q = F.normalize(self.q_proj(d16_c.flatten(2).transpose(1, 2)), dim=-1)
        K = F.normalize(self.k_proj(d16_r.flatten(2).transpose(1, 2)), dim=-1)
        V = self.v_proj(v16_r.flatten(2).transpose(1, 2))
        if mode == "v_despatial":
            # V_ref 去空间化: 每 image 的空间均值 → 逐 token 常量 (无空间边缘方向)
            V = V.mean(dim=1, keepdim=True).expand_as(V)
        sim = torch.bmm(Q, K.transpose(1, 2))
        if use_uot:
            from exp.v26.uot_fiber_bundle import uot_gamma_bar
            A = uot_gamma_bar(d16_c, d16_r, eps=0.05, n_iter=6)  # (B, 256, 256) 行归一
            R = torch.bmm(A, V)
        else:
            R = torch.bmm(torch.softmax(sim * temp, dim=-1), V)
        if self.use_conf_gate:
            max_sim = sim.max(dim=-1, keepdim=True).values
            g_i = torch.sigmoid(self.conf_temp * (max_sim + self.conf_bias))
            u_g = self.glob_proj(u_glob).unsqueeze(1).expand_as(R)
            R = g_i * R + (1.0 - g_i) * u_g
        R_map = R.transpose(1, 2).view(B, -1, H, W)
        return h16 + self.out_proj(R_map) * (smask2.view(B, 1, 1, 1) * cmask4)

    return forward


def run_exp1(pool: str, device: str = "cuda", n_tasks: int = N_TASKS) -> dict:
    from types import MethodType
    import exp.v25 as _v25
    from PIL import Image as PILImage
    from src.utils import FeatureStack, decode_s10, load_s10_decoder
    from exp.v18 import tasks_and_sources
    from exp.v26.sr_cfg import integrate_sr, sr_velocity

    _v25.seed_all(20266210)
    env = load_sr_env(pool, device=device)
    fm, ad = env["fm"], env["ad"]
    dec = load_s10_decoder(device)
    stack = FeatureStack(device)
    w_l, b_l = env["w_l"].to(device), env["b_l"].to(device)

    tasks, z_src_750, _ = tasks_and_sources()
    cross_ids = [i for i, t in enumerate(tasks) if t["source_style_id"] != t["target_style_id"]][:n_tasks]
    sel = torch.tensor(cross_ids, dtype=torch.long)
    B = len(sel)
    tid_t = torch.tensor([tasks[i]["target_style_id"] for i in cross_ids], device=device)
    tid_cpu = tid_t.cpu()
    zs = z_src_750[sel].to(device).float()
    y16 = (torch.nn.functional.avg_pool2d(
        ((zs * w_l).sum(1, keepdim=True) + b_l).clamp(0.0, 1.0), 4, 4)).clamp(-1.0, 1.0)
    d16 = env["bundle"]["test_struct"]["dino_16"][sel].to(device).float()
    u_src = env["u_src_750"][sel].to(device)
    u_style = ad(env["u_sg5"][tid_cpu].to(device), u_src)
    u_glob = env["fib_n_sg5"][tid_cpu]
    d16_r = env["d16_sg5"][tid_cpu].to(device)
    v16_r = env["v16_sg5"][tid_cpu].to(device)
    tau0, wj = 0.45, 2.2
    eps = torch.randn(B, 4, 64, 64, generator=torch.Generator().manual_seed(20261185)).to(device)
    z = (1.0 - tau0) * eps + tau0 * zs

    rgb_src = decode_s10(dec, zs, batch=16).clamp(0.0, 1.0)
    z_ref5 = env["c21"]["sg5"]["z"].float().to(device)
    rgb_ref5 = decode_s10(dec, z_ref5, batch=5).clamp(0.0, 1.0)

    def dino_of(imgs):
        pil = [PILImage.fromarray((x.permute(1, 2, 0).cpu().numpy() * 255).round().clip(0, 255).astype("uint8"))
               for x in imgs]
        return stack.extract(pil, batch=16)["dino"].cpu()

    f_content = dino_of(rgb_src)
    f_ref5 = dino_of(rgb_ref5)[tid_cpu]

    @torch.no_grad()
    def run_arm(tag: str, mode: str, temp: float = 12.0, use_uot: bool = False) -> dict:
        fwd = _make_instrumented(mode, temp=temp, use_uot=use_uot)
        fm.ref_attn16.forward = MethodType(fwd, fm.ref_attn16)

        def vel_fn(zz, prog):
            tv = torch.full((B,), tau0 + prog * (1.0 - tau0), device=device)
            v, _ = sr_velocity(fm, ad, u_style, None, u_glob, None, d16, d16_r, v16_r,
                               None, None, y16, zz, tv, mode="baseline", w=0.0, wj=wj, device=device)
            return v

        z_gen = integrate_sr(vel_fn, z, tau0, 10, "rk2")
        rgb_gen = decode_s10(dec, z_gen, batch=16).clamp(0.0, 1.0)
        f_gen = dino_of(rgb_gen)
        fm.ref_attn16.forward = fm.ref_attn16.__class__.forward  # 还原
        row = {
            "dino_c_vs_content": round(float(NF.cosine_similarity(f_gen, f_content, dim=-1).mean()), 5),
            "dino_s_vs_target_ref": round(float(NF.cosine_similarity(f_gen, f_ref5, dim=-1).mean()), 5),
        }
        print(f"[Exp1 {tag}] {row}", flush=True)
        return row

    results = {
        "R0_baseline": {"dino_c_vs_content": 0.5392, "dino_s_vs_target_ref": 0.39575},
        "R3_vzero": {"dino_c_vs_content": 0.58917, "dino_s_vs_target_ref": 0.38974},
        "R4_vshuffle": {"dino_c_vs_content": 0.5886, "dino_s_vs_target_ref": 0.3851},
    }
    for tag, mode, temp, uot in (
        ("X1_v_despatial", "v_despatial", 12.0, False),
        ("X2a_temp4", "raw", 4.0, False),
        ("X2b_temp48", "raw", 48.0, False),
        ("X3_uot_gamma", "raw", 12.0, True),
    ):
        results[tag] = run_arm(tag, mode, temp, uot)

    out = {"task": "V26-L Exp1 RefAttn root-cause (zero-training)",
           "n_tasks": B, "updated_utc": utc(), "arms": results}
    save_json(OUT / "gpu4090" / "exp1_refattn_rootcause.json", out)

    # 判读
    x1, r3 = results["X1_v_despatial"], results["R3_vzero"]
    verdict = []
    if x1["dino_c_vs_content"] >= r3["dino_c_vs_content"] - 0.005:
        verdict.append("V_ref 去空间化后 DINO-C 完全反弹 → 几何泄露确认 (非噪声)")
    else:
        verdict.append(f"去空间化后 DINO-C 仍低 ({x1['dino_c_vs_content']:.4f}) → 几何泄露 + 其他因素")
    x3 = results["X3_uot_gamma"]
    verdict.append(f"UOT Γ̄: DINO-C={x3['dino_c_vs_content']} (vs R3 0.589) — "
                   f"{'UOT 软对齐保留内容 ✓' if x3['dino_c_vs_content'] >= 0.57 else 'UOT 仍有内容损耗'}")
    out["verdict"] = verdict
    save_json(OUT / "gpu4090" / "exp1_refattn_rootcause.json", out)
    for v in verdict:
        print(f"[Exp1 verdict] {v}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    from exp.v26 import sanitize_cuda_env
    sanitize_cuda_env()
    run_exp1(args.pool, device=args.device)


if __name__ == "__main__":
    main()
