"""V25 机制 B: 双模型 Pareto 曲线蒸馏进单模型 —— γ 残差蒸馏学生 (B0/B1) — B01 实现.

学生形式 (agent2.md §二第 3 条, 修正 M5/M6 失败设计):

    z_student = z_P1 + γ · R_θ(z_P1, c_64, d_16, chord, γ)

γ=0 精确回退由"**冻结基座 + 显式 γ 残差**"构造性保证: z_p1 + 0·R_θ ≡ z_p1 逐元素严格相等
(IEEE754: 0·有限值 = ±0, x + (−0) = x), **不依赖零初始化技巧、不训练 PRISM-1 基座**——
SpatiallyRoutedChordNet 基座 forward 冻结复用 (训练侧直接消费缓存 z_p1_tr = 基座在同一
输入上的确定性输出; 评测侧经 compute_prism1_latents 同配方重算). R_θ 末层零初始化只是
训练稳定性的附加性质 (初始任意 γ 亦恒等), 不承担 γ=0 回退正确性.

γ 注入: γ → Fourier 编码 32 维 (sin/cos(2^k πγ), k=0..15) → 逐 ResBlock FiLM (B.2 第 1 点).

训练目标 (02_mechanism_plan.md B.3):
    z*(γ) = (1−γ)·z_p1_tr + γ·z_d6_tr,  γ ∈ {0, 0.25, 0.40, 0.55, 0.70} 均匀抽 + γ=0 权重×2
    L = MSE(z_student, z*) + 0.3 · L_cnx-mid(dec(z_student), dec(z*))   (可微 S10 子样本)
教师定义铁律: 与机制 A 共用 exp.v25.mechanism_a.CANONICAL_TEACHER_SPEC 及
runs/v25/cache/train_latents_4400.pt 的 {z_p1_tr, z_d6_tr, teacher_spec} (assert_same_teacher).
same-task bypass (输出 = z_src) 保持网络之外, 与 PRISM-1 推理一致 (B.2 第 3 点).

规模: B0 残差分支 ≤1M (width64×4blk ≈ 0.36M) / B1 宽化 ~3–5M (width176×8blk ≈ 4.7M).

评测: eval_student_gamma_sweep 在 γ∈{0,0.25,0.33,0.40,0.55,0.63,0.70} (0.33/0.63 为未见泛化点)
逐 γ 输出 vs 教师曲线逐任务残差 + per-style 分解 + 中间 γ 线性插值误差
(教师曲线对 γ 线性 → 完美学生插值 gap 应≈0, 学生偏离线性即泛化误差的独立刻画).

用法 (4090, 本任务只交付不启动):
  python -m exp.v25.mechanism_b --scale B1 --measure --pool ../pseudo_gt_24h/orig_mean_fp16
"""
from __future__ import annotations

import argparse
import copy
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import OUT, ROOT, STYLES, autocast, sanitize_cuda_env, save_json, seed_all, tier_micro_accum, utc
from src.utils import load_s10_decoder
from exp.v19.fiber_duality import build_or_load_v19_foundation
from exp.v20.submode_patchswd import build_or_load_core_knn_bank, load_convnext_tiny_offline
from exp.v25.r1_inference_ablation import eval_latent_v25
from exp.v25.mechanism_a import (
    CANONICAL_TEACHER_SPEC,
    build_teacher_latents_750,
    cnx_mid_content_loss,
    fiber_chords,
    load_teacher_cache,
)

B_SEED = 20263520                    # 全臂同 seed → 共享采样顺序 (r2_train_ablation 工程规范)
B_GAMMA_POOL = (0.0, 0.0, 0.25, 0.40, 0.55, 0.70)   # γ=0 权重 ×2 (B.3: P(γ=0)=1/3)
B_GAMMA_SWEEP = (0.0, 0.25, 0.33, 0.40, 0.55, 0.63, 0.70)  # 0.33/0.63 未见泛化点
B_TRAINED_GAMMAS = (0.0, 0.25, 0.40, 0.55, 0.70)
LAMBDA_STRUCT = 0.3                  # B.3: λ_struct = 0.3
B_EFFECTIVE = 64
FOURIER_FREQ = 16                    # → 32 维 Fourier 特征


# ============================================================================
# 1) 学生网络 (残差分支; 基座在函数之外冻结复用)
# ============================================================================

def gamma_fourier(gamma, n_freq: int = FOURIER_FREQ) -> torch.Tensor:
    """γ → (B, 2·n_freq) Fourier 特征: [sin(2^k πγ), cos(2^k πγ)]_{k=0..n_freq−1}.

    标量 / 0-dim / (B,) 张量均可; 输出确定性, γ=0 → sin≡0, cos≡1.
    """
    if not torch.is_tensor(gamma):
        gamma = torch.tensor([float(gamma)])
    gamma = gamma.reshape(-1).float()
    ks = torch.arange(n_freq, dtype=torch.float32, device=gamma.device)
    ang = (2.0 ** ks) * math.pi * gamma[:, None]  # (B, n_freq)
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)


class GammaResBlock(nn.Module):
    """Conv-ResBlock + 双 FiLM (风格弦全局 + γ-Fourier 逐块; B.2 "γ 拼入风格弦通路")."""

    def __init__(self, width: int, dilation: int, fourier_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(16, width)
        self.conv1 = nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(16, width)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1)
        self.film = nn.Linear(fourier_dim, width * 2)
        nn.init.zeros_(self.film.weight)   # 初始 γ-Fourier 不注入 (训练稳定; 与回退正确性无关)
        nn.init.zeros_(self.film.bias)

    def forward(self, h: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        x = self.norm1(h) * (1.0 + scale) + shift
        x = self.conv1(F.silu(x))
        x = self.conv2(F.silu(self.norm2(x)))
        return h + x


class GammaResidualNet(nn.Module):
    """R_θ: [z_p1(4) ⊕ c64(4)] 8ch stem + d16 1×1 注入 + 风格弦 FiLM + γ-Fourier 逐块 FiLM.

    out_conv 零初始化 → 初始 R_θ≡0 (附加稳定性质); γ=0 精确回退由外层 γ 乘子构造性保证.
    """

    def __init__(self, width: int = 64, blocks: int = 4, chord_dim: int = 128,
                 dilations: tuple = (1, 1, 2, 2), n_freq: int = FOURIER_FREQ):
        super().__init__()
        assert len(dilations) == blocks, "dilations 与 blocks 必须一一对应"
        self.n_freq = n_freq
        self.stem = nn.Sequential(
            nn.Conv2d(4 + 4, width, 3, padding=1),
            nn.GroupNorm(16, width),
            nn.SiLU(inplace=True),
        )
        self.dino_add = nn.Sequential(
            nn.Conv2d(384, width, 1),
            nn.GroupNorm(16, width),
            nn.SiLU(inplace=True),
        )
        self.chord_proj = nn.Linear(chord_dim, width * 2)
        self.blocks = nn.ModuleList([GammaResBlock(width, d, 2 * n_freq) for d in dilations])
        self.out_conv = nn.Conv2d(width, 4, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, z_p1: torch.Tensor, c64: torch.Tensor, d16: torch.Tensor,
                chord: torch.Tensor, gamma) -> torch.Tensor:
        h = self.stem(torch.cat([z_p1.float(), c64.float()], dim=1))
        h = h + F.interpolate(self.dino_add(d16.float()), size=h.shape[-2:],
                              mode="bilinear", align_corners=False)
        gfeat = gamma_fourier(gamma, self.n_freq).to(h.device)
        sc, sh = self.chord_proj(chord.float()).chunk(2, dim=-1)
        scale = sc[:, :, None, None].expand_as(h)
        shift = sh[:, :, None, None].expand_as(h)
        for blk in self.blocks:
            sg, st = blk.film(gfeat).chunk(2, dim=-1)
            h = blk(h, scale + sg[:, :, None, None], shift + st[:, :, None, None])
        return self.out_conv(F.silu(h))


class GammaResidualStudent(nn.Module):
    """完整学生 z_student = z_P1 + γ·R_θ(...). γ=0 → 与 z_P1 逐元素严格相等 (构造性, 单测断言)."""

    def __init__(self, **kw):
        super().__init__()
        self.residual = GammaResidualNet(**kw)

    def n_params(self) -> int:
        return self.residual.n_params()

    def forward(self, z_p1: torch.Tensor, c64: torch.Tensor, d16: torch.Tensor,
                chord: torch.Tensor, gamma) -> torch.Tensor:
        r = self.residual(z_p1, c64, d16, chord, gamma)
        g = torch.as_tensor(gamma, dtype=z_p1.dtype, device=z_p1.device).reshape(-1)
        if g.numel() == 1:
            g = g.expand(z_p1.shape[0])
        return z_p1 + g.view(-1, 1, 1, 1) * r


B_SCALES = {
    # B0: 残差分支 ≤1M (02 计划 B.3 "原 PRISM-1 宽度"档 → 本任务规约为残差 ≤1M)
    "B0": dict(width=64, blocks=4, dilations=(1, 1, 2, 2), updates=3000),
    # B1: 宽化 ~3–5M
    "B1": dict(width=176, blocks=8, dilations=(1, 1, 2, 2, 4, 4, 1, 1), updates=6000),
}


def sample_train_gammas(batch: int, generator: torch.Generator) -> torch.Tensor:
    """训练 γ 采样: pool 均匀抽, γ=0 占两席 → 权重 ×2 (确定性: 显式传入 generator)."""
    pool = torch.tensor(B_GAMMA_POOL)
    return pool[torch.randint(0, len(pool), (batch,), generator=generator)]


# ============================================================================
# 2) 训练 (4090 入口; 本任务只交付不启动)
# ============================================================================

def train_student(bundle: dict, cache: dict, scale: str, cnx, dec, dec_g, device: str,
                  updates: int | None = None, decode_sub: int = 6, log_every: int = 300):
    """只训残差分支 R_θ; 冻结基座输出以缓存 z_p1_tr 给出 (教师目标与基座输出同源无 gap).

    r2/v19 工程规范: seed_all + cosine lr (1.2e-3→8e-5, v19 学生配方) + micro/accum + EMA 0.996
    + clip 1.0. 已存在 ckpt 则复用 (4090 队列断点续跑).
    """
    assert scale in B_SCALES, f"unknown scale {scale}"
    cfg = {k: v for k, v in B_SCALES[scale].items() if k != "updates"}
    updates = updates or B_SCALES[scale]["updates"]
    out_p = OUT / "mechanism_b" / f"student_{scale}.pt"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if out_p.exists():
        print(f"[MechB {scale}] Reusing {out_p}", flush=True)
        ck = torch.load(out_p, map_location="cpu", weights_only=False)
        return out_p, ck["info"]

    seed_all(B_SEED)
    device_t = torch.device(device)
    tr = bundle["v18_cache"]["train"]
    z_p1_all = cache["z_p1_tr"].float()   # 冻结基座输出 = 学生在 γ=0 的精确回退目标
    z_d6_all = cache["z_d6_tr"].float()   # 教师混合上限
    c64_all = tr["c_64"].float()
    d16_all = tr["dino_16"].float()
    tid_all = cache["tid_tr"].long()
    chord_all = fiber_chords(bundle, bundle["fnd"]["tr_fib"].float(), tid_all, device).cpu()
    n = len(z_p1_all)

    student = GammaResidualStudent(**cfg).to(device_t)
    ema = copy.deepcopy(student).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(student.parameters(), lr=1.2e-3, betas=(0.9, 0.99), weight_decay=1e-4)
    micro, accum = tier_micro_accum(B_EFFECTIVE, default_micro_24gb=B_EFFECTIVE,
                                    max_micro_12gb=32, device=device)
    cpu_gen = torch.Generator().manual_seed(B_SEED + 1)
    t0, lat_v, st_v = time.time(), 0.0, 0.0

    for step in range(1, updates + 1):
        prog = step / max(1, updates)
        lr = 8e-5 + 0.5 * (1.2e-3 - 8e-5) * (1.0 + math.cos(math.pi * prog))
        for pg in opt.param_groups:
            pg["lr"] = lr
        opt.zero_grad(set_to_none=True)
        for _ in range(accum):
            idx = torch.randint(0, n, (micro,), generator=cpu_gen)
            g = sample_train_gammas(micro, cpu_gen).to(device_t)
            z_p1 = z_p1_all[idx].to(device_t)
            z_d6 = z_d6_all[idx].to(device_t)
            gw = g.view(-1, 1, 1, 1)
            z_star = (1.0 - gw) * z_p1 + gw * z_d6
            with autocast(device):
                z_stu = student(z_p1, c64_all[idx].to(device_t), d16_all[idx].to(device_t),
                                chord_all[idx].to(device_t), g)
                loss_lat = F.mse_loss(z_stu, z_star)
                loss = loss_lat
                loss_st_v = 0.0
                if step % 2 == 0:  # 可微 S10 + ConvNeXt 中尺度子样本 (v19/v23 模式, 控显存)
                    sub = slice(0, min(decode_sub, micro))
                    loss_st = LAMBDA_STRUCT * cnx_mid_content_loss(
                        cnx, dec_g, dec, z_stu[sub], z_star[sub])
                    loss = loss + loss_st
                    loss_st_v = float(loss_st.detach())
            (loss / accum).backward()
            lat_v += float(loss_lat.detach()) / accum
            st_v += loss_st_v / accum
        nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe, pw in zip(ema.parameters(), student.parameters()):
                pe.mul_(0.996).add_(pw, alpha=0.004)
        if step == 1 or step % log_every == 0 or step == updates:
            print(f"[MechB {scale}] {step:4d}/{updates} lat={lat_v/step:.5f} cnx={st_v/step:.5f} "
                  f"lr={lr:.2e} ({time.time()-t0:.0f}s)", flush=True)

    info = dict(scale=scale, updates=updates, n_params=student.n_params(), micro=micro,
                accum=accum, gamma_pool=list(B_GAMMA_POOL), lambda_struct=LAMBDA_STRUCT,
                final_lat=lat_v, final_cnx=st_v,
                teacher_spec=CANONICAL_TEACHER_SPEC.to_dict())
    torch.save({"ema": ema.state_dict(), "model": student.state_dict(), "info": info}, out_p)
    print(f"[MechB {scale}] Saved {out_p} in {(time.time()-t0)/60:.1f} min", flush=True)
    return out_p, info


# ============================================================================
# 3) γ 扫描评测 (student_fn 可注入 → CPU mock 可测; 网络外 same bypass)
# ============================================================================

@torch.no_grad()
def eval_student_gamma_sweep(student_fn, z_src: torch.Tensor, z_p1: torch.Tensor,
                             z_d6: torch.Tensor, same_mask: torch.Tensor, tid: torch.Tensor,
                             cond: dict | None = None, gammas=B_GAMMA_SWEEP,
                             trained=B_TRAINED_GAMMAS, batch: int = 50,
                             device: str = "cpu") -> dict:
    """逐 γ 学生 vs 教师曲线评测; 返回 records + z_by_gamma (供可选 measure 复用).

    student_fn(z_p1_b, gamma_vec_b, cond_b) -> z_student_b; cond_b = dict(c64, d16, chord) 切片.
    教师曲线 z*(γ) = (1−γ)z_p1 + γ·z_d6 (对 γ 线性); same 任务 bypass → 输出与残差恒 0.
    记录: 逐任务残差 (cross600) / per-style 分解 / 未见 γ 泛化点 + 线性插值误差.
    """
    z_p1, z_d6, z_src = z_p1.float(), z_d6.float(), z_src.float()
    cross = ~same_mask.bool()
    trained = tuple(float(t) for t in trained)

    z_by_gamma = {}
    for g in gammas:
        g = float(g)
        outs = []
        for st in range(0, len(z_p1), batch):
            sl = slice(st, min(st + batch, len(z_p1)))
            zp = z_p1[sl].to(device)
            gv = torch.full((len(zp),), g, device=device)
            cb = None if cond is None else {k: v[sl].to(device) for k, v in cond.items()}
            outs.append(student_fn(zp, gv, cb).cpu())
        z_by_gamma[g] = torch.cat(outs, 0)

    records = []
    for g in gammas:
        g = float(g)
        gw = z_by_gamma[g].clone()
        gw[same_mask] = z_src[same_mask]          # bypass 在网络之外 (B.2 第 3 点)
        teacher = (1.0 - g) * z_p1 + g * z_d6
        teacher[same_mask] = z_src[same_mask]
        res = ((gw - teacher).flatten(1).norm(dim=1)
               / teacher.flatten(1).norm(dim=1).clamp_min(1e-8))[cross]
        per_style = {s[:3]: round(float(res[tid[cross] == i].mean()), 6)
                     for i, s in enumerate(STYLES)}
        rec = dict(
            gamma=g, unseen=(g not in trained),
            residual_mean=round(float(res.mean()), 6), residual_p50=round(float(res.median()), 6),
            residual_max=round(float(res.max()), 6), residual_per_task=[round(float(v), 6) for v in res],
            residual_per_style=per_style, interp_gap=None, interp_neighbors=None,
        )
        if rec["unseen"]:
            lo = max(t for t in trained if t < g)
            hi = min(t for t in trained if t > g)
            w = (g - lo) / (hi - lo)
            stu_interp = (1.0 - w) * z_by_gamma[lo] + w * z_by_gamma[hi]
            gap = ((gw - stu_interp).flatten(1).norm(dim=1)
                   / teacher.flatten(1).norm(dim=1).clamp_min(1e-8))[cross]
            rec["interp_gap"] = round(float(gap.mean()), 6)
            rec["interp_neighbors"] = [lo, hi]
        records.append(rec)
    return dict(records=records, z_by_gamma=z_by_gamma)


# ============================================================================
# 4) main (4090 完整入口; 本任务只交付, 不启动 GPU)
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="both", help="B0 | B1 | both")
    ap.add_argument("--updates", type=int, default=0, help=">0 覆盖每规模默认 (B0=3000/B1=6000)")
    ap.add_argument("--gammas", default=",".join(str(g) for g in B_GAMMA_SWEEP))
    ap.add_argument("--measure", action="store_true", help="逐 γ 完整 test750 四裁判评测")
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    sanitize_cuda_env()
    cache_p = OUT / "cache/train_latents_4400.pt"
    if not cache_p.exists():
        raise FileNotFoundError(f"{cache_p} 不存在 — 先运行 python -m exp.v25.precompute_train_latents")
    cache = load_teacher_cache(cache_p)
    bundle = build_or_load_v19_foundation(args.pool, device=args.device)
    bank = build_or_load_core_knn_bank(bundle, k_sub=8, core_per_style=180, device=args.device)
    cnx = load_convnext_tiny_offline(args.device).eval()
    for p in cnx.parameters():
        p.requires_grad_(False)
    dec = load_s10_decoder(args.device)
    dec_g = copy.deepcopy(dec).to(device=args.device, dtype=torch.bfloat16).eval()
    for p in dec_g.parameters():
        p.requires_grad_(False)

    # 评测侧教师口径 latent (z_p1_750 / z_d6_750 由 spec 三元组逐位再生, 见 mechanism_a)
    ctx750 = build_teacher_latents_750(bundle, bank, args.device)
    cond750 = dict(c64=ctx750["c64"], d16=ctx750["d16"], chord=ctx750["chords"])
    gammas = tuple(float(g) for g in str(args.gammas).split(",") if g.strip())

    scales = ("B0", "B1") if args.scale == "both" else (args.scale,)
    out_json = OUT / "mechanism_b" / "mechanism_b.json"
    results = {}
    for scale in scales:
        ckpt_p, info = train_student(bundle, cache, scale, cnx, dec, dec_g, args.device,
                                     updates=(args.updates or None))
        student = GammaResidualStudent(**{k: v for k, v in B_SCALES[scale].items() if k != "updates"}).to(args.device)
        student.load_state_dict(torch.load(ckpt_p, map_location="cpu", weights_only=False)["ema"], strict=True)
        student.eval()

        def student_fn(zp, gv, cb, _s=student, _d=args.device):
            return _s(zp, cb["c64"], cb["d16"], cb["chord"], gv).to(_d)

        sweep = eval_student_gamma_sweep(student_fn, ctx750["z_src"], ctx750["z_p1"],
                                         ctx750["z_d6"], ctx750["same"], ctx750["tid"],
                                         cond=cond750, gammas=gammas, device=args.device)
        if args.measure:
            for rec in sweep["records"]:
                arm = f"MechB_{scale}_g{int(round(rec['gamma']*100)):02d}"
                z_bp = sweep["z_by_gamma"][rec["gamma"]].clone()
                z_bp[ctx750["same"]] = ctx750["z_src"][ctx750["same"]]  # A08 R2 修复: measure 路径传 bypass 后张量 (与 :303 残差口径一致, Same150 测 z_src 而非学生原始输出)
                m = eval_latent_v25(arm, z_bp, ctx750["tasks"],
                                    ctx750["dec"], device=args.device)
                rec["metrics"] = {k: v for k, v in m.items() if k != "arm"}
        results[scale] = dict(info=info, records=sweep["records"])
        save_json(out_json, {"updated_utc": utc(), "gammas": list(gammas),
                             "teacher_spec": CANONICAL_TEACHER_SPEC.to_dict(),
                             "results": results})
        print(f"[MechB {scale}] sweep done -> {out_json}", flush=True)

    print("\n=== V25 机制 B γ 扫描汇总 (逐任务 latent 残差, cross600) ===", flush=True)
    for scale, r in results.items():
        for rec in r["records"]:
            mt = rec.get("metrics", {})
            m = {k: mt[k] for k in ("dino_c", "dino_s", "lpips") if k in mt}
            star = "*" if rec["unseen"] else " "
            print(f"{scale} γ={rec['gamma']:.2f}{star} res={rec['residual_mean']:.4f} "
                  f"interp={rec['interp_gap']} {m}", flush=True)


if __name__ == "__main__":
    main()
