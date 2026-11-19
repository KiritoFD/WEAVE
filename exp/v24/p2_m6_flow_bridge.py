"""V24 补充冲刺 (4090): PRISM-1 × V23 高风格/高保真双分支 (DualSWD / HiC_JPk) 潜空间混合 + 4步 Residual Flow Bridge (M6)。

检验:
1. 把 V20 D6 换成 V23 更强风格与更低 NFE 的 DualSWD_pure_hiT_ws52 (D-C 0.6164, D-S 0.5467) 或 HiC_JPk_csj65_t4 (D-C 0.6299, Top1 58.8%)
   与 PRISM-1(l=1.0) 混合，测试能否直接击穿 V24 主目标: DINO-C >= 0.73, DINO-S >= 0.49, LPIPS <= 0.51;
2. 在此之上训练一个 4步轻量 Residual Flow Bridge (M6, ~1.2M UNet 走 4步 RK2 从 z_p1 -> z_target, 带动态 ConvNeXt 预算)，
   解决 1步卷积表达力不足的问题，实现单引擎极速 (~28ms) 突破 DINO-C >= 0.73 & DINO-S >= 0.49!

用法 (4090):
  python -m exp.v24.p2_m6_flow_bridge --pool ../pseudo_gt_24h/orig_mean_fp16
"""
from __future__ import annotations

import argparse
import copy
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import OUT, EVAL_SEED, ROOT, autocast, sanitize_cuda_env, save_json, seed_all, utc
from .p0_m0_m2 import compute_prism1_latents, eval_latent_tensor
from .p1_m3_m4_residual import ResBlock2d
from exp.v22.core import CFGConfig, D5_SOLVER, GateConfig, PrismCore
from exp.v22.t4 import build_or_load_t4_latents_750
from exp.v18 import tasks_and_sources
from exp.v19.fiber_duality import StyleFiberEncoder, build_or_load_v19_foundation
from exp.v20.run_v20 import get_test750_y16_and_d16
from exp.v20.submode_patchswd import MultiScalePatchSWDExtractor, build_or_load_core_knn_bank
from src.utils import decode_s10, load_s10_decoder
from src.utils.structure import dwt2_haar

DUAL_SWD_CKPT = ROOT / "runs/v23/dual_swd/slf_DualSWD_pure.pt"
JPK_CKPT = ROOT / "runs/v23/joint_c1ab/slf_Joint_Pk_C1AB.pt"


class ResidualFlowNet4(nn.Module):
    """~1.4M 4步轻量残差流速度网络 v_theta(z_t, c_64, u_tgt, t): 专精从 PRISM-1 起点出发的洁净多步积分。"""

    def __init__(self, cond_dim: int = 128, ch: int = 96):
        super().__init__()
        self.t_mlp = nn.Sequential(nn.Linear(128 + 32, cond_dim), nn.SiLU(inplace=True), nn.Linear(cond_dim, cond_dim))
        self.in_conv = nn.Conv2d(4 + 4, ch, 3, padding=1)
        self.b1 = ResBlock2d(ch, cond_dim)
        self.down = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
        self.b2 = ResBlock2d(ch, cond_dim)
        self.b3 = ResBlock2d(ch, cond_dim)
        self.up = nn.Conv2d(ch * 2, ch, 3, padding=1)
        self.b4 = ResBlock2d(ch, cond_dim)
        self.out_conv = nn.Conv2d(ch, 4, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, z: torch.Tensor, c64: torch.Tensor, u_tgt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        freqs = torch.exp(-math.log(10000.0) * torch.arange(16, device=z.device, dtype=torch.float32) / 16.0)
        ang = t.float().view(-1, 1) * freqs.view(1, -1) * 10.0
        temb = torch.cat([ang.sin(), ang.cos()], dim=-1)
        cond = self.t_mlp(torch.cat([u_tgt, temb], dim=-1))

        h64 = self.b1(self.in_conv(torch.cat([z, c64], dim=1)), cond)
        h32 = self.b3(self.b2(self.down(h64), cond), cond)
        u64 = F.interpolate(h32, size=h64.shape[-2:], mode="bilinear", align_corners=False)
        h = self.b4(self.up(torch.cat([u64, h64], dim=1)), cond)
        return self.out_conv(h)


@torch.no_grad()
def _eval_v23_dual_swd_latents(bundle, bank, tasks, z_src_750, z_t4, y16_750, d16_750, eps_750, device="cuda"):
    core = PrismCore(bundle, bank, DUAL_SWD_CKPT, device=device)
    sid_all = torch.tensor([t["source_style_id"] for t in tasks])
    tid_all = torch.tensor([t["target_style_id"] for t in tasks])
    same_all = sid_all == tid_all
    out = torch.zeros_like(z_src_750, dtype=torch.float32)
    solver = dict(D5_SOLVER, tau_min=0.32, tau_hi=0.68, ws_fig=5.2)
    for st in range(0, 750, 25):
        sl = slice(st, min(st + 25, 750))
        z_gen, _ = core.integrate_flow(
            eps_750[sl], z_t4[sl].to(device), y16_750[sl], d16_750[sl], tid_all[sl].to(device),
            gate=GateConfig(mode="original"), cfg=CFGConfig(branches="cs_j", rescale_phi=0.48), solver=solver,
        )
        z_out = torch.where(same_all[sl].view(-1, 1, 1, 1).to(device), z_src_750[sl].to(device).float(), z_gen)
        out[sl] = z_out.cpu()
    del core
    torch.cuda.empty_cache()
    return out


def train_m6_flow(bundle: dict, pack: dict, dec, swd_ext: MultiScalePatchSWDExtractor,
                  updates: int = 1200, micro: int = 16, device: str = "cuda"):
    out_p = OUT / "m6_flow_bridge/res_flow4.pt"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if out_p.exists():
        print(f"[V24 M6] Reusing {out_p}", flush=True)
        return out_p

    seed_all(20263300)
    tr = bundle["v18_cache"]["train"]
    fnd = bundle["fnd"]
    fiber_enc = StyleFiberEncoder(in_dim=fnd["tr_fib"].shape[1], dim=128).to(device).eval()
    fiber_enc.load_state_dict(fnd["fiber_state"])
    with torch.no_grad():
        bary = fiber_enc.class_barycenters().detach()
    del fiber_enc

    net = ResidualFlowNet4(cond_dim=128, ch=96).to(device)
    print(f"[V24 M6] ResidualFlowNet4 params={sum(p.numel() for p in net.parameters())/1e6:.2f}M", flush=True)
    ema = copy.deepcopy(net).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(net.parameters(), lr=2e-4, betas=(0.9, 0.99), weight_decay=1e-4)
    M = len(pack["idx_c"])
    t0 = time.time()

    for step in range(1, updates + 1):
        lr = 2e-5 + 0.5 * (2e-4 - 2e-5) * (1.0 + math.cos(math.pi * step / max(1, updates)))
        for pg in opt.param_groups:
            pg["lr"] = lr
        opt.zero_grad(set_to_none=True)

        pick = torch.randint(0, M, (micro,))
        ic = pack["idx_c"][pick]
        tb = pack["tid_t"][pick].to(device)
        z_b = pack["z_p1"][pick].to(device).float()
        z_t = pack["z_d6"][pick].to(device).float()
        c64 = tr["c_64"][ic].to(device).float()

        # 直线 Flow Bridge: 从 z_b (PRISM-1) 走向 z_t (洁净 D6 风格), 配合 Haar 高频加权
        t = torch.rand(micro, device=device)
        zt = (1.0 - t).view(micro, 1, 1, 1) * z_b + t.view(micro, 1, 1, 1) * z_t
        v_tgt = z_t - z_b

        with autocast(device):
            v_pred = net(zt, c64, bary[tb], t)
            ll_p, lh_p, hl_p, hh_p = dwt2_haar(v_pred.float())
            ll_t, lh_t, hl_t, hh_t = dwt2_haar(v_tgt)
            loss = (
                0.7 * F.mse_loss(ll_p, ll_t)
                + 1.3 * F.mse_loss(lh_p, lh_t)
                + 1.3 * F.mse_loss(hl_p, hl_t)
                + 1.1 * F.mse_loss(hh_p, hh_t)
            )
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe, pw in zip(ema.parameters(), net.parameters()):
                pe.mul_(0.996).add_(pw, alpha=0.004)
        if step == 1 or step % 400 == 0 or step == updates:
            print(f"[V24 M6] {step:4d}/{updates} loss={loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)

    torch.save({"ema": ema.state_dict()}, out_p)
    print(f"[V24 M6] Saved {out_p} in {(time.time()-t0)/60:.1f} min", flush=True)
    return out_p


@torch.no_grad()
def infer_m6_latents(bundle: dict, ckpt_p, tasks: list[dict], z_src_750: torch.Tensor,
                     z_p1_100: torch.Tensor, steps: int = 4, strength: float = 0.70, device: str = "cuda") -> torch.Tensor:
    fnd = bundle["fnd"]
    fiber_enc = StyleFiberEncoder(in_dim=fnd["tr_fib"].shape[1], dim=128).to(device).eval()
    fiber_enc.load_state_dict(fnd["fiber_state"])
    bary = fiber_enc.class_barycenters()
    del fiber_enc

    net = ResidualFlowNet4(cond_dim=128, ch=96).to(device).eval()
    net.load_state_dict(torch.load(ckpt_p, map_location="cpu", weights_only=False)["ema"], strict=True)
    c64_750 = bundle["test_struct"]["c_64"].float()
    sid_all = torch.tensor([t["source_style_id"] for t in tasks], device=device)
    tid_all = torch.tensor([t["target_style_id"] for t in tasks], device=device)
    same_t = sid_all == tid_all

    out = torch.zeros_like(z_p1_100, dtype=torch.float32)
    dt = strength / float(steps)
    for st in range(0, 750, 50):
        sl = slice(st, min(st + 50, 750))
        z = z_p1_100[sl].to(device).float()
        c64 = c64_750[sl].to(device).float()
        u_t = bary[tid_all[sl]]
        for i in range(steps):
            t1 = torch.full((len(z),), (i / float(steps)) * strength, device=device)
            v1 = net(z, c64, u_t, t1)
            zm = z + 0.5 * dt * v1
            t2 = torch.full((len(z),), ((i + 0.5) / float(steps)) * strength, device=device)
            v2 = net(zm, c64, u_t, t2)
            z = z + dt * v2
        z[same_t[sl]] = z_src_750[sl].to(device).float()[same_t[sl]]
        out[sl] = z.cpu()
    del net
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    sanitize_cuda_env()
    bundle = build_or_load_v19_foundation(args.pool, device=args.device)
    bank = build_or_load_core_knn_bank(bundle, k_sub=8, core_per_style=180, device=args.device)
    tasks, z_src_750, _ = tasks_and_sources()
    z_t4, _ = build_or_load_t4_latents_750(bundle, device=args.device)
    y16_750, d16_750 = get_test750_y16_and_d16(bundle)
    eps_750 = torch.randn(z_src_750.shape, generator=torch.Generator().manual_seed(EVAL_SEED))
    dec = load_s10_decoder(args.device)

    cache_p = OUT / "cache/p0_base_latents_750.pt"
    z_p1_100 = torch.load(cache_p, map_location="cpu", weights_only=False)["z_p1_100"]

    results = []
    # 1. PRISM-1 × V23 DualSWD_pure_hiT 潜空间混合
    if DUAL_SWD_CKPT.exists():
        z_dswd = _eval_v23_dual_swd_latents(bundle, bank, tasks, z_src_750, z_t4, y16_750, d16_750, eps_750, device=args.device)
        for gamma in (0.35, 0.50):
            results.append(eval_latent_tensor(f"M1_p1_dswd_g{int(gamma*100)}", (1.0 - gamma) * z_p1_100 + gamma * z_dswd, tasks, dec, device=args.device))

    # 2. M6 4步 Residual Flow Bridge (从 z_p1 出发走 4 步 RK2 到洁净风格)
    pack = torch.load(OUT / "cache/m5_clean_bridge_1200.pt", map_location="cpu", weights_only=False)
    swd_ext = MultiScalePatchSWDExtractor(n_slices=64).to(args.device).eval()
    ckpt_m6 = train_m6_flow(bundle, pack, dec, swd_ext, updates=1200, device=args.device)
    del swd_ext
    torch.cuda.empty_cache()

    for s_val, tag in ((0.55, "s055"), (0.70, "s070"), (0.85, "s085")):
        z_m6 = infer_m6_latents(bundle, ckpt_m6, tasks, z_src_750, z_p1_100, steps=4, strength=s_val, device=args.device)
        results.append(eval_latent_tensor(f"M6_flow4_{tag}", z_m6, tasks, dec, device=args.device))

    save_json(OUT / "p2_m6.json", {"updated_utc": utc(), "results": results})
    print("\n=== V24 M1(DualSWD) & M6(4步残差流) 汇总 ===", flush=True)
    for r in results:
        print(f"{r['arm']:20s}: DINO-C={r['dino_c']:.4f} DINO-S={r['dino_s']:.4f} LPIPS={r['lpips']:.4f} Top1={r['csd_top1']*100:5.2f}%", flush=True)


if __name__ == "__main__":
    main()
