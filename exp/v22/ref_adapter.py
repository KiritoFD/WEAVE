"""V22-P2-1 Reference Adapter（3060 one-shot 线）：冻结 FM 主干，只训连续参考条件 Adapter。

设计（writing-block(1) §3.2/§6.2）:
  u_ref  = StyleFiberEncoder(fiber_stats(x_ref))  — 冻结
  u_src  = StyleFiberEncoder(fiber_stats(x_src))  — 冻结
  u_style = Normalize(u_ref + alpha · A_delta(u_ref − u_src))   — A_delta 轻量 MLP (~66K)
  onehot(5) 输入位置保留但恒喂零（无类别信息）

训练（§6.2 Adapter-only 热启动）:
  内容 = 训练池画作 z_i；参考 = 同画派异图（distinct neighbor，无放回）；
  目标 = z_i 本身；drop_style 0.12（供 cs_j 分支）；content 恒开启；layout_scale=1.0；
  主干全冻结，只训 Adapter，lr 2e-4，2000 updates。
"""
from __future__ import annotations

import argparse
import copy
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import OUT, ROOT, autocast, sanitize_cuda_env, save_json, seed_all, utc
from exp.v20.slf_flow import SLFUNet, coarse_y16_lcn
from exp.v18.ccf_model import sinusoidal_time_emb
from exp.v19.fiber_duality import StyleFiberEncoder, build_or_load_v19_foundation


class RefAdapter(nn.Module):
    """u_style = Normalize(u_ref + alpha · A_delta(u_ref − u_src))；A_delta = 128→256→128 MLP。"""

    def __init__(self, dim: int = 128, hidden: int = 256):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.3))
        self.a_delta = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(inplace=True),
            nn.Linear(hidden, dim),
        )

    def forward(self, u_ref: torch.Tensor, u_src: torch.Tensor) -> torch.Tensor:
        delta = self.a_delta(u_ref - u_src)
        return F.normalize(u_ref + self.alpha * delta, dim=-1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class ZeroOnehotSLFUNet(SLFUNet):
    """onehot(5) 恒零的 SLFUNet（无类别信息）；其余与父类逐位一致。"""

    def forward(self, z_tau, y16_lcn, dino_16, style_id, u_vivid, tau, *,
                drop_style=None, drop_content=None, layout_scale=None):
        B = z_tau.shape[0]
        dev = z_tau.device
        if drop_style is None:
            drop_style = torch.zeros(B, dtype=torch.bool, device=dev)
        if drop_content is None:
            drop_content = torch.zeros(B, dtype=torch.bool, device=dev)
        cmask4 = (~drop_content).float().view(B, 1, 1, 1)
        smask2 = (~drop_style).float().view(B, 1)
        if layout_scale is None:
            lscale = (~drop_content).float()
        else:
            lscale = layout_scale.to(dev).float().view(B).clamp(0.0, 1.0) * (~drop_content).float()
        lscale4 = lscale.view(B, 1, 1, 1)

        d_feat = self.dino_proj(dino_16 * cmask4) * lscale4
        s16 = torch.cat([y16_lcn * lscale4, d_feat], dim=1)

        # onehot(5) 槽位恒零 — 无任何类别信息
        oh = torch.zeros(B, self.N_STYLE, device=dev)
        uv = u_vivid * smask2
        temb = sinusoidal_time_emb(tau, 64)
        flags = torch.stack([drop_style.float(), 1.0 - lscale], dim=-1)
        cond = self.cond_mlp(torch.cat([temb, oh, uv, flags], dim=-1))

        h64 = self.in_conv(z_tau)
        for blk in self.enc64:
            h64 = blk(h64, cond)
        h32 = self.down32(h64)
        for blk in self.enc32:
            h32 = blk(h32, cond)
        h16 = self.down16(h32)
        h16 = self.mid1(h16, s16, cond)
        h16 = self.mid_attn(h16)
        h16 = self.mid2(h16, s16, cond)
        u32 = self.up32(h16.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1))
        u32 = self.skip32(torch.cat([u32, h32], dim=1))
        for blk in self.dec32:
            u32 = blk(u32, cond)
        u64 = self.up64(u32.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1))
        u64 = self.skip64(torch.cat([u64, h64], dim=1))
        for blk in self.dec64:
            u64 = blk(u64, cond)
        return self.out_conv(F.silu(self.out_norm(u64)))


class RefAdapterModel(nn.Module):
    """打包: 冻结 SLFUNet(ZeroOnehot) + 冻结 FiberEncoder + 可训 RefAdapter。"""

    def __init__(self, fm_ckpt: Path, fiber_state: dict, fiber_in_dim: int, device: str = "cuda"):
        super().__init__()
        ck = torch.load(fm_ckpt, map_location="cpu", weights_only=False)
        self.fm = ZeroOnehotSLFUNet(base_dim=144, u_dim=128).to(device).eval()
        self.fm.load_state_dict(ck["ema"], strict=True)
        for p in self.fm.parameters():
            p.requires_grad_(False)

        self.fiber_enc = StyleFiberEncoder(in_dim=fiber_in_dim, dim=128).to(device).eval()
        self.fiber_enc.load_state_dict(fiber_state)
        for p in self.fiber_enc.parameters():
            p.requires_grad_(False)

        self.adapter = RefAdapter(dim=128).to(device)

    def u_from_fiber(self, fib: torch.Tensor) -> torch.Tensor:
        return self.fiber_enc.encode(fib.to(next(self.parameters()).device).float())


def train_ref_adapter(bundle: dict, c21: dict, updates: int = 2000,
                      micro: int = 24, device: str = "cuda") -> Path:
    out_p = OUT / "ref_adapter/ref_adapter.pt"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if out_p.exists():
        print(f"[V22 RefAdapter] Reusing {out_p}", flush=True)
        return out_p

    seed_all(20261220)
    tr = bundle["v18_cache"]["train"]
    w_luma = bundle["v18_cache"]["luma_proj"]["w"].to(device).float()
    b_luma = bundle["v18_cache"]["luma_proj"]["b"].to(device).float()

    pack = RefAdapterModel(
        fm_ckpt=OUT / "attn_flow/prism_dual_branch.pt",
        fiber_state=bundle["fnd"]["fiber_state"],
        fiber_in_dim=bundle["fnd"]["tr_fib"].shape[1],
        device=device,
    )
    n_adapter = pack.adapter.n_params()
    print(f"[V22 RefAdapter] adapter params={n_adapter} (主干冻结)", flush=True)

    z_all = tr["z"]
    sid_all = tr["sid"].long()
    d16_all = tr["dino_16"]
    y16_all = coarse_y16_lcn(tr["y_64"].float()).half()

    # fiber u per painting（冻结 encoder，一次性算好）
    fib_all = bundle["fnd"]["tr_fib"].float()
    with torch.no_grad():
        u_all = pack.u_from_fiber(fib_all).cpu()  # (4400, 128)
    nbr_idx = c21["train"]["nbr_idx"]  # (4400, 12) distinct neighbors

    u_all_dev = u_all.to(device)
    nbr_dev = nbr_idx

    opt = torch.optim.AdamW(pack.adapter.parameters(), lr=2e-4, weight_decay=1e-2)
    t0 = time.time()

    for step in range(1, updates + 1):
        prog = step / max(1, updates)
        lr = 5e-5 + 0.5 * (2e-4 - 5e-5) * (1.0 + math.cos(math.pi * prog))
        for pg in opt.param_groups:
            pg["lr"] = lr

        opt.zero_grad(set_to_none=True)

        idx = torch.randint(0, len(z_all), (micro,))
        n_shots = 1
        nbr_scores = torch.rand(micro, 12)
        perm_nbr = nbr_scores.topk(n_shots, dim=1).indices  # 无放回
        ref_idx = nbr_idx[idx[:, None], perm_nbr].squeeze(1)  # (micro,)

        z1 = z_all[idx].to(device, non_blocking=True).float()
        sid = sid_all[idx].to(device, non_blocking=True)
        d16 = d16_all[idx].to(device, non_blocking=True).float()
        y16 = y16_all[idx].to(device, non_blocking=True).float()

        u_src = u_all_dev[idx]           # 内容图自身 fiber
        u_ref = u_all_dev[ref_idx]       # 参考图 fiber（异图同派）
        u_style = pack.adapter(u_ref, u_src)

        d16_aug = d16 + 0.08 * torch.randn_like(d16)
        d16_aug = d16_aug - d16_aug.mean(dim=(2, 3), keepdim=True)
        y16_aug = (y16 * (0.85 + 0.30 * torch.rand(micro, 1, 1, 1, device=device)) + 0.04 * torch.randn_like(y16)).clamp(-1.1, 1.1)

        u_r = torch.rand(micro, device=device)
        ds = u_r < 0.12
        dc = torch.zeros(micro, dtype=torch.bool, device=device)
        lscale = torch.ones(micro, device=device)

        z0 = torch.randn_like(z1)
        tau = torch.rand(micro, device=device)
        zt = (1.0 - tau).view(-1, 1, 1, 1) * z0 + tau.view(-1, 1, 1, 1) * z1
        target_v = z1 - z0

        with autocast(device):
            pred_v = pack.fm(zt, y16_aug, d16_aug, sid, u_style, tau,
                             drop_style=ds, drop_content=dc, layout_scale=lscale)
            loss = F.mse_loss(pred_v.float(), target_v)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(pack.adapter.parameters(), 1.0)
        opt.step()

        if step == 1 or step % 400 == 0 or step == updates:
            print(f"[V22 RefAdapter] {step:4d}/{updates} loss={loss.item():.4f} alpha={float(pack.adapter.alpha):.3f} ({time.time()-t0:.0f}s)", flush=True)

    torch.save({"adapter": pack.adapter.state_dict(), "n_params": n_adapter,
                "updates": updates, "fm_ckpt": str(OUT / "attn_flow/prism_dual_branch.pt")}, out_p)
    print(f"[V22 RefAdapter] Saved {out_p} in {(time.time()-t0)/60:.1f} min", flush=True)
    return out_p


@torch.no_grad()
def eval_ref_adapter_1shot(bundle: dict, c21: dict, adapter_p: Path, device: str = "cuda") -> list[dict]:
    """1-Shot 评测: SG-5 参考图（与 StyleGallery 完全同参考）× Test750。"""
    from exp.v18 import tasks_and_sources
    from src.utils import decode_s10, load_s10_decoder
    from src.utils.raw_io import RawWriter, export_raw_pngs
    from src.eval.runner import run_measure_subprocess

    tasks, z_src_750, _ = tasks_and_sources()
    dec = load_s10_decoder(device)

    pack = RefAdapterModel(
        fm_ckpt=OUT / "attn_flow/prism_dual_branch.pt",
        fiber_state=bundle["fnd"]["fiber_state"],
        fiber_in_dim=bundle["fnd"]["tr_fib"].shape[1],
        device=device,
    )
    ad = RefAdapter(dim=128).to(device)
    ad.load_state_dict(torch.load(adapter_p, map_location="cpu", weights_only=False)["adapter"])
    pack.adapter = ad

    fib_750 = bundle["fnd"]["test750_fib"].float()
    with torch.no_grad():
        u_src_750 = pack.u_from_fiber(fib_750).cpu()  # (750,128) 内容图自身 fiber

    if "y_64" in bundle["test_struct"]:
        y16_750 = coarse_y16_lcn(bundle["test_struct"]["y_64"].float()).half()
    else:
        w_l = bundle["v18_cache"]["luma_proj"]["w"].float().view(1, 4, 1, 1)
        b_l = bundle["v18_cache"]["luma_proj"]["b"].float()
        y64_750 = ((z_src_750.float() * w_l).sum(1, keepdim=True) + b_l).clamp(0.0, 1.0)
        y16_750 = coarse_y16_lcn(y64_750).half()
    eps_750 = torch.randn(z_src_750.shape, generator=torch.Generator().manual_seed(20261185))

    sg5 = c21["sg5"]  # sg5 u/vref... sg5["u"] 是 fiber u? sg5 存的是 fiber u (5,128)
    u_sg5 = sg5["u"].float()  # (5, 128)

    arms = [
        ("P2A_1shot_vsc", "vsc"),
        ("P2A_1shot_cs_j", "cs_j"),
    ]
    results = []
    from src.utils.protocol import EVAL_SEED

    for arm_name, mode in arms:
        wr = RawWriter(arm_name, raw_root=OUT / "raw", kind="v22")
        t0 = time.time()
        for st in range(0, 750, 25):
            sl = slice(st, min(st + 25, 750))
            grp = tasks[sl]
            B = len(grp)
            zs = z_src_750[sl].to(device).float()
            sid_p = torch.tensor([t["source_style_id"] for t in grp], device=device)
            tid_t = torch.tensor([t["target_style_id"] for t in grp], device=device)
            same = sid_p == tid_t

            u_ref_b = u_sg5[tid_t.cpu()].to(device)     # SG-5 参考图 fiber
            u_src_b = u_src_750[sl].to(device)          # 内容图 fiber
            u_style = pack.adapter(u_ref_b, u_src_b)

            y16 = y16_750[sl].to(device).float()
            zt = eps_750[sl].to(device).float()  # 纯噪声起点（无 T4 依赖）
            f0 = torch.zeros(B, dtype=torch.bool, device=device)
            f1 = torch.ones(B, dtype=torch.bool, device=device)
            l1 = torch.ones(B, device=device)
            ls0 = torch.zeros(B, device=device)
            tv = torch.zeros(B, device=device)
            tv_end = torch.ones(B, device=device)
            steps = 10
            dt = 1.0 / steps

            def vel(zz, prog):
                tv_c = prog * tv_end
                if mode == "vsc":
                    with autocast(device):
                        return pack.fm(zz, y16, bundle["test_struct"]["dino_16"][sl].to(device).float(),
                                       sid_t_dummy(B, device), u_style, tv_c,
                                       drop_style=f0, drop_content=f0, layout_scale=l1).float()
                else:
                    zz2 = torch.cat([zz, zz], 0)
                    y2 = y16.repeat(2, 1, 1, 1)
                    d2 = bundle["test_struct"]["dino_16"][sl].to(device).float().repeat(2, 1, 1, 1)
                    s2 = sid_t_dummy(B, device).repeat(2)
                    u2 = u_style.repeat(2, 1)
                    t2 = (tv_c).repeat(2)
                    ds2 = torch.cat([f0, f1], 0)
                    dc2 = torch.cat([f0, f0], 0)
                    l2 = torch.cat([l1, ls0], 0)
                    with autocast(device):
                        v_sc, v_0c = pack.fm(zz2, y2, d2, s2, u2, t2, drop_style=ds2, drop_content=dc2, layout_scale=l2).float().chunk(2, 0)
                    return v_0c + 2.2 * (v_sc - v_0c)  # 双分支 CFG wj=2.2

            def sid_t_dummy(Bn, dv):
                return torch.zeros(Bn, dtype=torch.long, device=dv)

            z = zt
            for i in range(steps):
                v1 = vel(z, i / steps)
                zm = z + 0.5 * dt * v1
                v2 = vel(zm, (i + 0.5) / steps)
                z = z + dt * v2
            z_out = torch.where(same.view(B, 1, 1, 1), zs, z)
            wr.append([t["task_id"] for t in grp],
                      decode_s10(dec, z_out, batch=16).clamp(0.0, 1.0), z_out)
        wall = round(time.time() - t0, 1)
        meta = wr.finish(arm=arm_name, wall_sec=wall, ms_per_img=round(wall * 1000 / 750, 1))
        print(f"[V22 1Shot] {arm_name} done in {wall}s", flush=True)
        run_measure_subprocess(OUT / "raw" / arm_name, OUT / "measured" / arm_name, arm_name=arm_name, device=device)
        export_raw_pngs(OUT / f"raw/{arm_name}", OUT / f"png_750/{arm_name}", max_workers=8)
        results.append({"arm": arm_name, "wall_sec": wall})
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--updates", type=int, default=2000)
    ap.add_argument("--micro", type=int, default=24)  # 3060 12GB 预算（AGENTS.md micro≤24）
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    sanitize_cuda_env()
    bundle = build_or_load_v19_foundation(args.pool, device=args.device)
    c21_path = ROOT / "runs/v21/cache/v21_stiefel_ref_cache.pt"
    c21 = torch.load(c21_path, map_location="cpu", weights_only=False)

    adapter_p = train_ref_adapter(bundle, c21, updates=args.updates, micro=args.micro, device=args.device)
    if not args.skip_eval:
        res = eval_ref_adapter_1shot(bundle, c21, adapter_p, device=args.device)
        save_json(OUT / "p2a_1shot_results.json", {"updated_utc": utc(), "arms": res})


if __name__ == "__main__":
    main()
