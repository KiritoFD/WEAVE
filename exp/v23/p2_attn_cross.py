"""V23 P2 §2.2 & §2.3 (3060): 16×16 轻量 Reference Attention + 置信度门控 + 跨风格配对训练 (M0/M1/M2)。

架构（严格遵从 V23 §2.2 最小增量原则）:
- 仅在 16×16 bottleneck (mid1 与 mid_attn 之间) 插入一层轻量 Reference Cross-Attention:
    Q = content dino_16 (B, 256, 384) -> proj(D)
    K = reference dino_16 (B, 256, 384) -> proj(D)
    V = reference dino_16 (B, 256, 384) -> proj(D) + 全局风格向量残差
  输出经零初始化线性投影 + 可学习门控 beta 加回 h16（初始严格等同于原模型，diff == 0）。
- 置信度门控 g_i（R6）：基于 max_j cos(Q_i, K_j) 计算局部匹配置信度，低置信回退到零/全局分支而非强行贴错局部区域。

受控实验臂（完全无目标类别 ID, onehot=0, 无测试标签, 修复双分支 cs_j, 1200 updates × micro=16×3）:
- R4_global   : 仅全局参考向量 u_style (无 16×16 Attn)
- R5_refattn  : 全局向量 + 16×16 RefAttn (无置信度门控)
- R6_gated_m0 : R5 + 置信度门控, 100% 自配对 (M0)
- M1_cross25  : R6 + 75% 自配对 + 25% 训练内部跨风格伪配对
- M2_cross50  : R6 + 50% 自配对 + 50% 训练内部跨风格伪配对

附加 §8.4 测试: 固定内容图更换 5 张 SG-5 参考图，测量输出特征随参考变化的距离。

用法 (3060):
  python -m exp.v23.p2_attn_cross --pool ../pseudo_gt_24h/orig_mean_fp16
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

from . import OUT, EVAL_SEED, ROOT, RawWriterV23, autocast, config_hash, export_raw_pngs, file_sha, git_sha, read_json, run_measure_v23, sanitize_cuda_env, save_json, seed_all, utc, validate_raw_meta
from .p0c_oneshot_reval import CustomOnehotSLFUNet
from exp.v22.ref_adapter import RefAdapter
from exp.v18 import tasks_and_sources
from exp.v18.ccf_model import sinusoidal_time_emb
from exp.v19.fiber_duality import StyleFiberEncoder, build_or_load_v19_foundation
from exp.v20.slf_flow import SLFUNet, coarse_y16_lcn
from src.utils import FeatureStack, decode_s10, load_s10_decoder

DUAL_CKPT = ROOT / "runs/v22/attn_flow/prism_dual_branch.pt"


class LightRefAttn16(nn.Module):
    """16×16 轻量 Reference Cross-Attention + 可选置信度门控（零初始化输出投影）。"""

    def __init__(self, h_dim: int = 432, dino_dim: int = 384, attn_dim: int = 128, use_conf_gate: bool = False):
        super().__init__()
        self.use_conf_gate = use_conf_gate
        self.q_proj = nn.Linear(dino_dim, attn_dim)
        self.k_proj = nn.Linear(dino_dim, attn_dim)
        self.v_proj = nn.Linear(dino_dim, attn_dim)
        self.u_proj = nn.Linear(128, attn_dim)
        self.out_proj = nn.Conv2d(attn_dim, h_dim, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.conf_temp = nn.Parameter(torch.tensor(8.0))
        self.conf_bias = nn.Parameter(torch.tensor(-0.3))

    def forward(self, h16: torch.Tensor, d16_content: torch.Tensor,
                d16_ref: torch.Tensor | None, u_style: torch.Tensor,
                smask2: torch.Tensor, cmask4: torch.Tensor) -> torch.Tensor:
        if d16_ref is None:
            return h16
        B, _, H, W = h16.shape  # 16×16
        qc = d16_content.flatten(2).transpose(1, 2)  # (B, 256, 384)
        kr = d16_ref.flatten(2).transpose(1, 2)      # (B, 256, 384)
        Q = F.normalize(self.q_proj(qc), dim=-1)     # (B, 256, D)
        K = F.normalize(self.k_proj(kr), dim=-1)     # (B, 256, D)
        V = self.v_proj(kr)                          # (B, 256, D)

        sim = torch.bmm(Q, K.transpose(1, 2))        # (B, 256, 256)
        attn = torch.softmax(sim * 12.0, dim=-1)
        R = torch.bmm(attn, V)                       # (B, 256, D)

        if self.use_conf_gate:
            max_sim = sim.max(dim=-1, keepdim=True).values  # (B, 256, 1)
            g_i = torch.sigmoid(self.conf_temp * (max_sim + self.conf_bias))
            u_glob = self.u_proj(u_style).unsqueeze(1).expand_as(R)
            R = g_i * R + (1.0 - g_i) * u_glob

        R_map = R.transpose(1, 2).view(B, -1, H, W)
        # style 或 content 被 dropout 时不注入局部参考残差
        mask = smask2.view(B, 1, 1, 1) * cmask4
        return h16 + self.out_proj(R_map) * mask


class SLFUNetRefAttn(SLFUNet):
    """无类别 (onehot=0) + 可选 16×16 Reference Cross-Attention 的 SLFUNet。"""

    def __init__(self, base_dim: int = 144, u_dim: int = 128, cond_dim: int = 288,
                 use_attn: bool = True, use_conf_gate: bool = False):
        super().__init__(base_dim=base_dim, u_dim=u_dim, cond_dim=cond_dim)
        self.use_attn = use_attn
        if use_attn:
            self.ref_attn16 = LightRefAttn16(h_dim=base_dim * 3, dino_dim=384, attn_dim=128,
                                             use_conf_gate=use_conf_gate)

    def forward(self, z_tau, y16_lcn, dino_16, style_id, u_vivid, tau, *,
                drop_style=None, drop_content=None, layout_scale=None,
                d16_ref: torch.Tensor | None = None):
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

        # 正式无类别 One-Shot: onehot(5) 恒零
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
        if self.use_attn and d16_ref is not None:
            h16 = self.ref_attn16(h16, dino_16, d16_ref, u_vivid, smask2, cmask4)
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


ARM_SPECS = {
    "R4_global":   dict(use_attn=False, use_conf=False, cross_prob=0.0),
    "R5_refattn":  dict(use_attn=True,  use_conf=False, cross_prob=0.0),
    "R6_gated_m0": dict(use_attn=True,  use_conf=True,  cross_prob=0.0),
    "M1_cross25":  dict(use_attn=True,  use_conf=True,  cross_prob=0.25),
    "M2_cross50":  dict(use_attn=True,  use_conf=True,  cross_prob=0.50),
}


def train_attn_cross_arm(bundle: dict, c21: dict, cross_pack: dict | None,
                         arm: str, updates: int = 1200, micro: int = 16, accum: int = 3,
                         device: str = "cuda") -> Path:
    spec = ARM_SPECS[arm]
    out_p = OUT / "oneshot_attn" / f"{arm}.pt"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if out_p.exists():
        print(f"[V23 {arm}] Reusing {out_p}", flush=True)
        return out_p

    if spec["cross_prob"] > 0 and cross_pack is None:
        raise RuntimeError(f"{arm} 需要 train_cross_pairs_1000.pt")

    seed_all(20261700)
    tr = bundle["v18_cache"]["train"]
    z_all = tr["z"]
    sid_all = tr["sid"].long()
    d16_all = tr["dino_16"]
    y16_all = coarse_y16_lcn(tr["y_64"].float()).half()
    nbr_idx = c21["train"]["nbr_idx"]

    fiber_enc = StyleFiberEncoder(in_dim=bundle["fnd"]["tr_fib"].shape[1], dim=128).to(device).eval()
    fiber_enc.load_state_dict(bundle["fnd"]["fiber_state"])
    with torch.no_grad():
        u_all = fiber_enc.encode(bundle["fnd"]["tr_fib"].to(device).float())
    del fiber_enc

    fm = SLFUNetRefAttn(base_dim=144, u_dim=128, use_attn=spec["use_attn"], use_conf_gate=spec["use_conf"]).to(device)
    fm.load_state_dict(torch.load(DUAL_CKPT, map_location="cpu", weights_only=False)["ema"], strict=False)
    adapter = RefAdapter(dim=128).to(device)

    for p in fm.parameters():
        p.requires_grad_(False)
    unfreeze_prefixes = ("cond_mlp.", "dino_proj.", "mid1.", "mid_attn.", "mid2.",
                         "up32.", "skip32.", "dec32.", "up64.", "skip64.", "dec64.", "out_norm.", "out_conv.")
    fm_base_params, new_params = [], list(adapter.parameters())
    for n, p in fm.named_parameters():
        if n.startswith("ref_attn16."):
            p.requires_grad_(True)
            new_params.append(p)
        elif n.startswith(unfreeze_prefixes):
            p.requires_grad_(True)
            fm_base_params.append(p)

    opt = torch.optim.AdamW([
        {"params": new_params, "lr": 1.5e-4},
        {"params": fm_base_params, "lr": 3e-5},
    ], weight_decay=1e-3)
    base_lrs = [1.5e-4, 3e-5]

    ema_fm = copy.deepcopy(fm).eval()
    ema_ad = copy.deepcopy(adapter).eval()
    for p in list(ema_fm.parameters()) + list(ema_ad.parameters()):
        p.requires_grad_(False)

    t0 = time.time()
    n_cross_pool = len(cross_pack["idx_c"]) if cross_pack is not None else 0

    for step in range(1, updates + 1):
        prog = step / max(1, updates)
        scale = 0.2 + 0.4 * (1.0 + math.cos(math.pi * prog))
        for pg, blr in zip(opt.param_groups, base_lrs):
            pg["lr"] = blr * scale
        opt.zero_grad(set_to_none=True)

        loss_v = 0.0
        for _ in range(accum):
            n_cr = int(round(micro * spec["cross_prob"]))
            n_self = micro - n_cr

            # 自配对样本
            idx_s = torch.randint(0, len(z_all), (n_self,))
            perm_nbr = torch.rand(n_self, 12).topk(1, dim=1).indices
            ref_s = nbr_idx[idx_s[:, None], perm_nbr].squeeze(1)

            ic_list = [idx_s]
            ir_list = [ref_s]
            z1_list = [z_all[idx_s].to(device, non_blocking=True).float()]

            if n_cr > 0:
                pick = torch.randint(0, n_cross_pool, (n_cr,))
                ic_cr = cross_pack["idx_c"][pick]
                ir_cr = cross_pack["idx_r"][pick]
                z1_cr = cross_pack["z_cr"][pick].to(device, non_blocking=True).float()
                ic_list.append(ic_cr)
                ir_list.append(ir_cr)
                z1_list.append(z1_cr)

            ic = torch.cat(ic_list, 0)
            ir = torch.cat(ir_list, 0)
            z1 = torch.cat(z1_list, 0)
            B = len(ic)

            sid = sid_all[ir].to(device, non_blocking=True)
            d16 = d16_all[ic].to(device, non_blocking=True).float()
            d16_r = d16_all[ir].to(device, non_blocking=True).float()
            y16 = y16_all[ic].to(device, non_blocking=True).float()
            u_style = adapter(u_all[ir], u_all[ic])

            d16_aug = d16 + 0.08 * torch.randn_like(d16)
            d16_aug = d16_aug - d16_aug.mean(dim=(2, 3), keepdim=True)
            y16_aug = (y16 * (0.85 + 0.30 * torch.rand(B, 1, 1, 1, device=device)) + 0.04 * torch.randn_like(y16)).clamp(-1.1, 1.1)

            ds = torch.rand(B, device=device) < 0.12
            dc = torch.zeros(B, dtype=torch.bool, device=device)
            lscale = torch.ones(B, device=device)
            z0 = torch.randn_like(z1)
            tau = torch.rand(B, device=device)
            zt = (1.0 - tau).view(B, 1, 1, 1) * z0 + tau.view(B, 1, 1, 1) * z1
            target_v = z1 - z0

            with autocast(device):
                pred_v = fm(zt, y16_aug, d16_aug, sid, u_style, tau,
                            drop_style=ds, drop_content=dc, layout_scale=lscale, d16_ref=d16_r)
                loss = F.mse_loss(pred_v.float(), target_v) / accum
            loss.backward()
            loss_v += float(loss.item())

        all_p = [p for g in opt.param_groups for p in g["params"]]
        nn.utils.clip_grad_norm_(all_p, 1.0)
        opt.step()
        with torch.no_grad():
            for pe, pw in zip(ema_fm.parameters(), fm.parameters()):
                pe.mul_(0.996).add_(pw, alpha=0.004)
            for pe, pw in zip(ema_ad.parameters(), adapter.parameters()):
                pe.mul_(0.996).add_(pw, alpha=0.004)

        if step == 1 or step % 300 == 0 or step == updates:
            print(f"[V23 {arm}] {step:4d}/{updates} loss={loss_v:.4f} ({time.time()-t0:.0f}s)", flush=True)

    torch.save({"fm": ema_fm.state_dict(), "adapter": ema_ad.state_dict(), "spec": spec, "arm": arm}, out_p)
    print(f"[V23 {arm}] Saved {out_p} in {(time.time()-t0)/60:.1f} min", flush=True)
    return out_p


@torch.no_grad()
def _sg5_ref_d16(bundle: dict, c21: dict, device: str = "cuda") -> torch.Tensor:
    """获取 SG-5 5 张参考图的 16×16 DINO patch 特征 (5, 384, 16, 16)（不读类别 ID）。"""
    if "d16" in c21["sg5"]:
        return c21["sg5"]["d16"].float()
    # 从 q32 (5, 384, 32, 32) 下采样或从 sg5 原始图提取
    if "q32" in c21["sg5"]:
        return F.avg_pool2d(c21["sg5"]["q32"].float(), 2, 2)
    raise KeyError("sg5 cache lacks d16/q32")


@torch.no_grad()
def eval_attn_cross_arm(bundle: dict, c21: dict, ckpt_p: Path, arm: str, device: str = "cuda") -> list[dict]:
    spec = ARM_SPECS[arm]
    tasks, z_src_750, _ = tasks_and_sources()
    dec = load_s10_decoder(device)

    ck = torch.load(ckpt_p, map_location="cpu", weights_only=False)
    fm = SLFUNetRefAttn(base_dim=144, u_dim=128, use_attn=spec["use_attn"], use_conf_gate=spec["use_conf"]).to(device).eval()
    fm.load_state_dict(ck["fm"], strict=True)
    ad = RefAdapter(dim=128).to(device).eval()
    ad.load_state_dict(ck["adapter"], strict=True)

    fiber_enc = StyleFiberEncoder(in_dim=bundle["fnd"]["tr_fib"].shape[1], dim=128).to(device).eval()
    fiber_enc.load_state_dict(bundle["fnd"]["fiber_state"])
    u_src_750 = fiber_enc.encode(bundle["fnd"]["test750_fib"].to(device).float()).cpu()
    del fiber_enc

    if "y_64" in bundle["test_struct"]:
        y16_750 = coarse_y16_lcn(bundle["test_struct"]["y_64"].float()).half()
    else:
        w_l = bundle["v18_cache"]["luma_proj"]["w"].float().view(1, 4, 1, 1)
        b_l = bundle["v18_cache"]["luma_proj"]["b"].float()
        y16_750 = coarse_y16_lcn(((z_src_750.float() * w_l).sum(1, keepdim=True) + b_l).clamp(0.0, 1.0)).half()
    eps_750 = torch.randn(z_src_750.shape, generator=torch.Generator().manual_seed(20261185))
    u_sg5 = c21["sg5"]["u"].float()
    d16_sg5 = _sg5_ref_d16(bundle, c21, device=device)

    res = []
    for tau0, anc_tag in ((0.0, "noise"), (0.40, "srcA0")):
        name = f"{arm}_{anc_tag}"
        cfg_hash = config_hash({"arm": name, "ckpt": file_sha(ckpt_p)[:12], "tau0": tau0})
        if (OUT / "raw" / name / "meta.json").exists() and validate_raw_meta(read_json(OUT / "raw" / name / "meta.json")):
            s = read_json(OUT / f"measured/{name}/summary_v23.json")
        else:
            wr = RawWriterV23(name, dec=dec, device=device)
            t0 = time.time()
            for st in range(0, 750, 25):
                sl = slice(st, min(st + 25, 750))
                grp = tasks[sl]
                B = len(grp)
                zs = z_src_750[sl].to(device).float()
                sid_p = torch.tensor([t["source_style_id"] for t in grp], device=device)
                tid_t = torch.tensor([t["target_style_id"] for t in grp], device=device)
                same = sid_p == tid_t

                u_style = ad(u_sg5[tid_t.cpu()].to(device), u_src_750[sl].to(device))
                d16_r = d16_sg5[tid_t.cpu()].to(device)
                y16 = y16_750[sl].to(device).float()
                d16 = bundle["test_struct"]["dino_16"][sl].to(device).float()
                eps_b = eps_750[sl].to(device).float()
                z = (1.0 - tau0) * eps_b + tau0 * zs if tau0 > 0 else eps_b

                f0 = torch.zeros(B, dtype=torch.bool, device=device)
                f1 = torch.ones(B, dtype=torch.bool, device=device)
                l1 = torch.ones(B, device=device)
                sid0 = torch.zeros(B, dtype=torch.long, device=device)
                steps = 10
                dt = (1.0 - tau0) / float(steps)

                def vel(zz, prog):
                    tv = torch.full((B,), tau0 + prog * (1.0 - tau0), device=device)
                    zz2 = torch.cat([zz, zz], 0)
                    y2, d2, dr2 = y16.repeat(2, 1, 1, 1), d16.repeat(2, 1, 1, 1), d16_r.repeat(2, 1, 1, 1)
                    s2, u2, t2 = sid0.repeat(2), u_style.repeat(2, 1), tv.repeat(2)
                    ds2, dc2 = torch.cat([f0, f1], 0), torch.cat([f0, f0], 0)
                    l2 = torch.cat([l1, l1], 0)
                    with autocast(device):
                        v_sc, v_0c = fm(zz2, y2, d2, s2, u2, t2, drop_style=ds2, drop_content=dc2,
                                        layout_scale=l2, d16_ref=dr2).float().chunk(2, 0)
                    return v_0c + 2.4 * (v_sc - v_0c)

                for i in range(steps):
                    v1 = vel(z, i / float(steps))
                    zm = z + 0.5 * dt * v1
                    v2 = vel(zm, (i + 0.5) / float(steps))
                    z = z + dt * v2
                z_out = torch.where(same.view(B, 1, 1, 1), zs, z)
                wr.write_batch(z_out, grp)
            meta = wr.finalize(cfg_hash=cfg_hash, arm=name, spec=spec, tau0=tau0, wall_sec=round(time.time() - t0, 1))
            assert validate_raw_meta(meta)
            s = run_measure_v23(name, device=device)
            export_raw_pngs(OUT / f"raw/{name}", OUT / f"png_750/{name}", max_workers=8)
        c = s["cross600"]
        res.append({"arm": name, "csd_top1": round(c["csd_top1"], 5), "dino_s": round(c["dino_s"], 5),
                    "dino_c": round(c["dino_c"], 5), "style_affinity": round(c["style_affinity"], 5),
                    "lpips": round(c["lpips"], 5), "pred_eq_src": round(c.get("pred_eq_src", 0.0), 4)})
    del fm, ad
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--arms", default="R4_global,R5_refattn,R6_gated_m0,M1_cross25,M2_cross50")
    ap.add_argument("--updates", type=int, default=1200)
    args = ap.parse_args()

    sanitize_cuda_env()
    bundle = build_or_load_v19_foundation(args.pool, device=args.device)
    c21 = torch.load(ROOT / "runs/v21/cache/v21_stiefel_ref_cache.pt", map_location="cpu", weights_only=False)
    cross_p = OUT / "cache/train_cross_pairs_1000.pt"
    cross_pack = torch.load(cross_p, map_location="cpu", weights_only=False) if cross_p.exists() else None

    results = []
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        if ARM_SPECS[arm]["cross_prob"] > 0 and cross_pack is None:
            print(f"[V23 {arm}] SKIP (waiting for {cross_p})", flush=True)
            continue
        ckpt_p = train_attn_cross_arm(bundle, c21, cross_pack, arm, updates=args.updates, device=args.device)
        results.extend(eval_attn_cross_arm(bundle, c21, ckpt_p, arm, device=args.device))

    save_json(OUT / "p2_attn_cross.json", {"updated_utc": utc(), "results": results})
    print("\n=== V23 P2 Reference Attention + 跨风格配对汇总 ===", flush=True)
    for r in results:
        print(f"{r['arm']:20s}: Top1={r['csd_top1']*100:5.2f}% DINO-S={r['dino_s']:.4f} DINO-C={r['dino_c']:.4f} LPIPS={r['lpips']:.4f}", flush=True)


if __name__ == "__main__":
    main()
