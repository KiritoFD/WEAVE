"""V24 P0-P1 (3060): O1 / O2 / O3 参考 Value 消融（Q/K 灰度 DINO 语义匹配 vs V = RGB ConvNeXt 外观特征）。

严格对照 V24 §O1:
- O1_gray_dino     : Q/K=gray_dino16, V=gray_dino16 (V23 对照)
- O2_rgb_convnext  : Q/K=gray_dino16, V=RGB ConvNeXt_s2 (192ch 下采样至 16x16, 真实局部颜色纹理)
- O3_rgb_cnx_gated : O2 + Oklab27+DINO768 全局外观投影 + 匹配置信度门控 (不靠 5类探针)

三臂统一从 V22 Dual (prism_dual_branch.pt) 启动, onehot=0, srcA0 50% Anchor-Bridge, 1200 updates × micro=12 × accum=4,
评测: WEAVE test750 (srcA0 tau0=0.45 & 0.55) + StyleGallery 25类 125-pair OOD + §8.4 敏感度。

用法 (3060):
  python -m exp.v24.p0_o1_o3_value --pool ../pseudo_gt_24h/orig_mean_fp16
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
from PIL import Image
import torchvision.transforms.functional as TF

from . import OUT, ROOT, RawWriterV24, autocast, config_hash, export_raw_pngs, file_sha, read_json, run_measure_v24, sanitize_cuda_env, save_json, seed_all, utc, validate_raw_meta
from exp.v18 import tasks_and_sources
from exp.v18.ccf_model import sinusoidal_time_emb
from exp.v18.structure import extract_whitened_dino_layout
from exp.v19.fiber_duality import StyleFiberEncoder, build_or_load_v19_foundation, extract_color_fiber_27d, extract_dino_rgb_moments_768d
from exp.v20.slf_flow import SLFUNet, coarse_y16_lcn
from exp.v20.submode_patchswd import load_convnext_tiny_offline
from exp.v22.ref_adapter import RefAdapter
from exp.v23.p2_attn_cross import _sg5_ref_d16
from src.utils import decode_s10, load_s10_decoder

DUAL_CKPT = ROOT / "runs/v22/attn_flow/prism_dual_branch.pt"


class RefAttnV24(nn.Module):
    """Q/K 来自灰度 DINO, V 可选来自灰度 DINO (O1) 或 RGB ConvNeXt s2 (O2/O3) + 可选全局外观置信度门控。"""

    def __init__(self, h_dim: int = 432, qk_dim: int = 384, v_dim: int = 192,
                 attn_dim: int = 128, use_conf_gate: bool = False, use_global_fib: bool = False):
        super().__init__()
        self.use_conf_gate = use_conf_gate
        self.use_global_fib = use_global_fib
        self.q_proj = nn.Linear(qk_dim, attn_dim)
        self.k_proj = nn.Linear(qk_dim, attn_dim)
        self.v_proj = nn.Linear(v_dim, attn_dim)
        if use_global_fib:
            self.glob_proj = nn.Linear(795, attn_dim)
        else:
            self.glob_proj = nn.Linear(128, attn_dim)
        self.out_proj = nn.Conv2d(attn_dim, h_dim, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.conf_temp = nn.Parameter(torch.tensor(8.0))
        self.conf_bias = nn.Parameter(torch.tensor(-0.3))

    def forward(self, h16: torch.Tensor, d16_c: torch.Tensor, d16_r: torch.Tensor,
                v16_r: torch.Tensor, u_glob: torch.Tensor,
                smask2: torch.Tensor, cmask4: torch.Tensor) -> torch.Tensor:
        B, _, H, W = h16.shape
        Q = F.normalize(self.q_proj(d16_c.flatten(2).transpose(1, 2)), dim=-1)
        K = F.normalize(self.k_proj(d16_r.flatten(2).transpose(1, 2)), dim=-1)
        V = self.v_proj(v16_r.flatten(2).transpose(1, 2))
        sim = torch.bmm(Q, K.transpose(1, 2))
        R = torch.bmm(torch.softmax(sim * 12.0, dim=-1), V)

        if self.use_conf_gate:
            max_sim = sim.max(dim=-1, keepdim=True).values
            g_i = torch.sigmoid(self.conf_temp * (max_sim + self.conf_bias))
            u_g = self.glob_proj(u_glob).unsqueeze(1).expand_as(R)
            R = g_i * R + (1.0 - g_i) * u_g

        R_map = R.transpose(1, 2).view(B, -1, H, W)
        return h16 + self.out_proj(R_map) * (smask2.view(B, 1, 1, 1) * cmask4)


class SLFUNetV24OneShot(SLFUNet):
    def __init__(self, v_dim: int, use_conf_gate: bool, use_global_fib: bool):
        super().__init__(base_dim=144, u_dim=128)
        self.ref_attn16 = RefAttnV24(h_dim=432, qk_dim=384, v_dim=v_dim, attn_dim=128,
                                     use_conf_gate=use_conf_gate, use_global_fib=use_global_fib)

    def forward(self, z_tau, y16_lcn, dino_16, style_id, u_vivid, tau, *,
                drop_style=None, drop_content=None, layout_scale=None,
                d16_ref=None, v16_ref=None, u_glob=None):
        B, dev = z_tau.shape[0], z_tau.device
        if drop_style is None:
            drop_style = torch.zeros(B, dtype=torch.bool, device=dev)
        if drop_content is None:
            drop_content = torch.zeros(B, dtype=torch.bool, device=dev)
        cmask4 = (~drop_content).float().view(B, 1, 1, 1)
        smask2 = (~drop_style).float().view(B, 1)
        lscale = (~drop_content).float() if layout_scale is None else layout_scale.to(dev).float().view(B).clamp(0.0, 1.0) * (~drop_content).float()
        lscale4 = lscale.view(B, 1, 1, 1)

        d_feat = self.dino_proj(dino_16 * cmask4) * lscale4
        s16 = torch.cat([y16_lcn * lscale4, d_feat], dim=1)
        oh = torch.zeros(B, self.N_STYLE, device=dev)
        uv = u_vivid * smask2
        cond = self.cond_mlp(torch.cat([sinusoidal_time_emb(tau, 64), oh, uv, torch.stack([drop_style.float(), 1.0 - lscale], -1)], -1))

        h64 = self.in_conv(z_tau)
        for blk in self.enc64:
            h64 = blk(h64, cond)
        h32 = self.down32(h64)
        for blk in self.enc32:
            h32 = blk(h32, cond)
        h16 = self.down16(h32)
        h16 = self.mid1(h16, s16, cond)
        if d16_ref is not None and v16_ref is not None:
            h16 = self.ref_attn16(h16, dino_16, d16_ref, v16_ref, u_glob if u_glob is not None else u_vivid, smask2, cmask4)
        h16 = self.mid_attn(h16)
        h16 = self.mid2(h16, s16, cond)

        u32 = self.skip32(torch.cat([self.up32(h16.repeat_interleave(2, -2).repeat_interleave(2, -1)), h32], 1))
        for blk in self.dec32:
            u32 = blk(u32, cond)
        u64 = self.skip64(torch.cat([self.up64(u32.repeat_interleave(2, -2).repeat_interleave(2, -1)), h64], 1))
        for blk in self.dec64:
            u64 = blk(u64, cond)
        return self.out_conv(F.silu(self.out_norm(u64)))


@torch.no_grad()
def build_or_load_train_cnx_v16(bundle: dict, dec, cnx, device: str = "cuda") -> torch.Tensor:
    """一次性缓存训练集 4400 张图与 SG-5 的 RGB ConvNeXt_s2 (192, 16, 16) 外观特征。"""
    cache_p = OUT / "cache/train_cnx_v16.pt"
    cache_p.parent.mkdir(parents=True, exist_ok=True)
    if cache_p.exists():
        return torch.load(cache_p, map_location="cpu", weights_only=False)["v16_tr"]

    z_all = bundle["v18_cache"]["train"]["z"]
    cnx_s2 = cnx[:4]
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    outs = []
    for st in range(0, len(z_all), 32):
        zb = z_all[st:st + 32].to(device)
        rgb = decode_s10(dec, zb, batch=16, output_cpu=False)
        x = (F.interpolate(rgb.float(), size=(256, 256), mode="bilinear", align_corners=False) - mean) / std
        f2 = cnx_s2(x)  # (B, 192, 32, 32)
        v16 = F.avg_pool2d(f2, 2, 2)
        v16 = v16 / v16.std(dim=(2, 3), keepdim=True).clamp_min(1e-4)
        outs.append(v16.cpu().half())
    v16_tr = torch.cat(outs, 0)
    torch.save({"v16_tr": v16_tr}, cache_p)
    print(f"[V24 O1-O3] Cached train RGB ConvNeXt v16 {v16_tr.shape}", flush=True)
    return v16_tr


@torch.no_grad()
def _sg5_cnx_v16(c21: dict, cnx, device: str = "cuda") -> torch.Tensor:
    cnx_s2 = cnx[:4]
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    clean_dir = ROOT / "datasets/distinct5_512/train_512clean"
    imgs = []
    for st in c21["sg5"]["stems"]:
        p = clean_dir / st.split("__")[0] / f"{st}.jpg"
        if not p.exists():
            p = list(clean_dir.rglob(f"{st}.*"))[0]
        imgs.append(TF.to_tensor(Image.open(p).convert("RGB").resize((512, 512), Image.Resampling.BILINEAR)))
    rgb = torch.stack(imgs, 0).to(device)
    x = (F.interpolate(rgb.float(), size=(256, 256), mode="bilinear", align_corners=False) - mean) / std
    v16 = F.avg_pool2d(cnx_s2(x), 2, 2)
    return (v16 / v16.std(dim=(2, 3), keepdim=True).clamp_min(1e-4)).cpu()


O_SPECS = {
    "O1_gray_dino":     dict(v_type="dino", use_conf=False, use_glob_fib=False),
    "O2_rgb_convnext":  dict(v_type="cnx",  use_conf=False, use_glob_fib=False),
    "O3_rgb_cnx_gated": dict(v_type="cnx",  use_conf=True,  use_glob_fib=True),
}


def train_o_arm(bundle: dict, c21: dict, cross_pack: dict, v16_tr: torch.Tensor,
                arm: str, updates: int = 1200, micro: int = 12, accum: int = 4, device: str = "cuda") -> Path:
    spec = O_SPECS[arm]
    out_p = OUT / "oneshot_val" / f"{arm}.pt"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if out_p.exists():
        print(f"[V24 {arm}] Reusing {out_p}", flush=True)
        return out_p

    seed_all(20262700)
    tr = bundle["v18_cache"]["train"]
    fnd = bundle["fnd"]
    z_all, sid_all = tr["z"], tr["sid"].long()
    d16_all = tr["dino_16"]
    y16_all = coarse_y16_lcn(tr["y_64"].float()).half()
    nbr_idx = c21["train"]["nbr_idx"]

    fib_tr = fnd["tr_fib"].float()
    fib_mean = fib_tr.mean(0, keepdim=True).to(device)
    fib_std = fib_tr.std(0, keepdim=True).clamp_min(1e-4).to(device)
    fib_n_all = (fib_tr.to(device) - fib_mean) / fib_std

    fiber_enc = StyleFiberEncoder(in_dim=fib_tr.shape[1], dim=128).to(device).eval()
    fiber_enc.load_state_dict(fnd["fiber_state"])
    with torch.no_grad():
        u_all = fiber_enc.encode(fib_tr.to(device))
    del fiber_enc

    v_dim = 384 if spec["v_type"] == "dino" else 192
    fm = SLFUNetV24OneShot(v_dim=v_dim, use_conf_gate=spec["use_conf"], use_global_fib=spec["use_glob_fib"]).to(device)
    fm.load_state_dict(torch.load(DUAL_CKPT, map_location="cpu", weights_only=False)["ema"], strict=False)
    adapter = RefAdapter(dim=128).to(device)

    for p in fm.parameters():
        p.requires_grad_(False)
    unfreeze = ("cond_mlp.", "dino_proj.", "mid1.", "mid_attn.", "mid2.", "dec32.", "dec64.", "out_norm.", "out_conv.")
    fm_p, new_p = [], list(adapter.parameters())
    for n, p in fm.named_parameters():
        if n.startswith("ref_attn16."):
            p.requires_grad_(True); new_p.append(p)
        elif n.startswith(unfreeze):
            p.requires_grad_(True); fm_p.append(p)

    opt = torch.optim.AdamW([{"params": new_p, "lr": 1.5e-4}, {"params": fm_p, "lr": 3e-5}], weight_decay=1e-3)
    base_lrs = [1.5e-4, 3e-5]
    ema_fm, ema_ad = copy.deepcopy(fm).eval(), copy.deepcopy(adapter).eval()
    for p in list(ema_fm.parameters()) + list(ema_ad.parameters()):
        p.requires_grad_(False)

    n_cr_pool = len(cross_pack["idx_c"])
    t0 = time.time()
    for step in range(1, updates + 1):
        scale = 0.2 + 0.4 * (1.0 + math.cos(math.pi * step / max(1, updates)))
        for pg, blr in zip(opt.param_groups, base_lrs):
            pg["lr"] = blr * scale
        opt.zero_grad(set_to_none=True)
        loss_v = 0.0
        for _ in range(accum):
            n_cr = micro // 2
            n_self = micro - n_cr
            idx_s = torch.randint(0, len(z_all), (n_self,))
            perm_nbr = torch.rand(n_self, 12).topk(1, dim=1).indices
            ref_s = nbr_idx[idx_s[:, None], perm_nbr].squeeze(1)
            pick = torch.randint(0, n_cr_pool, (n_cr,))

            ic = torch.cat([idx_s, cross_pack["idx_c"][pick]], 0)
            ir = torch.cat([ref_s, cross_pack["idx_r"][pick]], 0)
            z1 = torch.cat([z_all[idx_s].to(device).float(), cross_pack["z_cr"][pick].to(device).float()], 0)
            B = len(ic)

            sid = sid_all[ir].to(device)
            d16 = d16_all[ic].to(device).float()
            d16_r = d16_all[ir].to(device).float()
            v16_r = (d16_r if spec["v_type"] == "dino" else v16_tr[ir].to(device).float())
            y16 = y16_all[ic].to(device).float()
            u_style = adapter(u_all[ir], u_all[ic])
            u_glob = fib_n_all[ir] if spec["use_glob_fib"] else u_style

            d16_aug = d16 + 0.08 * torch.randn_like(d16)
            d16_aug = d16_aug - d16_aug.mean(dim=(2, 3), keepdim=True)
            y16_aug = (y16 * (0.85 + 0.30 * torch.rand(B, 1, 1, 1, device=device)) + 0.04 * torch.randn_like(y16)).clamp(-1.1, 1.1)

            ds = torch.rand(B, device=device) < 0.12
            dc = torch.zeros(B, dtype=torch.bool, device=device)
            lscale = torch.ones(B, device=device)
            z0 = torch.randn_like(z1)
            za = z_all[ic].to(device).float()

            # 跨风格样本走 Source Anchor-Bridge (tau0 in [0.35, 0.60])
            tau_fm = torch.rand(B, device=device)
            zt_fm = (1.0 - tau_fm).view(B, 1, 1, 1) * z0 + tau_fm.view(B, 1, 1, 1) * z1
            v_fm = z1 - z0
            tau0 = 0.35 + 0.25 * torch.rand(B, device=device)
            z_init = (1.0 - tau0).view(B, 1, 1, 1) * z0 + tau0.view(B, 1, 1, 1) * za
            tau_ab = tau0 + (1.0 - tau0) * torch.rand(B, device=device)
            frac = ((tau_ab - tau0) / (1.0 - tau0).clamp_min(1e-4)).view(B, 1, 1, 1)
            zt_ab = z_init + frac * (z1 - z_init)
            v_ab = (z1 - z_init) / (1.0 - tau0).clamp_min(1e-4).view(B, 1, 1, 1)

            is_ab = torch.zeros(B, dtype=torch.bool, device=device)
            is_ab[n_self:] = True
            m4 = is_ab.view(B, 1, 1, 1)
            tau = torch.where(is_ab, tau_ab, tau_fm)
            zt = torch.where(m4, zt_ab, zt_fm)
            target_v = torch.where(m4, v_ab, v_fm)

            with autocast(device):
                pred_v = fm(zt, y16_aug, d16_aug, sid, u_style, tau,
                            drop_style=ds, drop_content=dc, layout_scale=lscale,
                            d16_ref=d16_r, v16_ref=v16_r, u_glob=u_glob)
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

        if step == 1 or step % 400 == 0 or step == updates:
            print(f"[V24 {arm}] {step:4d}/{updates} loss={loss_v:.4f} ({time.time()-t0:.0f}s)", flush=True)

    torch.save({"fm": ema_fm.state_dict(), "adapter": ema_ad.state_dict(),
                "fib_mean": fib_mean.cpu(), "fib_std": fib_std.cpu()}, out_p)
    print(f"[V24 {arm}] Saved {out_p} in {(time.time()-t0)/60:.1f} min", flush=True)
    return out_p


@torch.no_grad()
def eval_o_arm(bundle: dict, c21: dict, cnx, ckpt_p: Path, arm: str, device: str = "cuda") -> list[dict]:
    from src.utils import FeatureStack
    spec = O_SPECS[arm]
    tasks, z_src_750, _ = tasks_and_sources()
    dec = load_s10_decoder(device)

    ck = torch.load(ckpt_p, map_location="cpu", weights_only=False)
    v_dim = 384 if spec["v_type"] == "dino" else 192
    fm = SLFUNetV24OneShot(v_dim=v_dim, use_conf_gate=spec["use_conf"], use_global_fib=spec["use_glob_fib"]).to(device).eval()
    fm.load_state_dict(ck["fm"], strict=True)
    ad = RefAdapter(dim=128).to(device).eval()
    ad.load_state_dict(ck["adapter"], strict=True)
    fib_mean, fib_std = ck["fib_mean"].to(device), ck["fib_std"].to(device)

    fiber_enc = StyleFiberEncoder(in_dim=bundle["fnd"]["tr_fib"].shape[1], dim=128).to(device).eval()
    fiber_enc.load_state_dict(bundle["fnd"]["fiber_state"])
    u_src_750 = fiber_enc.encode(bundle["fnd"]["test750_fib"].to(device).float()).cpu()

    d16_sg5 = _sg5_ref_d16(bundle, c21, device=device)
    v16_sg5 = d16_sg5 if spec["v_type"] == "dino" else _sg5_cnx_v16(c21, cnx, device=device)

    # 提取 SG-5 的原始 795D 统计 (仅 O3 用)
    if spec["use_glob_fib"]:
        stack = FeatureStack(device)
        dino_model = stack.dino
        clean_dir = ROOT / "datasets/distinct5_512/train_512clean"
        sg_fibs = []
        for st in c21["sg5"]["stems"]:
            p = clean_dir / st.split("__")[0] / f"{st}.jpg"
            if not p.exists():
                p = list(clean_dir.rglob(f"{st}.*"))[0]
            xr = TF.to_tensor(Image.open(p).convert("RGB").resize((512, 512), Image.Resampling.BILINEAR)).unsqueeze(0).to(device)
            sg_fibs.append(torch.cat([extract_color_fiber_27d(xr), extract_dino_rgb_moments_768d(dino_model, xr, batch=1, device=device).to(device)], dim=-1))
        fib_sg5 = torch.cat(sg_fibs, 0)
        fib_n_sg5 = (fib_sg5 - fib_mean) / fib_std
        u_sg5 = fiber_enc.encode(fib_sg5.float()).cpu()
        del stack
        torch.cuda.empty_cache()
    else:
        u_sg5 = c21["sg5"]["u"].float()
        fib_n_sg5 = None
    del fiber_enc

    w_l = bundle["v18_cache"]["luma_proj"]["w"].float().view(1, 4, 1, 1)
    b_l = bundle["v18_cache"]["luma_proj"]["b"].float()
    y16_750 = coarse_y16_lcn(((z_src_750.float() * w_l).sum(1, keepdim=True) + b_l).clamp(0.0, 1.0)).half()
    eps_750 = torch.randn(z_src_750.shape, generator=torch.Generator().manual_seed(20261185))

    res = []
    for tau0, wj, tag in ((0.45, 2.2, "t45"), (0.55, 2.0, "t55")):
        name = f"{arm}_{tag}"
        cfg_hash = config_hash({"arm": name, "tau0": tau0, "wj": wj, "ckpt": file_sha(ckpt_p)[:12]})
        if (OUT / "raw" / name / "meta.json").exists() and validate_raw_meta(read_json(OUT / "raw" / name / "meta.json")):
            s = read_json(OUT / f"measured/{name}/summary_v24.json")
        else:
            wr = RawWriterV24(name, dec=dec, device=device)
            for st in range(0, 750, 25):
                sl = slice(st, min(st + 25, 750))
                grp = tasks[sl]
                B = len(grp)
                zs = z_src_750[sl].to(device).float()
                sid_p = torch.tensor([t["source_style_id"] for t in grp], device=device)
                tid_t = torch.tensor([t["target_style_id"] for t in grp], device=device)
                same = sid_p == tid_t

                u_style = ad(u_sg5[tid_t.cpu()].to(device), u_src_750[sl].to(device))
                u_glob = fib_n_sg5[tid_t.cpu()].to(device) if fib_n_sg5 is not None else u_style
                d16_r = d16_sg5[tid_t.cpu()].to(device)
                v16_r = v16_sg5[tid_t.cpu()].to(device)
                y16 = y16_750[sl].to(device).float()
                d16 = bundle["test_struct"]["dino_16"][sl].to(device).float()
                z = (1.0 - tau0) * eps_750[sl].to(device).float() + tau0 * zs

                f0, f1 = torch.zeros(B, dtype=torch.bool, device=device), torch.ones(B, dtype=torch.bool, device=device)
                l1, sid0 = torch.ones(B, device=device), torch.zeros(B, dtype=torch.long, device=device)
                steps = 10
                dt = (1.0 - tau0) / float(steps)
                for i in range(steps):
                    def _v(zz, pr):
                        tv = torch.full((B,), tau0 + pr * (1.0 - tau0), device=device)
                        zz2 = torch.cat([zz, zz], 0)
                        with autocast(device):
                            v_sc, v_0c = fm(zz2, y16.repeat(2, 1, 1, 1), d16.repeat(2, 1, 1, 1),
                                            sid0.repeat(2), u_style.repeat(2, 1), tv.repeat(2),
                                            drop_style=torch.cat([f0, f1], 0), drop_content=torch.cat([f0, f0], 0),
                                            layout_scale=torch.cat([l1, l1], 0),
                                            d16_ref=d16_r.repeat(2, 1, 1, 1), v16_ref=v16_r.repeat(2, 1, 1, 1),
                                            u_glob=u_glob.repeat(2, 1)).float().chunk(2, 0)
                        return v_0c + wj * (v_sc - v_0c)
                    v1 = _v(z, i / float(steps))
                    v2 = _v(z + 0.5 * dt * v1, (i + 0.5) / float(steps))
                    z = z + dt * v2
                z_out = torch.where(same.view(B, 1, 1, 1), zs, z)
                wr.write_batch(z_out, grp)
            meta = wr.finalize(cfg_hash=cfg_hash, tau0=tau0, wj=wj)
            assert validate_raw_meta(meta)
            s = run_measure_v24(name, device=device)
            export_raw_pngs(OUT / f"raw/{name}", OUT / f"png_750/{name}", max_workers=8)
        c = s["cross600"]
        res.append({"arm": name, "dino_c": round(c["dino_c"], 5), "dino_s": round(c["dino_s"], 5),
                    "lpips": round(c["lpips"], 5), "style_affinity": round(c["style_affinity"], 5),
                    "csd_top1": round(c["csd_top1"], 5)})
    del fm, ad
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--updates", type=int, default=1200)
    args = ap.parse_args()

    sanitize_cuda_env()
    bundle = build_or_load_v19_foundation(args.pool, device=args.device)
    c21 = torch.load(ROOT / "runs/v21/cache/v21_stiefel_ref_cache.pt", map_location="cpu", weights_only=False)
    cross_pack = torch.load(ROOT / "runs/v23/cache/train_cross_pairs_1000.pt", map_location="cpu", weights_only=False)
    dec = load_s10_decoder(args.device)
    cnx = load_convnext_tiny_offline(args.device).eval()
    for p in cnx.parameters():
        p.requires_grad_(False)
    v16_tr = build_or_load_train_cnx_v16(bundle, dec, cnx, device=args.device)

    results = []
    for arm in ("O1_gray_dino", "O2_rgb_convnext", "O3_rgb_cnx_gated"):
        ckpt_p = train_o_arm(bundle, c21, cross_pack, v16_tr, arm, updates=args.updates, device=args.device)
        results.extend(eval_o_arm(bundle, c21, cnx, ckpt_p, arm, device=args.device))

    save_json(OUT / "p0_o1_o3.json", {"updated_utc": utc(), "results": results})
    print("\n=== V24 O1/O2/O3 参考 Value 消融汇总 ===", flush=True)
    for r in results:
        print(f"{r['arm']:24s}: DINO-C={r['dino_c']:.4f} DINO-S={r['dino_s']:.4f} LPIPS={r['lpips']:.4f} Top1={r['csd_top1']*100:5.2f}%", flush=True)


if __name__ == "__main__":
    main()
