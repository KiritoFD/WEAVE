"""V23 P2 §3.4 (3060): 无五分类监督外观直通分支 (Direct Oklab27 + DINO768 -> 128D Residual) + M3 跨风格 Anchor-Bridge。

解决 SG-25 OOD 25类诊断发现的问题:
- 仅依赖 5类监督 StyleFiberEncoder 会在 OOD 新风格 (二次元/水墨/赛博朋克) 上丧失原色盘与纹理统计。
- 本脚本从 M3_ab_cross50 初始化，新增零初始化的外观直通残差投影 DirectAppProj: 795D (Oklab 27D + DINO RGB 768D 标准化统计) -> 128D，
  让任意参考图的真实色盘与纹理统计不经 5 分类瓶颈直接注入 u_style。

训练 1200 updates (50% 跨风格 Anchor-Bridge + 50% 自配对)，之后直接跑:
1. WEAVE test750 (srcA0 tau0=0.40 & tau0=0.50);
2. StyleGallery 25 类 125-pair 端到端 OOD 评测 (验证 DINO-S/CLIP-S/Gram 是否显著反超 M3)!

用法 (3060):
  python -m exp.v23.p2_direct_app --pool ../pseudo_gt_24h/orig_mean_fp16
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
from .p2_attn_cross import SLFUNetRefAttn, _sg5_ref_d16
from .p2_sg_bench import build_sg_25_manifest, load_a0_or_sd0_encoder, encode_rgb_sd0
from exp.v18 import tasks_and_sources
from exp.v18.structure import extract_whitened_dino_layout
from exp.v19.fiber_duality import StyleFiberEncoder, build_or_load_v19_foundation, extract_color_fiber_27d, extract_dino_rgb_moments_768d
from exp.v20.slf_flow import coarse_y16_lcn
from exp.v22.ref_adapter import RefAdapter
from src.utils import FeatureStack, decode_s10, load_s10_decoder
from src.utils.protocol import load_vgg16_features_offline
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF

M3_CKPT = OUT / "oneshot_attn/M3_ab_cross50.pt"


class DirectAppProj(nn.Module):
    """零初始化的 795D 无分类外观统计残差投影 (u_style = Normalize(u_adapter + gamma * W(fib_norm_ref - fib_norm_src)))。"""

    def __init__(self, in_dim: int = 795, out_dim: int = 128):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(0.2))
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.SiLU(inplace=True),
            nn.Linear(256, out_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, u_base: torch.Tensor, fib_n_ref: torch.Tensor, fib_n_src: torch.Tensor) -> torch.Tensor:
        delta = self.net(fib_n_ref - fib_n_src)
        return F.normalize(u_base + self.gamma * delta, dim=-1)


def train_direct_app(bundle: dict, c21: dict, cross_pack: dict,
                     updates: int = 1200, micro: int = 16, accum: int = 3, device: str = "cuda") -> Path:
    out_p = OUT / "oneshot_attn/O7_direct_app.pt"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if out_p.exists():
        print(f"[V23 O7] Reusing {out_p}", flush=True)
        return out_p

    seed_all(20262500)
    tr = bundle["v18_cache"]["train"]
    fnd = bundle["fnd"]
    z_all = tr["z"]
    sid_all = tr["sid"].long()
    d16_all = tr["dino_16"]
    y16_all = coarse_y16_lcn(tr["y_64"].float()).half()
    nbr_idx = c21["train"]["nbr_idx"]

    fib_tr = fnd["tr_fib"].float()
    fib_mean = fib_tr.mean(dim=0, keepdim=True).to(device)
    fib_std = fib_tr.std(dim=0, keepdim=True).clamp_min(1e-4).to(device)
    fib_n_all = ((fib_tr.to(device) - fib_mean) / fib_std)

    fiber_enc = StyleFiberEncoder(in_dim=fib_tr.shape[1], dim=128).to(device).eval()
    fiber_enc.load_state_dict(fnd["fiber_state"])
    with torch.no_grad():
        u_all = fiber_enc.encode(fib_tr.to(device))
    del fiber_enc

    ck = torch.load(M3_CKPT, map_location="cpu", weights_only=False)
    fm = SLFUNetRefAttn(base_dim=144, u_dim=128, use_attn=True, use_conf_gate=True).to(device)
    fm.load_state_dict(ck["fm"], strict=True)
    adapter = RefAdapter(dim=128).to(device)
    adapter.load_state_dict(ck["adapter"], strict=True)
    app_proj = DirectAppProj(in_dim=fib_tr.shape[1], out_dim=128).to(device)

    for p in fm.parameters():
        p.requires_grad_(False)
    unfreeze = ("cond_mlp.", "ref_attn16.", "mid1.", "mid_attn.", "mid2.", "dec32.", "dec64.", "out_norm.", "out_conv.")
    fm_p = []
    for n, p in fm.named_parameters():
        if n.startswith(unfreeze):
            p.requires_grad_(True)
            fm_p.append(p)

    opt = torch.optim.AdamW([
        {"params": list(app_proj.parameters()) + list(adapter.parameters()), "lr": 1.5e-4},
        {"params": fm_p, "lr": 2.5e-5},
    ], weight_decay=1e-3)
    base_lrs = [1.5e-4, 2.5e-5]

    ema_fm = copy.deepcopy(fm).eval()
    ema_ad = copy.deepcopy(adapter).eval()
    ema_ap = copy.deepcopy(app_proj).eval()
    for p in list(ema_fm.parameters()) + list(ema_ad.parameters()) + list(ema_ap.parameters()):
        p.requires_grad_(False)

    n_cross_pool = len(cross_pack["idx_c"])
    t0 = time.time()

    for step in range(1, updates + 1):
        prog = step / max(1, updates)
        scale = 0.2 + 0.4 * (1.0 + math.cos(math.pi * prog))
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

            pick = torch.randint(0, n_cross_pool, (n_cr,))
            ic = torch.cat([idx_s, cross_pack["idx_c"][pick]], 0)
            ir = torch.cat([ref_s, cross_pack["idx_r"][pick]], 0)
            z1 = torch.cat([z_all[idx_s].to(device).float(), cross_pack["z_cr"][pick].to(device).float()], 0)
            B = len(ic)

            sid = sid_all[ir].to(device)
            d16 = d16_all[ic].to(device).float()
            d16_r = d16_all[ir].to(device).float()
            y16 = y16_all[ic].to(device).float()
            u_base = adapter(u_all[ir], u_all[ic])
            u_style = app_proj(u_base, fib_n_all[ir], fib_n_all[ic])

            d16_aug = d16 + 0.08 * torch.randn_like(d16)
            d16_aug = d16_aug - d16_aug.mean(dim=(2, 3), keepdim=True)
            y16_aug = (y16 * (0.85 + 0.30 * torch.rand(B, 1, 1, 1, device=device)) + 0.04 * torch.randn_like(y16)).clamp(-1.1, 1.1)

            ds = torch.rand(B, device=device) < 0.12
            dc = torch.zeros(B, dtype=torch.bool, device=device)
            lscale = torch.ones(B, device=device)
            z0 = torch.randn_like(z1)

            # Anchor-Bridge 对跨风格样本 (tau0 in [0.25, 0.55])
            za = z_all[ic].to(device).float()
            tau_fm = torch.rand(B, device=device)
            zt_fm = (1.0 - tau_fm).view(B, 1, 1, 1) * z0 + tau_fm.view(B, 1, 1, 1) * z1
            v_fm = z1 - z0
            tau0 = 0.25 + 0.30 * torch.rand(B, device=device)
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
            for pe, pw in zip(ema_ap.parameters(), app_proj.parameters()):
                pe.mul_(0.996).add_(pw, alpha=0.004)

        if step == 1 or step % 300 == 0 or step == updates:
            print(f"[V23 O7] {step:4d}/{updates} loss={loss_v:.4f} gamma={float(app_proj.gamma.detach()):.3f} ({time.time()-t0:.0f}s)", flush=True)

    torch.save({
        "fm": ema_fm.state_dict(), "adapter": ema_ad.state_dict(), "app_proj": ema_ap.state_dict(),
        "fib_mean": fib_mean.cpu(), "fib_std": fib_std.cpu(),
    }, out_p)
    print(f"[V23 O7] Saved {out_p} in {(time.time()-t0)/60:.1f} min", flush=True)
    return out_p


@torch.no_grad()
def eval_o7_test750_and_sg25(bundle: dict, c21: dict, ckpt_p: Path, device: str = "cuda"):
    tasks, z_src_750, _ = tasks_and_sources()
    dec = load_s10_decoder(device)
    ck = torch.load(ckpt_p, map_location="cpu", weights_only=False)
    fm = SLFUNetRefAttn(base_dim=144, u_dim=128, use_attn=True, use_conf_gate=True).to(device).eval()
    fm.load_state_dict(ck["fm"], strict=True)
    ad = RefAdapter(dim=128).to(device).eval()
    ad.load_state_dict(ck["adapter"], strict=True)
    ap_m = DirectAppProj(in_dim=bundle["fnd"]["tr_fib"].shape[1], out_dim=128).to(device).eval()
    ap_m.load_state_dict(ck["app_proj"], strict=True)
    fib_mean, fib_std = ck["fib_mean"].to(device), ck["fib_std"].to(device)

    fiber_enc = StyleFiberEncoder(in_dim=bundle["fnd"]["tr_fib"].shape[1], dim=128).to(device).eval()
    fiber_enc.load_state_dict(bundle["fnd"]["fiber_state"])

    # 提取 SG-5 的原始 795D fib（不走 5 分类）
    stack = FeatureStack(device)
    dino_model = stack.dino
    stems = c21["sg5"]["stems"]
    clean_dir = ROOT / "datasets/distinct5_512/train_512clean"
    sg_fibs = []
    for st in stems:
        p = clean_dir / st.split("__")[0] / f"{st}.jpg"
        if not p.exists():
            p = list(clean_dir.rglob(f"{st}.*"))[0]
        xr = TF.to_tensor(Image.open(p).convert("RGB").resize((512, 512), Image.Resampling.BILINEAR)).unsqueeze(0).to(device)
        sg_fibs.append(torch.cat([extract_color_fiber_27d(xr), extract_dino_rgb_moments_768d(dino_model, xr, batch=1, device=device).to(device)], dim=-1))
    fib_sg5 = torch.cat(sg_fibs, 0)
    fib_n_sg5 = (fib_sg5 - fib_mean) / fib_std
    u_sg5 = fiber_enc.encode(fib_sg5.float())
    d16_sg5 = _sg5_ref_d16(bundle, c21, device=device).to(device)

    fib_750 = bundle["fnd"]["test750_fib"].to(device).float()
    fib_n_750 = (fib_750 - fib_mean) / fib_std
    u_src_750 = fiber_enc.encode(fib_750)

    w_l = bundle["v18_cache"]["luma_proj"]["w"].float().view(1, 4, 1, 1)
    b_l = bundle["v18_cache"]["luma_proj"]["b"].float()
    y16_750 = coarse_y16_lcn(((z_src_750.float() * w_l).sum(1, keepdim=True) + b_l).clamp(0.0, 1.0)).half()
    eps_750 = torch.randn(z_src_750.shape, generator=torch.Generator().manual_seed(20261185))

    # 1. WEAVE test750 (srcA0 tau0=0.40 & tau0=0.50)
    res_750 = []
    for tau0, wj, tag in ((0.40, 2.2, "t40"), (0.50, 2.0, "t50")):
        name = f"O7_directapp_{tag}"
        wr = RawWriterV23(name, dec=dec, device=device)
        for st in range(0, 750, 25):
            sl = slice(st, min(st + 25, 750))
            grp = tasks[sl]
            B = len(grp)
            zs = z_src_750[sl].to(device).float()
            sid_p = torch.tensor([t["source_style_id"] for t in grp], device=device)
            tid_t = torch.tensor([t["target_style_id"] for t in grp], device=device)
            same = sid_p == tid_t

            u_base = ad(u_sg5[tid_t], u_src_750[sl])
            u_style = ap_m(u_base, fib_n_sg5[tid_t], fib_n_750[sl])
            d16_r = d16_sg5[tid_t]
            y16 = y16_750[sl].to(device).float()
            d16 = bundle["test_struct"]["dino_16"][sl].to(device).float()
            eps_b = eps_750[sl].to(device).float()
            z = (1.0 - tau0) * eps_b + tau0 * zs

            f0, f1 = torch.zeros(B, dtype=torch.bool, device=device), torch.ones(B, dtype=torch.bool, device=device)
            l1, sid0 = torch.ones(B, device=device), torch.zeros(B, dtype=torch.long, device=device)
            steps = 10
            dt = (1.0 - tau0) / float(steps)
            for i in range(steps):
                def _v(zz, pr):
                    tv = torch.full((B,), tau0 + pr * (1.0 - tau0), device=device)
                    zz2 = torch.cat([zz, zz], 0)
                    with autocast(device):
                        v_sc, v_0c = fm(zz2, y16.repeat(2, 1, 1, 1), d16.repeat(2, 1, 1, 1), sid0.repeat(2),
                                        u_style.repeat(2, 1), tv.repeat(2),
                                        drop_style=torch.cat([f0, f1], 0), drop_content=torch.cat([f0, f0], 0),
                                        layout_scale=torch.cat([l1, l1], 0), d16_ref=d16_r.repeat(2, 1, 1, 1)).float().chunk(2, 0)
                    return v_0c + wj * (v_sc - v_0c)
                v1 = _v(z, i / steps)
                v2 = _v(z + 0.5 * dt * v1, (i + 0.5) / steps)
                z = z + dt * v2
            z_out = torch.where(same.view(B, 1, 1, 1), zs, z)
            wr.write_batch(z_out, grp)
        wr.finalize(arm=name, tau0=tau0, wj=wj)
        s = run_measure_v23(name, device=device)
        c = s["cross600"]
        res_750.append({"arm": name, "csd_top1": round(c["csd_top1"], 5), "dino_s": round(c["dino_s"], 5),
                        "dino_c": round(c["dino_c"], 5), "style_affinity": round(c["style_affinity"], 5), "lpips": round(c["lpips"], 5)})

    # 2. StyleGallery 25 类 125-pair 端到端全盲对比 (对比同 tau0=0.35, wj=2.2 的 M3)
    vae = load_a0_or_sd0_encoder(device)
    sg_root = Path("I:/Latent_Style/StyleGallery") if Path("I:/Latent_Style/StyleGallery").exists() else ROOT.parent / "StyleGallery"
    pairs = build_sg_25_manifest(sg_root, n_content=5)
    vgg = load_vgg16_features_offline(device).eval()

    def _vgg_gram(rgb01):
        mean = rgb01.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = rgb01.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x = (rgb01.to(device).float() - mean) / std
        grams = []
        for idx, m in enumerate(vgg):
            x = m(x)
            if idx in (3, 8, 15, 22):
                B_, C_, H_, W_ = x.shape
                f = x.view(B_, C_, H_ * W_)
                grams.append(torch.bmm(f, f.transpose(1, 2)) / (C_ * H_ * W_))
        return grams

    w_l_d, b_l_d = w_l.to(device), b_l.to(device)
    dc_l, ds_l, cs_l, lp_l, gr_l = [], [], [], [], []
    seed_all(20262200)
    tau0 = 0.35
    wj = 2.2
    out_dir = OUT / "sg25_outputs/O7_directapp"
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in pairs:
        im_c = Image.open(p["content_path"]).convert("RGB").resize((512, 512), Image.Resampling.BILINEAR)
        im_r = Image.open(p["style_path"]).convert("RGB").resize((512, 512), Image.Resampling.BILINEAR)
        xc, xr = TF.to_tensor(im_c).unsqueeze(0).to(device), TF.to_tensor(im_r).unsqueeze(0).to(device)
        zc = encode_rgb_sd0(vae, xc)
        y16 = coarse_y16_lcn(((zc * w_l_d).sum(1, keepdim=True) + b_l_d).clamp(0.0, 1.0))
        d16 = extract_whitened_dino_layout(dino_model, xc, batch=1, device=device)
        d16_r = extract_whitened_dino_layout(dino_model, xr, batch=1, device=device)
        fib_c = torch.cat([extract_color_fiber_27d(xc), extract_dino_rgb_moments_768d(dino_model, xc, batch=1, device=device).to(device)], dim=-1)
        fib_r = torch.cat([extract_color_fiber_27d(xr), extract_dino_rgb_moments_768d(dino_model, xr, batch=1, device=device).to(device)], dim=-1)
        u_style = ap_m(ad(fiber_enc.encode(fib_r.float()), fiber_enc.encode(fib_c.float())),
                       (fib_r - fib_mean) / fib_std, (fib_c - fib_mean) / fib_std)
        z = (1.0 - tau0) * torch.randn_like(zc) + tau0 * zc
        f0, f1 = torch.zeros(1, dtype=torch.bool, device=device), torch.ones(1, dtype=torch.bool, device=device)
        l1, sid0 = torch.ones(1, device=device), torch.zeros(1, dtype=torch.long, device=device)
        dt = (1.0 - tau0) / 10.0
        for i in range(10):
            def _v(zz, pr):
                t2 = torch.full((2,), tau0 + pr * (1.0 - tau0), device=device)
                with autocast(device):
                    v_sc, v_0c = fm(torch.cat([zz, zz], 0), y16.repeat(2, 1, 1, 1), d16.repeat(2, 1, 1, 1),
                                    sid0.repeat(2), u_style.repeat(2, 1), t2,
                                    drop_style=torch.cat([f0, f1], 0), drop_content=torch.cat([f0, f0], 0),
                                    layout_scale=torch.cat([l1, l1], 0), d16_ref=d16_r.repeat(2, 1, 1, 1)).float().chunk(2, 0)
                return v_0c + wj * (v_sc - v_0c)
            v1 = _v(z, i / 10.0)
            v2 = _v(z + 0.5 * dt * v1, (i + 0.5) / 10.0)
            z = z + dt * v2
        rgb_out = decode_s10(dec, z, batch=1, output_cpu=False).clamp(0.0, 1.0)
        pil_out = Image.fromarray((rgb_out[0].permute(1, 2, 0).cpu().numpy() * 255).round().clip(0, 255).astype("uint8"))
        pil_out.save(out_dir / f"{p['pair_id']}.png")
        fo = stack.extract([im_c, im_r, pil_out], batch=3)
        dc_l.append(float((fo["dino"][2] * fo["dino"][0]).sum()))
        ds_l.append(float((fo["dino"][2] * fo["dino"][1]).sum()))
        cs_l.append(float((fo["clip"][2] * fo["clip"][1]).sum()))
        lp_l.append(float(stack.lpips_vs((rgb_out.cpu() * 255).round().clamp(0, 255).byte(), xc.cpu(), batch=1).item()))
        gr_l.append(float(sum(F.mse_loss(a, b) for a, b in zip(_vgg_gram(rgb_out), _vgg_gram(xr))).item()))

    sg25_res = {
        "arm": "O7_directapp_t35", "n_pairs": len(pairs),
        "dino_s": round(float(np.mean(ds_l)), 4), "clip_s": round(float(np.mean(cs_l)), 4),
        "style_affinity": round(0.5 * (float(np.mean(ds_l)) + float(np.mean(cs_l))), 4),
        "dino_c": round(float(np.mean(dc_l)), 4), "lpips": round(float(np.mean(lp_l)), 4),
        "vgg_gram_loss": round(float(np.mean(gr_l)), 5),
    }
    print(f"[SG-Bench O7 DirectApp 125-Pair] {sg25_res}", flush=True)
    save_json(OUT / "p2_o7_direct_app.json", {"updated_utc": utc(), "test750": res_750, "sg25": sg25_res})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--updates", type=int, default=1200)
    args = ap.parse_args()

    sanitize_cuda_env()
    bundle = build_or_load_v19_foundation(args.pool, device=args.device)
    c21 = torch.load(ROOT / "runs/v21/cache/v21_stiefel_ref_cache.pt", map_location="cpu", weights_only=False)
    cross_pack = torch.load(OUT / "cache/train_cross_pairs_1000.pt", map_location="cpu", weights_only=False)

    ckpt_p = train_direct_app(bundle, c21, cross_pack, updates=args.updates, device=args.device)
    eval_o7_test750_and_sg25(bundle, c21, ckpt_p, device=args.device)


if __name__ == "__main__":
    main()
