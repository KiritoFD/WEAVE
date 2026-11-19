"""V24 P1/P3 (3060): O4-D 共享轻量残差编辑器的真 One-Shot 模型 (Content Anchor + E_s(x_ref) -> Delta_theta)。

统一 V24 §三 & §五 核心架构:
  z_out = z_c + Delta_theta(z_c, c_64, d_16, E_s(x_ref))
其中:
- 残差主干复用主线 PaletteTextureResidualNet (仅增加 1层轻量 16×16 Q/K=gray_dino, V=RGB_ConvNeXt Cross-Attention 将局部参考外观注入 Texture 分支!);
- 风格编码 E_s(x_ref) 完全不经过五分类监督: 直接融合参考图 Oklab27 + DINO_RGB768 线性外观编码 (128D) 与 16×16 RGB ConvNeXt 局部特征;
- 训练使用 O2 真参考条件无类别伪配对 (ref_teacher_pairs_2000.pt) + 实时参考自监督 SWD/Oklab/ConvNeXt 预算约束, 彻底摆脱五分类标签!

用法 (3060):
  python -m exp.v24.o4d_residual_oneshot --pool ../pseudo_gt_24h/orig_mean_fp16
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
from .p0_o1_o3_value import RefAttnV24, _sg5_cnx_v16, build_or_load_train_cnx_v16
from .p1_m3_m4_residual import ResBlock2d
from exp.v18 import tasks_and_sources
from exp.v18.structure import extract_whitened_dino_layout
from exp.v19.fiber_duality import build_or_load_v19_foundation, extract_color_fiber_27d, extract_dino_rgb_moments_768d
from exp.v20.submode_patchswd import MultiScalePatchSWDExtractor, load_convnext_tiny_offline, sliced_wasserstein_1d
from exp.v23.p2_attn_cross import _sg5_ref_d16
from exp.v23.p2_sg_bench import build_sg_25_manifest, encode_rgb_scaled, load_a0_or_sd0_encoder
from src.utils import FeatureStack, decode_s10, load_s10_decoder
from src.utils.structure import dwt2_haar, idwt2_haar


class OneShotResidualNet(nn.Module):
    """~1.3M 统一 One-Shot 残差编辑器: Palette+Texture 双路径 + 16×16 Q/K灰度DINO × V=RGB_ConvNeXt 局部参考注意力。"""

    def __init__(self, fib_dim: int = 795, cond_dim: int = 128, ch: int = 96):
        super().__init__()
        # 无五分类监督的直接参考编码器 E_s: (fib_ref - fib_src) + fib_ref -> 128D
        self.ref_enc = nn.Sequential(
            nn.Linear(fib_dim * 2, 256), nn.SiLU(inplace=True),
            nn.Linear(256, cond_dim), nn.SiLU(inplace=True),
        )
        # Palette 分支 (32x32 -> 16x16 -> 64x64)
        self.pal_in = nn.Conv2d(4, 48, 3, stride=2, padding=1)
        self.pal_down = nn.Conv2d(48, 64, 3, stride=2, padding=1)
        self.pal_blk = ResBlock2d(64, cond_dim)
        self.pal_out = nn.Conv2d(64, 4, 3, padding=1)
        nn.init.zeros_(self.pal_out.weight)
        nn.init.zeros_(self.pal_out.bias)

        # Texture 分支 + 16x16 局部参考注意力 (Q/K=gray_dino, V=RGB ConvNeXt 192ch)
        self.dino_reduce = nn.Conv2d(384, 48, 1)
        self.ref_attn16 = RefAttnV24(h_dim=48, qk_dim=384, v_dim=192, attn_dim=96,
                                     use_conf_gate=True, use_global_fib=True)
        self.tex_in = nn.Conv2d(4 + 4 + 48, ch, 3, padding=1)
        self.tex_blks = nn.ModuleList([ResBlock2d(ch, cond_dim) for _ in range(4)])
        self.tex_out = nn.Conv2d(ch, 4, 3, padding=1)
        nn.init.zeros_(self.tex_out.weight)
        nn.init.zeros_(self.tex_out.bias)

        self.alpha_ll = nn.Parameter(torch.tensor(0.25))
        self.alpha_m = nn.Parameter(torch.tensor(1.0))
        self.alpha_h = nn.Parameter(torch.tensor(1.0))

    def forward(self, z_c: torch.Tensor, c_64: torch.Tensor, d16_c: torch.Tensor,
                d16_r: torch.Tensor, v16_r: torch.Tensor, fib_n_c: torch.Tensor, fib_n_r: torch.Tensor,
                same_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cond = self.ref_enc(torch.cat([fib_n_r, fib_n_r - fib_n_c], dim=-1))

        hp = F.silu(self.pal_down(F.silu(self.pal_in(z_c))))
        hp = self.pal_blk(hp, cond)
        d_pal = F.interpolate(self.pal_out(hp), size=z_c.shape[-2:], mode="bilinear", align_corners=False)

        B = z_c.shape[0]
        d_feat = self.dino_reduce(d16_c)  # (B, 48, 16, 16)
        smask = torch.ones(B, 1, device=z_c.device)
        cmask = torch.ones(B, 1, 1, 1, device=z_c.device)
        d_feat_ref = self.ref_attn16(d_feat, d16_c, d16_r, v16_r, fib_n_r, smask, cmask)
        d_up = F.interpolate(d_feat_ref, size=z_c.shape[-2:], mode="bilinear", align_corners=False)

        ht = self.tex_in(torch.cat([z_c, c_64, d_up], dim=1))
        for blk in self.tex_blks:
            ht = blk(ht, cond)
        t_raw = self.tex_out(ht)
        ll, lh, hl, hh = dwt2_haar(t_raw)
        d_tex = idwt2_haar(self.alpha_ll * ll, self.alpha_m * lh, self.alpha_m * hl, self.alpha_h * hh)

        delta = d_pal + d_tex
        if same_mask is not None:
            m = (~same_mask).float().view(-1, 1, 1, 1)
            delta, d_pal, d_tex = delta * m, d_pal * m, d_tex * m
        return delta, d_pal, d_tex


def train_o4d(bundle: dict, v16_tr: torch.Tensor, dec, swd_ext: MultiScalePatchSWDExtractor,
              teacher_pack: dict | None, updates: int = 600, micro: int = 4, accum: int = 2,
              device: str = "cuda") -> Path:
    out_p = OUT / "o4d_residual/o4d_net.pt"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if out_p.exists():
        print(f"[V24 O4-D] Reusing {out_p}", flush=True)
        return out_p

    seed_all(20262900)
    tr = bundle["v18_cache"]["train"]
    fnd = bundle["fnd"]
    z_all = tr["z"]
    d16_all = tr["dino_16"]
    c64_all = tr["c_64"]

    fib_tr = fnd["tr_fib"].float()
    fib_mean = fib_tr.mean(0, keepdim=True).to(device)
    fib_std = fib_tr.std(0, keepdim=True).clamp_min(1e-4).to(device)
    fib_n_all = (fib_tr.to(device) - fib_mean) / fib_std

    net = OneShotResidualNet(fib_dim=fib_tr.shape[1], cond_dim=128, ch=96).to(device)
    print(f"[V24 O4-D] OneShotResidualNet params={sum(p.numel() for p in net.parameters())/1e6:.2f}M", flush=True)
    ema = copy.deepcopy(net).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(net.parameters(), lr=1.5e-4, betas=(0.9, 0.99), weight_decay=1e-4)
    lam_c = 1.6
    eps_budget = 0.085
    t0 = time.time()

    n_teach = len(teacher_pack["idx_c"]) if teacher_pack is not None else 0

    for step in range(1, updates + 1):
        lr = 2e-5 + 0.5 * (1.5e-4 - 2e-5) * (1.0 + math.cos(math.pi * step / max(1, updates)))
        for pg in opt.param_groups:
            pg["lr"] = lr
        opt.zero_grad(set_to_none=True)

        ls_v, lc_v = 0.0, 0.0
        for _ in range(accum):
            if n_teach > 0:
                pick = torch.randint(0, n_teach, (micro,))
                ic, ir = teacher_pack["idx_c"][pick], teacher_pack["idx_r"][pick]
                z_tgt_lat = teacher_pack["z_cr"][pick].to(device).float()
                use_lat_sup = True
            else:
                ic = torch.randint(0, len(z_all), (micro,))
                ir = (ic + torch.randint(1, len(z_all), (micro,))) % len(z_all)
                z_tgt_lat = None
                use_lat_sup = False

            z_src = z_all[ic].to(device).float()
            z_ref = z_all[ir].to(device).float()
            c64 = c64_all[ic].to(device).float()
            d16_c = d16_all[ic].to(device).float()
            d16_r = d16_all[ir].to(device).float()
            v16_r = v16_tr[ir].to(device).float()

            with torch.no_grad():
                rgb_src = decode_s10(dec, z_src, batch=micro, output_cpu=False)
                rgb_ref = decode_s10(dec, z_ref, batch=micro, output_cpu=False)
                toks_src = swd_ext.extract_tokens(rgb_src)
                toks_ref = swd_ext.extract_tokens(rgb_ref)
                pal_ref = extract_color_fiber_27d(rgb_ref)

            with autocast(device):
                delta, d_pal, d_tex = net(z_src, c64, d16_c, d16_r, v16_r, fib_n_all[ic], fib_n_all[ir])
                z_out = (z_src + delta).clamp(-5.0, 5.0)
                rgb_out = decode_s10(dec, z_out, batch=micro, enable_grad=True, output_cpu=False)
                toks_out = swd_ext.extract_tokens(rgb_out)

                l_swd = (
                    1.0 * sliced_wasserstein_1d(toks_out["rgb"], toks_ref["rgb"], swd_ext.p_rgb)
                    + 0.35 * sliced_wasserstein_1d(toks_out["s1"], toks_ref["s1"], swd_ext.p_s1)
                    + 0.45 * sliced_wasserstein_1d(toks_out["s2"], toks_ref["s2"], swd_ext.p_s2)
                    + 0.55 * sliced_wasserstein_1d(toks_out["s3"], toks_ref["s3"], swd_ext.p_s3)
                ).mean()
                l_pal = F.smooth_l1_loss(extract_color_fiber_27d(rgb_out), pal_ref)
                l_style = l_swd + 0.65 * l_pal
                if use_lat_sup:
                    ll_p, lh_p, hl_p, hh_p = dwt2_haar(z_out - z_src)
                    ll_t, lh_t, hl_t, hh_t = dwt2_haar(z_tgt_lat - z_src)
                    l_lat = (F.smooth_l1_loss(ll_p, ll_t) + F.smooth_l1_loss(lh_p, lh_t)
                             + F.smooth_l1_loss(hl_p, hl_t) + F.smooth_l1_loss(hh_p, hh_t))
                    l_style = l_style + 1.2 * l_lat

                l_c = 0.5 * (
                    (1.0 - (F.normalize(toks_out["s2"], dim=-1) * F.normalize(toks_src["s2"], dim=-1)).sum(-1)).mean()
                    + (1.0 - (F.normalize(toks_out["s3"], dim=-1) * F.normalize(toks_src["s3"], dim=-1)).sum(-1)).mean()
                )
                loss = (l_style + lam_c * l_c) / accum

            loss.backward()
            ls_v += float(l_style.item()) / accum
            lc_v += float(l_c.item()) / accum

        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe, pw in zip(ema.parameters(), net.parameters()):
                pe.mul_(0.996).add_(pw, alpha=0.004)
        if step % 10 == 0:
            lam_c = float(min(8.0, max(0.2, lam_c + 0.40 * (lc_v - eps_budget))))

        if step == 1 or step % 300 == 0 or step == updates:
            print(f"[V24 O4-D] {step:4d}/{updates} style={ls_v:.4f} lc={lc_v:.4f}(lam_c={lam_c:.2f}) ({time.time()-t0:.0f}s)", flush=True)

    torch.save({"ema": ema.state_dict(), "fib_mean": fib_mean.cpu(), "fib_std": fib_std.cpu()}, out_p)
    print(f"[V24 O4-D] Saved {out_p} in {(time.time()-t0)/60:.1f} min", flush=True)
    return out_p


@torch.no_grad()
def eval_o4d(bundle: dict, c21: dict, cnx, ckpt_p: Path, dec, device: str = "cuda"):
    tasks, z_src_750, _ = tasks_and_sources()
    ck = torch.load(ckpt_p, map_location="cpu", weights_only=False)
    net = OneShotResidualNet(fib_dim=bundle["fnd"]["tr_fib"].shape[1], cond_dim=128, ch=96).to(device).eval()
    net.load_state_dict(ck["ema"], strict=True)
    fib_mean, fib_std = ck["fib_mean"].to(device), ck["fib_std"].to(device)

    # 直接从 c21["sg5"]["z"] 经 S10 解码获取 SG-5 RGB (跨机器通用, 不依赖本地 train_512clean 目录)
    from exp.v18.structure import extract_whitened_dino_layout
    sg5_rgb = decode_s10(dec, c21["sg5"]["z"].to(device), batch=5, output_cpu=False).clamp(0.0, 1.0)
    stack = FeatureStack(device)
    dino_model = stack.dino
    d16_sg5 = extract_whitened_dino_layout(dino_model, sg5_rgb, batch=5, device=device).to(device)
    mean_im = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_im = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    x_sg = (F.interpolate(sg5_rgb.float(), size=(256, 256), mode="bilinear", align_corners=False) - mean_im) / std_im
    v16_sg5 = F.avg_pool2d(cnx[:4](x_sg), 2, 2)
    v16_sg5 = (v16_sg5 / v16_sg5.std(dim=(2, 3), keepdim=True).clamp_min(1e-4)).to(device)

    sg_fibs = torch.cat([
        extract_color_fiber_27d(sg5_rgb),
        extract_dino_rgb_moments_768d(dino_model, sg5_rgb, batch=5, device=device).to(device),
    ], dim=-1)
    fib_n_sg5 = (sg_fibs - fib_mean) / fib_std
    del stack
    torch.cuda.empty_cache()
    fib_n_750 = (bundle["fnd"]["test750_fib"].to(device).float() - fib_mean) / fib_std
    c64_750 = bundle["test_struct"]["c_64"].float()
    d16_750 = bundle["test_struct"]["dino_16"].float()

    res_750 = []
    for sc, tag in ((1.0, "s100"), (1.35, "s135")):
        arm = f"O4D_res_{tag}"
        wr = RawWriterV24(arm, dec=dec, device=device)
        t0 = time.time()
        for st in range(0, 750, 25):
            sl = slice(st, min(st + 25, 750))
            grp = tasks[sl]
            zs = z_src_750[sl].to(device).float()
            sid_p = torch.tensor([t["source_style_id"] for t in grp], device=device)
            tid_t = torch.tensor([t["target_style_id"] for t in grp], device=device)
            same = sid_p == tid_t
            delta, _, _ = net(zs, c64_750[sl].to(device).float(), d16_750[sl].to(device).float(),
                              d16_sg5[tid_t], v16_sg5[tid_t], fib_n_750[sl], fib_n_sg5[tid_t], same_mask=same)
            z_out = zs + sc * delta
            z_out[same] = zs[same]
            wr.write_batch(z_out, grp)
        wr.finalize(arm=arm, scale=sc, wall_sec=round(time.time() - t0, 1))
        s = run_measure_v24(arm, device=device)
        export_raw_pngs(OUT / f"raw/{arm}", OUT / f"png_750/{arm}", max_workers=8)
        c = s["cross600"]
        res_750.append({"arm": arm, "dino_c": round(c["dino_c"], 5), "dino_s": round(c["dino_s"], 5),
                        "lpips": round(c["lpips"], 5), "style_affinity": round(c["style_affinity"], 5),
                        "csd_top1": round(c["csd_top1"], 5)})

    save_json(OUT / "p1_o4d.json", {"updated_utc": utc(), "test750": res_750})
    print("\n=== V24 O4-D 统一轻量残差 One-Shot 汇总 ===", flush=True)
    for r in res_750:
        print(f"{r['arm']:16s}: DINO-C={r['dino_c']:.4f} DINO-S={r['dino_s']:.4f} LPIPS={r['lpips']:.4f} Top1={r['csd_top1']*100:5.2f}%", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--updates", type=int, default=600)
    args = ap.parse_args()

    sanitize_cuda_env()
    bundle = build_or_load_v19_foundation(args.pool, device=args.device)
    c21 = torch.load(ROOT / "runs/v21/cache/v21_stiefel_ref_cache.pt", map_location="cpu", weights_only=False)
    dec = load_s10_decoder(args.device)
    cnx = load_convnext_tiny_offline(args.device).eval()
    for p in cnx.parameters():
        p.requires_grad_(False)
    v16_tr = build_or_load_train_cnx_v16(bundle, dec, cnx, device=args.device)
    swd_ext = MultiScalePatchSWDExtractor(n_slices=96).to(args.device).eval()
    for p in swd_ext.parameters():
        p.requires_grad_(False)

    teach_p = OUT / "cache/ref_teacher_pairs_2000.pt"
    teacher_pack = torch.load(teach_p, map_location="cpu", weights_only=False) if teach_p.exists() else None
    ckpt_p = train_o4d(bundle, v16_tr, dec, swd_ext, teacher_pack, updates=args.updates, device=args.device)
    del swd_ext
    torch.cuda.empty_cache()
    eval_o4d(bundle, c21, cnx, ckpt_p, dec, device=args.device)


if __name__ == "__main__":
    main()
