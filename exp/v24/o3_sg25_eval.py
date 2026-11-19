"""V24 P1 (3060): 在 O3_rgb_cnx_gated (t45 / t55) 上跑 StyleGallery 25类 OOD 125-pair 与 §8.4 敏感度测试。

直接与 V23 M3_ab_cross50 (DINO-S 0.2625, DINO-C 0.4177) 对比:
验证 V = RGB ConvNeXt + 全局 Oklab 置信度门控 在真实 25 类 OOD 上的迁移增益!

用法 (3060):
  python -m exp.v24.o3_sg25_eval --pool ../pseudo_gt_24h/orig_mean_fp16
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF

from . import OUT, ROOT, autocast, sanitize_cuda_env, save_json, seed_all, utc
from .p0_o1_o3_value import O_SPECS, SLFUNetV24OneShot, _sg5_cnx_v16
from exp.v18 import tasks_and_sources
from exp.v18.structure import extract_whitened_dino_layout
from exp.v19.fiber_duality import StyleFiberEncoder, build_or_load_v19_foundation, extract_color_fiber_27d, extract_dino_rgb_moments_768d
from exp.v20.slf_flow import coarse_y16_lcn
from exp.v20.submode_patchswd import load_convnext_tiny_offline
from exp.v22.ref_adapter import RefAdapter
from exp.v23.p2_attn_cross import _sg5_ref_d16
from exp.v23.p2_sg_bench import _find_sg_root, build_sg_25_manifest, encode_rgb_scaled, load_a0_or_sd0_encoder
from src.utils import FeatureStack, decode_s10, load_s10_decoder
from src.utils.protocol import load_vgg16_features_offline

O3_CKPT = OUT / "oneshot_val/O3_rgb_cnx_gated.pt"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    sanitize_cuda_env()
    sg_root = _find_sg_root()
    bundle = build_or_load_v19_foundation(args.pool, device=args.device)
    c21 = torch.load(ROOT / "runs/v21/cache/v21_stiefel_ref_cache.pt", map_location="cpu", weights_only=False)
    vae = load_a0_or_sd0_encoder(args.device)
    dec = load_s10_decoder(args.device)
    stack = FeatureStack(args.device)
    dino_model = stack.dino
    cnx = load_convnext_tiny_offline(args.device).eval()
    cnx_s2 = cnx[:4]
    vgg = load_vgg16_features_offline(args.device).eval()

    ck = torch.load(O3_CKPT, map_location="cpu", weights_only=False)
    fm = SLFUNetV24OneShot(v_dim=192, use_conf_gate=True, use_global_fib=True).to(args.device).eval()
    fm.load_state_dict(ck["fm"], strict=True)
    ad = RefAdapter(dim=128).to(args.device).eval()
    ad.load_state_dict(ck["adapter"], strict=True)
    fib_mean, fib_std = ck["fib_mean"].to(args.device), ck["fib_std"].to(args.device)

    fiber_enc = StyleFiberEncoder(in_dim=bundle["fnd"]["tr_fib"].shape[1], dim=128).to(args.device).eval()
    fiber_enc.load_state_dict(bundle["fnd"]["fiber_state"])
    w_l = bundle["v18_cache"]["luma_proj"]["w"].float().view(1, 4, 1, 1).to(args.device)
    b_l = bundle["v18_cache"]["luma_proj"]["b"].float().to(args.device)

    mean_im = torch.tensor([0.485, 0.456, 0.406], device=args.device).view(1, 3, 1, 1)
    std_im = torch.tensor([0.229, 0.224, 0.225], device=args.device).view(1, 3, 1, 1)

    def _cnx_v16(rgb01):
        x = (F.interpolate(rgb01.float(), size=(256, 256), mode="bilinear", align_corners=False) - mean_im) / std_im
        v = F.avg_pool2d(cnx_s2(x), 2, 2)
        return v / v.std(dim=(2, 3), keepdim=True).clamp_min(1e-4)

    def _vgg_gram(rgb01):
        x = (rgb01.to(args.device).float() - mean_im) / std_im
        grams = []
        for idx, m in enumerate(vgg):
            x = m(x)
            if idx in (3, 8, 15, 22):
                B_, C_, H_, W_ = x.shape
                f = x.view(B_, C_, H_ * W_)
                grams.append(torch.bmm(f, f.transpose(1, 2)) / (C_ * H_ * W_))
        return grams

    # 1. §8.4 参考图更换敏感度
    tasks, z_src_750, _ = tasks_and_sources()
    sel = [i for i, t in enumerate(tasks) if t["source_style_id"] != t["target_style_id"]][:10]
    sel_t = torch.tensor(sel, dtype=torch.long)
    fib_src_b = bundle["fnd"]["test750_fib"][sel_t].to(args.device).float()
    u_src_b = fiber_enc.encode(fib_src_b)
    y16_b = coarse_y16_lcn(((z_src_750[sel_t].to(args.device).float() * w_l).sum(1, keepdim=True) + b_l).clamp(0.0, 1.0))
    d16_b = bundle["test_struct"]["dino_16"][sel_t].to(args.device).float()
    d16_sg5 = _sg5_ref_d16(bundle, c21, device=args.device).to(args.device)
    v16_sg5 = _sg5_cnx_v16(c21, cnx, device=args.device).to(args.device)

    clean_dir = ROOT / "datasets/distinct5_512/train_512clean"
    sg_fibs = []
    for st in c21["sg5"]["stems"]:
        p = clean_dir / st.split("__")[0] / f"{st}.jpg"
        if not p.exists():
            p = list(clean_dir.rglob(f"{st}.*"))[0]
        xr = TF.to_tensor(Image.open(p).convert("RGB").resize((512, 512), Image.Resampling.BILINEAR)).unsqueeze(0).to(args.device)
        sg_fibs.append(torch.cat([extract_color_fiber_27d(xr), extract_dino_rgb_moments_768d(dino_model, xr, batch=1, device=args.device).to(args.device)], dim=-1))
    fib_sg5 = torch.cat(sg_fibs, 0)
    fib_n_sg5 = (fib_sg5 - fib_mean) / fib_std
    u_sg5 = fiber_enc.encode(fib_sg5.float())

    eps_b = torch.randn(len(sel), 4, 64, 64, generator=torch.Generator().manual_seed(20262100)).to(args.device)
    feats_r = []
    B = len(sel)
    f0, f1 = torch.zeros(B, dtype=torch.bool, device=args.device), torch.ones(B, dtype=torch.bool, device=args.device)
    l1, sid0 = torch.ones(B, device=args.device), torch.zeros(B, dtype=torch.long, device=args.device)
    tau0_s, wj_s = 0.45, 2.2
    dt_s = (1.0 - tau0_s) / 10.0
    for r_i in range(5):
        z = (1.0 - tau0_s) * eps_b + tau0_s * z_src_750[sel_t].to(args.device).float()
        u_style = ad(u_sg5[r_i:r_i+1].expand(B, -1), u_src_b)
        u_glob = fib_n_sg5[r_i:r_i+1].expand(B, -1)
        d16_r = d16_sg5[r_i:r_i+1].expand(B, -1, -1, -1)
        v16_r = v16_sg5[r_i:r_i+1].expand(B, -1, -1, -1)
        for i in range(10):
            def _v(zz, pr):
                tv = torch.full((B,), tau0_s + pr * (1.0 - tau0_s), device=args.device)
                with autocast(args.device):
                    v_sc, v_0c = fm(torch.cat([zz, zz], 0), y16_b.repeat(2, 1, 1, 1), d16_b.repeat(2, 1, 1, 1),
                                    sid0.repeat(2), u_style.repeat(2, 1), tv.repeat(2),
                                    drop_style=torch.cat([f0, f1], 0), drop_content=torch.cat([f0, f0], 0),
                                    layout_scale=torch.cat([l1, l1], 0),
                                    d16_ref=d16_r.repeat(2, 1, 1, 1), v16_ref=v16_r.repeat(2, 1, 1, 1),
                                    u_glob=u_glob.repeat(2, 1)).float().chunk(2, 0)
                return v_0c + wj_s * (v_sc - v_0c)
            v1 = _v(z, i / 10.0)
            z = z + dt_s * _v(z + 0.5 * dt_s * v1, (i + 0.5) / 10.0)
        rgb = decode_s10(dec, z, batch=10).clamp(0.0, 1.0)
        pils = [Image.fromarray((x.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype("uint8")) for x in rgb]
        feats_r.append(stack.extract(pils, batch=10)["dino"].mean(0).cpu())
    dists = [float(F.pairwise_distance(feats_r[i].unsqueeze(0), feats_r[j].unsqueeze(0))) for i in range(5) for j in range(i + 1, 5)]
    swap_dist = round(sum(dists) / len(dists), 4)
    print(f"[V24 O3 §8.4 SwapSensitivity] mean_pairwise_dist={swap_dist}", flush=True)

    # 2. StyleGallery 25 类 125-pair 端到端全盲对比 (报告 delta_S = S(out,ref) - S(src,ref))
    pairs = build_sg_25_manifest(sg_root, n_content=5)
    sg_runs = []
    for tau0, wj, tag in ((0.40, 2.2, "O3_sg25_t40"), (0.50, 2.0, "O3_sg25_t50")):
        out_dir = OUT / "sg25_outputs" / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        dc_l, ds_l, dsrc_l, cs_l, lp_l, gr_l, ts = [], [], [], [], [], [], []
        seed_all(20262200)
        torch.cuda.reset_peak_memory_stats(args.device)
        dt = (1.0 - tau0) / 10.0
        for p in pairs:
            im_c = Image.open(p["content_path"]).convert("RGB").resize((512, 512), Image.Resampling.BILINEAR)
            im_r = Image.open(p["style_path"]).convert("RGB").resize((512, 512), Image.Resampling.BILINEAR)
            xc, xr = TF.to_tensor(im_c).unsqueeze(0).to(args.device), TF.to_tensor(im_r).unsqueeze(0).to(args.device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            zc = encode_rgb_scaled(vae, xc)
            y16 = coarse_y16_lcn(((zc * w_l).sum(1, keepdim=True) + b_l).clamp(0.0, 1.0))
            d16 = extract_whitened_dino_layout(dino_model, xc, batch=1, device=args.device)
            d16_r = extract_whitened_dino_layout(dino_model, xr, batch=1, device=args.device)
            v16_r = _cnx_v16(xr)
            fib_c = torch.cat([extract_color_fiber_27d(xc), extract_dino_rgb_moments_768d(dino_model, xc, batch=1, device=args.device).to(args.device)], dim=-1)
            fib_r = torch.cat([extract_color_fiber_27d(xr), extract_dino_rgb_moments_768d(dino_model, xr, batch=1, device=args.device).to(args.device)], dim=-1)
            u_style = ad(fiber_enc.encode(fib_r.float()), fiber_enc.encode(fib_c.float()))
            u_glob = (fib_r - fib_mean) / fib_std

            z = (1.0 - tau0) * torch.randn_like(zc) + tau0 * zc
            f0_1, f1_1 = torch.zeros(1, dtype=torch.bool, device=args.device), torch.ones(1, dtype=torch.bool, device=args.device)
            l1_1, sid0_1 = torch.ones(1, device=args.device), torch.zeros(1, dtype=torch.long, device=args.device)
            for i in range(10):
                def _v1(zz, pr):
                    t2 = torch.full((2,), tau0 + pr * (1.0 - tau0), device=args.device)
                    with autocast(args.device):
                        v_sc, v_0c = fm(torch.cat([zz, zz], 0), y16.repeat(2, 1, 1, 1), d16.repeat(2, 1, 1, 1),
                                        sid0_1.repeat(2), u_style.repeat(2, 1), t2,
                                        drop_style=torch.cat([f0_1, f1_1], 0), drop_content=torch.cat([f0_1, f0_1], 0),
                                        layout_scale=torch.cat([l1_1, l1_1], 0),
                                        d16_ref=d16_r.repeat(2, 1, 1, 1), v16_ref=v16_r.repeat(2, 1, 1, 1),
                                        u_glob=u_glob.repeat(2, 1)).float().chunk(2, 0)
                    return v_0c + wj * (v_sc - v_0c)
                v1 = _v1(z, i / 10.0)
                z = z + dt * _v1(z + 0.5 * dt * v1, (i + 0.5) / 10.0)
            rgb_out = decode_s10(dec, z, batch=1, output_cpu=False).clamp(0.0, 1.0)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000.0)

            pil_out = Image.fromarray((rgb_out[0].permute(1, 2, 0).cpu().numpy() * 255).round().clip(0, 255).astype("uint8"))
            pil_out.save(out_dir / f"{p['pair_id']}.png")
            fo = stack.extract([im_c, im_r, pil_out], batch=3)
            dc_l.append(float((fo["dino"][2] * fo["dino"][0]).sum()))
            ds_l.append(float((fo["dino"][2] * fo["dino"][1]).sum()))
            dsrc_l.append(float((fo["dino"][0] * fo["dino"][1]).sum()))
            cs_l.append(float((fo["clip"][2] * fo["clip"][1]).sum()))
            lp_l.append(float(stack.lpips_vs((rgb_out.cpu() * 255).round().clamp(0, 255).byte(), xc.cpu(), batch=1).item()))
            gr_l.append(float(sum(F.mse_loss(a, b) for a, b in zip(_vgg_gram(rgb_out), _vgg_gram(xr))).item()))

        s_out = float(np.mean(ds_l))
        s_src = float(np.mean(dsrc_l))
        row = {
            "arm": tag, "n_pairs": len(pairs),
            "dino_c": round(float(np.mean(dc_l)), 4),
            "dino_s": round(s_out, 4),
            "delta_dino_s": round(s_out - s_src, 4),
            "clip_s": round(float(np.mean(cs_l)), 4),
            "style_affinity": round(0.5 * (s_out + float(np.mean(cs_l))), 4),
            "lpips": round(float(np.mean(lp_l)), 4),
            "vgg_gram": round(float(np.mean(gr_l)), 5),
            "e2e_ms": round(float(np.mean(ts[2:])), 1),
            "peak_vram_mb": round(torch.cuda.max_memory_allocated(args.device) / (1024 ** 2), 1),
        }
        sg_runs.append(row)
        print(f"[V24 O3 SG-25] {row}", flush=True)

    save_json(OUT / "o3_sg25_eval.json", {"updated_utc": utc(), "swap_dist": swap_dist, "sg25": sg_runs})


if __name__ == "__main__":
    main()
