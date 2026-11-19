"""V25 Round 3 (3060): One-Shot 收敛最小集 —— τ 单滑块补扫 + O3_label_free 盲测补全 + O3_self_only 单变量训练。

落实 docs/19_pull_together/01_convergence_plan.md §4:
1. 零训练 τ 补扫: O3_rgb_cnx_gated 同权重补 tau0=0.40/0.60 (与已测 t45/t55 构成单一 strength 曲线);
2. 零训练盲测补全: O3_label_free 补 §8.4 swap_dist + SG-25 OOD 125-pair (t40/t50, 与 O3 已有行同口径);
3. 单变量训练 O3_self_only: 同 O3_rgb_cnx_gated 结构/步数/初始化, 仅把 50% 跨风格对换成 100% 自配对重建,
   t45/t55 评测 → 定量证明跨风格监督是否不可或缺 (标签归因实验后置到机制 C1, 见 01 文档 §4.1)。

复用 exp.v24 全部 One-Shot 基础设施 (SLFUNetV24OneShot / train_o_arm / A0 encoder), 仅将产出重定向 runs/v25。

用法 (3060):
  python -m exp.v25.r3_oneshot_minimal --pool ../pseudo_gt_24h/orig_mean_fp16
"""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF

from . import OUT, ROOT, autocast, sanitize_cuda_env, save_json, seed_all, utc
from exp.v24 import RawWriterV24, config_hash, file_sha, read_json, run_measure_v24, validate_raw_meta
import exp.v24 as _v24
import exp.v24.p0_o1_o3_value as _o_mod
from exp.v24.p0_o1_o3_value import O_SPECS, SLFUNetV24OneShot, _sg5_cnx_v16, _sg5_ref_d16, build_or_load_train_cnx_v16, train_o_arm
from exp.v18 import tasks_and_sources
from exp.v18.structure import extract_whitened_dino_layout
from exp.v19.fiber_duality import StyleFiberEncoder, build_or_load_v19_foundation, extract_color_fiber_27d, extract_dino_rgb_moments_768d
from exp.v20.slf_flow import coarse_y16_lcn
from exp.v20.submode_patchswd import load_convnext_tiny_offline
from exp.v22.ref_adapter import RefAdapter
from exp.v23.p2_sg_bench import _find_sg_root, build_sg_25_manifest, encode_rgb_scaled, load_a0_or_sd0_encoder
from src.utils import FeatureStack, decode_s10, load_s10_decoder
from src.utils.protocol import load_vgg16_features_offline

V25_OUT = OUT  # exp.v25.__init__ 的 OUT 即 runs/v25
O3_CKPT = ROOT / "runs/v24/oneshot_val/O3_rgb_cnx_gated.pt"
LF_CKPT = ROOT / "runs/v24/oneshot_val/O3_label_free.pt"


def reused_v24_summary(arm_name: str) -> dict:
    s = read_json(ROOT / f"runs/v24/measured/{arm_name}/summary_v24.json")
    c = s["cross600"]
    return {"arm": arm_name, "reused": "runs/v24", "dino_c": round(c["dino_c"], 5), "dino_s": round(c["dino_s"], 5),
            "lpips": round(c["lpips"], 5), "style_affinity": round(c["style_affinity"], 5), "csd_top1": round(c["csd_top1"], 5)}


@torch.no_grad()
def eval_o_tau_list(bundle: dict, c21: dict, cnx, ckpt_p: Path, arm: str,
                    tau_list: list[tuple[float, float, str]], device: str = "cuda") -> list[dict]:
    """exp.v24.p0_o1_o3_value.eval_o_arm 的 τ 参数化版本 (产出重定向 runs/v25)。"""
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
    for tau0, wj, tag in tau_list:
        name = f"{arm}_{tag}"
        cfg_hash = config_hash({"arm": name, "tau0": tau0, "wj": wj, "ckpt": file_sha(ckpt_p)[:12]})
        if (V25_OUT / "raw" / name / "meta.json").exists() and validate_raw_meta(read_json(V25_OUT / "raw" / name / "meta.json")):
            s = read_json(V25_OUT / f"measured/{name}/summary_v24.json")
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
                        with autocast(device):
                            v_sc, v_0c = fm(torch.cat([zz, zz], 0), y16.repeat(2, 1, 1, 1), d16.repeat(2, 1, 1, 1),
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
        c = s["cross600"]
        res.append({"arm": name, "tau0": tau0, "wj": wj, "dino_c": round(c["dino_c"], 5), "dino_s": round(c["dino_s"], 5),
                    "lpips": round(c["lpips"], 5), "style_affinity": round(c["style_affinity"], 5), "csd_top1": round(c["csd_top1"], 5)})
    del fm, ad
    torch.cuda.empty_cache()
    return res


@torch.no_grad()
def sg25_and_swap(bundle: dict, c21: dict, cnx, ckpt_p: Path, tau_list: list[tuple[float, float, str]],
                  device: str = "cuda") -> tuple[float, list[dict]]:
    """exp.v24.o3_sg25_eval.main 的参数化版本: 任意 O3 系权重 × 任意 τ 列表 (swap_dist + SG-25 125-pair)。"""
    sg_root = _find_sg_root()
    vae = load_a0_or_sd0_encoder(device)
    dec = load_s10_decoder(device)
    stack = FeatureStack(device)
    dino_model = stack.dino
    cnx_s2 = cnx[:4]
    vgg = load_vgg16_features_offline(device).eval()

    ck = torch.load(ckpt_p, map_location="cpu", weights_only=False)
    fm = SLFUNetV24OneShot(v_dim=192, use_conf_gate=True, use_global_fib=True).to(device).eval()
    fm.load_state_dict(ck["fm"], strict=True)
    ad = RefAdapter(dim=128).to(device).eval()
    ad.load_state_dict(ck["adapter"], strict=True)
    fib_mean, fib_std = ck["fib_mean"].to(device), ck["fib_std"].to(device)

    fiber_enc = StyleFiberEncoder(in_dim=bundle["fnd"]["tr_fib"].shape[1], dim=128).to(device).eval()
    fiber_enc.load_state_dict(bundle["fnd"]["fiber_state"])
    w_l = bundle["v18_cache"]["luma_proj"]["w"].float().view(1, 4, 1, 1).to(device)
    b_l = bundle["v18_cache"]["luma_proj"]["b"].float().to(device)

    mean_im = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_im = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def _cnx_v16(rgb01):
        x = (F.interpolate(rgb01.float(), size=(256, 256), mode="bilinear", align_corners=False) - mean_im) / std_im
        v = F.avg_pool2d(cnx_s2(x), 2, 2)
        return v / v.std(dim=(2, 3), keepdim=True).clamp_min(1e-4)

    def _vgg_gram(rgb01):
        x = (rgb01.to(device).float() - mean_im) / std_im
        grams = []
        for idx, m in enumerate(vgg):
            x = m(x)
            if idx in (3, 8, 15, 22):
                B_, C_, H_, W_ = x.shape
                f = x.view(B_, C_, H_ * W_)
                grams.append(torch.bmm(f, f.transpose(1, 2)) / (C_ * H_ * W_))
        return grams

    # 1. §8.4 参考图更换敏感度 (SG-5 五参考 × 10 内容, τ=0.45 固定口径与 O3 既有 swap 0.442 可比)
    tasks, z_src_750, _ = tasks_and_sources()
    sel = [i for i, t in enumerate(tasks) if t["source_style_id"] != t["target_style_id"]][:10]
    sel_t = torch.tensor(sel, dtype=torch.long)
    fib_src_b = bundle["fnd"]["test750_fib"][sel_t].to(device).float()
    u_src_b = fiber_enc.encode(fib_src_b)
    y16_b = coarse_y16_lcn(((z_src_750[sel_t].to(device).float() * w_l).sum(1, keepdim=True) + b_l).clamp(0.0, 1.0))
    d16_b = bundle["test_struct"]["dino_16"][sel_t].to(device).float()
    d16_sg5 = _sg5_ref_d16(bundle, c21, device=device).to(device)
    v16_sg5 = _sg5_cnx_v16(c21, cnx, device=device).to(device)

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
    u_sg5 = fiber_enc.encode(fib_sg5.float())

    eps_b = torch.randn(len(sel), 4, 64, 64, generator=torch.Generator().manual_seed(20262100)).to(device)
    feats_r = []
    B = len(sel)
    f0, f1 = torch.zeros(B, dtype=torch.bool, device=device), torch.ones(B, dtype=torch.bool, device=device)
    l1, sid0 = torch.ones(B, device=device), torch.zeros(B, dtype=torch.long, device=device)
    tau0_s, wj_s = 0.45, 2.2
    dt_s = (1.0 - tau0_s) / 10.0
    for r_i in range(5):
        z = (1.0 - tau0_s) * eps_b + tau0_s * z_src_750[sel_t].to(device).float()
        u_style = ad(u_sg5[r_i:r_i + 1].expand(B, -1), u_src_b)
        u_glob = fib_n_sg5[r_i:r_i + 1].expand(B, -1)
        d16_r = d16_sg5[r_i:r_i + 1].expand(B, -1, -1, -1)
        v16_r = v16_sg5[r_i:r_i + 1].expand(B, -1, -1, -1)
        for i in range(10):
            def _v(zz, pr):
                tv = torch.full((B,), tau0_s + pr * (1.0 - tau0_s), device=device)
                with autocast(device):
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

    # 2. SG-25 OOD 125-pair 全盲 (delta_S = S(out,ref) - S(src,ref))
    pairs = build_sg_25_manifest(sg_root, n_content=5)
    sg_runs = []
    for tau0, wj, tag in tau_list:
        out_dir = V25_OUT / "sg25_outputs" / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        dc_l, ds_l, dsrc_l, cs_l, lp_l, gr_l, ts = [], [], [], [], [], [], []
        seed_all(20262200)
        torch.cuda.reset_peak_memory_stats(device)
        dt = (1.0 - tau0) / 10.0
        for p in pairs:
            im_c = Image.open(p["content_path"]).convert("RGB").resize((512, 512), Image.Resampling.BILINEAR)
            im_r = Image.open(p["style_path"]).convert("RGB").resize((512, 512), Image.Resampling.BILINEAR)
            xc, xr = TF.to_tensor(im_c).unsqueeze(0).to(device), TF.to_tensor(im_r).unsqueeze(0).to(device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            zc = encode_rgb_scaled(vae, xc)
            y16 = coarse_y16_lcn(((zc * w_l).sum(1, keepdim=True) + b_l).clamp(0.0, 1.0))
            d16 = extract_whitened_dino_layout(dino_model, xc, batch=1, device=device)
            d16_r = extract_whitened_dino_layout(dino_model, xr, batch=1, device=device)
            v16_r = _cnx_v16(xr)
            fib_c = torch.cat([extract_color_fiber_27d(xc), extract_dino_rgb_moments_768d(dino_model, xc, batch=1, device=device).to(device)], dim=-1)
            fib_r = torch.cat([extract_color_fiber_27d(xr), extract_dino_rgb_moments_768d(dino_model, xr, batch=1, device=device).to(device)], dim=-1)
            u_style = ad(fiber_enc.encode(fib_r.float()), fiber_enc.encode(fib_c.float()))
            u_glob = (fib_r - fib_mean) / fib_std

            z = (1.0 - tau0) * torch.randn_like(zc) + tau0 * zc
            f0_1, f1_1 = torch.zeros(1, dtype=torch.bool, device=device), torch.ones(1, dtype=torch.bool, device=device)
            l1_1, sid0_1 = torch.ones(1, device=device), torch.zeros(1, dtype=torch.long, device=device)
            for i in range(10):
                def _v1(zz, pr):
                    t2 = torch.full((2,), tau0 + pr * (1.0 - tau0), device=device)
                    with autocast(device):
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

        s_out, s_src = float(np.mean(ds_l)), float(np.mean(dsrc_l))
        row = {"arm": tag, "n_pairs": len(pairs), "dino_c": round(float(np.mean(dc_l)), 4), "dino_s": round(s_out, 4),
               "delta_dino_s": round(s_out - s_src, 4), "clip_s": round(float(np.mean(cs_l)), 4),
               "style_affinity": round(0.5 * (s_out + float(np.mean(cs_l))), 4), "lpips": round(float(np.mean(lp_l)), 4),
               "vgg_gram": round(float(np.mean(gr_l)), 5), "e2e_ms": round(float(np.mean(ts[2:])), 1),
               "peak_vram_mb": round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 1)}
        sg_runs.append(row)
        print(f"[V25 R3 SG-25] {row}", flush=True)
    del fm, ad, stack, vae, fiber_enc
    torch.cuda.empty_cache()
    return swap_dist, sg_runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--updates", type=int, default=1200)
    args = ap.parse_args()

    sanitize_cuda_env()
    bundle = build_or_load_v19_foundation(args.pool, device=args.device)
    c21 = torch.load(ROOT / "runs/v21/cache/v21_stiefel_ref_cache.pt", map_location="cpu", weights_only=False)
    dec = load_s10_decoder(args.device)
    cnx = load_convnext_tiny_offline(args.device).eval()
    for p in cnx.parameters():
        p.requires_grad_(False)
    v16_tr = build_or_load_train_cnx_v16(bundle, dec, cnx, device=args.device)  # 复用 runs/v24 缓存

    # 之后所有 exp.v24 写路径重定向 runs/v25
    V25_OUT.mkdir(parents=True, exist_ok=True)
    _v24.OUT = V25_OUT
    _o_mod.OUT = V25_OUT

    out_json = V25_OUT / "r3_oneshot_minimal.json"
    pack: dict = {"updated_utc": utc(), "reused": {}, "tau_curve": [], "label_free_blind": {}, "self_only": []}

    # 复用 V24 已测 (t45/t55 两档四臂)
    for arm in ("O3_rgb_cnx_gated_t45", "O3_rgb_cnx_gated_t55", "O3_label_free_t45", "O3_label_free_t55"):
        pack["reused"][arm] = reused_v24_summary(arm)
    save_json(out_json, pack)

    # ---- 任务 1: O3 单 strength τ 曲线补扫 (零训练) ----
    pack["tau_curve"] = eval_o_tau_list(bundle, c21, cnx, O3_CKPT, "O3_rgb_cnx_gated",
                                        [(0.40, 2.3, "t40"), (0.60, 1.9, "t60")], device=args.device)
    pack["updated_utc"] = utc()
    save_json(out_json, pack)
    for r in pack["tau_curve"]:
        print(f"[V25 R3 tau-curve] {r['arm']}: D-C={r['dino_c']:.4f} D-S={r['dino_s']:.4f} Top1={r['csd_top1']*100:.2f}%", flush=True)

    # ---- 任务 3 (单变量训练): O3_self_only = 100% 自配对重建, 其余同 O3 ----
    _o_mod.O_SPECS["O3_self_only"] = copy.deepcopy(_o_mod.O_SPECS["O3_rgb_cnx_gated"])
    tr_z = bundle["v18_cache"]["train"]["z"]
    n_self = 1200
    self_pack = {"idx_c": torch.arange(n_self), "idx_r": torch.arange(n_self), "z_cr": tr_z[:n_self].float()}
    ckpt_p = train_o_arm(bundle, c21, self_pack, v16_tr, "O3_self_only", updates=args.updates, device=args.device)
    pack["self_only"] = eval_o_tau_list(bundle, c21, cnx, ckpt_p, "O3_self_only",
                                        [(0.45, 2.2, "t45"), (0.55, 2.0, "t55")], device=args.device)
    pack["updated_utc"] = utc()
    save_json(out_json, pack)
    for r in pack["self_only"]:
        print(f"[V25 R3 self-only] {r['arm']}: D-C={r['dino_c']:.4f} D-S={r['dino_s']:.4f} Top1={r['csd_top1']*100:.2f}%", flush=True)

    # ---- 任务 2: O3_label_free 盲测补全 (swap_dist + SG-25 t40/t50, 与 O3 同口径) ----
    if LF_CKPT.exists():
        swap_lf, sg_lf = sg25_and_swap(bundle, c21, cnx, LF_CKPT,
                                       [(0.40, 2.2, "O3LF_sg25_t40"), (0.50, 2.0, "O3LF_sg25_t50")], device=args.device)
        pack["label_free_blind"] = {"swap_dist": swap_lf, "sg25": sg_lf}
    else:
        pack["label_free_blind"] = {"error": f"missing {LF_CKPT}"}
    pack["updated_utc"] = utc()
    save_json(out_json, pack)

    print("\n=== V25 Round 3 (3060 One-Shot minimal) 汇总 ===", flush=True)
    print(f"tau_curve: {[r['arm'] for r in pack['tau_curve']]}", flush=True)
    print(f"self_only: {[r['arm'] for r in pack['self_only']]}", flush=True)
    print(f"label_free_blind: swap={pack['label_free_blind'].get('swap_dist')} sg25={[r['arm'] for r in pack['label_free_blind'].get('sg25', [])]}", flush=True)


if __name__ == "__main__":
    main()
