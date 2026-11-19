"""V23 P2 §2.4 & §7.1 & §8.4 (3060): 严格独立 StyleGallery 25类 One-Shot 评测 + RGB→RGB 端到端测速 + 参考敏感度诊断。

包含四项功能:
1. Encoder 双向验证: 验证 A0/SD0 Encoder + S10 Decoder 往返 DINO-C 与像素误差 (确保 RGB→latent 域兼容);
2. §8.4 防止忽略参考诊断: 固定 10 张内容图, 轮换 5 张 SG-5 参考图, 测输出特征 pairwise_dist (排除死输出/参考泄漏);
3. 独立 StyleGallery 25 类盲测 (125 配对: 5 内容图 × 25 新风格家族):
   - 无任何目标类别 ID, onehot=0, 每张参考图实时编码 DINO+Oklab fiber & d16_ref;
   - 报告: 输出-参考 DINO-S, 输出-内容 DINO-C, Style Affinity, LPIPS, VGG Gram Loss,
     以及实测完整 RGB→RGB 端到端延迟 (Encoder + 参考特征 + Flow + S10) 与峰值显存。

用法 (3060):
  python -m exp.v23.p2_sg_bench --ckpt runs/v23/oneshot_attn/M1_cross25.pt --arm M1_cross25 --spec M1_cross25
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF

from . import OUT, ROOT, autocast, sanitize_cuda_env, save_json, seed_all, utc
from .p2_attn_cross import ARM_SPECS, SLFUNetRefAttn
from exp.v18.structure import extract_whitened_dino_layout
from exp.v19.fiber_duality import StyleFiberEncoder, build_or_load_v19_foundation
from exp.v20.slf_flow import coarse_y16_lcn
from exp.v22.ref_adapter import RefAdapter
from src.utils import FeatureStack, decode_s10, load_s10_decoder
from src.utils.protocol import load_vgg16_features_offline
from src.utils._local_eval_refs import load_dino_model

SG_ROOT_CANDS = [
    ROOT.parent / "StyleGallery",
    Path("I:/Latent_Style/StyleGallery"),
]
SD0_VAE_DIR = ROOT / "deliver/sd0_s10_profile/sd0_vae"
A0_ENC_CANDS = [
    ROOT / "runs/a0/a0_encoder_only.pt",
    ROOT / "deliver/checkpoints/runs/a0/a0_encoder_only.pt",
]


def _find_sg_root() -> Path:
    for p in SG_ROOT_CANDS:
        if p.exists() and (p / "content").exists() and (p / "style").exists():
            return p
    raise FileNotFoundError(f"StyleGallery root not found in {SG_ROOT_CANDS}")


def load_a0_or_sd0_encoder(device: str = "cuda"):
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(str(SD0_VAE_DIR), torch_dtype=torch.float32, local_files_only=True).to(device).eval()
    for p in A0_ENC_CANDS:
        if p.exists():
            ck = torch.load(p, map_location="cpu", weights_only=False)
            vae.load_state_dict(ck["state_dict"], strict=False)
            print(f"[SG-Bench] Loaded A0 encoder weights from {p}", flush=True)
            break
    vae = vae.half()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


@torch.no_grad()
def encode_rgb_scaled(vae, rgb01: torch.Tensor) -> torch.Tensor:
    """将 [0,1] RGB (B,3,512,512) 编码为仓库缩放规范的 (B,4,64,64) latent (* 0.18215)。"""
    x_m11 = (rgb01.to(vae.device, dtype=vae.dtype) * 2.0 - 1.0)
    z = vae.encode(x_m11).latent_dist.mean * 0.18215
    return z.float()


@torch.no_grad()
def verify_encoder_s10_roundtrip(vae, dec, stack: FeatureStack, sg_root: Path, device: str = "cuda") -> dict:
    imgs = sorted((sg_root / "content").glob("*.*"))[:6]
    xs = [TF.to_tensor(Image.open(p).convert("RGB").resize((512, 512), Image.Resampling.BILINEAR)) for p in imgs]
    rgb_in = torch.stack(xs, 0).to(device)
    t0 = time.perf_counter()
    z = encode_rgb_scaled(vae, rgb_in)
    torch.cuda.synchronize()
    enc_ms = (time.perf_counter() - t0) * 1000.0 / len(imgs)
    rgb_rec = decode_s10(dec, z, batch=8).clamp(0.0, 1.0)

    pils_in = [Image.open(p).convert("RGB").resize((512, 512), Image.Resampling.BILINEAR) for p in imgs]
    pils_rec = [Image.fromarray((r.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype("uint8")) for r in rgb_rec]
    f_in = stack.extract(pils_in, batch=6)
    f_rc = stack.extract(pils_rec, batch=6)
    dino_c = float((f_in["dino"] * f_rc["dino"]).sum(-1).mean())
    mae = float((rgb_in.cpu() - rgb_rec).abs().mean())
    res = {"n": len(imgs), "roundtrip_dino_c": round(dino_c, 4), "mae_01": round(mae, 4), "enc_ms_per_img": round(enc_ms, 2)}
    print(f"[SG-Bench Encoder Roundtrip] {res}", flush=True)
    return res


def build_sg_25_manifest(sg_root: Path, n_content: int = 5) -> list[dict]:
    """冻结 5 张内容图 × 25 个风格家族各第 1 张图 = 125 个 OOD 1-Shot 配对。"""
    out_p = OUT / "sg_25_manifest_125.json"
    if out_p.exists():
        return json.loads(out_p.read_text(encoding="utf-8"))["pairs"]
    contents = sorted([p for p in (sg_root / "content").iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")])[:n_content]
    styles = sorted([d for d in (sg_root / "style").iterdir() if d.is_dir()])
    pairs = []
    for c_p in contents:
        for s_dir in styles:
            refs = sorted([p for p in s_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
            if not refs:
                continue
            pairs.append({
                "pair_id": f"{c_p.stem}__{s_dir.name}",
                "content_path": str(c_p),
                "style_family": s_dir.name,
                "style_path": str(refs[0]),
            })
    save_json(out_p, {"n_pairs": len(pairs), "n_content": n_content, "n_styles": len(styles), "pairs": pairs})
    return pairs


@torch.no_grad()
def test_ref_swap_sensitivity_v23(fm: SLFUNetRefAttn, ad: RefAdapter, bundle: dict, c21: dict,
                                  dec, stack: FeatureStack, arm_name: str, device: str = "cuda") -> dict:
    """V23 §8.4: 固定 10 张内容图，更换 5 张 SG-5 参考图，测输出随参考图变化的距离。"""
    from exp.v18 import tasks_and_sources
    from .p2_attn_cross import _sg5_ref_d16

    tasks, z_src_750, _ = tasks_and_sources()
    sel = [i for i, t in enumerate(tasks) if t["source_style_id"] != t["target_style_id"]][:10]
    sel_t = torch.tensor(sel, dtype=torch.long)

    fiber_enc = StyleFiberEncoder(in_dim=bundle["fnd"]["tr_fib"].shape[1], dim=128).to(device).eval()
    fiber_enc.load_state_dict(bundle["fnd"]["fiber_state"])
    u_src_b = fiber_enc.encode(bundle["fnd"]["test750_fib"][sel_t].to(device).float())
    del fiber_enc

    w_l = bundle["v18_cache"]["luma_proj"]["w"].float().view(1, 4, 1, 1)
    b_l = bundle["v18_cache"]["luma_proj"]["b"].float()
    y16_b = coarse_y16_lcn(((z_src_750[sel_t].float() * w_l).sum(1, keepdim=True) + b_l).clamp(0.0, 1.0)).to(device)
    d16_b = bundle["test_struct"]["dino_16"][sel_t].to(device).float()
    eps_b = torch.randn(len(sel), 4, 64, 64, generator=torch.Generator().manual_seed(20262100)).to(device)
    u_sg5 = c21["sg5"]["u"].float().to(device)
    d16_sg5 = _sg5_ref_d16(bundle, c21, device=device).to(device)

    feats_per_ref = []
    B = len(sel)
    f0, f1 = torch.zeros(B, dtype=torch.bool, device=device), torch.ones(B, dtype=torch.bool, device=device)
    l1, sid0 = torch.ones(B, device=device), torch.zeros(B, dtype=torch.long, device=device)

    for r_i in range(5):
        u_style = ad(u_sg5[r_i:r_i+1].expand(B, -1), u_src_b)
        d16_r = d16_sg5[r_i:r_i+1].expand(B, -1, -1, -1)
        z = eps_b.clone()
        steps, dt = 10, 0.1
        for i in range(steps):
            def _v(zz, pr):
                zz2 = torch.cat([zz, zz], 0)
                t2 = torch.full((B * 2,), pr, device=device)
                with autocast(device):
                    v_sc, v_0c = fm(zz2, y16_b.repeat(2, 1, 1, 1), d16_b.repeat(2, 1, 1, 1),
                                    sid0.repeat(2), u_style.repeat(2, 1), t2,
                                    drop_style=torch.cat([f0, f1], 0), drop_content=torch.cat([f0, f0], 0),
                                    layout_scale=torch.cat([l1, l1], 0), d16_ref=d16_r.repeat(2, 1, 1, 1)).float().chunk(2, 0)
                return v_0c + 2.4 * (v_sc - v_0c)
            v1 = _v(z, i / steps)
            v2 = _v(z + 0.5 * dt * v1, (i + 0.5) / steps)
            z = z + dt * v2
        rgb = decode_s10(dec, z, batch=10).clamp(0.0, 1.0)
        pils = [Image.fromarray((x.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype("uint8")) for x in rgb]
        fo = stack.extract(pils, batch=10)
        feats_per_ref.append(fo["dino"].mean(0).cpu())

    dists = []
    for i in range(5):
        for j in range(i + 1, 5):
            dists.append(float(F.pairwise_distance(feats_per_ref[i].unsqueeze(0), feats_per_ref[j].unsqueeze(0))))
    res = {"arm": arm_name, "mean_pairwise_dist": round(sum(dists) / len(dists), 4), "dists": [round(d, 4) for d in dists]}
    print(f"[SG-Bench §8.4 SwapSensitivity] {arm_name}: mean_pairwise_dist={res['mean_pairwise_dist']}", flush=True)
    return res


@torch.no_grad()
def eval_sg_25_oneshot(fm: SLFUNetRefAttn, ad: RefAdapter, vae, dec, stack: FeatureStack,
                       bundle: dict, pairs: list[dict], arm_name: str, tau0: float = 0.35,
                       wj: float = 2.2, device: str = "cuda") -> dict:
    """在 125 个 StyleGallery 25 类 OOD 配对上跑完整端到端 RGB->RGB 1-Shot 评测。"""
    from exp.v19.fiber_duality import extract_color_fiber_27d, extract_dino_rgb_moments_768d

    dino_model = stack.dino
    fiber_enc = StyleFiberEncoder(in_dim=bundle["fnd"]["tr_fib"].shape[1], dim=128).to(device).eval()
    fiber_enc.load_state_dict(bundle["fnd"]["fiber_state"])
    w_l = bundle["v18_cache"]["luma_proj"]["w"].float().view(1, 4, 1, 1).to(device)
    b_l = bundle["v18_cache"]["luma_proj"]["b"].float().to(device)

    vgg = load_vgg16_features_offline(device).eval()
    vgg_layers = (3, 8, 15, 22)

    def _vgg_gram(rgb01: torch.Tensor) -> list[torch.Tensor]:
        mean = rgb01.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = rgb01.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x = (rgb01.to(device).float() - mean) / std
        grams = []
        for idx, m in enumerate(vgg):
            x = m(x)
            if idx in vgg_layers:
                B_, C_, H_, W_ = x.shape
                f = x.view(B_, C_, H_ * W_)
                g = torch.bmm(f, f.transpose(1, 2)) / (C_ * H_ * W_)
                grams.append(g)
        return grams

    out_dir = OUT / "sg25_outputs" / arm_name
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats(device)

    all_dino_c, all_dino_s, all_clip_s, all_lpips, all_gram = [], [], [], [], []
    t_e2e_list = []
    seed_all(20262200)

    for p in pairs:
        im_c = Image.open(p["content_path"]).convert("RGB").resize((512, 512), Image.Resampling.BILINEAR)
        im_r = Image.open(p["style_path"]).convert("RGB").resize((512, 512), Image.Resampling.BILINEAR)
        xc = TF.to_tensor(im_c).unsqueeze(0).to(device)
        xr = TF.to_tensor(im_r).unsqueeze(0).to(device)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        zc = encode_rgb_scaled(vae, xc)
        y64 = ((zc * w_l).sum(1, keepdim=True) + b_l).clamp(0.0, 1.0)
        y16 = coarse_y16_lcn(y64)
        d16 = extract_whitened_dino_layout(dino_model, xc, batch=1, device=device)
        d16_r = extract_whitened_dino_layout(dino_model, xr, batch=1, device=device)

        fib_c = torch.cat([extract_color_fiber_27d(xc), extract_dino_rgb_moments_768d(dino_model, xc, batch=1, device=device).to(device)], dim=-1)
        fib_r = torch.cat([extract_color_fiber_27d(xr), extract_dino_rgb_moments_768d(dino_model, xr, batch=1, device=device).to(device)], dim=-1)
        u_style = ad(fiber_enc.encode(fib_r.float()), fiber_enc.encode(fib_c.float()))

        eps = torch.randn_like(zc)
        z = (1.0 - tau0) * eps + tau0 * zc if tau0 > 0 else eps
        f0 = torch.zeros(1, dtype=torch.bool, device=device)
        f1 = torch.ones(1, dtype=torch.bool, device=device)
        l1 = torch.ones(1, device=device)
        sid0 = torch.zeros(1, dtype=torch.long, device=device)
        steps = 10
        dt = (1.0 - tau0) / float(steps)
        for i in range(steps):
            def _v(zz, pr):
                zz2 = torch.cat([zz, zz], 0)
                t2 = torch.full((2,), tau0 + pr * (1.0 - tau0), device=device)
                with autocast(device):
                    v_sc, v_0c = fm(zz2, y16.repeat(2, 1, 1, 1), d16.repeat(2, 1, 1, 1),
                                    sid0.repeat(2), u_style.repeat(2, 1), t2,
                                    drop_style=torch.cat([f0, f1], 0), drop_content=torch.cat([f0, f0], 0),
                                    layout_scale=torch.cat([l1, l1], 0), d16_ref=d16_r.repeat(2, 1, 1, 1)).float().chunk(2, 0)
                return v_0c + wj * (v_sc - v_0c)
            v1 = _v(z, i / steps)
            v2 = _v(z + 0.5 * dt * v1, (i + 0.5) / steps)
            z = z + dt * v2
        rgb_out = decode_s10(dec, z, batch=1, output_cpu=False).clamp(0.0, 1.0)
        torch.cuda.synchronize()
        t_e2e_list.append((time.perf_counter() - t0) * 1000.0)

        pil_out = Image.fromarray((rgb_out[0].permute(1, 2, 0).cpu().numpy() * 255).round().clip(0, 255).astype("uint8"))
        pil_out.save(out_dir / f"{p['pair_id']}.png")

        fo = stack.extract([im_c, im_r, pil_out], batch=3)
        all_dino_c.append(float((fo["dino"][2] * fo["dino"][0]).sum()))
        all_dino_s.append(float((fo["dino"][2] * fo["dino"][1]).sum()))
        all_clip_s.append(float((fo["clip"][2] * fo["clip"][1]).sum()))
        u8_out = (rgb_out.cpu() * 255.0).round().clamp(0, 255).byte()
        all_lpips.append(float(stack.lpips_vs(u8_out, xc.cpu(), batch=1).item()))

        go = _vgg_gram(rgb_out)
        gr = _vgg_gram(xr)
        all_gram.append(float(sum(F.mse_loss(a, b) for a, b in zip(go, gr)).item()))

    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    aff = 0.5 * (float(np.mean(all_dino_s)) + float(np.mean(all_clip_s)))
    summary = {
        "arm": arm_name, "n_pairs": len(pairs),
        "dino_s": round(float(np.mean(all_dino_s)), 4),
        "clip_s": round(float(np.mean(all_clip_s)), 4),
        "style_affinity": round(aff, 4),
        "dino_c": round(float(np.mean(all_dino_c)), 4),
        "lpips": round(float(np.mean(all_lpips)), 4),
        "vgg_gram_loss": round(float(np.mean(all_gram)), 5),
        "e2e_rgb2rgb_ms_per_img": round(float(np.mean(t_e2e_list[2:])), 1),
        "peak_vram_mb": round(peak_mb, 1),
    }
    print(f"[SG-Bench 25-Family 125-Pair] {summary}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--spec", default="M1_cross25")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tau0", type=float, default=0.35)
    ap.add_argument("--wj", type=float, default=2.2)
    ap.add_argument("--run-125", action="store_true", help="跑完整 125 对 25 类 StyleGallery 端到端评测")
    args = ap.parse_args()

    sanitize_cuda_env()
    sg_root = _find_sg_root()
    bundle = build_or_load_v19_foundation(args.pool, device=args.device)
    c21 = torch.load(ROOT / "runs/v21/cache/v21_stiefel_ref_cache.pt", map_location="cpu", weights_only=False)
    vae = load_a0_or_sd0_encoder(args.device)
    dec = load_s10_decoder(args.device)
    stack = FeatureStack(args.device)

    rt = verify_encoder_s10_roundtrip(vae, dec, stack, sg_root, device=args.device)

    spec = ARM_SPECS[args.spec]
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    fm = SLFUNetRefAttn(base_dim=144, u_dim=128, use_attn=spec["use_attn"], use_conf_gate=spec["use_conf"]).to(args.device).eval()
    fm.load_state_dict(ck["fm"], strict=True)
    ad = RefAdapter(dim=128).to(args.device).eval()
    ad.load_state_dict(ck["adapter"], strict=True)

    swap = test_ref_swap_sensitivity_v23(fm, ad, bundle, c21, dec, stack, args.arm, device=args.device)
    pairs = build_sg_25_manifest(sg_root, n_content=5)
    sg_eval = eval_sg_25_oneshot(fm, ad, vae, dec, stack, bundle, pairs, args.arm, tau0=args.tau0, wj=args.wj, device=args.device) if args.run_125 else None
    save_json(OUT / f"sg_bench_{args.arm}.json", {"updated_utc": utc(), "roundtrip": rt, "swap_sensitivity": swap, "sg25": sg_eval})


if __name__ == "__main__":
    main()
