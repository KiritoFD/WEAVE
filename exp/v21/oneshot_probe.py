"""Fast 30s diagnostic probe of 1-Shot Arbitrary-Exemplar Pyramid-SWD + Grayscale Semantic Lock."""
from __future__ import annotations

import copy
import time
import torch
import torch.nn.functional as F
from PIL import Image

from . import ROOT, sanitize_cuda_env
from exp.v18 import tasks_and_sources
from exp.v19.fiber_duality import build_or_load_v19_foundation
from exp.v20.submode_patchswd import load_convnext_tiny_offline, sliced_wasserstein_1d
from src.utils import FeatureStack, decode_s10, load_s10_decoder, reference_bank
from src.utils.protocol import load_vgg16_features_offline
from src.utils.structure import rgb_to_gray, rgb_to_oklab


def main():
    sanitize_cuda_env()
    dev = "cuda"
    bundle = build_or_load_v19_foundation(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16", device=dev)
    ho = bundle["v18_cache"]["holdout"]  # 300 UNSEEN holdout paintings (0 train/test/ref overlap!)
    ho_sid = ho["sid"].long()
    ho_fib = bundle["fnd"]["ho_fib"].float()
    ho_dmu = F.normalize(ho_fib[:, 27:27 + 384], dim=-1)

    dec_fp16 = load_s10_decoder(dev)
    dec_bf16 = copy.deepcopy(dec_fp16).to(device=dev, dtype=torch.bfloat16).eval()
    for p in dec_bf16.parameters():
        p.requires_grad_(False)

    # Select 1 SINGLE unseen holdout painting per style (medoid within holdout_300, 100% blind to train & refs!)
    oneshot_z = []
    for s in range(5):
        ms = (ho_sid == s).nonzero(as_tuple=True)[0]
        dm = ho_dmu[ms]
        pick = ms[(dm @ dm.T).mean(1).argmax()]
        oneshot_z.append(ho["z"][pick].float())
    oneshot_z = torch.stack(oneshot_z).to(dev)
    with torch.no_grad():
        oneshot_rgb = decode_s10(dec_fp16, oneshot_z, batch=5, output_cpu=False).clamp(0.0, 1.0)

    cnx = load_convnext_tiny_offline(dev)
    vgg = load_vgg16_features_offline(dev)
    im_m = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    im_s = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    g = torch.Generator(device=dev).manual_seed(20261121)
    projs = {}
    for k, cdim in [("ok", 27), ("c1", 96), ("c2", 192), ("c3", 384), ("c4", 768), ("v2", 128), ("v3", 256), ("v4", 512)]:
        w = torch.randn(cdim, 128, generator=g, device=dev)
        projs[k] = w / w.norm(dim=0, keepdim=True).clamp_min(1e-8)

    def extract_style_pyramid(rgb):
        x = F.interpolate(rgb.float(), size=(320, 320), mode="bilinear", align_corners=False)
        lab = rgb_to_oklab(x)
        ok_p = F.unfold(lab, kernel_size=3, padding=1, stride=4).transpose(1, 2)[:, ::4, :]
        xn = (x - im_m) / im_s
        c1 = cnx[:2](xn)
        c2 = cnx[2:4](c1)
        c3 = cnx[4:6](c2)
        c4 = cnx[6:8](c3)
        v1 = vgg[:4](xn)
        v2 = vgg[4:9](v1)
        v3 = vgg[9:16](v2)
        v4 = vgg[16:23](v3)
        return {
            "ok": ok_p,
            "c1": c1.flatten(2).transpose(1, 2)[:, ::4, :],
            "c2": c2.flatten(2).transpose(1, 2),
            "c3": c3.flatten(2).transpose(1, 2),
            "c4": c4.flatten(2).transpose(1, 2),
            "v2": v2.flatten(2).transpose(1, 2)[:, ::16, :],
            "v3": v3.flatten(2).transpose(1, 2)[:, ::4, :],
            "v4": v4.flatten(2).transpose(1, 2),
        }

    def extract_whitened_gray_content(rgb):
        y = rgb_to_gray(rgb.float()).repeat(1, 3, 1, 1)
        yn = (F.interpolate(y, size=(256, 256), mode="bilinear", align_corners=False) - im_m) / im_s
        c2 = cnx[:4](yn)
        c3 = cnx[4:6](c2)

        def whiten(f):
            f = (f - f.mean((2, 3), keepdim=True)) / f.std((2, 3), keepdim=True, unbiased=False).clamp_min(1e-5)
            return F.normalize(f, dim=1)

        w3 = whiten(c3)
        fl = w3.flatten(2)
        ssim = torch.bmm(fl.transpose(1, 2), fl)
        return whiten(c2), w3, ssim

    with torch.no_grad():
        ref_pyr_5 = extract_style_pyramid(oneshot_rgb)

    tasks, z_src_750, _ = tasks_and_sources()
    sel = [
        i
        for s in range(5)
        for i in [j for j, t in enumerate(tasks) if t["source_style_id"] != t["target_style_id"] and t["target_style_id"] == s][::12][:10]
    ]
    sel_t = torch.tensor(sel)
    zs = z_src_750[sel_t].to(dev).float()
    tid = torch.tensor([tasks[i]["target_style_id"] for i in sel], device=dev)

    with torch.no_grad():
        src_rgb = decode_s10(dec_fp16, zs, batch=25, output_cpu=False).clamp(0.0, 1.0)
        sc2_t, sc3_t, ssim_t = extract_whitened_gray_content(src_rgb)

    P_Y = bundle["fnd"]["priors"]["P_Y"].to(dev).float()
    P_C = bundle["fnd"]["priors"]["P_C"].to(dev).float()
    z_ref_b = oneshot_z[tid]
    zy_s = torch.einsum("ij,bjhw->bihw", P_Y, zs)
    zc_s = torch.einsum("ij,bjhw->bihw", P_C, zs)
    zc_r = torch.einsum("ij,bjhw->bihw", P_C, z_ref_b)
    mu_s, sd_s = zc_s.mean((2, 3), keepdim=True), zc_s.std((2, 3), keepdim=True).clamp_min(1e-4)
    mu_r, sd_r = zc_r.mean((2, 3), keepdim=True), zc_r.std((2, 3), keepdim=True).clamp_min(1e-4)
    z_init = zy_s + torch.einsum("ij,bjhw->bihw", P_C, (zc_s - mu_s) / sd_s * sd_r + mu_r)

    stack, rb = FeatureStack(dev), reference_bank(dev)
    rd = [torch.as_tensor(x, device=dev, dtype=torch.float32) for x in rb["dino"]]
    rc = [torch.as_tensor(x, device=dev, dtype=torch.float32) for x in rb["clip"]]
    cproto = torch.as_tensor(stack.csd_prototypes, device=dev, dtype=torch.float32)
    src_pil = [Image.fromarray((x.detach().cpu().permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype("uint8")) for x in src_rgb]
    src_df = F.normalize(torch.as_tensor(stack.extract(src_pil, batch=10)["dino"], device=dev, dtype=torch.float32), dim=-1)

    for w_content in (3.2, 2.0, 1.2, 0.7):
        z_opt = z_init.clone().detach()
        t0 = time.time()
        for ch in range(0, len(zs), 10):
            sl = slice(ch, ch + 10)
            z_var = z_opt[sl].clone().detach().requires_grad_(True)
            opt = torch.optim.Adam([z_var], lr=0.065)
            ti_c = tid[sl]
            r_sub = {k: v[ti_c] for k, v in ref_pyr_5.items()}
            for _ in range(14):
                opt.zero_grad(set_to_none=True)
                rgb_p = decode_s10(dec_bf16, z_var, batch=10, dtype=torch.bfloat16, enable_grad=True, output_cpu=False).clamp(0.0, 1.0)
                cur_p = extract_style_pyramid(rgb_p)
                l_swd = (
                    1.5 * sliced_wasserstein_1d(cur_p["ok"], r_sub["ok"], projs["ok"])
                    + 0.4 * sliced_wasserstein_1d(cur_p["c1"], r_sub["c1"], projs["c1"])
                    + 0.6 * sliced_wasserstein_1d(cur_p["c2"], r_sub["c2"], projs["c2"])
                    + 0.9 * sliced_wasserstein_1d(cur_p["c3"], r_sub["c3"], projs["c3"])
                    + 1.2 * sliced_wasserstein_1d(cur_p["c4"], r_sub["c4"], projs["c4"])
                    + 0.3 * sliced_wasserstein_1d(cur_p["v2"], r_sub["v2"], projs["v2"])
                    + 0.5 * sliced_wasserstein_1d(cur_p["v3"], r_sub["v3"], projs["v3"])
                    + 0.8 * sliced_wasserstein_1d(cur_p["v4"], r_sub["v4"], projs["v4"])
                ).mean()
                c2_p, c3_p, ss_p = extract_whitened_gray_content(rgb_p)
                l_cnt = (
                    (1.0 - (c3_p * sc3_t[sl]).sum(1)).mean()
                    + 0.40 * (1.0 - (c2_p * sc2_t[sl]).sum(1)).mean()
                    + 0.25 * F.mse_loss(ss_p, ssim_t[sl])
                )
                dz = z_var - z_init[sl]
                l_tv = (dz[:, :, :, 1:] - dz[:, :, :, :-1]).square().mean() + (dz[:, :, 1:, :] - dz[:, :, :-1, :]).square().mean()
                loss = l_swd + w_content * l_cnt + 0.08 * l_tv
                loss.backward()
                opt.step()
            z_opt[sl] = z_var.detach()

        with torch.no_grad():
            rgb_o = decode_s10(dec_fp16, z_opt, batch=25)
            pils = [Image.fromarray((x.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype("uint8")) for x in rgb_o]
            fo = stack.extract(pils, batch=10)
            df, cf, csdf = [F.normalize(torch.as_tensor(fo[k], device=dev, dtype=torch.float32), dim=-1) for k in ("dino", "clip", "csd")]
            dc_v = (df * src_df).sum(-1)
            ds_t = torch.tensor([(df[j : j + 1] @ rd[int(tid[j])].T).max().item() for j in range(len(zs))])
            cs_t = torch.tensor([(cf[j : j + 1] @ rc[int(tid[j])].T).max().item() for j in range(len(zs))])
            t1 = (csdf @ cproto.T).argmax(1).eq(tid).float().mean().item()
            print(
                f"1Shot_wC={w_content:3.1f} ({time.time()-t0:.1f}s) | Top1={t1*100:5.1f}% "
                f"DINO-S={ds_t.mean():.4f} CLIP-S={cs_t.mean():.4f} Aff={(ds_t.mean()+cs_t.mean())/2:.4f} "
                f"DINO-C={dc_v.mean():.4f} | D-S={[round(ds_t[tid.cpu()==s].mean().item(),3) for s in range(5)]} "
                f"| D-C={[round(dc_v[tid.cpu()==s].mean().item(),3) for s in range(5)]}",
                flush=True,
            )


if __name__ == "__main__":
    main()
