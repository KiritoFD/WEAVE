r"""V26-L L2: Full Decoder 同域匹配微调公平对照 (D0/D1/D2/S10) — S10 收益归因.

docs/1120/08_v26_long_plan.md §L2 (独立于 L1 成败必做).

  D0        Full SD-VAE decoder (HF ft-ema 基座, 冻结)          — 域外基线
  D1        Full SD-VAE + WikiArt 同域微调 (匹配步数)           — 域适配效应
  D2        剪枝 S10 + 同域微调 (同数据/同步数/同 lr)           — 剪枝+域适配
  S10_as_is 历史剪枝 S10 (冻结)                                 — 现役解码器

匹配 FT 口径: 蒸馏到 **D0 (冻结 Full) 的域内重建** (与 S10 历史训练的蒸馏对齐) —
D1/D2 同一批训练 latents、同 batch、同 lr、同步数. 评测: **完全相同的生成 latent**
(runs/v26/raw/A0_legacy_nbr_t45 的 750 输出) 分别解码 → 四裁判 (隔离进程) +
holdout 真图重建 PSNR/LPIPS/DINO-cos + 4090 解码耗时.

用法:  python -m exp.v26.gpu4090.l2_decoder_matched --pool ../pseudo_gt_24h/orig_mean_fp16
"""
from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from exp.v26 import OUT, ROOT, read_json, sanitize_cuda_env, save_json, utc
from src.utils import decode_s10, load_s10_decoder

L2_STEPS = 1000
L2_BATCH = 8
L2_LR = 5e-5
GEN_RAW = OUT / "raw" / "A0_legacy_nbr_t45"
_VAE_SCALE = 0.18215


# ------------------------------------------------- 解码器包装 ----

def _load_full_vae(device: str):
    """HF SD-VAE ft-ema 基座 (4090 联网下载, 本地缓存)."""
    from diffusers import AutoencoderKL
    import paths
    cache = str(paths.eval_cache_root("hf") / "hub")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema",
                                        cache_dir=cache).to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


def wrap_decoder(name: str, model, device: str):
    """统一解码包装 fn(z_scaled) → rgb01 [0,1]; name ∈ {full*, s10*}."""
    if name.startswith("full"):
        def fn(z):
            out = model.decode(z.float().to(device) / _VAE_SCALE).sample
            return (out / 2 + 0.5).clamp(0.0, 1.0)
        return fn
    def fn(z):
        return decode_s10(model, z, batch=min(len(z), 16), output_cpu=False).clamp(0.0, 1.0)
    return fn


# ------------------------------------------------- 同域匹配 FT ----

def _ft(s10_or_decoder_module, wrap_fn, train_latents: torch.Tensor, ref_fn,
        steps: int = L2_STEPS, batch: int = L2_BATCH, lr: float = L2_LR,
        device: str = "cuda", tag: str = "") -> None:
    """同域蒸馏 FT: 模型解码(latent) → 对齐 D0 (冻结 Full) 的重建输出.

    与 S10 历史训练口径一致 (蒸馏到 Full VAE 重建), D1/D2 完全同配.
    """
    import exp.v25 as _v25
    _v25.seed_all(20266200)
    params = [p for p in s10_or_decoder_module.parameters() if p.requires_grad]
    for p in s10_or_decoder_module.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=lr)
    order = torch.randperm(len(train_latents), generator=torch.Generator().manual_seed(20266200))
    t0 = time.time()
    for step in range(1, steps + 1):
        start = ((step - 1) * batch) % max(len(order) - batch, 1)
        idx = order[start:start + batch]
        z = train_latents[idx].to(device).float()
        with torch.no_grad():
            target = ref_fn(z)
        rgb = wrap_fn(z)
        loss = F.l1_loss(rgb, target) + 0.5 * F.mse_loss(rgb, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step == 1 or step % 200 == 0 or step == steps:
            print(f"[L2 FT {tag}] step {step}/{steps} loss={float(loss):.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    torch.cuda.empty_cache()


# ------------------------------------------------- 评测 ----

@torch.no_grad()
def eval_referee(bundle, dec_wraps: dict, device: str) -> dict:
    """同一生成 latent 分别解码 → 四裁判隔离进程对比."""
    from src.eval.runner import run_measure_subprocess
    from src.utils.raw_io import RawWriter

    lat_files = sorted(glob.glob(str(GEN_RAW / "*.pt")))
    assert lat_files, f"生成 latent 缺失: {GEN_RAW}"
    results = {}
    for name, fn in dec_wraps.items():
        arm = f"L2dec_{name}"
        wr = RawWriter(arm, raw_root=OUT / "raw", kind="v24")
        for f in lat_files:
            d = torch.load(f, map_location="cpu", weights_only=False)
            rgb = fn(d["z_final"].float().to(device))
            wr.append(d["ids"], rgb.cpu(), z_final=d["z_final"].cpu())
        wr.finish()
        s = run_measure_subprocess(OUT / "raw" / arm, OUT / "measured" / arm,
                                   arm_name=arm, device=device)
        c = s["cross600"]
        results[name] = {k: round(c[k], 5) for k in ("dino_c", "dino_s", "lpips", "csd_top1")}
        print(f"[L2 eval] {name}: {results[name]}", flush=True)
    return results


@torch.no_grad()
def eval_recon(bundle, dec_wraps: dict, device: str, n: int = 96) -> dict:
    """holdout 真图重建: PSNR / LPIPS / DINO-cos (每解码器一致的真图目标)."""
    from src.utils._local_eval_refs import FeatureStack
    from PIL import Image as PILImage

    ho = bundle["v18_cache"]["holdout"]
    z_all = ho["z"][:n].to(device).float()
    dec_s10 = load_s10_decoder(device)
    stack = FeatureStack(device)

    def to_pil(imgs):
        return [PILImage.fromarray((x.permute(1, 2, 0).cpu().numpy() * 255).round().clip(0, 255).astype("uint8"))
                for x in imgs]

    imgs_true = torch.cat([decode_s10(dec_s10, z_all[i:i + 16], batch=16).clamp(0.0, 1.0)
                           for i in range(0, len(z_all), 16)])
    f_true = stack.extract(to_pil(imgs_true), batch=16)["dino"].cpu()
    res = {}
    for name, fn in dec_wraps.items():
        imgs = torch.cat([fn(z_all[i:i + 16]).clamp(0.0, 1.0) for i in range(0, len(z_all), 16)])
        mse = float((imgs - imgs_true).square().mean())
        psnr = round(float(-10.0 * torch.log10(torch.tensor(max(mse, 1e-8)))), 2)
        f_rec = stack.extract(to_pil(imgs), batch=16)["dino"].cpu()
        dcos = round(float(F.cosine_similarity(f_rec, f_true, dim=-1).mean()), 5)
        lp = round(float(stack.lpips_vs((imgs * 255).round().to(torch.uint8), imgs_true, batch=8).mean()), 5)
        # 解码耗时 (batch=16, 4090)
        t0 = time.time()
        for _ in range(3):
            fn(z_all[:16])
        torch.cuda.synchronize(device)
        dt = round((time.time() - t0) / 3.0 / 16 * 1000, 2)
        res[name] = {"psnr_db": psnr, "dino_cos_recon": dcos, "lpips_recon": lp,
                     "decode_ms_b16": dt}
        print(f"[L2 recon] {name}: {res[name]}", flush=True)
    return res


# ------------------------------------------------- 主流程 ----

def run_l2(pool: str, device: str = "cuda") -> str:
    import exp.v25 as _v25
    _v25.seed_all(20266200)

    bundle = build_or_load_bundle(pool, device)
    train_latents = bundle["v18_cache"]["train"]["z"][:2400].clone()

    s10_asis = load_s10_decoder(device)
    for p in s10_asis.parameters():
        p.requires_grad_(False)

    full = _load_full_vae(device)
    out_p = OUT / "gpu4090" / "l2_decoder_matched.json"

    # D0 基线 (冻结) 蒸馏参考
    ref_fn = wrap_decoder("full_ref", full, device)

    # D1: Full 同域 FT (只解冻 decoder 子模块, 编码器冻结省显存)
    for n, p in full.named_parameters():
        p.requires_grad_(n.startswith("decoder") or n.startswith("post_quant_conv"))
    _ft(full, wrap_decoder("full_ft", full, device), train_latents, ref_fn,
        device=device, tag="D1_full_ft")

    # D2: S10 同域 FT (同同步数)
    s10_ft = load_s10_decoder(device)
    _ft(s10_ft, wrap_decoder("s10_ft", s10_ft, device), train_latents, ref_fn,
        device=device, tag="D2_s10_ft")
    for p in full.parameters():
        p.requires_grad_(False)
    torch.cuda.empty_cache()

    dec_wraps = {
        "D0_full_frozen": wrap_decoder("D0", full, device),
        "D1_full_ft": wrap_decoder("D1", full, device),
        "D2_s10_ft": wrap_decoder("D2", s10_ft, device),
        "S10_as_is": wrap_decoder("S10", s10_asis, device),
    }
    results = {"task": "V26-L L2 decoder matched-FT attribution",
               "steps": L2_STEPS, "updated_utc": utc(),
               "referee": eval_referee(bundle, dec_wraps, device),
               "recon": eval_recon(bundle, dec_wraps, device)}
    save_json(out_p, results)
    _write_report(results)
    print(f"[L2] DONE → {out_p}", flush=True)
    return "done"


def _write_report(results: dict) -> None:
    p = ROOT / "docs/v26/4090/L2_DECODER_MATCHED.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    L = ["# L2 — Full Decoder 同域匹配微调归因 (V26-L)", "",
         f"> 生成时间: {results['updated_utc']} ｜ FT 步数: {results['steps']} (D1/D2 同配)",
         "", "## 生成 latent 解码对比 (同一 750 输出, Cross600 四裁判)",
         "", "| 解码器 | DINO-C ↑ | DINO-S ↑ | LPIPS ↓ | CSD Top-1 ↑ |",
         "|---|---:|---:|---:|---:|"]
    for name, r in results["referee"].items():
        L.append(f"| {name} | {r['dino_c']} | {r['dino_s']} | {r['lpips']} | {r['csd_top1']*100:.2f}% |")
    L += ["", "## 真图重建 (holdout 96)", "",
          "| 解码器 | PSNR dB | DINO-cos | LPIPS | 解码 ms/b16 |", "|---|---:|---:|---:|---:|"]
    for name, r in results["recon"].items():
        L.append(f"| {name} | {r['psnr_db']} | {r['dino_cos_recon']} | {r['lpips_recon']} | {r['decode_ms_b16']} |")
    L += ["", "## 判读 (docs/1120/08 §L2)", "",
          "- D1 ≈ D2 >> D0 → 同域 FT 主导收益, S10 主要是轻量域适配解码器;",
          "- D2 > D1 → 剪枝+FT 有独立协同收益 (频谱瓶颈正则化证据);",
          "- 未收敛 → INCONCLUSIVE, 优先延长 matched 训练, 不得按早期曲线下结论."]
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[L2] Report → {p}", flush=True)


def build_or_load_bundle(pool: str, device: str):
    from exp.v19.fiber_duality import build_or_load_v19_foundation
    return build_or_load_v19_foundation(pool, device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    run_l2(args.pool, device=args.device)


if __name__ == "__main__":
    main()
