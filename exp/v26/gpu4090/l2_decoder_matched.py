r"""V26-L L2: Full Decoder 同域匹配微调公平对照 (D0/D1/D2/S10) — S10 收益归因.

docs/1120/08_v26_long_plan.md §L2 (独立于 L1 成败必做).

  D0        Full SD-VAE decoder (HF ft-ema 基座, 冻结)          — 域外基线
  D1        Full SD-VAE + WikiArt 同域微调 (匹配步数)           — 域适配效应
  D2        剪枝 S10 + 同域微调 (同数据/同步数/同 lr)           — 剪枝+域适配
  S10_as_is 历史剪枝 S10 (冻结)                                 — 现役解码器

匹配 FT 口径: **真图重建** (d5 子集, D1/D2 同 latents/同图/同步数) — D1 与 D0
同权重初始化, 若蒸馏到 D0 重建则自恒等必然空转; 真图目标才有域适配语义.
评测: **完全相同的生成 latent**
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
L2_BATCH = 1   # v2 修复: batch=8 fp32 全量 decoder FT 激活 ~21GB → OOM; batch=1 + bf16 + ckpt
L2_LR = 5e-5
GEN_RAW = OUT / "raw" / "A0_legacy_nbr_t45"
_VAE_SCALE = 0.18215


# ------------------------------------------------- 解码器包装 ----

def _load_full_vae(device: str):
    """Full VAE 基座: 本地 SD0 全量 VAE (deliver/checkpoints/sd0_vae → 上传至 4090
    runs/v26/cache/sd0_vae_full). HF 在 4090/本机均被网络阻断, 弃用在线下载."""
    from diffusers import AutoencoderKL
    local_dir = OUT / "cache" / "sd0_vae_full"
    assert local_dir.exists(), f"本地 Full VAE 缺失: {local_dir} (先上传 sd0_vae_full)"
    vae = AutoencoderKL.from_pretrained(str(local_dir), local_files_only=True).to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


def wrap_decoder(name: str, model, device: str, *, kind: str | None = None,
                 enable_grad: bool = False, dtype: torch.dtype = torch.float16):
    """统一解码包装 fn(z_scaled) → rgb01 [0,1].

    kind: "full" (diffusers AutoencoderKL.decode) 或 "s10" (decode_s10); 缺省按
    name 前缀推断. v4 修复: eval 臂名 ("D0"/"D1"/"D2") 不带 full 前缀, 原实现
    按名字分派把它们全部路由进 S10 分支 → decode_s10(Half 输入) × fp32 Full VAE
    → RuntimeError, L2 评测从未成功过.
    enable_grad: S10 路径 (decode_s10 默认 inference_mode) 训练时必须显式开启.
    dtype: S10 计算精度 (D2 FT=fp32; fp16 反传溢出 nan).
    """
    if kind is None:
        kind = "full" if name.startswith("full") else "s10"
    if kind == "full":
        def fn(z):
            out = model.decode(z.float().to(device) / _VAE_SCALE).sample
            return (out / 2 + 0.5).clamp(0.0, 1.0)
        return fn
    def fn(z):
        return decode_s10(model, z, batch=min(len(z), 16), dtype=dtype,
                          enable_grad=enable_grad, output_cpu=False).clamp(0.0, 1.0)
    return fn


# ------------------------------------------------- 同域匹配 FT ----

def _build_ft_targets(train_ids, max_n: int = 2400) -> tuple[list[int], torch.Tensor]:
    """真图 FT 目标 (d5 子集): 4090 无 wikiart 原图; distinct5_512 按
    train_512clean/<Style>/<id>.jpg 布局. 返回 (latent 索引, uint8 真图 (N,3,512,512)).

    v4 修复: 原实现 D1 蒸馏到 D0 冻结重建 — D1 与 D0 同权重初始化 → 自恒等
    (loss≈0 无梯度, 必然空转). 计划 "WikiArt 同域微调" 的语义 = 真图重建;
    D1/D2 共用同一 d5 子集 (协议匹配).
    """
    from PIL import Image as PILImage
    import numpy as np
    d5_root = ROOT / "datasets/distinct5_512/train_512clean"
    idxs, imgs = [], []
    for i, pid in enumerate(train_ids[:max_n]):
        pid = str(pid)
        if not pid.startswith("d5__"):
            continue
        parts = pid.split("__")
        # id 'd5__<style>__<style>__<stem>' → 文件 '<style>__<stem>.jpg' (无 d5 前缀)
        p = d5_root / parts[1] / f"{parts[1]}__{'__'.join(parts[3:])}.jpg"
        if not p.exists():
            continue
        im = PILImage.open(p).convert("RGB")
        if im.size != (512, 512):
            im = im.resize((512, 512))
        idxs.append(i)
        imgs.append(torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1))
    return idxs, torch.stack(imgs) if imgs else torch.zeros(0, 3, 512, 512, dtype=torch.uint8)


def _ft(s10_or_decoder_module, wrap_fn, train_latents: torch.Tensor, targets: torch.Tensor,
        ft_idx: list[int], steps: int = L2_STEPS, batch: int = L2_BATCH, lr: float = L2_LR,
        device: str = "cuda", tag: str = "", amp: bool = True) -> None:
    """同域真图重建 FT: decoder(z_i) → 真图 x_i (D1/D2 同一 latents/真图/batch/lr/步数)."""
    import exp.v25 as _v25
    _v25.seed_all(20266200)
    params = [p for p in s10_or_decoder_module.parameters() if p.requires_grad]
    assert params, f"[L2 FT {tag}] 无可训练参数 (requires_grad 全 False)"
    opt = torch.optim.AdamW(params, lr=lr)
    order = torch.randperm(len(ft_idx), generator=torch.Generator().manual_seed(20266200))
    t0 = time.time()
    for step in range(1, steps + 1):
        start = ((step - 1) * batch) % max(len(order) - batch, 1)
        sel = order[start:start + batch]
        idx = torch.tensor([ft_idx[j] for j in sel], dtype=torch.long)
        z = train_latents[idx].to(device).float()
        target = targets[sel].to(device).float() / 255.0
        if amp:
            with torch.autocast("cuda", torch.bfloat16):
                rgb = wrap_fn(z).float()
        else:
            rgb = wrap_fn(z).float()
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
    from src.utils import FeatureStack
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
    tr_cache = bundle["v18_cache"]["train"]
    train_latents = tr_cache["z"][:2400].clone()
    ft_idx, ft_imgs = _build_ft_targets(tr_cache["ids"])
    assert len(ft_idx) >= 512, f"d5 真图 FT 目标不足: {len(ft_idx)}"
    print(f"[L2] FT 目标: {len(ft_idx)} 张 d5 真图 (真图重建, 非 D0 自蒸馏)", flush=True)
    torch.cuda.empty_cache()

    s10_asis = load_s10_decoder(device)
    for p in s10_asis.parameters():
        p.requires_grad_(False)

    full = _load_full_vae(device)
    # v4 修复: 独立冻结副本作蒸馏参考 — 原实现 ref_fn 与 D1 共享同一 full 对象,
    # target 随训练同步漂移 → D1 loss 自恒等 (0.0008 空转), D2 目标被污染
    full_ref = _load_full_vae(device)
    try:
        full.enable_gradient_checkpointing()  # 反传激活重计算, 显存大头 (v2 修复)
        print("[L2] full VAE gradient checkpointing ON", flush=True)
    except Exception as exc:
        print(f"[L2] gradient checkpointing unavailable ({exc}) — 依赖 batch=1+bf16", flush=True)
    out_p = OUT / "gpu4090" / "l2_decoder_matched.json"

    # D1: Full 同域真图 FT (只解冻 decoder 子模块, 编码器冻结省显存)
    for n, p in full.named_parameters():
        p.requires_grad_(n.startswith("decoder") or n.startswith("post_quant_conv"))
    _ft(full, wrap_decoder("full_ft", full, device, kind="full"), train_latents, ft_imgs,
        ft_idx, device=device, tag="D1_full_ft")

    # D2: S10 同域真图 FT (同同步数/同图) — fp32 端到端 (fp16 反传溢出 → nan;
    # S10 仅 10.66M, batch=1 fp32 激活 ~3GB 完全安全)
    s10_ft = load_s10_decoder(device).to(torch.float32)
    _ft(s10_ft, wrap_decoder("s10_ft", s10_ft, device, kind="s10", enable_grad=True,
                             dtype=torch.float32), train_latents, ft_imgs,
        ft_idx, device=device, tag="D2_s10_ft", amp=False)
    s10_dt = next(s10_ft.parameters()).dtype
    for p in full.parameters():
        p.requires_grad_(False)
    torch.cuda.empty_cache()

    dec_wraps = {
        "D0_full_frozen": wrap_decoder("D0", full_ref, device, kind="full"),
        "D1_full_ft": wrap_decoder("D1", full, device, kind="full"),
        "D2_s10_ft": wrap_decoder("D2", s10_ft, device, kind="s10", dtype=s10_dt),
        "S10_as_is": wrap_decoder("S10", s10_asis, device, kind="s10"),
    }
    results = {"task": "V26-L L2 decoder matched-FT attribution",
               "steps": L2_STEPS, "updated_utc": utc(),
               "ft_target": {"kind": "real_images_d5", "n_images": len(ft_idx),
                             "note": "真图重建 (d5 子集); D0 冻结基座仅作 eval 臂, 不作蒸馏目标"},
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
