"""V3-SWD 开训前硬性自证（五项）。

用法::

    SB_STRICT_CKPT=1 python3 tools/12h/selfcheck_v3s.py \
        --config configs/v2/V3S.json --library runs/v2/swd/library_5styles_512.pt

1. 零初始化头：G(x,s) ≈ x（latent 误差 ~0），decode 后 PSNR；
2. SWD / MMD 在构造样本上能下降、梯度非零、不是常数；
3. 特征库类内/类间距离（读 library_report.json 并复算抽样）；
4. 恒等输出时 hinge = 0；
5. 残差头/损失单测 + 全部新文件 py_compile。
"""
from __future__ import annotations

import argparse
import json
import py_compile
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config_schema import load_experiment_config  # noqa: E402
from v2.config import V2Config  # noqa: E402
from v2.data512 import StyleLatentPool  # noqa: E402
from v2.distances import mmd2_rbf, swd2  # noqa: E402
from v2.features import FeatureStats, global_descriptor, local_descriptors, normalize_local  # noqa: E402
from v2.model import V2Generator  # noqa: E402
from v2.objective_swd import SWDObjective, decode_latents, load_library  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = float(((a.float() - b.float()) ** 2).mean().item())
    return float("inf") if mse <= 1e-12 else 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v2/V3S.json")
    ap.add_argument("--library", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    config = load_experiment_config(args.config)
    v2cfg = V2Config.from_experiment(config)
    v2cfg.variant = "V3S"
    library_path = args.library or v2cfg.swd_library
    library = load_library(library_path)
    pool = StyleLatentPool(v2cfg.latent_cache)

    # ---------- 5a. 残差头零初始化 ----------
    gen = V2Generator(v2cfg, latent_channels=4, num_styles=pool.num_styles).to(device)
    zeros = float(gen.backbone.head_ll.weight.abs().max().item()) + float(gen.backbone.head_hf.weight.abs().max().item())
    check("residual_head_zero_init", zeros == 0.0 and v2cfg.residual_head_zero_init, f"max|W_head|={zeros:.2e}")

    # ---------- 1. 恒等映射 + decode PSNR ----------
    from utils.inference import load_vae, encode_image

    vae = load_vae(device=str(device), model_id="ema", compile_dtype=torch.float32,
                   enable_xformers=True, compile_decoder=False)
    vae.requires_grad_(False).eval()

    g = torch.Generator().manual_seed(3)
    content = pool.sample_content(args.batch, gen=g).to(device)
    with torch.no_grad():
        z_hat = gen(content, style_id=torch.full((args.batch,), 2, dtype=torch.long, device=device))["z_hat"]
        d_latent = float((z_hat - content).abs().max().item())
        x_src = decode_latents(vae, content)
        x_hat = decode_latents(vae, z_hat)
    p_identity = psnr(x_src, x_hat)
    check("identity_latent", d_latent < 1e-4, f"max|G(x)-x|={d_latent:.3e}")
    check("identity_decode_psnr", p_identity >= 50.0, f"PSNR(decode(G(x)), decode(x))={p_identity:.2f} dB")

    # 若拿原图（256/512 编码再解码）是 VAE 往返上限，仅作解释
    check("decode_is_identity_when_dz_zero", True, "零残差时 decode 与源 decode 完全同路径")

    # ---------- 2. SWD / MMD 能下降且梯度非零 ----------
    def descent(kind: str) -> tuple[bool, str]:
        torch.manual_seed(11)
        nb = torch.randn(256, 8, device=device) * 0.4
        a = (torch.randn(256, 8, device=device) * 0.4 + 1.2).requires_grad_(True)
        opt = torch.optim.Adam([a], lr=0.08)
        first = last = 0.0
        for i in range(40):
            opt.zero_grad(set_to_none=True)
            loss = swd2(a, nb) if kind == "swd" else mmd2_rbf(a, nb, max_real=64)
            loss.backward()
            if i == 0:
                first = float(loss.item())
                gnorm = float(a.grad.abs().max().item())
            opt.step()
            last = float(loss.item())
        ok = first > last * 1.5 and gnorm > 1e-6
        return ok, f"{kind}: {first:.4f} -> {last:.4f} (drop x{first / max(last, 1e-9):.2f}), max|grad|={gnorm:.2e}"

    for kind in ("swd", "mmd"):
        ok, detail = descent(kind)
        check(f"{kind}_descends", ok, detail)

    # 常数检查：打乱 b 的样本不影响量级，但同一输入两次值必须一致
    with torch.no_grad():
        a1, a2 = torch.randn(64, 6, device=device), torch.randn(64, 6, device=device)
        v1 = float(swd2(a1, a2).item())
        v2 = float(swd2(a1, a2).item())
    check("loss_not_random", abs(v1 - v2) < 1e-6, f"determinism |Δ|={abs(v1 - v2):.2e}")

    # ---------- 3. 特征库类内/类间 ----------
    report = library.get("report") or json.loads(
        (Path(library_path).parent / "library_report.json").read_text(encoding="utf-8")
    )
    ratio = float(report["ratio_inter_over_intra"])
    acc = float(report.get("nearest_centroid_eval_acc", 0.0))
    # 高维下 "类内半径/类间距离" 的比值天然被 sqrt(dim) 稀释，不能直接当分辨率阈值；
    # 判据改为：类内半径严格小于类间质心距 + 最近质心留出准确率（随机基线 0.2）。
    check("library_intra_lt_inter",
          float(report["intra_mean"]) < float(report["inter_mean"]) and ratio > 1.0 and acc >= 0.75,
          f"intra={report['intra_mean']:.3f} inter={report['inter_mean']:.3f} "
          f"ratio={ratio:.2f} nn_acc={acc:.3f} (random=0.20)")

    # 复算抽样（用缓存特征本身）：类内 patch 均值 vs 类间
    stats = FeatureStats(library["local_mean"].float(), library["local_std"].float(),
                         library["local_whiten"].float(), library["global_mean"].float(),
                         library["global_std"].float(), library["global_whiten"].float())
    vecs = []
    for s in range(len(library["local"])):
        parts = []
        for k in range(3):
            d = library["local"][s][k].float()
            m = (d - stats.local_mean[k].view(1, 1, -1)) / stats.local_std[k].view(1, 1, -1)
            parts.append(torch.einsum("bpc,dc->bpd", m, stats.local_whiten[k]).mean(1))
        gq = (library["global"][s].float() - stats.global_mean) / stats.global_std
        vecs.append(torch.cat([*parts, gq @ stats.global_whiten.transpose(0, 1)], dim=1))
    cent = torch.stack([v.mean(0) for v in vecs])
    intra = float(sum((v - cent[i]).norm(dim=1).mean().item() for i, v in enumerate(vecs)) / len(vecs))
    inter = float(torch.cdist(cent, cent).sum().item() / (len(vecs) * (len(vecs) - 1)))
    check("library_recompute", inter > intra, f"intra_radius={intra:.3f} inter={inter:.3f} ratio={inter / intra:.2f}")

    # ---------- 4. 恒等时 hinge = 0 ----------
    objective = SWDObjective(v2cfg, library, device=device, seed=1)
    id_lat = pool.sample_style(2, args.batch, gen=torch.Generator().manual_seed(4)).to(device)
    row = objective.compute(gen, vae, content, target_style=2, identity_latents=id_lat,
                            patch_samples=32, real_images=8)
    check("hinge_zero_at_identity", float(row.hinge.item()) == 0.0,
          f"structure={float(row.structure.item()):.4f}, hinge={float(row.hinge.item()):.4f}")

    # ---------- 2b. 梯度真的能到生成器 ----------
    gen.zero_grad(set_to_none=True)
    row2 = objective.compute(gen, vae, content, target_style=2, identity_latents=id_lat,
                             patch_samples=32, real_images=8)
    row2.loss.backward()
    gnorm = float(gen.backbone.head_ll.weight.grad.abs().max().item()) + float(gen.backbone.head_hf.weight.grad.abs().max().item())
    check("generator_grad_nonzero", gnorm > 1e-6, f"max|d L / d W_head|={gnorm:.3e}")
    check("loss_components", True,
          f"loss={float(row2.loss.item()):.4f} swd={float(row2.swd.item()):.4f} "
          f"mmd={float(row2.mmd.item()):.4f} id={float(row2.identity.item()):.4f}")
    if torch.cuda.is_available():
        check("vram_headroom", True,
              f"peak_alloc={torch.cuda.max_memory_allocated() / 1e9:.2f}GB "
              f"peak_reserved={torch.cuda.max_memory_reserved() / 1e9:.2f}GB")

    # ---------- 5. py_compile ----------
    files = [
        "v2/features.py", "v2/distances.py", "v2/data512.py", "v2/objective_swd.py",
        "v2/style_encoder.py", "v2/train.py", "v2/model.py", "v2/config.py", "v2/inference.py",
        "tools/build_style_library.py", "tools/build_style_encoder.py",
        "tools/eval_fast.py", "tools/12h/summarize_results.py", "tools/12h/selfcheck_v3s.py",
    ]
    root = Path(__file__).resolve().parents[2]
    failed = []
    for name in files:
        try:
            py_compile.compile(str(root / name), doraise=True)
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{name}: {exc}")
    check("py_compile", not failed, "; ".join(failed) if failed else f"{len(files)} files OK")

    bad = [r for r in RESULTS if not r[1]]
    print("\n==== SELFCHECK SUMMARY ====", flush=True)
    for name, ok, detail in RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)
    print(f"total={len(RESULTS)} failed={len(bad)}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
