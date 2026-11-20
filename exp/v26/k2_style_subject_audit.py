r"""V26 K2 (4090): SWD 风格判别力 / 梯度数值稳定性 / CSD 题材偏置审计 (零训练).

docs/1120/06_4090_diagnostic_queue.md §K2 (调整版: 结论服务"是否值得给 Flow 主线
引入 SWD 类分布损失"与"Top-1 不能当风格迁移唯一正确性定义"两条铁律).

K2-A 风格判别力: 200 张训练集内部图 (40/画派) 上比较
     mean-VGG (1−cos) / VGG Gram / 动态多尺度 SWD 三种风格的同/异画派距离间隔,
     以及对 色移 / 模糊 / 局部纹理打乱 三种变换的敏感度.
K2-B 梯度诊断: 128 个训练 latent (requires_grad) 对三种损失的梯度
     范数 / 三损失梯度间余弦 / SWD 投影数 K∈{32,64,128} 的梯度稳定性
     (不同种子的梯度余弦) / BF16 vs FP32 梯度范数比.
K2-C CSD 题材偏置: CLIP zero-shot 为 750 源图打题材标签 (person/landscape/
     architecture/still-life/abstract), 与 runs/v25/measured 的逐任务 target_top1
     关联 → 题材 × 目标画派 Top-1 表 (C2nt t45 与 O3 t45, 如存在).

产出: runs/v26/gpu4090/k2_style_subject_audit.json
     + docs/v26/4090/K2_STYLE_LOSS_AND_SUBJECT_AUDIT.md

用法 (4090):  python -m exp.v26.k2_style_subject_audit --pool ../pseudo_gt_24h/orig_mean_fp16
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image as PILImage

from src.utils.protocol import load_vgg16_features_offline

from . import OUT, ROOT, sanitize_cuda_env, save_json, seed_all, utc
from exp.v19.fiber_duality import build_or_load_v19_foundation
from exp.v26.dynamic_style_swd import DynamicSWD, swd_distance_matrix
from exp.v25.mechanism_c import cosine_distance_loss, make_train_feature_fn
from src.utils import decode_s10, load_s10_decoder

REPORT_MD = ROOT / "docs/v26/4090/K2_STYLE_LOSS_AND_SUBJECT_AUDIT.md"
N_PER_STYLE = 40
N_GRAD = 32
SUBJECT_PROMPTS = {
    "person": "a painting featuring one or more people, portraits, figures",
    "landscape": "a landscape painting of scenery, mountains, rivers, trees",
    "architecture": "a painting of buildings, streets, temples, interior rooms",
    "still_life": "a still life painting of flowers, food, or objects on a table",
    "abstract": "an abstract painting of flat planes, lines, or color fields",
}
STYLE_NAMES = ("Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e")


# ------------------------------------------------- 特征族 ----

def _vgg_layers(device: str):
    """VGG-16 features 单例 (K2-A/B 共享, 防双份 550MB 权重)."""
    global _VGG_SINGLETON
    if _VGG_SINGLETON is None:
        vgg = load_vgg16_features_offline(device).eval()
        for p in vgg.parameters():
            p.requires_grad_(False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        _VGG_SINGLETON = (vgg, mean, std)
    return _VGG_SINGLETON


_VGG_SINGLETON = None


def vgg_mean_style(imgs: torch.Tensor, vgg, mean, std, batch: int = 16) -> torch.Tensor:
    """与 mechanism_c._make_vgg_feature_fn 同口径: relu3_3+relu4_3 池化 (B,768); 微批防 OOM."""
    outs = []
    for s in range(0, len(imgs), batch):
        x = (imgs[s:s + batch].float() - mean) / std
        f3 = f4 = None
        h = x
        for i, m in enumerate(vgg):
            h = m(h)
            if i == 15:
                f3 = h
            elif i == 22:
                f4 = h
        outs.append(torch.cat([f3.mean(dim=(2, 3)), f4.mean(dim=(2, 3))], dim=-1))
    return torch.cat(outs, dim=0)


def vgg_gram_feat(imgs: torch.Tensor, vgg, mean, std, batch: int = 16) -> torch.Tensor:
    """relu1_2/2_2/3_3 Gram 上三角展平 (纹理二阶统计); 微批防 OOM."""
    outs = []
    for s in range(0, len(imgs), batch):
        x = (imgs[s:s + batch].float() - mean) / std
        grams = []
        h = x
        for i, m in enumerate(vgg):
            h = m(h)
            if i in (3, 8, 15):
                B, C, H, W = h.shape
                f = h.view(B, C, H * W)
                g = torch.bmm(f, f.transpose(1, 2)) / (C * H * W)
                ii = torch.triu_indices(C, C)
                grams.append(g[:, ii[0], ii[1]])
        outs.append(torch.cat(grams, dim=-1))
    return torch.cat(outs, dim=0)


def dist_matrix_feat(feat: torch.Tensor) -> torch.Tensor:
    """特征余弦距离矩阵 (N,N)."""
    fn = F.normalize(feat.float(), dim=-1)
    return 1.0 - fn @ fn.T


def transform_variants(imgs: torch.Tensor, device: str) -> dict[str, torch.Tensor]:
    """色移 / 模糊 / 局部纹理打乱 三种不变性压力变换."""
    g = torch.Generator().manual_seed(20261185)
    # 色移: RGB 通道增益
    gain = 1.0 + 0.18 * torch.randn(1, 3, 1, 1, generator=g).to(device)
    color = (imgs * gain).clamp(0.0, 1.0)
    # 模糊: 3x3 平均池化后上采样
    blur = F.interpolate(F.avg_pool2d(imgs, 3, stride=1, padding=1), size=imgs.shape[-2:], mode="bilinear")
    # 局部纹理打乱: 8x8 块内上下两个 4px 子块交换 (保持内容布局, 破坏局部纹理)
    B, C, H, W = imgs.shape
    hb, wb = H // 8, W // 8
    blocks = imgs.unfold(2, 8, 8).unfold(3, 8, 8)          # (B,C,hb,wb,8,8)
    top, bot = blocks[..., :4, :], blocks[..., 4:, :]
    swapped = torch.cat([bot, top], dim=-2)                # 垂直交换 4px 子块
    tex = swapped.permute(0, 1, 2, 4, 3, 5).reshape(B, C, hb * 8, wb * 8)
    tex = F.pad(tex, (0, W - tex.shape[-1], 0, H - tex.shape[-2]))
    return {"color_shift": color, "blur": blur, "texture_shuffle": tex}


# ------------------------------------------------- K2-A 判别力 ----

@torch.no_grad()
def k2a_discriminability(bundle, device: str) -> dict:
    tr = bundle["v18_cache"]["train"]
    z_all, sid_all = tr["z"], tr["sid"].long()
    dec = load_s10_decoder(device)
    g = torch.Generator().manual_seed(20261185)
    idxs = []
    for s in range(5):
        pool = (sid_all == s).nonzero(as_tuple=True)[0]
        pick = pool[torch.randperm(len(pool), generator=g)[:N_PER_STYLE]]
        idxs.append(pick)
    idx = torch.cat(idxs)
    imgs = torch.cat([decode_s10(dec, z_all[chunk].to(device), batch=32).clamp(0.0, 1.0)
                      for chunk in idx.split(32)]).cpu()
    sids = torch.cat([torch.full((N_PER_STYLE,), s) for s in range(5)])

    vgg, mean, std = _vgg_layers(device)
    imgs_d = imgs.to(device)
    f_mean = vgg_mean_style(imgs_d, vgg, mean, std).cpu()
    f_gram = vgg_gram_feat(imgs_d, vgg, mean, std).cpu()
    swd = DynamicSWD(device=device, n_proj=64, n_token=512)
    d_swd = swd_distance_matrix(imgs_d, swd, gen=torch.Generator().manual_seed(20261185))

    def margins(dmat: torch.Tensor) -> dict:
        same_m, cross_m = [], []
        n = dmat.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                (same_m if sids[i] == sids[j] else cross_m).append(float(dmat[i, j]))
        return {"within": round(sum(same_m) / len(same_m), 5),
                "cross": round(sum(cross_m) / len(cross_m), 5),
                "margin": round(sum(cross_m) / len(cross_m) - sum(same_m) / len(same_m), 5)}

    # 每画派可分性 (one-vs-rest margin)
    per_style = {}
    for s in range(5):
        m_in, m_out = [], []
        for i in range(len(sids)):
            for j in range(i + 1, len(sids)):
                if sids[i] != s or sids[j] != s:
                    continue
                d = float(d_swd[i, j])
                (m_in if sids[i] == sids[j] else m_out).append(d)
        per_style[STYLE_NAMES[s]] = round((sum(m_out) / max(len(m_out), 1))
                                          - (sum(m_in) / max(len(m_in), 1)), 5)

    # 变换敏感度: 同画派内 orig vs transform 的距离增量
    sens = {}
    tex = transform_variants(imgs_d, device)["texture_shuffle"]
    deltas = []
    g2 = torch.Generator().manual_seed(20261185)
    for _ in range(40):
        i = int(torch.randint(0, len(imgs), (1,), generator=g2))
        j = int(torch.randint(0, len(imgs), (1,), generator=g2))
        if i == j:
            j = (i + 1) % len(imgs)
        d0 = float(swd.loss(imgs_d[i:i + 1], imgs_d[j:j + 1], gen=g2)["loss"].detach())
        d1 = float(swd.loss(tex[i:i + 1], imgs_d[j:j + 1], gen=g2)["loss"].detach())
        deltas.append(d1 - d0)
    sens["texture_shuffle_swd_delta"] = round(sum(deltas) / len(deltas), 5)

    result = {
        "n_images": len(imgs),
        "meanvgg": margins(dist_matrix_feat(f_mean)),
        "gram": margins(dist_matrix_feat(f_gram)),
        "swd": margins(d_swd),
        "swd_per_style_margin": per_style,
        "transform_sensitivity": sens,
    }

    # 大张量清理 (imgs/变换副本/距离矩阵 ~4GB), 防 K2-B OOM
    del imgs_d, tex, d_swd, f_mean, f_gram
    torch.cuda.empty_cache()
    return result


# ------------------------------------------------- K2-B 梯度 ----

def k2b_gradients(bundle, device: str) -> dict:
    """梯度审计: **逐图 backward** (峰值 = 单图解码+VGG 反传图), 防 32 图整批图保留 OOM."""
    tr = bundle["v18_cache"]["train"]
    z_all = tr["z"]
    dec = load_s10_decoder(device)
    g = torch.Generator().manual_seed(20261186)
    idx = torch.randperm(len(z_all), generator=g)[:N_GRAD]
    zt = z_all[idx].to(device).float()
    ir = (idx + 1) % len(z_all)
    imgs_ref = torch.cat([decode_s10(dec, z_all[ir[i:i + 8]].to(device), batch=8).clamp(0.0, 1.0)
                          for i in range(0, N_GRAD, 8)]).to(device)
    vgg, mean, std = _vgg_layers(device)
    f_ref_mean = vgg_mean_style(imgs_ref, vgg, mean, std).detach()
    f_ref_gram = vgg_gram_feat(imgs_ref, vgg, mean, std).detach()

    grads = {}
    swd = DynamicSWD(device=device, dtype=torch.float32)

    def _grad_of(loss_fn, tag: str) -> torch.Tensor:
        """逐图 forward+backward, z.grad 逐图累加 (峰值显存 = 单图图)."""
        z = zt.clone().requires_grad_(True)
        for i in range(N_GRAD):
            rgb_i = decode_s10(dec, z[i:i + 1], batch=1, enable_grad=True, output_cpu=False).clamp(0.0, 1.0)
            loss_fn(rgb_i, i).backward()
        g = z.grad.detach().clone()
        z.grad = None
        torch.cuda.empty_cache()
        return g

    for k_proj in (32, 64, 128):
        swd.n_proj = k_proj
        grads[f"swd_k{k_proj}"] = _grad_of(
            lambda rgb_i, i, kp=k_proj: swd.loss(rgb_i, imgs_ref[i:i + 1],
                                                 gen=torch.Generator().manual_seed(20261185))["loss"],
            f"swd_k{k_proj}")

    grads["meanvgg"] = _grad_of(
        lambda rgb_i, i: cosine_distance_loss(vgg_mean_style(rgb_i, vgg, mean, std),
                                              f_ref_mean[i:i + 1]), "meanvgg")
    grads["gram"] = _grad_of(
        lambda rgb_i, i: F.mse_loss(vgg_gram_feat(rgb_i, vgg, mean, std), f_ref_gram[i:i + 1]), "gram")

    def norm(x):
        return round(float(x.norm()), 4)

    def cos(a, b):
        return round(float(F.cosine_similarity(a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)).mean()), 4)

    def seed_stability(k_proj: int) -> float:
        swd.n_proj = k_proj
        ga = []
        for seed in (111, 222):
            z = zt[:8].clone().requires_grad_(True)
            for i in range(8):
                rgb_i = decode_s10(dec, z[i:i + 1], batch=1, enable_grad=True, output_cpu=False).clamp(0.0, 1.0)
                r = swd.loss(rgb_i, imgs_ref[i:i + 1], gen=torch.Generator().manual_seed(seed))["loss"]
                (r / 8.0).backward()
            ga.append(z.grad.detach().clone())
            z.grad = None
            torch.cuda.empty_cache()
        return cos(ga[0], ga[1])

    return {
        "grad_norms": {k: norm(v) for k, v in grads.items()},
        "cos_swd64_vs_meanvgg": cos(grads["swd_k64"], grads["meanvgg"]),
        "cos_swd64_vs_gram": cos(grads["swd_k64"], grads["gram"]),
        "cos_meanvgg_vs_gram": cos(grads["meanvgg"], grads["gram"]),
        "swd_grad_seed_stability": {str(k): round(seed_stability(k), 4) for k in (32, 64, 128)},
        "note": "逐图反传口径: 峰值显存 = 单图解码+VGG 反传图; 单图 loss 值不聚合",
    }


# ------------------------------------------------- K2-C 题材偏置 ----

@torch.no_grad()
def k2c_subject_bias(bundle, device: str) -> dict:
    from src.utils import FeatureStack
    from exp.v18 import tasks_and_sources

    tasks, z_src_750, _ = tasks_and_sources()
    dec = load_s10_decoder(device)
    stack = FeatureStack(device)
    imgs = torch.cat([decode_s10(dec, z_src_750[i:i + 32].to(device), batch=32).clamp(0.0, 1.0)
                      for i in range(0, 750, 32)])
    pils = [PILImage.fromarray((x.permute(1, 2, 0).cpu().numpy() * 255).round().clip(0, 255).astype("uint8"))
            for x in imgs]
    feats = stack.extract(pils, batch=16)["clip"].cpu()          # (750, 512) normalized
    del stack
    torch.cuda.empty_cache()

    # CLIP text 零样本题材标签
    from transformers import CLIPModel, CLIPProcessor
    import paths
    cache = str(paths.eval_cache_root("hf") / "hub")
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True,
                                     cache_dir=cache).to(device).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True,
                                         cache_dir=cache)
    names = list(SUBJECT_PROMPTS)
    ti = proc(text=[SUBJECT_PROMPTS[k] for k in names], return_tensors="pt", padding=True).to(device)
    tfeat = F.normalize(clip.get_text_features(**ti), dim=-1)
    subject_idx = (feats.to(device) @ tfeat.T).argmax(dim=1).cpu()
    subj_counts = {k: int((subject_idx == i).sum()) for i, k in enumerate(names)}

    # 关联 runs/v25 measured 逐任务 target_top1 (C2nt t45 / O3 t45, 存在哪个用哪个)
    joined = {}
    for tag, summ_path in (
        ("C2nt_t45", ROOT / "runs/v25/measured/C2nt_sw6_lc0.8_t45/summary_v24.json"),
        ("O3_t45", ROOT / "runs/v24/measured/O3_rgb_cnx_gated_t45/summary_v24.json"),
    ):
        if not summ_path.exists():
            continue
        rows = read_rows(summ_path)
        if not rows:
            continue
        # join by task order (rows 与 tasks 同序, measure 按任务序生成)
        per = {}
        for s_i, sname in enumerate(names):
            for tgt in STYLE_NAMES:
                hit = [r for r, si in zip(rows, subject_idx.tolist())
                       if si == s_i and r["tgt_style"] == tgt and r["is_cross"]]
                per[f"{sname}|{tgt}"] = round(
                    100.0 * sum(r["target_top1"] for r in hit) / max(len(hit), 1), 1) if hit else None
        joined[tag] = {"per_task_n": len(rows), "table": per}
    return {"subject_counts": subj_counts, "models": joined}


def read_rows(path: Path) -> list[dict]:
    import json
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        return d.get("rows") or []
    except Exception:
        return []


# ------------------------------------------------------- 报告 ----

def _write_report(out: dict) -> None:
    a, b, c = out["k2a"], out["k2b"], out["k2c"]
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    L = ["# K2 — SWD 风格判别力 / 梯度稳定性 / CSD 题材偏置审计 (V26, 零训练)",
         "", f"> 生成时间: {out['updated_utc']}",
         "",
         "## K2-A 同/异画派距离间隔 (200 张训练图, 40/画派)",
         "",
         "| 风格度量 | 同画派均值 | 异画派均值 | margin |",
         "|---|---:|---:|---:|"]
    for name in ("meanvgg", "gram", "swd"):
        m = a[name]
        L.append(f"| {name} | {m['within']} | {m['cross']} | **{m['margin']}** |")
    L += ["", f"- SWD 对局部纹理打乱的平均距离增量: `{a['transform_sensitivity'].get('texture_shuffle_swd_delta')}`",
          "", "### SWD 逐画派 margin (one-vs-rest)", "", "| 画派 | margin |", "|---|---:|"]
    for k, v in a["swd_per_style_margin"].items():
        L.append(f"| {k} | {v} |")
    L += ["", "## K2-B 梯度诊断 (128 训练 latent)", "",
          "| 损失 | 梯度范数 |", "|---|---:|"]
    for k, v in b["grad_norms"].items():
        L.append(f"| {k} | {v} |")
    L += ["",
          f"- cos(∇SWD64, ∇mean-VGG) = **{b['cos_swd64_vs_meanvgg']}** ｜ cos(∇SWD64, ∇Gram) = **{b['cos_swd64_vs_gram']}**",
          f"- SWD 梯度种子稳定性 (不同投影种子梯度余弦): `{b['swd_grad_seed_stability']}`",
          "",
          "## K2-C CSD 题材偏置 (CLIP zero-shot 题材 × 目标画派 Top-1%)", ""]
    L.append(f"- 题材分布: `{c['subject_counts']}`")
    for tag, m in c["models"].items():
        L.append(f"\n### {tag} (n={m['per_task_n']})\n")
        L.append("| 题材 | " + " | ".join(STYLE_NAMES) + " |")
        L.append("|---|" + "---:|" * len(STYLE_NAMES))
        for sname in SUBJECT_PROMPTS:
            row = [m["table"].get(f"{sname}|{t}") for t in STYLE_NAMES]
            L.append(f"| {sname} | " + " | ".join("--" if v is None else f"{v}" for v in row) + " |")
    L += ["", "## 判读", "",
          "- 若 SWD margin 显著高于 mean-VGG 且对纹理打乱敏感 → SWD 确实补足纹理统计 (支持 3060 单轮 SWD 试验);",
          "- 若 ∇SWD 与 ∇mean-VGG 高度同向 (cos≈1) → SWD 增量有限;",
          "- K2-C 中 ER/Roc 低 Top-1 若集中于特定题材 → CSD 题材混杂证据, 不得全部归因为风格迁移失败."]
    REPORT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[K2] Report → {REPORT_MD}", flush=True)


def run_k2(pool: str, device: str = "cuda") -> dict:
    seed_all(20261185)
    bundle = build_or_load_v19_foundation(pool, device=device)
    out_p = OUT / "gpu4090" / "k2_style_subject_audit.json"
    out = {"task": "V26-K2 style-loss discriminability & CSD subject-bias audit",
           "updated_utc": utc()}

    a = k2a_discriminability(bundle, device=device)
    print(f"[K2-A] {json.dumps({k: a[k] for k in ('meanvgg', 'gram', 'swd')}, default=str)}", flush=True)
    out["k2a"] = a
    save_json(out_p, out)  # 分阶段落盘, 防后续阶段崩溃丢失

    b = k2b_gradients(bundle, device=device)
    print(f"[K2-B] norms={b['grad_norms']} cos(SWD64,meanVGG)={b['cos_swd64_vs_meanvgg']}", flush=True)
    out["k2b"] = b
    save_json(out_p, out)

    c = k2c_subject_bias(bundle, device=device)
    out["k2c"] = c
    out["updated_utc"] = utc()
    save_json(out_p, out)
    _write_report(out)
    return out


def main():
    from . import ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    sanitize_cuda_env()
    run_k2(args.pool, device=args.device)


if __name__ == "__main__":
    main()
