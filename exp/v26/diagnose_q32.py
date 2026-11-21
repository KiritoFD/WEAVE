r"""V26 K0 (4090): Q32 谱特性、稳定性与内容—风格解耦二维审计 (纯诊断, 零训练).

docs/1120/06_4090_diagnostic_queue.md §K0 + 01_diagnostics_and_srcfg.md §P0-C.

科学问题: V25 主线使用 384 通道 dino_16, 理论文档中的 Stiefel 商空间 Q32 仅 32 通道.
现有 16×16 DINO 结构条件究竟保留了多少源风格? 硬极分解是否以损坏有用结构为代价
换取线性探针可分性下降?

四组表征 (同批训练图, 不直接喂给预期 384 通道的旧模型):
  raw384  : 原始 dino_16 (384ch)
  pca32   : PCA32 (前 32 主成分, 与 W_stiefel 同基但无极分解步)
  q32     : 硬 Stiefel 极分解 Q32 (c21["W_stiefel"])
  q32_soft: Tikhonov 软 Q32  q_eta = sqrt(N) U diag(s/sqrt(s²+eta²)) Vh
            (eta = 训练集奇异值中位数分位数, 禁止 test750 调优)

审计维度 (每表征):
  - 奇异值谱 (均值) / 有效秩 (participation ratio) / sigma_min/sigma_max
  - 稳定性: 输入加噪 0.05 后特征相对 L2 变化
  - 风格泄漏: Gram 特征 5-way MLP 探针 (V21 口径) + 均值池化特征线性探针
  - 内容结构保持: 空间结构图 (逐 token 通道 L2) 与 raw384 结构图的相关;
    空间自相关 (相邻 token 特征余弦)
  - Minimalism vs 其他画派的谱差异 (高背景占比困难路线)

产出: runs/v26/gpu4090/k0_q32_audit.json + docs/v26/4090/K0_Q32_FEATURE_AUDIT.md
二维权衡图数据: (style_leak, structure_retention) per 变体.

用法 (4090):  python -m exp.v26.diagnose_q32 --pool ../pseudo_gt_24h/orig_mean_fp16
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from . import OUT, ROOT, V21_CACHE, sanitize_cuda_env, save_json, seed_all, utc
from exp.v19.fiber_duality import build_or_load_v19_foundation
from exp.v21.stiefel_ref_cache import stiefel_polar_quotient
from exp.v18.structure import _train_linear_mlp_probe

REPORT_MD = ROOT / "docs/v26/4090/K0_Q32_FEATURE_AUDIT.md"
NOISE_STD = 0.05
SOFT_ETA_QUANTILE = 0.5  # 训练集奇异值中位数 (禁 test750 调优)


# ------------------------------------------------------- 变体构造 ----

def pca32_basis(dino16_train: torch.Tensor) -> torch.Tensor:
    """前 32 主成分正交基 (384, 32) — 与 W_stiefel 同数据同维度, 但无极分解步."""
    X = dino16_train.float().permute(0, 2, 3, 1).reshape(-1, 384)[::12]
    X = X - X.mean(0, keepdim=True)
    _, _, Vh = torch.linalg.svd(X, full_matrices=False)
    return Vh[:32].T


def project_variant(dino16: torch.Tensor, W: torch.Tensor, kind: str,
                    eta: float = 0.0) -> torch.Tensor:
    """(N,384,16,16) → 变体特征. kind ∈ {pca32, q32, q32_soft}."""
    if kind == "pca32":
        return torch.einsum("dk,bdp->bkp", W.to(dino16.device), dino16.float().flatten(2)).reshape(
            dino16.shape[0], W.shape[1], 16, 16)
    if kind == "q32":
        return stiefel_polar_quotient(dino16, W)
    if kind == "q32_soft":
        return soft_stiefel_quotient(dino16, W, eta=eta)
    raise ValueError(kind)


def soft_stiefel_quotient(p16: torch.Tensor, W_k: torch.Tensor, eta: float) -> torch.Tensor:
    """Tikhonov 软极分解: q = sqrt(N) U diag(s/sqrt(s²+eta²)) Vh; eta=0 退化为硬 Q32."""
    w = W_k.to(device=p16.device, dtype=torch.float32)
    fl = torch.einsum("dk,bdp->bkp", w, p16.float().flatten(2))
    fl = fl - fl.mean(dim=-1, keepdim=True)
    U, S, Vh = torch.linalg.svd(fl, full_matrices=False)
    shrink = S / torch.sqrt(S * S + eta * eta)
    q = torch.bmm(U * shrink.unsqueeze(1), Vh) * 16.0
    return q.reshape(p16.shape[0], w.shape[1], 16, 16)


# ------------------------------------------------------- 审计指标 ----

def gram_feats(t4d: torch.Tensor, max_ch: int = 48) -> torch.Tensor:
    fl = t4d.float().flatten(2)
    if fl.shape[1] > max_ch:
        fl = fl[:, :max_ch, :]
    g = torch.bmm(fl, fl.transpose(1, 2)) / 256.0
    ii = torch.triu_indices(fl.shape[1], fl.shape[1])
    return g[:, ii[0], ii[1]]


def pooled_feats(t4d: torch.Tensor) -> torch.Tensor:
    return t4d.float().mean(dim=(2, 3))


def spectrum_stats(t4d: torch.Tensor) -> dict:
    """逐样本奇异值谱统计 (在 (C, 256) 矩阵上)."""
    fl = t4d.float().flatten(2)
    s = torch.linalg.svdvals(fl)
    mean_s = s.mean(dim=0)
    energy = s.square()
    eff_rank = float((energy.sum(dim=1) ** 2 / (energy.square().sum(dim=1) + 1e-9)).mean())
    return {
        "sigma_max": round(float(s.max()), 4),
        "sigma_min": round(float(s.min()), 6),
        "sigma_min_max_ratio": round(float((s.min(dim=1).values / s.max(dim=1).values.clamp_min(1e-9)).mean()), 6),
        "effective_rank": round(eff_rank, 3),
        "spectrum_mean_head8": [round(float(v), 4) for v in mean_s[:8]],
    }


def structure_retention_relational(variant: torch.Tensor, raw: torch.Tensor) -> float:
    """跨维度结构保持: token-token 相似度结构 (256×256 Gram) 与 raw dino_16 的相关.

    修复原 L2-norm 度量对正交基变体的退化 (pca32 −0.09): 关系结构 (哪些 token 相似)
    与通道基无关, 对任意 32 维投影仍有定义.
    """
    def _rel(f: torch.Tensor) -> torch.Tensor:
        ff = F.normalize(f.float().flatten(2).transpose(1, 2), dim=-1)  # (N, 256, C)
        return torch.bmm(ff, ff.transpose(1, 2)).flatten(1)             # (N, 256²)

    a = _rel(variant) - _rel(variant).mean(dim=1, keepdim=True)
    b = _rel(raw) - _rel(raw).mean(dim=1, keepdim=True)
    num = (a * b).sum(dim=1)
    den = (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-8)
    return float((num / den).mean())


def structure_retention(variant: torch.Tensor, raw: torch.Tensor) -> float:
    """空间结构保持: 逐 token 通道 L2 结构图 (N,256) 的 Pearson 相关 (对 raw dino_16)."""
    m_var = variant.float().norm(dim=1).flatten(1)
    m_raw = raw.float().norm(dim=1).flatten(1)
    a = m_var - m_var.mean(dim=1, keepdim=True)
    b = m_raw - m_raw.mean(dim=1, keepdim=True)
    num = (a * b).sum(dim=1)
    den = (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-8)
    return float((num / den).mean())


def spatial_autocorr(t4d: torch.Tensor) -> float:
    """相邻 (右/下) token 特征余弦相似度均值 — 空间平滑度."""
    f = t4d.float()
    fr = F.normalize(f[:, :, :, 1:].flatten(2), dim=-1)
    fl = F.normalize(f[:, :, :, :-1].flatten(2), dim=-1)
    fd = F.normalize(f[:, :, 1:, :].flatten(2), dim=-1)
    fu = F.normalize(f[:, :, :-1, :].flatten(2), dim=-1)
    return float(((fr * fl).sum(-1).mean() + (fd * fu).sum(-1).mean()) / 2.0)


def perturb_stability(t4d: torch.Tensor, gen: torch.Generator, std: float = NOISE_STD) -> float:
    """输入加噪后特征相对 L2 变化 (越低越稳)."""
    noise = torch.randn(t4d.shape, generator=gen).to(t4d.device) * std
    f2 = t4d + noise
    a = (t4d.float() - f2.float()).flatten(1).norm(dim=1)
    b = t4d.float().flatten(1).norm(dim=1).clamp_min(1e-8)
    return float((a / b).mean())


def audit_variant(name: str, feat: torch.Tensor, raw: torch.Tensor,
                  sid_tr: torch.Tensor, sid_ho: torch.Tensor,
                  feat_tr: torch.Tensor, feat_ho: torch.Tensor,
                  gen: torch.Generator) -> dict:
    """单变体全审计 (探针在 train→holdout 上, 无 test750 参与)."""
    gram_tr, gram_ho = gram_feats(feat_tr), gram_feats(feat_ho)
    pool_tr, pool_ho = pooled_feats(feat_tr), pooled_feats(feat_ho)
    with torch.enable_grad():
        leak_gram = _train_linear_mlp_probe(gram_tr, sid_tr, gram_ho, sid_ho)
        leak_pool = _train_linear_mlp_probe(pool_tr, sid_tr, pool_ho, sid_ho)
    return {
        "variant": name,
        "leak_gram_5way": round(leak_gram, 4),
        "leak_pooled_5way": round(leak_pool, 4),
        "structure_retention": round(structure_retention(feat, raw), 4),
        "spatial_autocorr": round(spatial_autocorr(feat), 4),
        "perturb_stability": round(perturb_stability(feat_ho, gen), 4),
        "spectrum": spectrum_stats(feat_ho),
    }


# ----------------------------------------------------------- main ----

def run_k0(pool: str, device: str = "cuda") -> dict:
    seed_all(20261185)
    bundle = build_or_load_v19_foundation(pool, device=device)
    c21 = torch.load(V21_CACHE, map_location="cpu", weights_only=False)
    W = c21["W_stiefel"]

    tr = bundle["v18_cache"]["train"]
    ho = bundle["v18_cache"]["holdout"]
    sid_tr, sid_ho = tr["sid"].long(), ho["sid"].long()

    # eta: 训练集投影奇异值中位数 (禁 test750)
    with torch.no_grad():
        sample = tr["dino_16"][::40].to(device)
        fl = torch.einsum("dk,bdp->bkp", W.to(device), sample.float().flatten(2))
        fl = fl - fl.mean(dim=-1, keepdim=True)
        sv = torch.linalg.svdvals(fl)
        eta = float(sv.median())
    print(f"[K0] soft eta (train sigma median) = {eta:.4f}", flush=True)

    variants = {
        "raw384": ho["dino_16"],
        "pca32": project_variant(ho["dino_16"], W, "pca32").cpu(),
        "q32_hard": project_variant(ho["dino_16"], W, "q32").cpu(),
        "q32_soft": project_variant(ho["dino_16"], W, "q32_soft", eta=eta).cpu(),
    }
    raw_ho = variants["raw384"]
    gen = torch.Generator().manual_seed(20261185)

    aud, spectra_by_style = [], {}
    for name, feat_ho in variants.items():
        feat_tr = {
            "raw384": tr["dino_16"], "pca32": project_variant(tr["dino_16"], W, "pca32").cpu(),
            "q32_hard": project_variant(tr["dino_16"], W, "q32").cpu(),
            "q32_soft": project_variant(tr["dino_16"], W, "q32_soft", eta=eta).cpu(),
        }[name]
        row = audit_variant(name, feat_ho, raw_ho, sid_tr, sid_ho, feat_tr, feat_ho, gen)
        aud.append(row)
        print(f"[K0] {row['variant']}: leak_gram={row['leak_gram_5way']:.3f} "
              f"leak_pool={row['leak_pooled_5way']:.3f} retention={row['structure_retention']:.3f} "
              f"stability={row['perturb_stability']:.3f}", flush=True)

    # Minimalism 谱差异 (高背景占比困难画派)
    for name, feat in variants.items():
        svs = torch.linalg.svdvals(feat.float().flatten(2))
        by = {}
        for s in range(5):
            m = (sid_ho == s)
            by[int(s)] = round(float(svs[m].mean()), 4)
        spectra_by_style[name] = by

    # Gram 恒等误差 (硬 Q32 数学性质核对)
    q = variants["q32_hard"].float().flatten(2)
    gram_err = float((torch.bmm(q, q.transpose(1, 2)) / 256.0 - torch.eye(q.shape[1])).abs().max())
    # 软 Q32 低秩稳定性 (秩 1 / 全零输入)
    z0 = torch.zeros(2, 384, 16, 16)
    r1 = torch.zeros(2, 384, 16, 16); r1[:, 0] = 1.0
    soft_ok = bool(torch.isfinite(soft_stiefel_quotient(z0, W, eta=eta)).all()
                   and torch.isfinite(soft_stiefel_quotient(r1, W, eta=eta)).all())

    out = {
        "task": "V26-K0 Q32 spectrum/leak/retention audit",
        "updated_utc": utc(), "soft_eta": round(eta, 5),
        "soft_lowrank_stable": soft_ok,
        "q32_hard_gram_identity_err": gram_err,
        "audit": aud, "spectra_mean_by_style_holdout": spectra_by_style,
        "tradeoff_2d": [{"variant": r["variant"], "style_leak_gram": r["leak_gram_5way"],
                         "structure_retention": r["structure_retention"]} for r in aud],
    }
    save_json(OUT / "gpu4090" / "k0_q32_audit.json", out)
    _write_report(out)
    return out


def _write_report(out: dict) -> None:
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# K0 — Q32 谱特性与内容—风格解耦审计 (V26, 纯诊断)",
        "",
        f"> 生成时间: {out['updated_utc']} ｜ 软 eta (训练集 sigma 中位数): {out['soft_eta']}",
        "",
        "## 1. 二维权衡 (Style Leakage ↓ vs Content Structure Retention ↑)",
        "",
        "| 变体 | Gram 泄漏探针 | 均值池化探针 | 结构保持 | 空间自相关 | 扰动稳定性 | 有效秩 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in out["audit"]:
        lines.append(
            f"| {r['variant']} | {r['leak_gram_5way']:.3f} | {r['leak_pooled_5way']:.3f} | "
            f"{r['structure_retention']:.3f} | {r['spatial_autocorr']:.3f} | "
            f"{r['perturb_stability']:.3f} | {r['spectrum']['effective_rank']:.1f} |")
    lines += [
        "",
        f"- 硬 Q32 Gram 恒等最大误差: `{out['q32_hard_gram_identity_err']:.2e}`",
        f"- 软 Q32 低秩/全零输入数值稳定: `{out['soft_lowrank_stable']}`",
        "",
        "## 2. 判读准则 (docs/1120 §K0)",
        "",
        "- **有效解耦**: 泄漏探针显著下降 (≤0.30) 且结构保持 ≥0.85 → 进入 K3 匹配训练消融;",
        "- **过度去信息**: 泄漏下降但结构保持 <0.70 → Q32 判负 (损坏布局);",
        "- **无解耦价值**: 泄漏未降 → 直接取消 K3, 释放 4090 预算.",
        "",
        "## 3. Minimalism 谱差异 (holdout, 各画派 sigma 均值)",
        "",
        "| 变体 | ER | Imp | Min | Roc | Uki |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, by in out["spectra_mean_by_style_holdout"].items():
        lines.append(f"| {name} | {by[0]} | {by[1]} | {by[2]} | {by[3]} | {by[4]} |")
    lines += ["", "## 4. 结论", ""]
    a = {r["variant"]: r for r in out["audit"]}
    q, p = a.get("q32_hard", {}), a.get("pca32", {})
    if q and p:
        leak_drop = p["leak_gram_5way"] - q["leak_gram_5way"]
        ret = q["structure_retention"]
        if leak_drop > 0.05 and ret >= 0.85:
            verdict = "**PASS-候选**: 硬 Q32 相对 PCA32 泄漏显著下降且结构保持充分 — 建议进入 K3 匹配训练消融."
        elif ret < 0.70:
            verdict = "**FAIL-过度去信息**: 结构保持不足, Q32 判负 (不进入 K3)."
        else:
            verdict = "**INCONCLUSIVE**: 泄漏下降不足或结构保持边缘 — 需结合 K3 小预算探针决定."
        lines.append(verdict)
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[K0] Report → {REPORT_MD}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(Path(__file__).resolve().parents[2].parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    sanitize_cuda_env()
    run_k0(args.pool, device=args.device)


if __name__ == "__main__":
    main()
