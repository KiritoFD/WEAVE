"""路线 B 的训练诊断：曲线记录、AdaIN-off 风格增益、小样本过拟合探针。

三件工具（都只依赖 CPU 合成张量，秒级可跑）：

  * `CurveRecorder`   —— 按 2k / 10k / 30k 里程碑给出"训练集拟合"与"未见内容
    泛化"的完整曲线（JSONL 全量 + Markdown 摘要）。
  * `StyleGainDiagnostics` —— 关掉推理期 AdaIN 后仍能测出风格提升，证明风格
    来自网络而不是推理后处理。
  * `OverfitProbe`    —— 32–128 张小样本能否把 loss 拟合到初始值的一半以下；
    拟合不动就说明架构/目标有问题，应先回报而不是继续堆复杂度。

这些工具不依赖 model / flow / trainer，也不做任何教师前向。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import torch

from b1_style_operator import StyleTargetScreener
from b_pseudo_targets import BPseudoTargetProvider


DEFAULT_MILESTONES = (2000, 10000, 30000)


# ---------------------------------------------------------------------------
# 训练曲线
# ---------------------------------------------------------------------------

class CurveRecorder:
    """按里程碑汇总训练曲线。

    用法：
        rec = CurveRecorder()
        rec.record(step, "train", {"loss": float(logs["loss"])})
        rec.record(step, "holdout", provider.evaluate_z_star(...))
        rec.write(exp_dir)

    `summary()` 对每个里程碑取"步数最接近"的一条记录，并额外给出
    `train_fit`（train 的 loss）与 `holdout`（留出集指标）两个视图。
    """

    def __init__(
        self,
        milestones: Sequence[int] = DEFAULT_MILESTONES,
        *,
        splits: Sequence[str] = ("train", "holdout"),
    ) -> None:
        self.milestones = tuple(sorted(int(m) for m in milestones))
        self.splits = tuple(splits)
        self.records: List[Dict] = []

    # -- 写入 -------------------------------------------------------------
    def record(self, step: int, split: str, metrics: Mapping[str, float]) -> None:
        if str(split) not in self.splits:
            raise ValueError(f"split={split!r} 不在 {self.splits} 中")
        clean: Dict[str, float] = {}
        for k, v in metrics.items():
            if torch.is_tensor(v):
                v = float(v.detach())
            try:
                clean[str(k)] = float(v)
            except (TypeError, ValueError):
                continue
        self.records.append({"step": int(step), "split": str(split), "metrics": clean})

    def should_record(self, step: int) -> bool:
        """用于训练循环：每个里程碑命中时返回 True。"""
        return int(step) in self.milestones

    # -- 汇总 -------------------------------------------------------------
    def _nearest(self, milestone: int, split: str) -> Dict | None:
        candidates = [r for r in self.records if r["split"] == split]
        if not candidates:
            return None
        best = min(candidates, key=lambda r: (abs(r["step"] - milestone), r["step"]))
        return best

    def summary(self) -> Dict:
        by_milestone: Dict[int, Dict] = {}
        for m in self.milestones:
            row: Dict = {}
            for split in self.splits:
                hit = self._nearest(m, split)
                if hit is not None:
                    row[split] = {"step_actual": hit["step"], **hit["metrics"]}
            by_milestone[m] = row

        def _last_loss(split: str) -> float | None:
            vals = [r["metrics"].get("loss") for r in self.records if r["split"] == split]
            vals = [v for v in vals if v is not None]
            return float(vals[-1]) if vals else None

        return {
            "milestones": by_milestone,
            "train_fit": {"final_loss": _last_loss("train")},
            "holdout": {"final_loss": _last_loss("holdout")},
            "num_records": len(self.records),
        }

    def write(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "b_curve.jsonl").open("w", encoding="utf-8") as fh:
            for rec in self.records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

        summ = self.summary()
        lines = ["# Route B curve summary", ""]
        for m, row in summ["milestones"].items():
            lines.append(f"## milestone {m}")
            lines.append("")
            if not row:
                lines.append("_no records_")
                lines.append("")
                continue
            keys: List[str] = []
            for split_row in row.values():
                for k in split_row:
                    if k not in keys:
                        keys.append(k)
            lines.append("| split | " + " | ".join(keys) + " |")
            lines.append("|" + "---|" * (len(keys) + 1))
            for split, split_row in row.items():
                cells = []
                for k in keys:
                    v = split_row.get(k)
                    cells.append("" if v is None else f"{v:.5f}")
                lines.append(f"| {split} | " + " | ".join(cells) + " |")
            lines.append("")
        lines.append(f"- train final loss: {summ['train_fit']['final_loss']}")
        lines.append(f"- holdout final loss: {summ['holdout']['final_loss']}")
        (out / "b_curve_summary.md").write_text("\n".join(lines), encoding="utf-8")
        return out


# ---------------------------------------------------------------------------
# AdaIN-off 风格增益诊断
# ---------------------------------------------------------------------------

def channel_adain(z: torch.Tensor, ref: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    """逐通道 mean/std 对齐（模拟推理期 AdaIN），strength=0 时原样返回。"""
    s = float(strength)
    if s <= 0.0:
        return z
    zf = z.float()
    rf = ref.float()
    if rf.shape[0] == 1 and zf.shape[0] > 1:
        rf = rf.expand(zf.shape[0], -1, -1, -1)
    mu_z = zf.mean(dim=(2, 3), keepdim=True)
    sd_z = zf.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    mu_r = rf.mean(dim=(2, 3), keepdim=True)
    sd_r = rf.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    matched = (zf - mu_z) / sd_z * sd_r + mu_r
    return ((1.0 - s) * zf + s * matched).to(dtype=z.dtype)


class StyleGainDiagnostics:
    """测量网络自身（关闭推理期 AdaIN）能带来多少风格增益。"""

    def __init__(
        self,
        operator,
        *,
        screener: StyleTargetScreener | None = None,
        adain_strength: float = 0.3,
    ) -> None:
        self.operator = operator
        cfg = getattr(operator, "cfg", None)
        self.screener = screener or StyleTargetScreener(
            patch_size=getattr(cfg, "patch_size", 4),
            patch_stride=getattr(cfg, "patch_stride", 4),
        )
        self.adain_strength = float(adain_strength)

    @torch.no_grad()
    def evaluate(
        self,
        content_latent: torch.Tensor,
        reference_latent: torch.Tensor,
        *,
        use_adain: bool = False,
    ) -> Dict:
        z_star = self.operator(content_latent, reference_latent)["z_star"]
        if use_adain:
            z_star = channel_adain(z_star, reference_latent, self.adain_strength)
        report = self.screener.score(z_star, content_latent, reference_latent)
        report["used_adain"] = bool(use_adain)
        return report

    def compare(
        self,
        content_latent: torch.Tensor,
        reference_latent: torch.Tensor,
    ) -> Dict:
        off = self.evaluate(content_latent, reference_latent, use_adain=False)
        on = self.evaluate(content_latent, reference_latent, use_adain=True)
        contrib = float(on["style_gain"] - off["style_gain"])
        return {
            "adain_off": off,
            "adain_on": on,
            "adain_contribution": contrib,
            # 关掉 AdaIN 仍有风格增益 -> 风格来自网络。
            "network_style_ok": bool(off["style_gain"] >= self.screener.min_style_gain),
            "passed": bool(
                off["passed"]
                and off["style_gain"] >= self.screener.min_style_gain
            ),
        }


# ---------------------------------------------------------------------------
# 小样本过拟合探针
# ---------------------------------------------------------------------------

@dataclass
class OverfitConfig:
    steps: int = 200
    lr: float = 3e-3
    n_samples: int = 32
    grad_clip: float = 1.0
    unpaired_weight: float = 0.5
    target_ratio: float = 0.5
    seed: int = 0
    extra: Dict = field(default_factory=dict)


class OverfitProbe:
    """32–128 张小样本能否过拟合：final <= initial * target_ratio 才算目标可优化。"""

    def __init__(
        self,
        provider: BPseudoTargetProvider,
        cfg: OverfitConfig | None = None,
    ) -> None:
        self.provider = provider
        self.cfg = cfg or OverfitConfig()

    def _unpaired_from(self, samples: Sequence[Dict], idx: int) -> Dict:
        """用同批样本的 content + 另一张参考图构造无配对项。"""
        cur = samples[idx % len(samples)]
        donor = samples[(idx + 1) % len(samples)]
        return {
            "content_latent": cur["content_latent"],
            "reference_latent": donor["reference_latent"],
        }

    def run(self, samples: Iterable[Dict]) -> Dict:
        items = list(samples)[: int(self.cfg.n_samples)]
        if len(items) < 2:
            raise ValueError("OverfitProbe 至少需要 2 个配对样本")
        torch.manual_seed(self.cfg.seed)
        params = [p for p in self.provider.operator.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("OverfitProbe 需要可训练参数（阶段 1 尚未冻结）")
        opt = torch.optim.Adam(params, lr=self.cfg.lr)
        self.provider.operator.train()

        history: List[float] = []
        initial = None
        for step in range(1, int(self.cfg.steps) + 1):
            paired = items[(step - 1) % len(items)]
            unpaired = self._unpaired_from(items, step - 1)
            logs = self.provider.stage1_step(
                paired,
                unpaired,
                optimizer=opt,
                grad_clip=self.cfg.grad_clip,
                unpaired_weight=self.cfg.unpaired_weight,
            )
            loss = float(logs["loss"].detach())
            if initial is None:
                initial = loss
            history.append(loss)

        final = history[-1]
        ratio = final / max(initial, 1e-8)
        return {
            "n_samples": len(items),
            "steps": int(self.cfg.steps),
            "initial_loss": float(initial),
            "final_loss": float(final),
            "ratio": float(ratio),
            "target_ratio": float(self.cfg.target_ratio),
            "overfit_ok": bool(ratio <= self.cfg.target_ratio),
            "history": history,
        }


# ---------------------------------------------------------------------------
# 留出集指标
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_holdout(
    provider: BPseudoTargetProvider,
    samples: Iterable[Dict],
) -> Dict:
    """对留出内容跑 style_gain / content_keep / reference_copy 的均值。"""
    gains: List[float] = []
    keeps: List[float] = []
    copies: List[float] = []
    passed = 0
    total = 0
    for item in samples:
        rep = provider.evaluate_z_star(
            item["content_latent"], item["reference_latent"]
        )
        gains.append(float(rep["style_gain"]))
        keeps.append(float(rep["content_keep"]))
        copies.append(float(rep["reference_copy_corr"]))
        passed += int(bool(rep["passed"]))
        total += 1
    if total == 0:
        raise ValueError("evaluate_holdout 需要至少一个样本")
    return {
        "style_gain": sum(gains) / total,
        "content_keep": sum(keeps) / total,
        "reference_copy_corr": sum(copies) / total,
        "pass_rate": passed / total,
        "num_samples": total,
    }


__all__ = [
    "DEFAULT_MILESTONES",
    "CurveRecorder",
    "channel_adain",
    "StyleGainDiagnostics",
    "OverfitConfig",
    "OverfitProbe",
    "evaluate_holdout",
]
