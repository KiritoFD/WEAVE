"""路线 B 的三阶段伪目标供应器 + 离线特征 cache 接口。

阶段划分（与任务描述一致）：

  阶段 1：只训 T_phi（B1 的小波风格化网络）
      - 数据：B2 的严格对齐配对 (x', r_s, y) + 真实内容图上的无配对样本；
      - 监督：SWD / patch 统计（独立于 FM），然后由 StyleTargetScreener 自评；
      - 不达标（style_gain / content_keep / reference_copy 任一不过）不能进阶段 2。

  阶段 2：冻结 T_phi，把 z1 = stopgrad(T_phi(z_c, z_s)) 作为 FM 端点。
      - 本模块的 build_fm_endpoint 强制要求先 freeze_operator() 且 screen 已通过。

  阶段 3：补参考图条件，跑两条结构测试：
      ① 固定内容与 style_id，换参考画 -> 输出有可测风格变化；
      ② 平移/打乱参考图空间位置 -> 输出不跟着搬物体轮廓。

关于教师特征：真实 DINO/VGG 特征只能通过 OfflineStyleFeatureCache 读取；
本模块不在任何训练 step 内构造教师或做教师前向。cache 未命中时返回 None，
由调用方决定回退（不回退到在线教师）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch

from b1_style_operator import (
    B1StyleDistributionLoss,
    B1StyleOperator,
    StyleTargetScreener,
    reference_permutation_probe,
    reference_swap_probe,
)
from pseudo_targets import PseudoTargetCacheWriter


STAGE_DESCRIPTIONS = {
    1: "train T_phi only with B2 aligned pairs + unpaired content; gate on z* screening",
    2: "freeze T_phi; z1 = stopgrad(T_phi(z_c, z_s)) supplies FM endpoints",
    3: "add reference conditioning; run swap / permutation structural tests",
}

STYLE_FEATURE_CACHE_VERSION = 1


# ---------------------------------------------------------------------------
# 离线风格特征 cache（DINO / VGG 特征只从这里读）
# ---------------------------------------------------------------------------

class OfflineStyleFeatureCacheWriter:
    """把离线提取的风格特征写成 shard。

    每个 entry 至少包含其一：
      * global_op:   (style_dim,) 预投影后的全局风格操作向量（阶段 2 可直接用）
      * patch_tokens: (N,D) 预提取的参考 patch 特征（可选，供离线分析）
    """

    def __init__(self, out_dir: str | Path, *, shard_size: int = 2048) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = int(shard_size)
        self._entries: List[Dict] = []
        self._shard_idx = 0
        self._total = 0

    def add(
        self,
        key: str,
        *,
        global_op: torch.Tensor | None = None,
        patch_tokens: torch.Tensor | None = None,
        style_id: int | None = None,
    ) -> None:
        entry = {
            "key": str(key),
            "style_id": int(style_id) if style_id is not None else None,
            "global_op": global_op.detach().float().cpu() if global_op is not None else None,
            "patch_tokens": patch_tokens.detach().float().cpu() if patch_tokens is not None else None,
        }
        self._entries.append(entry)
        if len(self._entries) >= self.shard_size:
            self._flush(self.shard_size)

    def _flush(self, n: int) -> None:
        payload = {
            "version": STYLE_FEATURE_CACHE_VERSION,
            "entries": self._entries[:n],
        }
        torch.save(payload, self.out_dir / f"style_shard_{self._shard_idx:06d}.pt")
        self._shard_idx += 1
        self._total += n
        del self._entries[:n]

    def close(self, *, meta: Dict | None = None) -> Path:
        if self._entries:
            self._flush(len(self._entries))
        manifest = {
            "version": STYLE_FEATURE_CACHE_VERSION,
            "count": self._total,
            "shards": self._shard_idx,
            "meta": dict(meta or {}),
        }
        (self.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return self.out_dir


class OfflineStyleFeatureCache:
    """只读 cache。get(key) 未命中返回 None；绝不触发教师前向。"""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        manifest_path = self.cache_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"OfflineStyleFeatureCache: no manifest.json in {self.cache_dir}")
        manifest = json.loads(manifest_path.read_text())
        if int(manifest.get("version", 0)) != STYLE_FEATURE_CACHE_VERSION:
            raise ValueError(f"Unsupported style feature cache version {manifest.get('version')}")
        self._index: Dict[str, Tuple[Path, int]] = {}
        self._shards: Dict[Path, Dict] = {}
        for shard in sorted(self.cache_dir.glob("style_shard_*.pt")):
            payload = torch.load(shard, map_location="cpu", weights_only=False)
            for i, entry in enumerate(payload["entries"]):
                self._index[str(entry["key"])] = (shard, i)
        self.count = len(self._index)

    def _load(self, path: Path) -> Dict:
        if path not in self._shards:
            self._shards[path] = torch.load(path, map_location="cpu", weights_only=False)
        return self._shards[path]

    def get(self, key: str) -> Dict | None:
        hit = self._index.get(str(key))
        if hit is None:
            return None
        shard, i = hit
        entry = self._load(shard)["entries"][i]
        return {
            "key": entry["key"],
            "style_id": entry["style_id"],
            "global_op": entry["global_op"],
            "patch_tokens": entry["patch_tokens"],
        }

    def get_global_op(self, key: str) -> torch.Tensor | None:
        entry = self.get(key)
        if entry is None:
            return None
        return entry["global_op"]


# ---------------------------------------------------------------------------
# 供应器
# ---------------------------------------------------------------------------

@dataclass
class ProviderConfig:
    unpaired_weight: float = 0.5
    require_unpaired: bool = True
    # 阶段 3 结构测试阈值（相对输出幅度）
    min_swap_delta_rel: float = 1.0e-4
    max_permutation_delta: float = 1.0e-4
    extra: Dict = field(default_factory=dict)


class BPseudoTargetProvider:
    """B1 算子 + B2 配对 + 三阶段门禁的统一入口。"""

    def __init__(
        self,
        operator: B1StyleOperator,
        *,
        style_loss: B1StyleDistributionLoss | None = None,
        screener: StyleTargetScreener | None = None,
        style_feature_cache: OfflineStyleFeatureCache | None = None,
        stage: int = 1,
        cfg: ProviderConfig | None = None,
    ) -> None:
        self.operator = operator
        self.style_loss = style_loss or B1StyleDistributionLoss(
            patch_size=operator.cfg.patch_size,
            patch_stride=operator.cfg.patch_stride,
        )
        self.screener = screener or StyleTargetScreener(
            patch_size=operator.cfg.patch_size,
            patch_stride=operator.cfg.patch_stride,
        )
        self.style_feature_cache = style_feature_cache
        self.stage = int(stage)
        self.cfg = cfg or ProviderConfig()
        self.screen_passed = False
        self.screen_report: Dict | None = None

    # -- 阶段 1 -----------------------------------------------------------
    def stage1_loss(
        self,
        paired: Dict[str, torch.Tensor],
        unpaired: Dict[str, torch.Tensor] | None = None,
        *,
        unpaired_weight: float | None = None,
        require_unpaired: bool | None = None,
    ) -> Dict[str, torch.Tensor]:
        """阶段 1 的独立风格监督。

        Args:
            paired: 需含 content_latent(x'), reference_latent(r_s), target_latent(y)。
            unpaired: 需含 content_latent(真实内容图), reference_latent(风格参考)。
                这是必须项：只用配对会退化成 "从 x' 复原 y" 的重建任务。
        """
        weight = self.cfg.unpaired_weight if unpaired_weight is None else float(unpaired_weight)
        need_unpaired = self.cfg.require_unpaired if require_unpaired is None else bool(require_unpaired)
        if need_unpaired and unpaired is None:
            raise ValueError("阶段 1 必须混入真实内容图上的无配对训练（unpaired 不能为空）")

        xp = paired["content_latent"]
        rs = paired["reference_latent"]
        y = paired["target_latent"]
        out_paired = self.operator(xp, rs)
        logs_paired = self.style_loss(
            out_paired["z_star"], xp, y, ll_delta=out_paired["ll_delta"]
        )

        logs: Dict[str, torch.Tensor] = {
            f"paired/{k}": v for k, v in logs_paired.items()
        }
        total = logs_paired["loss"]
        if unpaired is not None:
            c = unpaired["content_latent"]
            r = unpaired["reference_latent"]
            out_unpaired = self.operator(c, r)
            logs_unpaired = self.style_loss(
                out_unpaired["z_star"], c, r, ll_delta=out_unpaired["ll_delta"]
            )
            for k, v in logs_unpaired.items():
                logs[f"unpaired/{k}"] = v
            total = total + weight * logs_unpaired["loss"]
        logs["loss"] = total
        return logs

    def stage1_step(
        self,
        paired: Dict[str, torch.Tensor],
        unpaired: Dict[str, torch.Tensor] | None = None,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        grad_clip: float = 0.0,
        unpaired_weight: float | None = None,
        require_unpaired: bool | None = None,
    ) -> Dict[str, torch.Tensor]:
        logs = self.stage1_loss(
            paired,
            unpaired,
            unpaired_weight=unpaired_weight,
            require_unpaired=require_unpaired,
        )
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            logs["loss"].backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.operator.parameters(), grad_clip)
            optimizer.step()
        return logs

    # -- 自评闸门 ---------------------------------------------------------
    @torch.no_grad()
    def evaluate_z_star(
        self,
        content_latent: torch.Tensor,
        reference_latent: torch.Tensor,
        *,
        global_style_op: torch.Tensor | None = None,
    ) -> Dict:
        out = self.operator(content_latent, reference_latent, global_style_op=global_style_op)
        report = self.screener.score(out["z_star"], content_latent, reference_latent)
        report["gate"] = float(out["gate"])
        return report

    def mark_screened(self, report: Dict | None = None) -> Dict:
        """登记筛选结果。只有 report['passed'] 为真才打开阶段 2 端点闸门。"""
        if report is None:
            if self.screen_report is None:
                raise ValueError("mark_screened 需要 report，或先调用 evaluate_z_star")
            report = self.screen_report
        self.screen_report = dict(report)
        self.screen_passed = bool(report.get("passed", False))
        return self.screen_report

    def evaluate_and_mark(
        self,
        content_latent: torch.Tensor,
        reference_latent: torch.Tensor,
        *,
        global_style_op: torch.Tensor | None = None,
    ) -> Dict:
        return self.mark_screened(
            self.evaluate_z_star(content_latent, reference_latent, global_style_op=global_style_op)
        )

    # -- 阶段 2 -----------------------------------------------------------
    def freeze_operator(self) -> None:
        for p in self.operator.parameters():
            p.requires_grad_(False)
        self.operator.eval()
        self.stage = 2

    def _lookup_global_op(
        self,
        style_key: str | Sequence[str] | None,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if style_key is None or self.style_feature_cache is None:
            return None
        if isinstance(style_key, str):
            op = self.style_feature_cache.get_global_op(style_key)
            if op is None:
                return None
            return op.unsqueeze(0).to(device=device, dtype=dtype)
        ops = []
        for key in style_key:
            op = self.style_feature_cache.get_global_op(key)
            if op is None:
                return None
            ops.append(op)
        return torch.stack(ops).to(device=device, dtype=dtype)

    @torch.no_grad()
    def build_fm_endpoint(
        self,
        content_latent: torch.Tensor,
        reference_latent: torch.Tensor | None = None,
        *,
        style_key: str | Sequence[str] | None = None,
        allow_unscreened: bool = False,
    ) -> Dict[str, torch.Tensor] | None:
        """阶段 2：z1 = stopgrad(T_phi(z_c, z_s))。

        未通过阶段 1 筛选时直接拒绝；没有参考图也没有离线 style_key 时返回 None，
        由 trainer 回退到原 target_hf，而不是在线跑教师。
        """
        if self.stage < 2 and not allow_unscreened:
            raise RuntimeError("build_fm_endpoint 需要先 freeze_operator()（阶段 2）")
        if not self.screen_passed and not allow_unscreened:
            raise RuntimeError("T_phi 尚未通过阶段 1 筛选，不允许提供 FM 端点")
        if reference_latent is None and style_key is None:
            return None

        B = content_latent.shape[0]
        global_op = self._lookup_global_op(
            style_key, B, content_latent.device, content_latent.dtype
        )
        if reference_latent is None and global_op is None:
            return None

        out = self.operator(content_latent, reference_latent, global_style_op=global_op)
        z1 = out["z_star"].detach()
        return {
            "z1": z1,
            "velocity": z1 - content_latent.detach(),
            "style_ops": {
                k: v.detach() for k, v in out["region_params"].items()
            },
            "global_op": out["global_op"].detach(),
            "screen": self.screen_report,
        }

    # -- 阶段 3 -----------------------------------------------------------
    def stage3_checks(
        self,
        content_latent: torch.Tensor,
        reference_a: torch.Tensor,
        reference_b: torch.Tensor,
        *,
        permutation: torch.Tensor | None = None,
    ) -> Dict:
        """跑两条结构测试并返回 pass/fail。"""
        swap = reference_swap_probe(
            self.operator, content_latent, reference_a, reference_b
        )
        perm = reference_permutation_probe(
            self.operator, content_latent, reference_b, permutation=permutation
        )
        swap_ok = swap["output_delta_rel"] >= self.cfg.min_swap_delta_rel
        perm_ok = perm["output_delta_max"] <= self.cfg.max_permutation_delta
        return {
            "swap": swap,
            "permutation": perm,
            "swap_passed": bool(swap_ok),
            "permutation_passed": bool(perm_ok),
            "passed": bool(swap_ok and perm_ok),
        }

    # -- 离线端点导出 ------------------------------------------------------
    @torch.no_grad()
    def export_endpoints(
        self,
        samples: Iterable[Dict],
        out_dir: str | Path,
        *,
        shard_size: int = 4096,
        only_passed: bool = True,
    ) -> Path:
        """把 z1 写成 pseudo-target cache，供后续 FM 直接读取。

        samples 每项需含：
          content_key: str | list[str]
          style_id:    int | list[int]
          content_latent: (B,C,H,W)
          reference_latent: (B,C,H,W)（可选；若用 style_key 则从 cache 读全局操作）
        """
        writer = PseudoTargetCacheWriter(out_dir, shard_size=shard_size)
        written = 0
        for item in samples:
            c = item["content_latent"]
            r = item.get("reference_latent")
            style_key = item.get("style_key")
            keys = item["content_key"]
            style_ids = item["style_id"]
            if isinstance(keys, str):
                keys = [keys]
            if isinstance(style_ids, int):
                style_ids = [style_ids]

            B = c.shape[0]
            global_op = self._lookup_global_op(style_key, B, c.device, c.dtype)
            if r is None and global_op is None:
                continue
            out = self.operator(c, r, global_style_op=global_op)
            z1 = out["z_star"]
            keep_idx: List[int] = []
            scores: List[float] = []
            if r is None:
                # 没有参考图时无法做"是否复制参考结构"评估；只在非严格模式下落盘。
                if only_passed:
                    continue
                for i in range(B):
                    keep_idx.append(i)
                    scores.append(0.0)
            else:
                for i in range(B):
                    r_i = r[i : i + 1] if r.shape[0] > 1 else r
                    rep = self.screener.score(z1[i : i + 1], c[i : i + 1], r_i)
                    if (not only_passed) or rep["passed"]:
                        keep_idx.append(i)
                        # 分数用于 cache 质量过滤；负 gain 落盘为 0，避免默认阈值丢弃。
                        scores.append(float(max(0.0, rep["style_gain"])))
            if not keep_idx:
                continue
            idx = torch.tensor(keep_idx, device=c.device, dtype=torch.long)
            writer.add(
                z1.index_select(0, idx),
                [keys[i] for i in keep_idx],
                [style_ids[i] for i in keep_idx],
                scores,
            )
            written += len(keep_idx)
        return writer.close(meta={"route": "B", "stage": 2, "written": written})


__all__ = [
    "STAGE_DESCRIPTIONS",
    "OfflineStyleFeatureCache",
    "OfflineStyleFeatureCacheWriter",
    "ProviderConfig",
    "BPseudoTargetProvider",
]
