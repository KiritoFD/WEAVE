"""V25 机制 C 教师包 v2 (3060 可跑): 内容权重参数化 + 双指标筛选 + 参考特征缓存 —— B02 实现.

复用 exp.v24.o2_ref_teacher_pairs 的优化结构 (可微 S10 × MultiScalePatchSWD × Oklab 27D 色彩
× 局部 DINO-match, 24 步 Adam), 仅改三处 (docs/19_pull_together/02_mechanism_plan.md §C.2,
agent2.md §二 G2-2):

1. 内容权重参数化:  ``content_w ∈ {2.2, 1.2, 0.7}`` (T-conservative / T-medium / T-expressive),
   其余超参 (lr / 正则 / 步数) 三档完全一致 —— "同初始化、同优化, 仅改内容权重" 的校准协议.
2. 双指标筛选取代 o2 的 mid_cos top-k (A03 诊断: top-k 偏向保守样本, 是 O3_label_free
   "少改即安全" 的直接原因):
       keep = (mid_cos ∈ [lo, hi]) AND (style_gain ≥ gain_min)
   - mid_cos    内容稳定性 = 输出 vs 源图的 ConvNeXt s2/s3 逐 patch 余弦 (与 o2 同公式);
   - style_gain 风格增量   = pooled_style_cos(fin, ref) − pooled_style_cos(src, ref),
     即相对源图基线, 输出外观向参考移动的净幅度 (label-free, 只用训练专用 ConvNeXt 特征).
   - 阈值起点: mid_cos 带 (0.80, 0.92) 来自 02_mechanism_plan §C.2 预期带
     (T-medium 0.86–0.90 / T-expressive 0.80–0.86; 上限 0.92 用于滤除 ≥0.965 的保守臂);
     style_gain_min=0.10 为代码分析自定起点 —— **待 A03 校准**
     (docs/v25/audits/a03_teacher_quality.md 尚未产出; R2 三档诊断将给出实测分布).
3. 参考特征缓存 v2: ReferenceFeatureCache 按参考图 latent 内容 sha256 键控 (文件存
   --cache-dir, 默认 runs/v25/cache/ref_feats/), 同一参考重复编码直接命中 —— C3 需要同一批
   (idx_c, idx_r) 在三档权重下各优化一次, 参考侧 (SWD tokens + Oklab 27D) 只算一次.
   形状契约 (3060 R2 现场修复): 缓存条目必须带批维 (tokens (1,N,C) / pal (1,C)) —— o2 的
   批量 extract_tokens 输出即 (B,N,C), 单样本落盘丢批维会使 cat 把 batch 折进 spatial 维
   (12×(256,384)→(3072,384)) 并在 expand_as 处崩溃; batch_cached_ref_tokens 在读取侧归一.

3060 性能契约 (2026-11-20, R2 实测 54s/对 → 修复后估算):
  - 逐行对照 o2: 优化循环 (批量 12 对同时过可微 S10 + SWD + Adam, 24 步) 与 o2 **结构逐式
    相同**; 无逐对串行 batch=1、步内无重编码、无 synchronize/.item() 往返、dtype 与 o2 一致
    (解码 fp16, 提取 fp32). 54s/对 vs o2 的 3.6s/对 差距主体是 **4090→3060 硬件**
    (每步 = SD-VAE 级解码器 512² fwd+bwd ≈ 45-70 TFLOP@B12; 3060 fp16-TC/fp32-accum
    有效吞吐 ~3.5-5 TFLOPS → 步级硬件地板 ~10-20s ≈ 20-40s/对; o2 基线是 4090 实测).
    CPU 实测非解码组件: SWD≈0.55s / Oklab≈0.27s / DINO 匹配≈9ms / sha≈0.5ms (12 对, 均为
    GPU 上更小的相对量) → 步级成本由解码器主导.
  - 本次精确等价修复 (逐位不改轨迹): ① 缓存 v2 = fp32 精确存储 + "s"/"r" 角色前缀键,
    全命中时 **零 decode/extract** (三档共享同配对 → 第 2/3 档前置阶段 ~5-8s/批 → ~0.5-2s/批);
    ② miss 时单次批量 extract 替代 12 次 batch-1 (切片落盘再 cat = 逐位等于 o2 批量输出);
    ③ 常量 src 归一外提步循环 (o2 每步重复 normalize 同一常量).
  - 旋钮: --fast = cudnn.benchmark 固定形状自动调优 (步级 ~1.2-1.4×, 轨迹近似一致, 无 RNG);
    --batch ↑ 利用 Adam 尺度不变性 → 单对轨迹与 batch 无关 (显存允许时建议试 16);
    --steps/--n-cand 供 R2 诊断降档 (R3 生产保持 24 步全量).
  - 估算 (B12): 精确修复后 ≈ 50-52s/对; +--fast ≈ 38-45s/对; +batch16 ≈ 32-38s/对.
    R2 建议口径: --content-w all --n-cand 150 --steps 24 --fast ≈ 4.5-5.5h (三档),
    或 --n-cand 100 ≈ 3-3.7h; R3 2000 对生产 (24 步) 建议 4090 (o2 口径 ≈ 2h).

无画派 ID: 优化目标、筛选条件与损失全程不读取任何画派标签/类重心 (与 o2 铁律一致);
唯一例外是 **诊断报告** ``family_retention_report`` (A03 审计维度"不同风格的保留比例"),
它只在包落盘后对 kept 掩码做分组统计, 永不进入优化/筛选/任何损失.

swap 配对: 为 C2 参考替换一致性预留 —— 前 n_swap 个内容各优化两个不同参考 (多产出 n_swap
对教师目标), pack 额外携带 ``swap_groups`` (G,2, 指向 kept 数组内共享同内容的对索引);
双指标筛选按对独立执行, swap 组仅在两成员同时 kept 时保留 (风格语义一致性要求).

用法 (3060, 本任务只交付不启动 GPU):
  R2 教师三档诊断:  python -m exp.v25.reference_teacher_v2 --content-w all --n-cand 150 --steps 24 --fast
  R3 2000 对生成:   python -m exp.v25.reference_teacher_v2 --content-w <R2胜者> --n-cand 2400 --n-keep 2000 --fast
3060 12GB 约束: 教师优化批 bs=12 起步 (S10 可微解码 + SWD, 与 o2 同量级), 显存有余可 --batch 16;
R2 诊断规模建议 100-150 对/档 (分布标定足够, 见上方性能契约); R3 全量生产建议改排 4090 队列.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from . import OUT, ROOT, STYLES, sanitize_cuda_env, save_json, seed_all, utc
from exp.v19.fiber_duality import build_or_load_v19_foundation, extract_color_fiber_27d
from exp.v20.submode_patchswd import MultiScalePatchSWDExtractor, sliced_wasserstein_1d
from src.utils import decode_s10, load_s10_decoder

# ---- 阈值常量 (可由 CLI/参数覆盖; 起点 <待 A03 校准>) ----
MID_COS_BAND = (0.80, 0.92)   # (lo, hi) 闭区间; lo 出自 plan §C.2, hi 滤除保守臂 (0.965)
STYLE_GAIN_MIN = 0.10         # 风格增量下限, 代码分析起点 <待 A03 校准>
TEACHER_W_GRID = (2.2, 1.2, 0.7)
TEACHER_W_TAGS = {2.2: "tcons", 1.2: "tmed", 0.7: "texpr"}
DEFAULT_REF_CACHE_DIR = OUT / "cache/ref_feats"
CACHE_FMT = 2                 # 缓存条目格式: v2 = fp32 精确存储 + s/r 角色前缀键 (v1 条目惰性重编)
STYLE_GAIN_PROVENANCE = (
    "mid_cos 带 (0.80,0.92) 出自 docs/19_pull_together/02_mechanism_plan.md §C.2; "
    "style_gain_min=0.10 为代码分析自定起点 <待 A03 校准> "
    "(docs/v25/audits/a03_teacher_quality.md 未产出, R2 三档诊断后由 A03 标定)"
)


def enable_fast_conv() -> None:
    """--fast 优化开关 (默认关闭 → 与 o2 数值逐位可比).

    cudnn.benchmark 对固定输入形状 (解码 (B,4,64,64)→512² / ConvNeXt 提取 (B,3,256,²))
    自动选最快卷积算法 —— 本工作负载每步 ~24 步同形状卷积, 预计步级 1.2-1.4×
    (3060/Ampere fp16 conv 经验值; 未实测, 以 --fast 前后日志 step_s 为准).
    数值影响: 算法选择改变卷积求和顺序 → 优化轨迹近似一致 (教师优化为确定性 Adam
    下降, 无 RNG 参与; 损失/指标为平滑聚合, 分布等价).
    """
    torch.backends.cudnn.benchmark = True


def teacher_tag(content_w: float) -> str:
    """内容权重 → 产物标签 (三档用 plan 命名, 其余 w<p>格式)."""
    tag = TEACHER_W_TAGS.get(round(float(content_w), 3))
    return tag if tag is not None else "w" + str(float(content_w)).replace(".", "p")


# ---------------------------------------------------------------- 缓存 ----

def bytes_sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tensor_sha(t: torch.Tensor) -> str:
    """张量内容 sha256 (shape+dtype 参与键控; numpy tobytes 跨平台字节稳定)."""
    tt = t.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(f"{tuple(tt.shape)}|{tt.dtype}".encode("utf-8"))
    if tt.dtype == torch.bfloat16:
        h.update(tt.float().numpy().tobytes())
    else:
        h.update(tt.numpy().tobytes())
    return h.hexdigest()


class ReferenceFeatureCache:
    """按参考图内容 sha 键控的磁盘特征缓存 (get/put 原子写, 命中计数可审计).

    契约 (B02 验收 5): 键 = tensor_sha(参考内容), 同内容重复编码直接命中;
    缓存文件存 <cache_dir>/ref_<sha>.pt, 默认 runs/v25/cache/ref_feats/.
    """

    def __init__(self, cache_dir: str | Path | None = None):
        self.dir = Path(cache_dir) if cache_dir is not None else DEFAULT_REF_CACHE_DIR
        self.dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> Path:
        return self.dir / f"ref_{key}.pt"

    def get(self, key: str):
        p = self._path(key)
        if p.exists():
            self.hits += 1
            return torch.load(p, map_location="cpu", weights_only=False)
        self.misses += 1
        return None

    def put(self, key: str, value) -> None:
        tmp = self.dir / f"ref_{key}.pt.tmp"
        torch.save(value, tmp)
        os.replace(tmp, self._path(key))

    def get_or_encode(self, key: str, encode_fn):
        v = self.get(key)
        if v is not None:
            return v
        v = encode_fn()
        self.put(key, v)
        return v

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses, "dir": self.dir.as_posix()}


def batch_cached_ref_tokens(items: list[dict], device: str = "cpu") -> tuple[dict, torch.Tensor | None]:
    """缓存条目列表 → 批量参考特征 (o2 形状契约: tokens (B,N,C), pal (B,C) 或 None).

    3060 R2 现场修复 (expand([B,256,1] → [3072,384]) RuntimeError): 单样本缓存条目若为
    2D (N,C) (batch 维丢失), cat 沿 dim 0 会把 batch 折进 spatial 维 —— 12×(256,384) →
    (3072,384), 使 best_j.unsqueeze(-1).expand_as(ref_s3) 崩溃. o2 的批量 extract_tokens
    输出本就是 (B,N,C) 3D; 本归一保证 cat 前每个条目都带批维:
      - tokens: 2D (N,C) (历史条目) → unsqueeze(0); 3D (1,N,C) (新条目) 原样;
      - pal:    1D (C,) (历史条目) → unsqueeze(0); 2D (1,C) (新条目) 原样;
                None (内容侧条目, collect_cached_features role="s") → 返回 None.
    """
    assert len(items) > 0, "缓存条目为空"
    toks = {}
    for k in items[0]["tokens"]:
        parts = []
        for it in items:
            t = it["tokens"][k]
            assert t.dim() in (2, 3), f"token '{k}' 维度异常: {tuple(t.shape)} (应为 (N,C) 或 (1,N,C))"
            parts.append(t.unsqueeze(0) if t.dim() == 2 else t)
        toks[k] = torch.cat(parts, 0).to(device).float()
    if items[0].get("pal") is None:
        assert all(it.get("pal") is None for it in items), "pal 字段必须全 None 或全有值 (不可混排)"
        return toks, None
    pal_parts = []
    for it in items:
        t = it["pal"]
        assert t.dim() in (1, 2), f"pal 维度异常: {tuple(t.shape)} (应为 (C,) 或 (1,C))"
        pal_parts.append(t.unsqueeze(0) if t.dim() == 1 else t)
    pal = torch.cat(pal_parts, 0).to(device).float()
    assert pal.dim() == 2, f"pal_ref 形状契约破坏: {tuple(pal.shape)} (应为 (B,C))"
    return toks, pal


def collect_cached_features(cache: ReferenceFeatureCache, z_batch: torch.Tensor, dec, swd_ext, *,
                            role: str, need_pal: bool, device: str = "cpu") -> list[dict]:
    """预匹配特征收集: sha 缓存优先, miss 时整批 decode+extract **一次**并按样本切片落盘.

    提速契约 (R2 3060 54s/对 → 目标 ≤40s/对):
    1. 全命中时 **完全不调用 dec/swd_ext** —— R2 第二/三档与 R3 增量重跑的前置阶段
       从 ~5-8s/批 降到 ~0.5-2s/批 (缓存盘读);
    2. miss 时单次批量 extract_tokens 替代 v1 的 12 次 batch-1 调用 (内核启动开销),
       切片 t[b:b+1] 落盘后再 cat 重建 = 逐位等于 o2 的批量 extract 输出;
    3. 条目为 fp32 精确存储 (CACHE_FMT=2) → 命中值与重算逐位一致 (v1 fp16 条目自动惰性重编);
    4. role="s"/"r" 角色前缀键: 内容侧与参考侧同池可撞 sha, 分命名空间 (v1 裸 sha 键
       因此全部失效 → 惰性重编, 旧文件成无害孤儿).

    返回条目列表 (交 batch_cached_ref_tokens 批量化); dec/swd_ext 为真实或 mock 模型.
    """
    assert role in ("s", "r")
    keys = [role + tensor_sha(z_batch[b].detach().cpu()) for b in range(len(z_batch))]
    items = [cache.get(k) for k in keys]
    items = [it if (it is not None and it.get("fmt") == CACHE_FMT) else None for it in items]
    miss = [b for b, it in enumerate(items) if it is None]
    if miss:
        # 只解码缺失行 (混合命中场景零浪费); 批量 extract 一次, 切片 (1,N,C) 落盘
        z_miss = z_batch[miss]
        rgb = decode_s10(dec, z_miss, batch=len(z_miss), output_cpu=False)
        toks_full = swd_ext.extract_tokens(rgb)
        pal_full = extract_color_fiber_27d(rgb) if need_pal else None
        for j, b in enumerate(miss):
            it = {"tokens": {k: t[j:j + 1].cpu().float() for k, t in toks_full.items()},
                  "pal": pal_full[j:j + 1].cpu().float() if need_pal else None,
                  "fmt": CACHE_FMT}
            cache.put(keys[b], it)
            items[b] = it
    return items


# ---------------------------------------------------------- 纯函数指标 ----

def pooled_style_cos(tokens_a: dict, tokens_b: dict, scales=("s2", "s3")) -> torch.Tensor:
    """Label-free 风格相似度: 各尺度 token 空间池化 (B,C) → 余弦, 跨尺度平均. 返回 (B,)."""
    vals = []
    for k in scales:
        a = tokens_a[k].mean(dim=1)
        b = tokens_b[k].mean(dim=1)
        vals.append(F.cosine_similarity(a, b, dim=-1))
    return torch.stack(vals, 0).mean(0)


def dual_metric_filter(mid_cos, style_gain, *,
                       lo: float = MID_COS_BAND[0], hi: float = MID_COS_BAND[1],
                       gain_min: float = STYLE_GAIN_MIN) -> torch.Tensor:
    """双指标筛选 (B02 验收): (lo ≤ mid_cos ≤ hi) AND (style_gain ≥ gain_min); NaN 判拒绝."""
    mid = torch.as_tensor(mid_cos, dtype=torch.float32)
    gain = torch.as_tensor(style_gain, dtype=torch.float32)
    return (mid >= lo) & (mid <= hi) & (gain >= gain_min)


def _percentiles(x: torch.Tensor) -> dict:
    if x.numel() == 0:
        return {}
    qs = torch.quantile(x.float(), torch.tensor([0.05, 0.25, 0.50, 0.75, 0.95]))
    return {f"p{int(q * 100)}": round(float(v), 5) for q, v in zip([0.05, 0.25, 0.50, 0.75, 0.95], qs)}


def filter_diagnostics(mid_cos, style_gain, keep, edit_norm=None) -> dict:
    """筛选报告: 保留率 + 被筛/保留样本的幅度分布 (B02 验收"被筛样本幅度分布")."""
    mid = torch.as_tensor(mid_cos, dtype=torch.float32)
    gain = torch.as_tensor(style_gain, dtype=torch.float32)
    kp = torch.as_tensor(keep, dtype=torch.bool)
    n = int(mid.numel())
    diag = {
        "n_total": n,
        "n_kept": int(kp.sum()),
        "keep_rate": round(float(kp.float().mean()) if n else 0.0, 5),
    }
    if n:
        diag["mid_cos"] = {"kept": _percentiles(mid[kp]), "rejected": _percentiles(mid[~kp])}
        diag["style_gain"] = {"kept": _percentiles(gain[kp]), "rejected": _percentiles(gain[~kp])}
    if edit_norm is not None:
        en = torch.as_tensor(edit_norm, dtype=torch.float32)
        diag["edit_norm"] = {"kept": _percentiles(en[kp]), "rejected": _percentiles(en[~kp]),
                             "kept_mean": round(float(en[kp].mean()) if kp.any() else 0.0, 5)}
    return diag


def family_retention_report(sid_all, keep_mask, n_styles: int = 5,
                            style_names=None) -> dict:
    """A03 审计维度: 分画派保留率 (诊断专用 —— 绝不进入优化/筛选/损失).

    sid_all: (N,) 训练集画派 id (仅用于报告分组); keep_mask: (n_kept_from,) bool,
    与 idx_c 对齐 → 调用方先用 sid_all[idx_c] 取每对的内容画派再传入.
    """
    sid = torch.as_tensor(sid_all).long().reshape(-1)
    kp = torch.as_tensor(keep_mask, dtype=torch.bool).reshape(-1)
    assert sid.numel() == kp.numel(), "sid 与 keep_mask 必须逐对对齐"
    names = list(style_names) if style_names else [f"fam{i}" for i in range(n_styles)]
    per_family = {}
    for f in range(min(n_styles, len(names))):
        m = sid == f
        n_f = int(m.sum())
        if n_f == 0:
            continue  # 该画派无样本: 不产出误导性保留率
        per_family[names[f]] = {
            "n_total": n_f,
            "n_kept": int(kp[m].sum()),
            "keep_rate": round(float(kp[m].float().mean()), 5),
        }
    rates = [v["keep_rate"] for v in per_family.values()]
    return {
        "per_family": per_family,
        "keep_rate_spread": round(max(rates) - min(rates), 5) if rates else 0.0,
        "note": "diagnostic-only; sid 不进入优化目标/筛选条件/损失",
    }


# ------------------------------------------------------------ 教师优化 ----

def _draw_pairs(n_cand: int, n_swap: int, n_total: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """随机内容-参考配对 + swap 复用: 前 n_swap 个内容各配两个互异参考.

    返回 (idx_c, idx_r, swap_groups); swap_groups[j] = (j, n_content + j) 共享内容 base_c[j].
    """
    assert 0 <= n_swap < n_cand // 2, "n_swap 须 ∈ [0, n_cand/2) (前段内容复用两次)"
    n_content = n_cand - n_swap
    g = torch.Generator().manual_seed(seed)
    base_c = torch.randint(0, n_total, (n_content,), generator=g)

    def _ref_for(c: torch.Tensor) -> torch.Tensor:
        r = (c + torch.randint(1, n_total, (len(c),), generator=g)) % n_total
        return r

    r_first = _ref_for(base_c)
    r_second = _ref_for(base_c[:n_swap])
    # 第二参考尽量避开内容与第一参考 (best-effort 单次重抽)
    clash = (r_second == base_c[:n_swap]) | (r_second == r_first[:n_swap])
    r_second = torch.where(clash, (r_second + 1) % n_total, r_second)
    idx_c = torch.cat([base_c, base_c[:n_swap]])
    idx_r = torch.cat([r_first, r_second])
    swap_groups = torch.stack([torch.arange(n_swap), n_content + torch.arange(n_swap)], 1)
    return idx_c, idx_r, swap_groups


def build_teacher_pairs(
    content_w: float,
    *,
    pool: str | None = None,
    device: str = "cuda",
    n_cand: int = 2400,
    n_keep: int = 2000,
    steps: int = 24,
    n_swap: int = 400,
    mid_cos_band: tuple[float, float] = MID_COS_BAND,
    style_gain_min: float = STYLE_GAIN_MIN,
    lr: float = 0.055,
    batch: int = 12,
    seed: int = 20265100,
    fixed_pairs: tuple[torch.Tensor, torch.Tensor] | None = None,
    ref_cache_dir: str | Path | None = None,
    force: bool = False,
) -> dict:
    """生成 v2 教师包 (返回 pack dict; 落盘 runs/v25/cache/ref_teacher_pairs_v2_<tag>.pt).

    结构复用 exp.v24.o2_ref_teacher_pairs (可微 S10 + SWD + Oklab + 局部 DINO-match, Adam),
    仅内容权重参数化 + 双指标筛选 + 参考特征缓存. 签名无任何画派 ID (B02 验收 4);
    per-family 保留率仅作为落盘诊断 (见 family_retention_report).

    超参策略: 三档 content_w 共用同 (idx_c, idx_r) 与全部优化超参 —— C3 强度混合要求
    同对齐; 传 fixed_pairs 可显式钉死配对 (与第一档产物一致).
    """
    tag = teacher_tag(content_w)
    out_p = OUT / f"cache/ref_teacher_pairs_v2_{tag}.pt"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if out_p.exists() and not force:
        print(f"[V25 T-v2/{tag}] Reusing {out_p}", flush=True)
        return torch.load(out_p, map_location="cpu", weights_only=False)

    seed_all(seed)
    bundle = build_or_load_v19_foundation(pool, device=device)
    tr = bundle["v18_cache"]["train"]
    z_all = tr["z"].float()
    d16_all = tr["dino_16"].float()
    n_total = len(z_all)

    if fixed_pairs is not None:
        idx_c, idx_r = fixed_pairs[0].long(), fixed_pairs[1].long()
        assert len(idx_c) == n_cand, f"fixed_pairs 长度 {len(idx_c)} != n_cand {n_cand}"
        swap_groups = torch.zeros(0, 2, dtype=torch.long)
    else:
        n_swap = min(n_swap, n_cand // 4)  # 小诊断子集自动缩量 (R2 用 600 对时 n_swap→150)
        idx_c, idx_r, swap_groups = _draw_pairs(n_cand, n_swap, n_total, seed)

    dec = load_s10_decoder(device)
    swd_ext = MultiScalePatchSWDExtractor(n_slices=96).to(device).eval()
    for p in swd_ext.parameters():
        p.requires_grad_(False)
    cache = ReferenceFeatureCache(ref_cache_dir)

    z_cr = torch.zeros(n_cand, 4, 64, 64, dtype=torch.float16)
    mid_scores = torch.zeros(n_cand, dtype=torch.float32)
    style_gains = torch.zeros(n_cand, dtype=torch.float32)
    edit_norms = torch.zeros(n_cand, dtype=torch.float32)

    t0 = time.time()
    for st in range(0, n_cand, batch):
        sl = slice(st, min(st + batch, n_cand))
        ic, ir = idx_c[sl], idx_r[sl]  # CPU 索引 (与 o2 一致)
        z_src = z_all[ic].to(device)
        z_ref = z_all[ir].to(device)
        t_b = time.time()

        with torch.no_grad():
            # 前置特征 (内容侧 + 参考侧) 走 sha 缓存 v2: 全命中时零 decode/extract —— 三档
            # 权重共享同一批配对, 第 2/3 档前置阶段只剩缓存盘读 (collect_cached_features).
            src_items = collect_cached_features(cache, z_src, dec, swd_ext, role="s",
                                                need_pal=False, device=device)
            ref_items = collect_cached_features(cache, z_ref, dec, swd_ext, role="r",
                                                need_pal=True, device=device)
            toks_src, _ = batch_cached_ref_tokens(src_items, device)
            toks_ref, pal_ref = batch_cached_ref_tokens(ref_items, device)
            # 常量归一外提 (逐位等价: o2 在 24 步内对同一常量重复 normalize)
            src_n2 = F.normalize(toks_src["s2"], dim=-1)
            src_n3 = F.normalize(toks_src["s3"], dim=-1)

            # Q/K 灰度 DINO 局部语义匹配 (与 o2 同式: 内容 DINO 布局 ↔ 参考 DINO 布局,
            # 再 gather 参考 ConvNeXt s3 token; d16_all 在内存直接索引, 无需缓存)
            dc = F.normalize(d16_all[ic].to(device).flatten(2).transpose(1, 2), dim=-1)
            dr = F.normalize(d16_all[ir].to(device).flatten(2).transpose(1, 2), dim=-1)
            sim = torch.bmm(dc, dr.transpose(1, 2))
            best_sim, best_j = sim.max(dim=-1)
            gate_loc = (best_sim > 0.25).float().unsqueeze(-1)
            ref_s3 = toks_ref["s3"]
            assert ref_s3.dim() == 3 and ref_s3.shape[0] == len(ir), \
                f"toks_ref['s3'] 形状契约破坏: {tuple(ref_s3.shape)} (应为 (B,256,384))"
            matched_s3_n = F.normalize(torch.gather(ref_s3, 1, best_j.unsqueeze(-1).expand_as(ref_s3)), dim=-1)

        z_opt = z_src.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([z_opt], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            rgb_hat = decode_s10(dec, z_opt.clamp(-5.0, 5.0), batch=batch, enable_grad=True, output_cpu=False)
            toks_hat = swd_ext.extract_tokens(rgb_hat)
            l_swd = (
                1.0 * sliced_wasserstein_1d(toks_hat["rgb"], toks_ref["rgb"], swd_ext.p_rgb)
                + 0.35 * sliced_wasserstein_1d(toks_hat["s1"], toks_ref["s1"], swd_ext.p_s1)
                + 0.45 * sliced_wasserstein_1d(toks_hat["s2"], toks_ref["s2"], swd_ext.p_s2)
                + 0.55 * sliced_wasserstein_1d(toks_hat["s3"], toks_ref["s3"], swd_ext.p_s3)
            ).mean()
            l_pal = F.smooth_l1_loss(extract_color_fiber_27d(rgb_hat), pal_ref)
            l_loc = ((1.0 - (F.normalize(toks_hat["s3"], dim=-1) * matched_s3_n).sum(-1, keepdim=True)) * gate_loc).mean()
            l_content = 0.5 * (
                (1.0 - (F.normalize(toks_hat["s2"], dim=-1) * src_n2).sum(-1)).mean()
                + (1.0 - (F.normalize(toks_hat["s3"], dim=-1) * src_n3).sum(-1)).mean()
            )
            l_reg = (z_opt - z_src).square().mean()
            loss = (l_swd + 0.7 * l_pal + 0.25 * l_loc) + content_w * l_content + 0.15 * l_reg
            loss.backward()
            opt.step()

        with torch.no_grad():
            z_fin = z_opt.detach().clamp(-5.0, 5.0)
            rgb_fin = decode_s10(dec, z_fin, batch=batch, output_cpu=False)
            tf = swd_ext.extract_tokens(rgb_fin)
            cos_mid = 0.5 * (
                (F.normalize(tf["s2"], dim=-1) * src_n2).sum(-1).mean(dim=1)
                + (F.normalize(tf["s3"], dim=-1) * src_n3).sum(-1).mean(dim=1)
            )
            gain = pooled_style_cos(tf, toks_ref) - pooled_style_cos(toks_src, toks_ref)
            z_cr[sl] = z_fin.cpu().half()
            mid_scores[sl] = cos_mid.cpu()
            style_gains[sl] = gain.cpu()
            edit_norms[sl] = (z_fin - z_src).flatten(1).norm(dim=1).cpu()

        if (st // batch) % 20 == 0 or st + batch >= n_cand:
            done = min(st + batch, n_cand)
            batch_s = time.time() - t_b
            print(f"[V25 T-v2/{tag}] {done}/{n_cand} mean_mid_cos={float(mid_scores[:done].mean()):.4f} "
                  f"mean_gain={float(style_gains[:done].mean()):.4f} "
                  f"batch_s={batch_s:.0f} step_s={batch_s / max(1, steps):.1f} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    # ---- 双指标筛选 (取代 o2 的 mid_cos top-k) ----
    keep = dual_metric_filter(mid_scores, style_gains, lo=mid_cos_band[0], hi=mid_cos_band[1],
                              gain_min=style_gain_min)
    n_kept = int(keep.sum())
    if n_kept > n_keep:  # 超额: 按 style_gain 降序取前 n_keep (最大化风格信号, 确定性)
        order = style_gains[keep].argsort(descending=True)
        sel = keep.nonzero(as_tuple=True)[0][order[:n_keep]]
        keep = torch.zeros_like(keep)
        keep[sel] = True
        n_kept = n_keep
    if n_kept < n_keep:
        print(f"[V25 T-v2/{tag}] WARNING: 双指标仅保留 {n_kept}/{n_keep} (阈值过紧, 待 A03 校准)", flush=True)

    # swap 组仅在两成员同时 kept 时保留
    if swap_groups.numel():
        both = keep[swap_groups[:, 0]] & keep[swap_groups[:, 1]]
        kept_groups = swap_groups[both]
        kept_groups = _remap_groups(kept_groups, keep)
    else:
        kept_groups = swap_groups

    diag = filter_diagnostics(mid_scores, style_gains, keep, edit_norm=edit_norms)
    # 诊断专用: 分画派保留率 (A03 审计维度; 不进入优化/筛选/损失)
    fam = family_retention_report(tr["sid"].long()[idx_c], keep, n_styles=5, style_names=STYLES)
    diag["per_family_content"] = fam

    pack = {
        "kind": "ref_teacher_pairs_v2", "tag": tag, "content_w": float(content_w),
        "idx_c": idx_c[keep].cpu(), "idx_r": idx_r[keep].cpu(), "z_cr": z_cr[keep].cpu(),
        "mid_cos": mid_scores[keep].cpu(), "style_gain": style_gains[keep].cpu(),
        "edit_norm": edit_norms[keep].cpu(), "swap_groups": kept_groups.cpu(),
        "mid_cos_all": mid_scores.cpu(), "style_gain_all": style_gains.cpu(),
        "edit_norm_all": edit_norms.cpu(), "keep_mask_all": keep.cpu(),
        "idx_c_all": idx_c.cpu(), "idx_r_all": idx_r.cpu(),
        "pair_uid_all": (idx_c.long() * 100000 + idx_r.long()).cpu(),  # n_total < 100000 (训练集 4400)
        "filter": {"mid_cos_band": list(mid_cos_band), "style_gain_min": float(style_gain_min),
                   "provenance": STYLE_GAIN_PROVENANCE},
        "meta": {"n_cand": n_cand, "n_keep": n_keep, "n_swap": n_swap, "steps": steps, "lr": lr,
                 "optimizer": "adam(o2 结构)", "seed": seed, "batch": batch,
                 "thresholds": STYLE_GAIN_PROVENANCE, "ref_cache": cache.stats(),
                 "n_swap_groups_kept": int(kept_groups.shape[0]), "updated_utc": utc()},
    }
    torch.save(pack, out_p)
    save_json(OUT / f"cache/ref_teacher_pairs_v2_{tag}_diag.json", diag)
    print(f"[V25 T-v2/{tag}] keep={n_kept}/{n_cand} ({diag['keep_rate']:.2%}) "
          f"mid_cos_all={float(mid_scores.mean()):.4f} gain_all={float(style_gains.mean()):.4f} "
          f"cache={cache.stats()} -> {out_p}", flush=True)
    return pack


def _remap_groups(groups: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """kept 掩码下的旧对索引 → 新 kept 数组索引."""
    if groups.numel() == 0:
        return groups
    new_id = torch.full((len(keep),), -1, dtype=torch.long)
    new_id[keep] = torch.arange(int(keep.sum()))
    return new_id[groups]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--content-w", default="1.2", help="2.2 | 1.2 | 0.7 | all")
    ap.add_argument("--n-cand", type=int, default=2400)
    ap.add_argument("--n-keep", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--n-swap", type=int, default=400)
    ap.add_argument("--batch", type=int, default=12,
                    help="每步并行优化的对数 (Adam 尺度不变 → 单对轨迹与 batch 大小无关; "
                         "3060 12GB 建议先 12, 显存有余再试 16)")
    ap.add_argument("--mid-cos-lo", type=float, default=MID_COS_BAND[0])
    ap.add_argument("--mid-cos-hi", type=float, default=MID_COS_BAND[1])
    ap.add_argument("--style-gain-min", type=float, default=STYLE_GAIN_MIN)
    ap.add_argument("--ref-cache-dir", default=None)
    ap.add_argument("--fast", action="store_true",
                    help="cudnn.benchmark 固定形状自动调优 (步级 ~1.2-1.4×; 轨迹近似一致, 见 enable_fast_conv)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    sanitize_cuda_env()
    if args.fast:
        enable_fast_conv()
        print("[V25 T-v2] --fast: cudnn.benchmark=True (固定形状卷积自动调优)", flush=True)
    ws = list(TEACHER_W_GRID) if args.content_w == "all" else [float(args.content_w)]
    summary = {}
    for w in ws:
        pack = build_teacher_pairs(
            w, pool=args.pool, device=args.device, n_cand=args.n_cand, n_keep=args.n_keep,
            steps=args.steps, n_swap=args.n_swap, batch=args.batch,
            mid_cos_band=(args.mid_cos_lo, args.mid_cos_hi), style_gain_min=args.style_gain_min,
            ref_cache_dir=args.ref_cache_dir, force=args.force)
        summary[teacher_tag(w)] = {
            "n_kept": int(len(pack["idx_c"])),
            "keep_rate": round(float(pack["keep_mask_all"].float().mean()), 5),
            "mean_mid_cos": round(float(pack["mid_cos_all"].mean()), 5),
            "mean_style_gain": round(float(pack["style_gain_all"].mean()), 5),
            "swap_groups_kept": int(pack["swap_groups"].shape[0]),
        }
    save_json(OUT / "reference_teacher_v2.json", {"updated_utc": utc(), "args": vars(args), "summary": summary})
    for tag, s in summary.items():
        print(f"[V25 T-v2] {tag}: {s}", flush=True)


if __name__ == "__main__":
    main()
