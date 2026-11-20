r"""V25 机制 C: One-Shot 监督强度校准与参考替换一致性 (C0–C3) —— B02 实现.

docs/19_pull_together/02_mechanism_plan.md §机制 C + agent2.md §二 G2-2 修正式 / §三 B02 验收.

数学要点
--------
1. **风格项一律 (1 − cos) 形式** (B02 验收 1, 禁止直接最小化正余弦相似度):

       style(f_out, f_ref) = 1 − cos(f_out, f_ref) ∈ [0, 2]
       cos → 1 (同特征) 当且仅当 loss → 0; cos = 0 (正交) → loss = 1; cos = −1 → loss = 2.
       ∇_f(1 − cos) = −∇_f cos: 最小化 (1−cos) 把 f_out 推向 f_ref;
       若误写 "最小化 cos" 则把 f_out 推向 −f_ref (负相关, 方向相反) —— 单测断言两种写法
       对 f_out 的梯度逐元素相反, 且 GD 迭代后余弦一个升一个降.

2. **参考替换损失** (agent2.md §二 G2-2 修正式; 两个输出分别受内容保持约束):

       L_swap = style(G_a, r_a) + style(G_b, r_b)
                + λ_c · [ L_c(G_a, x_c) + L_c(G_b, x_c) ],   λ_c 默认 0.5 (参数化)

   G_a = 学生对 (x_c, r_a) 的输出, G_b = 学生对 (x_c, r_b) 的输出. 内容项 λ_c 逐输出施加
   (取代 plan §C.3 的输出差 |C(G_a)−C(G_b)| 形式 —— 逐输出约束在梯度上更直接且无对抗塌缩).
   swap 项是一致性正则 (默认权重 0.5): 主风格信号仍是教师包 flow-MSE (C1 同款).

3. **四条件参考评测钩子** ``eval_ref_conditions`` (B02 验收 3): 正确参考 / 错误参考 /
   同风格不同参考 / 完全不同风格参考, 输出每条件 (dino_c vs 内容, dino_s vs 各自参考,
   dino_s vs 正确参考) 表. Smoke 断言 (CPU mock 可测, GPU 评测执行):
       S1: 错误参考输出对"正确参考"的 dino_s 不得高于正确参考条件 (防对齐反转);
       S2: 错误参考条件对自身参考的 dino_s 不得高于正确参考条件 (风格必须真的跟随).

4. **无画派 ID** (B02 验收 4): 本模块全部训练/评测接口签名不含 sid/onehot/目标类别输入
   (``assert_label_free_interfaces`` 用 inspect 机械断言, 每次 run_stage 前自检);
   SLFUNetV24OneShot.forward 的 style_id 槽位是 v24 legacy 兼容参数, 其内部恒置零
   onehot (oh = zeros) —— 本模块调用一律传 0 张量, 与 eval 同口径; 四条件评测的参考
   选择在评测器内部读取数据集元数据 (与 eval_o_arm 读 tid 同性质), 不构成模型输入.

5. **参考特征缓存** (B02 验收 5): exp.v25.reference_teacher_v2.ReferenceFeatureCache
   按参考图 latent 内容 sha256 键控, 文件存 --cache-dir (默认 runs/v25/cache/ref_feats/),
   同参考重复编码直接命中.

3060 12GB 显存约束 (硬性)
-------------------------
- flow-MSE micro ≤ 12 (参照 train_o_arm micro=12 × accum=4);
- C2 swap 预览子批 swap_micro ≤ 4 组 (= 2·4=8 张 512² 可微 S10 解码 + VGG 前向反向);
  若 OOM: 降 --swap-micro 2 或 --swap-weight 0 (纯 C1 口径);
- 训练臂 1200 updates ≈ 20–30 min/臂 (C2 略慢); 评测执行在独立进程 (src.eval.runner).

阶段
----
- C0   只评已有 O3 (runs/v24/oneshot_val/O3_rgb_cnx_gated.pt) —— τ45/55 + 四条件钩子.
- C1   校准教师包训练: reference_teacher_v2.build_teacher_pairs(content_w) → train_o_arm
       (复用 r3_oneshot_minimal 的 OUT 重定向手法, 产出落 runs/v25).
- C2   C1 + swap 损失: train_c2_arm, 特征 fn 参数化 (VGG-16 / ConvNeXt, 训练专用);
       **禁用评测裁判 DINOv2/CSD/CLIP/LPIPS 作训练损失** (assert_train_feature_kind).
- C3   连续教师强度混合: γ_t ∈ {0.4, 0.7, 1.0} 逐对混合保守(2.2)/表现(0.7)教师目标
       (对级对齐, 双包 kept 交集), 复用 train_o_arm; γ 仅作用于目标侧, 架构零改动.
- C2nt 方案③ 无教师参考正则训练 (绕开教师问题本身):
       C1 教师两轮结构验证证明 latent-SWD 优化下 mid_cos 频带不可达 (w_c/步数/锚/信任域
       全排除), 且 self-pair 单独训练必然恒等坍缩 (O3_self_only Top-1 12.33%) —— 方案③
       直接把风格信号从"伪目标 MSE"换成"参考正则损失":
         训练对 = 自配对 (z 目标 = z_src 恒等, 内容锚 + same 任务语义);
         损失   = flow_MSE(自配对) + λ_sw·style(一阶预览, r_i) + λ_c·content(预览, z_src);
         预览   = z_init + v·(1−τ0) → 可微 S10 → 训练专用特征 (VGG/ConvNeXt);
         参考池 = 训练集 4400 张自身, 每样本参考 = 其跨风格邻居 (nbr_idx, sid 仅用于参考
                  抽样选择, 非模型输入); 同内容双参考组 ≥1 组/步 (batch 内 ≥2 不同参考
                  结构性保证); 参考特征走 ReferenceFeatureCache (fp32 v2, 独立子目录
                  ref_feats_c2nt/).
       MSE 锚住恒等, swap 风格项提供"离开恒等"的梯度 —— 编辑幅度由 λ_sw 隐式决定,
       无教师、无频带问题. 风格 dropout 关闭 (ds≡False): swap 项需要每个样本的风格条件
       速度, o3 的 12% dropout 会清零被丢弃样本的风格梯度.
       3060 12GB 预算: 每 update = 4×flow fwd/bwd (micro 12) + 1×swap pass
       (S10 可微解码 12×512² + 训练特征 fwd/bwd) → 1200 updates ≈ 25-40 min;
       OOM 降 --micro 8. λ 校准计划: 先 --updates 200 探针跑 λ_sw ∈ {0.2, 0.5, 1.0}
       (诊断每 200 步输出 preview_content_cos / preview_style_cos —— 训练即可见
       "编辑幅度-内容保持"曲线, 无需等评测), 再定正式 λ_sw.
       用法: python -m exp.v25.mechanism_c --stage C2nt --lambda-sw 0.5 --updates 1200
       判读: t45/t55 vs C0/O3 与 O3_self_only —— Top-1 ≥ 20% 且 DINO-S ≥ 0.48 → 方案③成立.

用法 (3060, 本任务只交付不启动 GPU; G2 队列 R2–R4/R5)
------------------------------------------------------
  R2 教师三档诊断:  python -m exp.v25.reference_teacher_v2 --content-w all --n-cand 600 --steps 24
  R3 2000 对生成:   python -m exp.v25.reference_teacher_v2 --content-w <R2胜者> --n-cand 2400 --n-keep 2000
  R4 O3 新教师训练: python -m exp.v25.mechanism_c --stage C1 --teacher-w <R2胜者> --updates 1200
  R5 参考替换:      python -m exp.v25.mechanism_c --stage C2 --teacher-w <R2胜者> --feature-kind vgg
  C3 强度混合:      python -m exp.v25.mechanism_c --stage C3 --updates 1200
  C0 基线:          python -m exp.v25.mechanism_c --stage C0
"""
from __future__ import annotations

import argparse
import copy
import inspect
import math
import re
import time
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pathlib import Path

from . import OUT, ROOT, autocast, sanitize_cuda_env, save_json, seed_all, utc
from exp.v25.reference_teacher_v2 import (
    MID_COS_BAND,
    ReferenceFeatureCache,
    STYLE_GAIN_MIN,
    TEACHER_W_GRID,
    build_teacher_pairs,
    teacher_tag,
)
import exp.v24 as _v24
import exp.v24.p0_o1_o3_value as _o_mod
from exp.v24.p0_o1_o3_value import (
    DUAL_CKPT,
    O_SPECS,
    SLFUNetV24OneShot,
    build_or_load_train_cnx_v16,
    train_o_arm,
)
from exp.v19.fiber_duality import StyleFiberEncoder, build_or_load_v19_foundation
from exp.v20.slf_flow import coarse_y16_lcn
from exp.v20.submode_patchswd import load_convnext_tiny_offline
from exp.v22.ref_adapter import RefAdapter
from src.utils import FeatureStack, decode_s10, load_s10_decoder
from src.utils.protocol import load_vgg16_features_offline


def _lazy_r3():
    """延迟导入 r3_oneshot_minimal 的评测器 (仅 GPU 阶段入口需要).

    其评测链 exp.v23.p2_sg_bench → src.utils._local_eval_refs → tools/12h/csd_model
    会在 import 期初始化 CUDA 内部状态 (torch 2.13: CVD='' 下 device_count 失真为 1),
    延迟到 GPU 阶段调用可保持本模块在 CPU 单测环境 import 干净.
    """
    from exp.v25.r3_oneshot_minimal import O3_CKPT, eval_o_tau_list

    return O3_CKPT, eval_o_tau_list

# ---- 预注册常量 (不得在臂间私下改动) ----
LAMBDA_C_DEFAULT = 0.5          # B02 验收 2: swap 损失内容权重, 参数化默认 0.5
SWAP_WEIGHT_DEFAULT = 0.5       # swap 项相对 flow-MSE 的总权重
C3_GAMMA_POOL = (0.4, 0.7, 1.0) # plan §C.4: 连续教师强度混合池
TRAIN_FEATURE_KINDS = ("vgg", "cnx")
FORBIDDEN_TRAIN_FEATURES = ("dino", "csd", "clip", "lpips")  # 评测裁判, 禁作训练损失
MICRO_3060 = 12                 # 3060 12GB: flow micro 上限 (train_o_arm 口径)
SWAP_MICRO_3060 = 4             # 3060 12GB: swap 预览组数上限 (2×4=8 张可微解码)
COND_CORRECT = "correct_ref"
COND_WRONG = "wrong_ref"
COND_SAME_STYLE = "same_style_alt_ref"
COND_DIFF_STYLE = "diff_style_ref"
CONDITION_KEYS = (COND_CORRECT, COND_WRONG, COND_SAME_STYLE, COND_DIFF_STYLE)

# 接口签名禁用参数名 (B02 验收 4; inspect 机械断言)
_FORBIDDEN_PARAM_RE = re.compile(
    r"(^|_)(sid|tid|style_id|styleid|target_id|targetid|target_class|class_id|classid|"
    r"onehot|one_hot|labels?)(_|$)",
    re.IGNORECASE,
)
LABEL_FREE_INTERFACES = (
    "build_teacher_pairs", "train_c2_arm", "train_c2nt_arm", "eval_ref_conditions",
    "run_ref_condition_eval", "make_train_feature_fn", "swap_loss", "swap_loss_features",
    "swap_loss_images", "run_stage",
)


def assert_label_free_interface(fn, name: str | None = None) -> None:
    """inspect 断言: 接口签名不含 sid/onehot/目标类别参数 (B02 验收 4)."""
    bad = [p for p in inspect.signature(fn).parameters if _FORBIDDEN_PARAM_RE.search(p)]
    assert not bad, f"{name or getattr(fn, '__name__', fn)} 接口含画派 ID 参数: {bad}"


def assert_label_free_interfaces() -> None:
    for fname in LABEL_FREE_INTERFACES:
        assert_label_free_interface(globals()[fname], fname)


# ------------------------------------------------ (1−cos) 损失族 ----

def _pooled(feat: torch.Tensor) -> torch.Tensor:
    """(B,C,H,W) → 空间均值池化 (B,C); (B,D) 原样."""
    return feat.mean(dim=(2, 3)) if feat.dim() == 4 else feat


def cosine_distance_loss(feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
    """(1 − cos) 标量损失: 同特征 → 0, 正交 → 1, 反向 → 2; 逐样本均值."""
    a, b = _pooled(feat_a).float(), _pooled(feat_b).float()
    return (1.0 - F.cosine_similarity(a, b, dim=-1)).mean()


def style_cos_loss(feat_out: torch.Tensor, feat_ref: torch.Tensor) -> torch.Tensor:
    """风格跟随项 (1 − cos) —— 语义命名, 数学同 cosine_distance_loss."""
    return cosine_distance_loss(feat_out, feat_ref)


def content_cos_loss(feat_out: torch.Tensor, feat_src: torch.Tensor) -> torch.Tensor:
    """内容保持项 (1 − cos) —— 语义命名, 数学同 cosine_distance_loss."""
    return cosine_distance_loss(feat_out, feat_src)


def swap_loss(f_a, f_b, f_ra, f_rb, f_c, lambda_c: float = LAMBDA_C_DEFAULT) -> torch.Tensor:
    """特征级参考替换损失 (B02 验收 2, G2-2 修正式):

        L_swap = (1−cos)(f_a, f_ra) + (1−cos)(f_b, f_rb)
                 + λ_c · [(1−cos)(f_a, f_c) + (1−cos)(f_b, f_c)]

    两个输出 (f_a/f_b) 分别受内容保持约束; λ_c=0 时退化为纯风格项.
    """
    return (
        cosine_distance_loss(f_a, f_ra)
        + cosine_distance_loss(f_b, f_rb)
        + lambda_c * (cosine_distance_loss(f_a, f_c) + cosine_distance_loss(f_b, f_c))
    )


def swap_loss_features(fa: dict, fb: dict, fra: dict, frb: dict, fc: dict,
                       lambda_c: float = LAMBDA_C_DEFAULT) -> torch.Tensor:
    """特征 dict 级 ({"style": feat, "content": feat}) swap 损失: style 项用 style 特征,
    内容项用 content 特征 —— 供 C2 训练循环在特征已预计算时使用 (避免重复 VGG 前向)."""
    return (
        cosine_distance_loss(fa["style"], fra["style"])
        + cosine_distance_loss(fb["style"], frb["style"])
        + lambda_c * (cosine_distance_loss(fa["content"], fc["content"])
                      + cosine_distance_loss(fb["content"], fc["content"]))
    )


def swap_loss_images(rgb_a, rgb_b, rgb_ra, rgb_rb, rgb_c, feature_fn, lambda_c: float = LAMBDA_C_DEFAULT) -> torch.Tensor:
    """图像级入口: feature_fn(rgb01) → {"style": feat, "content": feat}, 逐图提取一次后套 swap_loss_features."""
    fa, fb = feature_fn(rgb_a), feature_fn(rgb_b)
    fra, frb, fc = feature_fn(rgb_ra), feature_fn(rgb_rb), feature_fn(rgb_c)
    return swap_loss_features(fa, fb, fra, frb, fc, lambda_c)


# -------------------------------------------- 训练专用特征 (非评测裁判) ----

def assert_train_feature_kind(kind: str) -> None:
    """评测裁判 (DINOv2/CSD/CLIP/LPIPS) 禁止作训练损失特征; 只放行 VGG/ConvNeXt."""
    k = str(kind).lower()
    for bad in FORBIDDEN_TRAIN_FEATURES:
        if bad in k:
            raise ValueError(f"特征 '{kind}' 含评测裁判 '{bad}' —— 铁律: 不直接优化评测裁判 (允许: {TRAIN_FEATURE_KINDS})")
    if k not in TRAIN_FEATURE_KINDS:
        raise ValueError(f"未知训练特征 '{kind}' (允许: {TRAIN_FEATURE_KINDS})")


def _make_vgg_feature_fn(device: str):
    """VGG-16 relu3_3+relu4_3 空间均值池化: style=(B,768), content=(B,512) (训练专用)."""
    vgg = load_vgg16_features_offline(device).eval()
    for p in vgg.parameters():
        p.requires_grad_(False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    relu_idx = (15, 22)  # relu3_3 (256ch) / relu4_3 (512ch)

    def fn(rgb01: torch.Tensor) -> dict:
        x = (rgb01.float() - mean) / std
        f3 = f4 = None
        h = x
        for i, m in enumerate(vgg):
            h = m(h)
            if i == relu_idx[0]:
                f3 = h
            elif i == relu_idx[1]:
                f4 = h
        style = torch.cat([f3.mean(dim=(2, 3)), f4.mean(dim=(2, 3))], dim=-1)
        return {"style": style, "content": f4.mean(dim=(2, 3))}

    return fn


def _make_cnx_feature_fn(device: str):
    """ConvNeXt-Tiny s2+s3 空间均值池化: style=(B,576), content=(B,192) (训练专用, plan A.3 口径)."""
    cnx = load_convnext_tiny_offline(device).eval()
    for p in cnx.parameters():
        p.requires_grad_(False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def fn(rgb01: torch.Tensor) -> dict:
        x = (F.interpolate(rgb01.float(), size=(256, 256), mode="bilinear", align_corners=False) - mean) / std
        f2 = cnx[:4](x)    # (B,192,32,32)
        f3 = cnx[4:6](f2)  # (B,384,16,16)
        return {"style": torch.cat([f2.mean(dim=(2, 3)), f3.mean(dim=(2, 3))], dim=-1),
                "content": f2.mean(dim=(2, 3))}

    return fn


def make_train_feature_fn(kind: str = "vgg", device: str = "cuda"):
    """构造训练专用特征 fn (冻结, 可微输入); 评测裁判特征在此处被结构性拒绝."""
    assert_train_feature_kind(kind)
    return _make_vgg_feature_fn(device) if kind == "vgg" else _make_cnx_feature_fn(device)


# ------------------------------------------------- 四条件参考评测钩子 ----

def eval_ref_conditions(gen_feats: dict, ref_feats: dict, content_feat: torch.Tensor) -> list[dict]:
    """四条件参考评测钩子 (纯函数, CPU mock 特征可测; B02 验收 3).

    gen_feats : {cond: (T,D) 生成输出特征} — GPU 路径喂评测裁判 DINOv2 特征 (仅评测允许)
    ref_feats : {cond: (T,D) 该条件所用参考的特征}, 须含全部四条件
    content_feat: (T,D) 内容图特征

    返回逐条件表: dino_c (vs 内容), dino_s_own (vs 各自参考), dino_s_correct (vs 正确参考).
    """
    for cond in CONDITION_KEYS:
        assert cond in gen_feats, f"gen_feats 缺条件 {cond}"
        assert cond in ref_feats, f"ref_feats 缺条件 {cond}"
    fc = content_feat
    rows = []
    for cond in CONDITION_KEYS:
        fg = gen_feats[cond]  # (T, D), 逐任务与 content/ref 特征同索引对齐
        rows.append({
            "condition": cond,
            "dino_c": round(float(F.cosine_similarity(fg, fc, dim=-1).mean()), 5),
            "dino_s_own": round(float(F.cosine_similarity(fg, ref_feats[cond], dim=-1).mean()), 5),
            "dino_s_correct": round(float(F.cosine_similarity(fg, ref_feats[COND_CORRECT], dim=-1).mean()), 5),
        })
    return rows


def check_ref_conditions_smoke(table: list[dict], *, tol: float = 0.0) -> dict:
    """Smoke 断言逻辑 (CPU 可测): 错误参考条件不得获得高于正确参考的 dino_s.

    S1 (agent2.md §三 B02): 错误参考输出对"正确参考"的得分 ≤ 正确参考条件;
    S2 (任务书 smoke):    错误参考条件对自身参考的 dino_s ≤ 正确参考条件.
    """
    by = {r["condition"]: r for r in table}
    missing = [c for c in CONDITION_KEYS if c not in by]
    if missing:
        return {"ok": False, "failures": [f"missing conditions: {missing}"]}
    wc, cc = by[COND_WRONG], by[COND_CORRECT]
    fails = []
    if wc["dino_s_correct"] > cc["dino_s_correct"] + tol:
        fails.append(
            f"S1 错误参考输出获得更高正确参考得分: wrong={wc['dino_s_correct']:.4f} > correct={cc['dino_s_correct']:.4f}")
    if wc["dino_s_own"] > cc["dino_s_own"] + tol:
        fails.append(
            f"S2 错误参考条件自身参考 dino_s 高于正确参考条件: wrong={wc['dino_s_own']:.4f} > correct={cc['dino_s_own']:.4f}")
    return {"ok": not fails, "failures": fails}


def assert_ref_conditions_smoke(table: list[dict], *, tol: float = 0.0) -> None:
    r = check_ref_conditions_smoke(table, tol=tol)
    assert r["ok"], "参考条件 smoke 失败: " + "; ".join(r["failures"])


# ------------------------------------------------------- C3 强度混合 ----

def mix_teacher_strengths(z_cons: torch.Tensor, z_expr: torch.Tensor, gamma) -> torch.Tensor:
    """z_mix = (1−γ)·z_cons + γ·z_expr (γ=0 保守 / γ=1 表现)."""
    g = torch.as_tensor(gamma, dtype=z_cons.dtype, device=z_cons.device).view(-1, 1, 1, 1)
    return (1.0 - g) * z_cons + g * z_expr


def build_strength_mix_pack(cons_pack: dict, expr_pack: dict, gammas=C3_GAMMA_POOL,
                            *, seed: int = 20265300) -> dict:
    """C3: γ_t ∈ {0.4,0.7,1.0} 逐对混合保守(2.2)/表现(0.7)教师目标.

    要求两包同配对 (同 pair_uid_all, 同 seed 或 fixed_pairs 保证); 训练对取双包 kept 交集,
    γ 逐对均匀取自池 (pack 级固定). 架构零改动: γ 不新增任何输入接口 —— 单模型能否在
    相同输入条件下吸收多强度目标, 正是 C3 的科学问题 (判负出口: "目标侧强度混合不足以
    产生可控 strength", 不归因于架构). 混合包不含 swap 组 (C2 专用).
    """
    assert torch.equal(cons_pack["pair_uid_all"], expr_pack["pair_uid_all"]), \
        "C3 两档教师包必须同配对 (同 seed 或 fixed_pairs)"
    uid_k = cons_pack["pair_uid_all"][cons_pack["keep_mask_all"]]
    uid_e = expr_pack["pair_uid_all"][expr_pack["keep_mask_all"]]
    n_e = len(uid_e)
    if n_e and len(uid_k):
        order_e = uid_e.argsort()
        uid_e_s = uid_e[order_e]
        pos = torch.searchsorted(uid_e_s, uid_k).clamp_max(n_e - 1)
        hit = uid_e_s[pos] == uid_k
    else:
        hit = torch.zeros(len(uid_k), dtype=torch.bool)
    i_c = hit.nonzero(as_tuple=True)[0]           # cons kept 数组内位置 (升序)
    i_e = order_e[pos[hit]] if n_e and len(uid_k) else torch.zeros(0, dtype=torch.long)
    g = torch.Generator().manual_seed(seed)
    gamma = torch.tensor(gammas, dtype=torch.float32)[
        torch.randint(0, len(gammas), (len(i_c),), generator=g)]
    z_mix = mix_teacher_strengths(cons_pack["z_cr"][i_c].float(), expr_pack["z_cr"][i_e].float(), gamma)
    return {
        "kind": "ref_teacher_pairs_v2_mix", "gammas": list(gammas), "seed": seed,
        "idx_c": cons_pack["idx_c"][i_c], "idx_r": cons_pack["idx_r"][i_c],
        "z_cr": z_mix.half(), "gamma": gamma,
        "n_pairs": len(i_c),
        "n_cons_only": int(cons_pack["keep_mask_all"].sum()) - len(i_c),
        "n_expr_only": int(expr_pack["keep_mask_all"].sum()) - len(i_c),
        "swap_groups": torch.zeros(0, 2, dtype=torch.long),
        "meta": {"updated_utc": utc(), "note": "gamma 仅作用教师目标 (架构零改动)"},
    }


# ------------------------------------------------- One-Shot 积分 (GPU) ----

def _oneshot_integrate(fm, z_init, y16, d16, u_style, u_glob, d16_r, v16_r,
                       tau0: float, wj: float, device: str, steps: int = 10) -> torch.Tensor:
    """与 eval_o_tau_list 同口径: 10 步 Heun + 风格 CFG (单批, 无 same bypass)."""
    B, dev = z_init.shape[0], z_init.device
    f0 = torch.zeros(B, dtype=torch.bool, device=dev)
    f1 = torch.ones(B, dtype=torch.bool, device=dev)
    l1 = torch.ones(B, device=dev)
    slot = torch.zeros(B, dtype=torch.long, device=dev)  # legacy onehot 槽位恒零 (标签自由)
    dt = (1.0 - tau0) / float(steps)

    def _v(zz, pr):
        tv = torch.full((B,), tau0 + pr * (1.0 - tau0), device=dev)
        with autocast(device):
            out = fm(torch.cat([zz, zz], 0), y16.repeat(2, 1, 1, 1), d16.repeat(2, 1, 1, 1),
                     slot.repeat(2), u_style.repeat(2, 1), tv.repeat(2),
                     drop_style=torch.cat([f0, f1], 0), drop_content=torch.cat([f0, f0], 0),
                     layout_scale=torch.cat([l1, l1], 0),
                     d16_ref=d16_r.repeat(2, 1, 1, 1), v16_ref=v16_r.repeat(2, 1, 1, 1),
                     u_glob=u_glob.repeat(2, 1)).float()
        v_sc, v_0c = out.chunk(2, 0)
        return v_0c + wj * (v_sc - v_0c)

    z = z_init
    for i in range(steps):
        v1 = _v(z, i / float(steps))
        z = z + dt * _v(z + 0.5 * dt * v1, (i + 0.5) / float(steps))
    return z


def _to_pils(rgb01) -> list:
    return [Image.fromarray((x.permute(1, 2, 0).cpu().numpy() * 255).round().clip(0, 255).astype("uint8"))
            for x in rgb01]


@torch.no_grad()
def run_ref_condition_eval(bundle, c21, cnx, v16_tr, ckpt_p, *, n_tasks: int = 48, chunk: int = 12,
                           tau0: float = 0.45, wj: float = 2.2, steps: int = 10, device: str = "cuda",
                           seed: int = 20265400, strict: bool = False) -> dict:
    """GPU 四条件评测执行器 → eval_ref_conditions 纯函数表 + smoke 断言.

    四条件参考选择 (评测器内部读取数据集元数据, 与 eval_o_arm 读 tid 同性质; 非模型输入):
      correct        = 随机互异参考 (与训练对分布一致);
      wrong          = 第三画派参考 (sid ∉ {sid(content), sid(correct)});
      same_style_alt = 与 correct 同画派的另一训练图;
      diff_style     = ConvNeXt pooled 特征与 correct 参考余弦最小的候选 (风格最远).
    strict=False 时仅记录 smoke 结果 (队列安全); True 时断言失败即抛.
    """
    seed_all(seed)
    ck = torch.load(ckpt_p, map_location="cpu", weights_only=False)
    fm = SLFUNetV24OneShot(v_dim=192, use_conf_gate=True, use_global_fib=True).to(device).eval()
    fm.load_state_dict(ck["fm"], strict=True)
    ad = RefAdapter(dim=128).to(device).eval()
    ad.load_state_dict(ck["adapter"], strict=True)
    fib_mean, fib_std = ck["fib_mean"].to(device), ck["fib_std"].to(device)
    fiber_enc = StyleFiberEncoder(in_dim=bundle["fnd"]["tr_fib"].shape[1], dim=128).to(device).eval()
    fiber_enc.load_state_dict(bundle["fnd"]["fiber_state"])
    tr = bundle["v18_cache"]["train"]
    z_all, d16_all, sid_all = tr["z"], tr["dino_16"], tr["sid"].long()
    y16_all = coarse_y16_lcn(tr["y_64"].float()).half()
    u_all = fiber_enc.encode(bundle["fnd"]["tr_fib"].to(device).float())
    fib_n_all = (bundle["fnd"]["tr_fib"].to(device).float() - fib_mean) / fib_std
    n_total = len(z_all)

    g = torch.Generator().manual_seed(seed + 1)
    cand_pool = torch.randperm(n_total, generator=g)[:max(n_tasks * 4, 256)]
    ar = torch.arange(n_total)
    ic_list, ref_sel = [], {k: [] for k in CONDITION_KEYS}
    for i in range(n_tasks):
        c = int(cand_pool[i % len(cand_pool)])
        r = int((c + 1 + int(torch.randint(0, n_total - 1, (1,), generator=g))) % n_total)
        sc, sr = int(sid_all[c]), int(sid_all[r])
        third = ((sid_all != sc) & (sid_all != sr)).nonzero(as_tuple=True)[0]
        wrong = int(third[int(torch.randint(0, len(third), (1,), generator=g))])
        same = ((sid_all == sr) & (ar != r)).nonzero(as_tuple=True)[0]
        alt = int(same[int(torch.randint(0, len(same), (1,), generator=g))]) if len(same) else r
        cand = third[torch.randperm(len(third), generator=g)[:32]]
        vr = F.normalize(v16_tr[r].float().flatten(), dim=0)
        cosc = F.normalize(v16_tr[cand].float().flatten(1), dim=1) @ vr
        far = int(cand[int(cosc.argmin())])
        ic_list.append(c)
        ref_sel[COND_CORRECT].append(r)
        ref_sel[COND_WRONG].append(wrong)
        ref_sel[COND_SAME_STYLE].append(alt)
        ref_sel[COND_DIFF_STYLE].append(far)
    del fiber_enc

    dec = load_s10_decoder(device)
    stack = FeatureStack(device)
    gen_feats = {k: [] for k in CONDITION_KEYS}
    ref_feats = {k: [] for k in CONDITION_KEYS}
    content_feat = []
    for st in range(0, n_tasks, chunk):
        sl = slice(st, min(st + chunk, n_tasks))
        ic = torch.tensor(ic_list[sl])
        B = len(ic)
        zs = z_all[ic].to(device).float()
        eps = torch.randn(B, 4, 64, 64, generator=torch.Generator().manual_seed(seed + 100 + st)).to(device)
        z_init = (1.0 - tau0) * eps + tau0 * zs
        y16 = y16_all[ic].to(device).float()
        d16 = d16_all[ic].to(device).float()
        u_src = u_all[ic]
        rgb_c = decode_s10(dec, zs, batch=chunk)
        content_feat.append(stack.extract(_to_pils(rgb_c), batch=chunk)["dino"].cpu())
        for cond in CONDITION_KEYS:
            ir = torch.tensor(ref_sel[cond][sl])
            u_style = ad(u_all[ir], u_src)
            z_out = _oneshot_integrate(fm, z_init, y16, d16, u_style, fib_n_all[ir],
                                       d16_all[ir].to(device).float(), v16_tr[ir].to(device).float(),
                                       tau0, wj, device, steps)
            gen_feats[cond].append(stack.extract(_to_pils(decode_s10(dec, z_out, batch=chunk)), batch=chunk)["dino"].cpu())
            rgb_r = decode_s10(dec, z_all[ir].to(device).float(), batch=chunk)
            ref_feats[cond].append(stack.extract(_to_pils(rgb_r), batch=chunk)["dino"].cpu())
    del fm, ad, stack
    torch.cuda.empty_cache()

    gen_feats = {k: torch.cat(v, 0) for k, v in gen_feats.items()}
    ref_feats = {k: torch.cat(v, 0) for k, v in ref_feats.items()}
    table = eval_ref_conditions(gen_feats, ref_feats, torch.cat(content_feat, 0))
    smoke = check_ref_conditions_smoke(table)
    if not smoke["ok"]:
        print(f"[V25 C-ref-cond] WARNING smoke: {smoke['failures']}", flush=True)
        if strict:
            assert_ref_conditions_smoke(table)
    return {"table": table, "smoke": smoke, "n_tasks": n_tasks, "tau0": tau0, "wj": wj,
            "ckpt": str(ckpt_p).replace("\\", "/").split("runs/")[-1], "updated_utc": utc()}


# ------------------------------------------------ C2nt 无教师参考正则 ----

def c2nt_swap_active(lambda_sw: float, lambda_c: float) -> bool:
    """λ 双零 → swap 正则停用, C2nt 严格退化为 O3_self_only 配方 (纯自配对 flow-MSE)."""
    return (float(lambda_sw) > 0.0) or (float(lambda_c) > 0.0)


def c2nt_swap_losses(f_gen: dict, f_ref: dict, f_src: dict):
    """C2nt swap 正则逐项 (训练专用特征, dict 契约 {"style","content"}).

    返回 (l_style, l_cont, cos_style, cos_cont): 前两者为 (1−cos) 批均值标量 (梯度项),
    后两者为逐样本余弦 (B,) —— 训练期"编辑幅度-内容保持"曲线的诊断量
    (cos_cont = mid_cos 代理, 在所选训练特征空间; --feature-kind cnx 时即 ConvNeXt 余弦).
    """
    cos_style = F.cosine_similarity(_pooled(f_gen["style"]), _pooled(f_ref["style"]), dim=-1)
    cos_cont = F.cosine_similarity(_pooled(f_gen["content"]), _pooled(f_src["content"]), dim=-1)
    return (1.0 - cos_style).mean(), (1.0 - cos_cont).mean(), cos_style, cos_cont


def assign_c2nt_refs(micro: int, n_total: int, nbr_idx: torch.Tensor, sid_all: torch.Tensor,
                     generator: torch.Generator, *, n_group: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """C2nt 参考分配: 每样本一个参考 (跨风格邻居优先), 末 2·n_group 行 = 同内容双参考组.

    - 跨风格: 从该内容的 nbr_idx 行随机抽 4 列, 取首个 sid≠内容 的邻居
      (sid 为数据集元数据, 仅用于参考**抽样选择**, 非模型输入 —— 与四条件评测同性质;
      无合适跨风格邻居时回退首列);
    - 组内两参考强制互异 → batch 内 ≥2 不同参考为结构性保证;
    - 返回 (ic (micro,), ir (micro,)).
    """
    assert micro >= 2 * n_group + 1, "micro 须容纳 ≥1 单样本行 + 2·n_group 组行"
    nbr_w = nbr_idx.shape[1]

    def _cross_ref(content: int) -> int:
        cols = torch.randint(0, nbr_w, (4,), generator=generator).tolist()
        sc = int(sid_all[content])
        for c in cols:
            r = int(nbr_idx[content, c])
            if int(sid_all[r]) != sc:
                return r
        for c in range(nbr_w):  # 全行扫描兜底: 行内存在跨风格邻居则必取
            r = int(nbr_idx[content, c])
            if int(sid_all[r]) != sc:
                return r
        return int(nbr_idx[content, 0])

    n_single = micro - 2 * n_group
    ic_single = torch.randint(0, n_total, (n_single,), generator=generator)
    ic_parts, ir_parts = [ic_single], [torch.tensor([_cross_ref(int(i)) for i in ic_single.tolist()],
                                                    dtype=torch.long)]
    for _ in range(n_group):
        c = int(torch.randint(0, n_total, (1,), generator=generator))
        ra = _cross_ref(c)
        rb = ra
        for _ in range(8):
            rb = _cross_ref(c)
            if rb != ra:
                break
        if rb == ra:
            alt = [int(x) for x in nbr_idx[c].tolist() if int(x) != ra]
            rb = alt[0] if alt else (ra + 1) % n_total
        ic_parts.append(torch.tensor([c, c], dtype=torch.long))
        ir_parts.append(torch.tensor([ra, rb], dtype=torch.long))
    ic, ir = torch.cat(ic_parts), torch.cat(ir_parts)
    assert len(set(ir.tolist())) >= 2, "参考多样性保证失效"
    return ic, ir


def train_c2nt_arm(bundle, c21, v16_tr, arm, *, lambda_sw: float = 0.5, lambda_c: float = 1.0,
                   feature_kind: str = "vgg", updates: int = 1200, micro: int = MICRO_3060,
                   accum: int = 4, device: str = "cuda", ref_cache_dir: str | Path | None = None,
                   seed: int = 20265500, diag_every: int = 200):
    """方案③ C2nt: 无教师参考正则训练 (绕开教师 mid_cos 频带不可达问题).

    损失 = flow_MSE(自配对) + λ_sw·style(预览, r_i) + λ_c·content(预览, z_src):
    - flow-MSE: O3_self_only 配方 (z 目标 = z_src 恒等, τ 均匀采样, 内容锚 + same 语义);
    - swap 正则: 每 update 一次独立 pass 覆盖 micro 个样本 —— 一阶预览 z_init+v·(1−τ0)
      → 可微 S10 → 训练专用特征 (VGG/ConvNeXt), 逐样本拉向其跨风格邻居参考;
      同内容双参考组 ≥1 组/步 (结构保证 batch 内 ≥2 不同参考);
    - MSE 锚住恒等, swap 风格项提供"离开恒等"的梯度 —— 编辑幅度由 λ_sw 隐式决定,
      无教师、无频带问题;
    - 风格 dropout 关闭 (ds≡False): swap 项需要每个样本的风格条件速度, o3 的 12% dropout
      会清零被丢弃样本的风格梯度;
    - sid 仅用于参考抽样选择 (数据集元数据), 模型输入保持 label-free (onehot 槽位恒零).

    3060 12GB 显存预算: 每 update = 4×flow fwd/bwd (micro 12) + 1×swap pass
    (S10 可微解码 12×512² + 训练特征 fwd/bwd) → 1200 updates ≈ 25-40 min;
    OOM 时降 --micro 8 (swap 覆盖随之 8). λ 校准计划: 先 --updates 200 探针跑
    λ_sw ∈ {0.2, 0.5, 1.0} 看诊断的 preview_content_cos / preview_style_cos 曲线再定.

    返回 (ckpt Path, diag_rows list); 诊断 JSON 落 OUT/oneshot_val/<arm>_train_diag.json.
    """
    assert micro <= MICRO_3060, f"3060 12GB: flow micro ≤ {MICRO_3060} (train_o_arm 口径)"
    spec = O_SPECS[arm]
    out_p = OUT / "oneshot_val" / f"{arm}.pt"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if out_p.exists():
        print(f"[V25 {arm}] Reusing {out_p}", flush=True)
        return out_p, []
    swap_on = c2nt_swap_active(lambda_sw, lambda_c)

    seed_all(seed)
    tr = bundle["v18_cache"]["train"]
    fnd = bundle["fnd"]
    z_all = tr["z"]
    d16_all = tr["dino_16"]
    sid_all = tr["sid"].long()
    y16_all = coarse_y16_lcn(tr["y_64"].float()).half()
    nbr_idx = c21["train"]["nbr_idx"]

    fib_tr = fnd["tr_fib"].float()
    fib_mean = fib_tr.mean(0, keepdim=True).to(device)
    fib_std = fib_tr.std(0, keepdim=True).clamp_min(1e-4).to(device)
    fib_n_all = (fib_tr.to(device) - fib_mean) / fib_std
    fiber_enc = StyleFiberEncoder(in_dim=fib_tr.shape[1], dim=128).to(device).eval()
    fiber_enc.load_state_dict(fnd["fiber_state"])
    with torch.no_grad():
        u_all = fiber_enc.encode(fib_tr.to(device))
    del fiber_enc

    v_dim = 384 if spec["v_type"] == "dino" else 192
    fm = SLFUNetV24OneShot(v_dim=v_dim, use_conf_gate=spec["use_conf"], use_global_fib=spec["use_glob_fib"]).to(device)
    fm.load_state_dict(torch.load(DUAL_CKPT, map_location="cpu", weights_only=False)["ema"], strict=False)
    adapter = RefAdapter(dim=128).to(device)
    feature_fn = make_train_feature_fn(feature_kind, device)
    dec = load_s10_decoder(device)
    cache = ReferenceFeatureCache(ref_cache_dir if ref_cache_dir is not None
                                  else OUT / "cache/ref_feats_c2nt")

    for p in fm.parameters():
        p.requires_grad_(False)
    unfreeze = ("cond_mlp.", "dino_proj.", "mid1.", "mid_attn.", "mid2.", "dec32.", "dec64.", "out_norm.", "out_conv.")
    fm_p, new_p = [], list(adapter.parameters())
    for n, p in fm.named_parameters():
        if n.startswith("ref_attn16."):
            p.requires_grad_(True); new_p.append(p)
        elif n.startswith(unfreeze):
            p.requires_grad_(True); fm_p.append(p)
    opt = torch.optim.AdamW([{"params": new_p, "lr": 1.5e-4}, {"params": fm_p, "lr": 3e-5}], weight_decay=1e-3)
    base_lrs = [1.5e-4, 3e-5]
    ema_fm, ema_ad = copy.deepcopy(fm).eval(), copy.deepcopy(adapter).eval()
    for p in list(ema_fm.parameters()) + list(ema_ad.parameters()):
        p.requires_grad_(False)

    n_total = len(z_all)
    diag_rows = []
    t0 = time.time()
    for step in range(1, updates + 1):
        scale = 0.2 + 0.4 * (1.0 + math.cos(math.pi * step / max(1, updates)))
        for pg, blr in zip(opt.param_groups, base_lrs):
            pg["lr"] = blr * scale
        opt.zero_grad(set_to_none=True)
        loss_v = 0.0
        for _ in range(accum):  # ---- flow-MSE 自配对 (O3_self_only 配方) ----
            ic, ir = assign_c2nt_refs(micro, n_total, nbr_idx, sid_all,
                                      torch.Generator().manual_seed(seed * 7919 + step))
            z1 = z_all[ic].to(device).float()          # 自配对: 目标 = 源 (恒等)
            B = len(ic)
            d16 = d16_all[ic].to(device).float()
            d16_r = d16_all[ir].to(device).float()
            v16_r = v16_tr[ir].to(device).float()
            y16 = y16_all[ic].to(device).float()
            u_style = adapter(u_all[ir], u_all[ic])
            u_glob = fib_n_all[ir]
            d16_aug = d16 + 0.08 * torch.randn_like(d16)
            d16_aug = d16_aug - d16_aug.mean(dim=(2, 3), keepdim=True)
            y16_aug = (y16 * (0.85 + 0.30 * torch.rand(B, 1, 1, 1, device=device)) + 0.04 * torch.randn_like(y16)).clamp(-1.1, 1.1)
            slot = torch.zeros(B, dtype=torch.long, device=device)  # legacy onehot 槽位恒零
            z0 = torch.randn_like(z1)
            tau = torch.rand(B, device=device)
            zt = (1.0 - tau).view(B, 1, 1, 1) * z0 + tau.view(B, 1, 1, 1) * z1
            target_v = z1 - z0
            with autocast(device):
                pred_v = fm(zt, y16_aug, d16_aug, slot, u_style, tau,
                            drop_style=torch.zeros(B, dtype=torch.bool, device=device),
                            drop_content=torch.zeros(B, dtype=torch.bool, device=device),
                            layout_scale=torch.ones(B, device=device),
                            d16_ref=d16_r, v16_ref=v16_r, u_glob=u_glob)
                loss = F.mse_loss(pred_v.float(), target_v) / accum
            loss.backward()
            loss_v += float(loss.item())

        l_sw_v = l_c_v = 0.0
        if swap_on:  # ---- 参考正则 pass (每 update 一次, 覆盖 micro 个样本) ----
            ic_sw, ir_sw = assign_c2nt_refs(micro, n_total, nbr_idx, sid_all,
                                            torch.Generator().manual_seed(seed * 104729 + step))
            za = z_all[ic_sw].to(device).float()
            zr = z_all[ir_sw].to(device).float()
            S = len(za)
            d16_c = d16_all[ic_sw].to(device).float()
            y16_c = y16_all[ic_sw].to(device).float()
            u_src = u_all[ic_sw]
            tau0_s = 0.35 + 0.25 * torch.rand(S, device=device)
            z0_s = torch.randn_like(za)
            z_init = (1.0 - tau0_s).view(S, 1, 1, 1) * z0_s + tau0_s.view(S, 1, 1, 1) * za
            slot0 = torch.zeros(S, dtype=torch.long, device=device)
            d16_aug = d16_c + 0.08 * torch.randn_like(d16_c)
            d16_aug = d16_aug - d16_aug.mean(dim=(2, 3), keepdim=True)
            y16_aug = (y16_c * (0.85 + 0.30 * torch.rand(S, 1, 1, 1, device=device))
                       + 0.04 * torch.randn_like(y16_c)).clamp(-1.1, 1.1)
            d16_r_sw = d16_all[ir_sw].to(device).float()
            v16_r_sw = v16_tr[ir_sw].to(device).float()
            u_glob_sw = fib_n_all[ir_sw]
            # 分块(<=4)过 adapter+fm+可微S10+feature_fn 并逐块反传, 控制 3060 12GB 峰值显存且每块图独立
            sw_chunk = 4
            cos_st_parts, cos_ct_parts, en_parts = [], [], []
            for c0 in range(0, S, sw_chunk):
                c1 = min(c0 + sw_chunk, S)
                w_frac = float(c1 - c0) / float(S)
                u_style_c = adapter(u_all[ir_sw[c0:c1]], u_src[c0:c1])
                with autocast(device):
                    v_prev = fm(z_init[c0:c1], y16_aug[c0:c1], d16_aug[c0:c1], slot0[c0:c1],
                                u_style_c, tau0_s[c0:c1],
                                drop_style=torch.zeros(c1 - c0, dtype=torch.bool, device=device),
                                drop_content=torch.zeros(c1 - c0, dtype=torch.bool, device=device),
                                layout_scale=torch.ones(c1 - c0, device=device),
                                d16_ref=d16_r_sw[c0:c1],
                                v16_ref=v16_r_sw[c0:c1],
                                u_glob=u_glob_sw[c0:c1]).float()
                z_prev = z_init[c0:c1] + v_prev * (1.0 - tau0_s[c0:c1].view(-1, 1, 1, 1))
                rgb_prev = decode_s10(dec, z_prev.clamp(-5.0, 5.0), batch=c1 - c0, enable_grad=True, output_cpu=False)
                with torch.no_grad():
                    rgb_r = decode_s10(dec, zr[c0:c1], batch=c1 - c0, enable_grad=True, output_cpu=False)
                    rgb_s = decode_s10(dec, za[c0:c1], batch=c1 - c0, enable_grad=True, output_cpu=False)
                    with autocast(device):
                        f_ref = feature_fn(rgb_r)
                        f_src = feature_fn(rgb_s)
                with autocast(device):
                    f_gen = feature_fn(rgb_prev)
                l_style_c, l_cont_c, cos_st_c, cos_ct_c = c2nt_swap_losses(f_gen, f_ref, f_src)
                ((lambda_sw * l_style_c + lambda_c * l_cont_c) * w_frac).backward()
                l_sw_v += float(l_style_c.item()) * w_frac
                l_c_v += float(l_cont_c.item()) * w_frac
                cos_st_parts.append(cos_st_c.detach())
                cos_ct_parts.append(cos_ct_c.detach())
                en_parts.append((z_prev.detach() - za[c0:c1]).flatten(1).norm(dim=1))
            cos_style = torch.cat(cos_st_parts)
            cos_cont = torch.cat(cos_ct_parts)
            edit_norms = torch.cat(en_parts)
            if step == 1 or step % diag_every == 0 or step == updates:
                row = {"step": step,
                       "preview_content_cos": round(float(cos_cont.mean()), 5),
                       "preview_style_cos": round(float(cos_style.mean()), 5),
                       "content_cos_min": round(float(cos_cont.min()), 5),
                       "style_cos_max": round(float(cos_style.max()), 5),
                       "preview_edit_norm": round(float(edit_norms.mean()), 4),
                       "l_style": round(l_sw_v, 5), "l_cont": round(l_c_v, 5),
                       "lambda_sw": float(lambda_sw), "lambda_c": float(lambda_c)}
                diag_rows.append(row)
                print(f"[V25 {arm}] diag {row}", flush=True)

        all_p = [p for g_ in opt.param_groups for p in g_["params"]]
        nn.utils.clip_grad_norm_(all_p, 1.0)
        opt.step()
        with torch.no_grad():
            for pe, pw in zip(ema_fm.parameters(), fm.parameters()):
                pe.mul_(0.996).add_(pw, alpha=0.004)
            for pe, pw in zip(ema_ad.parameters(), adapter.parameters()):
                pe.mul_(0.996).add_(pw, alpha=0.004)

    torch.save({"fm": ema_fm.state_dict(), "adapter": ema_ad.state_dict(),
                "fib_mean": fib_mean.cpu(), "fib_std": fib_std.cpu(),
                "meta": {"lambda_sw": lambda_sw, "lambda_c": lambda_c, "feature_kind": feature_kind,
                         "micro": micro, "accum": accum, "updates": updates, "updated_utc": utc()}}, out_p)
    save_json(OUT / "oneshot_val" / f"{arm}_train_diag.json",
              {"arm": arm, "lambda_sw": lambda_sw, "lambda_c": lambda_c,
               "feature_kind": feature_kind, "rows": diag_rows, "updated_utc": utc()})
    print(f"[V25 {arm}] Saved {out_p} in {(time.time() - t0) / 60:.1f} min", flush=True)
    return out_p, diag_rows


# ----------------------------------------------------------- C2 训练 ----

def train_c2_arm(bundle, c21, cross_pack, v16_tr, arm, *, feature_kind: str = "vgg",
                 lambda_c: float = LAMBDA_C_DEFAULT, swap_weight: float = SWAP_WEIGHT_DEFAULT,
                 swap_micro: int = SWAP_MICRO_3060, updates: int = 1200, micro: int = MICRO_3060,
                 accum: int = 4, device: str = "cuda") -> Path:
    """C2 训练臂: O3 配方 flow-MSE (镜像 train_o_arm 采样/优化器/EMA) + 参考替换 swap 正则.

    显存预算 (3060 12GB): micro ≤ 12 (flow) + swap_micro ≤ 4 组 (= 8 张 512² 可微 S10 解码
    + 训练专用 VGG/ConvNeXt 特征); swap 常量侧 (参考/内容特征) 在 no_grad 下预算, 仅学生
    输出分支带梯度. swap 组来自教师包 swap_groups (同一内容的两个 kept 参考).
    """
    assert micro <= MICRO_3060, f"3060 12GB: flow micro ≤ {MICRO_3060} (train_o_arm 口径)"
    assert swap_micro <= SWAP_MICRO_3060, f"3060 12GB: swap_micro ≤ {SWAP_MICRO_3060}"
    spec = O_SPECS[arm]
    out_p = OUT / "oneshot_val" / f"{arm}.pt"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if out_p.exists():
        print(f"[V25 {arm}] Reusing {out_p}", flush=True)
        return out_p

    seed_all(20265200)
    tr = bundle["v18_cache"]["train"]
    fnd = bundle["fnd"]
    z_all = tr["z"]
    d16_all = tr["dino_16"]
    y16_all = coarse_y16_lcn(tr["y_64"].float()).half()
    nbr_idx = c21["train"]["nbr_idx"]

    fib_tr = fnd["tr_fib"].float()
    fib_mean = fib_tr.mean(0, keepdim=True).to(device)
    fib_std = fib_tr.std(0, keepdim=True).clamp_min(1e-4).to(device)
    fib_n_all = (fib_tr.to(device) - fib_mean) / fib_std
    fiber_enc = StyleFiberEncoder(in_dim=fib_tr.shape[1], dim=128).to(device).eval()
    fiber_enc.load_state_dict(fnd["fiber_state"])
    with torch.no_grad():
        u_all = fiber_enc.encode(fib_tr.to(device))
    del fiber_enc

    v_dim = 384 if spec["v_type"] == "dino" else 192
    fm = SLFUNetV24OneShot(v_dim=v_dim, use_conf_gate=spec["use_conf"], use_global_fib=spec["use_glob_fib"]).to(device)
    fm.load_state_dict(torch.load(DUAL_CKPT, map_location="cpu", weights_only=False)["ema"], strict=False)
    adapter = RefAdapter(dim=128).to(device)
    feature_fn = make_train_feature_fn(feature_kind, device)
    dec = load_s10_decoder(device)

    for p in fm.parameters():
        p.requires_grad_(False)
    unfreeze = ("cond_mlp.", "dino_proj.", "mid1.", "mid_attn.", "mid2.", "dec32.", "dec64.", "out_norm.", "out_conv.")
    fm_p, new_p = [], list(adapter.parameters())
    for n, p in fm.named_parameters():
        if n.startswith("ref_attn16."):
            p.requires_grad_(True); new_p.append(p)
        elif n.startswith(unfreeze):
            p.requires_grad_(True); fm_p.append(p)
    opt = torch.optim.AdamW([{"params": new_p, "lr": 1.5e-4}, {"params": fm_p, "lr": 3e-5}], weight_decay=1e-3)
    base_lrs = [1.5e-4, 3e-5]
    ema_fm, ema_ad = copy.deepcopy(fm).eval(), copy.deepcopy(adapter).eval()
    for p in list(ema_fm.parameters()) + list(ema_ad.parameters()):
        p.requires_grad_(False)

    swap_groups = cross_pack.get("swap_groups")
    has_swap = swap_weight > 0 and swap_groups is not None and len(swap_groups) > 0
    if swap_weight > 0 and not has_swap:
        print(f"[V25 {arm}] WARNING: swap_groups 为空, swap 项停用 (纯 C1 口径)", flush=True)
    n_cr_pool = len(cross_pack["idx_c"])

    t0 = time.time()
    for step in range(1, updates + 1):
        scale = 0.2 + 0.4 * (1.0 + math.cos(math.pi * step / max(1, updates)))
        for pg, blr in zip(opt.param_groups, base_lrs):
            pg["lr"] = blr * scale
        opt.zero_grad(set_to_none=True)
        loss_v = 0.0
        for _ in range(accum):  # ---- flow-MSE (O3 配方) ----
            n_cr = micro // 2
            n_self = micro - n_cr
            idx_s = torch.randint(0, len(z_all), (n_self,))
            perm_nbr = torch.rand(n_self, 12).topk(1, dim=1).indices
            ref_s = nbr_idx[idx_s[:, None], perm_nbr].squeeze(1)
            pick = torch.randint(0, n_cr_pool, (n_cr,))
            ic = torch.cat([idx_s, cross_pack["idx_c"][pick]], 0)
            ir = torch.cat([ref_s, cross_pack["idx_r"][pick]], 0)
            z1 = torch.cat([z_all[idx_s].to(device).float(), cross_pack["z_cr"][pick].to(device).float()], 0)
            B = len(ic)
            d16 = d16_all[ic].to(device).float()
            d16_r = d16_all[ir].to(device).float()
            v16_r = v16_tr[ir].to(device).float()
            y16 = y16_all[ic].to(device).float()
            u_style = adapter(u_all[ir], u_all[ic])
            u_glob = fib_n_all[ir]
            d16_aug = d16 + 0.08 * torch.randn_like(d16)
            d16_aug = d16_aug - d16_aug.mean(dim=(2, 3), keepdim=True)
            y16_aug = (y16 * (0.85 + 0.30 * torch.rand(B, 1, 1, 1, device=device)) + 0.04 * torch.randn_like(y16)).clamp(-1.1, 1.1)
            ds = torch.rand(B, device=device) < 0.12
            dc = torch.zeros(B, dtype=torch.bool, device=device)
            lscale = torch.ones(B, device=device)
            slot = torch.zeros(B, dtype=torch.long, device=device)  # legacy onehot 槽位恒零
            z0 = torch.randn_like(z1)
            za = z_all[ic].to(device).float()
            tau_fm = torch.rand(B, device=device)
            zt_fm = (1.0 - tau_fm).view(B, 1, 1, 1) * z0 + tau_fm.view(B, 1, 1, 1) * z1
            v_fm = z1 - z0
            tau0 = 0.35 + 0.25 * torch.rand(B, device=device)
            z_init = (1.0 - tau0).view(B, 1, 1, 1) * z0 + tau0.view(B, 1, 1, 1) * za
            tau_ab = tau0 + (1.0 - tau0) * torch.rand(B, device=device)
            frac = ((tau_ab - tau0) / (1.0 - tau0).clamp_min(1e-4)).view(B, 1, 1, 1)
            zt_ab = z_init + frac * (z1 - z_init)
            v_ab = (z1 - z_init) / (1.0 - tau0).clamp_min(1e-4).view(B, 1, 1, 1)
            is_ab = torch.zeros(B, dtype=torch.bool, device=device)
            is_ab[n_self:] = True
            m4 = is_ab.view(B, 1, 1, 1)
            tau = torch.where(is_ab, tau_ab, tau_fm)
            zt = torch.where(m4, zt_ab, zt_fm)
            target_v = torch.where(m4, v_ab, v_fm)
            with autocast(device):
                pred_v = fm(zt, y16_aug, d16_aug, slot, u_style, tau,
                            drop_style=ds, drop_content=dc, layout_scale=lscale,
                            d16_ref=d16_r, v16_ref=v16_r, u_glob=u_glob)
                loss = F.mse_loss(pred_v.float(), target_v) / accum
            loss.backward()
            loss_v += float(loss.item())

        l_swap_v = 0.0
        if has_swap:  # ---- 参考替换一致性 (一阶预览 + 可微 S10 + 训练专用特征) ----
            gi = torch.randint(0, len(swap_groups), (swap_micro,))
            gsel = swap_groups[gi]
            ic_g = cross_pack["idx_c"][gsel[:, 0]]
            ir_a = cross_pack["idx_r"][gsel[:, 0]]
            ir_b = cross_pack["idx_r"][gsel[:, 1]]
            za = z_all[ic_g].to(device).float()
            S = len(za)
            d16_c = d16_all[ic_g].to(device).float()
            y16_c = y16_all[ic_g].to(device).float()
            u_src = u_all[ic_g]
            u_a = adapter(u_all[ir_a], u_src)
            u_b = adapter(u_all[ir_b], u_src)
            tau0_s = 0.35 + 0.25 * torch.rand(S, device=device)
            z0_s = torch.randn_like(za)
            z_init_s = (1.0 - tau0_s).view(S, 1, 1, 1) * z0_s + tau0_s.view(S, 1, 1, 1) * za
            rep2 = torch.cat([z_init_s, z_init_s], 0)
            tau_rep = torch.cat([tau0_s, tau0_s], 0)
            slot2 = torch.zeros(2 * S, dtype=torch.long, device=device)
            ds2 = torch.zeros(2 * S, dtype=torch.bool, device=device)
            dc2 = torch.zeros(2 * S, dtype=torch.bool, device=device)
            l2 = torch.ones(2 * S, device=device)
            d16_aug = d16_c.repeat(2, 1, 1, 1) + 0.08 * torch.randn_like(d16_c.repeat(2, 1, 1, 1))
            d16_aug = d16_aug - d16_aug.mean(dim=(2, 3), keepdim=True)
            y16_aug = (y16_c.repeat(2, 1, 1, 1) * (0.85 + 0.30 * torch.rand(2 * S, 1, 1, 1, device=device))
                       + 0.04 * torch.randn_like(y16_c.repeat(2, 1, 1, 1))).clamp(-1.1, 1.1)
            with autocast(device):
                v_rep = fm(rep2, y16_aug, d16_aug, slot2, torch.cat([u_a, u_b], 0), tau_rep,
                           drop_style=ds2, drop_content=dc2, layout_scale=l2,
                           d16_ref=torch.cat([d16_all[ir_a].to(device).float(), d16_all[ir_b].to(device).float()], 0),
                           v16_ref=torch.cat([v16_tr[ir_a].to(device).float(), v16_tr[ir_b].to(device).float()], 0),
                           u_glob=torch.cat([fib_n_all[ir_a], fib_n_all[ir_b]], 0)).float()
            z_prev = rep2 + v_rep * (1.0 - tau_rep.view(-1, 1, 1, 1))
            # 学生输出分支: 可微 S10 解码 (enable_grad 保留输入梯度路径, 不解冻权重)
            rgb_prev = decode_s10(dec, z_prev.clamp(-5.0, 5.0), batch=swap_micro, enable_grad=True, output_cpu=False)
            with torch.no_grad():
                # 常量侧 (参考/内容) 特征: enable_grad=True 使 decode 产出普通张量 (非 inference
                # tensor), 可安全参与反向保存; 显存与精度一致性: 特征提取同样走 autocast.
                rgb_ra = decode_s10(dec, z_all[ir_a].to(device).float(), batch=swap_micro,
                                    enable_grad=True, output_cpu=False)
                rgb_rb = decode_s10(dec, z_all[ir_b].to(device).float(), batch=swap_micro,
                                    enable_grad=True, output_cpu=False)
                rgb_c = decode_s10(dec, za, batch=swap_micro, enable_grad=True, output_cpu=False)
                with autocast(device):
                    f_ra = feature_fn(rgb_ra)
                    f_rb = feature_fn(rgb_rb)
                    f_c = feature_fn(rgb_c)
            with autocast(device):
                f_gen_a = feature_fn(rgb_prev[:S])
                f_gen_b = feature_fn(rgb_prev[S:])
            l_swap = swap_loss_features(f_gen_a, f_gen_b, f_ra, f_rb, f_c, lambda_c) / accum
            (l_swap * swap_weight).backward()
            l_swap_v = float(l_swap.item()) * swap_weight

        all_p = [p for g_ in opt.param_groups for p in g_["params"]]
        nn.utils.clip_grad_norm_(all_p, 1.0)
        opt.step()
        with torch.no_grad():
            for pe, pw in zip(ema_fm.parameters(), fm.parameters()):
                pe.mul_(0.996).add_(pw, alpha=0.004)
            for pe, pw in zip(ema_ad.parameters(), adapter.parameters()):
                pe.mul_(0.996).add_(pw, alpha=0.004)
        if step == 1 or step % 400 == 0 or step == updates:
            print(f"[V25 {arm}] {step:4d}/{updates} loss={loss_v:.4f} swap={l_swap_v:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    torch.save({"fm": ema_fm.state_dict(), "adapter": ema_ad.state_dict(),
                "fib_mean": fib_mean.cpu(), "fib_std": fib_std.cpu(),
                "meta": {"feature_kind": feature_kind, "lambda_c": lambda_c,
                         "swap_weight": swap_weight, "swap_micro": swap_micro,
                         "micro": micro, "accum": accum, "updated_utc": utc()}}, out_p)
    print(f"[V25 {arm}] Saved {out_p} in {(time.time() - t0) / 60:.1f} min", flush=True)
    return out_p


# ------------------------------------------------------------- 阶段 ----

class _TrainEnv(SimpleNamespace):
    pass


def _redirect_v24_out() -> None:
    """r3_oneshot_minimal 的 OUT 重定向手法: 之后 exp.v24 写路径落 runs/v25."""
    _v24.OUT = OUT
    _o_mod.OUT = OUT


def _load_train_env(args) -> _TrainEnv:
    bundle = build_or_load_v19_foundation(args.pool, device=args.device)
    c21 = torch.load(ROOT / "runs/v21/cache/v21_stiefel_ref_cache.pt", map_location="cpu", weights_only=False)
    dec = load_s10_decoder(args.device)
    cnx = load_convnext_tiny_offline(args.device).eval()
    for p in cnx.parameters():
        p.requires_grad_(False)
    v16_tr = build_or_load_train_cnx_v16(bundle, dec, cnx, device=args.device)  # 复用 runs/v24 缓存
    _redirect_v24_out()  # 与 r3 同序: 缓存读 runs/v24, 之后训练/评测写 runs/v25
    return _TrainEnv(bundle=bundle, c21=c21, dec=dec, cnx=cnx, v16_tr=v16_tr)


def _build_pack(args, w: float, fixed_pairs=None) -> dict:
    return build_teacher_pairs(
        w, pool=args.pool, device=args.device, n_cand=args.n_cand, n_keep=args.n_keep,
        steps=args.steps, mid_cos_band=(args.mid_cos_lo, args.mid_cos_hi),
        style_gain_min=args.style_gain_min, fixed_pairs=fixed_pairs, force=args.force)


def _register_arm(arm: str) -> None:
    _o_mod.O_SPECS[arm] = copy.deepcopy(_o_mod.O_SPECS["O3_rgb_cnx_gated"])


def run_stage_c0(args, env) -> dict:
    """C0 基线: 只评已有 O3 (τ45/55 + 四条件钩子), 零训练."""
    O3_CKPT, eval_o_tau_list = _lazy_r3()
    rows = eval_o_tau_list(env.bundle, env.c21, env.cnx, O3_CKPT, "O3_rgb_cnx_gated",
                           [(0.45, 2.2, "t45"), (0.55, 2.0, "t55")], device=args.device)
    ref_cond = run_ref_condition_eval(env.bundle, env.c21, env.cnx, env.v16_tr, O3_CKPT,
                                      n_tasks=args.n_tasks, device=args.device)
    return {"stage": "C0", "o3_tau": rows, "ref_conditions": ref_cond}


def run_stage_c1(args, env) -> dict:
    """C1: 校准教师包 (content_w=--teacher-w) → train_o_arm (O3 配方, OUT 重定向 runs/v25)."""
    w = float(args.teacher_w)
    if w not in TEACHER_W_GRID:
        print(f"[V25 C1] WARNING: teacher_w={w} 非校准三档 {TEACHER_W_GRID} (需 R2 诊断支撑)", flush=True)
    pack = _build_pack(args, w)
    arm = f"C1_{teacher_tag(w)}"
    _register_arm(arm)
    _, eval_o_tau_list = _lazy_r3()
    ckpt = train_o_arm(env.bundle, env.c21, pack, env.v16_tr, arm, updates=args.updates, device=args.device)
    rows = eval_o_tau_list(env.bundle, env.c21, env.cnx, ckpt, arm,
                           [(0.45, 2.2, "t45"), (0.55, 2.0, "t55")], device=args.device)
    ref_cond = run_ref_condition_eval(env.bundle, env.c21, env.cnx, env.v16_tr, ckpt,
                                      n_tasks=args.n_tasks, device=args.device)
    return {"stage": "C1", "arm": arm, "teacher_w": w, "n_pairs": int(len(pack["idx_c"])),
            "swap_groups_kept": int(pack["swap_groups"].shape[0]), "o3_tau": rows,
            "ref_conditions": ref_cond}


def run_stage_c2(args, env) -> dict:
    """C2: C1 + 参考替换 swap 损失 (训练专用特征 fn, 评测裁判禁用)."""
    w = float(args.teacher_w)
    pack = _build_pack(args, w)
    arm = f"C2_{teacher_tag(w)}"
    _register_arm(arm)
    _, eval_o_tau_list = _lazy_r3()
    ckpt = train_c2_arm(env.bundle, env.c21, pack, env.v16_tr, arm,
                        feature_kind=args.feature_kind,
                        lambda_c=(LAMBDA_C_DEFAULT if args.lambda_c is None else args.lambda_c),
                        swap_weight=args.swap_weight, swap_micro=args.swap_micro,
                        updates=args.updates, device=args.device)
    rows = eval_o_tau_list(env.bundle, env.c21, env.cnx, ckpt, arm,
                           [(0.45, 2.2, "t45"), (0.55, 2.0, "t55")], device=args.device)
    ref_cond = run_ref_condition_eval(env.bundle, env.c21, env.cnx, env.v16_tr, ckpt,
                                      n_tasks=args.n_tasks, device=args.device)
    return {"stage": "C2", "arm": arm, "teacher_w": w, "feature_kind": args.feature_kind,
            "lambda_c": (LAMBDA_C_DEFAULT if args.lambda_c is None else args.lambda_c),
            "swap_weight": args.swap_weight,
            "n_pairs": int(len(pack["idx_c"])),
            "swap_groups_kept": int(pack["swap_groups"].shape[0]),
            "o3_tau": rows, "ref_conditions": ref_cond}


def run_stage_c3(args, env) -> dict:
    """C3: 连续教师强度混合 γ_t ∈ {0.4,0.7,1.0} (保守 2.2 × 表现 0.7 逐对混合, 架构零改动)."""
    pack_cons = _build_pack(args, 2.2)
    pack_expr = _build_pack(args, 0.7, fixed_pairs=(pack_cons["idx_c_all"], pack_cons["idx_r_all"]))
    pack = build_strength_mix_pack(pack_cons, pack_expr, gammas=C3_GAMMA_POOL)
    print(f"[V25 C3] mixed pairs={pack['n_pairs']} (cons-only={pack['n_cons_only']}, "
          f"expr-only={pack['n_expr_only']})", flush=True)
    arm = "C3_mix"
    _register_arm(arm)
    _, eval_o_tau_list = _lazy_r3()
    ckpt = train_o_arm(env.bundle, env.c21, pack, env.v16_tr, arm, updates=args.updates, device=args.device)
    rows = eval_o_tau_list(env.bundle, env.c21, env.cnx, ckpt, arm,
                           [(0.45, 2.2, "t45"), (0.55, 2.0, "t55")], device=args.device)
    ref_cond = run_ref_condition_eval(env.bundle, env.c21, env.cnx, env.v16_tr, ckpt,
                                      n_tasks=args.n_tasks, device=args.device)
    return {"stage": "C3", "arm": arm, "gammas": list(C3_GAMMA_POOL), "n_pairs": pack["n_pairs"],
            "o3_tau": rows, "ref_conditions": ref_cond}


def run_stage_c2nt(args, env) -> dict:
    """方案③ C2nt: 无教师参考正则训练 (自配对 flow-MSE + λ_sw·swap_style + λ_c·content).

    不加载任何教师包 (接口与实现均无 pack); 判读框架: t45/t55 vs C0/O3 与 O3_self_only ——
    Top-1 ≥ 20% 且 DINO-S ≥ 0.48 → 方案③成立.
    """
    lambda_c = 1.0 if args.lambda_c is None else args.lambda_c
    arm = f"C2nt_sw{args.lambda_sw:g}_lc{lambda_c:g}"
    _register_arm(arm)
    ckpt, diag = train_c2nt_arm(env.bundle, env.c21, env.v16_tr, arm,
                                lambda_sw=args.lambda_sw, lambda_c=lambda_c,
                                feature_kind=args.feature_kind, updates=args.updates,
                                device=args.device)
    _, eval_o_tau_list = _lazy_r3()
    rows = eval_o_tau_list(env.bundle, env.c21, env.cnx, ckpt, arm,
                           [(0.45, 2.2, "t45"), (0.55, 2.0, "t55")], device=args.device)
    ref_cond = run_ref_condition_eval(env.bundle, env.c21, env.cnx, env.v16_tr, ckpt,
                                      n_tasks=args.n_tasks, device=args.device)
    return {"stage": "C2nt", "arm": arm, "lambda_sw": args.lambda_sw, "lambda_c": lambda_c,
            "feature_kind": args.feature_kind, "train_diag": diag,
            "o3_tau": rows, "ref_conditions": ref_cond}


_STAGES = {"C0": run_stage_c0, "C1": run_stage_c1, "C2": run_stage_c2, "C3": run_stage_c3,
           "C2nt": run_stage_c2nt}
_STAGES_BY_LOWER = {k.lower(): k for k in _STAGES}


def run_stage(stage: str, args) -> dict:
    """阶段分发入口 (run_stage 前机械自检接口标签自由)."""
    assert_label_free_interfaces()
    s = _STAGES_BY_LOWER.get(str(stage).strip().lower())
    if s is None:
        raise ValueError(f"未知阶段 {stage} (允许: {sorted(_STAGES)})")
    env = _load_train_env(args)
    result = _STAGES[s](args, env)
    result["updated_utc"] = utc()
    result["args"] = {k: (v if isinstance(v, (int, float, str, bool, list)) else str(v))
                      for k, v in vars(args).items()}
    save_json(OUT / f"mechanism_c_{s.lower()}.json", result)
    print(f"\n=== V25 机制 C {s} 完成 → {OUT / f'mechanism_c_{s.lower()}.json'} ===", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=tuple(_STAGES))
    ap.add_argument("--teacher-w", type=float, default=1.2, help="教师内容权重 {2.2, 1.2, 0.7}")
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--updates", type=int, default=1200)
    ap.add_argument("--n-cand", type=int, default=2400)
    ap.add_argument("--n-keep", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--mid-cos-lo", type=float, default=MID_COS_BAND[0])
    ap.add_argument("--mid-cos-hi", type=float, default=MID_COS_BAND[1])
    ap.add_argument("--style-gain-min", type=float, default=STYLE_GAIN_MIN)
    ap.add_argument("--feature-kind", default="vgg", choices=TRAIN_FEATURE_KINDS)
    ap.add_argument("--lambda-sw", type=float, default=0.5,
                    help="C2nt: swap 风格项权重 (校准: --updates 200 探针 × {0.2, 0.5, 1.0})")
    ap.add_argument("--lambda-c", type=float, default=None,
                    help="内容保持项权重 (C2 swap 项默认 0.5 / C2nt 预览项默认 1.0; 缺省按阶段解析)")
    ap.add_argument("--swap-weight", type=float, default=SWAP_WEIGHT_DEFAULT)
    ap.add_argument("--swap-micro", type=int, default=SWAP_MICRO_3060)
    ap.add_argument("--n-tasks", type=int, default=48)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    sanitize_cuda_env()
    run_stage(args.stage, args)


if __name__ == "__main__":
    main()
