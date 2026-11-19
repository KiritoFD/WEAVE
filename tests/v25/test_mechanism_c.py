"""V25 B02: 机制 C 教师与参考替换 Loss 修正 —— CPU 单测 (mock 特征 / 小张量 / 禁 GPU / 禁真实 ckpt).

覆盖 (agent2.md §三 B02 验收逐条):
  1. 风格项 (1 − cos) 方向正确: 同特征→0 / 正交→1 / 反向→2;
     "(1−cos) 写法" 与 "直接最小化 cos 写法" 对输入的梯度逐元素相反, GD 后余弦一升一降;
  2. L_swap = style(G_a,r_a) + style(G_b,r_b) + λ_c[L_c(G_a,x_c)+L_c(G_b,x_c)],
     λ_c 默认 0.5 可参数化; 两输出分别受内容约束; 内容漂移方向数值梯度 sanity;
  3. 四条件参考评测钩子 eval_ref_conditions: mock 特征跑通; 错误参考 smoke 断言
     (S1 正确参考得分反超 / S2 自身参考得分反超) 两条触发路径 + 通过路径;
  4. 无画派 ID: inspect 断言训练/评测接口签名无 sid/onehot/目标类别参数;
     并实证复用模型 SLFUNetV24OneShot 的 legacy style_id 槽位对输出零影响 (torch.equal);
  5. 参考特征缓存: 内容 sha 键稳定, 同参考重复编码直接命中 (tmp_path);
  6. 双指标筛选全保留/全拒绝边界 + NaN 拒绝 + 闭区间端点.

另: C3 强度混合公式与配对对齐; 训练特征 fn 拒绝评测裁判 (DINO/CSD/CLIP/LPIPS).
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # 必须在 import torch 之前 → 纯 CPU
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import inspect

import pytest
import torch
import torch.nn.functional as F

from exp.v25.mechanism_c import (
    C3_GAMMA_POOL,
    CONDITION_KEYS,
    COND_CORRECT,
    COND_DIFF_STYLE,
    COND_SAME_STYLE,
    COND_WRONG,
    FORBIDDEN_TRAIN_FEATURES,
    LAMBDA_C_DEFAULT,
    MICRO_3060,
    SWAP_MICRO_3060,
    SWAP_WEIGHT_DEFAULT,
    TRAIN_FEATURE_KINDS,
    assert_label_free_interface,
    assert_label_free_interfaces,
    assert_ref_conditions_smoke,
    assert_train_feature_kind,
    build_strength_mix_pack,
    check_ref_conditions_smoke,
    content_cos_loss,
    cosine_distance_loss,
    eval_ref_conditions,
    make_train_feature_fn,
    mix_teacher_strengths,
    style_cos_loss,
    swap_loss,
    swap_loss_features,
    swap_loss_images,
    train_c2_arm,
)
from exp.v25.reference_teacher_v2 import (
    MID_COS_BAND,
    STYLE_GAIN_MIN,
    TEACHER_W_GRID,
    TEACHER_W_TAGS,
    ReferenceFeatureCache,
    batch_cached_ref_tokens,
    build_teacher_pairs,
    bytes_sha,
    dual_metric_filter,
    family_retention_report,
    filter_diagnostics,
    pooled_style_cos,
    tensor_sha,
    teacher_tag,
)

torch.manual_seed(0)


# ---------------- 验收 1: (1 − cos) 方向与数值边界 ----------------

def test_cosine_distance_bounds_and_4d_pooling():
    a = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    same = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    ortho = torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    oppo = torch.tensor([[-1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]])
    assert float(cosine_distance_loss(a, same)) <= 1e-6
    assert abs(float(cosine_distance_loss(a, ortho)) - 1.0) <= 1e-6
    assert abs(float(cosine_distance_loss(a, oppo)) - 2.0) <= 1e-6
    # 数值边界: 任意特征对 loss ∈ [0, 2]
    x = torch.randn(8, 16)
    y = torch.randn(8, 16)
    l = float(cosine_distance_loss(x, y))
    assert 0.0 <= l <= 2.0
    # 4D 输入 → 空间均值池化后与手工池化 2D 等价
    xa = torch.randn(2, 8, 4, 4)
    xb = torch.randn(2, 8, 4, 4)
    manual = (1.0 - F.cosine_similarity(xa.mean(dim=(2, 3)), xb.mean(dim=(2, 3)), dim=-1)).mean()
    assert torch.allclose(cosine_distance_loss(xa, xb), manual, atol=1e-6)


def test_one_minus_cos_gradient_direction_opposite_to_cos_form():
    """验收 1 核心: (1−cos) 写法与'直接最小化 cos'写法的梯度逐元素相反; GD 后余弦一升一降."""
    torch.manual_seed(7)
    f_b = torch.randn(4, 8)
    f_a0 = torch.randn(4, 8)

    def grad_at(loss_fn):
        f_a = f_a0.clone().requires_grad_(True)
        loss_fn(f_a).backward()
        return f_a.grad.detach().clone()

    g_right = grad_at(lambda fa: style_cos_loss(fa, f_b))                          # (1 − cos)
    g_wrong = grad_at(lambda fa: F.cosine_similarity(fa, f_b, dim=-1).mean())      # 直接 cos
    assert torch.allclose(g_right, -g_wrong, atol=1e-6), "两种写法梯度必须逐元素相反"

    # GD 行为验证: 最小化 (1−cos) → cos 上升; 最小化 cos → cos 下降 (方向相反)
    def gd(loss_fn, steps=30, lr=0.05):
        f_a = f_a0.clone().requires_grad_(True)
        c0 = float(F.cosine_similarity(f_a, f_b, dim=-1).mean())
        for _ in range(steps):
            g = torch.autograd.grad(loss_fn(f_a).sum(), f_a)[0]
            f_a = (f_a.detach() - lr * g).requires_grad_(True)
        return c0, float(F.cosine_similarity(f_a, f_b, dim=-1).mean())

    c0_r, c1_r = gd(lambda fa: style_cos_loss(fa, f_b))
    c0_w, c1_w = gd(lambda fa: F.cosine_similarity(fa, f_b, dim=-1).mean())
    assert c1_r > c0_r, "(1−cos) 最小化必须把余弦推向 +1"
    assert c1_w < c0_w, "直接最小化 cos 会把余弦推向 −1 (错误方向, 禁止)"


def test_style_and_content_names_share_math():
    x = torch.randn(3, 8)
    y = torch.randn(3, 8)
    assert torch.allclose(style_cos_loss(x, y), cosine_distance_loss(x, y))
    assert torch.allclose(content_cos_loss(x, y), cosine_distance_loss(x, y))


# ---------------- 验收 2: swap 损失 ----------------

def _swap_fixture(D=8, B=3, seed=5):
    g = torch.Generator().manual_seed(seed)
    mk = lambda: torch.randn(B, D, generator=g)
    return dict(f_a=mk(), f_b=mk(), f_ra=mk(), f_rb=mk(), f_c=mk())


def test_swap_loss_lambda_default_and_linearity():
    """λ_c 默认 0.5 (参数化); λ=0 退化纯风格项; loss 对 λ 线性: L(λ)=style_sum+λ·content_sum."""
    sig = inspect.signature(swap_loss)
    assert sig.parameters["lambda_c"].default == LAMBDA_C_DEFAULT == 0.5
    fx = _swap_fixture()
    style_sum = (cosine_distance_loss(fx["f_a"], fx["f_ra"]) + cosine_distance_loss(fx["f_b"], fx["f_rb"]))
    content_sum = (cosine_distance_loss(fx["f_a"], fx["f_c"]) + cosine_distance_loss(fx["f_b"], fx["f_c"]))
    assert torch.allclose(swap_loss(**fx, lambda_c=0.0), style_sum, atol=1e-6)
    for lam in (0.25, 0.5, 1.0):
        assert torch.allclose(swap_loss(**fx, lambda_c=lam), style_sum + lam * content_sum, atol=1e-5)


def test_swap_loss_content_drift_penalty_direction():
    """数值梯度 sanity: 输出特征向内容移动 → loss 降; 远离内容 → loss 升 (两个输出各自如此).

    控制世界: 风格目标已满足 (f_ra=f_a, f_rb=f_b) → 风格项梯度为零 (cos=1 处二阶),
    内容项一阶主导 → 符号确定性; 风格方向单独在 λ_c=0 下验证 (沿 (ref−out) 方向
    d(1−cos)/dε = −|ref|(1−cos²)/|out| ≤ 0 严格解析负, 见 test_swap_loss_style_direction).
    """
    torch.manual_seed(11)
    f_c = torch.randn(2, 8)
    f_a = torch.randn(2, 8)
    f_b = torch.randn(2, 8)
    world = dict(f_a=f_a, f_b=f_b, f_ra=f_a.clone(), f_rb=f_b.clone(), f_c=f_c)
    base = float(swap_loss(**world))
    for key, v0 in (("f_a", f_a), ("f_b", f_b)):
        to_c = F.normalize(f_c - v0, dim=-1)
        for eps in (0.01, 0.05, 0.10):
            assert float(swap_loss(**{**world, key: v0 + eps * to_c})) < base, \
                f"{key} 向内容移动必须降低 loss"
            assert float(swap_loss(**{**world, key: v0 - eps * to_c})) > base, \
                f"{key} 远离内容必须抬高 loss (内容漂移惩罚)"


def test_swap_loss_style_direction():
    """风格项方向 (λ_c=0 隔离): 沿 (ref−out) 方向 (1−cos) 严格下降, 反向严格上升."""
    torch.manual_seed(13)
    f_a, f_b = torch.randn(2, 8), torch.randn(2, 8)
    f_ra, f_rb, f_c = torch.randn(2, 8), torch.randn(2, 8), torch.randn(2, 8)
    base = float(swap_loss(f_a, f_b, f_ra, f_rb, f_c, lambda_c=0.0))
    toward_a = f_a + 0.10 * F.normalize(f_ra - f_a, dim=-1)
    toward_b = f_b + 0.10 * F.normalize(f_rb - f_b, dim=-1)
    away_a = f_a - 0.10 * F.normalize(f_ra - f_a, dim=-1)
    assert float(swap_loss(toward_a, f_b, f_ra, f_rb, f_c, lambda_c=0.0)) < base
    assert float(swap_loss(f_a, toward_b, f_ra, f_rb, f_c, lambda_c=0.0)) < base
    assert float(swap_loss(away_a, f_b, f_ra, f_rb, f_c, lambda_c=0.0)) > base


def test_swap_loss_gradient_flows_to_both_outputs():
    fx = {k: (v.clone().requires_grad_(True) if k in ("f_a", "f_b") else v.clone())
          for k, v in _swap_fixture().items()}
    swap_loss(**fx).backward()
    for key in ("f_a", "f_b"):
        assert fx[key].grad is not None and float(fx[key].grad.abs().sum()) > 0, \
            f"{key} (生成输出) 必须有非零梯度 (两输出分别受风格+内容约束)"


def test_swap_loss_images_mock_feature_fn():
    calls = []

    def mock_fn(rgb):
        calls.append(tuple(rgb.shape))
        g = torch.Generator().manual_seed(int(rgb.sum().item() * 100) % (2 ** 31))
        base = torch.randn(2, 12, generator=g)
        return {"style": base[:, :6], "content": base[:, 6:]}

    B = 2
    rgb = [torch.rand(B, 3, 8, 8) for _ in range(5)]  # a, b, ra, rb, c
    l_img = swap_loss_images(*rgb, mock_fn)
    assert len(calls) == 5, "feature_fn 必须逐图调用一次"
    fa, fb = mock_fn(rgb[0]), mock_fn(rgb[1])
    fra, frb, fc = mock_fn(rgb[2]), mock_fn(rgb[3]), mock_fn(rgb[4])
    assert fa["style"].shape == (B, 6) and fa["content"].shape == (B, 6)
    manual = (cosine_distance_loss(fa["style"], fra["style"]) + cosine_distance_loss(fb["style"], frb["style"])
              + 0.5 * (cosine_distance_loss(fa["content"], fc["content"]) + cosine_distance_loss(fb["content"], fc["content"])))
    assert torch.allclose(l_img, manual, atol=1e-6)
    # dict 级入口与图像级入口等价 (C2 训练循环用预计算特征走 swap_loss_features)
    l_feat = swap_loss_features(fa, fb, fra, frb, fc)
    assert torch.allclose(l_feat, l_img, atol=1e-6)


# ---------------- 验收 3: 四条件参考评测钩子 ----------------

def _mock_ref_condition_world(T=4, D=8, seed=9):
    """构造 mock 特征世界: correct 输出强跟随其参考; wrong 输出跟随弱于 correct."""
    g = torch.Generator().manual_seed(seed)
    content = torch.randn(T, D, generator=g)
    refs = {k: torch.randn(T, D, generator=g) for k in CONDITION_KEYS}

    def blend(target, ref, alpha):
        v = (1 - alpha) * target + alpha * ref
        return v

    gen = {
        COND_CORRECT: blend(content, refs[COND_CORRECT], 0.7),
        COND_WRONG: blend(content, refs[COND_WRONG], 0.3),       # 错误参考 → 跟随弱
        COND_SAME_STYLE: blend(content, refs[COND_SAME_STYLE], 0.5),
        COND_DIFF_STYLE: blend(content, refs[COND_DIFF_STYLE], 0.2),
    }
    return gen, refs, content


def test_eval_ref_conditions_table_shape_and_range():
    gen, refs, content = _mock_ref_condition_world()
    table = eval_ref_conditions(gen, refs, content)
    assert [r["condition"] for r in table] == list(CONDITION_KEYS)
    for r in table:
        assert set(r) == {"condition", "dino_c", "dino_s_own", "dino_s_correct"}
        for k in ("dino_c", "dino_s_own", "dino_s_correct"):
            assert -1.0 <= r[k] <= 1.0


def test_eval_ref_conditions_smoke_pass_with_mock():
    gen, refs, content = _mock_ref_condition_world()
    table = eval_ref_conditions(gen, refs, content)
    by = {r["condition"]: r for r in table}
    # 正确条件对自身参考的跟随强于错误条件; 错误条件对正确参考的"误分"不高于正确条件
    assert by[COND_CORRECT]["dino_s_own"] > by[COND_WRONG]["dino_s_own"]
    assert by[COND_WRONG]["dino_s_correct"] <= by[COND_CORRECT]["dino_s_correct"]
    assert_ref_conditions_smoke(table)  # 不抛


def test_eval_ref_conditions_smoke_s1_triggers_on_correct_ref_inversion():
    """S1: 错误参考输出若获得更高'正确参考'得分 → 断言触发 (mock 构造反例)."""
    gen, refs, content = _mock_ref_condition_world()
    # 反例: 错误参考的输出反而更靠近正确参考 (对齐反转)
    gen[COND_WRONG] = 0.2 * content + 0.8 * refs[COND_CORRECT]
    table = eval_ref_conditions(gen, refs, content)
    res = check_ref_conditions_smoke(table)
    assert not res["ok"] and any("S1" in f for f in res["failures"])
    with pytest.raises(AssertionError, match="S1"):
        assert_ref_conditions_smoke(table)


def test_eval_ref_conditions_smoke_s2_triggers_on_own_ref_inversion():
    """S2: 错误参考条件对自身参考的 dino_s 反超正确参考条件 → 断言触发."""
    gen, refs, content = _mock_ref_condition_world()
    gen[COND_WRONG] = 0.1 * content + 0.95 * refs[COND_WRONG]  # 错误参考被"过度跟随"
    table = eval_ref_conditions(gen, refs, content)
    by = {r["condition"]: r for r in table}
    assert by[COND_WRONG]["dino_s_own"] > by[COND_CORRECT]["dino_s_own"]
    res = check_ref_conditions_smoke(table)
    assert not res["ok"] and any("S2" in f for f in res["failures"])
    with pytest.raises(AssertionError, match="S2"):
        assert_ref_conditions_smoke(table)


def test_eval_ref_conditions_smoke_missing_condition():
    gen, refs, content = _mock_ref_condition_world()
    gen_partial = {k: v for k, v in gen.items() if k != COND_DIFF_STYLE}
    with pytest.raises(AssertionError, match=COND_DIFF_STYLE):
        eval_ref_conditions(gen_partial, refs, content)


# ---------------- 验收 4: 无画派 ID (inspect 断言 + legacy 槽位惰性实证) ----------------

def test_label_free_interfaces_pass():
    """训练/评测接口签名不含 sid/onehot/目标类别输入 (inspect 机械断言)."""
    assert_label_free_interfaces()  # 不抛


@pytest.mark.parametrize("bad_param", ["sid", "tid", "style_id", "target_style_id",
                                       "onehot", "one_hot", "class_id", "label", "labels"])
def test_label_free_detector_catches_forbidden_params(bad_param):
    src = f"def dummy({bad_param}=None):\n    return {bad_param}\n"
    ns: dict = {}
    exec(src, ns)
    with pytest.raises(AssertionError):
        assert_label_free_interface(ns["dummy"], "dummy")
    # 允许列表: 正常参数名不误伤
    def ok(reference_feats=None, content_feat=None, lambda_c=0.5):
        return None
    assert_label_free_interface(ok, "ok")


def test_legacy_style_slot_is_inert_in_reused_model():
    """复用模型 SLFUNetV24OneShot 的 style_id 槽位是 v24 legacy 兼容参数: forward 内部
    恒置零 onehot → 传入不同画派 id 输出逐位相同 (torch.equal), 实证"模型条件无画派 ID"."""
    from exp.v24.p0_o1_o3_value import SLFUNetV24OneShot
    torch.manual_seed(3)
    fm = SLFUNetV24OneShot(v_dim=192, use_conf_gate=True, use_global_fib=True).eval()
    B = 1
    z, y16, d16 = torch.randn(B, 4, 64, 64), torch.randn(B, 1, 16, 16), torch.randn(B, 384, 16, 16)
    d16r, v16r, ug = torch.randn(B, 384, 16, 16), torch.randn(B, 192, 16, 16), torch.randn(B, 795)
    u, tau = torch.randn(B, 128), torch.rand(B)
    with torch.no_grad():
        o_a = fm(z, y16, d16, torch.zeros(B, dtype=torch.long), u, tau, d16_ref=d16r, v16_ref=v16r, u_glob=ug)
        o_b = fm(z, y16, d16, torch.full((B,), 3, dtype=torch.long), u, tau, d16_ref=d16r, v16_ref=v16r, u_glob=ug)
    assert torch.equal(o_a, o_b), "legacy style_id 槽位必须对输出零影响 (内部 onehot 恒零)"


def test_train_c2_arm_defaults_match_3060_budget():
    """3060 12GB 约束写进签名默认值: micro=12 (train_o_arm 口径), swap_micro=4, λ_c=0.5."""
    sig = inspect.signature(train_c2_arm)
    assert sig.parameters["micro"].default == MICRO_3060 == 12
    assert sig.parameters["swap_micro"].default == SWAP_MICRO_3060 == 4
    assert sig.parameters["lambda_c"].default == LAMBDA_C_DEFAULT == 0.5
    assert sig.parameters["swap_weight"].default == SWAP_WEIGHT_DEFAULT == 0.5
    assert sig.parameters["accum"].default == 4


def test_oneshot_integrate_cpu_smoke():
    """GPU 评测积分器 (10 步 Heun + 风格 CFG) 的 CPU 小步烟测: 形状/有限值/CFG 接线正确."""
    from exp.v25.mechanism_c import _oneshot_integrate
    from exp.v24.p0_o1_o3_value import SLFUNetV24OneShot
    torch.manual_seed(19)
    fm = SLFUNetV24OneShot(v_dim=192, use_conf_gate=True, use_global_fib=True).eval()
    B = 1
    z_init = torch.randn(B, 4, 64, 64)
    y16, d16 = torch.randn(B, 1, 16, 16), torch.randn(B, 384, 16, 16)
    u, ug = torch.randn(B, 128), torch.randn(B, 795)
    d16r, v16r = torch.randn(B, 384, 16, 16), torch.randn(B, 192, 16, 16)
    z_out = _oneshot_integrate(fm, z_init, y16, d16, u, ug, d16r, v16r,
                               tau0=0.45, wj=2.2, device="cpu", steps=2)
    assert z_out.shape == z_init.shape and bool(torch.isfinite(z_out).all())


# ---------------- 验收 5: 参考特征缓存 ----------------

def test_tensor_sha_key_stability():
    t = torch.randn(4, 64, 64).half()
    k1, k2 = tensor_sha(t), tensor_sha(t.clone())
    assert k1 == k2 and len(k1) == 64
    assert bytes_sha(b"abc") == bytes_sha(b"abc")
    assert bytes_sha(b"abc") != bytes_sha(b"abd")
    assert tensor_sha(t * 2) != k1, "数值变化必须换键"
    assert tensor_sha(t.float()) != k1, "dtype 参与键控"
    assert tensor_sha(t[None]) != k1, "shape 参与键控"


def test_ref_feature_cache_hit_and_key_stability(tmp_path):
    """同参考内容重复编码直接命中; 键稳定; 缓存文件落 --cache-dir (tmp_path)."""
    encode_calls = []

    def encode():
        encode_calls.append(1)
        return {"tokens": {"s3": torch.randn(1, 4, 8).half()}, "pal": torch.randn(1, 27).half()}

    cache = ReferenceFeatureCache(tmp_path)
    z_ref = torch.randn(1, 4, 64, 64).half()
    key = tensor_sha(z_ref)
    v1 = cache.get_or_encode(key, encode)
    v2 = cache.get_or_encode(key, encode)
    v3 = cache.get_or_encode(tensor_sha(z_ref.clone()), encode)
    assert len(encode_calls) == 1, "同参考重复编码必须命中缓存"
    assert torch.equal(v1["pal"], v2["pal"]) and torch.equal(v2["pal"], v3["pal"])
    assert cache.hits == 2 and cache.misses == 1
    assert (tmp_path / f"ref_{key}.pt").exists()
    # 不同参考 → 新键 miss
    other = tensor_sha(torch.randn(1, 4, 64, 64).half())
    cache.get_or_encode(other, encode)
    assert cache.misses == 2 and len(encode_calls) == 2
    assert cache.stats()["dir"] == tmp_path.as_posix()


def test_ref_feature_cache_default_dir_matches_spec():
    """默认缓存目录 = runs/v25/cache/ref_feats (B02 验收 5 指定路径)."""
    from exp.v25.reference_teacher_v2 import DEFAULT_REF_CACHE_DIR
    assert DEFAULT_REF_CACHE_DIR.as_posix().endswith("runs/v25/cache/ref_feats")
    assert ReferenceFeatureCache(None).dir == DEFAULT_REF_CACHE_DIR


def test_batch_cached_ref_tokens_3060_regression():
    """3060 R2 现场 bug 回归: 单样本 2D (N,C) 缓存条目 cat 后不得把 batch 折进 spatial 维.

    现场错误链: 12×(256,384) 沿 dim 0 cat → (3072,384) →
    torch.gather(ref_s3, 1, best_j.unsqueeze(-1).expand_as(ref_s3)) 处
    RuntimeError expand(LongTensor[12,256,1], size=[3072,384]).
    修复后 batch_cached_ref_tokens 必须重建 o2 契约形状 (B,256,384), 且崩溃点走通.
    """
    B, N, C = 12, 256, 384
    stale_items = [  # 3060 现场修复前的历史条目: tokens 2D (N,C) (批维丢失), pal 2D (1,27)
        {"tokens": {"s2": torch.randn(N, 192), "s3": torch.randn(N, C)}, "pal": torch.randn(1, 27)}
        for _ in range(B)
    ]
    toks, pal = batch_cached_ref_tokens(stale_items)
    assert toks["s3"].shape == (B, N, C), "batch 维必须恢复 (不得折进 spatial 维)"
    assert toks["s2"].shape == (B, N, 192)
    assert pal.shape == (B, 27)
    # 现场崩溃点: gather 的 expand_as 必须走通
    best_j = torch.randint(0, N, (B, N))
    matched = torch.gather(toks["s3"], 1, best_j.unsqueeze(-1).expand_as(toks["s3"]))
    assert matched.shape == (B, N, C)


def test_batch_cached_ref_tokens_new_format_and_mixed_entries():
    """新条目 (单样本 extract_tokens 输出, 3D 带批维) 原样; 新旧条目混排与 pal 1D 兼容."""
    B, N, C = 4, 8, 6
    new_items = [{"tokens": {"s3": torch.randn(1, N, C)}, "pal": torch.randn(1, 27)} for _ in range(B)]
    toks, pal = batch_cached_ref_tokens(new_items)
    assert toks["s3"].shape == (B, N, C) and pal.shape == (B, 27)
    mixed = new_items[:2] + [{"tokens": {"s3": torch.randn(N, C)}, "pal": torch.randn(27)} for _ in range(2)]
    toks2, pal2 = batch_cached_ref_tokens(mixed)
    assert toks2["s3"].shape == (B, N, C) and pal2.shape == (B, 27)
    with pytest.raises(AssertionError, match="条目为空"):
        batch_cached_ref_tokens([])
    with pytest.raises(AssertionError, match="维度异常"):
        batch_cached_ref_tokens([{"tokens": {"s3": torch.randn(N)}, "pal": torch.randn(1, 27)}])


def test_teacher_first_batch_math_shapes_with_mock_tokens():
    """教师优化首个 batch 的全链形状回归 (mock tokens/projs, 零 checkpoint 加载).

    复现 build_teacher_pairs 预匹配 + 单步损失的每一步张量契约 (与 o2 逐式对照):
    历史缓存条目 (2D) → batch_cached_ref_tokens → DINO 局部匹配 gather/expand →
    SWD/Oklab/局部项/content_w·l_content/l_reg → backward. 3060 现场崩溃发生在
    gather/expand; 本测试保证修复后整链在 (B,N,C) 契约下走通.
    """
    from exp.v20.submode_patchswd import sliced_wasserstein_1d

    B, steps_hw = 12, (16, 16)
    g = torch.Generator().manual_seed(21)
    # 参考侧: 历史 2D 缓存条目 (现场格式) → 批量重建
    items = [{"tokens": {"rgb": torch.randn(1024, 27), "s1": torch.randn(1024, 96),
                         "s2": torch.randn(1024, 192), "s3": torch.randn(256, 384)},
              "pal": torch.randn(27)} for _ in range(B)]
    toks_ref, pal_ref = batch_cached_ref_tokens(items)
    assert toks_ref["s3"].shape == (B, 256, 384) and pal_ref.shape == (B, 27)

    # 内容/输出侧: 批量 extract_tokens 输出契约 (B,N,C)
    mk = lambda n, c: torch.randn(B, n, c, generator=g)
    toks_src = {"rgb": mk(1024, 27), "s1": mk(1024, 96), "s2": mk(1024, 192), "s3": mk(256, 384)}
    toks_hat = {"rgb": mk(1024, 27), "s1": mk(1024, 96), "s2": mk(1024, 192), "s3": mk(256, 384)}

    # Q/K 灰度 DINO 局部语义匹配 (与 o2 同式)
    d16_c = torch.randn(B, 384, *steps_hw, generator=g)
    d16_r = torch.randn(B, 384, *steps_hw, generator=g)
    dc = F.normalize(d16_c.flatten(2).transpose(1, 2), dim=-1)
    dr = F.normalize(d16_r.flatten(2).transpose(1, 2), dim=-1)
    sim = torch.bmm(dc, dr.transpose(1, 2))
    best_sim, best_j = sim.max(dim=-1)
    gate_loc = (best_sim > 0.25).float().unsqueeze(-1)
    ref_s3 = toks_ref["s3"]
    matched_s3_n = F.normalize(torch.gather(ref_s3, 1, best_j.unsqueeze(-1).expand_as(ref_s3)), dim=-1)

    # 单步损失 (content_w=1.2 档)
    z_opt = torch.randn(B, 4, 64, 64, generator=g).requires_grad_(True)
    projs = {k: F.normalize(torch.randn(c, 8, generator=g), dim=0)
             for k, c in (("rgb", 27), ("s1", 96), ("s2", 192), ("s3", 384))}
    l_swd = (
        1.0 * sliced_wasserstein_1d(toks_hat["rgb"], toks_ref["rgb"], projs["rgb"])
        + 0.35 * sliced_wasserstein_1d(toks_hat["s1"], toks_ref["s1"], projs["s1"])
        + 0.45 * sliced_wasserstein_1d(toks_hat["s2"], toks_ref["s2"], projs["s2"])
        + 0.55 * sliced_wasserstein_1d(toks_hat["s3"], toks_ref["s3"], projs["s3"])
    ).mean()
    l_pal = F.smooth_l1_loss(torch.randn(B, 27, generator=g), pal_ref)
    l_loc = ((1.0 - (F.normalize(toks_hat["s3"], dim=-1) * matched_s3_n).sum(-1, keepdim=True)) * gate_loc).mean()
    l_content = 0.5 * (
        (1.0 - (F.normalize(toks_hat["s2"], dim=-1) * F.normalize(toks_src["s2"], dim=-1)).sum(-1)).mean()
        + (1.0 - (F.normalize(toks_hat["s3"], dim=-1) * F.normalize(toks_src["s3"], dim=-1)).sum(-1)).mean()
    )
    l_reg = z_opt.square().mean()
    loss = (l_swd + 0.7 * l_pal + 0.25 * l_loc) + 1.2 * l_content + 0.15 * l_reg
    assert torch.isfinite(loss)
    loss.backward()
    assert z_opt.grad is not None and bool(torch.isfinite(z_opt.grad).all())


# ---------------- 验收 6: 双指标筛选边界 ----------------

def test_dual_metric_filter_all_keep_and_all_reject():
    n = 6
    mid_ok = torch.full((n,), 0.86)
    gain_ok = torch.full((n,), 0.20)
    assert bool(dual_metric_filter(mid_ok, gain_ok).all()), "全在带内 → 全保留"
    # 单指标违反即全拒绝
    assert not bool(dual_metric_filter(torch.full((n,), 0.95), gain_ok).any()), "mid_cos > hi → 全拒 (保守臂)"
    assert not bool(dual_metric_filter(torch.full((n,), 0.75), gain_ok).any()), "mid_cos < lo → 全拒 (内容漂移)"
    assert not bool(dual_metric_filter(mid_ok, torch.full((n,), 0.05)).any()), "style_gain 不足 → 全拒"
    # NaN → 拒绝
    nan_mid = mid_ok.clone()
    nan_mid[0] = float("nan")
    k = dual_metric_filter(nan_mid, gain_ok)
    assert not bool(k[0]) and bool(k[1:].all())


def test_dual_metric_filter_inclusive_boundaries():
    lo, hi = MID_COS_BAND
    mid = torch.tensor([lo, hi])
    gain = torch.tensor([STYLE_GAIN_MIN, STYLE_GAIN_MIN])
    assert bool(dual_metric_filter(mid, gain).all()), "闭区间端点 (lo/hi/gain_min) 必须保留"


def test_filter_diagnostics_magnitude_distribution():
    n = 10
    mid = torch.linspace(0.78, 0.96, n)
    gain = torch.linspace(0.02, 0.30, n)
    edit = torch.linspace(0.1, 3.0, n)
    keep = dual_metric_filter(mid, gain)
    diag = filter_diagnostics(mid, gain, keep, edit_norm=edit)
    assert diag["n_total"] == n and diag["n_kept"] == int(keep.sum())
    assert abs(diag["keep_rate"] - keep.float().mean().item()) < 1e-6
    assert "kept" in diag["edit_norm"] and "rejected" in diag["edit_norm"]
    assert abs(diag["edit_norm"]["kept_mean"] - edit[keep].mean().item()) < 1e-5
    # 全拒绝/全保留极角
    diag0 = filter_diagnostics(mid, torch.zeros(n), torch.zeros(n, dtype=torch.bool))
    assert diag0["n_kept"] == 0 and diag0["keep_rate"] == 0.0
    diag1 = filter_diagnostics(mid, gain, torch.ones(n, dtype=torch.bool))
    assert diag1["n_kept"] == n and diag1["keep_rate"] == 1.0


def test_family_retention_report_per_family_rates():
    sid = torch.tensor([0, 0, 1, 1, 2, 3, 3, 3, 3, 3])
    keep = torch.tensor([True, False, True, True, False, True, True, False, False, True])
    rep = family_retention_report(sid, keep, n_styles=5, style_names=["A", "B", "C", "D", "E"])
    pf = rep["per_family"]
    assert pf["A"]["keep_rate"] == 0.5 and pf["B"]["keep_rate"] == 1.0
    assert pf["C"]["keep_rate"] == 0.0 and pf["D"]["keep_rate"] == 0.6
    assert "E" not in pf, "无样本画派不产出误导性保留率"
    assert rep["keep_rate_spread"] == pytest.approx(1.0, abs=1e-6)
    with pytest.raises(AssertionError):
        family_retention_report(sid[:-1], keep)


# ---------------- 教师包 v2 常量与纯函数 ----------------

def test_teacher_w_grid_and_tags():
    assert TEACHER_W_GRID == (2.2, 1.2, 0.7)
    assert [teacher_tag(w) for w in TEACHER_W_GRID] == ["tcons", "tmed", "texpr"]
    assert teacher_tag(1.5) == "w1p5"


def test_pooled_style_cos_self_similarity_and_shape():
    toks = {"s2": torch.randn(3, 32, 192), "s3": torch.randn(3, 16, 384)}
    cos_self = pooled_style_cos(toks, toks)
    assert cos_self.shape == (3,) and torch.allclose(cos_self, torch.ones(3), atol=1e-5)
    toks2 = {"s2": torch.randn(3, 32, 192), "s3": torch.randn(3, 16, 384)}
    cos_ab = pooled_style_cos(toks, toks2)
    assert cos_ab.shape == (3,) and bool((cos_ab.abs() <= 1.0).all())
    with pytest.raises(KeyError):
        pooled_style_cos(toks, {"s2": toks2["s2"]})  # 缺尺度键必须显式报错


def test_build_teacher_pairs_signature_label_free_and_knobs():
    sig = inspect.signature(build_teacher_pairs)
    assert_label_free_interface(build_teacher_pairs, "build_teacher_pairs")
    for name, default in (("n_cand", 2400), ("n_keep", 2000), ("steps", 24)):
        assert sig.parameters[name].default == default
    assert sig.parameters["mid_cos_band"].default == MID_COS_BAND
    assert sig.parameters["style_gain_min"].default == STYLE_GAIN_MIN


# ---------------- C3 强度混合 ----------------

def test_mix_teacher_strengths_formula():
    z_c = torch.ones(2, 1, 2, 2)
    z_e = torch.zeros(2, 1, 2, 2)
    assert torch.equal(mix_teacher_strengths(z_c, z_e, 0.0), z_c)
    assert torch.equal(mix_teacher_strengths(z_c, z_e, 1.0), z_e)
    assert torch.allclose(mix_teacher_strengths(z_c, z_e, 0.4), torch.full_like(z_c, 0.6))


def _mock_pack(seed, n_cand=8, keep_n=6, value=1.0):
    g = torch.Generator().manual_seed(seed)
    idx_c = torch.randint(0, 50, (n_cand,), generator=g)
    idx_r = (idx_c + 1 + torch.randint(0, 40, (n_cand,), generator=g)) % 50
    keep = torch.zeros(n_cand, dtype=torch.bool)
    keep[:keep_n] = True
    return {
        "kind": "ref_teacher_pairs_v2",
        "idx_c": idx_c[keep], "idx_r": idx_r[keep],
        "z_cr": torch.full((keep_n, 1, 2, 2), value),
        "keep_mask_all": keep,
        "idx_c_all": idx_c, "idx_r_all": idx_r,
        "pair_uid_all": idx_c.long() * 100000 + idx_r.long(),
        "meta": {"n_cand": n_cand},
    }


def test_build_strength_mix_pack_alignment_formula_and_reproducible():
    p_cons, p_expr = _mock_pack(11, value=1.0), _mock_pack(11, value=0.0)  # 同 seed → 同配对
    pack = build_strength_mix_pack(p_cons, p_expr, gammas=(0.4, 0.7, 1.0), seed=77)
    n = pack["n_pairs"]
    assert n == 6 and pack["idx_c"].shape[0] == n
    assert torch.equal(pack["idx_c"], p_cons["idx_c"]) and torch.equal(pack["idx_r"], p_cons["idx_r"])
    assert bool(torch.isin(pack["gamma"], torch.tensor(C3_GAMMA_POOL)).all())
    # 逐对公式: z_mix = (1−γ)·1 + γ·0 = 1−γ (fp16 量化容差)
    expected = (1.0 - pack["gamma"]).view(-1, 1, 1, 1).expand(n, 1, 2, 2)
    assert torch.allclose(pack["z_cr"].float(), expected, atol=1e-3)
    # 可复现: 同 seed 同分配
    pack2 = build_strength_mix_pack(p_cons, p_expr, gammas=(0.4, 0.7, 1.0), seed=77)
    assert torch.equal(pack["gamma"], pack2["gamma"]) and torch.equal(pack["z_cr"], pack2["z_cr"])
    # 配对不一致 → 断言
    p_expr_bad = _mock_pack(12, value=0.0)
    with pytest.raises(AssertionError, match="同配对"):
        build_strength_mix_pack(p_cons, p_expr_bad)


# ---------------- 训练特征 fn: 拒绝评测裁判 ----------------

@pytest.mark.parametrize("bad", ["dino", "csd", "clip", "lpips"])
def test_train_feature_kind_rejects_referees(bad):
    with pytest.raises(ValueError, match=bad):
        assert_train_feature_kind(bad)
    with pytest.raises(ValueError):
        make_train_feature_fn(bad)  # 必须在加载任何权重前拒绝


def test_train_feature_kind_allows_only_train_kinds():
    for k in TRAIN_FEATURE_KINDS:
        assert_train_feature_kind(k)  # 不抛
    with pytest.raises(ValueError, match="未知训练特征"):
        assert_train_feature_kind("resnet")


def test_forbidden_list_covers_all_referees():
    assert set(FORBIDDEN_TRAIN_FEATURES) == {"dino", "csd", "clip", "lpips"}


# ---------------- 环境 ----------------

def test_no_cuda_in_tests():
    assert not torch.cuda.is_available() or torch.cuda.device_count() == 0
