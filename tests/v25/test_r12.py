"""V25 R12 (A08 R1/R2 补救) CPU 单测: bypass 修复语义 / ctx 键完整性 / γ=0 恒等 / 臂名计划.

覆盖 (R12 任务书验收逐条, 全部小张量 mock, 禁真实 ckpt / 禁 GPU):
  - same-task bypass 语义: bypass 后 same 行输出 == z_src (逐元素), cross 行不动;
    未 bypass 的学生 same 输出 != z_src (负对照 — 证明旧 :382 伪迹可被本组断言捕获)
  - mechanism_b.py:382 静态守卫: measure 路径必须传 bypass 后张量 (A08 R2 指定修复)
  - ctx750_B4 键完整性与形状一致性 (与 mechanism_a.build_teacher_latents_750 返回结构逐键对齐,
    AST 契约测试防键集漂移)
  - γ=0 恒等: 真实小学生网络 γ=0 逐元素精确回退 z_p1, 扫描记录 residual 精确为 0 (回归断言)
  - R1 七臂计划与臂名 (MechA_B4anc_* / MechB_B0fix_g*), renamed_mecha_measure 前缀插入与恢复
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # 必须在 import torch 之前 → 纯 CPU
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import ast
import math
import re
from pathlib import Path

import pytest
import torch

from exp.v25 import mechanism_a, mechanism_b
from exp.v25.mechanism_a import GateNet
from exp.v25.mechanism_b import B_GAMMA_SWEEP, GammaResidualStudent, eval_student_gamma_sweep
from exp.v25.r12_anchor_reeval import (
    CTX_KEYS,
    MECHA_TOKEN,
    apply_same_bypass,
    mechanism_a_plan,
    mechb_arm_name,
    renamed_mecha_measure,
    run_mechanism_a_reeval,
    run_mechanism_b_reeval,
    validate_ctx750,
)

torch.manual_seed(0)


# ---------------- fixtures (小张量 mock, 无 ckpt / 无 GPU) ----------------

def _mock_ctx(B=6, seed=17):
    """合法 mock ctx750: 6 任务 (5 cross 覆盖全部 5 画派 + 1 same), 64×64 latent (A 门控池化 4→16 需求)."""
    g = torch.Generator().manual_seed(seed)
    z_src = torch.randn(B, 4, 64, 64, generator=g) * 0.6
    sid = torch.tensor([0, 1, 2, 3, 4, 0])
    tid = torch.tensor([1, 2, 3, 4, 0, 0])  # 末位 same
    return dict(
        tasks=[{"source_style_id": int(s), "target_style_id": int(t)} for s, t in zip(sid, tid)],
        z_src=z_src,
        z_p1=z_src + torch.randn(B, 4, 64, 64, generator=g) * 0.2,
        z_d6=z_src + torch.randn(B, 4, 64, 64, generator=g) * 0.4,
        c64=torch.randn(B, 4, 64, 64, generator=g) * 0.5,
        d16=torch.randn(B, 384, 16, 16, generator=g),
        chords=torch.randn(B, 128, generator=g),
        sid=sid, tid=tid, same=sid == tid,
        dec=object(), tau_stats={"tau_mean": 0.55, "tau_p50": 0.55},
    )


def _small_student(seed=11, width=32, blocks=2):
    """真实 GammaResidualStudent (小宽度), 权重随机化以打破零初始化 (same 行输出非零)."""
    torch.manual_seed(seed)
    stu = GammaResidualStudent(width=width, blocks=blocks, dilations=tuple((1, 2)[:blocks]))
    torch.manual_seed(seed + 100)
    for m in stu.modules():
        if isinstance(m, torch.nn.Conv2d) and m.weight.requires_grad:
            torch.nn.init.normal_(m.weight, std=0.05)
            if m.bias is not None:
                torch.nn.init.normal_(m.bias, std=0.05)
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, std=0.05)
            if m.bias is not None:
                torch.nn.init.normal_(m.bias, std=0.05)
    return stu.eval()


# ---------------- 1) bypass 修复语义 (A08 R2) ----------------

def test_apply_same_bypass_semantics():
    """bypass 后: same 行输出逐元素 == z_src; cross 行逐元素不动; 与 :382 手工覆写形式逐位等价."""
    ctx = _mock_ctx()
    same, z_src = ctx["same"], ctx["z_src"]
    z = torch.randn_like(z_src) * 0.3  # 任意待 bypass 张量 (模拟学生原始输出)
    z_bp = apply_same_bypass(z, same, z_src)
    assert torch.equal(z_bp[same], z_src[same]), "same 行必须逐元素等于 z_src"
    assert torch.equal(z_bp[~same], z[~same]), "cross 行必须保持学生原始输出"
    z_manual = z.clone()
    z_manual[same] = z_src[same]  # mechanism_b.py:382 修复行的手工形式
    assert torch.equal(z_bp, z_manual)


def test_student_same_output_differs_from_src_before_bypass():
    """负对照: 未 bypass 的学生 same 任务输出 != z_src (旧 :382 伪迹正是漏掉这一步)."""
    ctx = _mock_ctx()
    stu = _small_student()
    z_out = stu(ctx["z_p1"], ctx["c64"], ctx["d16"], ctx["chords"],
                torch.full((len(ctx["z_p1"]),), 0.55))
    assert not torch.equal(z_out[ctx["same"]], ctx["z_src"][ctx["same"]])
    z_bp = apply_same_bypass(z_out, ctx["same"], ctx["z_src"])
    assert torch.equal(z_bp[ctx["same"]], ctx["z_src"][ctx["same"]])


def test_mechanism_b_382_static_bypass_fix():
    """静态守卫 (A08 R2 指定修复, 所有权约束下只许改 :382 一处): main 的 measure 调用必须传
    bypass 后张量 — z_by_gamma 克隆 + same 行 z_src 覆写, 且不得再直接传未 bypass 张量."""
    src = Path(mechanism_b.__file__).read_text(encoding="utf-8")
    assert re.search(r'z_bp = sweep\["z_by_gamma"\]\[rec\["gamma"\]\]\.clone\(\)', src), \
        "measure 路径缺少 z_by_gamma 的 bypass 克隆"
    assert re.search(r'z_bp\[ctx750\["same"\]\] = ctx750\["z_src"\]\[ctx750\["same"\]\]', src), \
        "measure 路径缺少 same 行 z_src 覆写"
    calls = re.findall(r'eval_latent_v25\(arm, ([\w\[\]"\.]+),', src)
    assert len(calls) == 1 and calls[0] == "z_bp", \
        f"mechanism_b measure 必须且只能传 bypass 后张量 z_bp, 得到 {calls}"


# ---------------- 2) ctx750_B4 键完整性 (R1) ----------------

def _return_dict_keys(fn_name: str, module_file: Path) -> set[str]:
    tree = ast.parse(module_file.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fn_name:
            for sub in ast.walk(node):
                if (isinstance(sub, ast.Return) and isinstance(sub.value, ast.Call)
                        and getattr(sub.value.func, "id", "") == "dict"):
                    return {kw.arg for kw in sub.value.keywords}
    raise AssertionError(f"no dict-return found in {fn_name}")


def test_ctx_keys_align_with_build_teacher_latents_750():
    """AST 契约: CTX_KEYS 与 mechanism_a.build_teacher_latents_750 返回 dict 键集逐键相等."""
    keys = _return_dict_keys("build_teacher_latents_750", Path(mechanism_a.__file__))
    assert keys == set(CTX_KEYS), f"ctx 键集漂移: {keys ^ set(CTX_KEYS)}"


def test_validate_ctx750_accepts_mock():
    ctx = _mock_ctx()
    validate_ctx750(ctx)  # 不抛即通过


def test_validate_ctx750_rejects_key_drift_and_shape_mismatch():
    ctx = _mock_ctx()
    broken = dict(ctx)
    del broken["tau_stats"]
    with pytest.raises(ValueError, match="missing"):
        validate_ctx750(broken)
    broken = dict(ctx, z_anc_extra=torch.zeros(1))  # 多余键同样拒绝 (键集漂移防护)
    with pytest.raises(ValueError, match="extra"):
        validate_ctx750(broken)
    for key, bad in (("z_d6", torch.zeros(6, 4, 32, 32)),
                     ("c64", torch.zeros(6, 4, 32, 32)),
                     ("chords", torch.zeros(6, 64)),
                     ("d16", torch.zeros(6, 384, 8, 8))):
        broken = dict(ctx)
        broken[key] = bad
        with pytest.raises(ValueError, match=key):
            validate_ctx750(broken)
    broken = dict(ctx)
    broken["same"] = ~ctx["same"]
    with pytest.raises(ValueError, match="same must equal"):
        validate_ctx750(broken)
    broken = dict(ctx)
    broken["dec"] = None
    with pytest.raises(ValueError, match="dec"):
        validate_ctx750(broken)


# ---------------- 3) γ=0 恒等 (回归断言语义) ----------------

def test_gamma_zero_exact_identity_and_sweep_residual_zero():
    """γ=0: 学生输出逐元素 == z_p1 (torch.equal); 扫描记录 residual 精确为 0; bypass 后 same==z_src."""
    ctx = _mock_ctx()
    stu = _small_student()
    z_out = stu(ctx["z_p1"], ctx["c64"], ctx["d16"], ctx["chords"],
                torch.zeros(len(ctx["z_p1"])))
    assert torch.equal(z_out, ctx["z_p1"]), "γ=0 必须逐元素精确回退 z_p1"

    def student_fn(zp, gv, cb):
        return stu(zp, cb["c64"], cb["d16"], cb["chord"], gv)

    sweep = eval_student_gamma_sweep(student_fn, ctx["z_src"], ctx["z_p1"], ctx["z_d6"],
                                     ctx["same"], ctx["tid"],
                                     cond=dict(c64=ctx["c64"], d16=ctx["d16"], chord=ctx["chords"]),
                                     gammas=B_GAMMA_SWEEP, device="cpu")
    by_g = {r["gamma"]: r for r in sweep["records"]}
    assert len(sweep["records"]) == len(B_GAMMA_SWEEP)
    assert by_g[0.0]["residual_mean"] == 0.0 and by_g[0.0]["residual_max"] == 0.0, \
        "γ=0 residual 必须精确为 0 (R12 回归断言)"
    for g in (0.25, 0.55, 0.70):
        assert by_g[g]["residual_mean"] > 0.0, f"γ={g} 随机权重学生残差应 > 0"
    assert torch.equal(sweep["z_by_gamma"][0.0], ctx["z_p1"])
    z_bp = apply_same_bypass(sweep["z_by_gamma"][0.55], ctx["same"], ctx["z_src"])
    assert torch.equal(z_bp[ctx["same"]], ctx["z_src"][ctx["same"]])


# ---------------- 4) R1/R2 臂名计划与复用机制 ----------------

def test_mechanism_a_plan_seven_arms():
    plan = mechanism_a_plan(0.55)
    assert len(plan) == 7
    names = [p["measured_arm"] for p in plan]
    assert names == [
        "MechA_B4anc_A0_global_g55",
        "MechA_B4anc_A1_heuristic_k50_g55",
        "MechA_B4anc_A1_heuristic_k100_g55",
        "MechA_B4anc_A2_learned_g55",
        "MechA_B4anc_A2_learned_em_g55",
        "MechA_B4anc_A3_smooth_g55",
        "MechA_B4anc_A3_smooth_em_g55",
    ]
    assert plan[0]["net"] is None and plan[1]["kappa"] == 0.5 and plan[2]["kappa"] == 1.0
    em_flags = [p["energy_match"] for p in plan]
    assert em_flags == [False, False, False, False, True, False, True], \
        "A2/A3 必须各带一个 energy_match 臂 (em)"
    assert all(p["arm"] in ("A0_global", "A1_heuristic", "A2_learned", "A3_smooth") for p in plan), \
        "arm 字段必须是 eval_mechanism_a_arm / arm_gate_field 可分发的原名"


def test_mechb_arm_names():
    assert [mechb_arm_name(g) for g in B_GAMMA_SWEEP] == [
        "MechB_B0fix_g00", "MechB_B0fix_g25", "MechB_B0fix_g33", "MechB_B0fix_g40",
        "MechB_B0fix_g55", "MechB_B0fix_g63", "MechB_B0fix_g70",
    ]


def test_renamed_mecha_measure_prefixes_and_restores():
    """renamed_mecha_measure: eval_mechanism_a_arm 命名空间内的测量臂名插入 B4anc 前缀, 退出恢复."""
    seen = []
    orig = mechanism_a.eval_latent_v25

    def spy(arm_name, z_750, tasks, dec, device="cuda"):
        seen.append(arm_name)
        return {"arm": arm_name}

    mechanism_a.eval_latent_v25 = spy
    try:
        with renamed_mecha_measure(MECHA_TOKEN):
            mechanism_a.eval_latent_v25("MechA_A2_learned_em_g55", None, [], None)
            with pytest.raises(ValueError, match="MechA"):
                mechanism_a.eval_latent_v25("MechB_B0_g55", None, [], None)
        mechanism_a.eval_latent_v25("MechA_A0_global_g55", None, [], None)  # 退出后不再重命名
    finally:
        mechanism_a.eval_latent_v25 = orig
    assert seen == ["MechA_B4anc_A2_learned_em_g55", "MechA_A0_global_g55"]
    assert mechanism_a.eval_latent_v25 is orig


# ---------------- 5) 端到端 CPU mock (measure=False, 无 ckpt / 无 GPU) ----------------

def test_run_mechanism_a_reeval_cpu_mock():
    """ctx750 mock + 随机门控: 7 臂全部产出且臂名带 B4anc 前缀; A0 能量恒等于 γ (复用语义不漂移)."""
    ctx = _mock_ctx()
    torch.manual_seed(3)
    nets = {"A2_learned": GateNet().eval(), "A3_smooth": GateNet().eval()}
    recs = run_mechanism_a_reeval(ctx, "cpu", gamma=0.55, nets=nets, measure=False)
    assert len(recs) == 7
    assert [r["arm"] for r in recs] == [p["measured_arm"] for p in mechanism_a_plan(0.55)]
    assert all(r["mech_a_tag"] == p["tag"] for r, p in zip(recs, mechanism_a_plan(0.55)))
    assert abs(recs[0]["edit_energy_mean"] - 0.55) < 1e-6, "A0 (M≡1) 能量必须恒等于名义 γ"
    assert all(math.isfinite(r["edit_energy_mean"]) for r in recs)
    assert recs[0]["tau_stats"] == ctx["tau_stats"], "rec 必须携带 ctx 的 tau_stats"
    assert "dino_c" not in recs[0], "measure=False 时不得产出裁判指标"


def test_run_mechanism_b_reeval_cpu_mock():
    """ctx750 mock + 真实小学生: γ=0 回归断言通过, 臂名 B0fix, same 行 bypass 进测量张量语义."""
    ctx = _mock_ctx()
    stu = _small_student(seed=23)
    recs = run_mechanism_b_reeval(ctx, stu, "cpu", gammas=(0.0, 0.25), measure=False)
    assert [r["arm"] for r in recs] == ["MechB_B0fix_g00", "MechB_B0fix_g25"]
    assert recs[0]["residual_mean"] == 0.0 and recs[0]["residual_max"] == 0.0
    assert recs[1]["residual_mean"] > 0.0
    assert all(len(r["residual_per_task"]) == int((~ctx["same"]).sum()) for r in recs)
    # γ=0 残差为 0 的构造性: z_by_gamma[0.0] == z_p1 逐位 (run_mechanism_b_reeval 内部断言已覆盖)


def test_no_cuda_in_tests():
    """护栏: 本测试文件必须保持纯 CPU (CUDA_VISIBLE_DEVICES='' 生效)."""
    assert not torch.cuda.is_available() or torch.cuda.device_count() == 0
