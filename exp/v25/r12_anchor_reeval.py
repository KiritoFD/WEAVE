"""V25 R12 (A08 审查补救 R1/R2, 全程零训练): B4 锚一致重评 —— 机制 A 7 臂 + 机制 B0 修复后 γ 扫描.

A08 (docs/v25/audits/a08_science_review.md §二) FAIL 根因:
  训练侧教师 z_d6_tr 用 **B4 锚** (precompute_train_latents.py:75-83,117-119), 评测侧 ctx750
  再生却用 **T4 锚** (mechanism_a.build_teacher_latents_750 → exp.v22.t4) —— A2/A3 门控与 B0 学生
  在 eval 时消费 OOD 锚输入; 且 mechanism_b.py 旧 :382 把**未 bypass 的** z_by_gamma 传给 measure,
  Same150 崩塌 (96.7%→64.7%) 是 measure 路径缺 bypass 的伪迹.

R12 补救 (评测-only, 4090):
  R1  ctx750_B4: 按 precompute_train_latents.py:75-83 配方对 test750 z_src 构建 B4 锚
        z_anc_750 = b4(canon(z_src, c1_gentle), 0, canon, tid),
      z_d6_750 = run_core_flow(slf_core_v2, anchor=z_anc_750, branches=cs_j, gate=original,
        solver=D6H_SOLVER) (spec 三元组逐字再生, 仅锚换成 B4),
      z_p1_750 = compute_prism1_latents(λ=1.0), 键对齐 build_teacher_latents_750 返回结构 (含 tau_stats);
      加载 gate_A2_learned_g55.pt / gate_A3_smooth_g55.pt (**EMA 权重**, 与原评测口径一致),
      经 renamed_mecha_measure 复用 eval_mechanism_a_arm 重评 7 臂
      (A0 / A1×2κ / A2 / A2_em / A3 / A3_em; arm 名前缀 MechA_B4anc_, 与旧 T4 锚臂不碰撞).
  R2  mechanism_b.py:382 bypass 修复后, 加载 student_B0.pt (EMA), 以 ctx750_B4 重跑
      γ∈{0,0.25,0.33,0.40,0.55,0.63,0.70} 扫描 (arm 名 MechB_B0fix_g*; same-task 走 z_src bypass);
      γ=0 residual==0 回归断言内置于 run_mechanism_b_reeval.
  CI  a01 配对 CI: 6 个 B4anc 臂 vs MechA_B4anc_A0_global_g55 (A08 附录 A 同工具/同 seed),
      另加 MechB_B0fix_g55 vs MechA_B4anc_A0_global_g55 —— A0 全局混合即 B4 教师曲线 z*(0.55)
      的同 ctx 实测点, 该 CI 即"B0fix 是否恢复到教师曲线 ±0.01"的预注册带判定.

判定解读框架 (总控读取 r12_anchor_reeval.json 时使用):
  - A2/A2_em 在锚一致后 PASS≈0 → 机制 A 判负维持, 归因注脚改"标定残差 + 结构退化(门控零信号)";
    若仍 FAIL (ΔS CI 上界 < 0 且 Minimalism 守卫违) → 判负加强 (OOD 惩罚假设被排除).
  - MechB_B0fix: same150 各 γ 应回到 ≈96.7% (bypass 伪迹证实); g55 对教师曲线 CI PASS (±带内)
    → B0 改判"欠拟合而非发散", 重试协议 (3000 upd) 获得载体; 若 g55 仍 FAIL 且 residual 随 γ 增长
    → A08 §9.2 发散结论成立.

用法 (4090, 总控执行; 幂等: raw meta / measured summary / ctx 缓存全部复用):
  python -m exp.v25.r12_anchor_reeval --pool ../pseudo_gt_24h/orig_mean_fp16
"""
from __future__ import annotations

import argparse
import contextlib
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import torch

from . import EVAL_SEED, OUT, ROOT, read_json, sanitize_cuda_env, save_json, utc
from src.infer import canonicalizer, load_c3unet
from src.utils import load_s10_decoder
from exp.v18 import tasks_and_sources
from exp.v19.fiber_duality import build_or_load_v19_foundation
from exp.v20.run_v20 import get_test750_y16_and_d16
from exp.v20.submode_patchswd import build_or_load_core_knn_bank
from exp.v24.p0_m0_m2 import V20_CKPT, compute_prism1_latents
from exp.v25.mechanism_a import (
    CANONICAL_TEACHER_SPEC,
    TRAIN_GAMMA_DEFAULT,
    GateNet,
    assert_same_teacher,
    eval_mechanism_a_arm,
    fiber_chords,
)
from exp.v25.mechanism_b import (
    B_GAMMA_SWEEP,
    B_SCALES,
    GammaResidualStudent,
    eval_student_gamma_sweep,
)
from exp.v25.precompute_train_latents import B4_CKPT
from exp.v25.r1_inference_ablation import eval_latent_v25, run_core_flow

# ---- R12 臂名 token (与旧 T4 锚臂 MechA_* / MechB_B0_* 不碰撞 → raw meta 幂等复用互不干扰) ----
MECHA_TOKEN = "B4anc"   # R1: MechA_B4anc_<tag>_g55
MECHB_TOKEN = "B0fix"   # R2: MechB_B0fix_g*
A1_KAPPAS = (0.5, 1.0)  # 与原 T4 口径一致 (MechA_A1_heuristic_k50/k100)

GATE_CKPTS = {
    "A2_learned": OUT / "mechanism_a" / "gate_A2_learned_g55.pt",
    "A3_smooth": OUT / "mechanism_a" / "gate_A3_smooth_g55.pt",
}
STUDENT_B0_CKPT = OUT / "mechanism_b" / "student_B0.pt"
CTX_CACHE = OUT / "cache" / "ctx750_b4_anchor_reeval.pt"
OUT_JSON = OUT / "r12_anchor_reeval.json"

B4_ANCHOR_RECIPE = {
    "ckpt": "runs/v6/P1_B4_distill50/P1_B4_distill50.pt",
    "canonicalizer": "src.model.canonicalizer frozen c1_gentle stats; mode='c1_gentle', augment=False",
    "formula": "z_anc = b4(canon(z_src, c1_gentle), 0.0, canon, tid)",
    "mirror_of": "exp/v25/precompute_train_latents.py:75-83 (训练侧 z_anc_tr 同配方)",
    "core_flow": "run_core_flow(slf_core_v2, anchor=z_anc_750, branches=cs_j, gate=original, solver=D6H_SOLVER)",
    "p1": "compute_prism1_latents(s2_prism1_v2, lam=1.0)",
    "eps": "torch.randn(generator=manual_seed(EVAL_SEED)) — 与 build_teacher_latents_750 同 seed",
}


# ============================================================================
# 1) 通用: same-task bypass + ctx750 校验 (纯函数, CPU 可测)
# ============================================================================

def apply_same_bypass(z: torch.Tensor, same_mask: torch.Tensor, z_src: torch.Tensor) -> torch.Tensor:
    """same-task bypass (网络之外): same 行输出覆写为 z_src, 其余行不动.

    与 run_core_flow 的 torch.where 掩码覆写及 eval_student_gamma_sweep 内部 :303 残差口径同语义;
    mechanism_b.py:382 修复后的 measure 路径消费同一语义 (A08 R2).
    """
    m = same_mask.bool()
    z_bp = z.clone()
    z_bp[m] = z_src[m]
    return z_bp


CTX_KEYS = ("tasks", "z_src", "z_p1", "z_d6", "c64", "d16", "chords", "sid", "tid",
            "same", "dec", "tau_stats")


def validate_ctx750(ctx: dict) -> None:
    """ctx750_B4 键完整性与形状一致性 (与 mechanism_a.build_teacher_latents_750 返回结构逐键对齐).

    多余键同样拒绝 (键集漂移防护); 缺键/形状不符/掩码不自洽一律 ValueError.
    """
    missing = [k for k in CTX_KEYS if k not in ctx]
    extra = [k for k in ctx if k not in CTX_KEYS]
    if missing or extra:
        raise ValueError(f"ctx750 key mismatch: missing={missing} extra={extra}; expected {list(CTX_KEYS)}")
    n = len(ctx["z_src"])
    if len(ctx["tasks"]) != n:
        raise ValueError(f"ctx750 tasks/z_src length mismatch: {len(ctx['tasks'])} vs {n}")
    for k in ("z_p1", "z_d6", "c64"):
        if tuple(ctx[k].shape) != tuple(ctx["z_src"].shape):
            raise ValueError(f"ctx750.{k} shape {tuple(ctx[k].shape)} != z_src {tuple(ctx['z_src'].shape)}")
    if ctx["d16"].shape[0] != n or ctx["d16"].shape[1] != 384 or tuple(ctx["d16"].shape[-2:]) != (16, 16):
        raise ValueError(f"ctx750.d16 must be (n,384,16,16), got {tuple(ctx['d16'].shape)}")
    if tuple(ctx["chords"].shape) != (n, 128):
        raise ValueError(f"ctx750.chords must be (n,128), got {tuple(ctx['chords'].shape)}")
    for k in ("sid", "tid"):
        if ctx[k].shape != (n,) or ctx[k].dtype != torch.long:
            raise ValueError(f"ctx750.{k} must be long (n,), got {tuple(ctx[k].shape)}/{ctx[k].dtype}")
    if ctx["same"].shape != (n,) or ctx["same"].dtype != torch.bool:
        raise ValueError(f"ctx750.same must be bool (n,), got {tuple(ctx['same'].shape)}/{ctx['same'].dtype}")
    if not torch.equal(ctx["same"], ctx["sid"] == ctx["tid"]):
        raise ValueError("ctx750.same must equal (sid == tid)")
    if not bool((~ctx["same"]).any()) or not bool(ctx["same"].any()):
        raise ValueError("ctx750 must contain both cross and same tasks")
    if ctx["dec"] is None or not isinstance(ctx["tau_stats"], dict):
        raise ValueError("ctx750.dec must be a decoder handle and tau_stats a dict")


def mechb_arm_name(gamma: float) -> str:
    """R2 臂名: MechB_B0fix_g{int(round(γ*100)):02d} (与 mechanism_b.main 同构, token 换 B0fix)."""
    return f"MechB_{MECHB_TOKEN}_g{int(round(gamma * 100)):02d}"


# ============================================================================
# 2) ctx750_B4 构建 (R1 核心; 缓存幂等)
# ============================================================================

@torch.no_grad()
def build_b4_anchor(tasks: list[dict], z_src_750: torch.Tensor, device: str,
                    batch: int = 50) -> torch.Tensor:
    """test750 B4 锚: z_anc = b4(canon(z_src, c1_gentle), 0, canon, tid).

    逐位镜像 exp/v25/precompute_train_latents.py:75-83 训练侧配方 (B4_distill50 + c1_gentle);
    tid 取任务 correct-target, same 行的锚下游被 run_core_flow 掩码覆写为 z_src (不消费).
    """
    tid = torch.tensor([t["target_style_id"] for t in tasks])
    b4, _ = load_c3unet(B4_CKPT, device)
    canon = canonicalizer(device)
    z_anc = torch.zeros_like(z_src_750, dtype=torch.float32)
    for st in range(0, len(z_src_750), batch):
        sl = slice(st, min(st + batch, len(z_src_750)))
        zb = z_src_750[sl].to(device)
        c_can = canon(zb, mode="c1_gentle", augment=False)
        z_anc[sl] = b4(c_can, 0.0, c_can, tid[sl].to(device)).cpu().float()
    del b4, canon
    torch.cuda.empty_cache()
    return z_anc


@torch.no_grad()
def build_or_load_ctx750_b4(bundle: dict, bank: dict, device: str,
                            spec=CANONICAL_TEACHER_SPEC) -> tuple[dict, torch.Tensor]:
    """构建/复用 ctx750_B4, 返回 (ctx, z_anc_750).

    ctx 键对齐 mechanism_a.build_teacher_latents_750 返回结构 (含 tau_stats); 仅 z_d6 的锚从
    T4 换成 B4, 其余 (branches/gate/solver/p1) 逐字取 spec. 缓存于 CTX_CACHE:
    teacher_spec 断言 + sid/tid 校验通过才复用; spec 变更须删除缓存重建.
    """
    tasks, z_src_750, _ = tasks_and_sources()
    sid = torch.tensor([t["source_style_id"] for t in tasks])
    tid = torch.tensor([t["target_style_id"] for t in tasks])
    same = sid == tid
    if CTX_CACHE.exists():
        ck = torch.load(CTX_CACHE, map_location="cpu", weights_only=False)
        try:
            assert_same_teacher(ck["teacher_spec"], spec, context=str(CTX_CACHE))
        except AssertionError as exc:
            raise RuntimeError(f"ctx cache {CTX_CACHE.name} teacher_spec 过期 — 删除该文件重建: {exc}") from exc
        if not (torch.equal(ck["sid"].long(), sid) and torch.equal(ck["tid"].long(), tid)):
            raise RuntimeError(f"ctx cache {CTX_CACHE.name} sid/tid 与 tasks_750 不一致 — 删除重建")
        _, d16_750 = get_test750_y16_and_d16(bundle)
        ctx = dict(tasks=tasks, z_src=z_src_750.float(), z_p1=ck["z_p1"].float(),
                   z_d6=ck["z_d6"].float(), c64=bundle["test_struct"]["c_64"].float(),
                   d16=d16_750.float(), chords=ck["chords"].float(), sid=sid, tid=tid,
                   same=same, dec=load_s10_decoder(device), tau_stats=ck["tau_stats"])
        z_anc = ck["z_anc"].float()
        print(f"[R12 ctx] Reusing {CTX_CACHE}", flush=True)
    else:
        y16_750, d16_750 = get_test750_y16_and_d16(bundle)
        eps_750 = torch.randn(z_src_750.shape, generator=torch.Generator().manual_seed(EVAL_SEED))
        z_anc = build_b4_anchor(tasks, z_src_750, device)
        print("[R12 ctx] B4 anchors done", flush=True)
        z_p1 = compute_prism1_latents(bundle, tasks, z_src_750, d16_750,
                                      lam=float(spec.p1_lambda), device=device)
        print("[R12 ctx] PRISM-1 latents done", flush=True)
        z_d6, tau_stats = run_core_flow(
            bundle, bank, V20_CKPT, tasks, z_src_750, z_anc, y16_750, d16_750, eps_750,
            branches=spec.branches, gate_mode=spec.gate, solver=dict(spec.solver), device=device,
        )
        print("[R12 ctx] B4-anchored D6 core flow done", flush=True)
        chords = fiber_chords(bundle, bundle["fnd"]["test750_fib"].float(), tid, device).cpu()
        chords[same] = 0.0  # same 任务走网络外 bypass, 弦置零保持门控输入良性 (同 build_teacher_latents_750)
        ctx = dict(tasks=tasks, z_src=z_src_750.float(), z_p1=z_p1, z_d6=z_d6,
                   c64=bundle["test_struct"]["c_64"].float(), d16=d16_750.float(),
                   chords=chords, sid=sid, tid=tid, same=same,
                   dec=load_s10_decoder(device), tau_stats=tau_stats)
        CTX_CACHE.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"z_anc": z_anc, "z_p1": z_p1, "z_d6": z_d6, "chords": chords,
                    "sid": sid, "tid": tid, "tau_stats": tau_stats,
                    "teacher_spec": spec.to_dict(), "updated_utc": utc()}, CTX_CACHE)
        print(f"[R12 ctx] Built and cached {CTX_CACHE}", flush=True)
    validate_ctx750(ctx)
    return ctx, z_anc


# ============================================================================
# 3) ckpt 加载 (运行时才加载; EMA 与原评测口径一致)
# ============================================================================

def load_gate_nets(device: str, gamma: float = TRAIN_GAMMA_DEFAULT) -> dict:
    """加载已训练门控 **EMA** 权重 (mechanism_a.main 评测消费 train_gate_arm 返回的 ema.eval())."""
    nets = {}
    for arm, p in GATE_CKPTS.items():
        if not p.exists():
            raise FileNotFoundError(f"gate ckpt missing: {p.name} — 先运行 python -m exp.v25.mechanism_a")
        ck = torch.load(p, map_location="cpu", weights_only=False)
        info = ck.get("info", {})
        if abs(float(info.get("gamma", gamma)) - gamma) > 1e-6:
            raise ValueError(f"{p.name}: ckpt gamma={info.get('gamma')} != eval gamma={gamma}")
        net = GateNet().to(device)
        net.load_state_dict(ck["ema"], strict=True)
        net.eval()
        nets[arm] = net
        print(f"[R12 MechA] loaded {p.name} (ema, n_params={info.get('n_params')}, "
              f"updates={info.get('updates')})", flush=True)
    return nets


def load_student_b0(device: str) -> tuple[GammaResidualStudent, dict]:
    """加载 B0 学生 **EMA** 权重 (mechanism_b.main 评测口径: strict=True 加载 ema)."""
    if not STUDENT_B0_CKPT.exists():
        raise FileNotFoundError(f"student ckpt missing: {STUDENT_B0_CKPT.name} — 先运行 python -m exp.v25.mechanism_b")
    ck = torch.load(STUDENT_B0_CKPT, map_location="cpu", weights_only=False)
    cfg = {k: v for k, v in B_SCALES["B0"].items() if k != "updates"}
    student = GammaResidualStudent(**cfg).to(device)
    student.load_state_dict(ck["ema"], strict=True)
    student.eval()
    print(f"[R12 MechB] loaded {STUDENT_B0_CKPT.name} (ema, info={ck.get('info', {})})", flush=True)
    return student, ck.get("info", {})


# ============================================================================
# 4) R1: 机制 A 7 臂重评 (复用 eval_mechanism_a_arm, 臂名经 renamed_mecha_measure 插 token)
# ============================================================================

def mechanism_a_plan(gamma: float) -> list[dict]:
    """R1 七臂计划: A0 / A1×2κ / A2 / A2_em / A3 / A3_em (传参语义与 mechanism_a.main 全同)."""
    gtag = f"g{int(round(gamma * 100)):02d}"
    plan = []
    for arm, net_key, kappa, em in (
        ("A0_global", None, None, False),
        ("A1_heuristic", None, A1_KAPPAS[0], False),
        ("A1_heuristic", None, A1_KAPPAS[1], False),
        ("A2_learned", "A2_learned", None, False),
        ("A2_learned", "A2_learned", None, True),
        ("A3_smooth", "A3_smooth", None, False),
        ("A3_smooth", "A3_smooth", None, True),
    ):
        tag = arm if kappa is None else f"{arm}_k{int(round(kappa * 100)):02d}"
        tag = f"{tag}_em" if em else tag
        plan.append(dict(arm=arm, net=net_key, kappa=kappa, energy_match=em, tag=tag,
                         measured_arm=f"MechA_{MECHA_TOKEN}_{tag}_{gtag}"))
    return plan


@contextlib.contextmanager
def renamed_mecha_measure(token: str):
    """临时替换 exp.v25.mechanism_a 命名空间内的 eval_latent_v25, 使 eval_mechanism_a_arm
    逐字复用 (门控/能量/bypass/记录逻辑零漂移) 而测量臂名插入 token:
    MechA_<tag>_g55 → MechA_<token>_<tag>_g55. 退出时恢复原函数."""
    import exp.v25.mechanism_a as _ma
    orig = _ma.eval_latent_v25

    def _renamed(arm_name, z_750, tasks, dec, device="cuda"):
        head, sep, rest = str(arm_name).partition("_")
        if head != "MechA" or not sep:
            raise ValueError(f"renamed_mecha_measure expects 'MechA_*' arm, got {arm_name!r}")
        return orig(f"{head}_{token}_{rest}", z_750, tasks, dec, device=device)

    _ma.eval_latent_v25 = _renamed
    try:
        yield _renamed
    finally:
        _ma.eval_latent_v25 = orig


def run_mechanism_a_reeval(ctx: dict, device: str, gamma: float = TRAIN_GAMMA_DEFAULT,
                           nets: dict | None = None, measure: bool = True,
                           on_result=None) -> list[dict]:
    """R1: ctx750_B4 上复用 eval_mechanism_a_arm 重评 7 臂 (arm 名 MechA_B4anc_*)."""
    validate_ctx750(ctx)
    nets = load_gate_nets(device, gamma=gamma) if nets is None else nets
    results = []
    cm = renamed_mecha_measure(MECHA_TOKEN) if measure else contextlib.nullcontext(None)
    with cm:
        for item in mechanism_a_plan(gamma):
            rec = eval_mechanism_a_arm(ctx, item["arm"], gamma, device,
                                       net=nets.get(item["net"]) if item["net"] else None,
                                       kappa=item["kappa"], energy_match=item["energy_match"],
                                       measure=measure)
            rec["mech_a_tag"] = rec["arm"]       # 原始 tag (如 A2_learned_em), 供与旧 JSON 对照
            rec["arm"] = item["measured_arm"]    # 统一为测量臂名 MechA_B4anc_*_g55
            results.append(rec)
            if on_result is not None:
                on_result(rec)
            print(f"[R12 MechA] {rec['arm']}: dino_c={rec.get('dino_c')} dino_s={rec.get('dino_s')} "
                  f"lpips={rec.get('lpips')} top1={rec.get('csd_top1')} "
                  f"E={rec['edit_energy_mean']}", flush=True)
    return results


# ============================================================================
# 5) R2: 机制 B0 修复后 γ 扫描重评 (same-task 走 z_src bypass)
# ============================================================================

def run_mechanism_b_reeval(ctx: dict, student: GammaResidualStudent, device: str,
                           gammas=B_GAMMA_SWEEP, measure: bool = True,
                           on_result=None) -> list[dict]:
    """R2: 修复后 measure 路径 (same-task bypass) + ctx750_B4 重跑 γ 扫描 (arm 名 MechB_B0fix_g*).

    回归断言: γ=0 residual 必须精确为 0 (构造性回退 z_p1 + 0·R_θ ≡ z_p1, 教师 z*(0)=z_p1;
    A08 §9.2 认定的可靠工程结论, bypass 修复不得破坏).
    """
    validate_ctx750(ctx)
    cond = dict(c64=ctx["c64"], d16=ctx["d16"], chord=ctx["chords"])

    def student_fn(zp, gv, cb):
        return student(zp, cb["c64"], cb["d16"], cb["chord"], gv).to(device)

    sweep = eval_student_gamma_sweep(student_fn, ctx["z_src"], ctx["z_p1"], ctx["z_d6"],
                                     ctx["same"], ctx["tid"], cond=cond, gammas=gammas,
                                     device=device)
    g0 = next(r for r in sweep["records"] if r["gamma"] == 0.0)
    assert g0["residual_mean"] == 0.0 and g0["residual_max"] == 0.0, (
        f"γ=0 回退回归断言失败: residual_mean={g0['residual_mean']} (应为精确 0)")
    records = []
    for rec in sweep["records"]:
        rec["arm"] = mechb_arm_name(rec["gamma"])
        if measure:
            z_bp = apply_same_bypass(sweep["z_by_gamma"][rec["gamma"]], ctx["same"], ctx["z_src"])
            m = eval_latent_v25(rec["arm"], z_bp, ctx["tasks"], ctx["dec"], device=device)
            rec["metrics"] = {k: v for k, v in m.items() if k != "arm"}
        records.append(rec)
        if on_result is not None:
            on_result(rec)
        mt = rec.get("metrics", {})
        print(f"[R12 MechB] {rec['arm']}: res={rec['residual_mean']:.4f} "
              f"dino_c={mt.get('dino_c')} dino_s={mt.get('dino_s')} same={mt.get('same_top1')}",
              flush=True)
    return records


# ============================================================================
# 6) a01 配对 CI + 参照读数 (报告层)
# ============================================================================

def compute_paired_ci(pairs: list[tuple[str, str]], n_boot: int = 1500, seed: int = 2026) -> dict:
    """a01 配对源聚类 bootstrap CI (A08 附录 A 同工具; 方向 = candidate − baseline).

    pairs = [(candidate_arm, baseline_arm), ...]; 基线 = MechA_B4anc_A0_global_g55 (同 ctx750_B4
    内部自洽配对). eval.json 缺失时记 error 不中断 (幂等重跑友好).
    """
    from tools.v25_cpu.a01_noninferiority import compare_arms
    ci = {}
    for cand, base in pairs:
        try:
            res = compare_arms(cand, base, n_boot=n_boot, seed=seed)
            ci[cand] = dict(baseline=base, direction=f"{cand} - {base}", verdict=res["verdict"],
                            final=res["final"], reasons=res["reasons"], ci=res["ci"],
                            guards=res["guards"], n_tasks=res["n_tasks"], n_clusters=res["n_clusters"])
            print(f"[R12 CI] {cand} vs {base}: {res['final']} {res['reasons']}", flush=True)
        except (FileNotFoundError, ValueError) as exc:
            ci[cand] = dict(baseline=base, error=f"{type(exc).__name__}: {exc}")
            print(f"[R12 CI] {cand}: skipped ({exc})", flush=True)
    return ci


def load_old_reference() -> dict:
    """旧 T4 锚口径读数摘录 (runs/v25/mechanism_{a,b}/*.json, 只读) — 供 lineage 差分解对照."""
    ref = {}
    pa = OUT / "mechanism_a" / "mechanism_a.json"
    if pa.exists():
        d = read_json(pa)
        ref["mechanism_a_t4_anchor"] = [
            {k: r.get(k) for k in ("arm", "dino_c", "dino_s", "lpips", "csd_top1", "edit_energy_mean")}
            for r in d.get("results", [])]
    pb = OUT / "mechanism_b" / "mechanism_b.json"
    if pb.exists():
        d = read_json(pb)
        b0 = (d.get("results") or {}).get("B0") or {}
        ref["mechanism_b0_t4_anchor"] = [
            {"gamma": r.get("gamma"), "residual_mean": r.get("residual_mean"),
             "metrics": r.get("metrics")} for r in b0.get("records", [])]
    return ref


@torch.no_grad()
def anchor_lineage_stats(ctx: dict, z_anc: torch.Tensor, bundle: dict, device: str) -> dict | None:
    """B4 锚 lineage 诊断 (仅当 v22 T4 缓存命中时计算, 绝不触发 T4 重生成):
    cross600 任务级 L2 的 p50 —— d(z_anc_B4, z_src) / d(z_anc_B4, z_t4) / d(z_t4, z_src) 与比值,
    直接量化 A08 §二 的 B4↔T4 锚差是否在 test750 上同样非平凡."""
    from exp.v22 import t4 as _t4
    if not _t4.T4_V22_CACHE.exists():
        return None
    z_t4, _ = _t4.build_or_load_t4_latents_750(bundle, device=device)
    cross = ~ctx["same"]

    def p50(a, b):
        d = (a.float() - b.float()).flatten(1).norm(dim=1)
        return round(float(d[cross].median()), 3)

    d_as, d_at, d_ts = p50(z_anc, ctx["z_src"]), p50(z_anc, z_t4), p50(z_t4, ctx["z_src"])
    return {"l2_p50_ancb4_vs_src": d_as, "l2_p50_ancb4_vs_t4": d_at, "l2_p50_t4_vs_src": d_ts,
            "ratio_ancb4_t4_over_ancb4_src": round(d_at / max(d_as, 1e-8), 3), "scope": "cross600"}


# ============================================================================
# 7) main (4090 入口; 本任务只交付, 不启动 GPU)
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gamma", type=float, default=TRAIN_GAMMA_DEFAULT,
                    help="机制 A 工作点 (须与 gate_*_g55.pt 一致)")
    ap.add_argument("--gammas", default=",".join(str(g) for g in B_GAMMA_SWEEP))
    ap.add_argument("--skip-a", action="store_true", help="跳过机制 A 重评")
    ap.add_argument("--skip-b", action="store_true", help="跳过机制 B 重评")
    ap.add_argument("--skip-measure", action="store_true",
                    help="只构建 ctx 与 latent 统计, 不写 raw/measure (调试)")
    ap.add_argument("--skip-ci", action="store_true", help="跳过 a01 配对 CI")
    args = ap.parse_args()

    sanitize_cuda_env()
    measure = not args.skip_measure
    gammas = tuple(float(g) for g in str(args.gammas).split(",") if g.strip())
    out = dict(task="R12", updated_utc=utc(), gamma=args.gamma, gammas=list(gammas),
               b4_anchor_recipe=B4_ANCHOR_RECIPE,
               teacher_spec=CANONICAL_TEACHER_SPEC.to_dict(),
               mechanism_a=[], mechanism_b=None, paired_ci=None,
               anchor_lineage=None, reference_old_t4=load_old_reference())

    def flush():
        save_json(OUT_JSON, out)

    bundle = build_or_load_v19_foundation(args.pool, device=args.device)
    bank = build_or_load_core_knn_bank(bundle, k_sub=8, core_per_style=180, device=args.device)
    ctx, z_anc = build_or_load_ctx750_b4(bundle, bank, device=args.device)
    out["ctx750_b4"] = dict(
        cache="runs/v25/cache/ctx750_b4_anchor_reeval.pt",
        tau_stats=ctx["tau_stats"],
        anchor_l2_p50_vs_src=round(float((z_anc - ctx["z_src"]).flatten(1).norm(dim=1).median()), 3),
    )
    out["anchor_lineage"] = anchor_lineage_stats(ctx, z_anc, bundle, args.device)
    flush()

    if not args.skip_a:
        collected_a: list[dict] = []
        out["mechanism_a"] = collected_a
        run_mechanism_a_reeval(ctx, args.device, gamma=args.gamma, measure=measure,
                               on_result=lambda rec: (collected_a.append(rec), flush()))

    if not args.skip_b:
        student, s_info = load_student_b0(args.device)
        out["student_b0_info"] = s_info
        collected_b: list[dict] = []
        out["mechanism_b"] = dict(gammas=list(gammas), records=collected_b)
        run_mechanism_b_reeval(ctx, student, args.device, gammas=gammas, measure=measure,
                               on_result=lambda rec: (collected_b.append(rec), flush()))
        flush()

    if measure and not args.skip_ci:
        baseline = mechanism_a_plan(args.gamma)[0]["measured_arm"]
        pairs = [(r["arm"], baseline) for r in out["mechanism_a"] if r["arm"] != baseline]
        if not args.skip_b:
            # A0 全局混合 = B4 教师曲线 z*(γ=0.55) 的同 ctx 实测点 → 该 CI 即
            # "B0fix 恢复到教师曲线 ±带" 的预注册判定 (A08 R2 解读框架).
            pairs.append((mechb_arm_name(0.55), baseline))
        out["paired_ci"] = compute_paired_ci(pairs)
        flush()

    print(f"\n=== R12 完成: {len(out['mechanism_a'])} MechA_B4anc_* + "
          f"{len((out['mechanism_b'] or {}).get('records', []))} MechB_B0fix_* 臂 -> {OUT_JSON.name} ===",
          flush=True)
    for r in out["mechanism_a"]:
        print(f"{r['arm']:34s}: dino_c={r.get('dino_c')} dino_s={r.get('dino_s')} "
              f"lpips={r.get('lpips')} top1={r.get('csd_top1')}", flush=True)
    for r in (out["mechanism_b"] or {}).get("records", []):
        mt = r.get("metrics", {})
        print(f"{r['arm']:34s}: res={r['residual_mean']:.4f} dino_c={mt.get('dino_c')} "
              f"dino_s={mt.get('dino_s')} same={mt.get('same_top1')}", flush=True)


if __name__ == "__main__":
    main()
