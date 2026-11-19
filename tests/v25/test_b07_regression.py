"""B07: Minimal PRISM CPU 测试套件整合 —— 只补缺口, 不重复既有测试 (agent2.md §六 B07).

现有 9 个测试文件 (test_stats / test_mechanism / test_mechanism_c / test_a02_lineage /
test_a03_teacher / test_registry / test_result_writer / test_prism_v25 / test_r12) 已覆盖
B07 清单 11 项中的大部分 (覆盖矩阵见 docs/v25/audits/b07_coverage_report.md)。本文件只补
盘点出的真缺口:

  1. 输出 schema: eval_latent_v25 返回 dict 的键集合与类型 (合成 summary + fake OUT, 纯 CPU
     复用"raw meta 完整 → 直接读 measured summary"分支, 不跑 GPU);
  2. 缓存状态机分支: 空缓存 (无 raw meta) → 必须走再生成分支 (sentinel 拦截, 证明不静默);
     H1 半完成窗口 (meta 完整但 summary 缺失, B06 §1.2) → 响亮 FileNotFoundError;
  3. ReferenceFeatureCache 损坏 .pt 条目 → 响亮失败 (绝不静默重编码/返回垃圾, 绝不删改文件);
  4. Windows 路径: ArtifactRegistry 血缘键的 posix/反斜杠/大小写等价 (4090 Linux 写 ↔
     3060 Windows 读 的跨平台契约, normcase 语义);
  5. One-Shot 参考变化: 复用 SLFUNetV24OneShot 对参考替换必须响应, 同参考逐位确定;
  6. A01 验收第 5 条防线: 均值外推伪造的常数配对样本能通过 a01 结构闸 (pair_cross_rows /
     compare_arms 不抛), 但必然产生零宽度退化 CI —— 本文件提供上游断言包装
     assert_paired_ci_not_degenerate 作为机器化防线 (只测不改 a01 本体);
  7. CPU-only 全链冒烟: CUDA_VISIBLE_DEVICES="" 子进程按文件路径导入 tests/v25 全部测试
     模块 (注: 本机 site-packages 存在同名 'tests' 包遮蔽仓库 namespace package,
     'import tests.v25' 字面不可用 —— B07 新发现, 见覆盖报告), 以 B02 发现的 NVML 陷阱
     判据 (torch.cuda.is_initialized()==False 为权威) + device_count()==0 辅证。

全 CPU / tmp_path / 无真实 checkpoint / 无 CUDA / 无绝对路径。
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # 必须在 import torch 之前 → 纯 CPU
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import exp.v25.r1_inference_ablation as r1_mod
from exp.v24.p0_o1_o3_value import SLFUNetV24OneShot
from exp.v25.reference_teacher_v2 import ReferenceFeatureCache
from src.v25.artifact_registry import ArtifactRegistry
from tools.v25_cpu.a01_noninferiority import (
    BAND_METRICS,
    SEED,
    VERDICT_PASS,
    compare_arms,
    pair_cross_rows,
)

torch.manual_seed(0)

#: eval_latent_v25 返回 schema 的权威键集 (r1_inference_ablation.py:81-85)
EVAL_LATENT_V25_KEYS = {
    "arm", "dino_c", "dino_s", "lpips", "style_affinity", "csd_top1", "same_top1",
}

_STYLES = ("Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e")


# --------------------------------------------------------------------------- #
# 公共小工具
# --------------------------------------------------------------------------- #
def _fake_raw_meta(n_tasks: int = 750, batch: int = 25) -> dict:
    """镜像 RawWriter meta.json 结构 (src/utils/raw_io.py): batches[] 含 ids,
    恰好 n_tasks 个唯一 task_id (validate_raw_meta 的 750 唯一 id 闸)。"""
    batches = []
    for off in range(0, n_tasks, batch):
        ids = [f"task_{off + i:04d}" for i in range(min(batch, n_tasks - off))]
        batches.append({"offset": off, "file": f"batch_{off:05d}.pt", "ids": ids})
    return {"kind": "v25", "arm": "test_arm", "n_tasks": n_tasks,
            "decoder": "S10", "batches": batches}


def _protocol_rows(dino_c: float, dino_s: float, lpips: float, *,
                   noise: float = 0.0, seed: int = 0, top1: float = 1.0,
                   n_sources: int = 150, tasks_per_source: int = 4) -> list[dict]:
    """完整协议形态的 cross 行 (150 源 × 4 任务 = 600): noise=0 即 A01 禁止的
    '均值外推伪造' 形态 (常数行); noise>0 为含任务级噪声的真实评测形态。
    tgt 分配对 seed 确定 → simple/complex 两臂 style 集合恒一致 (_guards 契约)。"""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_sources):
        stem = f"src_{i:03d}"
        for j in range(tasks_per_source):
            tgt = _STYLES[(i + j + 1) % 5]
            n = noise * float(rng.standard_normal())
            rows.append({
                "task_id": f"{stem}/{tgt}", "src_stem": stem,
                "src_style": _STYLES[i % 5], "tgt_style": tgt,
                "is_cross": True,
                "dino_c": dino_c + n, "dino_s": dino_s + n, "lpips": lpips + n,
                "target_top1": top1,
            })
    return rows


def assert_paired_ci_not_degenerate(res: dict, tol: float = 1e-12) -> None:
    """B07 上游防线 (只测不改 a01 本体): compare_arms 结果进入任何删除/保留决策前必须过本闸。

    全部 band 指标 CI 宽度为 0 (lo == hi == mean) 意味着配对样本是常数伪造 (均值外推) ——
    真实 600 任务评测必有任务级噪声, 不可能三指标同时零宽度。这是 agent2.md A01 验收
    第 5 条 ('无法凭均值外推构造假的配对样本') 的机器化防线; B08 可将本函数提升进
    a01 的调用方 (compare_arms 返回后、verdict 消费前)。
    """
    degenerate = [m for m in BAND_METRICS
                  if float(res["ci"][m]["hi"]) - float(res["ci"][m]["lo"]) <= tol]
    if degenerate:
        raise ValueError(
            f"degenerate paired CI (zero width) on {degenerate}: paired rows look "
            "fabricated / mean-extrapolated; real per-task evals carry task-level "
            "noise. Refusing to use as decision input (A01 acceptance #5)."
        )


# --------------------------------------------------------------------------- #
# 1) 输出 schema: eval_latent_v25 返回 dict (CPU 分支: meta 完整 → 读 measured summary)
# --------------------------------------------------------------------------- #
def test_eval_latent_v25_output_schema_from_cached_summary(tmp_path, monkeypatch):
    """键集合与类型钉死: 7 键, arm 为 str, 其余为 5 位舍入 float, 值逐项来自 summary。
    (走 'raw meta 完整 → 直接读 summary' 分支, 不触 GPU。)"""
    arm = "schema_arm"
    (tmp_path / "raw" / arm).mkdir(parents=True)
    (tmp_path / "raw" / arm / "meta.json").write_text(
        json.dumps(_fake_raw_meta(750)), encoding="utf-8")
    summary = {
        "cross600": {"dino_c": 0.654321, "dino_s": 0.50123456, "lpips": 0.52109876,
                     "style_affinity": 0.6012345, "csd_top1": 0.443277},
        "same150": {"csd_top1": 0.99333},
    }
    (tmp_path / "measured" / arm).mkdir(parents=True)
    (tmp_path / "measured" / arm / "summary_v25.json").write_text(
        json.dumps(summary), encoding="utf-8")
    monkeypatch.setattr(r1_mod, "OUT", tmp_path)

    res = r1_mod.eval_latent_v25(arm, None, [], None)

    assert set(res) == EVAL_LATENT_V25_KEYS, f"schema 键集漂移: {set(res) ^ EVAL_LATENT_V25_KEYS}"
    assert isinstance(res["arm"], str) and res["arm"] == arm
    for k in ("dino_c", "dino_s", "lpips", "style_affinity", "csd_top1"):
        assert isinstance(res[k], float), k
        assert res[k] == round(summary["cross600"][k], 5), k
    assert isinstance(res["same_top1"], float)
    assert res["same_top1"] == round(summary["same150"]["csd_top1"], 5)
    # 钉死 5 位舍入口径 (台账对比约定): 原值必须被舍入而非透传
    assert res["dino_c"] != summary["cross600"]["dino_c"]  # 0.654321 → 0.65432


def test_eval_latent_v25_cache_state_branches(tmp_path, monkeypatch):
    """缓存状态机两分支 (对应 B06 §2 矩阵):
    (1) 空缓存 (无 raw meta) → 必须走再生成分支 (GPU 路径) —— sentinel 拦截证明
        绝不静默吞掉空缓存直接'编造'结果;
    (2) H1 半完成窗口 (meta 完整但 measured summary 缺失) → 响亮 FileNotFoundError,
        与 B06 的 scan_status 'summary_missing' flag / '需补跑 measure' 语义一致。"""
    monkeypatch.setattr(r1_mod, "OUT", tmp_path)
    arm = "branch_arm"

    def _no_gpu_writer(*args, **kwargs):
        raise RuntimeError("GPU generation branch reached — forbidden in CPU tests")

    monkeypatch.setattr(r1_mod, "RawWriterV25", _no_gpu_writer)
    with pytest.raises(RuntimeError, match="GPU generation branch"):
        r1_mod.eval_latent_v25(arm, None, [], None)  # 空缓存 → 再生成分支

    (tmp_path / "raw" / arm).mkdir(parents=True)
    (tmp_path / "raw" / arm / "meta.json").write_text(
        json.dumps(_fake_raw_meta(750)), encoding="utf-8")
    with pytest.raises(FileNotFoundError):  # H1: meta 有效 + summary 缺失 → 响亮, 不静默
        r1_mod.eval_latent_v25(arm, None, [], None)


# --------------------------------------------------------------------------- #
# 2) 损坏缓存: ReferenceFeatureCache .pt 条目损坏 → 响亮失败
# --------------------------------------------------------------------------- #
def test_ref_feature_cache_corrupt_entry_fails_loud(tmp_path):
    """损坏 (截断) 的 .pt 缓存条目: torch.load 必须响亮抛错 —— 绝不静默重编码
    (encode_fn 不得被调用)、绝不返回垃圾、绝不自动删除/改写磁盘条目。
    (与 b06 read_json_tolerant 的 JSON 宽容读 / b03 registry 坏行跳过相区分:
    特征缓存的损坏以响亮失败暴露, 由人工或上层决定重编, 不做静默自愈。)"""
    cache = ReferenceFeatureCache(tmp_path)
    key = "deadbeef" * 8  # tensor_sha 形态 (64 hex)
    p = cache.dir / f"ref_{key}.pt"
    payload = b"PK\x03\x04corrupted-truncated-entry-payload"  # 截断 zip 头
    p.write_bytes(payload)

    encode_calls = []
    with pytest.raises(Exception):  # torch.load 具体异常类型随版本 (BadZipFile/UnpicklingError/…)
        cache.get_or_encode(key, lambda: encode_calls.append(1) or {})
    assert encode_calls == [], "损坏条目绝不能静默重编码"
    assert p.read_bytes() == payload, "损坏条目绝不能被自动删除/改写"
    # 空缓存路径 (无条目 → miss → encode) 已由 test_mechanism_c 覆盖, 此处不重复


# --------------------------------------------------------------------------- #
# 3) Windows 路径: ArtifactRegistry 血缘键跨平台等价
# --------------------------------------------------------------------------- #
def test_registry_lookup_windows_path_key_equivalence(tmp_path):
    """4090 (Linux, posix 分隔符) 写入的血缘注册表在 3060 (Windows, 反斜杠/大小写)
    读取必须命中同一记录 —— _path_key 的 resolve+normcase 语义是跨平台队列契约。
    既有测试只用同形态 Path 对象查过 (test_registry), 字符串形态等价未钉。"""
    art = tmp_path / "cache.pt"
    art.write_bytes(b"payload")
    reg = ArtifactRegistry(tmp_path / "lineage" / "artifacts.jsonl")
    reg.register(art, "cache")

    assert reg.lookup(art) is not None                      # Path 对象 (既有形态)
    assert reg.lookup(str(art)) is not None                 # 原生分隔符字符串 (Win 反斜杠)
    assert reg.lookup(art.as_posix()) is not None           # posix 字符串 (Linux 写入形态)
    assert reg.lookup(str(art).replace("\\", "/")) is not None
    assert reg.lookup(art.as_posix().replace("\\", "/")) is not None
    # normcase 大小写不敏感是 Windows 专属语义 (Linux normcase 为恒等, 大小写敏感)
    if os.path.normcase("A") == "a":
        assert reg.lookup(str(art).upper()) is not None
        assert reg.lookup(art.as_posix().upper()) is not None
    # 查到的就是刚登记的那条 (kind 往返)
    assert reg.lookup(art).kind == "cache"


# --------------------------------------------------------------------------- #
# 4) One-Shot 参考变化: 复用模型必须响应参考替换
# --------------------------------------------------------------------------- #
def test_oneshot_output_responds_to_reference_change():
    """B07 清单 'One-Shot 参考变化': 参考条件 (d16_ref/v16_ref) 替换 → 输出必须改变;
    同参考重复前向 → 逐位确定 (eval 模式, 无 dropout)。既有覆盖只有缓存键 (参考内容
    sha 变 → 新键) 与单参考烟测, 未钉'模型输出对参考的实际响应'。

    两处零初始化热启动约定 (SLFUNet.out_conv / RefAttnV24.out_proj) 使初始化网络
    输出恒为 0 且参考分支恒为 0 —— 本测试按 test_mechanism 的 randomize_out 模式
    刻意打破, 以考察参考条件到输出的通路真实存在 (而非数值恒等的巧合)。"""
    torch.manual_seed(29)
    fm = SLFUNetV24OneShot(v_dim=192, use_conf_gate=True, use_global_fib=True).eval()
    torch.manual_seed(41)  # 打破 out_conv (输出头) 与 ref_attn16.out_proj (参考出射) 零初始化
    for mod in (fm.out_conv, fm.ref_attn16.out_proj):
        torch.nn.init.normal_(mod.weight, std=0.05)
        torch.nn.init.normal_(mod.bias, std=0.05)
    g = torch.Generator().manual_seed(31)
    B = 1
    z = torch.randn(B, 4, 64, 64, generator=g)
    y16 = torch.randn(B, 1, 16, 16, generator=g)
    d16 = torch.randn(B, 384, 16, 16, generator=g)
    u = torch.randn(B, 128, generator=g)
    tau = torch.rand(B, generator=g)
    d16r = torch.randn(B, 384, 16, 16, generator=g)
    v16r = torch.randn(B, 192, 16, 16, generator=g)
    ug = torch.randn(B, 795, generator=g)

    def fwd(d16_ref, v16_ref):
        with torch.no_grad():
            return fm(z, y16, d16, torch.zeros(B, dtype=torch.long), u, tau,
                      d16_ref=d16_ref, v16_ref=v16_ref, u_glob=ug)

    o_ref = fwd(d16r, v16r)
    assert torch.equal(o_ref, fwd(d16r, v16r)), "同参考必须逐位确定 (eval, 无随机层)"
    # 参考替换: 强扰动参考条件 (缩放/平移/翻转) → 输出必须响应
    o_alt = fwd(d16r * 3.0 + 1.0, v16r.flip(-1))
    assert not torch.equal(o_ref, o_alt), "参考替换后输出必须改变 (One-Shot 条件必须实际生效)"
    assert o_ref.shape == o_alt.shape and bool(torch.isfinite(o_alt).all())


# --------------------------------------------------------------------------- #
# 5) A01 验收第 5 条: 均值外推伪造样本的防线 (只测不改 a01 本体)
# --------------------------------------------------------------------------- #
def test_forged_mean_extrapolated_rows_pass_structural_gate_but_fail_ci_guard():
    """A01 禁止的均值外推伪造 (常数行) 能通过 a01 全部结构闸 (配对/计数/簇完整性/有限值
    —— pair_cross_rows 与 compare_arms 均不抛), 三指标 CI 退化为零宽度 (可检测签名)。
    上游防线 assert_paired_ci_not_degenerate 必须在此形态上抛错; 对照组: 含任务级噪声
    的真实形态行 CI 非退化 → 闸放行。完整协议规模 (600 cross / 150 簇 × 4)。"""
    rows_c = _protocol_rows(0.70, 0.50, 0.53, seed=0)
    rows_s = _protocol_rows(0.695, 0.498, 0.535, seed=0)  # 均值外推差 (常数)

    # (i) 结构闸不拦常数伪造 —— a01 的自动防线缺位, 由上游包装补 (报告已记录该残差)
    a, b, clusters = pair_cross_rows(rows_s, rows_c)  # 不抛
    assert len(a) == len(b) == 600 and len(set(clusters)) == 150

    res = compare_arms("forged_simple", "forged_complex",
                       rows_simple=rows_s, rows_complex=rows_c,
                       n_boot=400, seed=SEED)
    assert res["verdict"] == VERDICT_PASS  # 伪造行会给出"完美 PASS" —— 危险形态
    for m in BAND_METRICS:
        ci = res["ci"][m]
        assert ci["lo"] == ci["hi"] == ci["mean"], f"{m}: 常数伪造 → 零宽度 CI"
    with pytest.raises(ValueError, match="degenerate"):
        assert_paired_ci_not_degenerate(res)  # 防线拦截

    # (ii) 对照: 含任务级噪声的真实形态行 → CI 非退化 → 闸放行
    rows_c_n = _protocol_rows(0.70, 0.50, 0.53, noise=0.003, seed=11)
    rows_s_n = _protocol_rows(0.695, 0.498, 0.535, noise=0.003, seed=12)
    res2 = compare_arms("noisy_simple", "noisy_complex",
                        rows_simple=rows_s_n, rows_complex=rows_c_n,
                        n_boot=400, seed=SEED)
    for m in BAND_METRICS:
        assert res2["ci"][m]["hi"] > res2["ci"][m]["lo"], f"{m}: 真实噪声行 CI 必须非退化"
    assert_paired_ci_not_degenerate(res2)  # 不抛


# --------------------------------------------------------------------------- #
# 6) CPU-only 全链冒烟 (agent2.md §六 B07 'CPU-only 运行')
# --------------------------------------------------------------------------- #
def test_cpu_only_full_suite_import_smoke():
    """CUDA_VISIBLE_DEVICES='' 子进程按文件路径导入 tests/v25 全部测试模块 (全链),
    以 B02 发现的 NVML 陷阱判据为准: torch.cuda.is_initialized()==False 为权威
    (device_count() 在 torch 2.13 的 NVML 路径下可能失真, 仅作辅证 —— 本机全新
    子进程实测为 0, 与 test_prism_v25 的 lazy-import 判据一致)。

    注: 不能用 'import tests.v25.<mod>' —— 本机 site-packages 存在同名 'tests'
    常规包 (遮蔽仓库 namespace package, sys.path 首位的 cwd 只贡献 namespace
    portion, 常规包胜出); pytest 下模块名实为 v25.test_*。按文件路径加载最稳。"""
    repo = Path(__file__).resolve().parents[2]
    code = "\n".join((
        "import sys, os, json, glob, importlib.util",
        "sys.path.insert(0, os.getcwd())",
        "files = sorted(glob.glob(os.path.join('tests', 'v25', 'test_*.py')))",
        "loaded = []",
        "for f in files:",
        "    name = 'v25_smoke_' + os.path.basename(f)[:-3]",
        "    spec = importlib.util.spec_from_file_location(name, f)",
        "    mod = importlib.util.module_from_spec(spec)",
        "    spec.loader.exec_module(mod)",
        "    loaded.append(os.path.basename(f))",
        "import torch",
        "print(json.dumps({'modules': loaded,"
        " 'cuda_initialized': torch.cuda.is_initialized(),"
        " 'device_count': torch.cuda.device_count()}))",
    ))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(repo), env=env,
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, f"stdout={proc.stdout[-1000:]} stderr={proc.stderr[-2000:]}"
    info = json.loads(proc.stdout.strip().splitlines()[-1])
    assert len(info["modules"]) >= 10 and "test_b07_regression.py" in info["modules"]
    assert info["cuda_initialized"] is False, \
        "CUDA context 被初始化 (存在隐藏 GPU 触点) —— B02 NVML 陷阱判据阳性"
    assert info["device_count"] == 0, \
        f"device_count={info['device_count']} (若 NVML 在其它机器使该值失真, " \
        "以 is_initialized=False 为权威判据后放宽本条)"


def test_no_cuda_in_tests():
    """护栏: 本测试文件必须保持纯 CPU (CUDA_VISIBLE_DEVICES='' 生效)。"""
    assert os.environ.get("CUDA_VISIBLE_DEVICES", "unset") == "", "CVD='' 环境契约失效"
    assert not torch.cuda.is_initialized(), "CUDA context 被初始化 (存在隐藏 GPU 触点)"
    assert torch.zeros(3).device.type == "cpu"
