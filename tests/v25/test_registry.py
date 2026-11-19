"""B03 CPU tests: 统一 checkpoint/cache 血缘检查器 (src/v25/artifact_registry.py).

验收项 (agent2.md §四 B03 / 任务卡):
- register -> validate 通过;
- 篡改依赖 -> invalid + invalidate_log 追加 (who/when/which-dep-mismatch);
- param_hash 确定性 / 键序无关 (并与 A02 solver_hash / exp.v22.config_hash 对齐, 防两套 hash);
- probe_legacy 无记录 -> unknown 而非报错;
- format_version 检查 (旧格式被 min_format_version 拦截, B02 ref_feats 教训);
- JSONL 逐行可解析且 append-only (重复 register 不覆盖).

全 CPU / tmp_path / 纯 stdlib; 绝不触碰仓库真实缓存; 绝不删除任何文件
(测试中删除/覆写的只是 pytest tmp_path 内自建文件)。
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import json

import pytest

from src.v25 import (
    FORMAT_REF_FEATS_V1_2D_NO_BATCH,
    FORMAT_REF_FEATS_V2_BATCH_DIM,
    STATUS_INVALID,
    STATUS_MISSING,
    STATUS_UNKNOWN,
    STATUS_VALID,
    ArtifactRegistry,
    DEP_CKPT_SHA256,
    DEP_PARAMS_HASH,
    file_sha256,
    normalize_dep_value,
    param_hash,
    resolve_git_sha,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
GIT = "0123456789abcdef0123456789abcdef01234567"

# A02 登记的 D6 solver 字面量 (tools/v25_cpu/a02_teacher_lineage.py:324-328),
# 仅用于与 A02 solver_hash / exp.v22.config_hash 的对齐漂移测试。
D6_SOLVER_LIT = dict(
    tau_min=0.40, tau_hi=0.74, steps=8,
    ws_fig=4.4, wsp_fig_late=1.8, wsp_abs=4.8,
    wc_fig=1.95, wc_abs=1.00, lscale_abs=0.44, rescale_phi=0.48,
)

try:
    from tools.v25_cpu.a02_teacher_lineage import solver_hash as a02_solver_hash
    A02_IMPORTABLE = True
except Exception:  # pragma: no cover - environment dependent
    A02_IMPORTABLE = False

try:
    import exp.v22 as _v22
    V22_IMPORTABLE = True
except Exception:  # pragma: no cover - environment dependent
    V22_IMPORTABLE = False


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture()
def reg(tmp_path):
    return ArtifactRegistry(tmp_path / "lineage" / "artifacts.jsonl")


def _make_artifact(tmp_path, name="cache.pt", payload=b"payload-v1"):
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(payload)
    return p


def _teacher_like_deps(params: dict) -> dict:
    """与 train_latents_4400.pt 接入提案同形的 depends (ckpt 列表 + params dict)。"""
    return {
        DEP_CKPT_SHA256: [SHA_A, SHA_B],
        DEP_PARAMS_HASH: params,
        "decoder_version": "s10_p5_lowlr_sha12",
        "feature_encoder_version": "dinov2_small_patch16_224",
    }


# --------------------------------------------------------------------------- #
# 1) register -> validate 通过
# --------------------------------------------------------------------------- #
def test_register_then_validate_pass(tmp_path, reg):
    art = _make_artifact(tmp_path, "teacher_latents.pt")
    params = {"solver": D6_SOLVER_LIT, "branches": "cs_j", "seed": 20263502}
    rec = reg.register(art, "teacher_latent_cache", depends=_teacher_like_deps(params),
                       format_version=2, git_sha=GIT, notes="B03 smoke")

    assert rec.artifact_path.endswith("teacher_latents.pt")
    assert rec.kind == "teacher_latent_cache"
    assert rec.git_sha == GIT
    assert rec.created_utc  # ISO 时间戳已自动登记
    assert rec.format_version == 2
    # dict 依赖在登记时已 param_hash 归一
    assert rec.depends[DEP_PARAMS_HASH] == param_hash(params)
    assert isinstance(rec.depends[DEP_CKPT_SHA256], str)

    res = reg.validate(art, depends=_teacher_like_deps(params), who="unit-test")
    assert res["status"] == STATUS_VALID
    assert res["mismatches"] == {}
    assert res["reasons"] == []
    assert res["dep_check"] == "performed"
    assert res["record"]["git_sha"] == GIT
    # validate 通过: 失效日志不得产生任何事件
    assert reg.read_invalidate_log() == []


def test_validate_with_depends_none_skips_dep_check_explicitly(tmp_path, reg):
    art = _make_artifact(tmp_path)
    reg.register(art, "cache", depends={"params_hash": {"a": 1}}, git_sha=GIT)
    res = reg.validate(art, depends=None, who="probe-only")
    assert res["status"] == STATUS_VALID
    assert res["dep_check"] == "skipped"


# --------------------------------------------------------------------------- #
# 2) 篡改依赖 -> invalid + invalidate_log 追加; 不自动删文件
# --------------------------------------------------------------------------- #
def test_dependency_tamper_invalidates_and_logs(tmp_path, reg):
    art = _make_artifact(tmp_path, "train_latents_4400.pt")
    params_v1 = {"solver": D6_SOLVER_LIT, "seed": 20263502}
    reg.register(art, "teacher_latent_cache", depends=_teacher_like_deps(params_v1),
                 git_sha=GIT)

    params_v2 = {"solver": D6_SOLVER_LIT, "seed": 99999999}  # 配置漂移
    res = reg.validate(art, depends=_teacher_like_deps(params_v2), who="mechanism_a.load")
    assert res["status"] == STATUS_INVALID
    assert DEP_PARAMS_HASH in res["mismatches"]
    mm = res["mismatches"][DEP_PARAMS_HASH]
    assert mm["stored"] == param_hash(params_v1)
    assert mm["current"] == param_hash(params_v2)
    assert any(DEP_PARAMS_HASH in r for r in res["reasons"])
    # 不自动删文件
    assert art.is_file()

    events = reg.read_invalidate_log()
    assert len(events) == 1
    ev = events[0]
    assert ev["event"] == "invalidate"
    assert ev["who"] == "mechanism_a.load"
    assert ev["when_utc"]  # when
    assert ev["artifact_path"].endswith("train_latents_4400.pt")
    assert ev["status"] == STATUS_INVALID
    assert DEP_PARAMS_HASH in ev["mismatches"]  # which-dep-mismatch

    # 每次失败都追加 (append-only 事件流), 供 A02/DATA-01 审计回放
    reg.validate(art, depends=_teacher_like_deps({"totally": "different"}), who="mechanism_b.load")
    assert len(reg.read_invalidate_log()) == 2


def test_checkpoint_set_tamper_detected(tmp_path, reg):
    art = _make_artifact(tmp_path, "ck.pt")
    reg.register(art, "cache", depends={DEP_CKPT_SHA256: [SHA_A, SHA_B]}, git_sha=GIT)
    # 换掉其中一个 checkpoint 的 sha -> invalid
    res = reg.validate(art, depends={DEP_CKPT_SHA256: [SHA_A, "c" * 64]}, who="t")
    assert res["status"] == STATUS_INVALID
    assert DEP_CKPT_SHA256 in res["mismatches"]


def test_ckpt_sha256_list_is_order_insensitive(tmp_path, reg):
    art = _make_artifact(tmp_path)
    reg.register(art, "cache", depends={DEP_CKPT_SHA256: [SHA_A, SHA_B]}, git_sha=GIT)
    assert reg.validate(art, depends={DEP_CKPT_SHA256: [SHA_B, SHA_A]},
                        who="t")["status"] == STATUS_VALID


def test_added_or_removed_dependency_is_a_mismatch(tmp_path, reg):
    art = _make_artifact(tmp_path)
    reg.register(art, "cache", depends={"alpha": 1.0}, git_sha=GIT)
    res_add = reg.validate(art, depends={"alpha": 1.0, "beta": 2.0}, who="t")
    assert res_add["status"] == STATUS_INVALID and "beta" in res_add["mismatches"]
    res_rm = reg.validate(art, depends={}, who="t")
    assert res_rm["status"] == STATUS_INVALID and "alpha" in res_rm["mismatches"]


def test_missing_artifact_file_reports_missing_and_logs(tmp_path, reg):
    art = _make_artifact(tmp_path, "gone.pt")
    params = {"seed": 1}
    reg.register(art, "cache", depends={"params_hash": params}, git_sha=GIT)
    art.unlink()  # 仅删除 tmp_path 内测试自建文件, 模拟磁盘清理
    res = reg.validate(art, depends={"params_hash": params}, who="janitor")
    assert res["status"] == STATUS_MISSING
    assert any("no longer exists" in r for r in res["reasons"])
    events = reg.read_invalidate_log()
    assert len(events) == 1 and events[0]["status"] == STATUS_MISSING


# --------------------------------------------------------------------------- #
# 3) param_hash: 确定性 / 键序无关 / 全精度浮点 / 与 A02 对齐
# --------------------------------------------------------------------------- #
def test_param_hash_deterministic_and_order_independent():
    cfg = {
        "solver": D6_SOLVER_LIT,
        "etas": [0.055, 0.105],
        "nested": {"b": {"y": 1, "x": [3, 1, 2]}, "a": None},
        "lr": 0.055,
        "third": 1 / 3,
    }
    shuffled = {
        "third": 1 / 3,
        "lr": 0.055,
        "nested": {"a": None, "b": {"x": [3, 1, 2], "y": 1}},
        "etas": [0.055, 0.105],
        "solver": {k: D6_SOLVER_LIT[k] for k in reversed(list(D6_SOLVER_LIT))},
    }
    h = param_hash(cfg)
    assert len(h) == 64 and all(c in "0123456789abcdef" for c in h)
    assert param_hash(cfg) == h                      # 确定性
    assert param_hash(shuffled) == h                 # 键序无关 (含嵌套 dict)
    assert param_hash(list(D6_SOLVER_LIT)) == param_hash(tuple(D6_SOLVER_LIT))  # tuple==list
    assert param_hash(cfg, n=12) == h[:12]


def test_param_hash_full_precision_floats():
    third = 1 / 3
    base = {"w": third}
    assert param_hash(base) == param_hash({"w": third})          # 同值同 hash
    assert param_hash(base) != param_hash({"w": 0.33})           # 精度差异可区分
    assert param_hash({"n": 1}) != param_hash({"n": 1.0})        # int/float 序列化区分
    # 大整数与小数的最短往返表示稳定
    assert param_hash({"t": 20263502, "e": 2.718281828459045}) == \
        param_hash({"e": 2.718281828459045, "t": 20263502})


def test_normalize_dep_value_semantics():
    assert normalize_dep_value("x", None) == "unknown"
    assert normalize_dep_value("decoder_version", "s10") == "s10"
    assert normalize_dep_value("p", {"b": 2, "a": 1}) == param_hash({"a": 1, "b": 2})
    assert normalize_dep_value(DEP_CKPT_SHA256, (SHA_B, SHA_A)) == \
        normalize_dep_value(DEP_CKPT_SHA256, [SHA_A, SHA_B])
    # 非 ckpt 列表保序 (顺序语义依赖)
    assert normalize_dep_value("etas", [0.055, 0.105]) != normalize_dep_value("etas", [0.105, 0.055])


@pytest.mark.skipif(not A02_IMPORTABLE, reason="a02 tool not importable")
def test_param_hash_matches_a02_solver_hash_convention():
    """单一 hash 约定: B03 param_hash == A02 solver_hash (同序列化字节)。"""
    assert param_hash(D6_SOLVER_LIT) == a02_solver_hash(D6_SOLVER_LIT)
    assert param_hash(D6_SOLVER_LIT, n=12) == a02_solver_hash(D6_SOLVER_LIT, n=12)
    assert param_hash({"a": [1, 2.5, "x"]}) == a02_solver_hash({"a": [1, 2.5, "x"]})


@pytest.mark.skipif(not V22_IMPORTABLE, reason="exp.v22 not importable on CPU")
def test_param_hash_matches_v22_config_hash_convention():
    assert param_hash(D6_SOLVER_LIT, n=12) == _v22.config_hash(D6_SOLVER_LIT)


# --------------------------------------------------------------------------- #
# 4) probe_legacy: 无记录 -> unknown, 不报错
# --------------------------------------------------------------------------- #
def test_probe_legacy_unknown_for_unregistered_and_missing(tmp_path, reg):
    missing = tmp_path / "never" / "existed.pt"
    info = reg.probe_legacy(missing)  # 不抛错
    assert info["status"] == STATUS_UNKNOWN
    assert info["record"] is None
    assert info["file"]["exists"] is False

    existing = _make_artifact(tmp_path, "legacy_old_format.pt")
    info2 = reg.probe_legacy(existing)
    assert info2["status"] == STATUS_UNKNOWN  # 有文件但注册表无记录: 仍 unknown
    assert info2["file"]["exists"] is True
    assert info2["file"]["size_bytes"] == len(b"payload-v1")
    assert "modified_utc" in info2["file"]


def test_probe_legacy_registered_after_register(tmp_path, reg):
    art = _make_artifact(tmp_path)
    reg.register(art, "cache", git_sha=GIT)
    info = reg.probe_legacy(art)
    assert info["status"] == "registered"
    assert info["record"]["kind"] == "cache"


def test_validate_unregistered_path_is_unknown_and_not_logged(tmp_path, reg):
    art = _make_artifact(tmp_path, "unregistered.pt")
    res = reg.validate(art, depends={"anything": 1}, who="audit")
    assert res["status"] == STATUS_UNKNOWN
    assert res["record"] is None
    # unknown 不是 validate 失败: 不入失效日志
    assert reg.read_invalidate_log() == []


# --------------------------------------------------------------------------- #
# 5) format_version + min_format_version (B02 ref_feats 教训)
# --------------------------------------------------------------------------- #
def test_stale_format_version_blocked_by_min_format_version(tmp_path, reg):
    ref = _make_artifact(tmp_path, "ref_deadbeef.pt")
    # 历史条目: 2D 无批维 (o2 时代) -> 如实登记为 v1
    reg.register(ref, "ref_feature_entry",
                 depends={"ref_content_sha": "d" * 64},
                 format_version=FORMAT_REF_FEATS_V1_2D_NO_BATCH,
                 notes="2D tokens (N,C) / pal (C,), no batch dim",
                 git_sha=GIT)

    # 读取方不设版本门槛: 可兼容读取 (batch_cached_ref_tokens 归一) -> valid
    assert reg.validate(ref, who="reader")["status"] == STATUS_VALID

    # 新消费者要求带批维条目 -> v1 被拦截, 且记入失效日志
    res = reg.validate(ref, min_format_version=FORMAT_REF_FEATS_V2_BATCH_DIM,
                       who="strict_reader")
    assert res["status"] == STATUS_INVALID
    assert any("format_version" in r and "min_format_version" in r for r in res["reasons"])
    assert res["mismatches"]["format_version"]["stored"] == "1"
    events = reg.read_invalidate_log()
    assert len(events) == 1 and events[0]["status"] == STATUS_INVALID

    # v2 条目 (带批维) 通过同一门槛
    ref2 = _make_artifact(tmp_path, "ref_cafebabe.pt", payload=b"batch-dim")
    reg.register(ref2, "ref_feature_entry", depends={"ref_content_sha": "e" * 64},
                 format_version=FORMAT_REF_FEATS_V2_BATCH_DIM, git_sha=GIT)
    assert reg.validate(ref2, min_format_version=FORMAT_REF_FEATS_V2_BATCH_DIM,
                        who="strict_reader")["status"] == STATUS_VALID


def test_format_version_defaults_to_1_and_accepts_equal_min(tmp_path, reg):
    art = _make_artifact(tmp_path)
    reg.register(art, "cache", git_sha=GIT)  # 默认 format_version=1
    assert reg.validate(art, min_format_version=1, who="t")["status"] == STATUS_VALID
    assert reg.validate(art, min_format_version=2, who="t")["status"] == STATUS_INVALID


# --------------------------------------------------------------------------- #
# 6) JSONL 逐行可解析 + append-only (重复 register 不覆盖)
# --------------------------------------------------------------------------- #
def test_jsonl_parseable_and_append_only(tmp_path, reg):
    art = _make_artifact(tmp_path, "append.pt")
    reg.register(art, "cache", depends={"params_hash": {"v": 1}}, git_sha=GIT)
    reg.register(art, "cache", depends={"params_hash": {"v": 2}}, git_sha=GIT)  # 重复登记
    reg.register(_make_artifact(tmp_path, "other.pt"), "cache", git_sha=GIT)

    lines = reg.registry_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3  # 不覆盖: 三条记录都在
    parsed = [json.loads(ln) for ln in lines]  # 逐行可解析
    assert all(p["record_type"] == "artifact" for p in parsed)
    assert parsed[0]["depends"]["params_hash"] == param_hash({"v": 1})
    assert parsed[1]["depends"]["params_hash"] == param_hash({"v": 2})
    assert parsed[2]["depends"] == {}  # 无 depends 的登记保持空 dict
    # lookup 取最新一条 (append-only 的"后写为准")
    latest = reg.lookup(art)
    assert latest.depends["params_hash"] == param_hash({"v": 2})
    # 历史记录仍可审计回放
    recs = [r for r in reg.records() if r.artifact_path == latest.artifact_path]
    assert len(recs) == 2 and recs[0].depends["params_hash"] == param_hash({"v": 1})


def test_registry_and_invalidate_log_are_separate_files(tmp_path, reg):
    art = _make_artifact(tmp_path)
    reg.register(art, "cache", git_sha=GIT)
    reg.validate(art, depends={"params_hash": {"x": 0}}, who="t")  # 失败
    assert reg.registry_path.is_file() and reg.invalidate_log_path.is_file()
    assert reg.registry_path != reg.invalidate_log_path
    # 注册表文件里没有任何 invalidate 事件混入
    for ln in reg.registry_path.read_text(encoding="utf-8").splitlines():
        assert json.loads(ln)["record_type"] == "artifact"


def test_corrupt_line_tolerated_on_read(tmp_path, reg):
    art = _make_artifact(tmp_path)
    reg.register(art, "cache", git_sha=GIT)
    with open(reg.registry_path, "a", encoding="utf-8") as fh:
        fh.write("{not-json\n")
    reg.register(_make_artifact(tmp_path, "b.pt"), "cache", git_sha=GIT)
    assert len(reg.records()) == 2  # 坏行被跳过, 绝不抛错
    assert reg.lookup(art) is not None


# --------------------------------------------------------------------------- #
# 7) git_sha 解析与 artifact hash 复核
# --------------------------------------------------------------------------- #
def test_git_sha_env_resolution(monkeypatch, tmp_path, reg):
    monkeypatch.setenv("WEAVE_GIT_SHA", "deadbeefcafe")
    assert resolve_git_sha() == "deadbeefcafe"
    art = _make_artifact(tmp_path)
    rec = reg.register(art, "cache")  # 未显式给 git_sha -> 取环境变量
    assert rec.git_sha == "deadbeefcafe"


def test_explicit_git_sha_overrides_env(monkeypatch, tmp_path, reg):
    monkeypatch.setenv("WEAVE_GIT_SHA", "envsha")
    art = _make_artifact(tmp_path)
    rec = reg.register(art, "cache", git_sha=GIT)
    assert rec.git_sha == GIT


def test_artifact_hash_optin_detects_overwrite(tmp_path, reg):
    art = _make_artifact(tmp_path, "h.pt", payload=b"v1")
    reg.register(art, "cache", git_sha=GIT, hash_artifact=True)
    rec = reg.lookup(art)
    assert rec.depends["artifact_sha256"] == file_sha256(art)
    assert reg.validate(art, check_artifact_hash=True, who="t")["status"] == STATUS_VALID

    art.write_bytes(b"v2-overwritten")  # 模拟缓存被覆写/损坏
    res = reg.validate(art, check_artifact_hash=True, who="t")
    assert res["status"] == STATUS_INVALID
    assert "artifact_sha256" in res["mismatches"]
    # 不要求复核时不做文件哈希 (大缓存友好), 且不因登记了 hash 键而误判
    assert reg.validate(art, who="t")["status"] == STATUS_VALID


# --------------------------------------------------------------------------- #
# 8) 记录结构: JSON 安全 / 前向兼容 / notes+extra 往返
# --------------------------------------------------------------------------- #
def test_record_json_safe_and_forward_compatible(tmp_path, reg):
    art = _make_artifact(tmp_path)
    rec = reg.register(art, "cache", git_sha=GIT, notes="n1", extra_kw=42)
    d = rec.to_dict()
    assert json.loads(json.dumps(d)) == d  # JSON-safe
    # 前向兼容: 未知顶层字段被忽略, 不破坏解析
    d2 = {**d, "future_field": {"a": 1}}
    obj = json.loads(json.dumps(d2))
    from src.v25.artifact_registry import ArtifactRecord
    rec2 = ArtifactRecord.from_dict(obj)
    assert rec2.kind == rec.kind and "future_field" not in rec2.to_dict()
    # notes / extra kwargs 往返
    back = ArtifactRecord.from_dict(json.loads(json.dumps(rec.to_dict())))
    assert back.notes == "n1" and back.extra == {"extra_kw": 42}
    # notes / extra kwargs 往返
    back = ArtifactRecord.from_dict(json.loads(json.dumps(rec.to_dict())))
    assert back.notes == "n1" and back.extra == {"extra_kw": 42}


def test_register_does_not_touch_repo_caches(tmp_path, reg):
    """护栏: 注册/校验失败绝不影响仓库内真实缓存路径 (tmp registry, 仓库零写入)。"""
    art = _make_artifact(tmp_path)
    reg.register(art, "cache", depends={"params_hash": {"v": 1}}, git_sha=GIT)
    reg.validate(art, depends={"params_hash": {"v": 2}}, who="t")
    assert art.read_bytes() == b"payload-v1"  # 工件字节未动
    # 所有写入都落在 tmp_path 内
    assert str(tmp_path) in str(reg.registry_path)
    assert str(tmp_path) in str(reg.invalidate_log_path)
