"""B03: 统一 checkpoint/cache 血缘检查器（append-only JSONL 注册表）。

背景（agent2.md §四 B03）：仓库大量 ``if path.exists(): return torch.load(path)``
式缓存在训练配置改变后会**静默复用旧缓存**；本模块提供兼容包装层 —— 只登记
与校验血缘，**不重写 / 不删除 / 不迁移任何旧缓存**。

核心对象 ``ArtifactRegistry``：一个 JSONL 注册表（路径参数化，测试用 tmp_path），
每条记录::

    {"record_type": "artifact",
     "artifact_path": "runs/v25/cache/train_latents_4400.pt",   # 仓内相对 posix
     "kind": "teacher_latent_cache",
     "created_utc": "2026-11-20T12:00:00+00:00",
     "git_sha": "<40-hex | unknown>",
     "depends": {"ckpt_sha256": "<param_hash of sorted sha list>", ...},
     "format_version": 2,
     "notes": "", "extra": {...}}

依赖键（``depends``，任一变化 -> ``validate`` 返回 invalid，**绝不自动删文件**）：

- ``ckpt_sha256``            checkpoint 文件 sha256 列表（无序集合语义，经 param_hash 归一）；
- ``input_manifest_sha``     输入 manifest 内容 sha；
- ``params_hash``            配置 dict/list 的稳定排序 JSON sha256（= A02 ``solver_hash`` /
                             ``exp.v22.config_hash`` 同一序列化约定，避免两套 hash；
                             浮点走 ``repr`` 最短往返 = 全精度）；
- ``decoder_version``        解码器版本（S10 契约见 ``s10_decoder_version``）；
- ``feature_encoder_version`` 特征编码器（DINO/ConvNeXt/CSD）版本；
- 任意 extra 键              dict/list 值统一 param_hash 归一，标量转 str。

缓存格式版本（吸收 B02 教训）：``format_version`` + ``validate(min_format_version=...)``。
旧格式（如 runs/v25/cache/ref_feats/ 的 2D 无批维历史条目）读取方
（reference_teacher_v2.batch_cached_ref_tokens）可兼容，但注册表**如实记录版本**，
为 B05 / 血缘审计提供依据 —— 版本不达标判 invalid，而不是悄悄放过。

``probe_legacy(path)``：注册表无记录 -> 返回 ``status="unknown"`` 而非报错
（legacy-probe 契约，对齐 A02 ``probe_pt`` 的 unknown-not-raise 语义）。

``invalidate_log``：每次 ``validate`` 失败（invalid / missing）向独立 JSONL 追加一条
``{"event": "invalidate", "who", "when_utc", "artifact_path", "mismatches", "reasons"}``，
供 A02 / DATA-01 后续审计；unknown（无记录）不算失败，不入日志。

约束：本模块纯 stdlib（无 torch/numpy 依赖），CPU 即可全量测试；不执行任何
文件删除；git_sha 仅在未显式给出时经环境变量 ``WEAVE_GIT_SHA`` 或只读
``git rev-parse HEAD`` 探测（任何失败 -> "unknown"，绝不抛错）。
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

__all__ = [
    "ArtifactRegistry",
    "ArtifactRecord",
    "REPO_ROOT",
    "UNKNOWN",
    "STATUS_VALID",
    "STATUS_INVALID",
    "STATUS_UNKNOWN",
    "STATUS_MISSING",
    "REGISTRY_SCHEMA_VERSION",
    "DEP_CKPT_SHA256",
    "DEP_INPUT_MANIFEST_SHA",
    "DEP_PARAMS_HASH",
    "DEP_DECODER_VERSION",
    "DEP_FEATURE_ENCODER_VERSION",
    "FORMAT_REF_FEATS_V1_2D_NO_BATCH",
    "FORMAT_REF_FEATS_V2_BATCH_DIM",
    "S10_DECODER_RELPATH",
    "param_hash",
    "file_sha256",
    "ckpt_sha256_from_paths",
    "normalize_dep_value",
    "resolve_git_sha",
    "utc_now",
    "s10_decoder_version",
]

REPO_ROOT = Path(__file__).resolve().parents[2]

UNKNOWN = "unknown"

STATUS_VALID = "valid"          # 记录存在 + 依赖全匹配 + 格式达标 + 工件文件在
STATUS_INVALID = "invalid"      # 依赖漂移 / format_version 低于 min_format_version
STATUS_UNKNOWN = "unknown"      # 注册表无记录（probe_legacy 契约：不报错）
STATUS_MISSING = "missing"      # 记录在但工件文件已消失（仍不自动重建/删除任何东西）

#: 注册表 / 事件日志自身的 schema 版本（记录结构变化时递增）
REGISTRY_SCHEMA_VERSION = 1

# ---- 规范依赖键 -----------------------------------------------------------
DEP_CKPT_SHA256 = "ckpt_sha256"
DEP_INPUT_MANIFEST_SHA = "input_manifest_sha"
DEP_PARAMS_HASH = "params_hash"
DEP_DECODER_VERSION = "decoder_version"
DEP_FEATURE_ENCODER_VERSION = "feature_encoder_version"

# ---- 缓存格式版本常量（B02 二次交付教训，reference_teacher_v2.py）----------
#: ref_feats 旧条目：单样本落盘丢批维 —— tokens (N,C) / pal (C,)（o2 时代）
FORMAT_REF_FEATS_V1_2D_NO_BATCH = 1
#: ref_feats 新条目：整体带批维 —— tokens (1,N,C) / pal (1,C)（reference_teacher_v2）
FORMAT_REF_FEATS_V2_BATCH_DIM = 2

# ---- 钉住的 S10 decoder（AGENTS.md 契约；仓内相对路径）---------------------
S10_DECODER_RELPATH = "deliver/checkpoints/runs/vae_decoder_prune/s10_p5_lowlr/decoder.pt"


# --------------------------------------------------------------------------- #
# 基础原语：hash / 时间 / git
# --------------------------------------------------------------------------- #
def utc_now() -> str:
    """ISO-8601 UTC 时间戳（与 src.utils.runtime.utc 同格式约定）。"""
    return datetime.now(timezone.utc).isoformat()


def _sorted_json(obj: Any) -> str:
    """与 A02 ``solver_hash`` / ``exp.v22.config_hash`` 完全相同的序列化约定。"""
    return json.dumps(obj, sort_keys=True, default=str)


def param_hash(obj: Any, n: int | None = None) -> str:
    """dict/list 配置的稳定 sha256（键序无关；浮点 repr 全精度）。

    序列化约定 = ``json.dumps(obj, sort_keys=True, default=str)`` ->
    sha256 hexdigest（与 A02 ``solver_hash`` / ``exp.v22.config_hash`` 逐字节一致，
    由 tests/v25/test_registry.py 的漂移测试钉住）。要点：

    - ``sort_keys=True`` 递归排序 -> 键序无关；
    - 浮点用 ``float.__repr__``（最短往返表示，无损 = 全精度），``1`` 与 ``1.0``
      序列化不同（int/float 区分），是确定性特征而非误差；
    - tuple 序列化为 JSON 数组 -> ``param_hash((1,2)) == param_hash([1,2])``；
    - 不可 JSON 化对象（Path / dtype / torch 张量外的自定义类）经 ``default=str``
      降级为 str。
    """
    digest = hashlib.sha256(_sorted_json(obj).encode("utf-8")).hexdigest()
    return digest if n is None else digest[:n]


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    """分块文件 sha256；文件缺失返回 ``UNKNOWN``，绝不抛错（A02 同契约）。"""
    p = Path(path)
    if not p.is_file():
        return UNKNOWN
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def ckpt_sha256_from_paths(paths: Iterable[str | Path]) -> list[str]:
    """显式把 checkpoint 路径列表 -> sha256 列表（供 ``depends[DEP_CKPT_SHA256]``）。"""
    return [file_sha256(p) for p in paths]


def resolve_git_sha() -> str:
    """登记时的 git 版本探测：环境变量 ``WEAVE_GIT_SHA`` 优先，其次只读
    ``git rev-parse HEAD``；任何失败（git 缺失 / 非仓库 / 超时）-> ``UNKNOWN``。
    绝不在 import 期执行，也绝不抛错。"""
    env = os.environ.get("WEAVE_GIT_SHA", "").strip()
    if env:
        return env
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT), check=True, capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or UNKNOWN
    except Exception:
        return UNKNOWN


def s10_decoder_version() -> str:
    """钉住 S10 decoder 的版本键 = decoder.pt 文件 sha256 前 12 位（缺文件 -> UNKNOWN）。

    依据 AGENTS.md S10 契约：解码一律走
    ``deliver/checkpoints/runs/vae_decoder_prune/s10_p5_lowlr/decoder.pt``；
    版本漂移（重训/替换 decoder）必须使所有登记了 ``decoder_version`` 的缓存失效。
    """
    full = file_sha256(REPO_ROOT / S10_DECODER_RELPATH)
    return UNKNOWN if full == UNKNOWN else full[:12]


# --------------------------------------------------------------------------- #
# 依赖归一
# --------------------------------------------------------------------------- #
def normalize_dep_value(key: str, value: Any) -> str:
    """depends 值 -> 可比较的规范字符串。

    - dict                      -> ``param_hash``（键序无关）；
    - list/tuple                -> ``param_hash``；仅 ``ckpt_sha256`` 先排序去重
                                   （checkpoint 集合无序语义），其余 list 保序；
    - None                      -> ``UNKNOWN``；
    - 其余标量                  -> ``str(value)``。
    """
    if value is None:
        return UNKNOWN
    if isinstance(value, Mapping):
        return param_hash(dict(value))
    if isinstance(value, (list, tuple)):
        items = list(value)
        if key == DEP_CKPT_SHA256:
            items = sorted({str(v) for v in items})
        return param_hash(items)
    return str(value)


def normalize_depends(depends: Mapping[str, Any] | None) -> dict[str, str]:
    if not depends:
        return {}
    return {str(k): normalize_dep_value(str(k), v) for k, v in depends.items()}


# --------------------------------------------------------------------------- #
# 记录
# --------------------------------------------------------------------------- #
class ArtifactRecord:
    """一条工件血缘记录（JSONL 行的内存形态；宽容解析，前向兼容多余键）。"""

    __slots__ = ("artifact_path", "kind", "created_utc", "git_sha", "depends",
                 "format_version", "notes", "extra")

    def __init__(self, artifact_path: str, kind: str, created_utc: str, git_sha: str,
                 depends: dict[str, str], format_version: int,
                 notes: str = "", extra: dict | None = None):
        self.artifact_path = artifact_path
        self.kind = kind
        self.created_utc = created_utc
        self.git_sha = git_sha
        self.depends = dict(depends)
        self.format_version = int(format_version)
        self.notes = notes
        self.extra = dict(extra or {})

    def to_dict(self) -> dict:
        d = {
            "record_type": "artifact",
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "artifact_path": self.artifact_path,
            "kind": self.kind,
            "created_utc": self.created_utc,
            "git_sha": self.git_sha,
            "depends": dict(self.depends),
            "format_version": self.format_version,
        }
        if self.notes:
            d["notes"] = self.notes
        if self.extra:
            d["extra"] = dict(self.extra)
        return d

    @classmethod
    def from_dict(cls, d: Mapping) -> "ArtifactRecord":
        return cls(
            artifact_path=str(d.get("artifact_path", "")),
            kind=str(d.get("kind", "")),
            created_utc=str(d.get("created_utc", "")),
            git_sha=str(d.get("git_sha", UNKNOWN)),
            depends=dict(d.get("depends") or {}),
            format_version=int(d.get("format_version") or 0),
            notes=str(d.get("notes", "")),
            extra=dict(d.get("extra") or {}),
        )


def _path_key(p: str | Path) -> str:
    """路径匹配键：相对路径锚定 REPO_ROOT，normcase（Windows 大小写/分隔符稳健）。"""
    pp = Path(p)
    if not pp.is_absolute():
        pp = REPO_ROOT / pp
    try:
        pp = pp.resolve()
    except OSError:  # pragma: no cover - 平台相关
        pp = pp.absolute()
    return os.path.normcase(str(pp))


def _display_path(p: str | Path) -> str:
    """存储用路径串：仓内文件存 REPO_ROOT 相对 posix，仓外存绝对 posix（如实记录）。"""
    pp = Path(p)
    try:
        rel = pp.resolve().relative_to(REPO_ROOT)
        return rel.as_posix()
    except (ValueError, OSError):
        return Path(p).absolute().as_posix()


# --------------------------------------------------------------------------- #
# 注册表
# --------------------------------------------------------------------------- #
class ArtifactRegistry:
    """append-only JSONL 血缘注册表 + 失效事件日志（纯 stdlib，CPU 可测）。

    - ``register`` 永远追加新行，**不覆盖**历史记录（审计可回放）；查找取最新一条。
    - ``validate`` 依赖漂移 / 格式版本不达标 / 工件缺失 -> 失败状态 + invalidate
      事件日志追加；**绝不删除或改写任何缓存文件**。
    - ``probe_legacy`` 无记录 -> ``unknown``，不报错（历史缓存兼容包装层）。
    """

    def __init__(self, registry_path: str | Path, invalidate_log_path: str | Path | None = None):
        self.registry_path = Path(registry_path)
        self.invalidate_log_path = (
            Path(invalidate_log_path) if invalidate_log_path is not None
            else self.registry_path.with_suffix(self.registry_path.suffix + ".invalidate_log.jsonl")
        )
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.invalidate_log_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- 读 ----
    def records(self) -> list[ArtifactRecord]:
        """全部记录（按文件顺序；坏行宽容跳过，绝不抛错）。"""
        out: list[ArtifactRecord] = []
        if not self.registry_path.is_file():
            return out
        with open(self.registry_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, Mapping):
                    out.append(ArtifactRecord.from_dict(obj))
        return out

    def lookup(self, artifact_path: str | Path) -> ArtifactRecord | None:
        """该工件路径的**最新**记录；无 -> None（append-only 语义：后写为准）。"""
        key = _path_key(artifact_path)
        found: ArtifactRecord | None = None
        for rec in self.records():
            if _path_key(rec.artifact_path) == key:
                found = rec
        return found

    def read_invalidate_log(self) -> list[dict]:
        """全部失效事件（宽容解析；供 A02 / DATA-01 审计读取）。"""
        out: list[dict] = []
        if not self.invalidate_log_path.is_file():
            return out
        with open(self.invalidate_log_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, Mapping):
                    out.append(dict(obj))
        return out

    # ---------------------------------------------------------------- 写 ----
    def register(self, artifact_path: str | Path, kind: str, *,
                 depends: Mapping[str, Any] | None = None,
                 format_version: int = 1,
                 git_sha: str | None = None,
                 notes: str = "",
                 hash_artifact: bool = False,
                 **extra: Any) -> ArtifactRecord:
        """登记一个工件（追加一行；同路径重复登记不覆盖旧行）。

        ``depends`` 中 dict/list 值在落盘前经 :func:`normalize_dep_value` 归一
        （params -> param_hash），因此登记与校验两侧天然可比。
        ``hash_artifact=True`` 额外登记 ``artifact_sha256``（大文件慎用；
        校验侧可用 ``validate(check_artifact_hash=True)`` 复核防覆写/损坏）。
        """
        deps = normalize_depends(depends)
        if hash_artifact:
            deps["artifact_sha256"] = file_sha256(artifact_path)
        rec = ArtifactRecord(
            artifact_path=_display_path(artifact_path),
            kind=str(kind),
            created_utc=utc_now(),
            git_sha=str(git_sha) if git_sha is not None else resolve_git_sha(),
            depends=deps,
            format_version=int(format_version),
            notes=notes,
            extra=extra,
        )
        self._append_json(self.registry_path, rec.to_dict())
        return rec

    def append_invalidate_event(self, event: Mapping[str, Any]) -> None:
        self._append_json(self.invalidate_log_path, dict(event))

    @staticmethod
    def _append_json(path: Path, obj: Mapping[str, Any]) -> None:
        """单行 JSON 追加（flush + fsync；崩溃最多丢最后一行，绝不破坏旧行）。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(obj, sort_keys=True, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    # ------------------------------------------------------------ 探测 ------
    def probe_legacy(self, artifact_path: str | Path) -> dict:
        """历史缓存兼容探测：注册表无记录 -> ``status="unknown"``，绝不报错。

        返回 ``{"path", "status": "registered"|"unknown", "record", "file": {...}}``；
        ``file`` 子对象给出存在性 / 大小 / mtime（纯 stat，不 load 内容，不依赖 torch）。
        """
        p = Path(artifact_path)
        rec = self.lookup(p)
        stat_info: dict[str, Any] = {"exists": p.is_file()}
        try:
            st = p.stat()
            stat_info["exists"] = True
            stat_info["size_bytes"] = int(st.st_size)
            stat_info["modified_utc"] = datetime.fromtimestamp(
                st.st_mtime, tz=timezone.utc).isoformat()
        except OSError:
            stat_info["exists"] = False
        return {
            "path": _display_path(p),
            "status": "registered" if rec is not None else STATUS_UNKNOWN,
            "record": rec.to_dict() if rec is not None else None,
            "file": stat_info,
        }

    # ------------------------------------------------------------ 校验 ------
    def validate(self, artifact_path: str | Path, *,
                 depends: Mapping[str, Any] | None = None,
                 min_format_version: int | None = None,
                 who: str = "",
                 check_file: bool = True,
                 check_artifact_hash: bool = False) -> dict:
        """血缘校验（缓存加载前的闸门）。

        - 无记录               -> ``unknown``（不入 invalidate 日志）；
        - 依赖漂移 / 格式过老   -> ``invalid``（记录 mismatch 逐键对照）；
        - 工件文件缺失         -> ``missing``；
        - 其余                 -> ``valid``。

        invalid / missing 均向 invalidate_log 追加一条
        ``{"event", "who", "when_utc", "artifact_path", "status", "mismatches", "reasons"}``。
        ``depends=None`` 时跳过依赖比对（仅做格式/文件检查），结果以
        ``"dep_check": "skipped"`` 显式标记，便于审计区分。
        **任何分支都不删除 / 不重建 / 不改写缓存文件。**
        """
        rec = self.lookup(artifact_path)
        display = _display_path(artifact_path)
        if rec is None:
            return {
                "status": STATUS_UNKNOWN,
                "artifact_path": display,
                "record": None,
                "mismatches": {},
                "reasons": ["no registry record (legacy/unregistered cache)"],
                "who": who,
                "dep_check": "skipped",
            }

        mismatches: dict[str, dict[str, str]] = {}
        reasons: list[str] = []

        current = normalize_depends(depends) if depends is not None else {}
        if check_artifact_hash and "artifact_sha256" in rec.depends:
            # 防覆写/损坏复核：current 值直接取磁盘文件现算 sha（非调用方提供）
            current["artifact_sha256"] = file_sha256(artifact_path)
        # depends=None 且未要求 hash 复核 -> 完全跳过依赖比对（显式标记 skipped）；
        # depends=None 但要求 hash 复核 -> 只比对 artifact_sha256 这一个本地可验证键。
        perform_dep_check = depends is not None or bool(current)
        dep_check = "performed" if perform_dep_check else "skipped"
        if perform_dep_check:
            keys = (sorted(set(rec.depends) | set(current)) if depends is not None
                    else sorted(current))
            for key in keys:
                if key == "artifact_sha256" and not check_artifact_hash:
                    continue  # 登记过但本次未要求复核：跳过，避免大文件重复哈希
                stored_v = rec.depends.get(key)
                cur_v = current.get(key)
                if stored_v is None:
                    mismatches[key] = {"stored": "<absent>", "current": cur_v or ""}
                    reasons.append(f"dependency '{key}' added vs registered record")
                elif cur_v is None:
                    mismatches[key] = {"stored": stored_v, "current": "<absent>"}
                    reasons.append(f"dependency '{key}' missing from current inputs")
                elif stored_v != cur_v:
                    mismatches[key] = {"stored": stored_v, "current": cur_v}
                    reasons.append(f"dependency '{key}' drifted: {stored_v[:12]} -> {cur_v[:12]}")

        if min_format_version is not None and rec.format_version < int(min_format_version):
            reasons.append(
                f"format_version {rec.format_version} < required min_format_version "
                f"{int(min_format_version)} (stale cache layout; see B02 ref_feats lesson)")
            mismatches.setdefault("format_version", {
                "stored": str(rec.format_version),
                "current": str(int(min_format_version)),
            })

        if check_file and not Path(artifact_path).is_file():
            reasons.append("artifact file no longer exists on disk")

        status = STATUS_VALID
        if check_file and not Path(artifact_path).is_file():
            status = STATUS_MISSING
        elif mismatches:
            status = STATUS_INVALID

        if status in (STATUS_INVALID, STATUS_MISSING):
            self.append_invalidate_event({
                "record_type": "invalidate",
                "event": "invalidate",
                "who": str(who),
                "when_utc": utc_now(),
                "artifact_path": display,
                "status": status,
                "mismatches": mismatches,
                "reasons": reasons,
                "registry_schema_version": REGISTRY_SCHEMA_VERSION,
            })

        return {
            "status": status,
            "artifact_path": display,
            "record": rec.to_dict(),
            "mismatches": mismatches,
            "reasons": reasons,
            "who": who,
            "dep_check": dep_check,
        }
