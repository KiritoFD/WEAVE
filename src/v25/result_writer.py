"""B06: 结果 Writer、任务状态与恢复检查（agent2.md §五 B06 / §九文件所有权）。

两件事，纯 stdlib（无 torch/numpy 依赖，CPU/无 GPU 可直接使用）：

1. ``atomic_save_json(obj, path)`` —— tmp + ``os.replace`` 的原子 JSON 落盘，
   Windows 兼容（tmp 与目标同目录保证同卷；``os.replace`` 在目标已存在时仍原子覆盖）。
   现有 ``src.utils.runtime.save_json`` 语义保持兼容（ensure_ascii=False / default=str /
   allow_nan=False），新增 pid 唯一 tmp 名（并发写者互不踩踏）与 indent 可选（compact
   口径：B04 实测 0.9MB eval.json indent=2 体积翻倍、dumps 3.9x CPU，绝对值仍 <0.1s）。
   不替换 ``save_json`` —— 新脚本可选。

2. ``ResultWriter`` —— raw 目录的状态标记薄类（started/running/done/failed/incomplete），
   供**新**实验脚本可选使用；与 ``src.utils.raw_io.RawWriter`` 完全兼容，不替换它：
   RawWriter 继续负责 batch 载荷，ResultWriter 只在旁边持久化"任务状态"，
   使中断的队列可以据状态而非文件存在性启发式分诊续跑。

    状态文件 = ``<arm_dir>/status.json``（原子写，最新状态缓存）
              + ``<arm_dir>/status_log.jsonl``（append-only 历史，fsync，权威记录）
    读序    = status.json 可解析 → 用之；损坏/缺失 → 回放 JSONL 最后一条有效事件；
              两者皆不可用 → unknown。崩溃中途撕裂的最后一行 JSONL 被宽容跳过。

状态语义（修复审计发现的两个缺口，详见 docs/v25/audits/b06_recovery_report.md）：
    - done 必须过 task_id 完整性闸：``done(meta=...)`` 给出 raw meta 或目录内
      ``meta.json`` 存在时，未通过 ``validate_raw_meta_dict``（恰好 expected_tasks 个
      唯一 task_id）则**拒绝记 done**，降级记 incomplete —— 堵住"失败仍记 done"；
    - 整条流水线第一次有了持久化的 failed/incomplete 标记（此前全栈无任何失败状态，
      "done 代理"= eval.json 文件存在性，半写文件也会被当作已评测）。

巡检入口（总控 one-liner，3060/4090 通用）::

    python -m src.v25.result_writer runs/v25 --stale-min 30
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "STATE_STARTED",
    "STATE_RUNNING",
    "STATE_DONE",
    "STATE_FAILED",
    "STATE_INCOMPLETE",
    "STATE_UNKNOWN",
    "STATUS_FILENAME",
    "LOG_FILENAME",
    "atomic_save_json",
    "read_json_tolerant",
    "append_jsonl",
    "validate_raw_meta_dict",
    "ResultWriter",
    "read_state",
    "scan_status",
    "main",
]

STATE_STARTED = "started"
STATE_RUNNING = "running"
STATE_DONE = "done"
STATE_FAILED = "failed"
STATE_INCOMPLETE = "incomplete"
STATE_UNKNOWN = "unknown"

STATUS_FILENAME = "status.json"
LOG_FILENAME = "status_log.jsonl"

_VALID_STATES = (STATE_STARTED, STATE_RUNNING, STATE_DONE, STATE_FAILED, STATE_INCOMPLETE)


# --------------------------------------------------------------------------- #
# 原子 IO 原语
# --------------------------------------------------------------------------- #
def _utc() -> str:
    """ISO-8601 UTC（与 src.utils.runtime.utc 同格式约定，避免为时间戳 import torch 栈）。"""
    return datetime.now(timezone.utc).isoformat()


def atomic_save_json(obj: Any, path: str | Path, *, indent: int | None = None,
                     fsync: bool = False) -> Path:
    """原子写 JSON：tmp（同目录、pid 唯一名）+ ``os.replace``，Windows 兼容。

    - 崩溃模拟：replace 之前任何一步（序列化/写 tmp）失败，目标文件保持旧内容，
      残留 ``<name>.tmp<pid>`` 由下一次同路径写自然复用/覆盖 —— 永不出现"半写目标"；
    - ``os.replace`` 在 Windows 上对**已存在的目标**同样原子覆盖（非 os.rename 语义）；
      tmp 与目标同目录 ⇒ 同卷 ⇒ replace 原子；
    - 序列化约定与 ``src.utils.runtime.save_json`` 一致（ensure_ascii=False /
      default=str / allow_nan=False），NaN/Inf 直接 ValueError 拒写；
    - 注意：Windows 上若另一进程正以独占方式打开目标文件，replace 可能 PermissionError
      —— 读方用 :func:`read_json_tolerant`（开-读-关）即可避免长期持有句柄。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    data = json.dumps(obj, ensure_ascii=False, indent=indent, default=str, allow_nan=False)
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(data)
        if fsync:
            fh.flush()
            os.fsync(fh.fileno())
    os.replace(tmp, p)
    return p


def read_json_tolerant(path: str | Path) -> dict | list | None:
    """宽容读 JSON：缺失 / 半写损坏 / 编码异常 -> ``None``，绝不抛错。

    专治审计发现的"半写 eval.json 被当作已评测"类问题：读方永远拿到
    "完整 JSON 或 None"，不存在解析到一半的中间态。
    """
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None


def append_jsonl(path: str | Path, obj: Mapping[str, Any], *, fsync: bool = True) -> None:
    """单行 JSON 追加（flush + fsync；崩溃最多丢最后一行，绝不破坏旧行）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False, default=str, sort_keys=True)
    with open(p, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(line + "\n")
        fh.flush()
        if fsync:
            os.fsync(fh.fileno())


def _iter_valid_jsonl(path: Path) -> list[dict]:
    """逐行解析 JSONL，坏行（含崩溃撕裂的最后一行）宽容跳过。"""
    out: list[dict] = []
    if not path.is_file():
        return out
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def validate_raw_meta_dict(meta: Any, expected_tasks: int = 750) -> bool:
    """raw/meta.json 完整性闸（stdlib 版，语义与 ``exp.v22.validate_raw_meta`` 逐位一致）：
    恰好 ``expected_tasks`` 个唯一 task_id 才算完整。"""
    try:
        ids = [tid for b in meta["batches"] for tid in b["ids"]]
    except (KeyError, TypeError, IndexError):
        return False
    return len(ids) == expected_tasks and len(set(ids)) == expected_tasks


# --------------------------------------------------------------------------- #
# ResultWriter：raw 目录状态标记（兼容 RawWriter，不替换）
# --------------------------------------------------------------------------- #
class ResultWriter:
    """raw 臂目录的任务状态薄标记（started → running → done|failed|incomplete）。

    用法（新实验脚本可选；RawWriter 照旧管 batch 载荷）::

        rw = ResultWriter(OUT / "raw" / arm_name, expected_tasks=750)
        rw.start(cmd="python -m exp.v25.foo --arm X")
        for ...:
            rw.running(progress=f"{k}/{n}")
        meta = wr.finalize(...)            # RawWriter 照常
        rw.done(meta=meta)                 # meta 未过 750 唯一 id 闸 → 自动记 incomplete
        # 异常路径:  except Exception as e: rw.failed(f"{type(e).__name__}: {e}")

    写序铁律：先 append JSONL（fsync，权威），再原子写 status.json（缓存）。
    status.json 损坏时由 JSONL 回放恢复 —— 两个文件不会同时处于半写状态。
    """

    def __init__(self, arm_dir: str | Path, *, task_id: str | None = None,
                 expected_tasks: int = 750, log_path: str | Path | None = None):
        self.dir = Path(arm_dir)
        self.task_id = task_id or self.dir.name
        self.expected_tasks = int(expected_tasks)
        self.status_path = self.dir / STATUS_FILENAME
        self.log_path = Path(log_path) if log_path is not None else self.dir / LOG_FILENAME

    # ------------------------------------------------------------- 读 ----
    def current(self) -> dict:
        """当前状态记录；status.json 损坏/缺失时回放 JSONL（宽容撕裂尾行）。"""
        status = read_json_tolerant(self.status_path)
        if isinstance(status, dict) and status.get("state") in _VALID_STATES:
            return {**status, "source": STATUS_FILENAME}
        for ev in reversed(_iter_valid_jsonl(self.log_path)):
            if ev.get("state") in _VALID_STATES:
                return {**ev, "source": LOG_FILENAME}
        return {"task_id": self.task_id, "state": STATE_UNKNOWN, "source": "none"}

    def history(self) -> list[dict]:
        """全部有效状态事件（按序；坏行已跳过）。"""
        return _iter_valid_jsonl(self.log_path)

    # ------------------------------------------------------------- 写 ----
    def start(self, **info: Any) -> dict:
        return self._transition(STATE_STARTED, **info)

    def running(self, progress: Any = None, **info: Any) -> dict:
        if progress is not None:
            info.setdefault("progress", progress)
        return self._transition(STATE_RUNNING, **info)

    def failed(self, reason: str, **info: Any) -> dict:
        info["reason"] = str(reason)
        return self._transition(STATE_FAILED, **info)

    def incomplete(self, reason: str, **info: Any) -> dict:
        info["reason"] = str(reason)
        return self._transition(STATE_INCOMPLETE, **info)

    def done(self, meta: Mapping[str, Any] | None = None, **info: Any) -> dict:
        """标记完成；带 raw meta 时**必须**先过 task_id 完整性闸。

        - ``meta`` 显式给出 → 校验之；未过闸 → 降级 :meth:`incomplete`（绝不记 done）；
        - ``meta=None`` 且目录内有 ``meta.json`` → 自动读取校验（同上）；
          **meta.json 存在但半写损坏** → 同样降级 :meth:`incomplete`
          （损坏的完整性凭证不能当作"无凭证"放行）；
        - ``meta=None`` 且无 ``meta.json`` → 仍记 done，但如实记录
          ``meta_validated: null``（非 750 评测类任务，如纯训练臂，由调用方自担）。
        """
        if meta is not None:
            meta_dict: dict | None = dict(meta)
            corrupt = False
        else:
            meta_p = self.dir / "meta.json"
            if meta_p.exists():
                loaded = read_json_tolerant(meta_p)
                meta_dict = loaded if isinstance(loaded, dict) else None
                corrupt = meta_dict is None
            else:
                meta_dict, corrupt = None, False

        if corrupt:
            return self.incomplete("raw meta.json corrupt (done refused)", meta_validated=False, **info)
        if meta_dict is not None:
            if not validate_raw_meta_dict(meta_dict, self.expected_tasks):
                return self.incomplete(
                    "raw meta failed completeness validation (done refused)",
                    meta_validated=False, **info)
            info["meta_validated"] = True
        else:
            info.setdefault("meta_validated", None)
        return self._transition(STATE_DONE, **info)

    # --------------------------------------------------------- 内部 ----
    def _transition(self, state: str, **info: Any) -> dict:
        prev = self.current()
        record: dict[str, Any] = {
            "event": "state_change",
            "task_id": self.task_id,
            "prev_state": prev.get("state", STATE_UNKNOWN),
            "state": state,
            "when_utc": _utc(),
        }
        record.update(info)
        append_jsonl(self.log_path, record)           # 权威历史（fsync）
        cache = {k: record[k] for k in ("task_id", "state", "when_utc") if k in record}
        cache.update(info)
        atomic_save_json(cache, self.status_path, indent=2)  # 最新状态缓存
        return record


def read_state(arm_dir: str | Path, *, expected_tasks: int = 750) -> dict:
    """模块级状态读取（总控巡检用，不产生任何写）。"""
    return ResultWriter(arm_dir, expected_tasks=expected_tasks).current()


# --------------------------------------------------------------------------- #
# 队列巡检：runs 根目录健康扫描（对接 §六 schtasks 失败识别手册）
# --------------------------------------------------------------------------- #
def _meta_state(arm_dir: Path, expected_tasks: int) -> str:
    meta_p = arm_dir / "meta.json"
    if not meta_p.exists():
        return "missing"
    meta = read_json_tolerant(meta_p)
    if meta is None:
        return "corrupt"
    return "valid" if validate_raw_meta_dict(meta, expected_tasks) else "incomplete_count"


def _latest_age_min(paths: list[Path]) -> float | None:
    stamps = []
    for p in paths:
        try:
            stamps.append(p.stat().st_mtime)
        except OSError:
            continue
    return round((time.time() - max(stamps)) / 60.0, 1) if stamps else None


def scan_status(root: str | Path, *, expected_tasks: int = 750) -> list[dict]:
    """扫描一个 runs 输出根目录，输出每臂健康行 + 恢复分诊 flag（纯读，不写）。

    支持两种布局：
    - 配对布局 ``root/raw/<arm>/`` + ``root/measured/<arm>/``（src.eval.runner 约定）；
    - 平铺布局 ``root/<arm>/``（直接含 meta.json / status.json 的子目录）。

    flag 分诊（与 b06_recovery_report.md 的半写窗口编号对应）：
    - ``summary_missing``    : raw meta 完整但 measured 摘要缺失 —— 半写窗口 H1，
                               只需补跑 measure，raw 不必重生成；
    - ``raw_meta_incomplete`` / ``raw_meta_corrupt`` / ``orphan_raw``:
                               raw 不完整 —— 整臂重新生成（orphan = 有 batch 分片但
                               无 meta.json，即生成中途被 kill）；
    - ``done_but_meta_bad``  : 状态记 done 但 meta 不完整 —— 历史脏状态，需人工复核；
    - 状态 unknown / failed / incomplete 直接透传给总控。
    """
    root_p = Path(root)
    paired = (root_p / "raw").is_dir() and (root_p / "measured").is_dir()
    rows: list[dict] = []

    if paired:
        arm_iter = sorted((root_p / "raw").iterdir()) if (root_p / "raw").is_dir() else []
        arm_dirs = [d for d in arm_iter if d.is_dir()]
    else:
        arm_dirs = [d for d in sorted(root_p.iterdir()) if d.is_dir()
                    and ((d / "meta.json").exists() or (d / STATUS_FILENAME).exists()
                         or (d / LOG_FILENAME).exists())] if root_p.is_dir() else []

    for arm_dir in arm_dirs:
        name = arm_dir.name
        measured_dir = root_p / "measured" / name if paired else None
        state = read_state(arm_dir, expected_tasks=expected_tasks)
        meta_state = _meta_state(arm_dir, expected_tasks)

        has_eval = bool(measured_dir and (measured_dir / "eval.json").is_file())
        has_summary = bool(measured_dir and any(
            f.is_file() and f.name.startswith("summary") and f.suffix == ".json"
            for f in measured_dir.iterdir())) if measured_dir and measured_dir.is_dir() else False
        has_batches = any(f.is_file() and f.name.startswith("batch_") for f in arm_dir.iterdir()) \
            if arm_dir.is_dir() else False

        flags: list[str] = []
        if meta_state in ("incomplete_count", "corrupt"):
            flags.append(f"raw_meta_{meta_state}")
            if state.get("state") == STATE_DONE:
                flags.append("done_but_meta_bad")
        elif meta_state == "missing" and has_batches:
            if state.get("state") not in (STATE_STARTED, STATE_RUNNING):
                flags.append("orphan_raw")  # 运行中臂有分片无 meta 属正常，不告警
        elif meta_state == "valid":
            if paired and not has_summary:
                flags.append("summary_missing")
            if state.get("state") == STATE_DONE and "meta_validated" not in state:
                flags.append("done_but_meta_bad")

        rows.append({
            "arm": name,
            "state": state.get("state", STATE_UNKNOWN),
            "state_source": state.get("source", "none"),
            "raw_meta": meta_state,
            "eval_json": has_eval,
            "summary": has_summary,
            "flags": flags,
            "age_min": _latest_age_min([
                arm_dir / STATUS_FILENAME, arm_dir / LOG_FILENAME,
                arm_dir / "meta.json",
                measured_dir / "eval.json" if measured_dir else arm_dir / "meta.json",
            ]),
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    """总控巡检 one-liner：``python -m src.v25.result_writer runs/v25 [--stale-min 30]``。"""
    ap = argparse.ArgumentParser(description="B06 queue health scan (pure stdlib, CPU-only)")
    ap.add_argument("root", help="runs 输出根目录（如 runs/v25 或 runs/v25/raw 的父目录）")
    ap.add_argument("--expected-tasks", type=int, default=750)
    ap.add_argument("--stale-min", type=float, default=None,
                    help="最新产物 mtime 超过该分钟数的臂标记 stale")
    ap.add_argument("--json", action="store_true", help="机器可读 JSON 输出")
    args = ap.parse_args(argv)

    rows = scan_status(args.root, expected_tasks=args.expected_tasks)
    n_flagged = 0
    for r in rows:
        if args.stale_min is not None and r["age_min"] is not None and r["age_min"] > args.stale_min:
            r["flags"] = list(r["flags"]) + ["stale"]
        if r["flags"]:
            n_flagged += 1

    if args.json:
        print(json.dumps({"root": str(args.root), "n_arms": len(rows),
                          "n_flagged": n_flagged, "rows": rows}, ensure_ascii=False, indent=2))
    else:
        print(f"scan {args.root}: {len(rows)} arms, {n_flagged} flagged")
        for r in rows:
            line = (f"  {r['arm']:28s} state={r['state']:<10s} meta={r['raw_meta']:<16s}"
                    f" eval={int(r['eval_json'])} summary={int(r['summary'])}")
            if r["age_min"] is not None:
                line += f" age={r['age_min']}min"
            if r["flags"]:
                line += f"  FLAGS={','.join(r['flags'])}"
            print(line)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
