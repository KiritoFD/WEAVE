"""B08: Windows/Linux 跨平台运行器与恢复助手（Slot B，纯 CPU / 纯 stdlib）。

职责边界（docs/19_pull_together/agent2.md §六 B08 / §九文件所有权）：
- 只做纯文本脚本渲染（Linux tmux + .sh；Windows .bat + schtasks）、
  五状态联合判别（对接 src/v25/result_writer.py::ResultWriter）与幂等恢复建议；
- 绝不执行 ssh、绝不启动真实 GPU/CUDA 进程、绝不修改现有实验脚本；
- 全部路径使用相对路径（workdir_rel / log_rel / err_rel），无硬编码绝对路径。

吸收 V25 实操教训（docs/19_pull_together/PROGRESS.md & b06_recovery_report.md）：
1. Linux (4090 tmux):
   - 禁止 tmux 内联复杂多级引号命令 → 固定生成独立 launch.sh，tmux 只执行 ``bash <launch.sh>``；
   - 禁止裸 ``pkill -f <pattern>`` 自杀 → 统一使用 ``[x]`` 括号正则技巧（如 ``[e]xp.v25.foo``）；
   - 默认注入 ``OMP_NUM_THREADS=2``、``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``、
     ``HF_HUB_OFFLINE=1``、``TRANSFORMERS_OFFLINE=1``。
2. Windows (3060 schtasks):
   - 禁止 schtasks ``/TR`` 内联引号参数（exit=2 根因）→ ``/TR`` 只指向固定 ``.bat`` 文件；
   - 必须使用 ``python -u`` 防止 stdout 缓冲假死；
   - 必须注入 ``HF_HUB_OFFLINE=1`` / ``TRANSFORMERS_OFFLINE=1`` 防 httpx 网络瞬断；
   - 禁用 ``expandable_segments``（3060 Windows Server 铁律）；
   - 固定 ``PYTHONUTF8=1`` 与 ``OMP_NUM_THREADS=2``；路径统一反斜杠。
3. 五状态联合判别（started / running / done / failed / incomplete）与 B06 H1/H2 半完成窗口分诊。
"""
from __future__ import annotations

import os

# ---- agent2.md §十一 CPU 资源隔离：纯 CPU 入口固定环境变量 ----
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import argparse
import json
import re
import shlex
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.v25.result_writer import (  # noqa: E402
    STATE_DONE,
    STATE_FAILED,
    STATE_INCOMPLETE,
    STATE_RUNNING,
    STATE_STARTED,
    STATE_UNKNOWN,
    ResultWriter,
    read_json_tolerant,
    read_state,
    validate_raw_meta_dict,
)

__all__ = [
    "VALID_HOSTS",
    "VALID_QUEUES",
    "SCHTASKS_RUNNING",
    "SCHTASKS_NOT_RUN",
    "SCHTASKS_TERMINATED",
    "JobSpec",
    "RenderedJobBundle",
    "JobStateReport",
    "RecoveryPlan",
    "default_env_for_host",
    "make_bracket_regex",
    "render_safe_pkill_cmd",
    "render_linux_script",
    "render_windows_bat",
    "render_job_bundle",
    "write_job_bundle",
    "parse_exit_code",
    "inspect_job_state",
    "recommend_recovery",
    "main",
]

VALID_HOSTS = ("linux", "windows")
VALID_QUEUES = ("4090", "3060")

# Windows Task Scheduler 标准状态码（十进制）
SCHTASKS_RUNNING = 267009       # 0x41301 SCHED_S_TASK_RUNNING
SCHTASKS_NOT_RUN = 267011       # 0x41303 SCHED_S_TASK_HAS_NOT_RUN
SCHTASKS_TERMINATED = 267014    # 0x41306 SCHED_S_TASK_TERMINATED

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_SAFE_MODULE_RE = re.compile(r"^[A-Za-z0-9_.]+$")
_ABS_PATH_RE = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\|/)")

# --------------------------------------------------------------------------- #
# 失败特征模式（按优先级匹配）
# --------------------------------------------------------------------------- #
_OOM_PATTERNS = (
    re.compile(r"CUDA out of memory", re.IGNORECASE),
    re.compile(r"OutOfMemoryError", re.IGNORECASE),
    re.compile(r"CUDNN_STATUS_ALLOC_FAILED", re.IGNORECASE),
    re.compile(r"torch\.cuda\.OutOfMemoryError", re.IGNORECASE),
    re.compile(r"\boom[- ]killer\b", re.IGNORECASE),
)

_HTTPX_HF_PATTERNS = (
    re.compile(r"httpx\.(?:ConnectError|ReadTimeout|RemoteProtocolError|ConnectTimeout|HTTPError)"),
    re.compile(r"requests\.exceptions\.(?:ConnectionError|ReadTimeout|SSLError)"),
    re.compile(r"huggingface_hub\.errors\.", re.IGNORECASE),
    re.compile(r"LocalEntryNotFoundError"),
    re.compile(r"OfflineModeIsEnabled"),
    re.compile(r"MaxRetryError.*huggingface", re.IGNORECASE),
)

_LAUNCH_ERR_PATTERNS = (
    re.compile(r"is not recognized as an internal or external command", re.IGNORECASE),
    re.compile(r"The system cannot find the (?:file|path) specified", re.IGNORECASE),
    re.compile(r"No module named\s+"),
    re.compile(r"can't open file '.*': \[Errno 2\]"),
    re.compile(r"No such file or directory"),
)

_NAN_PATTERNS = (
    re.compile(r"\bloss\s*[:=]\s*nan\b", re.IGNORECASE),
    re.compile(r"\bnan detected\b", re.IGNORECASE),
    re.compile(r"ValueError:.*nan", re.IGNORECASE),
)

_TRACEBACK_PATTERNS = (
    re.compile(r"Traceback \(most recent call last\):"),
    re.compile(r"^(?:RuntimeError|FileNotFoundError|KeyError|NameError|ValueError|AssertionError|OSError|TypeError):", re.MULTILINE),
    re.compile(r"json\.decoder\.JSONDecodeError"),
    re.compile(r"subprocess\.CalledProcessError"),
)

_PARTIAL_INTERRUPT_PATTERNS = (
    re.compile(r"KeyboardInterrupt"),
    re.compile(r"SIGTERM|SIGINT|UserWarning:.*killed", re.IGNORECASE),
)


def _is_relative_path(p: str) -> bool:
    if not p or "\n" in p or "\r" in p:
        return False
    return not _ABS_PATH_RE.match(p.strip())


def default_env_for_host(host_os: str) -> dict[str, str]:
    """返回平台默认环境变量（纯 CPU/GPU 隔离与离线护栏）。"""
    if host_os not in VALID_HOSTS:
        raise ValueError(f"unsupported host_os={host_os!r}, expected one of {VALID_HOSTS}")
    env = {
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONUNBUFFERED": "1",
    }
    if host_os == "linux":
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    else:
        env["PYTHONUTF8"] = "1"
    return env


def make_bracket_regex(pattern: str) -> str:
    """将 ``exp.v25.foo`` 转为 ``[e]xp.v25.foo``，防止 pgrep/pkill -f 匹配自身命令行。"""
    s = pattern.strip()
    if not s:
        raise ValueError("pattern must be non-empty")
    if s.startswith("[") and "]" in s:
        return s
    first = s[0]
    if first in r"\^$.|?*+(){}[]":
        return s
    return f"[{first}]{s[1:]}"


def render_safe_pkill_cmd(pattern: str) -> str:
    """渲染防自杀 pkill 命令（强制使用 [x] 括号正则技巧）。"""
    return f"pkill -f {shlex.quote(make_bracket_regex(pattern))}"


# --------------------------------------------------------------------------- #
# JobSpec 与包装脚本渲染
# --------------------------------------------------------------------------- #
@dataclass
class JobSpec:
    """跨平台实验任务规格（全相对路径，纯文本可序列化）。"""

    job_id: str
    host_os: str
    module: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    workdir_rel: str = "."
    log_rel: str = ""
    err_rel: str = ""
    gpu_queue: str = ""

    def __post_init__(self) -> None:
        if not self.job_id or not _SAFE_ID_RE.match(self.job_id):
            raise ValueError(
                f"job_id={self.job_id!r} must match [A-Za-z0-9_.-]+ (no spaces/slashes/quotes)"
            )
        if self.host_os not in VALID_HOSTS:
            raise ValueError(f"host_os={self.host_os!r} must be one of {VALID_HOSTS}")
        if not self.module or not _SAFE_MODULE_RE.match(self.module):
            raise ValueError(f"module={self.module!r} must be a valid dotted Python module")

        self.args = [str(a) for a in self.args]
        for a in self.args:
            if "\n" in a or "\r" in a:
                raise ValueError(f"arg contains newline: {a!r}")

        if not self.gpu_queue:
            self.gpu_queue = "4090" if self.host_os == "linux" else "3060"
        if self.gpu_queue not in VALID_QUEUES:
            raise ValueError(f"gpu_queue={self.gpu_queue!r} must be one of {VALID_QUEUES}")

        if not self.workdir_rel:
            self.workdir_rel = "."
        if not self.log_rel:
            self.log_rel = f"runs/v25/{self.job_id}.log"
        if not self.err_rel:
            self.err_rel = f"runs/v25/{self.job_id}.err.log"

        for field_name, rel_val in (
            ("workdir_rel", self.workdir_rel),
            ("log_rel", self.log_rel),
            ("err_rel", self.err_rel),
        ):
            if not _is_relative_path(rel_val):
                raise ValueError(f"{field_name}={rel_val!r} must be a relative path")

        merged_env = default_env_for_host(self.host_os)
        for k, v in self.env.items():
            ks, vs = str(k), str(v)
            if not ks or any(ch in ks for ch in ("=", '"', "'", "\n", "\r", " ")):
                raise ValueError(f"invalid env var name: {ks!r}")
            if "\n" in vs or "\r" in vs:
                raise ValueError(f"env value for {ks!r} contains newline")
            merged_env[ks] = vs
        if self.host_os == "windows":
            # Windows 3060 强制禁止 expandable_segments
            alloc_conf = merged_env.get("PYTORCH_CUDA_ALLOC_CONF", "")
            if "expandable_segments" in alloc_conf:
                raise ValueError(
                    "PYTORCH_CUDA_ALLOC_CONF with expandable_segments is disallowed on Windows"
                )
        self.env = merged_env

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RenderedJobBundle:
    """渲染产物：包装脚本全文 + 调度/查询/安全清理命令。"""

    job_id: str
    host_os: str
    gpu_queue: str
    script_name: str
    script_rel: str
    script_text: str
    setup_cmd: str
    launch_cmd: str
    status_cmd: str
    commands: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _win_rel(path_str: str) -> str:
    norm = str(PureWindowsPath(path_str.replace("/", "\\")))
    return norm if norm else "."


def _posix_rel(path_str: str) -> str:
    norm = str(PurePosixPath(path_str.replace("\\", "/")))
    return norm if norm else "."


def _quote_bat_arg(arg: str) -> str:
    if '"' in arg or "%" in arg:
        raise ValueError(f"Windows batch arg cannot contain '\"' or '%': {arg!r}")
    if arg == "" or any(ch in arg for ch in (" ", "\t", "&", "|", "<", ">", "^", "(", ")")):
        return f'"{arg}"'
    return arg


def render_linux_script(spec: JobSpec, script_rel: str | None = None) -> RenderedJobBundle:
    """渲染 Linux (4090) ``launch.sh`` 与 ``tmux`` 启动命令。

    防呆设计：
    - 独立 ``.sh`` 文件 + ``tmux new-session -d -s <id> bash <script_rel>``，
      彻底杜绝 tmux 内联双引号/转义地狱；
    - 内置 ``pgrep -f "[x]..."`` 防重复启动守卫（绝不使用裸 ``pkill -f``）；
    - 自动从脚本目录向上探测仓库根（以 ``paths.py`` 为锚）并切换至 ``workdir_rel``。
    """
    if spec.host_os != "linux":
        raise ValueError(f"render_linux_script requires host_os='linux', got {spec.host_os!r}")
    script_name = f"{spec.job_id}_launch.sh"
    rel = _posix_rel(script_rel or f"runs/v25/jobs/{script_name}")
    if not _is_relative_path(rel):
        raise ValueError(f"script_rel={rel!r} must be relative")

    workdir = _posix_rel(spec.workdir_rel)
    log_p = _posix_rel(spec.log_rel)
    err_p = _posix_rel(spec.err_rel)
    log_dir = _posix_rel(str(PurePosixPath(log_p).parent))
    err_dir = _posix_rel(str(PurePosixPath(err_p).parent))

    safe_pat = make_bracket_regex(spec.module)
    args_joined = " ".join(shlex.quote(a) for a in spec.args)
    py_cmd = f"python -u -m {shlex.quote(spec.module)}" + (f" {args_joined}" if args_joined else "")

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"# Auto-generated by tools/v25_cpu/b08_host_runner.py (job_id={spec.job_id}, queue={spec.gpu_queue})",
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'REPO_DIR="$SCRIPT_DIR"',
        'while [ "$REPO_DIR" != "$(dirname "$REPO_DIR")" ] && [ ! -f "$REPO_DIR/paths.py" ]; do',
        '    REPO_DIR="$(dirname "$REPO_DIR")"',
        "done",
        'if [ ! -f "$REPO_DIR/paths.py" ]; then',
        '    REPO_DIR="$SCRIPT_DIR"',
        "fi",
        'cd "$REPO_DIR"',
    ]
    if workdir != ".":
        lines.append(f"cd {shlex.quote(workdir)}")

    mkdir_targets = [shlex.quote(log_dir)]
    if err_dir != log_dir:
        mkdir_targets.append(shlex.quote(err_dir))
    lines.append("mkdir -p " + " ".join(mkdir_targets))
    lines.append("")

    for k, v in spec.env.items():
        lines.append(f"export {k}={shlex.quote(v)}")
    lines.append("")

    lines.extend([
        "# Duplicate-launch guard via [x] bracket trick (never self-matches)",
        f"if pgrep -f {shlex.quote(safe_pat)} >/dev/null 2>&1; then",
        f'    echo "[b08] job module {spec.module} is already running; skipping duplicate launch" >&2',
        "    exit 0",
        "fi",
        "",
    ])

    if log_p == err_p:
        lines.append(f"{py_cmd} >> {shlex.quote(log_p)} 2>&1")
    else:
        lines.append(f"{py_cmd} >> {shlex.quote(log_p)} 2>> {shlex.quote(err_p)}")
    lines.extend([
        "RC=$?",
        'exit "$RC"',
        "",
    ])

    script_text = "\n".join(lines)
    tmux_cmd = f"tmux new-session -d -s {shlex.quote(spec.job_id)} {shlex.quote('bash ' + rel)}"
    setup_cmd = f"chmod +x {shlex.quote(rel)}"
    status_cmd = f"tmux has-session -t {shlex.quote(spec.job_id)} && tmux capture-pane -pt {shlex.quote(spec.job_id)} | tail -n 20"
    pkill_safe = render_safe_pkill_cmd(spec.module)

    return RenderedJobBundle(
        job_id=spec.job_id,
        host_os="linux",
        gpu_queue=spec.gpu_queue,
        script_name=script_name,
        script_rel=rel,
        script_text=script_text,
        setup_cmd=setup_cmd,
        launch_cmd=tmux_cmd,
        status_cmd=status_cmd,
        commands={
            "setup_cmd": setup_cmd,
            "tmux_cmd": tmux_cmd,
            "launch_cmd": tmux_cmd,
            "status_cmd": status_cmd,
            "pgrep_safe_cmd": f"pgrep -af {shlex.quote(safe_pat)}",
            "pkill_safe_cmd": pkill_safe,
            "tail_log_cmd": f"tail -n 40 {shlex.quote(log_p)}",
        },
    )


def render_windows_bat(spec: JobSpec, script_rel: str | None = None) -> RenderedJobBundle:
    """渲染 Windows (3060) ``.bat`` 包装脚本与 ``schtasks`` 注册/启动命令。

    防呆设计：
    - ``schtasks /TR`` 仅指向单一 ``.bat`` 路径，绝不内联复杂引号参数（避免 exit=2）；
    - 强制 ``python -u`` 防止 stdout 缓冲假死；
    - 固定 ``HF_HUB_OFFLINE=1`` / ``TRANSFORMERS_OFFLINE=1`` / ``PYTHONUTF8=1`` / ``OMP_NUM_THREADS=2``；
    - 禁止 PowerShell 嵌套转义，全部路径转为反斜杠。
    """
    if spec.host_os != "windows":
        raise ValueError(f"render_windows_bat requires host_os='windows', got {spec.host_os!r}")
    script_name = f"{spec.job_id}_launch.bat"
    rel = _win_rel(script_rel or f"runs\\v25\\jobs\\{script_name}")
    if not _is_relative_path(rel):
        raise ValueError(f"script_rel={rel!r} must be relative")

    workdir = _win_rel(spec.workdir_rel)
    log_p = _win_rel(spec.log_rel)
    err_p = _win_rel(spec.err_rel)
    log_dir = _win_rel(str(PureWindowsPath(log_p).parent))
    err_dir = _win_rel(str(PureWindowsPath(err_p).parent))

    quoted_args = " ".join(_quote_bat_arg(a) for a in spec.args)
    py_cmd = f"python -u -m {spec.module}" + (f" {quoted_args}" if quoted_args else "")

    lines = [
        "@echo off",
        "setlocal EnableExtensions",
        f"rem Auto-generated by tools/v25_cpu/b08_host_runner.py (job_id={spec.job_id}, queue={spec.gpu_queue})",
        'cd /d "%~dp0"',
        'if exist "paths.py" goto :found_repo',
        'if exist "..\\paths.py" cd /d ".." & goto :found_repo',
        'if exist "..\\..\\paths.py" cd /d "..\\.." & goto :found_repo',
        'if exist "..\\..\\..\\paths.py" cd /d "..\\..\\.." & goto :found_repo',
        ":found_repo",
    ]
    if workdir != ".":
        lines.append(f'cd /d "{workdir}"')

    if log_dir and log_dir != ".":
        lines.append(f'if not exist "{log_dir}" mkdir "{log_dir}"')
    if err_dir and err_dir != "." and err_dir != log_dir:
        lines.append(f'if not exist "{err_dir}" mkdir "{err_dir}"')
    lines.append("")

    for k, v in spec.env.items():
        if '"' in v or "%" in v:
            raise ValueError(f"Windows batch env value for {k!r} cannot contain '\"' or '%': {v!r}")
        lines.append(f'set "{k}={v}"')
    lines.append("")

    if log_p == err_p:
        lines.append(f'{py_cmd} >> "{log_p}" 2>&1')
    else:
        lines.append(f'{py_cmd} >> "{log_p}" 2>> "{err_p}"')

    lines.extend([
        'set "RC=%ERRORLEVEL%"',
        "exit /b %RC%",
        "",
    ])

    script_text = "\r\n".join(lines)

    tr_val = f'"{rel}"' if " " in rel else rel
    schtasks_create = f"schtasks /Create /TN {spec.job_id} /TR {tr_val} /SC ONCE /ST 00:00 /F"
    schtasks_run = f"schtasks /Run /TN {spec.job_id}"
    schtasks_query = f"schtasks /Query /TN {spec.job_id} /V /FO LIST"

    return RenderedJobBundle(
        job_id=spec.job_id,
        host_os="windows",
        gpu_queue=spec.gpu_queue,
        script_name=script_name,
        script_rel=rel,
        script_text=script_text,
        setup_cmd=schtasks_create,
        launch_cmd=schtasks_run,
        status_cmd=schtasks_query,
        commands={
            "schtasks_create_cmd": schtasks_create,
            "schtasks_run_cmd": schtasks_run,
            "schtasks_query_cmd": schtasks_query,
            "setup_cmd": schtasks_create,
            "launch_cmd": schtasks_run,
            "status_cmd": schtasks_query,
            "proc_check_cmd": 'tasklist /fi "imagename eq python.exe" /v',
        },
    )


def render_job_bundle(spec: JobSpec, script_rel: str | None = None) -> RenderedJobBundle:
    """按 ``spec.host_os`` 分派渲染包装脚本与调度命令。"""
    if spec.host_os == "linux":
        return render_linux_script(spec, script_rel=script_rel)
    return render_windows_bat(spec, script_rel=script_rel)


def write_job_bundle(
    spec: JobSpec,
    out_dir: str | Path,
    *,
    script_rel: str | None = None,
) -> tuple[Path, RenderedJobBundle]:
    """将渲染脚本写入 ``out_dir``（纯文本，不执行调度）。"""
    bundle = render_job_bundle(spec, script_rel=script_rel)
    out_p = Path(out_dir) / bundle.script_name
    out_p.parent.mkdir(parents=True, exist_ok=True)
    # 显式 newline="" 保留 Linux LF / Windows CRLF 精确换行
    with open(out_p, "w", encoding="utf-8", newline="") as fh:
        fh.write(bundle.script_text)
    if spec.host_os == "linux":
        try:
            out_p.chmod(0o755)
        except OSError:
            pass
    return out_p, bundle


# --------------------------------------------------------------------------- #
# 五状态联合判别（ResultWriter + Exit Code + Log/Stderr + B06 H1/H2 窗口）
# --------------------------------------------------------------------------- #
@dataclass
class JobStateReport:
    """统一五状态判别结果（支持属性访问与 dict 下标访问）。"""

    state: str
    failure_kind: str | None
    reasons: list[str]
    task_exit_code: int | None
    result_writer_state: str
    meta_status: str
    window: str | None
    log_stale: bool
    host_os: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and hasattr(self, key)


def parse_exit_code(val: int | str | None) -> int | None:
    """解析整数或 0x 十六进制退出码（兼容 Windows schtasks Last Result）。"""
    if val is None:
        return None
    if isinstance(val, bool):
        raise ValueError(f"invalid exit code bool: {val!r}")
    if isinstance(val, int):
        return val
    s = str(val).strip()
    if not s or s.lower() in ("none", "null", "n/a"):
        return None
    if s.lower().startswith("0x"):
        return int(s, 16)
    return int(s, 10)


def _detect_log_failure(log_text: str, err_text: str) -> tuple[str | None, list[str]]:
    combined = f"{log_text or ''}\n{err_text or ''}"
    reasons: list[str] = []
    if any(p.search(combined) for p in _OOM_PATTERNS):
        reasons.append("CUDA/host OOM signature found in output")
        return "cuda_oom", reasons
    if any(p.search(combined) for p in _HTTPX_HF_PATTERNS):
        reasons.append("httpx/HuggingFace network or offline-cache error found in output")
        return "hf_network_error", reasons
    if any(p.search(combined) for p in _LAUNCH_ERR_PATTERNS):
        reasons.append("script/command/module launch failure found in output")
        return "launch_error", reasons
    if any(p.search(combined) for p in _NAN_PATTERNS):
        reasons.append("NaN divergence detected in output")
        return "numerical_nan", reasons
    if any(p.search(combined) for p in _TRACEBACK_PATTERNS):
        reasons.append("Python traceback/exception found in output")
        return "python_traceback", reasons
    return None, reasons


def _inspect_result_dir(
    result_state_dir: str | Path | None,
    expected_tasks: int,
) -> dict[str, Any]:
    if result_state_dir is None:
        return {
            "rw_state": STATE_UNKNOWN,
            "meta_status": "unchecked",
            "has_batches": False,
            "window": None,
        }
    arm_dir = Path(result_state_dir)
    if not arm_dir.exists():
        return {
            "rw_state": STATE_UNKNOWN,
            "meta_status": "dir_missing",
            "has_batches": False,
            "window": None,
        }

    rw_info = read_state(arm_dir, expected_tasks=expected_tasks)
    rw_state = rw_info.get("state", STATE_UNKNOWN)

    meta_p = arm_dir / "meta.json"
    if not meta_p.exists():
        meta_status = "missing"
    else:
        meta_obj = read_json_tolerant(meta_p)
        if meta_obj is None:
            meta_status = "corrupt"
        elif validate_raw_meta_dict(meta_obj, expected_tasks):
            meta_status = "valid"
        else:
            meta_status = "incomplete_count"

    has_batches = any(
        f.is_file() and f.name.startswith("batch_") for f in arm_dir.iterdir()
    )

    # 检测 B06 H1/H2 半完成窗口（仅当位于 raw/<arm> 配对布局且兄弟 measured/ 目录存在时）
    window: str | None = None
    if arm_dir.parent.name == "raw":
        measured_root = arm_dir.parent.parent / "measured"
        if measured_root.is_dir():
            measured_dir = measured_root / arm_dir.name
            eval_p = measured_dir / "eval.json"
            has_summary = measured_dir.is_dir() and any(
                f.is_file() and f.name.startswith("summary") and f.suffix == ".json"
                for f in measured_dir.iterdir()
            )
            if eval_p.exists() and read_json_tolerant(eval_p) is None:
                window = "H2"
            elif meta_status == "valid" and not has_summary:
                window = "H1"

    return {
        "rw_state": rw_state,
        "meta_status": meta_status,
        "has_batches": has_batches,
        "window": window,
    }


def inspect_job_state(
    log_text: str = "",
    err_text: str = "",
    task_exit_code: int | str | None = None,
    result_state_dir: str | Path | None = None,
    *,
    expected_tasks: int = 750,
    log_age_min: float | None = None,
    stale_threshold_min: float = 15.0,
    process_alive: bool | None = None,
    host_os: str | None = None,
) -> JobStateReport:
    """统一五状态判别：``started / running / done / failed / incomplete``。

    判别优先级（严格防止"失败仍记 done"）：
    1. 确定性失败（``failed``）：
       - ``task_exit_code`` 为非零失败码（含 Windows ``exit=2``、``267014``、``0xC0000409`` 等）；
       - 日志/stderr 含 OOM、httpx/HF 断连、启动路径缺失、NaN、Traceback；
       - ``ResultWriter`` 已持久化 ``failed``。
    2. 进行中（``running``）：
       - Windows ``267009`` (``0x41301``) 或 ``process_alive=True`` 且无致命报错；
       - 或 ``ResultWriter`` 为 ``running`` 且未超时停更。
    3. 半完成/中断/脏态（``incomplete``）：
       - B06 H1（``meta.json`` 完整但 ``measured/<arm>`` 缺 summary）或 H2（``eval.json`` 半写损坏）；
       - ``meta.json`` 损坏或 task_id 计数不足（即使 ``ResultWriter`` 历史记了 done 或 exit=0）；
       - 孤立 ``batch_*.pt``（无 ``meta.json`` 且进程已结束）；
       - ``ResultWriter`` 为 ``incomplete``，或状态为 ``started/running`` 但日志停更 / 进程已死；
       - 检测到 ``KeyboardInterrupt`` 等中断信号。
    4. 已完成（``done``）：
       - 无上述错误/半完成信号，且满足：
         (a) ``result_state_dir`` 存在时 ``meta.json`` 必须 ``valid``（若无 meta 则需 ``ResultWriter == done`` 且 ``exit_code == 0``）；
         (b) 无 ``result_state_dir`` 时 ``exit_code == 0``。
    5. 已启动/待运行（``started``）：
       - Windows ``267011`` (``0x41303`` 未运行)、``ResultWriter == started``（未停更）、或初始空状态。
    """
    rc = parse_exit_code(task_exit_code)
    rinfo = _inspect_result_dir(result_state_dir, expected_tasks=expected_tasks)
    rw_state: str = rinfo["rw_state"]
    meta_status: str = rinfo["meta_status"]
    has_batches: bool = rinfo["has_batches"]
    window: str | None = rinfo["window"]

    log_stale = log_age_min is not None and float(log_age_min) > float(stale_threshold_min)
    reasons: list[str] = []

    # 1) 致命失败信号优先
    if rc == 2:
        reasons.append("task_exit_code=2 (Windows schtasks bat/path or inline /TR quote error)")
        return JobStateReport(
            state=STATE_FAILED,
            failure_kind="schtasks_path_or_quote_error",
            reasons=reasons,
            task_exit_code=rc,
            result_writer_state=rw_state,
            meta_status=meta_status,
            window=window,
            log_stale=log_stale,
            host_os=host_os,
        )

    if rc == SCHTASKS_TERMINATED:
        reasons.append(f"task_exit_code={rc} (schtasks task terminated)")
        return JobStateReport(
            state=STATE_FAILED,
            failure_kind="process_terminated",
            reasons=reasons,
            task_exit_code=rc,
            result_writer_state=rw_state,
            meta_status=meta_status,
            window=window,
            log_stale=log_stale,
            host_os=host_os,
        )

    if rc is not None and rc not in (0, SCHTASKS_RUNNING, SCHTASKS_NOT_RUN):
        log_kind, log_reasons = _detect_log_failure(log_text, err_text)
        reasons.append(f"non-zero task_exit_code={rc}")
        reasons.extend(log_reasons)
        return JobStateReport(
            state=STATE_FAILED,
            failure_kind=log_kind or "nonzero_exit",
            reasons=reasons,
            task_exit_code=rc,
            result_writer_state=rw_state,
            meta_status=meta_status,
            window=window,
            log_stale=log_stale,
            host_os=host_os,
        )

    log_kind, log_reasons = _detect_log_failure(log_text, err_text)
    if log_kind is not None:
        return JobStateReport(
            state=STATE_FAILED,
            failure_kind=log_kind,
            reasons=log_reasons,
            task_exit_code=rc,
            result_writer_state=rw_state,
            meta_status=meta_status,
            window=window,
            log_stale=log_stale,
            host_os=host_os,
        )

    if rw_state == STATE_FAILED:
        reasons.append("ResultWriter persisted state=failed")
        return JobStateReport(
            state=STATE_FAILED,
            failure_kind="result_writer_failed",
            reasons=reasons,
            task_exit_code=rc,
            result_writer_state=rw_state,
            meta_status=meta_status,
            window=window,
            log_stale=log_stale,
            host_os=host_os,
        )

    # 2) 运行中信号（267009 或进程存活）
    is_live = (rc == SCHTASKS_RUNNING) or (process_alive is True)
    if is_live:
        if rc == SCHTASKS_RUNNING:
            reasons.append("task_exit_code=267009 (0x41301 schtasks running)")
        if process_alive is True:
            reasons.append("process_alive=True")
        if log_stale:
            reasons.append(f"log stale ({log_age_min}min > {stale_threshold_min}min); check python -u buffering")
        return JobStateReport(
            state=STATE_RUNNING,
            failure_kind="log_stale_while_alive" if log_stale else None,
            reasons=reasons,
            task_exit_code=rc,
            result_writer_state=rw_state,
            meta_status=meta_status,
            window=window,
            log_stale=log_stale,
            host_os=host_os,
        )

    # 3) 半完成 / 中断 / 完整性闸降级（incomplete）
    combined = f"{log_text or ''}\n{err_text or ''}"
    if any(p.search(combined) for p in _PARTIAL_INTERRUPT_PATTERNS):
        reasons.append("interruption signal found in output")
        return JobStateReport(
            state=STATE_INCOMPLETE,
            failure_kind="interrupted",
            reasons=reasons,
            task_exit_code=rc,
            result_writer_state=rw_state,
            meta_status=meta_status,
            window=window,
            log_stale=log_stale,
            host_os=host_os,
        )

    if window == "H2":
        reasons.append("B06 H2 window: measured eval.json exists but is corrupt/half-written")
        return JobStateReport(
            state=STATE_INCOMPLETE,
            failure_kind="h2_corrupt_eval_json",
            reasons=reasons,
            task_exit_code=rc,
            result_writer_state=rw_state,
            meta_status=meta_status,
            window="H2",
            log_stale=log_stale,
            host_os=host_os,
        )

    if window == "H1":
        reasons.append("B06 H1 window: raw meta.json valid but measured summary missing")
        return JobStateReport(
            state=STATE_INCOMPLETE,
            failure_kind="h1_summary_missing",
            reasons=reasons,
            task_exit_code=rc,
            result_writer_state=rw_state,
            meta_status=meta_status,
            window="H1",
            log_stale=log_stale,
            host_os=host_os,
        )

    if meta_status in ("corrupt", "incomplete_count"):
        reasons.append(f"raw meta.json {meta_status} (done refused)")
        return JobStateReport(
            state=STATE_INCOMPLETE,
            failure_kind=f"raw_meta_{meta_status}",
            reasons=reasons,
            task_exit_code=rc,
            result_writer_state=rw_state,
            meta_status=meta_status,
            window=window,
            log_stale=log_stale,
            host_os=host_os,
        )

    if rw_state == STATE_INCOMPLETE:
        reasons.append("ResultWriter persisted state=incomplete")
        return JobStateReport(
            state=STATE_INCOMPLETE,
            failure_kind="result_writer_incomplete",
            reasons=reasons,
            task_exit_code=rc,
            result_writer_state=rw_state,
            meta_status=meta_status,
            window=window,
            log_stale=log_stale,
            host_os=host_os,
        )

    if meta_status == "missing" and has_batches and rw_state not in (STATE_STARTED, STATE_RUNNING):
        reasons.append("orphan raw batch_*.pt shards without meta.json")
        return JobStateReport(
            state=STATE_INCOMPLETE,
            failure_kind="orphan_raw",
            reasons=reasons,
            task_exit_code=rc,
            result_writer_state=rw_state,
            meta_status=meta_status,
            window=window,
            log_stale=log_stale,
            host_os=host_os,
        )

    if rw_state in (STATE_STARTED, STATE_RUNNING) and (log_stale or process_alive is False or rc == 0):
        if log_stale:
            reasons.append(f"state={rw_state} but log stale ({log_age_min}min > {stale_threshold_min}min)")
        if process_alive is False:
            reasons.append(f"state={rw_state} but process_alive=False")
        if rc == 0:
            reasons.append(f"exit_code=0 but ResultWriter stayed at {rw_state}")
        return JobStateReport(
            state=STATE_INCOMPLETE,
            failure_kind="log_stale" if log_stale else "stopped_midrun",
            reasons=reasons,
            task_exit_code=rc,
            result_writer_state=rw_state,
            meta_status=meta_status,
            window=window,
            log_stale=log_stale,
            host_os=host_os,
        )

    if rw_state == STATE_RUNNING:
        reasons.append("ResultWriter state=running (not stale)")
        return JobStateReport(
            state=STATE_RUNNING,
            failure_kind=None,
            reasons=reasons,
            task_exit_code=rc,
            result_writer_state=rw_state,
            meta_status=meta_status,
            window=window,
            log_stale=log_stale,
            host_os=host_os,
        )

    # 4) 已完成判据（done）
    if result_state_dir is not None:
        if meta_status == "valid" and (rw_state == STATE_DONE or rc == 0 or bool(log_text.strip())):
            reasons.append("raw meta.json valid (750 unique ids) with no error signals")
            return JobStateReport(
                state=STATE_DONE,
                failure_kind=None,
                reasons=reasons,
                task_exit_code=rc,
                result_writer_state=rw_state,
                meta_status=meta_status,
                window=window,
                log_stale=log_stale,
                host_os=host_os,
            )
        if rw_state == STATE_DONE and meta_status == "missing" and not has_batches and rc in (0, None):
            reasons.append("ResultWriter state=done (non-raw task)")
            return JobStateReport(
                state=STATE_DONE,
                failure_kind=None,
                reasons=reasons,
                task_exit_code=rc,
                result_writer_state=rw_state,
                meta_status=meta_status,
                window=window,
                log_stale=log_stale,
                host_os=host_os,
            )
    elif rc == 0:
        reasons.append("task_exit_code=0 and no error signals")
        return JobStateReport(
            state=STATE_DONE,
            failure_kind=None,
            reasons=reasons,
            task_exit_code=rc,
            result_writer_state=rw_state,
            meta_status=meta_status,
            window=window,
            log_stale=log_stale,
            host_os=host_os,
        )

    # 5) 未开始/刚启动（started）
    if log_stale and rc != SCHTASKS_NOT_RUN:
        reasons.append(f"no completion artifacts and log stale ({log_age_min}min > {stale_threshold_min}min)")
        return JobStateReport(
            state=STATE_INCOMPLETE,
            failure_kind="log_stale",
            reasons=reasons,
            task_exit_code=rc,
            result_writer_state=rw_state,
            meta_status=meta_status,
            window=window,
            log_stale=log_stale,
            host_os=host_os,
        )

    if rc == SCHTASKS_NOT_RUN:
        reasons.append("task_exit_code=267011 (0x41303 schtasks scheduled, not run yet)")
    elif rw_state == STATE_STARTED:
        reasons.append("ResultWriter state=started")
    else:
        reasons.append("task initialized/pending with no completion or error markers")

    return JobStateReport(
        state=STATE_STARTED,
        failure_kind=None,
        reasons=reasons,
        task_exit_code=rc,
        result_writer_state=rw_state,
        meta_status=meta_status,
        window=window,
        log_stale=log_stale,
        host_os=host_os,
    )


# --------------------------------------------------------------------------- #
# 恢复动作建议
# --------------------------------------------------------------------------- #
@dataclass
class RecoveryPlan:
    """幂等续跑与恢复检查清单。"""

    state: str
    action: str
    should_rerun: bool
    idempotent_safe: bool
    pre_checks: list[str]
    steps: list[str]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


def recommend_recovery(
    state: str | Mapping[str, Any] | JobStateReport,
    *,
    failure_kind: str | None = None,
    window: str | None = None,
    host_os: str | None = None,
) -> RecoveryPlan:
    """根据 ``inspect_job_state`` 结果或状态字符串给出幂等续跑检查项与恢复步骤。

    无论何种失败/半完成态，均强制先核 ``raw/meta.json`` 与 ``.pt`` ckpt 完整性，
    再给出最小代价恢复动作（包括 B06 H1 仅补测、H2 删半写 eval.json 后补测）。
    """
    if isinstance(state, JobStateReport):
        st = state.state
        fk = failure_kind or state.failure_kind
        win = window or state.window
        hos = host_os or state.host_os
    elif isinstance(state, Mapping):
        st = str(state.get("state", STATE_UNKNOWN))
        fk = failure_kind or state.get("failure_kind")
        win = window or state.get("window")
        hos = host_os or state.get("host_os")
    else:
        st = str(state)
        fk = failure_kind
        win = window
        hos = host_os

    common_pre_checks = [
        "核查队列健康与半写窗口：python -m src.v25.result_writer runs/v25 --stale-min 30",
        "核查 raw/meta.json 完整性（须恰好 750 个唯一 task_id）；若缺失/损坏而存在 batch_*.pt，按整臂重生成处置。",
        "核查训练 checkpoint (*.pt) 完整性：对使用存在性复用的脚本（B06 R1），确认 .pt 非截断（可用 torch.load(..., map_location='cpu') 校验，损坏则先删除再重启）。",
        "核查累计结果 JSON（B06 R4）：重启时勿仅用 --arms 子集覆盖既有完整 JSON；必要时先备份或跑全量臂列表。",
    ]

    if st == STATE_DONE:
        return RecoveryPlan(
            state=STATE_DONE,
            action="none",
            should_rerun=False,
            idempotent_safe=True,
            pre_checks=[
                "确认 ResultWriter status.json 为 done 且 meta_validated=True。",
            ],
            steps=[
                "任务已完成，禁止重复启动以免浪费 GPU 或覆盖既有结果。",
            ],
        )

    if st in (STATE_STARTED, STATE_RUNNING):
        steps = [
            "保持既有进程运行，禁止重复启动并发副本（agent2.md §0）。",
        ]
        if hos == "windows":
            steps.append("用 schtasks /Query /TN <job_id> /V /FO LIST 与 tasklist 确认 python.exe 存活。")
        elif hos == "linux":
            steps.append("用 tmux has-session -t <job_id> 与 pgrep -af '[x]...' 确认进程存活。")
        if fk == "log_stale_while_alive":
            steps.append("进程存活但日志停更：核查启动脚本是否漏加 python -u（stdout 缓冲假死）。")
        return RecoveryPlan(
            state=st,
            action="monitor",
            should_rerun=False,
            idempotent_safe=True,
            pre_checks=[
                "确认日志 mtime 与 ResultWriter status_log.jsonl 最新时间戳。",
            ],
            steps=steps,
        )

    # failed 或 incomplete：先核 raw meta / ckpt 完整性，再按故障类型精确恢复
    if win == "H1" or fk == "h1_summary_missing":
        return RecoveryPlan(
            state=st,
            action="resume_measure_only",
            should_rerun=True,
            idempotent_safe=True,
            pre_checks=common_pre_checks,
            steps=[
                "B06 H1 窗口命中：raw/meta.json 已完整通过 750 task_id 校验，无需重新生成推理 latent。",
                "只补跑评测阶段（run_measure_subprocess / python -m src.eval.measure --raw ... --out ...），生成 eval.json 与 summary_arm.json。",
                "评测完成后调用 ResultWriter.done(meta=meta) 落盘最终状态。",
            ],
        )

    if win == "H2" or fk == "h2_corrupt_eval_json":
        return RecoveryPlan(
            state=st,
            action="clean_corrupt_and_rerun",
            should_rerun=True,
            idempotent_safe=True,
            pre_checks=common_pre_checks,
            steps=[
                "B06 H2 窗口命中：measured/<arm>/eval.json 半写损坏；若不删除，runner.py 会因存在性跳过并读崩。",
                "先删除损坏的 measured/<arm>/eval.json（保留已验证完整的 raw/ 目录）。",
                "仅重跑评测阶段（src.eval.measure）生成完整 eval.json 与 summary_arm.json。",
            ],
        )

    if fk == "schtasks_path_or_quote_error":
        return RecoveryPlan(
            state=STATE_FAILED,
            action="fix_and_rerun",
            should_rerun=True,
            idempotent_safe=True,
            pre_checks=common_pre_checks,
            steps=[
                "识别 exit=2（schtasks /TR 路径找不到或内联引号转义损坏）。",
                "改用 b08_host_runner 生成的固定 .bat 包装脚本，/TR 只填 .bat 路径，禁止内联 python 参数与引号。",
                "确认 .bat 首部包含 cd /d \"%~dp0\"、python -u、HF_HUB_OFFLINE=1，并用 schtasks /Create ... /F 重新注册后执行 schtasks /Run。",
            ],
        )

    if fk == "hf_network_error":
        return RecoveryPlan(
            state=STATE_FAILED,
            action="fix_and_rerun",
            should_rerun=True,
            idempotent_safe=True,
            pre_checks=common_pre_checks,
            steps=[
                "检测到 httpx / HuggingFace Hub 断连或离线错误。",
                "确认包装脚本已注入 HF_HUB_OFFLINE=1 与 TRANSFORMERS_OFFLINE=1，并核实本地 HF 缓存模型完整。",
                "核实 raw meta / ckpt 无半写残留后，通过 tmux / schtasks 幂等重跑。",
            ],
        )

    if fk == "cuda_oom":
        return RecoveryPlan(
            state=STATE_FAILED,
            action="fix_and_rerun",
            should_rerun=True,
            idempotent_safe=True,
            pre_checks=common_pre_checks,
            steps=[
                "检测到 CUDA/主机 OOM：先清理残留僵尸进程并确认无外部进程抢占显存。",
                "Linux 4090 确认启用 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True；Windows 3060 确认未设置 expandable_segments 且 micro_batch 走 tier_micro_accum 限幅。",
                "清理崩溃时可能截断的 .pt checkpoint 与无 meta.json 的孤立 batch_*.pt，再从安全检查点重启。",
            ],
        )

    if fk in ("orphan_raw", "raw_meta_corrupt", "raw_meta_incomplete_count"):
        return RecoveryPlan(
            state=STATE_INCOMPLETE,
            action="rerun_full_arm",
            should_rerun=True,
            idempotent_safe=True,
            pre_checks=common_pre_checks,
            steps=[
                "raw 产物不完整（缺 meta.json 或 task_id 未过 750 唯一性校验）。",
                "清理损坏的 meta.json 与截断的 .pt ckpt（batch_*.pt 可同名原子覆写）。",
                "确认无存活进程后整臂重跑，并在 finalize 后由 ResultWriter.done(meta=meta) 校验落盘。",
            ],
        )

    if fk == "log_stale":
        return RecoveryPlan(
            state=st,
            action="fix_and_rerun",
            should_rerun=True,
            idempotent_safe=True,
            pre_checks=common_pre_checks,
            steps=[
                "日志长时间停更：先确认原进程确已退出（tmux has-session / tasklist），避免并发重复启动。",
                "确认包装脚本使用 python -u（消除缓冲假死错觉）。",
                "先核 raw meta 与 .pt ckpt 完整性（有有效 raw meta 时走 H1 只补测，否则删半写 ckpt 后重跑）。",
            ],
        )

    # 通用 failed / incomplete
    return RecoveryPlan(
        state=st if st in (STATE_FAILED, STATE_INCOMPLETE) else STATE_INCOMPLETE,
        action="fix_and_rerun",
        should_rerun=True,
        idempotent_safe=True,
        pre_checks=common_pre_checks,
        steps=[
            "读取 err_rel / log_rel 尾部 Traceback 定位根因，并确认原进程已退出。",
            "先核 raw/meta.json 与训练 .pt ckpt 完整性：有效 raw 可直接复用（H1 仅补测），截断 .pt 必须删除。",
            "修复配置/代码缺陷后，按原完整臂列表通过包装脚本（tmux / schtasks）幂等重启。",
        ],
    )


# --------------------------------------------------------------------------- #
# CLI（纯 CPU 只读检查或脚本渲染）
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="B08 cross-platform runner helper (CPU-only)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_render = sub.add_parser("render", help="Render .sh/.bat wrapper script and scheduler commands")
    p_render.add_argument("--job-id", required=True)
    p_render.add_argument("--host-os", required=True, choices=VALID_HOSTS)
    p_render.add_argument("--module", required=True)
    p_render.add_argument("--arg", action="append", default=[], dest="args")
    p_render.add_argument("--workdir-rel", default=".")
    p_render.add_argument("--log-rel", default="")
    p_render.add_argument("--err-rel", default="")
    p_render.add_argument("--gpu-queue", default="")
    p_render.add_argument("--out-dir", default="")

    p_insp = sub.add_parser("inspect", help="Inspect job state and recommend recovery")
    p_insp.add_argument("--log-file", default="")
    p_insp.add_argument("--err-file", default="")
    p_insp.add_argument("--exit-code", default=None)
    p_insp.add_argument("--result-state-dir", default=None)
    p_insp.add_argument("--expected-tasks", type=int, default=750)
    p_insp.add_argument("--stale-min", type=float, default=15.0)
    p_insp.add_argument("--log-age-min", type=float, default=None)
    p_insp.add_argument("--host-os", default=None, choices=VALID_HOSTS)

    ns = ap.parse_args(list(argv) if argv is not None else None)

    if ns.cmd == "render":
        spec = JobSpec(
            job_id=ns.job_id,
            host_os=ns.host_os,
            module=ns.module,
            args=ns.args,
            workdir_rel=ns.workdir_rel,
            log_rel=ns.log_rel,
            err_rel=ns.err_rel,
            gpu_queue=ns.gpu_queue,
        )
        if ns.out_dir:
            written, bundle = write_job_bundle(spec, ns.out_dir)
            payload = {"written_path": str(written), **bundle.to_dict()}
        else:
            payload = render_job_bundle(spec).to_dict()
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    log_txt = Path(ns.log_file).read_text(encoding="utf-8", errors="replace") if ns.log_file and Path(ns.log_file).is_file() else ""
    err_txt = Path(ns.err_file).read_text(encoding="utf-8", errors="replace") if ns.err_file and Path(ns.err_file).is_file() else ""
    report = inspect_job_state(
        log_text=log_txt,
        err_text=err_txt,
        task_exit_code=ns.exit_code,
        result_state_dir=ns.result_state_dir,
        expected_tasks=ns.expected_tasks,
        log_age_min=ns.log_age_min,
        stale_threshold_min=ns.stale_min,
        host_os=ns.host_os,
    )
    recovery = recommend_recovery(report)
    print(json.dumps({"inspection": report.to_dict(), "recovery": recovery.to_dict()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
