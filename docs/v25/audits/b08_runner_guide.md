# B08：Windows / Linux 双平台运行器与失败恢复操作手册

- 任务：B08（[agent2.md](../../19_pull_together/agent2.md) §六 B08 / §九文件所有权），Slot B，2026-11-20
- 交付代码：[tools/v25_cpu/b08_host_runner.py](../../../tools/v25_cpu/b08_host_runner.py)
- 交付单测：[tests/v25/test_b08_runner.py](../../../tests/v25/test_b08_runner.py)
- 关联基座：[src/v25/result_writer.py](../../../src/v25/result_writer.py)、[b06_recovery_report.md](b06_recovery_report.md)

---

## 1. 设计目标与 V25 现场教训

V25 双卡（4090 Linux + 3060 Windows Server）并发实验期间暴露出 5 类高频运维故障，B08 将其固化为“纯文本渲染 + 五状态机判别 + 幂等恢复建议”三层防护，且全工具纯 CPU / 纯 stdlib、全相对路径、不触碰 CUDA 与 ssh：

| 平台 | 调度机制 | 历史踩坑根因 | B08 固化防线 |
|---|---|---|---|
| Linux (4090) | `tmux` + `.sh` | `tmux` 内联复合引号转义失效；裸 `pkill -f` 匹配远程 shell 命令行自杀 | 固定生成 LF 换行 `*_launch.sh`，`tmux` 只执行 `bash <script_rel>`；重复检测/清理统一用 `[x]` 括号正则技巧（`[e]xp.v25...`） |
| Windows (3060) | `.bat` + `schtasks` | SSH 直启随会话退出被回收；`schtasks /TR` 内联引号报 `exit=2`；缺 `-u` 导致 stdout 缓冲假死；缺 `HF_HUB_OFFLINE=1` 触发 `httpx.ConnectError`；误设 `expandable_segments` | 固定生成 CRLF 换行 `*_launch.bat`，`/TR` 只指向 `.bat` 相对路径；强制 `python -u`、`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、`PYTHONUTF8=1`、`OMP_NUM_THREADS=2`；显式禁止 Windows 注入 `expandable_segments` |

---

## 2. `JobSpec` 与双平台包装脚本生成

### 2.1 `JobSpec` 默认契约

- `job_id`：合法标识符 `[A-Za-z0-9_.-]+`（同步用作 tmux session 名或 schtasks `/TN` 任务名）。
- `host_os`：`"linux"` 或 `"windows"`。
- `gpu_queue`：默认 `"4090"`（linux）/ `"3060"`（windows）。
- `workdir_rel` / `log_rel` / `err_rel`：强制相对路径（拒绝任何盘符或根目录前缀）。
- 默认环境变量：
  - 通用：`OMP_NUM_THREADS=2`、`MKL_NUM_THREADS=2`、`OPENBLAS_NUM_THREADS=2`、`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、`PYTHONUNBUFFERED=1`
  - Linux 追加：`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  - Windows 追加：`PYTHONUTF8=1`（显式拒绝 `expandable_segments`）

### 2.2 Linux（4090）启动模板与命令

```python
from tools.v25_cpu.b08_host_runner import JobSpec, write_job_bundle

spec = JobSpec(
    job_id="v25_mech_a",
    host_os="linux",
    module="exp.v25.mechanism_a",
    args=["--arm", "all", "--updates", "1000"],
    log_rel="runs/v25/mech_a.log",
    err_rel="runs/v25/mech_a.err.log",
)
sh_path, bundle = write_job_bundle(spec, "runs/v25/jobs")
```

生成产物要点：
1. `*_launch.sh` 首部自动从脚本位置向上寻找含 `paths.py` 的仓库根并 `cd` 进入，再切到 `workdir_rel`；
2. 内置 `pgrep -f '[e]xp.v25.mechanism_a'` 防重复启动守卫（`[x]` 括号正则绝不自匹配）；
3. 调度命令完全免内联引号：
   - 授权：`chmod +x runs/v25/jobs/v25_mech_a_launch.sh`
   - 启动：`tmux new-session -d -s v25_mech_a 'bash runs/v25/jobs/v25_mech_a_launch.sh'`
   - 巡检：`tmux has-session -t v25_mech_a && tmux capture-pane -pt v25_mech_a | tail -n 20`
   - 安全终止（需手工停任务时）：`pkill -f '[e]xp.v25.mechanism_a'`

### 2.3 Windows（3060）启动模板与命令

```bash
python -m tools.v25_cpu.b08_host_runner render \
  --job-id v25r3 \
  --host-os windows \
  --module exp.v25.r3_oneshot_minimal \
  --arg --pool --arg ../pseudo_gt_24h/orig_mean_fp16 \
  --log-rel runs/v25/v25_r3.log \
  --err-rel runs/v25/v25_r3.log \
  --out-dir runs/v25/jobs
```

生成产物要点：
1. `*_launch.bat` 首部固定 `cd /d "%~dp0"` 并向上定位 `paths.py` 仓库根，所有路径自动转为反斜杠 `\`；
2. 环境变量使用安全 `set "KEY=VALUE"` 语法，调用 `python -u -m <module> ...`；
3. 调度命令无嵌套双引号：
   - 注册：`schtasks /Create /TN v25r3 /TR runs\v25\jobs\v25r3_launch.bat /SC ONCE /ST 00:00 /F`
   - 启动：`schtasks /Run /TN v25r3`
   - 查询：`schtasks /Query /TN v25r3 /V /FO LIST`

---

## 3. 五状态机与联合判别规则

`inspect_job_state(log_text, err_text, task_exit_code, result_state_dir)` 将 [ResultWriter](../../../src/v25/result_writer.py) 的磁盘持久化状态、Windows/Linux 退出码、日志/stderr 异常特征与 B06 H1/H2 半完成窗口统一收口到五态：

| 状态 | 触发判据（优先级从高到低） | 典型 `failure_kind` / `window` |
|---|---|---|
| `failed` | `task_exit_code == 2`（schtasks `/TR` 引号或 `.bat` 路径缺失）；`267014`（被终止）；非零退出码（含 `0xC0000409` 等）；日志/stderr 命中 OOM、`httpx`/`LocalEntryNotFoundError`、启动缺失、NaN、`Traceback`；`ResultWriter == failed` | `schtasks_path_or_quote_error`、`hf_network_error`、`cuda_oom`、`python_traceback`、`nonzero_exit` |
| `running` | `task_exit_code == 267009`（`0x41301`）或 `process_alive=True` 且无致命报错；或 `ResultWriter == running` 且未停更 | 若进程存活但日志停更，附 `log_stale_while_alive` 提醒检查 `python -u` |
| `incomplete` | `B06 H2`（`measured/<arm>/eval.json` 半写损坏）；`B06 H1`（`raw/meta.json` 完整但 `measured/<arm>` 缺 `summary*.json`）；`meta.json` 损坏或不足 `expected_tasks`（即使 status 伪写 `done` 或 `exit=0`）；孤立 `batch_*.pt`；`ResultWriter == incomplete`；`started/running` 但日志停更或进程已死；`KeyboardInterrupt` | `h1_summary_missing`、`h2_corrupt_eval_json`、`raw_meta_corrupt`、`raw_meta_incomplete_count`、`orphan_raw`、`log_stale` |
| `done` | 无上述错误/半完成信号，且 `meta.json` 通过 750 唯一 `task_id` 完整性闸（或无 raw 任务的 `ResultWriter == done` + `exit=0`） | `None` |
| `started` | `task_exit_code == 267011`（`0x41303` 已注册未运行）、`ResultWriter == started` 且未停更、或初始无报错/无产出态 | `None` |

### Windows `schtasks Last Result` 速查

| Last Result | 十六进制 | 含义 | B08 判定 |
|---|---|---|---|
| `0` | `0x0` | 正常退出 | 仍须结合日志与 `meta.json` 验证才可记 `done` |
| `2` | `0x2` | 文件/路径找不到（常见于 `/TR` 内联引号损坏） | `failed` (`schtasks_path_or_quote_error`) |
| `267009` | `0x41301` | 正在运行 (`SCHED_S_TASK_RUNNING`) | `running` |
| `267011` | `0x41303` | 尚未运行 (`SCHED_S_TASK_HAS_NOT_RUN`) | `started` |
| `267014` | `0x41306` | 被强行终止 (`SCHED_S_TASK_TERMINATED`) | `failed` (`process_terminated`) |
| `3221226505` / `3221225477` | `0xC0000409` / `0xC0000005` | 栈缓冲溢出 / 访问违例崩溃 | `failed` |

---

## 4. 失败与半完成恢复 SOP（`recommend_recovery`）

总控巡检与恢复必须严格遵循**先核完整、再做动作、禁止并发重复启动**三原则：

1. **先跑只读状态扫描**（纯 CPU，不碰 GPU）：
   ```bash
   python -m src.v25.result_writer runs/v25 --stale-min 30
   python -m tools.v25_cpu.b08_host_runner inspect \
     --log-file runs/v25/v25_r3.log \
     --exit-code 2 \
     --result-state-dir runs/v25/raw/<arm> \
     --host-os windows
   ```
2. **核查两类底层产物完整性（强制前置检查）**：
   - **`raw/meta.json`**：须通过 `validate_raw_meta_dict`（恰好 750 个唯一 `task_id`）。若有 `batch_*.pt` 而无有效 `meta.json`（`orphan_raw`），不可跳过生成，须整臂重跑（分片同名原子覆写）。
   - **训练 `.pt` checkpoint（B06 R1）**：使用 `if out_p.exists(): reuse` 的脚本若在 `torch.save` 中途被杀，会留下截断 `.pt`。重跑前用 CPU 侧 `torch.load(p, map_location="cpu")` 或文件校验核实；截断文件必须先删除再重启。
   - **累计结果 JSON（B06 R4）**：重启带 `--arms` 子集会覆盖整份 JSON；重启前确认备份或使用完整臂列表。
3. **按 `recommend_recovery(state)` 精确恢复**：
   - **`done` / `running` / `started`**：`should_rerun=False`，禁止重复启动；`running` 且日志停更时优先排查脚本是否遗漏 `python -u`。
   - **B06 `H1`（`raw/meta.json` 完整，缺 `summary`）**：`action="resume_measure_only"`——保留已验证的 `raw/<arm>`，只补跑 `src.eval.measure`（省去推理重新生成）。
   - **B06 `H2`（`measured/<arm>/eval.json` 半写损坏）**：`action="clean_corrupt_and_rerun"`——必须先删除损坏的 `eval.json`（否则 `runner.py` 会因文件存在而跳过评测并崩溃），再仅重跑 `src.eval.measure`。
   - **`exit=2`（`schtasks_path_or_quote_error`）**：改用 B08 `.bat` 包装脚本，`schtasks /TR` 只填 `.bat` 相对路径后 `/Create ... /F` + `/Run`。
   - **`hf_network_error`**：补齐 `HF_HUB_OFFLINE=1` 与 `TRANSFORMERS_OFFLINE=1` 并核实本地模型缓存后重启。
   - **`cuda_oom`**：清理僵尸进程与截断 `.pt`，确认 Linux 开 `expandable_segments:True`、Windows 3060 禁用它且启用 `tier_micro_accum` 限幅后再重跑。
