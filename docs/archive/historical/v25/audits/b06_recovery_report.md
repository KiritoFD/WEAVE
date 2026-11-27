# B06：结果 Writer、任务状态与恢复检查 —— 审计报告

- 任务：B06（docs/19_pull_together/agent2.md §五 / §九），Slot B，2026-11-20
- 交付：`src/v25/result_writer.py`（`atomic_save_json` + `ResultWriter` + `scan_status` 巡检）、`tests/v25/test_result_writer.py`（29 测试）、本报告
- 边界遵守：只创建/修改以上三个文件；无 git 命令、无 CUDA/ssh、无绝对路径；`measure.py:109` 只给修复提案不动代码；`metrics.csv` 由总控管理不在范围。

---

## 1. 逐项审计结论（验收 1–5，全部给 文件:行号 证据）

### 1.1 任务执行中断后能否续跑 —— 幂等复用模式检查清单

| # | 模式 | 位置 | 安全性 |
|---|---|---|---|
| R1 | checkpoint 存在性复用 `if out_p.exists(): reuse` | `exp/v25/r2_train_ablation.py:60-62` | **有漏洞**：ckpt 由非原子 `torch.save(..., out_p)` 写入（`r2_train_ablation.py:163`）。kill 发生在 save 中途 → 截断 `.pt` 存在 → 重启被"复用" → 后续 `load_state_dict` 崩溃（响亮失败，不静默，但必须手工删文件；无状态标记区分完整/截断）。同型：`exp/v24/p0_o1_o3_value.py:187`（复用）+ `:308-309`（非原子保存） |
| R2 | raw-meta 复用 `if meta.json exists and validate_raw_meta(...)` | `exp/v25/r1_inference_ablation.py:78-79`、`exp/v24/p0_o1_o3_value.py:367-368`、`exp/v25/r3_oneshot_minimal.py:107-108` | **raw 生成安全**：无 meta.json（生成中途 kill）→ validate 失败 → 整臂重生成，分片被同名覆写（`src/utils/raw_io.py:60-70` 逐分片 tmp+replace，原子）。**但见 1.2 的半完成窗口 H1/H2** |
| R3 | eval.json 存在性跳过 measure | `src/eval/runner.py:62` `if force or not (meas_p/"eval.json").exists()` | **存在性即 done 代理**：eval.json 存在（哪怕半写）→ 跳过 measure → `summarize_measured_arm`（`runner.py:13-16`）读半写 JSON → 崩溃。eval.json 写入非原子（见 §1.4） |
| R4 | accumulating results JSON：进程内 `results=[]` 起步，逐臂 `save_json(out_json, {..., "results": results})` | `exp/v25/r2_train_ablation.py:187-198`、`exp/v25/mechanism_a.py:656-682`、`exp/v25/r1_inference_ablation.py:127-135`、`exp/v24/p0_o1_o3_value.py:439-444`、`exp/v25/r3_oneshot_minimal.py:347-384` | 双面：kill 后重启且臂列表完整时，复用 R1/R2 使结果确定性重建（收敛）；但**无 merge-on-start** —— 带 `--arms` 子集重启会整体覆盖 JSON，永久丢掉先前记录（见 §1.3） |
| R5 | `mechanism_a` 训练无 ckpt 复用 | `exp/v25/mechanism_a.py:658-663`（无条件 `train_gate_arm`）、`:479`（非原子 `torch.save` ckpt） | **重启即全量重训**：`--arm all` kill 后重启，即使 `gate_A2_learned_g55.pt` 在盘也重训 A2/A3（~2h 浪费）；λ_c 标定（`:497-529`，300 updates × 3 候选）同样每次重跑（seed 固定可复现，纯浪费） |

**回答任务卡具体问题："mechanism_a 训练中 kill 后 ckpt 缺失但 JSON 已写？"** —— **不会**。`mechanism_a.json` 只在 `eval_mechanism_a_arm` 完整返回后才 `save_json`（`mechanism_a.py:673-682`），train 阶段 kill 时 JSON 无该臂记录、ckpt 也未写（`:479` 在训练循环结束后）；JSON 与 ckpt 不存在"半对半错"组合。真正的坑是 R1 型（r2/v24 存在性复用遇截断 ckpt）与 R5 型（重启重训浪费）。

### 1.2 是否重复评测已完成任务 —— 半完成窗口

复用条件链：raw meta 完整 ⇒ **直接读 summary**，没有任何"measured 存在性/有效性"中间检查：

- **H1（高概率，必现于"meta 已写、measure 被 kill"）**：`r1_inference_ablation.py:79` / `p0_o1_o3_value.py:368` / `r3_oneshot_minimal.py:108` 在 meta validate 通过后直接 `read_json(measured/<arm>/summary_vXX.json)`，summary 不存在 → `FileNotFoundError` 崩溃。raw 生成被正确复用（不重复算），但 measure 步骤**不会自动补跑**，需人工介入。触发概率不低：raw finalize（`raw_io.py:72-87`）与 measure 完成之间隔着整个 750 任务特征提取（~50s/臂）。
- **H2（低概率但后果混淆）**：`src/eval/measure.py:109-112` 非原子 `write_text` 写 eval.json（~0.9MB，B04 实测 write 落盘 ≈5.3ms，窗口即毫秒级）——kill 落在窗口内 → 半写 eval.json 存在 → 重启后 `runner.py:62` 判"已测"跳过 → `runner.py:13` 读崩 `JSONDecodeError`；且必须手工删该文件才能恢复。
- **H3（自动恢复，无需人工）**：eval.json + summary_arm.json 已写（`runner.py:74-75`）但实验侧 `summary_v25.json` 未写（`exp/v25/__init__.py:42-44`）→ 重启跳过 measure、重跑廉价 summarize → 自愈。这条链设计正确。

结论：**不会重复评测**（复用闸有效），但**会以崩溃方式中断在"raw 完整、measured 缺失"的半完成态**，缺一个"补跑 measure"的自动降级路径；且全栈没有任何持久化 failed/incomplete 状态来区分"从未跑 / 跑了一半 / 跑完崩溃"。

### 1.3 是否会覆盖成功结果

- **确实存在一处真实覆盖路径**：R4 型 accumulating JSON 无 merge-on-start。实例：Round 2 完成后（JSON 含 8 条记录），用 `python -m exp.v25.r2_train_ablation --arms Tr_T0` 单臂重启 → `results=[]` → 首臂评完即 `save_json` 整文件覆盖 → Tr_T1–T3 的 6 条记录**永久丢失**（重启同臂列表则无损收敛）。同型 `mechanism_a.py:680-682`、`p0_o1_o3_value.py:444`。
- `save_json` 本身（`src/utils/runtime.py:28-36`）是原子 tmp+replace，但语义是"最后写者赢"，无历史。
- `run_measure_subprocess(force=True)`（`runner.py:62`）会用新评测**覆盖**旧 eval.json 且写入非原子 —— kill 于窗口内 = 好的旧结果也没了（force 路径的"覆盖成功结果"实例）。
- 安全面：raw 分片/meta 逐文件原子（`raw_io.py:66-68`，meta 经 `save_json`）；`finish()` 先抛 `expected_tasks`/重复 id 异常（`raw_io.py:73-77`）再写 meta，不会用坏 meta 覆盖好 meta；PNG 导出按 task_id 幂等覆写。

### 1.4 JSON 是否原子写入 —— 写点清单（`atomic_save_json` 已交付 `src/v25/result_writer.py`）

| 写点 | 位置 | 现状 | 处置 |
|---|---|---|---|
| 实验结果/状态 JSON（r1/r2/r3/mechA/p0 等全部） | `runtime.save_json` | **已原子**（tmp+replace） | 不动 |
| RawWriter batch 分片 + meta.json | `raw_io.py:66-68`、`:86` | **已原子** | 不动 |
| summary_arm.json / summary_v25.json / summary_v24.json | `runner.py:75`、`exp/v25/__init__.py:44`、`exp/v22/__init__.py:55` | **已原子** | 不动 |
| **eval.json** | `src/eval/measure.py:109-112` | **非原子** `write_text`（B04 确认；~0.9MB） | **提案 A（本任务不动代码）**：改为 `atomic_save_json(..., indent=None)`（compact 口径；B04 §8.3 实测 indent=2 使 0.9MB 体积翻倍、dumps 3.9x CPU，compact 绝对开销 <0.05s/臂，占 ~50s 墙钟 <0.1%）。注意 `src.utils.runtime.save_json` 固定 indent=2，直接替换会体积翻倍 —— 需 compact 参数版，即本次交付的 `atomic_save_json` |
| features.pt | `measure.py:113-115` | 非原子 `torch.save` 直写（~5.9MB） | **提案 B**：tmp+replace 包装（torch 层，约 +1ms/臂）；优先级低于 A |
| 训练 ckpt（r2 / mechanism_a / v24 O 臂） | `r2_train_ablation.py:163`、`mechanism_a.py:479`、`p0_o1_o3_value.py:308-309`、`:152` | 非原子 `torch.save` 直写（~30MB） | **提案 C**：tmp+replace（+~0.05s/ckpts，一次性）；配合 R1 的存在性复用才是完整修复。属 B03 ArtifactRegistry 血缘闸的补强方向 |
| trainercache v16 | `p0_o1_o3_value.py:135,152` | 非原子 + 存在性复用 | 并入提案 C |

### 1.5 task_id 是否完整 / 失败仍记 done —— 结论：raw 层安全，measured 层的 done 代理有洞

- **raw 层安全（证据充分）**：`RawWriter.finish`（`raw_io.py:72-87`）在写 meta 之前先做两道闸 —— 任务数不符即 raise（`:73-74`）、重复 task_id 即 raise（`:76-77`）；`save_json` 在两闸之后（`:86`）。磁盘上任何能通过 `validate_raw_meta`（`exp/v22/__init__.py:87-93`：恰好 750 个唯一 id）的 meta 都对应完整批次。**部分批次不可能被记为完成**。
- **残差风险（低）**：`append()` 不校验 id 是否属于 tasks750、也不查跨批重复（只在 finish 统一查）——id 集合错了会在 measure 侧 `measure.py:63` `by_id[i]` KeyError 处响亮失败，不产生静默错数据。
- **"失败仍记 done"的真实缺口在 measured/队列层**：全栈唯一的状态语义是文件存在性（eval.json、summary_*.json、ckpt），半写文件与完成产物不可区分（H2/H3），且**没有任何 failed/incomplete 标记被持久化** —— schtasks 队列 kill 后，总控只能靠日志推断。本次交付的 `ResultWriter`（started/running/done/failed/incomplete + 完整性闸：`done(meta=...)` 未过 750 唯一 id 闸自动降级 incomplete，损坏 meta.json 同样拒绝 done）即为堵这个缺口。

---

## 2. 续跑安全矩阵（kill 时点 × 产物状态 × 恢复动作）

| kill 时点 | raw/ | meta.json | measured/ | 恢复动作 | 自动? |
|---|---|---|---|---|---|
| batch 生成中 | 有分片无 meta | 缺 | 缺 | 整臂重生成（分片同名覆写） | 是（浪费已算分片） |
| RawWriter.finish 后、measure 前 | 完整 | **valid** | 缺 | **H1：崩溃**，需手工补跑 `run_measure_v25(arm)` | 否 |
| measure 子进程中 | 完整 | valid | eval.json 缺 | 同 H1（eval.json 只在末尾写） | 否 |
| eval.json 写入窗口（ms 级） | 完整 | valid | **eval.json 半写** | **H2：崩溃**，需删 eval.json 后补跑 measure | 否 |
| runner summarize 后、summary_vXX 写前 | 完整 | valid | eval+summary_arm 有 | H3：重启跳过 measure，重写 summary | 是 |
| 训练循环中（r2/mekanismA/v24O） | — | — | — | 重训该臂（r2/v24O 存在性复用遇截断 ckpt 会崩，删文件恢复；mechA 无复用全量重训） | 部分 |
| `torch.save(ckpt)` 窗口 | — | — | — | 截断 ckpt + R1 存在性复用 → 下游 load 崩；手工删除 | 否 |
| 结果 JSON `save_json` 窗口 | — | — | — | save_json 原子，无影响 | 是 |
| r3 阶段间（pack 分阶段落盘 `:353/359/372/384`） | — | — | — | pack 原子覆写，重启按 stage 重算（R2 复用生效） | 是 |

`scan_status`（新交付）把这张矩阵编译成分诊 flag：`orphan_raw` / `summary_missing`（=H1）/ `raw_meta_incomplete_count` / `raw_meta_corrupt` / `done_but_meta_bad` / `stale`。

---

## 3. Windows schtasks（3060 队列）失败识别手册

经验锚点（`docs/19_pull_together/PROGRESS.md:15,59`、`docs/v25/GPU3060_RESULTS.md`）：SSH 会话直启会被回收；schtasks 内联 `/TR` 引号转义损坏（**exit=2**）；改 bat 文件 + schtasks 成功；python 需 `-u` 防 stdout 缓冲假死；httpx 网络瞬断需 `HF_HUB_OFFLINE=1`；代码运行时 bug（`sg25_and_swap` 提前 del）发生在队列深处。

### 检查项（按成本从低到高）

| # | 检查 | 命令 / 判据 | 阳性含义 |
|---|---|---|---|
| 1 | schtasks Last Result | `schtasks /query /tn v25r3 /v /fo LIST` → `Last Result: 0`=正常退出；`2`=bat/路径未找到（内联 /TR 转义损坏型）；`267009`=正在运行(0x41301)；`267011`=未运行过(0x41303)；大码 `0xC0000409`/`0xC0000005`=python 运行时崩溃 | 任务级死亡原因 |
| 2 | python 进程存活 | `tasklist /fi "imagename eq python.exe" /v`（或 `wmic process where "commandline like '%r3_oneshot%'" get processid,creationdate`） | 进程消失而 Last Result≠0 → 任务死于运行期 |
| 3 | 日志停更 | `runs/v25_r3.log` mtime：`powershell -c "(Get-Item runs\v25_r3.log).LastWriteTime"`；>10–15 min 未动且进程不在 → 死亡；mtime 停但进程在 → 忘 `-u` 缓冲假死或 CPU 死循环 | 死亡时刻定位（对照日志尾） |
| 4 | 结果 JSON `updated_utc` 停更 | `runs/v25/r3_oneshot_minimal.json` 的 `updated_utc` 每 stage 刷新（`r3_oneshot_minimal.py:353-384`）；stage 边界停更 + 检查项 1/2 阳性 = 该 stage 内崩溃 | 崩溃到哪个 stage |
| 5 | 日志尾模式识别 | bat 内 `python -u -m ... 2>&1` 重定向单文件；尾行 `Traceback` → 代码 bug（httpx 网络瞬断型：`httpx.ConnectError` → 补 `HF_HUB_OFFLINE=1`）；`CUDA out of memory` → 3060 12GB 微批；无 traceback 但戛然而止 → 进程被杀（OOM/回收/断电） | 崩溃类型 |
| 6 | 半写产物分诊 | `python -m src.v25.result_writer runs/v25`（本任务新交付）→ flags：`summary_missing`=只需补 measure；`orphan_raw`=整臂重生成；`raw_meta_incomplete_count/corrupt`=raw 坏；`done_but_meta_bad`=历史脏状态 | 恢复动作选择 |

### 推荐总控巡检 one-liner

本机（3060，bat 同目录执行）：

```bat
schtasks /query /tn v25r3 /v /fo LIST | findstr /C:"Last Result" /C:"Last Run Time" /C:"Status" & powershell -c "(Get-Item runs\v25_r3.log).LastWriteTime; (Get-Item runs\v25\r3_oneshot_minimal.json).LastWriteTime" & tasklist /fi "imagename eq python.exe" /v | findstr /i python
```

仓内（总控在任何机器、纯 CPU、只读）：

```bash
python -m src.v25.result_writer runs/v25 --stale-min 30        # 表格 + flags
python -m src.v25.result_writer runs/v25 --stale-min 30 --json # 机器可读
```

恢复顺序建议：先 `--json` 看 flags（决定 raw 是否要重生成 / 只补 measure）→ 再查 schtasks Last Result 与日志尾（决定是否改 bat/补 HF_HUB_OFFLINE）→ 手工恢复后让队列继续，不重复启动已成功臂（R2 复用闸保证幂等）。

### bat 启动模板（吸收全部教训）

```bat
@echo off
cd /d %~dp0
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
python -u -m exp.v25.r3_oneshot_minimal --pool ..\pseudo_gt_24h\orig_mean_fp16 >> runs\v25_r3.log 2>&1
```

（schtasks `/TR` 只指 bat 文件路径，绝不内联引号参数 —— exit=2 教训；`-u` 防缓冲假死；单文件重定向便于 tail。）

---

## 4. 交付物：`ResultWriter` 薄类与新脚本接入方式

- `atomic_save_json(obj, path, indent=None, fsync=False)`：pid 唯一 tmp + `os.replace`，Windows 兼容（目标已存在时原子覆盖；同目录 tmp 保证同卷）；序列化约定与 `runtime.save_json` 一致（`allow_nan=False`）。崩溃语义经单测钉死：replace 失败不伤原文件、残留 tmp 被下次写自然清理。
- `ResultWriter(arm_dir, expected_tasks=750)`：`start()/running(progress)/done(meta)/failed(reason)/incomplete(reason)` 五状态；`status.json`（原子缓存）+ `status_log.jsonl`（fsync append-only 权威史，写序=先史后缓存）；status.json 半写损坏 → JSONL 回放恢复；JSONL 撕裂尾行宽容跳过。`done(meta)` 未过完整性闸/遇损坏 meta.json 一律降级 `incomplete`（结构性堵死"失败仍记 done"）。
- `scan_status(root)` + `python -m src.v25.result_writer`：半写窗口分诊（§2 矩阵 → flags），总控巡检 one-liner。
- **兼容性**：不替换 `RawWriter` / `runtime.save_json` / `validate_raw_meta`；新脚本可选接入（`rw = ResultWriter(OUT/"raw"/arm)`，RawWriter 照旧 finalize 后 `rw.done(meta=meta)`）；现有评测格式零改动。

---

## 5. 测试

```
python -m pytest tests/v25/test_result_writer.py -q   → 29 passed (0.38s)
python -m pytest tests/v25 -q                          → 214 passed (26.0s, 无回归)
python -m pytest tests/test_no_absolute_paths.py tests/test_markdown_links.py -q → 4 passed（护栏）
python -m src.v25.result_writer runs/v25 --stale-min 120 → 本机镜像 smoke：0 arms（raw/ 在 4090，本地镜像仅 measured/，配对布局检测正确处理空目录）
```

测试覆盖映射验收项：①中断模拟（os.replace 注入失败）→ `test_crash_midwrite_keeps_original`；②半写窗口 → `test_torn_tmp_file_does_not_affect_original`、`test_h1_window_meta_valid_summary_missing`；③损坏 JSON 容错 → `test_read_missing_and_corrupt_returns_none`、`test_corrupt_status_json_falls_back_to_log_replay`、`test_torn_jsonl_tail_still_recovers_state`；④状态机 JSONL → `test_started_to_done_jsonl`/`test_started_to_failed_jsonl`；⑤完整性闸 → `test_done_refuses_incomplete_meta`、`test_done_refuses_duplicate_ids`、`test_done_auto_loads_meta_from_dir`（含损坏 meta 拒绝 done）；⑥Windows 路径 → `test_windows_path_compat`；⑦巡检 → `TestScanStatus`（6 测试）+ CLI。

## 6. RISKS / NEXT

- RISKS：(a) `atomic_save_json` 在 Windows 上若读方长期独占句柄仍可能 PermissionError（读方应开-读-关；`read_json_tolerant` 即如此）；(b) `scan_status` 对平铺布局无法核 measured 对照（不误报，flag 留空）；(c) 审计发现的 H1/H2/R1/R4 修复提案均未动现有代码（所有权边界），现有脚本行为不变。
- NEXT_TASK_SUGGESTION：(1) **提案 A 落地**（`measure.py:109` eval.json 原子化，compact 口径）由总控统一集成（我无该文件所有权）；(2) 提案 C（ckpt tmp+replace）与 R4 的 merge-on-start（重启先读旧 results 按臂名合并）建议作为 B08"Windows/Linux 运行器整理"的一部分；(3) 新实验脚本（机制 B/C 的 R2–R7 队列）从第一个新臂开始接 `ResultWriter`，3060 队列即可用 §3 手册 + one-liner 巡检。
