# WEAVE V25：双子 Agent 并发下的 12 小时执行方案

## 0. 核心原则

本方案替代之前的 20+ 子 Agent 并行方案。

**实际资源：**

- 4090：一条独占 GPU 实验队列，Linux + tmux。
- 3060：一条独占 GPU 实验队列，Windows + schtasks。
- Coding Agent：最多同时运行 **2 个子 Agent**。
- 总控 Agent：负责调度、审查、集成和 Git 操作。
- 所有 Agent 在各自机器的现有仓库目录工作，**禁止创建 worktree，禁止子 Agent 切换分支**。
- 当前已运行的任务优先保持，不允许为了新计划重复启动或中断。

**核心策略：GPU 任务进程化，Agent 任务短周期化。**

不要让子 Agent 等待实验结束，也不要让一个子 Agent 长期占用名额负责监控。

实验一旦通过 tmux 或 schtasks 成功启动并确认能够继续运行，负责启动的子 Agent 就应结束任务、释放名额。

后续状态由总控通过日志和进程检查获取。

---

# 一、最终组织结构

```text
                    MAIN ORCHESTRATOR
                     总控（持续工作）
                           │
             ┌─────────────┴─────────────┐
             │                           │
       SUBAGENT A                  SUBAGENT B
       科学/测量优先                代码/工程优先
             │                           │
       每次领取一个任务              每次领取一个任务
       完成后立即交付                完成后立即交付
             │                           │
             └─────────────┬─────────────┘
                           │
                     总控统一集成
                           │
               ┌───────────┴───────────┐
               │                       │
          4090 GPU QUEUE          3060 GPU QUEUE
          tmux 持续运行           schtasks 持续运行
          不占 Agent 名额         不占 Agent 名额
```

两个子 Agent 不固定为具体人员，只有两条工作槽位。

### Slot A：Research / Measurement

优先承担：

- 数学与统计审查；
- CPU 诊断；
- 数据分析；
- 实验归因；
- 新结果分析；
- 科学文档。

### Slot B：Engineering / Implementation

优先承担：

- 机制代码开发；
- 单元测试；
- 代码重构；
- 缓存管理；
- 跨平台运行；
- 推理接口；
- 工程文档。

**允许动态调整。** 如果 4090 的机制代码尚未就绪，可以临时让 A、B 分别开发主线和 One-Shot；开发完成后再恢复科学/工程分工。

总控不得让某个子 Agent 长期处于 `sleep`、`wait` 或持续轮询状态。

---

# 二、立即执行：先让两张 GPU 独立运转

## G0-A：4090 队列

检查：

- `tmux v25r2q` 是否仍存在；
- Round 2 是否已完成；
- 已生成哪些 checkpoint；
- 是否有失败、OOM、NaN；
- 结果是否已经写入独立 JSON；
- 当前脚本和 checkpoint 的 hash。

**已有任务不得重复运行。**

在 Round 2 完成后，按依赖顺序执行：

| 顺序 | 工作 | 前置条件 |
|---|---|---|
| G1 | Round 2 收敛统计与配方冻结 | Round 2 完成 |
| G2 | 机制 A1：启发式空间门控 | 已有 V24 latent 缓存 |
| G3 | 机制 A2/A3：学习空间门控 | 新机制代码通过 CPU 测试 |
| G4 | 机制 B0：PRISM-1 热启动蒸馏 | 教师配置确定 |
| G5 | 机制 B1：更宽学生 | B0 基线完成 |
| G6 | 最优候选独立种子复验 | 初步比较完成 |
| G7 | 端到端性能与正式 test750 | 最优候选确定 |

G2–G5 不能因为前一机制判负就全部停止。

### 必须修改的实验设计

**1. D6 与 Hybrid 不可过早合并。**

D6 当前实测只覆盖到 DINO-C≈0.70，0.73 处的表现不能用外推直接定论。

**2. 教师定义固定。**

训练侧和评测侧必须使用相同的 checkpoint、anchor、CFG 与 solver。

**3. 机制 B 的 γ=0 精确回退。**

通过冻结基座＋显式 γ 残差实现，而不是仅依赖初始化。

**4. 不为了节省算力降低正式评测规模。**

所有用于质量结论的候选均完成 test750。

---

## G0-B：3060 队列

检查：

- `schtasks v25r3` 的实际运行状态；
- Round 3 的 O3_self_only；
- 已完成的 τ 扫描；
- O3_label_free 的 OOD 和 swap 测试。

然后按顺序执行：

| 顺序 | 工作 | 前置条件 |
|---|---|---|
| R1 | Round 3 收敛 | 已有任务 |
| R2 | 教师内容权重校准 | 教师脚本修订 |
| R3 | 2000 对参考条件伪目标生成 | 教师质量诊断通过 |
| R4 | O3 新教师训练 | 伪目标完成 |
| R5 | 参考替换一致性训练 | Loss 符号和内容约束单测通过 |
| R6 | 最佳模型复验 | 初步指标完成 |
| R7 | SG-25 OOD、多参考及端到端性能 | 最终候选 |

3060 的原始 O3 可以作为保底模型，不因新实验失败而丢失。

---

# 三、两个子 Agent 的任务队列

不再同时派发 20 个任务，而是建立一个**有优先级的任务池**。

每次最多领取两个任务：

- Slot A 领取一个；
- Slot B 领取一个。

完成后立即领取下一个。总控可以随 GPU 新结果改变优先级，但不得随意取消已经产生有效工作的任务。

## P0：第一批必须立即完成

| ID | 槽位 | 任务 | 交付 |
|---|---|---|---|
| A01 | A | 配对 Bootstrap 与非劣性统计工具 | Python 工具＋单测＋Round 1 CI |
| B01 | B | 机制 A/B 的数学与代码正确性测试 | CPU mock 测试＋待运行代码 |
| A02 | A | T4/PRISM-1/D6 教师与锚点一致性诊断 | 配置血缘表＋差异报告 |
| B02 | B | 机制 C 教师与参考替换 Loss 修正 | 实现＋CPU 单测 |

**这一批最重要。**

它们直接影响 GPU 机制实验是否有效，不能放到最后才做。

### A01 验收

以 `src/eval/metrics.py` 为唯一统计基础。

必须验证：

- 600 个 cross 任务完全配对；
- 150 个 source cluster；
- Bootstrap 使用相同重采样索引；
- LPIPS 指标方向正确；
- 非劣性与显著性不是同一个概念；
- 缺失任务或重复任务立即报错；
- 无法凭均值外推构造假的配对样本。

输出 Round 1 的：

`cs_j vs full`

`fixed vs original gate`

`PRISM-1 anchor vs T4`

`D6 vs Hybrid`

前三项可根据真实配对结果判断；D6 在尚未实测的 DINO-C=0.73 处必须标记为未决。

### B01 验收

重点测试：

- 空间门控初始化与全局 γ 混合等价；
- clamp 后门控归一化仍有效；
- 相同总体编辑能量下的门控比较；
- γ=0 精确恢复 PRISM-1；
- 教师训练和评测配置一致；
- 无需真实 GPU 即可测试。

### A02 验收

从现有运行配置和 checkpoint 元数据建立：

`model_sha → anchor_sha → solver_hash → output_hash`

比较训练侧与评测侧，发现不一致立即报告。

### B02 验收

验证：

- 风格相似度的最小化方向正确；
- 两个参考输出分别受到内容保持约束；
- 使用错误参考时不会错误获得更高的正确参考得分；
- 参考图不依赖目标画派 ID；
- 同一参考特征缓存结果稳定。

---

# 四、第二批：CPU 诊断与测量

第一批任务结束后，派发以下四个任务。

| ID | 槽位 | 任务 | 交付 |
|---|---|---|---|
| A03 | A | O2 Teacher 质量与筛选偏差诊断 | 统计表＋分布图＋改进意见 |
| B03 | B | 统一 checkpoint/cache 血缘检查器 | `src/v25/artifact_registry.py`＋测试 |
| A04 | A | 模型强度旋钮与 Pareto 曲线分析 | 单调性、饱和区、真实前沿图 |
| B04 | B | CPU/I/O 测量工具 | Benchmark 程序＋结果模板 |

## A03：重点科学诊断

分析现有 O2 Teacher：

- `mid_cos` 分布；
- latent 编辑幅度；
- 颜色变化量；
- 参考风格变化量；
- 被 Top-K 筛掉的样本；
- 筛选后剩余样本是否过度接近原图；
- 不同风格的保留比例。

目标是解释 O3_label_free 的保守行为，而不是仅仅展示几张样例。

尽可能从已有缓存读取，不重新执行教师优化。

## B03：缓存血缘

检查现有仓库大量：

```python
if path.exists():
    return torch.load(path)
```

是否可能在训练配置改变后静默复用旧缓存。

新增统一缓存元数据：

- 输入 manifest hash；
- checkpoint hash；
- 配置 hash；
- 数据预处理版本；
- decoder 版本；
- 输出形状；
- 创建时间；
- Git SHA。

本轮先新增兼容包装层，不直接重写所有历史模块。

## A04：旋钮分析

使用已有 V19–V25 数据，不运行 GPU。

比较：

- PRISM-1 的 λ；
- D6 的 τ；
- Hybrid 的 γ；
- O3 的 τ。

输出不同引擎的：

- DINO-C/DINO-S 曲线；
- LPIPS 曲线；
- 内容—风格斜率；
- 单调性异常；
- 最有效工作区间。

最终给出一个合理的 `strength` 映射。

## B04：CPU/I/O 性能诊断

结合：

- `src/utils/raw_io.py`
- `src/utils/runtime.py`
- `src/eval/runner.py`
- `tools/benchmark_timing.py`

测量：

- 750 任务结果序列化；
- PNG 压缩；
- JSON 汇总；
- 缓存元数据加载；
- 哈希开销；
- 多小文件和批量文件；
- Windows/Linux 路径处理。

**只运行 CPU 测试。**

大规模 I/O 压测不能与 GPU 训练同时抢共享磁盘。

---

# 五、第三批：代码整理与重构

| ID | 槽位 | 任务 | 交付 |
|---|---|---|---|
| A05 | A | 128D/795D 风格特征有效维度诊断 | PCA、相关性、冗余分析 |
| B05 | B | 统一 Minimal PRISM 配置 | `src/v25/config.py`＋测试 |
| A06 | A | 20 个跨风格迁移方向分析 | 热力图＋失败模式报告 |
| B06 | B | 结果 Writer、任务状态与恢复检查 | CPU 回归测试＋兼容包装层 |

## B05：配置规约

最终用户接口只有：

```python
PRISMCore.transfer(
    content,
    style,
    strength=0.5,
)

PRISMRef.transfer(
    content,
    reference,
    strength=0.5,
)
```

研究用的高级配置仍然保留，但不暴露为必要用户参数。

必须支持：

- 配置 hash；
- checkpoint 对应关系；
- 默认参数冻结；
- 精确恢复历史实验；
- 不同 strength 的确定性。

## B06：结果与恢复

重点检查：

- 任务执行中断后能否续跑；
- 是否重复评测已完成任务；
- 是否会覆盖成功结果；
- JSON 是否原子写入；
- task_id 是否完整；
- 是否有失败但状态仍记为 done；
- Windows schtasks 失败如何被识别。

不能破坏现有评测格式。

---

# 六、第四批：测量、测试与工程交付

| ID | 槽位 | 任务 | 交付 |
|---|---|---|---|
| A07 | A | 完整 RGB→RGB 性能归因模型 | 端到端成本模型 |
| B07 | B | Minimal PRISM CPU 测试套件 | `tests/v25/` |
| A08 | A | 机制 A/B/C 结果独立审查 | PASS/FAIL/INCONCLUSIVE |
| B08 | B | Windows/Linux 运行器整理 | 双平台脚本＋恢复说明 |

### A07：必须测量的阶段

`Encoder → DINO/ConvNeXt → Style Encoding → Anchor → Flow → S10 → I/O`

区分：

- 单张延迟；
- batch 吞吐；
- 首次加载；
- 热缓存；
- 同参考复用；
- 真实 NFE；
- 实际神经网络前向次数。

CPU Agent 负责定义工具和分析结果。

真实 GPU Benchmark 由总控加入对应 GPU 队列，不允许 CPU Agent 自行运行。

### B07：重点回归

不重复已有基础测试，应补齐：

- 非劣性统计边界；
- γ=0 回退；
- 空间门控均值与有效编辑能量；
- 双分支条件保持；
- One-Shot 参考变化；
- teacher/config hash；
- 损坏缓存；
- 重复任务；
- 输出 schema；
- Windows 路径；
- CPU-only 运行。

### B08：跨平台执行

Linux 继续使用 tmux。

Windows 继续使用 `.bat + schtasks`，避免复杂嵌套转义。

增加明确的：

`started / running / done / failed / incomplete`

状态记录，以及失败后恢复说明。

---

# 七、第五批：文档与论文

前面任务结束后，两名子 Agent 转为收束工作。

| ID | 槽位 | 任务 | 交付 |
|---|---|---|---|
| A09 | A | V19–V25 模型谱系与结论整理 | 模型谱系、有效/失败/未决清单 |
| B09 | B | 最小复现与部署文档 | Quickstart、checkpoint manifest |
| A10 | A | 结果表、Pareto、失败案例可视化 | SVG＋统计表 |
| B10 | B | 论文方法与消融章节整理 | `docs/icme2027` 草稿 |

A09 特别要纠正历史中“失败原因已证明”的过度归因：区分代码证实、受控实验支持和合理猜测。

B10 不得根据尚未完成的 GPU 实验预写成功结论。

---

# 八、12 小时轮转排程

注意，下面是**时间预算和优先顺序**，不是让子 Agent 在时间窗口内空等。

| 时间窗口 | Slot A | Slot B | GPU 状态 |
|---|---|---|---|
| 0–0.3h | 协助核查/启动 4090 队列 | 协助核查/启动 3060 队列 | 两条队列独立运行 |
| 0.3–2h | A01 → A02 | B01 → B02 | Round 2/3 收敛 |
| 2–4h | A03 → A04 | B03 → B04 | 机制 A、C1 |
| 4–6h | A05 → A06 | B05 → B06 | 机制 B、C1 |
| 6–8h | A07 → A08 | B07 → B08 | B 复验、C2 |
| 8–10h | A09 | B09 | 最佳候选继续运行 |
| 10–12h | A10 | B10 | 最终评测和验收 |

**如果某个任务提前完成，立即领取下一个任务，不等待时间窗口。**

如果 GPU 产生紧急结果，允许临时中断低优先级文档任务，插入结果分析。

但不得中断已经正常运行的 GPU 实验。

---

# 九、没有 worktree 的共享仓库协作规则

**所有子 Agent 共享当前 checkout，不创建 worktree，也不创建或切换分支。**

## 9.1 文件所有权

每个任务领取时，必须确定唯一修改文件。

建议：

```text
tools/v25_cpu/
    a01_noninferiority.py
    a02_teacher_lineage.py
    a03_teacher_quality.py
    a04_strength_analysis.py
    b04_io_benchmark.py

src/v25/
    config.py
    artifact_registry.py

tests/v25/
    test_stats.py
    test_mechanism.py
    test_registry.py

docs/v25/
    audits/
    performance/
    engineering/
```

两个子 Agent 不得同时修改同一文件。

需要修改共享代码时，先实现独立模块，由总控统一集成。

## 9.2 Git 权限

只有总控可以：

- `git add`
- `git commit`
- `git pull`
- `git push`
- `git checkout`

子 Agent 禁止：

- `git worktree`
- `git branch`
- `git switch`
- `git stash`
- `git reset --hard`
- `git clean`

总控提交时显式指定文件路径，不使用未经检查的 `git add -A`。

## 9.3 运行中的脚本

正在运行的 V25 脚本、环境和缓存不得被子 Agent 修改。

实验开始时记录：

- Git SHA；
- checkpoint SHA；
- config hash；
- seed；
- 任务 ID；
- 完整命令；
- 输出目录。

总控只能在 GPU 安全检查点后同步修改。

---

# 十、任务调度算法

总控维护一个简单任务池，而不是长期固定子 Agent。

```text
while V25 尚未完成:

    检查 GPU-4090 / GPU-3060 状态

    收集已完成子 Agent 的结果

    审查代码与测试

    对完成任务：
        标记 DONE
        释放子 Agent 名额

    对失败任务：
        记录错误与失败原因
        判断是否可重试
        禁止无限重试

    根据当前 GPU 进度调整任务优先级

    while 活跃子 Agent 数 < 2:
        领取最高优先级、无文件冲突的 READY 任务
        启动子 Agent

    定期进行安全的 Git 集成与提交
```

调度优先级：

**P0：阻断 GPU 实验正确性的错误**

→ **P1：科学诊断与必须的测试**

→ **P2：代码整理和性能测量**

→ **P3：文档、论文和图表**

如果 GPU 所需机制代码尚未准备，优先分配开发 Agent，而不是继续做无关的 CPU 分析。

---

# 十一、CPU 资源隔离

子 Agent 并发数为 2，不代表可以无限使用 CPU 核心。

默认规则：

- `CUDA_VISIBLE_DEVICES=""`；
- `OMP_NUM_THREADS=2`；
- `MKL_NUM_THREADS=2`；
- `OPENBLAS_NUM_THREADS=2`；
- 禁止调用真实 CUDA 模型；
- 大型数据分析使用已有缓存和分块读取；
- 不在训练期间进行全盘哈希或磁盘压力测试；
- 不在 GPU 机器上运行大量 multiprocessing worker；
- 不修改正在使用的 Python 环境。

如果 CPU 任务在 GPU 主机上明显影响数据加载吞吐，应暂停该任务或降低其资源配额。

---

# 十二、每个子 Agent 的交付格式

子 Agent 完成后，必须返回：

```text
TASK: A03
STATUS: DONE / FAILED / BLOCKED

FILES_CHANGED:
  ...

KEY_RESULTS:
  ...

TESTS:
  command:
  result:

SCIENTIFIC_CONCLUSION:
  confirmed:
  rejected:
  inconclusive:

RISKS:
  ...

NEXT_TASK_SUGGESTION:
  ...
```

不能只回复“已完成”。

总控据此决定是否合并、返工或启动下一任务。

---

# 十三、完成标准

12 小时结束时，优先保证：

**科研：**

- V25 收敛结果完整；
- 机制 A/B/C 有明确结果；
- 最终候选至少两种子复验；
- 完整 test750；
- Pareto 与非劣性可复算。

**诊断：**

- 锚点分布与教师定义清晰；
- O2 Teacher 偏保守的原因得到量化；
- One-Shot 参考响应有独立诊断；
- 训练/推理条件差异得到审计。

**工程：**

- 最简接口；
- 统一配置；
- checkpoint/cache 血缘；
- CPU 单元测试；
- 可恢复 GPU 队列；
- 端到端性能测量规范。

**文档：**

- 模型谱系；
- 实验结论账本；
- 复现手册；
- 最终指标表；
- 论文材料。

不要为了完成全部文档而牺牲实验正确性，也不要为了用满 12 小时而重复已经判负的实验。

---

# 十四、直接发给当前总控 Agent 的执行指令

你现在作为 WEAVE V25 的唯一总控。

**重要限制：最多同时运行 2 个子 Agent；严格禁止 worktree。**

请立即按以下顺序执行：

1. 检查本地 Git 状态、4090 tmux、3060 schtasks，确认既有任务，禁止重复启动。
2. 让 4090 和 3060 使用各自机器的持久化 GPU 队列继续实验。GPU 进程不需要子 Agent 持续监控。
3. 两个子 Agent 名额分别用于科学诊断和工程开发，按本计划的 A01–A10、B01–B10 顺序轮转。
4. 如果 GPU 队列需要新代码，优先派发该代码开发任务；不要让 GPU 等待低优先级文档工作。
5. 每个子 Agent 只领取一个有明确交付物的任务，完成后立即结束并释放并发名额。
6. 所有 Agent 共享当前仓库，禁止 worktree、分支切换和子 Agent Git 操作。
7. 严格文件所有权，公共代码由总控统一集成。
8. CPU Agent 不使用 GPU，也不得明显影响 GPU 训练的 CPU、内存和磁盘资源。
9. Round 1 已完成，不再重复；Round 2/3 根据实际状态接力。
10. 保持机制 A/B/C 独立推进，失败不阻断其他实验。
11. 统计结论必须由真实任务数据支持，不能把外推当作实测。
12. 每个子 Agent 交付代码、测试、报告和明确结论；总控负责审核与 Git 提交。
13. 每个 GPU 机制完成后，优先将结果交给空闲的科学子 Agent 分析。
14. 对最终候选运行完整 test750、独立种子复验和端到端性能测量。
15. 最终形成 Minimal PRISM 双接口、完整结果台账、科学诊断与可复现文档。

**不要创建 20 个等待中的子 Agent。维护 20 个任务，但始终只让 2 个 Agent 工作。GPU 持续运行，CPU Agent 持续轮转。**
