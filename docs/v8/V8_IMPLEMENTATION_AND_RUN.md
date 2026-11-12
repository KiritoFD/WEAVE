# WEAVE V8：实施与运行记录（生成器约束的目标风格优化）

更新：2026-11-13（§7–§10 接口/矩阵/缺口为整理时按代码补记）。**本文只记录已验证的事实；未完成的项明确标注。**

---

## 1. 一句话现状

V8 的**推理基础设施已重写并验证**（`src/`），**实验代码已实现**（`exp/v8/`），
**24h 串行队列已于 02:55 在 3060 tmux `v8queue` 启动**，P0 合同测试通过，
A3 训练正在推进（step 100/3000，margin 2.58→4.57）。

**尚无可报告的方法学结论**——正式臂还没有一个跑完 750。

---

## 2. 代码结构（本轮重构）

按用户要求：模型归 `src/model`，推理归 `src/infer`，评测归 `src/eval`，
实验归 `exp/v8`。删除了 `tools/v8`、`tools/v8_24h` 两个平行目录及其临时脚本。

```
src/
  model/     unet.py(C3UNet,ScoreNet) critic.py(R1) canon.py transport.py
             模型定义的唯一归属地；新变体加在这里，不 fork 已有网络
  infer/     __init__.py(注册表) variants.py(9种变体) run.py(CLI)
             ckpt -> raw latent/uint8 图；不算指标、不写 PNG
  eval/      measure.py(raw -> 指标+PNG) metrics.py(配对统计)
  utils/     vae.py(TRT加载/定batch解码) features.py(冻结评测参照)
exp/v8/      controls.py(A) guidance.py(B) energy.py(C) p0_contract.py
             arms.py train.py run_arm.py specs.py build_support.py
             probe_step.py bench_train.py
archive/v8_ot_geometry/   已停用的 OT/几何匹配路线（含停止理由）
```

**已注册的 9 种推理变体**：`direct / relative / oracle / r1_steps / control /
supported / random / student / identity`。V8 的 28 个推理臂全部映射到这些变体，
`exp/v8/specs.py` 负责映射，推理代码不需要 V8 专用分支。

### 2.1 为什么归档 OT 路线

`archive/v8_ot_geometry/README.md` 记录了理由，核心是一个代数事实：

> 固定权重下 `E(h) = Σ π_j ||h − a_j||² = ||h − μ||² + const`，只能回归加权均值。

实测确认了这个缺陷：

| 位置 | mixture 能量 | centroid 能量 |
|---|---:|---:|
| 真实模式上 | **0.347** | 18.000 |
| 两模式中间空洞 | 9.000 | **9.000** |

mixture 能区分真实笔触和空洞，centroid（旧 CPT）反而**偏好空洞**。新计划因此
转向在生成器内部做控制，而不是继续扩大 patch 库。

---

## 3. P0 合同测试（已通过）

`python3 -m exp.v8.p0_contract`，02:55:23 PASS。这些是启动前必须过的检查，
每一条都对应本仓库曾经真实踩过的坑：

| 检查 | 实测结果 | 为什么必须 |
|---|---|---|
| 冻结资产 | B4/Q2A/R1 路径均存在 | 曾有脚本猜路径 |
| **Oracle 符号** | margin −4.775 → **+grad 后 −2.193**（升）→ −grad 后 −9.692（降） | 历史上把 `+∇softplus(−margin)` 误标为正向 |
| **u=0 严格等价** | A0/A1/A2/A4 四组 `max_abs_diff = **0.0**` | 任何控制必须初始为无操作 |
| 控制非零有效 | 0.3895 | 防止 u=0 与 B=0 同时导致无梯度 |
| **逐样本隔离** | 改动样本0 时变化 `[2.24, 0, 0, 0]` | 旧代码曾把 `tid[0]` 用于整批 |
| 控制维度 | A0 204 / A1 160 / A2 204 / A4 208 | 远小于 16384 维裸 latent |

---

## 4. 训练显存与 batch 定案（实测）

`exp/v8/probe_step.py` 跑的是**真实训练步**（风格损失会反传穿过冻结生成器激活），
不是 detach 后的单步 bench——后者低估约 1.8× 显存，正是 384 batch 起初 OOM 的原因。

### A3/A4（生成器冻结，训练控制策略）

| microbatch | 显存 | sec/update | samples/s |
|---:|---:|---:|---:|
| 96 | 3.40 GB | 0.33 | 293 |
| 192 | 6.77 GB | 0.58 | 333 |
| **288** | **10.14 GB** | **0.79** | **364** ✅ |

**采用 288**：吞吐最高，显存正好压在 12GB 卡的 10GB 目标。384 会 OOM（约 13.5GB）。

### 其他训练组（`bench_train.py` 三轮探测，每组 ≤3 个配置）

| 实验组 | microbatch | 显存 | sec/update |
|---|---:|---:|---:|
| A3/A4 | 288 | 10.14 GB | 0.79 |
| L2/L3 (SPD K2) | 288 | 4.96 GB | 0.67 |
| L3K4 (SPD K4) | 288 | 7.12 GB | 0.98 |
| C3 (critic) | 512 | 9.31 GB | 0.84 |

**有效 batch 说明**：文档规定 A 用 240、B 用 128。实测下 microbatch 288/512 已
**大于**该值（accum=1），因此实际有效 batch 更大。这是更充分而非更少的数据量，
各臂 state 会记录实际值。

### 训练 ETA（按实测 sec/update）

| 臂 | updates | sec/upd | 估算 |
|---|---:|---:|---:|
| A3 | 3000 | 0.79 | 40 min |
| A4 | 3000 | 0.79 | 40 min |
| L2 | 2000 | 0.67 | 22 min |
| L3 | 2000 | 0.98 | 33 min |
| L4 | 2000 | 0.67 | 22 min |
| C3 | 2500 | 0.84 | 35 min |
| **合计** | **14500** | | **≈3.2 h** |

### 推理 ETA

- 无梯度臂（A0/A1/A2/L0/L1/C0/C1/C2，14 个）：生成 ~72s + 评测 ~160s ≈ 4 min/臂 → **~56 min**
- 有梯度臂（L1.topk/C4/I0–I3 等，10 个）：约 2–3× 生成时间 → **~80 min**
- 中间 checkpoint 评测：**~1.5 h**

**全队列估算 ≈ 8–9 h**，24h 预算内有约 2.7× 裕量。

---

## 5. 队列状态（02:57 核实）

- tmux `v8queue` 运行中；GPU 11.2 GB / 利用率 49–97%
- **P0 PASS**
- **A3 训练推进中**：step 1 loss=0.4801 margin=2.582 → step 100 loss=0.1781 **margin=4.567**
  （margin 上升说明控制策略确实在提升目标风格分数）
- A4/L2/L3/L4/C3 尚未开始
- **尚无 checkpoint 落盘**（首个 checkpoint 在 step 1500）

队列顺序：`P0 → support bank → A3/A4训练 → L2/L3/L4训练 → C3训练 →
A推理 → C推理 → B推理 → I联合`。单臂失败记 FAILED 但**不中断队列**
（科学负结果同样不取消后续独立臂）。

---

## 6. 本轮修复的真实缺陷（过程记录）

| 问题 | 现象 | 修复 |
|---|---|---|
| palette 切片 | 44 标量按 `[4,4,12,12]` 切却 view 成 4×4 | 改为 `[16,4,12,12]` |
| `Controls` 用 nn.Parameter | backward 报 `does not require grad` | 改为普通张量，由调用方设 requires_grad |
| **u=0 fast-path 切断梯度** | 控制臂首个 smoke 直接崩 | fast-path 仅在 `requires_grad=False` 时启用 |
| `_local_injection` 广播 | 4D pad 尺寸不匹配 | 所有基统一为 (1,1,H,W) |
| A3 384 OOM | 训练比 bench 多 1.8× 显存 | 经实测降为 288 |
| tar 同步不覆盖 | bench 一直跑旧配置 | 改用 `rm -rf` 后解包；后续改用 scp |

同步方式也踩了坑：scp 的 sftp subsystem 曾被禁用，恢复后目标是
`3060:I:/Latent_Style/WEAVE/`（Windows 盘符路径）。

---

## 7. 代码接口契约（以实现为准）

### 7.1 推理层 `src/infer/`

- `InferSpec`（dataclass）：`ckpt / kind / canon_mode / steps / step_rms / rho / energy / seed / extra`。所有臂参数最终收敛到这一个 spec。
- `Generator` Protocol：`gen(z_src, sid, tid, tasks) -> {"z0","final","trace",...}`，返回 raw latent，**不算指标、不写 PNG**。
- `MODELS` 注册表：`@register(name)` 装饰器登记变体，`build(spec, device)` 按 `spec.kind` 分发。已注册 9 种：`direct / relative / oracle / r1_steps / control / supported / random / student / identity`。
- `run(spec, out, tasks_file, src_cache, batch, limit, decode, device, resume)`：按 `batch_XXXX.pt` 分片原子写入（tmp+rename），断点续跑校验 task_id 对齐；`meta.json` 记录 wall_sec / ms_per_img / img_per_s。
- CLI：`python -m src.infer.run --kind ... --ckpt ... --out ...`。

### 7.2 评测层 `src/eval/`

- `measure(gen_dir, eval_dir, device, write_png, limit)`：消费 `src.infer` 的 raw latent + rgb_u8，补算 DINO-S/C、LPIPS、CLIP-S、Style Affinity、CSD Top1/Margin/ΔT、G 等，写 PNG 与逐任务行。与推理完全解耦，可离线重算。
- `metrics.py`：纯函数——逐任务富集、G（目标相对特异性）、I（目标切换响应）、源 cluster bootstrap CI（重复抽样不去重）。

### 7.3 模型层 `src/model/`

网络定义唯一归属地，全部**再导出**自历史实现以保证数值逐位一致，不 fork：
`C3UNet / ConditionalScoreNet / StyleAdaLNBlock / StyleChannelPaletteHead`（tools/v6/models_v6）、`LatentStyleNet`（tools/v6_next/probe_style，R1 critic）、`ContentCanonicalizer`（tools/v6/canonicalize，c1_gentle 冻结统计）、`SingleStepTransportNet`（tools/v7）。
`src/utils/`：`load_vae / load_trt / decode`（TensorRT 固定 batch=8 解码）、`FeatureStack`（仅测量用的冻结 DINO/CLIP/CSD/LPIPS 参照，禁止进训练损失）。

### 7.4 实验层 `exp/v8/`

| 文件 | 职责 |
|---|---|
| `arms.py` | 固定臂矩阵 `GROUPS`（P0/A_infer/A_train/A_eval/B_infer/B_train/C_infer/C_train/I），臂 ID 不与历史冲突 |
| `specs.py` | 臂 dict → `InferSpec` 的唯一映射处；推理代码无 V8 专用分支 |
| `controls.py` | A 路线：`ControlSpec`/`Controls`/`ControlledGenerator`，低维控制 u 注入 StyleAdaLN/Palette/局部低秩；`u=0` 与原 forward 逐位一致（有断言） |
| `guidance.py` | B 路线：`SPDHead`（P=diag(softplus)+UUᵀ，恒等初始化）、`UnsupervisedUnroll`（K 步一阶截断展开，**非精确二阶元梯度**）、`project_rms` 硬位移球 |
| `energy.py` | C 路线：`RealSupport`（每风格 K 原型 R1 embedding 支持库）、`SupportedEnergy`（F=m_t−λ·E_real）、`robustness_finetune_step`（C3 critic 微调） |
| `build_support.py` | 从训练池构建 `runs/v8/real_support.pt`（5×K×D 原型 + σ），只用训练画作 |
| `train.py` | 统一训练器 `train_arm(arm_id)`：A3/A4 控制策略、L2/L3/L4 SPD、C3 critic；microbatch 由 `probe_step.py` 实测定案 |
| `run_arm.py` | 单臂端到端：`infer_run` 生成 → `measure` 评测，写 `runs/v8/<arm>/{gen,eval,state.json}`，支持 `--skip-gen` 续跑 |
| `p0_contract.py` | 启动前合同测试（Oracle 符号、u=0 等价、逐样本隔离、资产路径） |
| `probe_step.py` / `bench_train.py` | 真实训练步显存/batch 实测工具 |
| `energies.py` / `edit.py` | **遗留/未接入**：patch 能量与约束编辑器原型；`energies.py` 依赖已归档的 `.features`，当前不可导入 |

### 7.5 队列 `sbflow/scripts/run_v8_queue.sh`

固定顺序：`P0 → support bank → A3/A4 训练 → L2/L3/L4 训练 → C3 训练 → A 推理(12臂) → C 推理(5臂) → B 推理(6臂) → I 联合(4臂)`。单臂失败记 FAILED 继续；`runs/v8/<arm>/state.json` 记 DONE/FAILED，重跑自动跳过 DONE。GPU 严格单卡串行，WSL `/mnt/i/Latent_Style/WEAVE`。

## 8. 实际臂矩阵（代码为准）

**训练臂（6 个，14500 updates，有效 batch：A/C=240、L=128，bf16 未启用、microbatch 见 train.py）**

| 臂 | 内容 | updates | ckpt |
|---|---|---:|---|
| A3 | Q2a + hier_palette 控制策略（全局+逐尺度+palette） | 3000 | 0/1500/3000 |
| A4 | A3 + rank-2 局部控制 | 3000 | 0/1500/3000 |
| L2 | B4 起点 latent 空间对角 SPD，K=2 | 2000 | 0/1000/2000 |
| L3 | B4 起点 latent 空间 4×4 SPD，K=4 | 2000 | 0/1000/2000 |
| L4 | A3 控制坐标空间 SPD，K=2 | 2000 | 0/1000/2000 |
| C3 | R1 真实画作小半径鲁棒微调 | 2500 | 0/1250/2500 |

**推理臂（28 个，全部 tasks_750）**

- A：`A0/A0.s2/A0.s4`（global）、`A1*`（hier）、`A2/A2.s2/A2.s4`（hier_palette，Q2a）、`A2b`（hier_palette，B4 起点）、`A3.eval/A4.eval`（训练后策略）
- B：`L0.r08/r15/r25`（Q2a+Oracle 单步三档 ρ）、`L1.b4.r25/r15`（B4+Oracle）、`L1.topk`（B4+R1 TopK25 四步）
- C：`C0`（纯 R1 margin）、`C1.l010/l030`（+真实支持 λ=0.1/0.3）、`C2`（局部支持）、`C4`（C3 critic）
- I：`I0–I3`（A3 控制 × L4 SPD × λ_ratio 0/0.1）

## 9. 与 24h 设计文档的差异（必须知道）

设计文档（`V8_24H_RESEARCH_PLAN.md` 等四份）规划的是 H0–H6 / 121 注册项 / M-G-T-D 体系 / 36000 教师任务 / 28000 学生 updates。**当前队列没有实现那套矩阵**：没有 mixture/OT 能量臂、没有 G0–G12 归因、没有教师蒸馏学生（D0–D6）、没有 121 项注册表。实际规模是上表 6 训练 + 28 推理臂。设计文档中的协议纪律（逐图独立、u=0 无操作、拒绝不前进、NO 不取消队列、只读历史）仍然有效并已在 P0/代码中体现；矩阵规模与阶段划分以本文件为准。

## 10. 已知缺口（整理时核对代码发现）

1. **`exp/v8/energies.py` 不可导入**：`from .features import N_PATCH`，但 `exp/v8/features.py` 已随 OT 路线归档（`archive/v8_ot_geometry/features_geometry.py`）。当前无任何活跃代码 import 它，属死代码；若要复用需先恢复 features 或删除该文件。
2. **`exp/v8/models.py` 不存在**：`src/infer/variants.py` 的 `student` 变体 `from exp.v8.models import build_corrector`，该模块未实现 → student 变体当前不可用。
3. **`spd_edit` 映射是占位**：`specs.py` 中 `I2/I3`（kind=`spd_edit`）返回 `kind="control"` 的 InferSpec 且注释 "replaced below"——**I2/I3 实际会按普通 control 臂跑，SPD/L4 不会被加载**，结果不代表联合臂。队列跑到 I 段前需要修复 `specs.py`/`variants.py` 或从队列移除 I2/I3。
4. **`control_ckpt` 未被消费**：`A3.eval/A4.eval` 经 `infer_spec_for` 把训练好的 policy 路径放进 `extra["control_ckpt"]`，但 `variants._control` 不读该字段——**A3.eval/A4.eval 目前等价于零训练控制臂**，不会加载 `runs/v8/train/A3/step_03000.pt`。同理 L4 的 `control_ckpt`/`spd_ckpt` 也无消费方。
5. **训练 ckpt 与推理变体未打通**：`train.py` 保存 `{"policy": ...}` / `{"head": ...}`，但没有对应的推理加载路径（见 3/4）。
6. 中间 checkpoint（A3/A4 的 1500、L* 的 1000、C3 的 1250）已登记但队列只评测最终推理臂，**非零 ckpt 的 750 评测未排入队列**。
7. 有效 batch 大于设计值（microbatch 288/512 > 240/128，accum=1），已在 §4 记录，非静默改动。

## 11. 待完成与不可宣称的部分

**未完成**：
- A3/A4/L2/L3/L4/C3 训练尚未跑完；无 checkpoint
- 28 个推理臂无一完成 750 评测（仅 8 张 smoke 通过）
- I0–I3 联合臂依赖 A3/L4 checkpoint，尚未开始
- 尚无视觉检查、无真实 b1/b8 端到端测速
- 无 `V8_RESULTS.md` 结果报告

**不可宣称**：
- 不能说 A/B/C 任何一条路线"有效"或"更强"
- 不能说控制空间比裸 latent 编辑更好
- 8 张 smoke 不是性能证据

**已知限制**：
- 有效 batch 大于文档规定（已记录，非静默改动）
- 训练使用一阶截断展开（每步 detach），**不是精确二阶元梯度**
- C3 的 support 项基于冻结 R1 embedding，是**支持代理**而非标定密度
- 所有统计均为 tasks_750 研发集内结果，无盲测

---

## 12. 复现入口

```bash
# 3060 WSL
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 -m exp.v8.p0_contract          # 合同测试
python3 -m exp.v8.build_support        # C 路线 support bank
python3 -m exp.v8.train A3             # 单臂训练
python3 -m exp.v8.run_arm --arm A2.s2  # 单臂评测（生成+测量）
python3 -m exp.v8.probe_step --arm A3 --batches 288   # 显存/batch 实测
bash sbflow/scripts/run_v8_queue.sh    # 完整串行队列

# 本地
python3 _tmp/sync.py exp/v8 src        # 同步到 3060
```

产物：`runs/v8/<arm>/{gen/,eval/,state.json}`、`logs/v8_*.log`、`runs/v8/train/<arm>/`。
