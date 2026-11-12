# WEAVE V8：实施与运行记录（生成器约束的目标风格优化）

更新：2026-11-13 02:58。**本文只记录已验证的事实；未完成的项明确标注。**

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

## 7. 待完成与不可宣称的部分

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

## 8. 复现入口

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
