# V10 实施与运行说明

> 依据 `docs/bridgev10.md`。数据口径见仓库根目录 `data.md`（5 风格并集 15,545 张）。
> 本轮交付：**40 组实验中 38 组可执行、全部通过测试**，并已拉成串行队列。

---

## 一、代码结构（沿用 sbflow，不新建代码树）

| 文件 | 作用 |
|---|---|
| `sbflow/network.py` | 五风格条件 DiT：`c = time(τ) + W[e_a;e_b] + direction`；可选 pair-ID embedding、独立正反头、pair 低秩适配器、深度可分离局部卷积 |
| `sbflow/preconditioning.py` | `StyleStats`（共享全局 mean/std + 五域逐域统计）、十对 EDM 矩、**直接速度 EDM**、端点期望 EDM、score/current 分解 |
| `sbflow/data5.py` | 五风格缓存加载、统一全局标准化、十对均衡调度、无放回/有放回采样器 |
| `sbflow/train5.py` | 训练主程序：4 种目标 × 全部消融开关 × D/E 接入 |
| `sbflow/infer5.py` | 任意 (源, 目标) 推理；规范序 a<b，a→b 正向积分、b→a 反向积分、同风格恒等 |
| `sbflow/gen5.py` | 600 / Quick200 / 同风格 150 的生成与 manifest |
| `sbflow/score_cpu.py` | **纯 CPU** 指标计算（无任何 CUDA 调用），按 `target_style` 分组 |
| `sbflow/style_loss.py` | 对比学习风格编码器、SWD²（全局/局部）、分块梯度 |
| `sbflow/online_bridge.py` | 真 SDE 正/反向采样、在线 α-DSBM 损失、replay buffer、动态交叉协方差 |
| `sbflow/pivot.py` | 高斯 pivot：`T_{i→j} = F_j ∘ F_i^{-1}` |
| `sbflow/arms.py` | 40 组实验表 |
| `sbflow/run_suite.py` | 串行驱动：训练 → 生成 → CPU 打分 → 记录 |
| `sbflow/time_sampling.py` | 分层时间采样 + 重要性权重（exp 15） |

---

## 二、40 组实现状态

| 组 | 编号 | 状态 |
|---|---|---|
| A 批/采样 | 01–04 | ✅ 全部实现 |
| B 参数化 | 05–10 | ✅ 全部实现 |
| C 桥/EDM | 11, 12, 14, 15 | ✅ 实现 |
| C 桥/EDM | **13**（32×32 通道协方差 EDM） | ❌ 未实现，见下 |
| D 分布监督 | 16–23 | ✅ 全部实现 |
| E 在线 α-DSBM | 24–30, 31 | ✅ 全部实现（31 含动态交叉协方差） |
| F 高斯 pivot | 32–35 | ✅ 全部实现 |
| G 共享结构 | 37, 38 | ✅ 实现 |
| G 共享结构 | **36**（三风格路径组合一致性） | ❌ 未实现，见下 |
| H OT | 39, 40 | ✅ 实现 |

**合计 38/40 可执行。** `arms.py` 把 13、36 标为 `implemented=False`，
`run_suite` 会**跳过**而不是静默替换成别的东西。

未实现的两个及其原因：

- **13（矩阵 EDM）**：需要把逐通道标量系数换成 32×32 矩阵预条件器
  （`Var_z^{-1/2}`、`Cov·Var_z^{-1}`、Schur 补 Cholesky），训练与推理两条路径都要改，
  且要在 FP32 里保证协方差正定。这是深层数值改动，本轮不做半成品。
- **36（路径组合一致性）**：`T_{j→k}(T_{i→j}(z)) ≈ T_{i→k}(z)` 需要每步两次可微 ODE，
  计算量翻倍。文档本身也标注它是**新的结构约束**、不是原论文定理。

---

## 三、测试结果（3060 实测，全部通过）

```
V10 数学（7 项）
  PASS test_style_stats_pair              # 十对矩由五域统计解析构造，交叉协方差为 0
  PASS test_direct_velocity_whitening     # 直接速度 EDM 残差 均值<0.03、方差偏离<0.06
  PASS test_direct_velocity_reconstruct
  PASS test_score_current_roundtrip
  PASS test_independent_closed_form_still_holds
  PASS test_direct_velocity_reduces_to_cfm_at_eps0
  PASS test_pair_ids                      # 十个 pair id 对称且覆盖 0..9

旧 sbflow 数学（7 项）全部通过

D/E 机制（10 项）
  PASS test_swd2_properties               # 可微、自距离为 0
  PASS test_contrastive_learns_separation # 4.769 -> 4.143
  PASS test_distribution_loss_modes       # 3 模式 × 2 编码器
  PASS test_sde_endpoints_are_stochastic_and_stay_close
  PASS test_ode_forward_is_deterministic
  PASS test_online_loss_variants          # 6 种在线配置
  PASS test_online_endpoints_have_no_grad # 端点 stop-gradient
  PASS test_replay_buffer
  PASS test_cross_cov_estimation
  PASS test_both_drifts_finite

训练冒烟（12 组目标/消融）+ 端到端（600/200/150 任务、恒等精确、四方向有限）全部通过
```

补充实测：8 个 D 臂、7 个 E 臂、4 个 F 臂各自跑通；arm 31 的
`cross_cov_mean` 轨迹稳定（1.0311 → 1.0286）。

---

## 四、关键实现细节（容易出错的地方）

1. **共享全局标准化**：五域用同一个 `mean/std`，禁止逐风格白化。
   测试 `test_shared_global_normalization` 断言逐风格均值仍有 0.74 的离散度。
2. **十对矩解析构造**：无 OT 时端点独立 → `Cov(z_a,z_b)=0`，
   所以不必再用大量随机配对估计交叉协方差。
3. **直接速度必须重推 EDM**：`v* = (z1-z0)+c(t)ξ` 的系数与双漂移完全不同，
   复用了会训练不起来。已在 `direct_velocity_coefficients` 中推导并数值验证。
4. **score 头必须标定**：不标定时 score 目标是 O(1/ε) 量级，会压倒速度损失
   （实测 loss 351 → 标定后 2.08）。
5. **在线端点必须 stop-gradient**，且必须用真 SDE（有测试断言 SDE 两次采样不同、
   ODE 两次采样相同）。
6. **规范序**：`a=min,b=max`，`a→b` 正向积分、`b→a` 反向积分；同风格直接恒等，
   不训练恒等网络（测试断言逐元素相等）。
7. **CPU 打分**：`score_cpu.py` 不含任何 `.cuda()` / `autocast`，
   可在 GPU 跑训练或生成时并行进行。

---

## 五、串行队列

`sbflow/scripts/launch_v10_suite.sh`：

1. 等五个风格缓存就绪（`runs/blog_sbflow/cache5/*.pt`）
2. **第一阶段闸门**：arm 04 从零训练 + **full600 生成**（未过闸门不启动后续）
3. 依次 A → B → C → G/H → D → E → F，每组 Quick200

每组记录（写入 `runs/blog_sbflow/v10/suite_state.json`）：
`experiment_id / git_commit / config_sha256 / seed / styles=5 / pair_schedule /
batch / updates / endpoint_exposures / train_seconds / peak_vram / checkpoint /
full_or_quick_eval / eval metrics`。

产物目录：
```
runs/blog_sbflow/v10/
  exp_<id>/            model_step*.pt, train_summary.json, style_encoder.pt
  gen_<id>_<subset>/   generated/*.png, manifest.json
  score_<id>_<subset>/ style_scores.json
  logs/<id>_train.log / _gen.log / _score.log
  suite_state.json
```

## 六、如何检查进度

```bash
# 整体进度
python3 -c "import json;d=json.load(open('runs/blog_sbflow/v10/suite_state.json'));print(len([k for k,v in d.items() if v.get('status')=='done']),'/',len(d))"
# 单组日志
tail -n 40 runs/blog_sbflow/v10/logs/04_train.log
# 当前 GPU
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
```

## 七、已知的评测成本

Quick200（200 张）纯 CPU 打分约 6–10 分钟/模型；full600 约 20–30 分钟。
这是刻意选择：CPU 打分不阻塞 GPU，可以边训练边补跑。
