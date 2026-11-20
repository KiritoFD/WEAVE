# V26 工程实施规范、`exp/v26/` 分卡目录结构与 `tests/v26/` 单元测试契约

> **✅ 实现状态（2026-11-21）**：`exp/v26/` 公共层（`sr_cfg.py` / `diagnose_q32.py` / `diagnose_refattn.py` / `k2_style_subject_audit.py` / `dynamic_style_swd.py` / `sg25_hook.py`）与双卡队列（`gpu4090/queue_k.py`、`gpu3060/queue_p0.py` + `train_c2nt_v26.py`）已落地；`tests/v26/` **13 项契约 / 19 个测试全绿**（`python -m pytest tests/v26/ -q`）。下方原始契约清单保留作验收对照。

---

> **隔离原则**：全部新代码放入 `exp/v26/`（按 `gpu3060/` 与 `gpu4090/` 隔离运行代码，共享模块放 `exp/v26/` 公共层）与 `tests/v26/`，**不得修改 `exp/v25/`、`docs/ours/` 或已有 V25 权重与台账**；全仓继续严格遵守 `tests/test_no_absolute_paths.py`（零硬编码绝对路径）与 `tests/test_markdown_links.py`。

---

## 一、`exp/v26/` 目录布局与模块职责

```text
exp/v26/
├── __init__.py                 # 公共路径锚定 (WEAVE_ROOT)、config_hash、seed、CUDA 清理与 JSON I/O
├── source_repulsive_cfg.py     # Source-Repulsive CFG (SR0–SR7, P=I / P_C, Same150 无 bypass 自参考校验)
├── dynamic_style_swd.py        # 冻结 VGG-16 relu1_2/2_2/3_3 动态 64 投影 SWD + 512 token 抽样 + 有界 Hinge
├── sampler_fixed.py            # 修复 (seed, step, accum_index) 采样流 + S0–S2 策略与路线覆盖/重复率审计
├── eval_v26.py                 # 统一全量 test750 (含无 bypass Same150)、SG-25 OOD、20 路线与 3060 测速
├── gpu3060/                    # RTX 3060 12GB 主训练与突破队列 (产物 → runs/v26/gpu3060/)
│   ├── train_swd.py            # L0–L3 动态 SWD 训练器 (5 重防灰保护 + 断点 EMA/opt/RNG 落盘)
│   ├── ref_multiscale.py       # P3-A M0–M2 轻量非对称参考纹理 (源 16x16 不变, T_ref32 <= 0.5M)
│   ├── train_rollout.py        # P3-B 3060 显存四级摸底 + k∈{2,3,4} 随机步截断 Rollout
│   ├── few_step.py             # P0-D / P4-A 少步减法 + P4-B 条件 Reflow 短路径蒸馏
│   └── queue_3060.py           # Windows .bat + schtasks 串行持久化队列
└── gpu4090/                    # RTX 4090 24GB 独立诊断与上界队列 (产物 → runs/v26/gpu4090/, 报告 → docs/v26/4090/)
    ├── k0_q32_audit.py         # K0: dino_16 / PCA32 / 硬 Q32 / Soft-Q32 谱与线性探针泄漏审计
    ├── k1_refattn_causal.py    # K1: RefAttnV24 Hook 遥测 + R0–R6 因果干预 + v_tgt−v_src 速度场解剖
    ├── k2_swd_subject_audit.py # K2: VGG/Gram/SWD 风格判别力 + 128 图梯度诊断 + CSD 题材偏置拆解
    ├── k3_q32_matched_ft.py    # K3: Q0–Q3 同规格 32→384 适配器 1200 upd 匹配训练消融
    ├── k4_decoder_matched.py   # K4: Full SD-VAE vs Full FT vs S10 同域同预算匹配微调对照
    ├── k5_full_rollout.py      # K5: T0 (1-step预览) vs T1 (中间截断) vs T2 (4步完整可微 Rollout 上界)
    ├── k6_highres_ref_value.py # K6: V0–V3 高分辨率参考 Value 上界实验
    ├── k7_curvature_cache.py   # K7: ODE 曲率/非均匀时间网格归因 + 训练集教师轨迹缓存生成
    └── queue_4090.py           # Linux tmux 串行持久化队列
```

---

## 二、`tests/v26/` 强制 CPU 单元测试清单

在启动任何 GPU 训练或诊断前，必须完成并全绿通过以下 CPU-only 单元测试：

1. **SR-CFG 自参考零差分**：当 `target_ref == source_ref`（且相同内容/时间/噪声）时，$v_{\text{pos}} - v_{\text{src}}$ 在数值精度内**精确为零**，无论 $w$ 与 $P$ 取何值。
2. **CFG 分支条件严格对应**：确认 `positive` 与 `negative` 分支输入完全一致的 `z, t, y16, content`，仅参考张量按预期切换。
3. **Q32 维度与适配器兼容**：`dino_16 (384)`、`PCA32 (32)`、`Q32 (32)`、`软Q32 (32)` 经统一 `32→384` 投影适配器输出维度、设备、梯度连通性完全对齐。
4. **软 Q32 低秩数值稳定**：在全零输入、秩为 1 或极小奇异值的病态特征上，Tikhonov 软 `Q32` 不产生 `NaN / Inf`。
5. **采样器非重复性校验**：同一 `update` 内不同 `accum_index` 产生的 `(source, reference)` 索引对不再重复，且给定 `(seed, step, accum_index)` 可精确复现。
6. **SWD 动态投影刷新**：相邻两个 `update` 之间随机投影方向真实更新。
7. **SWD 同向同索引对齐**：同一次 `loss` 调用内，输出 $\hat{x}$、目标参考 $x_{\text{ref}}$、源图 $x_{\text{src}}$ 共享同一随机投影方向与空间 token 抽样坐标。
8. **SWD-Hinge 符号与边界**：验证 $D_{\text{style}}(\text{out}, \text{ref}) \le D_{\text{style}}(\text{out}, \text{src}) - m$ 时 $L_{\text{dir}} \equiv 0$，反之为正且对 `out` 存在正确反向梯度；$\text{src} == \text{ref}$ 时 $m=0$。
9. **Rollout 梯度可达性**：在截断步与完整 4 步模式下，分别验证 `no_grad` 段零梯度累积及可微步经输出向主干参数反传非零有限梯度。
10. **断点续训等价性**：模拟在 step $N$ 中断并恢复 `model / ema / optimizer / rng` 状态后，继续训练 2 步与连续训练结果一致。
11. **Same150 无 bypass 防护**：开启 `allow_same_bypass=False` 时，同风格任务经过完整网络推理而非直接拷贝 `z_src`。
12. **4090 缓存污染隔离守卫（针对 `K7`）**：验证教师轨迹缓存生成器严格校验任务列表仅来自训练集 ID，拒绝任何 `test750` 或 `SG-25 OOD` 图像进入缓存。

---

## 三、工程运行红线

- **严禁启用 `dev200` 代替正式选型**：所有正式模型比较与筛选一律使用完整 `test750`（`Cross600` 读数 + `Same150` 控制）。
- **固定 Git SHA 与配置 Hash**：每条任务运行前记录 `git rev-parse HEAD`、超参 `config_hash`、随机种子并落盘至独立输出目录 `runs/v26/gpu{3060,4090}/<arm_id>_<hash>/`。
- **3060 显存安全线**：严格遵循 `tier_micro_accum` 与 `≤10.4 GB` 物理显存熔断保护，防止 Windows 共享内存分页导致测速失真。
- **4090 资产及时拉回**：4090 `K0–K8` 任何阶段跑完即拉回本地 `G:\WEAVE\runs\v26\gpu4090\` 并校验 SHA256，防止远程算力释放丢失数据。
