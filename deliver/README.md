# V6 下一阶段 · 交付包

**日期**：2026-11-10
**协议**：`paper_rgb_v1` 评分链路 / 750 任务 = 600 cross + 150 same / 150 源图 / 每风格 120·120

---

## 目录

| 路径 | 内容 | 大小 |
|---|---|---|
| `table/V6_RESULTS_TABLE.md` | **统一结果表**（每行标注来源） | 3 KB |
| `table/BOTTLENECK.md` | **瓶颈判定 + Pareto 说明 + 下一步** | 4 KB |
| `pareto_style_content.png` | **风格–内容 Pareto 图** | 100 KB |
| **`checkpoints/`** | **14 个 checkpoint（33 MB），SHA256 全部校验，本地加载前向全部通过** | **33 MB** |
| `verify_checkpoints.py` | 校验脚本：重建模型 + CPU 前向 | 4 KB |
| `docs/` | 阶段报告 12 篇 + 静默失败记录 4 篇 | 150 KB |
| `compact/` | 聚合指标（无逐任务行） | 75 KB |
| `full/` | 原始 JSON（含逐任务行） | 5.4 MB |
| `make_pareto.py` | 由 `full/` 重绘 Pareto 图 | 5 KB |

---

## Checkpoints（14 个，33 MB，已校验）

`deliver/checkpoints/` 保留原始的 `runs/...` 目录结构。`MANIFEST.json` 记录每个文件的 SHA256 前 16 位与字节数。

| Checkpoint | 参数 | 机制（存在 ckpt 的 `args` 里） | 用途 |
|---|---:|---|---|
| `Q2a_embed_style.pt` | 218K | `rank=0.001 embed=0.3` | **Oracle 的 Direct 部分；蒸馏教师** |
| `r1_r1.pt` | 816K | R1 critic（width 64） | **Oracle 的梯度来源** |
| `P1_B3_distill20.pt` | 218K | `distill=20` | **推荐工作点**（Cross600 20.8% / DINO-C 0.810） |
| `P1_B4_distill50.pt` | 218K | `distill=50` | 更强风格（24.2% / 0.796） |
| `P1_B0_ctrl218.pt` | 218K | 无辅助损失 | B3/B4 的对照 |
| `P1_B1_distill218.pt` | 218K | `distill=1` | λ 过小的证据 |
| `P1_B2c_ctrl845.pt` | 845K | 无辅助损失 | 等宽对照（证明容量不是瓶颈） |
| `P1_B2_distill845.pt` | 845K | `distill=1` | 等宽蒸馏 |
| `P2_G0_bf16.pt` | 218K | 无辅助损失 | P2 对照 |
| `P2_G1a_bf16.pt` | 218K | `rank=0.001` | 弱 ranking |
| `P2_G1b_bf16.pt` | 218K | `rank=0.003` | 强 ranking |
| `L11_k1024_dim16.pt` | 218K | 无辅助损失 | 原始轻量基线 |
| `L8_k1024_dim32.pt` | 845K | 无辅助损失 | 宽度对照 |
| `score_diff_net.pt` | 3.27M | ScoreDiff 24/12 步 | 强风格参照 |

**Oracle 必须同时用 `Q2a_embed_style.pt` + `r1_r1.pt`** —— 它不是独立模型：
先跑 Q2a 得到 `z_out`，再对冻结 R1 求一次 `∇_z margin_target`，沿该方向走 RMS 0.25。

**校验方式**：

```bash
python deliver/verify_checkpoints.py
# 14/14 OK: 每个都从 ckpt 的 args 重建模型结构、加载权重、CPU 前向，
# 并打印机制参数（rank / embed / distill / trust）
```


---

## 先看这三个

1. **[table/V6_RESULTS_TABLE.md](table/V6_RESULTS_TABLE.md)** —— 8 行实测 + 3 行外部参照
2. **[table/BOTTLENECK.md](table/BOTTLENECK.md)** —— 瓶颈判定与四条候选解释的排除过程
3. **[pareto_style_content.png](pareto_style_content.png)** —— 前沿一眼可见

---

## 关键数字

| 模型 | Cross600 Top-1 | DINO-C | 说明 |
|---|---:|---:|---|
| Q2a Direct | 6.7% | 0.879 | 轻量单步基线 |
| **Oracle 1 步** | **36.5%** | 0.710 | 逐图优化上限（需 2 个 ckpt） |
| ScoreDiff N=24 | 35.2% | 0.592 | 强风格参照 |
| ScoreDiff N=12 | 30.8% | 0.624 | 内容更均衡 |
| **B3 蒸馏 λ=20** | **20.8%** | **0.810** | 单步、218K、**推荐工作点** |
| B4 蒸馏 λ=50 | 24.2% | 0.796 | 更强但开始过冲 |

**瓶颈判定：监督/训练目标的权重不足 —— 不是容量（218K 达 20.8% > 845K 的 7.5%）。**

---

## 三条必须同时声明的限制

1. **Minimalism 误判率不是 0%** —— Direct ≈ 26%、Oracle ≈ 31%、ScoreDiff ≈ 36%。
   原报告 0.000 是 `pred_class` 整数索引与字符串比较的 bug（见
   `docs/pred_class_int_vs_str.md`）。**所以 Oracle 的 36.5% 仍可能部分利用分类边界偏置。**
2. **Oracle 不是独立 218K 模型** —— 需要 `Q2a_embed_style.pt` + `r1_r1.pt`，
   推理时先跑 Q2a 再算一次冻结 R1 的 latent 梯度。
3. **Cross600 指标一律 cross-only** —— `eval_tiers` 的行里不含 `is_cross`，
   按 `tgt_style != src_style` 分离。用默认值会把 750 全算成 cross，Top-1 虚高 3.4 倍。

---

## 复现

```bash
bash sbflow/scripts/run_p0_amortize.sh      # P0 五变体（A0-A4）
bash sbflow/scripts/run_p1_distill.sh       # P1 B0/B1/B2c/B2 (λ=1)
bash sbflow/scripts/sweep_distill_lambda.sh # λ 标定 + 拟合诊断
bash sbflow/scripts/run_p1b_lambda.sh       # P1 B3/B4 (λ=20/50)
bash sbflow/scripts/run_p1_750.sh           # 750 全协议
bash sbflow/scripts/run_750_scorediff12.sh  # ScoreDiff N=12
bash sbflow/scripts/gen_delivery_table.sh   # 重新生成本表
python deliver/make_pareto.py               # 重绘 Pareto
```

**权重守卫**：每个新机制臂训练后与对照比较 `max|dW|`，为 0 则 `exit 1`。
本轮 7 项比较全部 > 0（1.5e-03 ~ 2.2e-01，113/113 张量）。
