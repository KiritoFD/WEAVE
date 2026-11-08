# V9：Blog-Style Bidirectional Brownian Bridge Matching（实施记录）

> 依据：用户提供的 HTML 文章《【生成模型】薛定谔流：玩鸣潮的同时玩原神》及其复刻代码包
> `weave_blog_sbflow_repro.zip`（参考实现说明见 `E:\README.md`）。
>
> **本轮定位**：这是对文章**公开描述算法**的工程复刻，不是作者私有代码/权重的逐参数复现。
> 文章未公开网络宽度/深度、DCAE checkpoint、样本量、ε、OT 比例等，本仓库使用的是文档化重建默认值。

---

## 一、与 V8 的关键区别（为什么必须重做，而不是改 V8）

用户在复核中指出：**V8 B3 并不是博客算法的等价实现**。此前 P1 长训练建立在错误的训练目标上，已终止。

| 组件 | 现有 V8 | 博客方法（V9 实现） |
|---|---|---|
| 训练目标 | 单一速度场 `E[z1-z0 \| z_t]` | **共享网络同时学习正/反两个条件 SDE 漂移** |
| 随机路径 | CFM 直线 / 高斯插值 | **布朗桥** `z_t=(1-t)z0+t z1+sqrt(ε t(1-t))ξ` |
| 推理 | 直接积分预测速度 | **积分 `(b⁺-b⁻)/2` 概率流 ODE** |
| 网络 | ~2M Conv U-Net | **32.6M DiT，AdaLN-Zero，时间+方向条件** |
| 预条件 | 无 | **修正版 EDM，含 OT 端点相关性矩估计** |
| 潜空间 | SD-VAE 4×64×64 | **DCAE f32c32，32×16×16** |
| OT 特征 | C1 latent 结构特征 | **RGB 32×32 缩略图（或可选 DINO 语义）** |
| 风格监督 | 隐式目标域 | **隐式目标域，无任何额外 style loss** |

结论：V8 的 B0–B3 结果**不能**用于判断博客方法是否有效。

---

## 二、代码落地

`sbflow/` 已复制到 WEAVE 仓库根目录（与参考包同构，未改动数学）：

| 文件 | 功能 |
|---|---|
| `sbflow/bridge.py` | 布朗桥单时刻采样 + 双向漂移监督目标 |
| `sbflow/preconditioning.py` | 修正 EDM 系数，支持 OT 相关性（矩估计 `mu0,mu1,var0,var1,cov01`） |
| `sbflow/ot.py` | 对数域 Sinkhorn + 行条件采样 + OT/随机混合配对 |
| `sbflow/cache.py` | 冻结 DCAE 编码 + RGB 缩略图（可选 DINO）缓存 |
| `sbflow/network.py` | 共享双向 DiT，AdaLN-Zero，方向 embedding |
| `sbflow/train.py` | 双向漂移平衡 MSE、梯度累积、EMA warm-up、checkpoint |
| `sbflow/infer.py` | 双向概率流 ODE（Euler/Heun） |
| `sbflow/preflight.py` | DCAE 重建一致性 + Test600 泄漏审计 |
| `sbflow/eval_test600.py` | 冻结 Test600 子集生成 + 风格评测 + DCAE no-op 基线 |
| `sbflow/merge_test600.py` | 十个双域模型汇总到完整 600 |
| `sbflow/tests/` | 数学单测 + 真实 DCAE 检查 |

新增运行脚本（`sbflow/scripts/`）：`cache_two_domains.sh`、`run_preflight.sh`、
`bench_train.sh`、`train_pair.sh`、`eval_pair.sh`。

---

## 三、已验证事实（本机实测，非引用）

### 3.1 数学单测（本地 CPU 通过）

```
PASS test_bridge                      # (b+ - b-)/2 == 路径导数；时间反演一致
PASS test_edm_corrected_closed_form    # 矩系数在独立单位方差时退化为修正闭式
PASS test_edm_empirical_correlated     # 相关性端点下 residual 均值≈0、方差≈1
PASS test_ot                           # Sinkhorn 双边缘误差 < 1e-5
PASS test_network                      # DiT 形状 + 零初始化输出
PASS deterministic solver smoke
PASS analytic Gaussian bidirectional marginal transport
```

最后一项是核心：**零网络**下，正确的 EDM 残差表示 + `(b⁺-b⁻)/2` PFODE
能把一个高斯边缘完整搬运到另一个高斯边缘（双向），说明推理方程本身是对的。

### 3.2 真实 DCAE（3060 实测）

- checkpoint：`mit-han-lab/dc-ae-f32c32-mix-1.0-diffusers`，**1.23 GB**，
  由 3060 的 Windows 侧下载（WSL 无外网），落在
  `I:\Latent_Style\weights\dc-ae-f32c32-mix-1.0-diffusers`，
  WSL 侧路径 `/mnt/i/Latent_Style/weights/dc-ae-f32c32-mix-1.0-diffusers`。
- 参数量 **323.4M**，`scaling_factor = 0.4552`。
- 真实 WikiArt 图 512→**32×16×16**，往返 PSNR **20.15–26.84 dB**，峰值显存 1.5 GB。

### 3.3 数据与泄漏审计（3060 实测）

- 两域缓存：`Impressionism` 1000 张、`Ukiyo_e` 1000 张，均为 `32×16×16`。
- `sbflow.preflight` 输出：
  - `frozen Test600 SHA/path audit PASS; 2000 training images checked`
  - 缓存 latent 与当前 AE 重编码一致性 MSE 1e-5 ~ 3e-4
  - `REAL DCAE preflight PASS`

### 3.4 训练吞吐（3060 实测）

- 参数量 **32,635,136**（与参考说明一致）。
- 60 次优化更新墙钟 **31.9 s**（含 ~13 s 启动），即 **≈0.31 s/update**；
  8000 步实际约 50 分钟。
- 首个 step 的 `loss≈1.006`，`f≈1.006`、`b≈1.033`：符合"归一化残差目标均值 0、
  方差 1，网络零初始化"的预期，是正确性的直接证据。
- Sinkhorn 行边缘误差稳定在 **1.5e-8 ~ 2.2e-8**。

---

## 四、实验矩阵（按 README 建议的 style-first 顺序）

| 编号 | 配置 | 目的 |
|---|---|---|
| A | `ot-ratio=0.0`, `eps=0.04` | 纯随机配对的博客基础版 |
| B | `ot-ratio=0.5`, `eps=0.04`, RGB 缩略图 | 加 mini-OT 配对 |

两者共享 seed=42、8000 updates、effective batch 64、cosine+warmup。
评测对象是冻结 Test600 中 **Impressionism ↔ Ukiyo_e** 的 **60 个有序跨风格任务**
（30 源图 × 2 方向），并**同时**报告 **DCAE no-op** 基线（不是 SD-VAE no-op）。

> 边界：单个双域模型只覆盖 600 中的 60 项。要覆盖全部 600 需训练 10 个无序风格对
> 模型，见 `sbflow/merge_test600.py`。本轮**不**把 60 项结果说成完整 Test600。

---

## 五、待补：结果

（见 `docs/v9/V9_RESULTS.md`，训练与评测完成后写入。）
