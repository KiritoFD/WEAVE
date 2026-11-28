# F0：最终实验协议、资产核验与仓库同步审计

**审计日期**：2026-11-28  
**仓库基线状态**：`origin/main` 停留于 `6ce19e4`；最新 `3060` 蒸馏（`TRT/enc20/enc10/P3 CFG`）、`V36 Reference E-Core` 与 `V37` 完整报告已同步至本机工作区并完成文档落盘（详见 [蒸馏实测总报告](../optimization/DISTILLATION_RESULTS.md) 与 [V37_RESULTS.md](../../one-shot/V37_RESULTS.md)）。

---

## 1. 三套冻结模型契约与 NFE 核对

| 维度 | `CLEAVE-E` | `CLEAVE-P/E` | `CLEAVE-One` |
|---|---|---|---|
| 生成核参数 | `11.13M` (`ECore_NoDINO`) | `11.13M` E + `44.54M` P (`44M` 主干) | `45.22M` (`FoundationPCoreNet`) + `0.94M` (`StyleMLPEncoder`) |
| 编码 / 解码 | `A0` (`34.16M`) + `ConvNeXt` (`12.35M`) + `S10B` (`10.66M`) | 同左 + 免训练 `A3` 起点重构 | 同左 + 免训练 `Reference-A3` (`I2` 亮度+色度) |
| Flow 积分与实际评估数 | `1-NFE`（单步显式偏移） | 名义 6 步含首步 `midpoint`：**`7×4B + 1×E = 29` 次**条件/无条件评估 | `Euler 首步 + AB2`：**6 次**网络批量调用，共 **18 个** CFG 分支评估（无额外 E 草稿） |
| 风格先验与数据隔离 | Core-5 闭集原型 | Core-5 闭集原型 | `Seen-Train` 76,319 张（131 派），**Core-5 严格未见**，**Zero-DINO** |
| 解码输入契约 | `S10B` / `S10` 统一解码；repo 存储 scaled latent 经 `decode_s10` 自动 `/ 0.18215` 且 `post_quant_conv` 仅用一次 | 同左 | 同左 |

> **跨硬件/协议禁令**：RTX 3060 端到端暖机测速（`optimization/RESULTS.md` & `DISTILLATION_RESULTS.md`）与历史 4090 `P8` 吞吐严禁与另一套协议的质量读数直接混拼成新点；所有论文表格由单一权威 JSON（待 F8 固化）统一生成。

---

## 2. 外部基线有效样本审计（`eval_protocol/baseline_audit_strict.json`）

| 外部基线 | 有效总数 | `Cross` | `Same` | 缺失数 | 主前沿纳入决策 |
|---|---:|---:|---:|---:|---|
| `StyleGallery` (`eval_stylegallery.json`) | **750** | 600 | 150 | 0 | **纳入**（重型 Open-Ref 标杆） |
| `AdaIN / AesPA-Net / CUT / Identity / SaMAM / SaMST / SD-Turbo / StyleID / StyTr2 / WCT` | **750** | 600 | 150 | 0 | **纳入**（全量完整） |
| `StyleShot` | **745** | 596 | 149 | **5** | 纳入主前沿并**显式披露缺失 5 项**（`n_cross=596`） |
| `Seedream 4.5` | **720** | 576 | 144 | **30** | 纳入主前沿并**显式披露缺失 30 项**（`n_cross=576`） |
| `StyleAligned` | 15 | 12 | 3 | 735 | **排除出主前沿**（仅作样本覆盖不足注记） |
| `ZStar` | 15 | 12 | 3 | 735 | **排除出主前沿** |

---

## 3. F1–F7 资产就绪度与唯一阻塞项核查

| 阶段 | 核查对象 | 本地 / 远端可用状态 |
|---|---|---|
| **F1 (`Artist-Holdout-750`)** | 完整 WikiArt 图像与元数据 | ✅ `datasets/wikiart_new_512/`、`datasets/distinct5_512/`、`4090:/root/WEAVE_v36` 及 `eval_protocol/` 就位；需在正式抽样后核查与 `Seen-Train` / 原型库画家零重叠 |
| **F2 (`独立 VLM 盲测`)** | VLM 推理后端 | ⚠️ **待用户选择**：本机环境变量与 `4090` HF 缓存目前**无可用 API Key（`OPENAI/ANTHROPIC/GEMINI/DASHSCOPE/OPENROUTER` 均为 `False`），亦未下载本地多模态模型（如 `Qwen2.5-VL-7B-Instruct`）** |
| **F3 (`One-shot 实例控制`)** | `V36 P+A3`、SG-25、反事实对比 | ✅ `V36` 权重（`v36_main_p2_final.pt`）、`exp/v36_art_foundation/`、`V37` 反事实控制均已就位 |
| **F4 (`Decoder / Encoder 机理`)** | 三解码器 + `enc20` 传播 | ✅ 原版 SD-VAE (`deliver/checkpoints/sd0_vae` & `G:/eval_cache/hf/hub/models--stabilityai--sd-vae-ft-ema`)、`S10` (`s10_p5_lowlr/decoder.pt`)、`StyleVAE` (`C_edit_best.pt`, `C_edit_decoder_only.pt`) 均在 `deliver/checkpoints/` 就位；`enc20` 6 档数据已齐 |
| **F5 (`E/P 交界连续性`)** | `V35 Final` 生成链 | ✅ `exp/v35_final/`、`4090` 与 `3060` 均就位 |
| **F6 (`Photo-to-Art`)** | 固定 `P2A-256` 基准 | ✅ `docs/baseline/P2A-256/jpg/`（含 13 个基线输出）与 `state/dino/p2a_test_renamed/` 就位 |
| **F7 (`性能定型`)** | TRT / PyTorch / Stiefel | ✅ 3060 与 4090 均空闲就位；`A0 TRT FP16` 已完测，待补 `S10B TRT`、`A0+S10B TRT`、`E→P TRT` 与 Newton-Schulz Gram 误差统计 |
