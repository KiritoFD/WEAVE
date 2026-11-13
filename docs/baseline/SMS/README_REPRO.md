# SMS / StyleSSP / OmniStyle：复现可行性登记（2026-11-13）

这三个是 **P2/P3 候选**，仅登记现状与门槛，未做 12GB smoke。StyleGallery + AD 是当前 P1 主线。

## SMS（ShowLab, https://github.com/showlab/SMS）

- **任务形态**：Stable Diffusion 1.5 微调 + **test-time attention/store 优化**；论文设置多为 512×512、几十步去噪。
- **代码**：开源仓含训练与推理脚本；推理可直接加载其发布的微调 checkpoint（需确认权重是否随仓发布或需自训）。
- **12GB 风险**：微调版 SD1.5 推理本身轻（~5-8GB），风险在其 **store/attention bank** 机制占额外显存；若官方默认开高分辨率或多图 bank 需降档。
- **门槛**：需确认 (a) 官方推理 checkpoint 是否公开；(b) train.py 依赖的 prompt/标注管线；(c) 750 任务 manifest 的 prompt 字段——我们的任务是 **image-driven style transfer**，SMS 原生是 text-driven，需要确认其是否支持 image-reference 模式，否则适配成本高。
- **结论**：先核对 README 的 inference 接口是否接受 content+style 双图输入，再决定是否立项。

## StyleSSP（ByteDance, https://github.com/bytedance/StyleSSP）

- **任务形态**：基于 SD/SDXL 的风格迁移，含 projector/adapter 结构；不同版本权重差异大。
- **12GB 风险**：若基于 **SDXL**（base ~6.9GB fp16）+ adapter + 优化，12GB 很紧；若仅 SD1.5 版本则可行。
- **门槛**：必须先确认仓库提供的是 SD1.5 还是 SDXL 权重；SDXL 版本在 3060 上基本不可行（除非作者提供蒸馏小模型）。
- **结论**：查 README 权重表 → 若 SD1.5-only 则按 AD 同款三层流程立项；若 SDXL 则标记 BLOCKED-on-12GB。

## OmniStyle（https://github.com/wangkaiye/OmniStyle 或官方仓）

- **任务形态**：多风格统一 adapter（一次训练多画派），推理为单次前向+adapter——**比 SG/AD 便宜**。
- **12GB 风险低**：adapter 推理 ≈ SD1.5 推理 + 小模块，预计 6-8GB。
- **门槛**：需找官方代码与权重（开源状态需核实）；若权重在 HuggingFace 需下载。
- **结论**：优先级在 SG/AD 之后；若官方权重可得，是最省事的补充 baseline。

## 统一要求（所有 P2/P3 baseline）

1. 固定 commit hash + 权重版本写入 `execution.jsonl` provenance；
2. smoke 只验证可运行性，**质量结论以 750-task 指标为准**；
3. reduced 配置（降步数/降分辨率/关 checkpointing）必须显式标注，禁止与官方结果混排；
4. 若官方实现依赖 >12GB 显存且无等价降级路径 → 记 BLOCKED，不用 substitute 结果冒充。
