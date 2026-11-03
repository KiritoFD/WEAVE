# 03 实验时间线

> 口径提醒：2026-06~07 的 `clip_style/lpips/WFI` 都是 **750 all-pairs**（150 内容 × 5 目标），不是 2026-11 的 600 跨风格口径。数字跨阶段不可直接横比，详见 [04_RESULTS.md](04_RESULTS.md)。

## 1 一句话谱系

```
Cycle-NCE / Latent AdaCUT（2-5 月，历史大指标面）
  → SchrödingerBridge D0→D2 前沿（5 月）
  → 616 TopoGate + 620 Spatial Bridge（6 月上）
  → 622-625 FC-SB（6 月中，645+ 实验）
  → 627 clip 0.7348 天花板 → 628 HH 判死 → 630 精简+mask
  → 710 四指标协议 → 713 target-HF 定向残差 + 论文表（7 月）
  → 2026-11 下一代 A/C/D（进行中）
```

## 2 分阶段记录

| 时间 | 试了什么 | 结果 | 为什么失败/保留 | 证据 |
|---|---|---|---|---|
| 2026-02~05 | Cycle-NCE / Latent AdaCUT 大规模风格迁移；Phase A-C 考古 | 历史大指标面，metrics/summary/ref cache 保留 | 不是当前主线，只作早期方向证据 | `EXPERIMENT_ARCHAEOLOGY/ARCHAEOLOGY_CURRENT_STATUS_AND_CONCLUSIONS_ZH_20260605.md#L88-L96` |
| 2026-05-16 | D0→D2 破坏性消融（terminal SWD / kinetic / micro-HF SWD） | D0 clip 0.7014；去 SWD 0.6708；去 kinetic 0.7159 但 LPIPS 0.6375 崩；SaMST 0.7194 | **terminal SWD 必需；kinetic 是风格-内容主旋钮**；去掉即内容崩 | `EXPERIMENT_LOG.md#L14-L29` |
| 2026-06-17（616） | TopoGate 内生 attention + vertical FM + 禁用 pairing cache | 结构保持 LPIPS≈0.31 解决；style 卡在 **<0.72** | 根因：确定性 ODE 收敛到 E[X\|c]="平均笔触"；PureLatentSpatial tokenizer 零 ROI（白耗 1.2GB） | `docs/archive/616/README.md#L9-L35` |
| 2026-06-18（618） | 读 StyleGallery/CSGO/SCSA/StyleShot | 确认任务差异：我们是 `f(content, style_id)`，无参考图 | 参考图路径缺失是结构性问题；IDT CLIP-S 低=风格间距大=更难 | `docs/archive/618/summary.md#L9-L25` |
| 2026-06-19~22（619/620） | Spatial Bridge：DINOv2+LoRA→256 memory tokens；条件源消融 | `style_condition_source=target_dino_patches` 使 WFI 0.61-0.64；`latent` 过门；推荐配置 WFI 0.3757 / CLIP 0.6995 / LPIPS 0.3422 | **条件源选择是白化主因**；dim=128 无收益；edge loss 弊大于利 | `docs/archive/620/fog/ablation_audit/final_report.md#L16-L25` |
| 2026-06-23~26（622/625） | FC-SB 六个月、645+ 实验 | I7 基线 clip 0.7017 / LPIPS 0.3625 / WFI 0.3757；卡在 0.70/0.30-0.36 保守吸引子 | 训练目标偏好"少注入 style"：gate collapse 0.048、endpoint α=0.16、GroupNorm 白化 | `docs/archive/625/grand_summary.md#L6-L23` |
| 2026-06-27（627） | Phase 4 融合突破 35+ 推理消融 + 6 训练 | clip 触顶 **0.7348**；LPIPS 0.3403 只能单独达成 | 判定该架构 clip 物理天花板；无法同时 clip>0.74 & LPIPS<0.35 | `docs/archive/627/phase4_final_report.md#L7-L19` |
| 2026-06-28（628） | I1-I10 推理消融 + T1-T8 训练消融 | T5 ep7 all-pairs clip 0.7307 / LPIPS 0.3403；I5-I9 全"无效果"（gate≈0.3 未学到）；去 AdaIN LPIPS +0.048；avg_pool 替 DWT +0.047 | **AdaIN 与 DWT 是内容保持关键**；fiber/triband 推理侧惰性；HH 在 L8 判 DEAD | `docs/archive/628/ablation_conclusions.md#L3-L37` |
| 2026-06-30（630） | 代码精简 + style mask；10 epoch 从零训练 | 删 11346 行死代码；mask 0.5/random 最佳；clip 0.7288 / LPIPS 0.3369 | mask 是当时唯一稳定增益；确立"禁止借旧 ckpt"规则 | `docs/archive/630/HANDOVER.md#L13-L19` |
| 2026-07-02（72） | FC-SB 全文档 + T11/T18/T19 | T11 本地 Pareto 最佳 clip 0.7213 / LPIPS 0.2868；SaMam 真实 0.5816/0.2434（v4 的 0.7175 是编造值） | 明确 HH 删除；CLIP-S 领先 SaMam +0.14，训练快 ~141× | `docs/archive/72/README.md#L6-L9` |
| 2026-07-10（710） | 固定四指标协议（DINO-S/CLIP-S/DINO-C/LPIPS），B0 重训 | 历史 750 包 0.4874/0.7292/0.7688/0.3239 但 **ckpt 已丢**；B0 = 903K、10ep、8-step Heun | 停用 MUSIQ 做模型选择；禁止后处理指标替代原生结果 | `docs/archive/710/README.md#L14-L40` |
| 2026-07-13（713） | target-HF 条件路径探针：pooled / spatial / subband / texture 等 20+ 变体 | 安全最优 `subband` DINO-S 0.4886；spatial 0.4901 但 DINO-C 崩到 0.404；`nomem`/`memres`/`scale` 均不优 | **失败在条件路径而非监督目标**；spatial 泄漏布局；放大残差只换来 off-style | `docs/713/EXPERIMENT_SUMMARY_FOR_METHOD_AND_NEXT_PLAN.md#L9-L56` |
| 2026-07-15 | oriented-HF 从零 15 epoch，逐 epoch 评测 | ep4 = 0.4915 / 0.7126 / 0.8103 / 0.2596，成为论文主表 ckpt；AdaIN 2.0 封顶，2.5 崩 | 风格上限 +0.005，内容代价可接受；2.5 不再探索 | `docs/713/SUBMISSION_HANDOFF_2026-07-15.md#L144-L156` |
| 2026-11-03 | 同口径重评 + 路线 A 输出监督（500 步 smoke） | 基线 600：DINO-S 0.3977 / CLIP 0.6425 / DINO-C 0.7939 / LPIPS 0.3053；A 500 步 0.3938 / 0.6461 / 0.8258 / 0.2722 | 500 步不足以改变风格（−0.004），仅证明链路可跑；显存固定 batch=1 | `Exp.md#L8-L23` |

## 3 反复踩过的坑（不要重试）

1. **靠 scalar 放大残差/风格强度**：628 与 713 都验证，只升 off-style，all-pairs DINO-S 反而降。
2. **raw spatial target HF 注入**：DINO-S/CLIP 上升但 DINO-C 崩（713 表 spatial 行）。
3. **推理侧调 fiber/triband/gate**：628 I5-I9 全零效果，因为训练时 gate 没学到。
4. **AdaIN scale 2.5**：713 明确"severe collapse"。
5. **PureLatentSpatial tokenizer / dim=128 扩容**：616、620 均判零收益。
6. **增强 HH 头**：628 L8 判 DEAD；713 也确认 HH 低能量、相位敏感、与感知风格梯度弱对齐。
7. **avg_pool 替代 DWT、去掉 AdaIN**：LPIPS 各 +0.047/+0.048，直接回退。

## 4 当前 A/C/D 状态（2026-11-03）

- 路线 A：机制打通，500 步无收益；扫描矩阵未开始（`Plan.md#L29-L48`）。
- 路线 C：依赖 DINOv2 patch cache（已生成 content_5k + refs_test）与 18k latent（**缺**），未开始（`Plan.md#L50-L58`）。
- 路线 D：VAE 重建上限探针未跑（`Plan.md#L60-L66`）。
- P6 同口径重评：进行中（`runs/nextgen_eval_full`，`Plan.md#L27`）。
