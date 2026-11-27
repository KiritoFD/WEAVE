# V8 文档导航

**状态（2026-11-13）：代码已实现并在 3060 串行队列中执行；正式臂尚无一条完成 750，暂无方法学结论。**

V8 实际执行的是 **A/B/C/I 四路线矩阵**（生成器内控制、学习梯度几何、真实画作支持能量、联合臂），代码在 `src/` + `exp/v8/`，队列入口 `sbflow/scripts/run_v8_queue.sh`。

| 顺序 | 文档 | 看什么 |
|---|---|---|
| 0 | [实施与运行记录](V8_IMPLEMENTATION_AND_RUN.md) | **当前事实源**：代码结构、接口、臂矩阵、队列状态、已知缺口 |
| 1 | [实验结果记录](exp.md) · [F2 本地复核](F2_LOCAL_EVAL.md) | `N2C` / `F2` 实测数据与对比图表 |

*(注：设计阶段未落地的 121 项 H0–H6 草案 `WEAVE_V8_24H_FULL_MATRIX_PLAN.md`、`V8_24H_RESEARCH_PLAN.md`、`V8_IMPLEMENTATION_SPEC.md`、`V8_TEST_ACCEPTANCE.md`、`V8_MATRIX_AND_RUNBOOK.md` 已统一归档至 `docs/archive/`。)*

## 计划与实现的关系（重要）

`docs/v8/` 的四份实施文档（2–5）描述的是**设计契约**：H0–H6、M0–M8 能量、G0–G12 几何归因、T0–T17 内容约束、D0–D6 学生、121 个 750 注册项、36000 教师任务。

实际落地时路线收敛为更小的 **A（GCE 控制坐标）/ B（LGG SPD 算子）/ C（RSDE 真实支持能量）/ I（联合）** 矩阵：6 个训练臂（14500 updates）+ 28 个推理臂，patch-bank OT/几何匹配路线归档在 `tools/v8_ot_geometry/`（固定权重平方距离退化为均值回归，实验证伪）。两者的差异与当前队列的真实覆盖范围见 [实施与运行记录](<V8_IMPLEMENTATION_AND_RUN.md>)。

## 执行决定（沿用设计文档的有效部分）

> **⚠ 2026-11-14 修订**：下列第 2 条已被 **AGENTS.md 决策协议（2026-11-14）作废**——
> 实测 S10 在同协议下全面更优（CSD Top-1 .3767 vs .3450，bootstrap CI 排除 0；LPIPS 亦胜），
> **一切评测/出图解码一律用 S10**，"正式质量用标准 SD-VAE 口径"的说法不再成立。
> 证据见 [docs/HIGHLIGHTS.md](<../HIGHLIGHTS.md>) 与 `deliver/q2a_oracle_scores_baseline_protocol.json`。

1. 科学 NO 不取消独立臂；实现/资源错误才 BLOCKED 或 FAILED，队列继续。
2. ~~正式质量用标准 SD-VAE 口径；S10 仅作独立部署对照。~~（已作废，见上方修订）
3. 24h 持久 deadline 不因重启归零；不输出假的 ALL_COMPLETE。
4. 历史运行目录（`runs/v6*`、`runs/v7*`、旧 `tools/v8` DSBM 结果）只读。

## 下一步

按 [实施与运行记录](<V8_IMPLEMENTATION_AND_RUN.md>) 的队列推进；全部臂完成后生成 `docs/v8/V8_24H_RESULTS.md`（只在有真实结果时创建）。
