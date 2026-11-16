# docs/performance — 统一性能台账与可视化

单一入口：`python docs/performance/build_metrics.py`（可重复运行，随 V17 队列落地自动扩表）。

## 文件

| 文件 | 内容 |
|---|---|
| `metrics.json` / `metrics.csv` | 全部方法/臂的统一台账：method、family、**protocol**、decoder、dino_s/dino_c/csd_top1/csd_margin/lpips/clip_s/style_affinity、**pareto** 旗标、provenance（原始评测文件路径） |
| `pareto_*.png` | 每协议一张双联散点（CSD Top-1 vs LPIPS；DINO-S vs LPIPS），红星 = 四轴非支配前沿成员，蓝圆 = WEAVE 系，灰叉 = 基线 |
| `images/<method>/` | 前沿方法的 750 任务生成图（512px 无损 PNG，task_id 与评测 rows 键一致） |

## 当前条目（43 条，2026-11-17 起）

1. **32 个 baseline/oracle 方法**（`deliver/q2a_oracle_scores_baseline_protocol.json`，q2a_oracle_cross600 协议）
2. **V16 九臂**（runs/v16/measured，B/C 家族，cross600 均值重算）
3. **V15 Group A 十六臂**（runs/v15/eval rows → cross600）
4. **V10 锚点**（LD4 Top-1 .3517 / E4 DINO-S .4415，q2a_750_merged 口径）
5. **V17 臂**（队列每完成一个评测臂自动计入）

## 协议警告（不要跨协议比较）

- `q2a_oracle_cross600`：CSD 裁判 + S10 解码 + 固定配对噪声（V15/V16/V17 现行口径）
- `q2a_750_merged`：V10 时代 dev+test 合并 750
- 旧 paper_rgb_v1（evaluator_v2 / TRT 解码）数字见 `docs/baseline/RANKING.md`，**与本台账不可直接混排**，仅存档参考

## 帕累托前沿（q2a_oracle_cross600 四轴：DINO-S↑ DINO-C↑ CSD-Top1↑ LPIPS↓）

非支配成员（`pareto=true`）由脚本实时计算——含 StyleShot/StyleID/AesPA 等外部基线与 WEAVE 系 V16_B_adapter / V16_B_teacher / V16_C_h2_strong 等；V15 Group A 各容量臂均被支配（容量不换风格特异的定量证据）。重复运行脚本时前沿随 V17 臂落地自动更新。

## 图像库存（images/，2026-11-17 起全量）

| 目录 | 张数 | 说明 |
|---|---:|---|
| `baselines_D5_512/jpg/<方法>/` | 17 方法 × 750 | 全部基线/历史臂 750 输出（styleshot/styleid/stytr2/seedream/adain/aespa/cut/wct/samst/samam/sdturbo/stylealigned/identity/zstar + 我们的 weave/weave_oriented_e4/weave_yfix），自 `docs/baseline/D5-512/jpg` 整体挪入 |
| `WEAVE_published_reval/` | 750 | WEAVE published 原版 paper 重评图（自 archives 710_baseline_weave_paper_reval 挪入）|
| `V15_W_dim96_conv_base/` | 2250（t0=0/0.4/0.7 × 750）| 3060 payload → PNG（无损）|
| `B_adapter/`（V16 学生，0.2565M）| 750 | 3060 runs/v16/raw → PNG（无损）|
| 其余前沿臂 | 待补 | 3060 `runs/v16/raw/` 按 `exp.v17.encode_v16_png --arms <名>` 编码后 tar 拉回；V17 臂随队列落地同样入库 |

注意：`baselines_D5_512` 为 paper_rgb_v1 时代（evaluator_v2/TRT 口径）的 750 输出，与 `images/V15_*`（S10 口径）分属不同协议，仅供人工对照与重算，不跨协议混排。

## 重建

```bash
python docs/performance/build_metrics.py        # 台账 + 帕累托 + 图
# 新臂图像：3060 上编码 → tar 拉回 docs/performance/images/
```
