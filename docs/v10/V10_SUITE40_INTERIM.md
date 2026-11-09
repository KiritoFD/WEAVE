# suite40 中期进展（执行书第 6 节对照口径）

> 生成时间：2026-11-09 13:47 CST。核心 9 组串行运行中；本报告只记录已完成组的
> Quick200 结果，不对未完成组作结论。统一 4000 等效步预算、SD-VAE
> `stabilityai/sd-vae-ft-ema`（`4×64×64`，sf=0.18215）、15,545 张五风格并集。

## 一、已完成组

| ID | 配置 | 状态 | Top-1 | Margin | ΔT | ΔS | SWD | DINO-C | 训练秒 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 01 | patch8 / B512 / mixed，统一主基线 | done | 0.1000 | -0.1307 | -0.0351 | -0.3821 | 0.0245 | 0.6974 | 655 |
| 04 | patch4 / B256 / mixed，token 64→256 | done | 0.1100 | -0.1169 | -0.0284 | -0.3919 | 0.0250 | — | 2345 |
| 08 | single velocity + EDM | training | — | — | — | — | — | — | — |

## 二、执行书点名的三个困难目标（01 → 04）

| 目标风格 | Top-1 | CSD ΔT | SWD | 判断 |
|---|---:|---:|---:|---|
| Rococo | 0.000 → 0.000 | -0.1342 → **-0.1197** | 0.0295 → 0.0298 | 仅轻微改善，仍完全失败 |
| Early_Renaissance | 0.050 → 0.025 | -0.0800 → -0.0794 | 0.0280 → 0.0285 | 无实质改善 |
| Ukiyo_e | 0.000 → 0.000 | -0.0323 → **-0.0387** | 0.0270 → 0.0272 | 反而略差 |

易目标：

| 目标风格 | Top-1 | CSD ΔT | 判断 |
|---|---:|---:|---|
| Impressionism | 0.325 → 0.350 | +0.0546 → +0.0565 | 小幅提升 |
| Minimalism | 0.125 → 0.175 | +0.0166 → **+0.0392** | 提升最明显 |

**结论（中期）**：patch4 / 256 token 让整体 Top-1 +1pp、Margin +0.014、ΔT +0.0067，
主要来自 Minimalism 与 Impressionism；Rococo / Early_Renaissance / Ukiyo-e 依旧
ΔT 为负、Top-1 近零。**因此 04 目前不能认定为突破**，符合执行书"不能因为
Impressionism 提高就宣布整体突破"的要求。

ΔT 为正的方向数：01 = 8/20，04 = 8/20，未增加。

## 三、工程状态核验

- `exp01/model_equiv1000/2000/3000/4000.pt` 全部为完整可恢复 payload，
  D/E/F 的共享前缀比较成立。
- 前 50 步实测速度：01 `0.18 s/update`，04 `0.30 s/update`；
  每组曝光 4,096,000 端点 / 每对 409,600，固定 4000 等效 LR 周期。
- 每组 state 已补齐 `git_sha`、`vae`、`scaling_factor`、`latent_shape`、
  `seed`、`batch`、`patch`、`updates`、`per_pair_exposure`。
- 三个 tmux 会话：`suite40`（核心 9 组）、`suite40_continue`（剩余 31 组）、
  `suite40_full750`（40 组终态后自动选 3 个不同机制 + 01 做官方 Full750）。
- 当前无失败组；CPU 打分与 GPU 训练按规定重叠（CPU 线程限制为 4）。

## 四、后续判定规则不变

最终晋级按执行书优先级：20 方向宏平均 Margin / ΔT 最高，
Rococo / Early_Renaissance / Ukiyo-e 的改善最高，其次五目标 SWD/MMD，
DINO-C / LPIPS 仅作结构破坏约束，Bridge MSE 只作稳定性参考。
Full750 的 150 同风格严格 Identity 必须单独列出，不能并入 600 跨风格成绩。
