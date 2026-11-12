# 产物、代码入口与复现边界

## 机器与工作目录

| 侧 | 工作目录 | 主要对象 |
|---|---|---|
| 本地Windows | G:/WEAVE | Git源码、文档、deliver聚合与部分权重 |
| 3060 Windows/WSL Ubuntu22.04 | /mnt/i/Latent_Style/WEAVE（Windows I:/Latent_Style/WEAVE） | WEAVE训练、评测、R2、TRT |
| 4090 Linux | /root/Workspace/WEAVE_VAE | VAE/Decoder、swap评测、ArtEncoder与reader |

同名runs目录在不同机器不等价。远端完整路径、文件hash与聚合数据见附录快照；本地来源hash见source_inventory.csv。

## 本地交付包


**证据来源：** [deliver/checkpoints/INDEX.md](../../../deliver/checkpoints/INDEX.md)。

### 原报告表 1：checkpoints/ 索引

| 子目录 | 工作流 | 文件数 | 大小 | 清单 |
|---|---|---:|---:|---|
| `runs/v6/`、`runs/v6_next/` | **V6 风格迁移（本轮）** | **14** | **33.2 MB** | [`MANIFEST.json`](../../../deliver/checkpoints/MANIFEST.json) |
| `runs/vae_style_vae/`、`runs/vae_decoder_prune/` | VAE / Decoder（另一工作流，从 4090 拉回） | 4 | 293.2 MB | [`README_VAE_DECODERS.md`](../../../deliver/checkpoints/README_VAE_DECODERS.md) |

### 原报告表 2：V6 风格迁移（14 个，33.2 MB）

| Checkpoint | 参数 | 机制（在 ckpt 的 `args` 里） | 用途 |
|---|---:|---|---|
| `runs/v6/Q2a_embed_style/Q2a_embed_style.pt` | 217,824 | `rank=0.001 embed=0.3` | **Oracle 的 Direct 部分；蒸馏教师** |
| `runs/v6_next/probes/r1_r1.pt` | 815,941 | R1 critic（width 64） | **Oracle 的梯度来源** |
| `runs/v6/P1_B3_distill20/P1_B3_distill20.pt` | 217,824 | `distill=20` | **推荐工作点**（Cross600 20.8% / DINO-C 0.810） |
| `runs/v6/P1_B4_distill50/P1_B4_distill50.pt` | 217,824 | `distill=50` | 更强风格（24.2% / 0.796） |
| `runs/v6/P1_B0_ctrl218/P1_B0_ctrl218.pt` | 217,824 | 无辅助损失 | B3/B4 的对照 |
| `runs/v6/P1_B1_distill218/P1_B1_distill218.pt` | 217,824 | `distill=1` | λ 过小的证据 |
| `runs/v6/P1_B2c_ctrl845/P1_B2c_ctrl845.pt` | 845,008 | 无辅助损失 | **等宽对照（排除容量解释）** |
| `runs/v6/P1_B2_distill845/P1_B2_distill845.pt` | 845,008 | `distill=1` | 等宽蒸馏 |
| `runs/v6/P2_G0_bf16/P2_G0_bf16.pt` | 217,824 | 无辅助损失 | P2 对照 |
| `runs/v6/P2_G1a_bf16/P2_G1a_bf16.pt` | 217,824 | `rank=0.001` | 弱 ranking |
| `runs/v6/P2_G1b_bf16/P2_G1b_bf16.pt` | 217,824 | `rank=0.003` | 强 ranking |
| `runs/v6/L11_k1024_dim16/L11_k1024_dim16.pt` | 217,824 | 无辅助损失 | 原始轻量基线 |
| `runs/v6/L8_k1024_dim32/L8_k1024_dim32.pt` | 845,008 | 无辅助损失 | 宽度对照 |
| `runs/v6/score_diff/score_diff_net.pt` | 3,274,180 | ScoreDiff 24/12 步 | 强风格参照 |

WEAVE14个约33.2MB、VAE工作流4个约293.2MB，共约326MB；均是已有交付包记录，不是本次重新下载或重复前向验证。
`.pt`被gitignore排除，clone仓库不等于取得权重。4个VAE权重也不包含一切外部基础模型依赖：原始SD-VAE、A0完整Encoder、CSD/DINO/CLIP/LPIPS缓存等仍需相应机器的模型缓存。
原README“只靠checkpoint+testset即可推理评测”须在这些依赖也存在的前提下理解。Canonicalizer/参考缓存也要核对，不能只拿生成器单文件。


**证据来源：** [deliver/README.md](../../../deliver/README.md)；[deliver/checkpoints/README_VAE_DECODERS.md](../../../deliver/checkpoints/README_VAE_DECODERS.md)；[deliver/testset/MANIFEST.json](../../../deliver/testset/MANIFEST.json)。

## 推理组成与参数计数

- Q2a Oracle：217,824参数Direct + 815,941参数R1 critic + 逐图latent梯度；不是独立218K纯前向模型。
- B3/B4：217,824参数生成器；VAE编码/解码及条件参考统计另计。
- ScoreDiff：3,274,180参数主体，12/24步；迭代次数与VAE解码成本单列。
- S10：10.658M Decoder；A0原Decoder49.490M；不代表完整VAE仅10.658M。

## 代码与运行入口

| 用途 | 入口 |
|---|---|
| V6 Direct/ScoreDiff与消融 | [tools/v6_24h/eval_tiers.py](../../../tools/v6_24h/eval_tiers.py) |
| R2统一评测 | [tools/v6_next/r2_eval.py](../../../tools/v6_next/r2_eval.py) |
| R2串行调度 | [sbflow/scripts/run_r2_12h.sh](../../../sbflow/scripts/run_r2_12h.sh) |
| 20格换VAE | [tools/vae_swap_eval/run.py](../../../tools/vae_swap_eval/run.py) |
| DINO/CLIP补评 | [tools/vae_swap_eval/style_compare.py](../../../tools/vae_swap_eval/style_compare.py) |
| 配对归因/全因子 | [tools/vae_swap_eval/attribute.py](../../../tools/vae_swap_eval/attribute.py) |
| S10实际阶段与损失 | [tools/vae_suite/decoder_train.py](../../../tools/vae_suite/decoder_train.py) |
| 联合StyleVAE训练 | [tools/vae_style_vae/train.py](../../../tools/vae_style_vae/train.py) |
| ArtEncoder训练 | [tools/art_vae/train.py](../../../tools/art_vae/train.py) |
| 局部读取器训练/评测 | [tools/local_reader/run.py](../../../tools/local_reader/run.py) |

这些入口用于定位实现，不是建议不加参数直接重跑。复现时先读取脚本CLI、原checkpoint args、训练/任务/参考缓存hash和本章协议；运行旧eval_tiers会复现旧链路，不能冒充修正协议。

## 数值产物入口


**证据来源：** [deliver/vae_ranked_eval/results.json](../../../deliver/vae_ranked_eval/results.json)；[deliver/vae_ranked_eval/style_comparison.json](../../../deliver/vae_ranked_eval/style_comparison.json)；[deliver/vae_ranked_eval/attribution.json](../../../deliver/vae_ranked_eval/attribution.json)；[deliver/vae_ranked_eval/vae_swap_compat.json](../../../deliver/vae_ranked_eval/vae_swap_compat.json)；[eval_protocol/tasks_750.json](../../../eval_protocol/tasks_750.json)；[eval_protocol/references_v1.json](../../../eval_protocol/references_v1.json)。

本复盘直接从JSON生成20格均值、差分CI、same150、逐目标与2×2效应，避免手工转录。旧报告表保留来源和协议警示。
源码/文档哈希只是本次快照证据，不自动证明训练时源码完全相同；没有保存当时commit/args的旧实验，复现性应按已有记录有限声明。

## 本次工作未执行的计算

没有新训练、没有重新生成模型输出、没有运行一轮额外750评测；只读两边文件/进程和已有JSON，再生成文档。未启动周期自动化，未改任何现有实验任务。
已有测试通过记录只作为历史证据引用，本次不重复模型单测；只校验本目录文件、链接、行数、覆盖和表格转录。
