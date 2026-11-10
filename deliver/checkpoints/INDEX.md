# checkpoints/ 索引

**本目录同时存放两个互不相关的工作流的权重。** 按子目录区分，不要混用。

| 子目录 | 工作流 | 文件数 | 大小 | 清单 |
|---|---|---:|---:|---|
| `runs/v6/`、`runs/v6_next/` | **V6 风格迁移（本轮）** | **14** | **33.2 MB** | [`MANIFEST.json`](MANIFEST.json) |
| `runs/vae_style_vae/`、`runs/vae_decoder_prune/` | VAE / Decoder（另一工作流，从 4090 拉回） | 4 | 293.2 MB | [`README_VAE_DECODERS.md`](README_VAE_DECODERS.md) |

**两套权重没有任何依赖关系**，不要因为同在一个目录就配对使用。

---

## V6 风格迁移（14 个，33.2 MB）

`MANIFEST.json` 记录每个文件的 SHA256 前 16 位与字节数。

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

**Oracle 必须同时用 `Q2a_embed_style.pt` + `r1_r1.pt`** —— 它不是独立模型：
先跑 Q2a 得 `z_out`，再对冻结 R1 求一次 `∇_z margin_target`，沿该方向走 RMS 0.25。

**`P1_B2c_ctrl845` 必须与 `P1_B2_distill845` 一起用** —— 它是等宽无蒸馏对照，
单独给 B2 就无法支撑"容量不是瓶颈"这个结论。

校验：`python deliver/verify_checkpoints.py`（14/14 通过）

---

## 注意：`.pt` 被 `.gitignore` 排除

仓库的 `.gitignore` 里有 `*.pt`，所以**这些权重不在 git 里**，只存在于本地磁盘。
交付时需要拷贝 `deliver/` 目录本身，不能只 `git clone`。
