# WEAVE 服务器交割说明（2026-11-03）

本文件在 `I:\Latent_Style\HANDOVER.md`（WSL: `/mnt/i/Latent_Style/HANDOVER.md`）与仓库 `docs/HANDOVER.md` 各存一份。
本次会话只做了环境盘点与代码同步，**未启动任何训练/评测/解压**，也**未执行 pytest 与 P0 诊断**。

---

## 0. 一句话结论

| 项目 | 状态 |
|---|---|
| ssh + WSL + GPU | 可用（RTX 3060 12GB，驱动 581.08，torch 2.10.0+cu128，CUDA 可用） |
| 可用显存 | **仅约 2.7GB 空闲**（9.5GB 被 Windows 桌面/LM Studio/Edge 占用） |
| 代码同步 | 19 个文件已同步到 `/mnt/i/Latent_Style/WEAVE`，md5 抽样一致；远端**不是 git 仓库** |
| 数据 | `data.tar` 与现有 `data/` 完全同源，**无需解压**；5k 有 latent，18k 只有原图 |
| 依赖缺口 | 缺 `diffusers`（VAE 探针必需）、`cv2`、`timm`、`onnxruntime`、`skimage`；`pytest` 已装 9.2.0 |

---

## 1. 连接与执行方式（可直接复制）

本地是 Windows PowerShell，远程主机别名为 `3060`，默认 shell 是 **cmd.exe**（Windows 10.0.19045，用户 administrator）。
后端实际工作全部在 WSL 里做。

```powershell
# 连通性
ssh 3060 "wsl.exe -d Ubuntu-22.04 -u root -- uname -a"

# 单条命令
ssh 3060 "wsl.exe -d Ubuntu-22.04 -u root -- nvidia-smi --query-gpu=memory.used,memory.total --format=csv"

# 多行脚本：用 PowerShell 数组 + LF 拼装，末尾加 echo 收尾（见第 6 节坑 1）
$s = @(
'cd /mnt/i/Latent_Style/WEAVE',
'python3 --version',
'echo __END__'
) -join "`n"; $s | ssh 3060 "wsl.exe -d Ubuntu-22.04 -u root -- bash -s"

# 交互式进入 WSL
ssh -t 3060 "wsl.exe -d Ubuntu-22.04 -u root -- bash -l"
```

### 文件传输（重要）
远端 **sftp 子系统被禁用**，普通 `scp` 会报 `subsystem request failed on channel 0`。必须用 legacy 协议：

```powershell
scp -O "g:\GitHub\Latent_Style\WEAVE\features.py" "3060:I:/Latent_Style/WEAVE/"
```

批量：
```powershell
$root = "g:\GitHub\Latent_Style\WEAVE"
foreach ($f in @("features.py","swd.py","output_supervision.py","pseudo_targets.py","reference_encoder.py","texture_module.py","config_schema.py","flow.py","model.py","trainer.py")) { scp -O -q "$root\$f" "3060:I:/Latent_Style/WEAVE/" }
scp -O -q "$root\utils\dataset.py"                 "3060:I:/Latent_Style/WEAVE/utils/"
scp -O -q "$root\tests\test_nextgen_skeleton.py"   "3060:I:/Latent_Style/WEAVE/tests/"
scp -O -q "$root\docs\NEXTGEN_RUNBOOK.md"          "3060:I:/Latent_Style/WEAVE/docs/"
foreach ($t in @("gen_pseudo_targets.py","vae_recon_probe.py","diag_data_scale.py","diag_reference_swap.py","diag_three_way.py","run_eval_split.py")) { scp -O -q "$root\tools\$t" "3060:I:/Latent_Style/WEAVE/tools/" }
```

---

## 2. 代码位置与同步清单

- 远端仓库：`/mnt/i/Latent_Style/WEAVE`（Windows `I:\Latent_Style\WEAVE`）
- 远端 **没有 `.git`**：不能 `git pull/status/diff`，只能用 `scp -O` 覆盖文件
- 本地 HEAD：`737c809b9b84a5b99cfe368d8eecd94d9926bbf7`，本次改动均**未提交**
- 覆盖前已备份远端原版：
  `/mnt/i/Latent_Style/_diag/backup_pre_sync/weave_pre_sync_20261103_190110.tar.gz`
  （含 config_schema.py / flow.py / model.py / trainer.py / utils/dataset.py）

已同步的 19 个文件：

```
M  config_schema.py
M  flow.py
M  model.py
M  trainer.py
M  utils/dataset.py
?? features.py
?? swd.py
?? output_supervision.py
?? pseudo_targets.py
?? reference_encoder.py
?? texture_module.py
?? tests/test_nextgen_skeleton.py
?? docs/NEXTGEN_RUNBOOK.md
?? tools/gen_pseudo_targets.py
?? tools/vae_recon_probe.py
?? tools/diag_data_scale.py
?? tools/diag_reference_swap.py
?? tools/diag_three_way.py
?? tools/run_eval_split.py
```

回滚某文件：
```bash
cd /mnt/i/Latent_Style && tar -xzf _diag/backup_pre_sync/weave_pre_sync_20261103_190110.tar.gz -C WEAVE config_schema.py
```

---

## 3. 运行环境现状

| 项 | 值 |
|---|---|
| 主机 | Windows 10.0.19045（cmd 默认 shell），administrator |
| WSL | Ubuntu-22.04，`BasePath = I:\wsl\Ubuntu-22.04`，默认 root，内核 6.18.40.1-microsoft-standard-WSL2 |
| GPU | NVIDIA RTX 3060，12GB，compute capability 8.6（sm_86） |
| 驱动 | 581.08，驱动级 CUDA 13.0；torch 自带 CUDA 12.8 |
| Python | 系统 `/usr/bin/python3` = 3.10.12，**无 venv** |
| torch | 2.10.0+cu128，`cuda.is_available() == True` |
| torchvision | 0.25.0+cu128 |
| 已装 | numpy 2.2.6, scipy 1.15.3, PIL 11.3.0, transformers 5.15.0.dev0, safetensors 0.8.0, einops 0.8.2, pytest 9.2.0（本次安装） |
| 缺 | diffusers, cv2, timm, onnxruntime, skimage, torchaudio |
| 磁盘 | `I:\` 448G，已用 343G，**剩 106G** |
| 内存 | 25GB 物理，约 16GB 空闲 |

### 显存问题是当前最大约束
`nvidia-smi` 显示 **9.5GB / 12GB 已被占用**（dwm、explorer、Edge、LM Studio、Windows 侧 python.exe 等）。
训练前建议先关闭 LM Studio / Edge / 桌面占用程序，然后确认：

```powershell
ssh 3060 "wsl.exe -d Ubuntu-22.04 -u root -- nvidia-smi"
```

若仍不足，按 runbook 降 `batch_size` / `output_supervision_microbatch` / 分辨率，并记录。

---

## 4. 数据：`data.tar` 与已有数据

- 用户提到的 `data.rar` 实际不存在；目录里是 **`/mnt/i/Latent_Style/data.tar`（7.5GB，POSIX tar，未压缩）**。
- 只读比对结论：`data.tar` 内的条目与仓库现有 `data/` **同名、同大小、同时间戳**（抽查
  `data/train_d5/images_hwc_uint8.bin` = 3,932,160,000 字节、`data/train_d5/.latent_cache/packed/00_Early_Renaissance.pt` = 327,762,372 字节，与磁盘完全一致）。
- **因此没有解压，也不需要解压**（不会再占约 8GB）。若日后确认要留一份独立副本，解压到
  `/mnt/i/Latent_Style/data_extracted/`（约 8GB，当前剩余 106G 足够），不要覆盖 `WEAVE/data`。

### 真实数据路径与数量

D5（5k，与论文当前模型匹配；VAE = SD1.5 EMA，latent scale 0.18215）

| 路径 | 内容 |
|---|---|
| `WEAVE/datasets/d5/train` | 原图 5000 = 5 风格 × 1000（Early_Renaissance, Impressionism, Minimalism, Rococo, Ukiyo_e） |
| `WEAVE/datasets/distinct5_512/train` / `train_512clean` | 同上 5000（另一份清洗版） |
| `WEAVE/data/train` | **逐样本 latent `.pt`** 5000 = 5 × 1000；`.latent_cache/packed/manifest.json` 记录 1000/风格 |
| `WEAVE/data/train_d5/.latent_cache/packed` | **SD1.5 EMA fp16 packed latent**，5 × 1000 = 5000，约 1.6G（**训练默认用这个**） |
| `WEAVE/data/train_taesd/.latent_cache/packed` | TAESD latent，5 × 1000 = 5000，约 314M |
| `WEAVE/data/train_sdxl/.latent_cache/packed` | SDXL latent，5 × 1000 = 5000 |
| `WEAVE/data/test`（= `datasets/d5/test`） | 评测原图 150 = 5 × 30 |
| `WEAVE/datasets/distinct5_512/test` | 300（5 × 30 + processed_data 空目录） |
| `WEAVE/data/train_d5/images_hwc_uint8.bin` + `images_index.json` | 3.9GB 原始像素缓存（重编码用） |

18k

| 路径 | 内容 |
|---|---|
| `WEAVE/datasets/wikiart512_5style/train` | **18000 = 5 × 3600**，风格为 Expressionism, Impressionism, Post_Impressionism, Realism, Symbolism |
| `WEAVE/datasets/wikiart512_5style/test` | 150 = 5 × 30 |
| 18k 的 latent cache | **不存在**（`datasets/*latents*` 里只有 5k 的 distinct5/samam 版本、fewshot、random20、stress splits） |

> 重要：18k 与 D5 是**不同风格集合**（只有 Impressionism 重合）。原计划的“5k vs 18k 同风格对照”无法直接成立；
> 需要聚类/对齐风格，或改做“同协议不同规模”的近似对照。

其他可用集合（备选）：`datasets/r5/train` 5000（Cubism, Expressionism, Pop_Art, Romanticism, Symbolism）+ 150 test；
`datasets/wikiart_random20_512_latents_ema` 20 风格各 500（Abstract_Expressionism 890）≈ 9900 latent；
`datasets/wikiart_stress_splits_512/{wikiart_stress1,2,3}` 带 manifest。

### 600 跨风格 + 150 同风格评测清单
**不存在现成清单**。只有每风格 30 张的 test 集（150 张）。需要按
“150 内容图 × 4 个非本风格目标 = 600 跨风格 + 150 同风格”自行生成，
参考图选择策略固定写在结果里（`tools/run_eval_split.py --smoke` 里有 policy 字段）。

---

## 5. 已跑通 / 失败 / 未执行

已跑通：
- `ssh 3060`、`wsl.exe -d Ubuntu-22.04 -u root -- ...`、`nvidia-smi`（Windows 与 WSL 双侧）
- WSL 内 python/torch/torchvision CUDA 探测（capability 8.6，12GB）
- 数据集盘点脚本 `/mnt/i/Latent_Style/_diag/weave_probe_inventory.py`（只读，可重跑）
- `scp -O` 传输、base64 单文件写入 WSL、`pip3 install pytest`（9.2.0）
- `data.tar` 前 80 条 + 关键文件条目比对（只读，未解压）

失败/受限：
- 普通 `scp`：`subsystem request failed on channel 0` → 用 `scp -O`
- `scp`/`ssh` 二进制管道（`cmd /c ... | ssh`）：被本机工具策略禁止 → 用 `scp -O`
- `wsl.exe -l -v` 从 cmd 执行报“系统找不到指定的路径” → 发行版信息用注册表确认（Ubuntu-22.04）
- `nohup ... &` 后台任务：WSL VM 在最后一个会话结束时关停，进程被杀；`tmux` 未安装
- 远端无 `.git`，不能 `git pull`

未执行（交接给新会话）：
- `pytest tests/test_nextgen_skeleton.py`
- 全部 `tools/*.py --smoke`
- P0 诊断（VAE 重建探针、5k vs 18k、参考图交换、三方对比）
- 已知基线：本地全量跑过 `pytest tests` 得 2 failed / 62 passed，
  `test_710_inference_infra.py::test_decoder_only_vae_preserves_decode_scaling_contract`
  与 `test_submission_root_layout.py::test_oriented_hf_route_changes_only_matching_hf_heads`
  （后者缺 `experiments/architecture/hf_oriented_nohh.json`）为改动前既有失败，**不是新回归**。

---

## 6. 踩过的坑（务必先读）

1. PowerShell 管道给 `bash -s` 时会在**最后一行**尾部加 `\r`。末行不要放管道或敏感参数，用 `echo __END__` 收尾；已用 `-join "`n"` 拼装。
2. WSL 里 `nohup`/`&` 不会跨会话存活。先 `apt-get install -y tmux`，长任务一律 `tmux new -s <name>`。
3. 远端 `weave` 不是 git 仓库，任何“同步”都是直接覆盖文件，先备份。
4. `torch.hub`/`torchvision` 权重目录 `~/.cache/torch/hub/checkpoints` 为空 → 首次用 `vgg16` 需联网下载权重。
5. 系统 Python 3.10.12 无 venv，`pip3` 直接装全局（已有 pytest）；缺 `diffusers` 会让 `--vae taesd/sd15` 与 `build_latent_decoder('taesd')` 失败（本地 `data/taesd/{config.json,diffusion_pytorch_model.safetensors}` 已存在）。
6. `/mnt/i` 上 `du` 很慢（仓库 83G），盘点用 `find`/`ls`，别全盘 `du`。
7. 显存只剩 ~2.7GB，不先关 Windows 侧程序就跑训练必 OOM。

---

## 7. 下一步最省时的启动命令

```bash
# 进入
ssh -t 3060 "wsl.exe -d Ubuntu-22.04 -u root -- bash -l"

# 0) 长任务保护（只需一次）
apt-get update -qq && apt-get install -y -qq tmux

# 1) 依赖（VAE 探针与真实解码器必需）
pip3 install -q diffusers==0.31.0

# 2) 骨架单测（CPU 即可）
cd /mnt/i/Latent_Style/WEAVE && python3 -m pytest tests/test_nextgen_skeleton.py -q

# 3) 各 tools 冒烟（CPU，秒级）
python3 tools/diag_data_scale.py      --smoke --out /mnt/i/Latent_Style/_diag/smoke_data_scale.json
python3 tools/diag_reference_swap.py  --smoke --out /mnt/i/Latent_Style/_diag/smoke_ref_swap.json
python3 tools/diag_three_way.py       --smoke --out /mnt/i/Latent_Style/_diag/smoke_three_way.json
python3 tools/gen_pseudo_targets.py    --smoke --out /mnt/i/Latent_Style/_diag/smoke_pseudo
python3 tools/run_eval_split.py        --smoke --out /mnt/i/Latent_Style/_diag/smoke_eval_split.json

# 4) VAE 重建上限探针（真实 vgg16；首次会下载权重）
python3 tools/vae_recon_probe.py \
  --style_root /mnt/i/Latent_Style/WEAVE/data/test --vae taesd \
  --extractor vgg16 --layers relu2_2,relu3_3 --max_per_style 30 \
  --out /mnt/i/Latent_Style/_diag/vae_probe_taesd.json

# 5) 5k 数据规模诊断（D5 latent，固定更新次数 / 固定轮次两协议）
python3 tools/diag_data_scale.py \
  --base_config config.json \
  --packed_dir /mnt/i/Latent_Style/WEAVE/data/train_d5/.latent_cache/packed \
  --per_style_small 1000 --per_style_full 3600 \
  --out /mnt/i/Latent_Style/_diag/diag_data_scale_d5.json
```

注意：`--per_style_full 3600` 需要 18k latent，而 18k 目前**没有 latent cache**。
两条可行路线：
- 只用 5k：`--per_style_small 200 --per_style_full 1000`（同集合内规模梯度，最省时）；
- 或先用 `data/train_d5/encoder_b8.onnx` / SD1.5 VAE 把 `datasets/wikiart512_5style/train`
  编码成 packed latent（18k × 4×64×64，预计约 6GB 磁盘、3060 上约 20-30 分钟），再跑 full=3600。

显存不足时优先降 `--batch_size`（默认 96 → 32/16），再降分辨率。

---

## 8. 产物与临时文件位置

- 诊断输出统一放 `/mnt/i/Latent_Style/_diag/`（本次已创建）：
  - `weave_probe_inventory.py`：只读数据盘点脚本
  - `backup_pre_sync/`：同步前远端原版备份
- 未解压 `data.tar`，未创建 `data_extracted/`
- 本机临时文件：无残留（本地 `docs/HANDOVER.md` 与远端 `HANDOVER.md` 为本次唯一新增文档）
