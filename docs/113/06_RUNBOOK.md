# 06 运行手册

> 2026-11-03 实测。**另一会话正在 3060 上跑训练/评测**：除 `docs/113/` 外，不要写远端任何文件。

## 1 环境现状（实测）

| 项 | 值 |
|---|---|
| 连接 | 本地 PowerShell → `ssh 3060` → `wsl.exe -d Ubuntu-22.04 -u root -- bash -s` |
| 主机 | Windows 10.0.19045，默认 shell cmd.exe |
| WSL | Ubuntu-22.04，`BasePath=I:\wsl\Ubuntu-22.04`，默认 root |
| GPU | RTX 3060 12GB（sm_86），驱动 581.08；当前 **已用 7743 MiB**，可用约 4.5G |
| Python | `/usr/bin/python3` = 3.10.12，**无 venv** |
| torch | 2.10.0+cu128，torchvision 0.25.0+cu128，CUDA 可用 |
| 内存/磁盘 | 15G 物理（约 12G available）；`/mnt/i` 剩 **93G**（80% 已用） |
| tmux | 已安装，当前无会话 |

远端目录：`/mnt/i/Latent_Style/WEAVE`（Windows `I:\Latent_Style\WEAVE`），**不是 git 仓库**。

## 2 权重（离线）

| 权重 | 本地路径 |
|---|---|
| DINOv2-base | `/mnt/i/Latent_Style/weights/dinov2-base` |
| CLIP ViT-L/14 | `/mnt/i/Latent_Style/weights/clip-vit-large-patch14`；ModelScope 缓存 `weights/AI-ModelScope__clip-vit-large-patch14` |
| DINO patch cache | `/mnt/i/Latent_Style/dino_cache/{content_5k,refs_test}`（dinov2-base, 224, last_hidden_state） |

HF 不可达，统一走 ModelScope / 本地目录；需要离线环境变量：

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MODELSCOPE_CACHE=/mnt/i/Latent_Style/weights
```

## 3 常用命令

```powershell
# 连通 + GPU
ssh 3060 "wsl.exe -d Ubuntu-22.04 -u root -- nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv"

# 单文件传输（必须 -O）
scp -O "g:\GitHub\Latent_Style\WEAVE\docs\113\01_DATA.md" "3060:I:/Latent_Style/WEAVE/docs/113/"

# 批量 md5
$s = @('md5sum /mnt/i/Latent_Style/WEAVE/docs/113/*.md','echo __END__') -join "`n"; $s | ssh 3060 "wsl.exe -d Ubuntu-22.04 -u root -- bash -s"
```

```bash
# 训练（远端）
cd /mnt/i/Latent_Style/WEAVE
tmux new -s weave            # 必须在 tmux 里，ssh 断开会杀进程
python3 run.py --config config.json

# 评测（600/150 拆分）
python3 tools/run_eval_split.py --input runs/<run>/eval/results.json \
    --reference_policy first_image_of_target_style_test_dir --out runs/<run>/eval/results_split.json
```

## 4 P0 诊断（先跑这个再谈新路线）

```bash
cd /mnt/i/Latent_Style/WEAVE
# 数据规模 5k vs 18k
python3 tools/diag_data_scale.py --base_config config_d5_adain2.json \
    --packed_dir data/train_d5/.latent_cache/packed --out runs/diag_data_scale/manifest.json
# VAE 重建上限
python3 tools/vae_recon_probe.py --style_root datasets/d5/test --vae taesd \
    --extractor vgg16 --layers relu2_2,relu3_3 --out runs/vae_recon_probe.json
# 参考图交换 / 三方对比：按各自 docstring 组装
python3 tools/diag_reference_swap.py --help
python3 tools/diag_three_way.py --help
```

> `docs/NEXTGEN_RUNBOOK.md` 里的 `data/train_d5_taesd/...` 和 `I:/datasets/...` 都是错路径：实际是 `data/train_taesd` 与 `/mnt/i/Latent_Style/WEAVE/datasets`。

## 5 易踩坑（按踩坑频率排序）

1. **`scp` 必须加 `-O`**：远端 sftp 子系统被禁用，不加会报 `subsystem request failed on channel 0`。
2. **远端不是 git 仓库**：不能 `git pull/status/diff`；同步只能 `scp -O` 覆盖，覆盖前先备份。
3. **不要在 Windows 侧跑 Python**：用 WSL `python3`；多行远端脚本用 PowerShell 数组 + `` `n `` 拼接，末尾 `echo __END__` 收尾（否则 heredoc 会挂）。
4. **HF 不可达**：不要 `from_pretrained("facebook/dinov2-base")` 直连；用 `weights/dinov2-base` 或 ModelScope。
5. **显存只剩 ~4.5G**：训练固定 `batch_size=1`；`output_supervision` 在 batch=2 时会 OOM（`Exp.md#L18-L21`）。
6. **长任务必须 tmux**：ssh 断开即杀进程；当前唯一会话不要再开第二个 GPU 任务。
7. **`docs/archive/*/.codeartsdoer/node_modules`**：历史遗留巨量文件，`rsync/scp` 同步 docs 时要排除，否则会卡死。
8. **`run.py` 的 `--config` 是相对 cwd**：必须在 `/mnt/i/Latent_Style/WEAVE` 下执行。

## 6 同步纪律

- 本地是唯一 git 源；远端是 scp 副本（无 `.git`）。
- 改代码 → 本地 commit → 再 `scp -O` 覆盖；本次整理**不动**远端代码/配置/runs。
- 文档同步后做 `md5sum` 对照，别靠肉眼。
