# V8 第三轮队列断点记录（2026-11-13 ~22:30）

队列 `v8qf`（`sbflow/scripts/run_v8_queue_f.sh`）已按要求手动停止，为 StyleGallery
12GB 复现让出 GPU。队列是断点续跑设计：重启同一脚本会跳过所有 `state.json=DONE` 的臂。

## 训练臂状态（runs/v8/train/）

| 臂 | 状态 | 最后 checkpoint | 备注 |
|---|---|---|---|
| A3 | DONE | step_03000 | 修复后重训完成（内容条件 policy） |
| A4 | DONE | step_03000 | 修复后重训完成（1+tanh(g) 解死锁） |
| L3 | DONE | step_02000 | 修复后重训完成（SPD U 非零 bias） |
| L4 | DONE | step_02000 | 修复后重训完成 |
| C3 | DONE | step_02500 | 修复后重训完成（sid 标签） |
| N2A/B/C | DONE | step_04000 | 上一轮已完成，结果已并入 results.csv |
| F0 | DONE | step_04000 | train_actuation steps_k=0 |
| F1 | DONE | step_04000 | train_actuation steps_k=1 |
| **F2** | **中断** | **无 checkpoint** | 步数约 800–1000/4000 时被杀；首个 ckpt 在 step_2000，**无可恢复权重，重启从 0 训** |
| F3 | 未开始 | — | 需要 style_targets（167MB） |

## 待跑评测臂（队列 eval 段，全部未开始）

重跑臂：A3.e A4.e L3.e L4.e I0.e I2.e I3.e C1.e C2.e C3o.e C3o.r15 C3n.e
新臂：F0.e F1.e F2.e F3.e F2.s1 F2.s2 F2.s4（以 `exp/v8/arms.py` F_EVAL 为准）

## 恢复方法

```bash
# 3060 上
cd /mnt/i/Latent_Style/WEAVE
tmux new-session -d -s v8qf 'bash sbflow/scripts/run_v8_queue_f.sh'
```

脚本会自动跳过 11 个 DONE 训练臂，从 F2 重新训练，然后 F3 + 全部评测。

## 复现侧已就绪（不占训练结果）

- StyleGallery 源码：`/mnt/i/Latent_Style/StyleGallery_src/StyleGallery-main`（zip 解压 + 8 处 12GB 补丁已应用并编译通过）
- 权重：`pretrained_models/` 下 runwayml_stable-diffusion-v1-5、facebook/dinov2-base、lcm-lora-sd1.5 均已软链到 `/mnt/i/Latent_Style/weights/`
- SD1.5 实际只需 diffusers 组件 ~5GB；45GB 中大部分是原始 ckpt/safetensors 重复权重，后续可清理
- AttentionDistillation 仓库克隆仍在 tmux `clones` 会话走镜像重试
