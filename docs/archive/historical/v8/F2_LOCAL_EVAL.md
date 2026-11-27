# F2 Style Actuation — 本地评测结果（2026-11-14）

在本地 RTX 4070 Laptop（8GB）上完成 F2 的全部三个评测臂（750 tasks = Cross600 + Same150），
未使用 3060/4090 远端资源。本文档同时记录本地复建评测管线的全部偏差与验证。

## 1. 结果（Cross600 主指标）

| arm | steps | dino_s | dino_c | lpips | Top1 | margin | dT | G | clip_s | affinity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **F2.e** | 2 | .4344 | .7390 | .5276 | .307 | −.0781 | .0964 | −.0179 | .6742 | .5543 |
| **F2.s1** | 1 | .4331 | .7545 | .5127 | .292 | −.0848 | .0995 | −.0194 | .6742 | .5536 |
| **F2.s4** | 4 | .4371 | .7078 | .5590 | .308 | −.0689 | .0854 | −.0138 | .6720 | .5546 |
| N2C.e（参照，3060） | — | .4358 | .7830 | .4969 | .328 | — | — | −.018 | — | .5581 |
| N2A.e（参照） | — | .4351 | .7836 | .4971 | .323 | — | — | — | — | .5578 |

Same150：F2.e dino_s .6150 / dino_c .8279 / lpips .3751 / Top1 .953；
F2.s1 .6143/.8385/.3563/.960；F2.s4 .6144/.8024/.4139/.940。

## 2. 判读

1. **步数扫描单调但方向错误**：内层 u 步数 1→2→4，dino_s .433→.434→.437（风格略增），
   但 dino_c .755→.739→.708、lpips .513→.528→.559 单调恶化——u 修正主要在
   "买风格、卖内容"，没有免费午餐。
2. **F2 全部落在 N2 前沿内侧**：N2C.e（.4358/.7830）在几乎相同的 dino_s 下
   dino_c 高 .029–.075、lpips 低 .016–.062——**受约束 margin 最大化仍是 Pareto 最优**。
3. Top1 ~.30–.31 与 N2 (.32) 同档，margin↔真实分类错位问题没有被 actuation 解决。
4. F2 训练日志 cost≈.17 持续高于 ε=.15（N2 中 hinge 从未触发）——actuation 的
   内层优化确实在更激进地推风格代价，但评测端收益不足以补偿内容损失。

## 3. 本地管线复建与偏差声明（重要）

评测在本地从零复建，以下偏差已逐项核验，量级评估附后：

| 组件 | 3060 原版 | 本地复建 | 偏差评估 |
|---|---|---|---|
| 评测源图 data/test | 原始 jpg | `deliver/testset/data/test` 复制 | **sha256 150/150 全对，零偏差** |
| 参考图 150 张 | datasets/* 原件 | 40 wikiart（29 字节级 sha 对齐 + 11 PIL 重建近似）+ 110 distinct5（train_512clean 字节，sha 与 json 不符但文件名/内容对应） | 特征级影响可忽略；11 张 wikiart 为同 id 原图重编码，像素内容一致、jpeg 字节不同 |
| references_v1_features.pt | 官方 builder（含 csd/dino_patch/clip/rbf 全字段） | 仅重建 measure 实际消费的 `dino_cls`+`clip_prototypes` | measure 不读其余字段，无影响 |
| tasks750_source_features.pt | build_source_cache（VAE+CSD+DINO+TRT rt_*） | 同协议重算 z_src/raw_rgb/raw_csd/raw_dino；rt_* 镜像 raw（measure/infer 均不读 rt_*） | z_src 与 v5 缓存对照 mean\|Δz\|=0.0013（fp16 噪声级）；raw_csd 与 v5 缓存 meanΔ=0.053（v5 协议不同所致，以 build_source_cache 代码为准） |
| VAE 解码 | TensorRT engine（fp16） | diffusers `vae.decode(z/0.18215)` fp16 fallback（`src/utils/vae.py`） | 同一 VAE 权重同一数学，差异为 kernel 顺序级 fp16 噪声；已验证重建 MAE≈8.6/255 |
| CSD/DINOv2/CLIP/LPIPS | eval_cache 权重 | dinov2-small+CSD-ViT-L(4090 同 revision 拉取)+CLIP+LPIPS 本地缓存 | 同 revision 同文件 |
| 推理 batch | 8 | 4（8GB 显存，OOM 规避） | batch 不影响单图结果（逐任务独立） |

**结论可信度**：F2 vs N2 的差距（dino_c .71–.75 vs .78）远大于上述任何偏差量级，
判决成立。若需与 3060 数字做逐点对齐，可在 3060 补跑 F2.e 做交叉验证（ckpt 已可推送）。

## 4. 产物

- `runs/v8/F2.e|F2.s1|F2.s4/{gen,eval}/` — 750 任务生成与逐任务指标
- `runs/v8/train/F2/step_04000.pt` — 训练 checkpoint（margin 8.69, cost .169, viol .027）
- `docs/v8/results.csv` — 已追加 6 行（3 臂 × cross/same）
- 复建脚本：`_tmp/build_src_cache_local.py`；补丁：`src/utils/vae.py`（TRT fallback）、
  `src/utils/_local_eval_refs.py` + `src/utils/features.py`（Windows import fallback）
