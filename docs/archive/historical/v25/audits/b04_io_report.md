# B04：CPU/I/O 性能诊断报告（本地工作机 Windows）

- 任务：B04（docs/19_pull_together/agent2.md §四），模式 `full`，总耗时 11.6s，生成于 2026-11-19T19:34:20.913134+00:00
- 环境：Windows-11-10.0.26300-SP0 / Python 3.12.0 / 32 线程 / numpy 2.3.5 / Pillow 12.2.0 / torch 2.13.0+cu132 (cuda_device_count=0)
- 约束遵守：全程 `CUDA_VISIBLE_DEVICES=""`、`OMP/MKL/OPENBLAS_NUM_THREADS=2`；只写临时目录（本机 C:\），峰值写入 97.2MB（<200MB 预算）；无 ssh、无 CUDA、未修改任何现有文件、未读取任何 runs/ 大文件内容（交付前仅对臂元数据做过 meta.json 读取与 stat 标定）

口径：warm = 复用状态/同名文件覆盖重跑；cold = 全新文件名/全新路径首触。Windows 用户态无法冲洗 OS 写缓存，cold 是"首触"口径而非落盘冲洗口径，共享 SSD 页缓存下两者差异通常很小。每项取 warm/cold 各 ≥3 次（quick 模式各 2 次）的中位数。所有含"估算"的数字均为线性外推，不是实测端到端。

## 1. M1 750 任务 JSON 序列化/反序列化
合成 750 行（镜像 src/eval/measure.py:90-101 行 schema），compact JSON 373 KB（真实 eval.json stat 参考 897.0 KB），indent2 587.9 KB。
| 指标 | warm中位(ms) | cold中位(ms) | warm范围 | cold范围 | 样本 |
| --- | --- | --- | --- | --- | --- |
dumps compact（纯 CPU） | 4.098 | 4.157 | 4.053~4.432 | 3.953~4.26 | n=3+3
dumps indent2（纯 CPU） | 15.785 | 16.508 | 15.379~19.178 | 15.649~18.533 | n=3+3
write compact 落盘 | 5.274 | 5.844 | 5.23~5.392 | 5.806~7.561 | n=3+3
write indent2 原子（tmp+replace） | 19.456 | 22.325 | 18.752~21.903 | 20.209~24.426 | n=3+3
read+parse 整文件 | 2.76 | 3.062 | 2.655~2.792 | 2.877~3.445 | n=3+3

## 2. M2 PNG 编码与写盘（10 张 512² uint8 均匀噪声）
单图 PNG 大小：l1=811KB, l6=770KB（噪声=PNG 最坏情形；真艺术图更可压缩，此处吞吐是保守下界）。
| 指标 | warm中位(ms) | cold中位(ms) | 每图warm(ms) | 每图cold(ms) |
| --- | --- | --- | --- | --- |
仅编码 l1（raw_io 默认） | 122.621 | 120.984 | 12.262 | 12.098
仅编码 l6（PIL 默认） | 213.738 | 210.687 | 21.374 | 21.069
编码+写盘 串行 | 164.013 | 190.088 | 16.401 | 19.009
编码+写盘 8线程（镜像 save_png_batch_fast） | 43.098 | 47.465 | 4.31 | 4.747

外推（估算）：750 张 l1 + 8 线程落盘 ≈ **3.23s**（warm 中位）。

## 3. M3 sha256 吞吐（100MB 合成内存缓冲）
| 分块 | warm 中位(ms) | warm GB/s | cold GB/s |
| --- | --- | --- | --- |
chunk_1mb | 50.145 | 1.947 | 1.738
chunk_4mb | 46.862 | 2.084 | 2.078
single_shot | 47.979 | 2.035 | 1.921

外推（估算）：30MB checkpoint 全量校验 ≈ **0.015s**（1MB 分块；4MB 分块无显著增益，runtime.sha256_file 的 1MB 分块无需调整）。

## 4. M4 JSONL 注册表：逐行追加 vs 整体重写（对接 B03）
记录格式与 src/v25/artifact_registry.py::ArtifactRecord.to_dict 同构，单行 468B × 100 行。
| 指标 | warm中位(ms) | cold中位(ms) | warm范围 | cold范围 | 样本 |
| --- | --- | --- | --- | --- | --- |
逐行追加+fsync（镜像 _append_json） | 86.513 | 82.167 | 84.908~91.957 | 80.075~84.976 | n=3+3
逐行追加 无fsync | 34.16 | 35.024 | 33.937~34.772 | 32.193~35.294 | n=3+3
整体重写（原子 tmp+replace） | 2.781 | 2.56 | 2.116~3.597 | 2.413~2.958 | n=3+3
全文件逐行解析（records() 等价） | 0.534 | 0.669 | 0.527~0.785 | 0.613~0.697 | n=3+3

真实 ArtifactRegistry 端到端（registry 规模 7 条）：
| 指标 | 中位(ms) | 范围 | 说明 |
| --- | --- | --- | --- |
| register 同路径 warm | 39.23 | 32.288~39.532 | 含每次 git rev-parse 子进程 |
| register 同路径(免git探测) | 1.814 | 1.618~2.2 | 设 WEAVE_GIT_SHA 后纯 fsync 下界 |
| register 新路径 cold | 34.431 | 31.067~36.062 | registry 持续增长的真实形态 |
| validate(known) | 6.71 | 6.197~8.544 | 含 lookup 全文件解析 |
| validate(unknown) | 5.367 | 4.887~5.561 | 无记录不报错不入日志 |

**B03 对接数值**：高频 register 单次成本 ≈ **39.23ms**（默认，含 git rev-parse 子进程）/ **1.814ms**（设 WEAVE_GIT_SHA 后），吞吐上限 ≈ **551 次/秒**（免探测口径）；裸逐行追加+fsync = **0.8651ms/行**，无 fsync **0.3416ms/行**，fsync 开销倍数 ≈ 2.5x；100 条整体重写 **2.781ms**、全量解析 **0.534ms**。注意：ArtifactRegistry.lookup 每次 records() 全文件 O(n) 解析，注册表到 1k 条时单次 validate 的解析成本按线性外推约 5ms，高频调用建议进程内缓存记录索引。

## 5. M5 小文件 vs 大文件 + 目录创建
| 指标 | warm中位(ms) | cold中位(ms) | 摊薄 |
| --- | --- | --- | --- |
创建 100×1KB | 43.343 | 61.554 | 433.4µs/文件
读 100×1KB | 9.099 | 9.362 | 91.0µs/文件
写 1×100KB | 0.643 | 1.448 | 等量内容单文件
读 1×100KB | 0.206 | 0.152 | 等量内容单文件
mkdir 嵌套链(parents) | 2.344 | 2.707 | 单链
mkdir exist_ok×50 | 1.012 | 1.179 | 20.2µs/次

## 6. M6 Windows 路径处理（本机口径；不含 GPU 机器）
| 正确性断言 | 结果 | 说明 |
| --- | --- | --- |
slash_equivalence | PASS | PurePath 归一化等价
case_insensitive_stat | PASS | stat 大小写不敏感
long_path_over_260 | PASS | 路径长 357 字符读写成功
dotdot_resolve | PASS | dotdot resolve 收敛
normcase | PASS | normcase 斜杠+小写归一
relative_vs_absolute_anchor | PASS | 相对路径锚定 REPO 后与绝对路径一致

| 开销项 | µs/操作 | 样本数 |
| --- | --- | --- |
path_resolve_existing | 216.93 | 2000
path_stat_existing | 4.54 | 4000
path_exists | 3.41 | 4000
normcase_str | 0.74 | 10000
path_construct_join | 4.8 | 10000
open_close_existing | 46.41 | 1000
str_to_path_roundtrip | 5.21 | 10000

## 7. M7 torch 原始 batch 载荷往返（镜像 src/infer/run.py 载荷）
载荷 3.54MB（batch=4：3×(4,64,64)fp16 latent + (4,3,512,512)uint8 rgb），torch 2.13.0+cu132，CUDA 可见设备数 = 0（CUDA_VISIBLE_DEVICES=""，隔离确认）。
| 指标 | warm中位(ms) | cold中位(ms) | warm MB/s | cold MB/s |
| --- | --- | --- | --- | --- |
torch.save(tmp+replace) | 6.043 | 5.886 | 585.8 | 601.4
torch.load | 1.926 | 2.062 | 1838.0 | 1716.8

## 8. 单臂评测 ~50s 中的非 GPU 开销归因（全部为估算）

代码锚点（CPU/I/O 环节在评测流水线中的位置）：

| 环节 | 代码位置 | 触发条件 | 估算耗时/臂 |
| --- | --- | --- | --- |
原始 batch torch.load（~738MB） | src/eval/measure.py:62 | 每臂必经 ~188 个 .pt | 0.4s
PNG 编码+写盘 750 张(8线程,l1) | src/eval/measure.py:69 + src/utils/raw_io.py:13-33 | 仅 write_png=True（runner.run_measure_subprocess 默认 False，src/eval/runner.py:56）| 3.23s
eval.json dumps+写盘（~0.9MB） | src/eval/measure.py:109-112 | 每臂必经 | 0.023s
features.pt torch.save（~5.9MB） | src/eval/measure.py:113-115 | 每臂必经 | 0.01s
原始 batch torch.save（~738MB, infer 阶段） | src/infer/run.py:75-77 | 每臂生成期一次 | 1.26s
checkpoint sha256 校验(30MB) | src/utils/runtime.py:39-44 / src/v25/artifact_registry.py:143-152 | 仅 B03 hash 复核开启 | 0.015s
registry register（含 git 探测） | src/v25/artifact_registry.py:360-400 | 仅 B03 接入后 | 39.23ms/次
目录 mkdir(parents,exist_ok) | src/eval/measure.py:40-42, src/utils/raw_io.py:23 | 每臂 1-2 次 | <0.01s（可忽略）

汇总（估算）：单臂 750 任务评测里，纯 CPU/I/O 项合计 ≈ **4.92s（PNG 开）** / **1.69s（PNG 关）**，占 ~50s 臂墙钟的 **9.8% / 3.4%**。其余为 GPU 推理与特征裁判前向（DINO/CSD/CLIP/LPIPS）。

**结论（评测流水线里哪些耗时根本不是 GPU 推理）**：

1. **原始 batch 的 torch.save（infer 落盘）与 torch.load（measure 读回）是最大的非 GPU 项**：~738MB/臂往返，估算 1.26s + 0.4s；每臂必付的固定 I/O 税，与模型快慢无关，改用更强 GPU 也不会缩短它。
2. **PNG 编码是 write_png=True 场景下的第二大头**：750 张噪声上界 ≈ 3.23s（8 线程把单图 4.31ms 摊薄）；正式口径 runner 默认 --no-png（src/eval/runner.py:56,69-70），此时为 0；真艺术图比噪声更可压缩，实际编码时间低于此上界。
3. **JSON 写盘本身可忽略**：~0.9MB eval.json 序列化+落盘 0.023s，summary_arm.json（~3KB，src/eval/runner.py:75）微秒级；measure.py:109 用的是非原子 write_text，若 B06 要原子化，save_json 的 indent=2 会让体积翻倍、dumps 多花约 3.9x CPU，绝对值仍 <0.1s，代价可接受。
4. **哈希校验是条件项**：sha256 吞吐 1.947GB/s（1MB 分块；4MB 分块 2.084GB/s 无显著增益），30MB checkpoint 校验 ≈ 0.015s；只有 B03 注册表启用 artifact_sha256 复核（validate check_artifact_hash=True）时才发生，常规缓存闸门（params_hash 等小字符串比对）是纯内存操作。
5. **目录创建与路径处理均不是瓶颈**：mkdir exist_ok 20.2µs/次、Path.resolve 216.93µs/次，每臂万次级路径操作也 <0.1s；Windows 长路径(>260字符)/大小写/正反斜杠实测全部无坑（见 §6 断言表全 PASS）。
6. **Windows 上"大量小文件"是真实的反模式**：逐个创建 1KB 文件 ≈ 433.4µs/文件（100 个共 43.343ms），同量 100KB 打包单文件写+读仅 ≈ 0.643ms，按字节差约 67x；RawWriter 的 ~4MB batch .pt 分片（src/utils/raw_io.py:60-70）粒度合理，但任何"逐任务级"小文件落盘（逐任务 json/元数据）都应打包或走 JSONL 批量追加，不要逐文件创建。
7. **B03 高频 register 的真实成本由 git 子进程主导**：register 含 git rev-parse 探测 39.23ms/次（若 git 不可用回退 unknown 不抛错），设 WEAVE_GIT_SHA 后降至 1.814ms/次——fsync 本身只占 0.8651ms/行，高频注册建议批量登记或进程内复用 git_sha。

## 9. 风险与限制

- cold = 全新文件名/路径首触；Windows 用户态无法冲洗 OS 写缓存，非落盘冲洗口径。
- PNG 用均匀噪声（最坏情形），真艺术图更可压缩：PNG 编码吞吐为保守下界，写盘字节量为上界。
- 真实臂标定：runs/v8/F2.e/gen/meta.json（750 任务，batch=4，decode=True）stat 得原始载荷 737.8MB / 188 个 batch .pt；runs/v25/measured/A_p1_l130/eval.json stat 897KB。标定只读 meta.json 小文件与 stat，未读任何 .pt 大文件内容；运行期完全不访问 runs/。
- 单臂占比的分母 ~50s 为总控给定基线，具体阶段组成以 A07 端到端成本模型为准；本报告只量化 CPU/I/O 项的绝对秒数。
- GPU 训练在远程 4090/3060，本机测量不与其抢磁盘；测量期间本机无其他重载任务时数值最可靠。
- M4 真实 register 成本包含每次 git rev-parse 子进程（WEAVE_GIT_SHA 未设时的默认行为，git 失败回退 unknown 不抛错）；两项口径均已测出。
