# VAE 编码：ONNX 固定计算图 + TensorRT 优化文档

> 目标：把 SD-1.5 EMA encoder-only (fp16) 的"归一化 + 编码"做成**一个固定计算图文件**，
> 用 **onnxruntime + TensorRT EP** 运行——**免重编译、开箱即用、且比原 torch 路径更快**。
>
> 硬件基线：NVIDIA RTX 4070 Laptop GPU (8 GB)，驱动 610.74，Windows。
> 任务：把 `data/train_d5` 下 5000 张 512×512 图编码成 latent 并落盘。

---

## 1. 文件位置清单

| 文件 / 目录 | 路径 | 说明 |
|---|---|---|
| 主脚本 | `tools/fast_vae_encode_d5.py` | 所有路径（torch / onnx / trt）的实现与 CLI |
| **ONNX 固定图（核心产物）** | `data/train_d5/encoder_b8.onnx` | **单文件 68 MB，含全部权重**。固定 batch=8、fp16、uint8[N,3,512,512] → fp16 latent。这就是"固定计算图"，可直接分发给他人，无需 torch 源码 |
| TRT 引擎缓存 | `data/train_d5/trt_cache/` | TensorRT 构建出的 fp16 引擎，缓存到磁盘。**首次构建 ~6 min，之后秒加载、零重建** |
| 打包图像（uint8 内存映射） | `data/train_d5/images_hwc_uint8.bin` | 5000 张图打包成 3.93 GB 单文件，运行时用 `np.memmap` 直读，零拷贝预取 |
| 输出 latent 缓存 | `data/train_d5/.latent_cache/` | `--full` 编码全部 5000 张后的落盘结果（`packed/*.pt` + `meta.json`） |
| 项目根 | `tools/fast_vae_encode_d5.py` 的 `parent.parent`，即仓库根 `WEAVE/` | `_ROOT` |

> 注：ONNX 路径目前要求 `--use_packed`（从 `images_hwc_uint8.bin` 读取）。

---

## 2. 优化与编译方式

### 2.1 为什么走 ONNX + TRT，而不是 torch.compile / AOTInductor

| 方案 | 免重编译 | 单文件 | 速度 | 本机可行性 |
|---|---|---|---|---|
| torch `--graph --graph_compile` | ❌ 每次需 `torch.compile` | ❌ | 43.3 ms/img | ✅ |
| ONNX + ORT **CUDA** EP | ✅ | ✅ 68MB | 904 ms/img（慢 20×） | ✅ |
| **ONNX + ORT TensorRT EP** | ✅ | ✅ 68MB | **37.1 ms/img** | ✅（需装 TensorRT） |
| AOTInductor `.so` | ✅ | ✅ | 预计最快 | ❌ 本机缺 MSVC `cl.exe`，出不了包 |

结论：**ONNX + TensorRT EP 是唯一同时满足"固定图 + 零重编译 + 更快"三点的方案**。

### 2.2 ONNX 导出（编一次）

`export_onnx()` 把 `FusedEncoder`（归一化 `x/127.5-1` + encoder，通道后置）导出成
**固定形状**的 ONNX 图——`batch` 写死（无 dynamic axes）、`opset_version=17`、开启
常量折叠。导出的 `.onnx` 把**图结构 + 全部权重**封进一个文件。

关键设计：
- **固定 batch**：导出时 `batch = --batch_size`（默认 8）。运行 `--load_onnx` 时
  **必须**用同一个 batch，否则形状不匹配。
- **uint8 输入**：输入直接是 `uint8 [B,3,512,512]`，归一化融进图里，省去主机端
  float 转换与一次 H2D。
- **持久 GPU 输入缓冲**：`ORTEncoder` 在 GPU 上开一块固定 buffer，每步只做一次
  `copy_`（non-blocking）填充，避免每步主机分配；输出绑定到 GPU，测速时不做 D2H。

### 2.3 TensorRT EP 配置（让它真正快且不炸）

`ORTEncoder.__init__` 在导入 onnxruntime **之前**，把两个 DLL 目录加进搜索路径：
1. `torch/lib` —— 复用 torch 自带的 `cublasLt64_12.dll` + `cudnn64_9.dll`（无需系统装 CUDA toolkit）；
2. `tensorrt_libs/` —— 暴露 `nvinfer_10.dll` / `nvinfer_plugin_10.dll`（否则 TRT EP 加载失败、
   静默回退到 CUDA EP，假象"成功"但奇慢）。

TRT EP 的关键参数（**必须**生效的几项）：

```python
providers.append(("TensorrtExecutionProvider", {
    "device_id": 0,
    "trt_fp16_enable": True,                          # fp16 推理，更快更小
    "trt_max_workspace_size": 2_147_483_648,         # 2 GiB 上限
    "trt_engine_cache_enable": True,                  # 引擎缓存到磁盘
    "trt_engine_cache_path": "<onnx_dir>/trt_cache", # 复用，免二次构建
}))
```

- **`trt_max_workspace_size=2GiB`**：这是"不炸显存"的关键。TRT 首次构建会做 **tactic 搜索**
  （每个算子试几十种 kernel 并实测计时），瞬时显存可达 7.8 GB。加上限后，构建期峰值压到
  ~4.5 GB，在 8 GB 卡上安全。
- **`trt_engine_cache_enable`**：构建好的引擎序列化到 `trt_cache/`。**首次 ~6 min，之后
  直接加载缓存、零重建**——即"编一次，永不再编"。

> ⚠️ **ORT 1.21 不支持** `trt_tactic_source` / `trt_timing_cache_enable` 这类 key。
> 误用会报 `Invalid TensorRT EP option` 并回退到 CUDA EP（看起来跑通实则慢 20×）。
> 上面只保留已验证生效的参数。

---

## 3. 环境与安装（Windows）

```powershell
# 1) onnxruntime（GPU，CUDA12/cuDNN9 构建）—— 匹配 torch 自带 CUDA
python -m pip uninstall -y onnxruntime onnxruntime-gpu   # 先清掉冲突的 CPU 版 + CUDA13 版
python -m pip install "onnxruntime-gpu==1.21.0"

# 2) TensorRT（与 ORT 1.21 对应的 10.4.0）
python -m pip install "tensorrt==10.4.0"

# 3) 导出固定图（编一次）
python tools/fast_vae_encode_d5.py --export_onnx data/train_d5/encoder_b8.onnx --batch_size 8
```

依赖与坑：
- 原环境同时装了 `onnxruntime 1.26(CPU)` 和 `onnxruntime-gpu 1.27(CUDA13 构建)`，两包共用
  同一 `onnxruntime/` 目录互相冲突、GPU 不生效。已统一为仅 `onnxruntime-gpu==1.21.0`。
- TensorRT 的 DLL 在 `tensorrt_libs/`（`pip` 装的 `tensorrt` + `tensorrt-cu12` 包），脚本已自动
  `os.add_dll_directory` 暴露，无需手动设 PATH。

---

## 4. 使用命令

```powershell
# 导出固定图（只需一次）
python tools/fast_vae_encode_d5.py --export_onnx data/train_d5/encoder_b8.onnx --batch_size 8

# 仅测速（256 张探针，走 TRT 缓存后约几秒）
python tools/fast_vae_encode_d5.py --use_packed \
    --load_onnx data/train_d5/encoder_b8.onnx --batch_size 8 --onnx_trt

# 编码全部 5000 张并落盘（首次会先构建引擎 ~6min，之后走缓存约 3min）
python tools/fast_vae_encode_d5.py --use_packed \
    --load_onnx data/train_d5/encoder_b8.onnx --batch_size 8 --onnx_trt --full
```

> 运行前确保 `PYTHONPATH` 含仓库根（脚本依赖项目内模块）。

---

## 5. 最终成绩（RTX 4070 Laptop 8 GB，5000 张，batch=8）

| 方案 | ms/img | 5000 张 | 显存峰值 | 免重编译 | 单文件 |
|---|---|---|---|---|---|
| ONNX + **TensorRT** EP | **37.1** | **3.09 min** | 4540 MiB | ✅ | ✅ |
| torch `--graph --graph_compile` | 43.3 | 3.61 min | 109 MiB | ❌ | ❌ |
| ONNX + ORT CUDA EP | 904 | 75 min | — | ✅ | ✅ |

- **ONNX + TRT 达成目标**：固定图（68MB 单文件）+ 零重编译（引擎缓存）+ 更快（37.1 ms/img，
  比 torch graph 还快 23%，5000 张进 3 分钟）。
- 代价：**显存峰值高**（~4.5 GB，因 TRT runtime + workspace 的固定开销）与**首次构建 ~6 min**
  （已缓存，后续免构建）。若显存极度紧张或追求最小依赖，可退回 torch `--graph`（仅 109 MiB，但
  每次需 compile）。

---

## 6. 踩坑速查

1. **首次构建"显存炸"**：根因是 TRT tactic 搜索无 workspace 上限（瞬时 7.8 GB）。加
   `trt_max_workspace_size=2GiB` 解决。
2. **TRT 没生效却"跑通"**：DLL 未找到时 ORT 静默回退到 CUDA EP（904 ms/img）。确认
   `tensorrt_libs/` 已 `add_dll_directory`，且日志打印 `provider in use: TensorrtExecutionProvider`。
3. **无效参数**：ORT 1.21 不认 `trt_tactic_source` / `trt_timing_cache_enable`，会回退 CUDA EP。
4. **包冲突**：CPU 版 `onnxruntime` 与 `onnxruntime-gpu` 共用目录会互相损坏，统一装 GPU 版即可。
5. **batch 必须整除 5000**：ONNX 固定形状，运行 `--load_onnx` 的 `--batch_size` 须等于导出时的 batch。
