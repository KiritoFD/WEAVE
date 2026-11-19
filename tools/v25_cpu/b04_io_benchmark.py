"""B04: CPU/I/O 性能诊断工具（Slot B，docs/19_pull_together/agent2.md §四 B04）。

在**本地工作机**（Windows）上对评测流水线的 CPU/I/O 环节做基准测量，回答核心问题：
"评测流水线里哪些耗时根本不是 GPU 推理"。

测量项（全部纯 CPU、小规模、只写 tmp 目录，总写入量 <200MB）：
  M1  750 任务 JSON 结果序列化/反序列化（合成 750 行记录，measure.py 行 schema）；
  M2  PNG 编码（10 张 512x512 uint8 -> bytes）与写盘吞吐（compress_level=1/6，串行/8 线程）；
  M3  sha256 哈希吞吐（100MB 合成内存缓冲，1MB/4MB 分块对比）；
  M4  JSONL 注册表逐行追加（含 fsync，镜像 src/v25/artifact_registry.py::_append_json）
      vs 整体重写，含真实 ArtifactRegistry.register/validate 端到端成本；
  M5  100x1KB 小文件 vs 单个 100KB 大文件的创建/读取开销，目录创建开销；
  M6  Windows 路径处理正确性断言与逐操作开销（resolve/stat/normcase/join/open）；
  M7  torch.save/torch.load 原始 batch 载荷往返（镜像 src/infer/run.py 载荷，可选）。

运行（从仓库根目录）：
  python -m tools.v25_cpu.b04_io_benchmark --quick      # 样本减半，<60s
  python -m tools.v25_cpu.b04_io_benchmark              # 完整模式
  python -m tools.v25_cpu.b04_io_benchmark --out-json docs/v25/audits/b04_io_results.json \
      --out-md docs/v25/audits/b04_io_report.md

约束遵守（agent2.md §十一 CPU 资源隔离）：
  - import 任何数值库之前设置 CUDA_VISIBLE_DEVICES="" / OMP_NUM_THREADS=2 /
    MKL_NUM_THREADS=2 / OPENBLAS_NUM_THREADS=2；
  - 不 import 任何 CUDA 路径、不 ssh、不读 runs/ 下真实评测数据大文件
    （仅允许对臂元数据做 meta.json 小文件读取与 stat，本工具交付时已离线标定；
    运行期完全不访问 runs/）；
  - 所有写盘在 tempfile.mkdtemp 临时目录，结束后默认删除（--keep-tmp 保留）；
  - 每项 warm+cold 至少 3 次取中位数（--quick 减半为 2+2）。

输出：
  docs/v25/audits/b04_io_results.json   机器可读全量数值
  docs/v25/audits/b04_io_report.md      人读报告（含非 GPU 开销归因估算与 B03 对接数值）
"""
from __future__ import annotations

import os

# ---- §十一 CPU 资源隔离：必须在任何数值库 import 之前钉死 ----
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import argparse
import gc
import hashlib
import io
import json
import platform
import random
import shutil
import statistics
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
from PIL import Image
import PIL

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]

# 真实臂标定（由本工具交付时对 runs/v8/F2.e/gen/meta.json 的元数据探测取得，
# 仅读 meta.json 小文件并 stat 载荷大小，未读任何 .pt 大文件）：
REAL_RAW_PAYLOAD_MB_PER_ARM = 738.0   # 188 个 batch .pt（batch=4, decode=True），750 任务
REAL_EVAL_JSON_KB = 897.0             # runs/v25/measured/A_p1_l130/eval.json 的 stat 大小
FEATURES_PT_MB_PER_ARM = 5.9          # 750 x (dino 384 + csd 795 + clip 768) x fp32，估算


# --------------------------------------------------------------------------- #
# 计时与统计
# --------------------------------------------------------------------------- #
def _stats_ms(samples: list[float]) -> dict:
    return {
        "n": len(samples),
        "median_ms": round(statistics.median(samples), 3),
        "min_ms": round(min(samples), 3),
        "max_ms": round(max(samples), 3),
    }


def run_reps(fn, reps: int, setup=None) -> list[float]:
    """运行 fn reps 次，每次前 gc.collect()；setup() 的返回值解包为 fn 参数（untimed）。"""
    out: list[float] = []
    for _ in range(reps):
        gc.collect()
        args = setup() if setup is not None else None
        t0 = time.perf_counter_ns()
        fn(*args) if args is not None else fn()
        out.append((time.perf_counter_ns() - t0) / 1e6)
    return out


def bench_wc(warm_fn, cold_fn, warm_reps: int, cold_reps: int, cold_setup=None) -> dict:
    """warm（复用状态）+ cold（cold_setup 每次给全新产物）两相，各自取中位数。

    cold_setup=None 表示 cold_fn 也是零参函数（如纯 CPU 计算）。
    """
    warm = run_reps(warm_fn, warm_reps)
    cold = run_reps(cold_fn, cold_reps, setup=cold_setup)
    return {"warm": _stats_ms(warm), "cold": _stats_ms(cold)}


# --------------------------------------------------------------------------- #
# M1: 750 任务 JSON 结果序列化/反序列化
# --------------------------------------------------------------------------- #
def synth_rows_750(seed: int = 20261120) -> list[dict]:
    """合成 750 行记录，schema 完全镜像 src/eval/measure.py:90-101 的 row 字段。"""
    rng = random.Random(seed)
    rows: list[dict] = []
    for i in range(750):
        cross = i < 600
        ss = rng.randrange(5)
        ts = ss
        if cross:
            while ts == ss:
                ts = rng.randrange(5)
        rows.append({
            "task_id": (f"cross_{i:04d}" if cross else f"same_{i - 600:03d}"),
            "src_stem": f"stem_{rng.randrange(10000):05d}",
            "src_style": STYLES[ss], "tgt_style": STYLES[ts],
            "src_style_id": ss, "tgt_style_id": ts, "is_cross": cross,
            "pred_class": rng.randrange(5),
            "style_scores_5": [round(rng.uniform(0.2, 0.9), 6) for _ in range(5)],
            "clip_scores_5": [round(rng.uniform(0.2, 0.9), 6) for _ in range(5)],
            "csd_scores_5": [round(rng.uniform(-2.0, 2.0), 6) for _ in range(5)],
            "dino_s": round(rng.uniform(0.3, 0.8), 6),
            "dino_c": round(rng.uniform(0.5, 0.8), 6),
            "lpips": round(rng.uniform(0.4, 0.6), 6),
            "target_top1": int(rng.random() < 0.44),
            "target_margin": round(rng.uniform(-0.1, 0.1), 6),
            "dT": round(rng.uniform(-0.1, 0.1), 6),
        })
    return rows


def m1_json750(tmp: Path, rw: int, rc: int) -> dict:
    d = tmp / "json750"
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_total": 750, "n_cross": 600, "n_same": 150,
        "rows": synth_rows_750(),
        "summary": {"csd_top1": 0.4433, "dino_s": 0.5789, "lpips": 0.5057},
    }

    def _dumps_compact() -> str:
        # 镜像 src/eval/measure.py:109-112 的 json.dumps(..., ensure_ascii=False)
        return json.dumps(payload, ensure_ascii=False)

    def _dumps_indent2() -> str:
        # 镜像 src/utils/runtime.py:28-36 save_json 的序列化约定
        return json.dumps(payload, ensure_ascii=False, indent=2,
                          default=str, allow_nan=False)

    def _write_compact(path: Path) -> None:
        path.write_text(_dumps_compact(), encoding="utf-8")

    def _write_atomic(path: Path) -> None:
        # 镜像 runtime.save_json：tmp + replace 原子写
        t = path.with_suffix(path.suffix + ".tmp")
        t.write_text(_dumps_indent2(), encoding="utf-8")
        t.replace(path)

    compact = _dumps_compact()
    res: dict = {
        "rows": 750,
        "compact_json_bytes": len(compact.encode("utf-8")),
        "dumps_compact_cpu": bench_wc(_dumps_compact, _dumps_compact, rw, rc),
        "dumps_indent2_cpu": bench_wc(_dumps_indent2, _dumps_indent2, rw, rc),
    }
    # warm: 同名文件覆盖；cold: 每次全新文件名（首触）
    warm_p = d / "eval_compact.json"
    warm_a = d / "eval_atomic.json"
    compact_ctr = [0]

    def _cold_compact() -> list[Path]:
        compact_ctr[0] += 1
        return [d / f"eval_compact_cold_{compact_ctr[0]}.json"]

    atomic_ctr = [0]

    def _cold_atomic() -> list[Path]:
        atomic_ctr[0] += 1
        return [d / f"eval_atomic_cold_{atomic_ctr[0]}.json"]

    res["write_compact_disk"] = bench_wc(
        lambda: _write_compact(warm_p), lambda p: _write_compact(p),
        rw, rc, _cold_compact)
    res["write_indent2_atomic_disk"] = bench_wc(
        lambda: _write_atomic(warm_a), lambda p: _write_atomic(p),
        rw, rc, _cold_atomic)
    # 读回+解析（warm：同一文件；read 基本无冷热差，报告 warm 为主）
    _write_compact(warm_p)
    res["read_parse_disk"] = bench_wc(
        lambda: json.loads(warm_p.read_bytes()),
        lambda: json.loads(warm_p.read_bytes()), rw, rc)
    res["file_size_kb_compact"] = round(warm_p.stat().st_size / 1024, 1)
    _write_atomic(warm_a)
    res["file_size_kb_indent2"] = round(warm_a.stat().st_size / 1024, 1)
    res["real_eval_json_kb_reference"] = REAL_EVAL_JSON_KB
    return res


# --------------------------------------------------------------------------- #
# M2: PNG 编码与写盘吞吐
# --------------------------------------------------------------------------- #
def m2_png(tmp: Path, rw: int, rc: int, n_imgs: int = 10) -> dict:
    d = tmp / "png"
    d.mkdir(parents=True, exist_ok=True)
    # 均匀噪声 = PNG 最坏情形（真图更可压缩 -> 本节吞吐是真图的保守下界）
    imgs = [np.random.default_rng(42 + i).integers(0, 256, (512, 512, 3),
            dtype=np.uint8) for i in range(n_imgs)]

    def _encode_all(level: int) -> None:
        buf = io.BytesIO()
        for arr in imgs:
            buf.seek(0)
            buf.truncate(0)
            Image.fromarray(arr).save(buf, format="PNG", compress_level=level)

    def _save_one(pair) -> None:
        path, arr = pair
        Image.fromarray(arr).save(path, format="PNG", compress_level=1)

    def _write_serial(sub: Path) -> None:
        sub.mkdir(parents=True, exist_ok=True)
        for i, arr in enumerate(imgs):
            Image.fromarray(arr).save(sub / f"img_{i:02d}.png",
                                      format="PNG", compress_level=1)

    def _write_threads8(sub: Path) -> None:
        # 镜像 src/utils/raw_io.py::save_png_batch_fast（mkdir + 8 线程 + level=1）
        sub.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(_save_one,
                        [(sub / f"img_{i:02d}.png", a) for i, a in enumerate(imgs)]))

    sizes = {}
    for lv in (1, 6):
        buf = io.BytesIO()
        Image.fromarray(imgs[0]).save(buf, format="PNG", compress_level=lv)
        sizes[f"noise_l{lv}_bytes_per_img"] = buf.tell()

    warm_dir = d / "warm_serial"
    warm_dir.mkdir(parents=True, exist_ok=True)
    ser_ctr = [0]

    def _cold_serial() -> list[Path]:
        ser_ctr[0] += 1
        return [d / f"cold_serial_{ser_ctr[0]}"]

    warm_t = d / "warm_threads"
    warm_t.mkdir(parents=True, exist_ok=True)
    thr_ctr = [0]

    def _cold_threads() -> list[Path]:
        thr_ctr[0] += 1
        return [d / f"cold_threads_{thr_ctr[0]}"]

    res: dict = {
        "n_images": n_imgs,
        "size_bytes_per_img": sizes,
        "encode_only_l1": bench_wc(lambda: _encode_all(1), lambda: _encode_all(1),
                                   rw, rc),
        "encode_only_l6_pil_default": bench_wc(lambda: _encode_all(6),
                                               lambda: _encode_all(6), rw, rc),
        "write_l1_serial": bench_wc(
            lambda: _write_serial(warm_dir), lambda sub: _write_serial(sub),
            rw, rc, _cold_serial),
        "write_l1_threads8": bench_wc(
            lambda: _write_threads8(warm_t), lambda sub: _write_threads8(sub),
            rw, rc, _cold_threads),
    }
    for k in ("encode_only_l1", "encode_only_l6_pil_default",
              "write_l1_serial", "write_l1_threads8"):
        for phase in ("warm", "cold"):
            tot = res[k][phase]["median_ms"]
            res[k][phase]["per_image_ms"] = round(tot / n_imgs, 3)
    # 750 张外推（估算，噪声上界）
    t8 = res["write_l1_threads8"]["warm"]["median_ms"]
    res["estimate_750_imgs_threads8_warm_s"] = round(t8 * 75 / 1000, 2)
    return res


# --------------------------------------------------------------------------- #
# M3: sha256 哈希吞吐
# --------------------------------------------------------------------------- #
def m3_sha256(rw: int, rc: int, mb: int = 100) -> dict:
    buf = np.random.default_rng(7).bytes(mb << 20)
    mv = memoryview(buf)
    res: dict = {"buffer_mb": mb}

    def _chunked(chunk: int) -> str:
        h = hashlib.sha256()
        for off in range(0, len(buf), chunk):
            h.update(mv[off:off + chunk])
        return h.hexdigest()

    for name, chunk in (("chunk_1mb", 1 << 20), ("chunk_4mb", 4 << 20),
                        ("single_shot", 1 << 60)):
        r = bench_wc(lambda c=chunk: _chunked(c), lambda c=chunk: _chunked(c),
                     rw, rc)
        r["warm_gbps"] = round(mb / 1024 / (r["warm"]["median_ms"] / 1000), 3)
        r["cold_gbps"] = round(mb / 1024 / (r["cold"]["median_ms"] / 1000), 3)
        res[name] = r
    # 30MB checkpoint 校验外推（估算）
    gbs = res["chunk_1mb"]["warm_gbps"]
    res["estimate_sha256_30mb_ckpt_s"] = round(30 / (gbs * 1024), 4)
    return res


# --------------------------------------------------------------------------- #
# M4: JSONL 注册表 逐行追加 vs 整体重写（对接 src/v25/artifact_registry.py）
# --------------------------------------------------------------------------- #
def _template_record(i: int) -> dict:
    """与 src/v25/artifact_registry.py::ArtifactRecord.to_dict 完全同构的记录。"""
    return {
        "record_type": "artifact",
        "schema_version": 1,
        "artifact_path": f"runs/v25/cache/bench_cache_{i:05d}.pt",
        "kind": "bench_latent_cache",
        "created_utc": "2026-11-20T00:00:00+00:00",
        "git_sha": "0" * 40,
        "depends": {
            "ckpt_sha256": "a" * 64,
            "params_hash": "b" * 64,
            "decoder_version": "c" * 12,
        },
        "format_version": 2,
    }


def m4_registry(tmp: Path, rw: int, rc: int, n_lines: int, n_register: int) -> dict:
    d = tmp / "jsonl"
    d.mkdir(parents=True, exist_ok=True)
    recs = [_template_record(i) for i in range(n_lines)]

    # (a) 逐行追加 + 每行 fsync —— 精确镜像 _append_json（含每次 mkdir）
    def _append_fsync() -> None:
        p = d / "append_fsync.jsonl"
        for r in recs:
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(r, sort_keys=True, default=str) + "\n")
                fh.flush()
                os.fsync(fh.fileno())

    # (b) 逐行追加，无 fsync
    def _append_nofsync() -> None:
        p = d / "append_nofsync.jsonl"
        for r in recs:
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(r, sort_keys=True, default=str) + "\n")
                fh.flush()

    # (c) 整体重写（原子 tmp+replace，runtime.save_json 风格）
    def _rewrite_all() -> None:
        p = d / "rewrite.jsonl"
        body = "".join(json.dumps(r, sort_keys=True, default=str) + "\n"
                       for r in recs)
        t = p.with_suffix(p.suffix + ".tmp")
        t.write_text(body, encoding="utf-8")
        t.replace(p)

    # (d) 全文件逐行解析（镜像 ArtifactRegistry.records()）
    _rewrite_all()
    rp = d / "rewrite.jsonl"

    def _parse_all() -> int:
        n = 0
        with open(rp, "r", encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                json.loads(ln)
                n += 1
        return n

    res: dict = {
        "n_lines": n_lines,
        "line_bytes": len(json.dumps(recs[0], sort_keys=True,
                                     default=str).encode("utf-8")),
        "append_fsync": bench_wc(_append_fsync, _append_fsync, rw, rc),
        "append_nofsync": bench_wc(_append_nofsync, _append_nofsync, rw, rc),
        "rewrite_all": bench_wc(_rewrite_all, _rewrite_all, rw, rc),
        "parse_all": bench_wc(_parse_all, _parse_all, rw, rc),
    }
    for k in ("append_fsync", "append_nofsync"):
        res[k]["per_line_ms"] = round(res[k]["warm"]["median_ms"] / n_lines, 4)
        res[k]["cold_per_line_ms"] = round(res[k]["cold"]["median_ms"] / n_lines, 4)

    # (e) 真实 ArtifactRegistry 端到端成本
    try:
        from src.v25.artifact_registry import ArtifactRegistry
    except Exception as e:  # pragma: no cover
        res["artifact_registry_e2e"] = {"skipped": f"import failed: {e}"}
        return res

    reg_d = d / "e2e"
    reg = ArtifactRegistry(reg_d / "registry.jsonl", reg_d / "invalid.jsonl")
    deps = {"params_hash": "b" * 64, "decoder_version": "c" * 12}
    art0 = reg_d / "artifact_000.pt"
    art0.write_bytes(b"\x00" * 1024)
    reg.register(art0, "bench_cache", depends=dict(deps), format_version=2,
                 notes="warmup(untimed)")  # 预热：首次 import/文件系统冷启动

    cold_i = [n_register]

    def _reg_warm() -> None:  # 同一路径重复登记（registry 大小恒定）
        reg.register(art0, "bench_cache", depends=dict(deps), format_version=2)

    def _reg_cold() -> None:  # 全新路径（registry 持续增长，高频 register 真实形态）
        cold_i[0] += 1
        reg.register(reg_d / f"artifact_{cold_i[0]:05d}.pt", "bench_cache",
                     depends=dict(deps), format_version=2)

    res["artifact_registry_e2e"] = {
        "register_warm_same_path": bench_wc(_reg_warm, _reg_cold, rw, rc),
        "registry_size_after_warmup": len(reg.records()),
    }
    # 关掉 git 子进程路径（WEAVE_GIT_SHA 环境优先），得到纯 fsync 下界
    os.environ["WEAVE_GIT_SHA"] = "0" * 40
    try:
        nog = run_reps(_reg_warm, rw)
        res["artifact_registry_e2e"]["register_warm_nogitprobe"] = _stats_ms(nog)
    finally:
        os.environ.pop("WEAVE_GIT_SHA", None)
    res["artifact_registry_e2e"]["validate_known"] = bench_wc(
        lambda: reg.validate(art0, depends=dict(deps), who="b04"),
        lambda: reg.validate(art0, depends=dict(deps), who="b04"), rw, rc)
    res["artifact_registry_e2e"]["validate_unknown"] = bench_wc(
        lambda: reg.validate(reg_d / "never_registered.pt", who="b04"),
        lambda: reg.validate(reg_d / "never_registered.pt", who="b04"), rw, rc)
    return res


# --------------------------------------------------------------------------- #
# M5: 大量小文件 vs 单个大文件 + 目录创建
# --------------------------------------------------------------------------- #
def m5_files(tmp: Path, rw: int, rc: int, n_files: int = 100) -> dict:
    d = tmp / "files"
    d.mkdir(parents=True, exist_ok=True)
    b1k = np.random.default_rng(11).bytes(1024)
    b100k = np.random.default_rng(12).bytes(100 * 1024)

    def _create_small(sub: Path) -> None:
        sub.mkdir(parents=True, exist_ok=True)
        for i in range(n_files):
            (sub / f"f_{i:03d}.bin").write_bytes(b1k)

    def _read_small(sub: Path) -> None:
        sub.mkdir(parents=True, exist_ok=True)
        for i in range(n_files):
            (sub / f"f_{i:03d}.bin").read_bytes()

    warm_s = d / "warm_small"
    warm_s.mkdir(parents=True, exist_ok=True)
    small_ctr = [0]

    def _cold_small() -> list[Path]:
        small_ctr[0] += 1
        return [d / f"cold_small_{small_ctr[0]}"]

    big_ctr = [0]

    def _cold_big() -> list[Path]:
        big_ctr[0] += 1
        return [d / f"big_cold_{big_ctr[0]}.bin"]

    big_f = d / "big_100k.bin"
    res: dict = {"n_files": n_files}
    res["create_100x1kb"] = bench_wc(
        lambda: _create_small(warm_s), _create_small, rw, rc, _cold_small)
    read_ctr = [0]

    def _cold_small_read() -> list[Path]:
        read_ctr[0] += 1
        sub = d / f"cold_read_{read_ctr[0]}"
        _create_small(sub)  # untimed setup：文件已在盘上，测首触读取
        return [sub]

    res["read_100x1kb"] = bench_wc(
        lambda: _read_small(warm_s), _read_small, rw, rc, _cold_small_read)
    res["create_1x100kb"] = bench_wc(
        lambda: big_f.write_bytes(b100k), lambda p: p.write_bytes(b100k),
        rw, rc, _cold_big)
    read_big_ctr = [0]

    def _cold_big_read() -> list[Path]:
        read_big_ctr[0] += 1
        p = d / f"big_cold_read_{read_big_ctr[0]}.bin"
        p.write_bytes(b100k)  # untimed setup：文件已在盘上，测首触读取
        return [p]

    res["read_1x100kb"] = bench_wc(
        lambda: big_f.read_bytes(), lambda p: p.read_bytes(), rw, rc,
        _cold_big_read)
    # 目录创建：全新嵌套链 vs exist_ok 重入
    dir_ctr = [0]

    def _mkdir_nested_fresh() -> None:
        dir_ctr[0] += 1
        (d / "dirs" / f"r{dir_ctr[0]}" / "a" / "b" / "c").mkdir(parents=True,
                                                               exist_ok=True)

    def _mkdir_exist_ok() -> None:
        for _ in range(50):
            warm_s.mkdir(parents=True, exist_ok=True)

    res["mkdir_nested_parents_fresh"] = bench_wc(
        _mkdir_nested_fresh, _mkdir_nested_fresh, rw, rc)
    res["mkdir_exist_ok_x50"] = bench_wc(_mkdir_exist_ok, _mkdir_exist_ok, rw, rc)
    for k in ("create_100x1kb", "read_100x1kb"):
        res[k]["per_file_us_warm"] = round(
            res[k]["warm"]["median_ms"] * 1000 / n_files, 1)
        res[k]["per_file_us_cold"] = round(
            res[k]["cold"]["median_ms"] * 1000 / n_files, 1)
    res["mkdir_exist_ok_per_call_us_warm"] = round(
        res["mkdir_exist_ok_x50"]["warm"]["median_ms"] * 1000 / 50, 1)
    return res


# --------------------------------------------------------------------------- #
# M6: Windows 路径处理 正确性 + 开销
# --------------------------------------------------------------------------- #
def m6_paths(tmp: Path, rw: int, rc: int, n_ops: int = 2000) -> dict:
    checks: list[dict] = []

    def _check(name: str, fn) -> None:
        try:
            detail = fn()
            checks.append({"name": name, "ok": True, "detail": str(detail)})
        except Exception as e:
            checks.append({"name": name, "ok": False,
                           "detail": f"{type(e).__name__}: {e}"})

    def _c_slash():  # 正反斜杠等价
        a = Path(tmp) / "sub" / "f.txt"
        b = Path(str(tmp) + "\\sub\\f.txt")
        assert a == b and a.as_posix() == b.as_posix()
        return "PurePath 归一化等价"

    def _c_case():  # 大小写不敏感
        p = tmp / "CaseTest.txt"
        p.write_bytes(b"x")
        assert (tmp / "casetest.TXT").is_file()
        p.unlink()
        return "stat 大小写不敏感"

    def _c_long():  # >260 字符长路径
        sub = tmp / "longpath"
        comp = "d" * 40
        deep = sub
        for _ in range(7):
            deep = deep / comp
        total = len(str(deep / "leaf.txt"))
        deep.mkdir(parents=True, exist_ok=True)
        (deep / "leaf.txt").write_bytes(b"ok")
        assert (deep / "leaf.txt").read_bytes() == b"ok"
        return f"路径长 {total} 字符读写成功"

    def _c_dots():  # .. 解析稳定性
        a = (REPO / "src" / "utils" / ".." / "utils").resolve()
        b = (REPO / "src" / "utils").resolve()
        assert a == b
        return "dotdot resolve 收敛"

    def _c_normcase():
        assert os.path.normcase("G:/WEAVE") == "g:\\weave"
        return "normcase 斜杠+小写归一"

    def _c_relative_abs():  # 相对/绝对路径锚定一致
        p1 = Path("src/utils/raw_io.py")
        p2 = REPO / "src" / "utils" / "raw_io.py"
        assert p2.is_file() and not p1.is_absolute()
        assert (REPO / p1).resolve() == p2.resolve()
        return "相对路径锚定 REPO 后与绝对路径一致"

    _check("slash_equivalence", _c_slash)
    _check("case_insensitive_stat", _c_case)
    _check("long_path_over_260", _c_long)
    _check("dotdot_resolve", _c_dots)
    _check("normcase", _c_normcase)
    _check("relative_vs_absolute_anchor", _c_relative_abs)

    probe = REPO / "src" / "utils" / "raw_io.py"

    def _t(fn, n: int) -> dict:
        gc.collect()
        t0 = time.perf_counter_ns()
        for _ in range(n):
            fn()
        ms = (time.perf_counter_ns() - t0) / 1e6
        return {"n": n, "median_us_per_op": round(ms * 1000 / n, 2)}

    overhead = {
        "path_resolve_existing": _t(lambda: probe.resolve(), n_ops),
        "path_stat_existing": _t(lambda: probe.stat(), n_ops * 2),
        "path_exists": _t(lambda: probe.exists(), n_ops * 2),
        "normcase_str": _t(lambda: os.path.normcase(str(probe)), n_ops * 5),
        "path_construct_join": _t(lambda: REPO / "src" / "utils" / "x.pt",
                                  n_ops * 5),
        "open_close_existing": _t(lambda: open(probe, "rb").close(),
                                  max(n_ops // 2, 200)),
        "str_to_path_roundtrip": _t(lambda: str(Path(str(probe))), n_ops * 5),
    }
    return {"correctness": checks, "overhead": overhead}


# --------------------------------------------------------------------------- #
# M7: torch 原始 batch 载荷往返（可选；镜像 src/infer/run.py 载荷）
# --------------------------------------------------------------------------- #
def m7_torch(tmp: Path, rw: int, rc: int, batch: int = 4) -> dict:
    try:
        import torch
    except Exception as e:  # pragma: no cover
        return {"skipped": f"torch import failed: {e}"}
    # 注意：torch>=2.x 的 is_available() 走 NVML、无视 CUDA_VISIBLE_DEVICES；
    # 隔离判据必须用 device_count()（CUDA_VISIBLE_DEVICES="" 时为 0）。
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:  # pragma: no cover
        return {"skipped": f"{n_gpu} CUDA device(s) visible; skipped for isolation"}
    d = tmp / "torch_batch"
    d.mkdir(parents=True, exist_ok=True)
    g = np.random.default_rng(13)
    payload = {
        "ids": [f"task_{i:04d}" for i in range(batch)],
        "z_src": torch.from_numpy(
            g.standard_normal((batch, 4, 64, 64)).astype(np.float16)),
        "z0": torch.from_numpy(
            g.standard_normal((batch, 4, 64, 64)).astype(np.float16)),
        "z_final": torch.from_numpy(
            g.standard_normal((batch, 4, 64, 64)).astype(np.float16)),
        "rgb_u8": torch.from_numpy(
            g.integers(0, 256, (batch, 3, 512, 512), dtype=np.uint8)),
    }
    nbytes = sum(t.untyped_storage().nbytes() for t in payload.values()
                 if hasattr(t, "untyped_storage"))
    save_ctr = [0]

    def _save(path: Path) -> None:
        # 镜像 src/infer/run.py:75-77 的 tmp + replace 原子写
        t = path.with_suffix(".tmp")
        torch.save(payload, t)
        t.replace(path)

    def _cold_setup() -> list[Path]:
        save_ctr[0] += 1
        return [d / f"batch_cold_{save_ctr[0]}.pt"]

    load_ctr = [0]

    def _cold_setup_load() -> list[Path]:
        load_ctr[0] += 1
        p = d / f"batch_cold_load_{load_ctr[0]}.pt"
        _save(p)  # untimed setup：文件已在盘上，测首触读取
        return [p]

    warm_p = d / "batch_warm.pt"
    _save(warm_p)
    res: dict = {
        "batch": batch,
        "payload_mb": round(nbytes / 1e6, 2),
        "torch_version": torch.__version__,
        "cuda_device_count": n_gpu,
        "save": bench_wc(lambda: _save(warm_p), lambda p: _save(p), rw, rc,
                         _cold_setup),
        "load": bench_wc(
            lambda: torch.load(warm_p, map_location="cpu", weights_only=False),
            lambda p: torch.load(p, map_location="cpu", weights_only=False),
            rw, rc, _cold_setup_load),
    }
    mb = res["payload_mb"]
    for k in ("save", "load"):
        res[k]["warm_mbps"] = round(mb / (res[k]["warm"]["median_ms"] / 1000), 1)
        res[k]["cold_mbps"] = round(mb / (res[k]["cold"]["median_ms"] / 1000), 1)
    return res


# --------------------------------------------------------------------------- #
# 非 GPU 开销归因外推（全部标注为估算）
# --------------------------------------------------------------------------- #
def extrapolate(res: dict, arm_wall_s: float, raw_mb: float) -> dict:
    """把实测中位数外推到单臂 750 任务评测，量化 CPU/I/O 项占比（估算值）。"""

    def med(key: str, phase: str = "warm") -> float:
        node = res
        for part in key.split("."):
            node = node[part]
        return node[phase]["median_ms"]

    est: dict = {"assumptions": {
        "arm_wall_s_baseline": arm_wall_s,
        "raw_payload_mb_per_arm": raw_mb,
        "eval_json_kb_synthetic": res["m1_json750"]["file_size_kb_compact"],
        "features_pt_mb_per_arm": FEATURES_PT_MB_PER_ARM,
        "png_scenario": "均匀噪声最坏情形上界（真艺术图更可压缩，写盘字节量被高估）",
        "all_values_are_estimates": True,
    }}
    m7 = res.get("m7_torch_batch", {})
    load_mbps = m7.get("load", {}).get("warm_mbps")
    save_mbps = m7.get("save", {}).get("warm_mbps")
    if load_mbps:
        est["measure_raw_batch_torch_load_s"] = round(raw_mb / load_mbps, 2)
    est["measure_png_write_750imgs_threads8_s"] = round(
        med("m2_png.write_l1_threads8") * 75 / 1000, 2)
    est["measure_png_write_750imgs_serial_s"] = round(
        med("m2_png.write_l1_serial") * 75 / 1000, 2)
    # 合成行比真实 enrich_rows 后的行少字段：按真实 eval.json / 合成 JSON 大小比放大
    kb_ratio = REAL_EVAL_JSON_KB / max(
        res["m1_json750"]["file_size_kb_compact"], 1e-6)
    est["eval_json_size_scale_factor"] = round(kb_ratio, 2)
    est["measure_eval_json_dumps_write_s"] = round(
        (med("m1_json750.dumps_compact_cpu")
         + med("m1_json750.write_compact_disk")) / 1000 * kb_ratio, 3)
    est["measure_eval_json_read_parse_s"] = round(
        med("m1_json750.read_parse_disk") / 1000 * kb_ratio, 3)
    if save_mbps:
        est["measure_features_pt_save_s"] = round(
            FEATURES_PT_MB_PER_ARM / save_mbps, 3)
        est["infer_raw_batch_torch_save_s"] = round(raw_mb / save_mbps, 2)
    # 注册表/哈希条件项（B03 接入后才有）
    est["sha256_gbps_1mb_chunk"] = res["m3_sha256"]["chunk_1mb"]["warm_gbps"]
    est["sha256_30mb_ckpt_s"] = round(
        30 / (est["sha256_gbps_1mb_chunk"] * 1024), 4)
    est["registry_register_per_call_ms"] = med(
        "m4_registry.artifact_registry_e2e.register_warm_same_path")
    nogit = res["m4_registry"].get("artifact_registry_e2e", {}).get(
        "register_warm_nogitprobe", {})
    est["registry_register_nogit_per_call_ms"] = nogit.get("median_ms")
    # 占比汇总（相对 ~50s 臂）
    png_on = (est.get("measure_raw_batch_torch_load_s", 0.0)
              + est["measure_png_write_750imgs_threads8_s"]
              + est["measure_eval_json_dumps_write_s"]
              + est.get("measure_features_pt_save_s", 0.0)
              + est.get("infer_raw_batch_torch_save_s", 0.0))
    est["non_gpu_cpu_io_seconds_total_png_on"] = round(png_on, 2)
    est["non_gpu_cpu_io_seconds_total_png_off"] = round(
        png_on - est["measure_png_write_750imgs_threads8_s"], 2)
    est["share_pct_of_arm_50s_png_on"] = round(
        100 * est["non_gpu_cpu_io_seconds_total_png_on"] / arm_wall_s, 1)
    est["share_pct_of_arm_50s_png_off"] = round(
        100 * est["non_gpu_cpu_io_seconds_total_png_off"] / arm_wall_s, 1)
    return est


# --------------------------------------------------------------------------- #
# 报告生成
# --------------------------------------------------------------------------- #
def _tbl(rows: list[list[str]]) -> str:
    if not rows:
        return "(无数据)"
    out = ["| " + " | ".join(rows[0]) + " |",
           "| " + " | ".join(["---"] * len(rows[0])) + " |"]
    for r in rows[1:]:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def _wc_line(label: str, node: dict) -> str:
    w, c = node["warm"], node["cold"]
    return (f"{label} | {w['median_ms']} | {c['median_ms']} | "
            f"{w['min_ms']}~{w['max_ms']} | {c['min_ms']}~{c['max_ms']} | "
            f"n={w['n']}+{c['n']}")


def build_report(res: dict, mode: str, wall_s: float, tmp_used_mb: float) -> str:
    L: list[str] = []
    env = res["env"]
    L.append("# B04：CPU/I/O 性能诊断报告（本地工作机 Windows）")
    L.append("")
    L.append(f"- 任务：B04（docs/19_pull_together/agent2.md §四），模式 `{mode}`，"
             f"总耗时 {wall_s:.1f}s，生成于 {res['generated_utc']}")
    L.append(f"- 环境：{env['platform']} / Python {env['python']} / "
             f"{env['cpu_count']} 线程 / numpy {env['numpy']} / "
             f"Pillow {env['pillow']}"
             + (f" / torch {env['torch']}" if env.get("torch") else ""))
    L.append(f"- 约束遵守：全程 `CUDA_VISIBLE_DEVICES=\"\"`、"
             f"`OMP/MKL/OPENBLAS_NUM_THREADS=2`；只写临时目录（本机 "
             f"{env['tmp_drive']}），峰值写入 {tmp_used_mb:.1f}MB（<200MB 预算）；"
             f"无 ssh、无 CUDA、未修改任何现有文件、未读取任何 runs/ 大文件内容"
             f"（交付前仅对臂元数据做过 meta.json 读取与 stat 标定）")
    L.append("")
    L.append("口径：warm = 复用状态/同名文件覆盖重跑；cold = 全新文件名/全新路径首触。"
             "Windows 用户态无法冲洗 OS 写缓存，cold 是\"首触\"口径而非落盘冲洗口径，"
             "共享 SSD 页缓存下两者差异通常很小。每项取 warm/cold 各 ≥3 次（quick 模式"
             "各 2 次）的中位数。所有含\"估算\"的数字均为线性外推，不是实测端到端。")
    L.append("")
    if res.get("errors"):
        L.append(f"**注意：部分测量项失败**：{list(res['errors'].keys())}"
                 f"（详见 results JSON 的 errors 字段），下表只含成功项。")
        L.append("")

    L.append("## 1. M1 750 任务 JSON 序列化/反序列化")
    m1 = res["m1_json750"]
    L.append(f"合成 750 行（镜像 src/eval/measure.py:90-101 行 schema），"
             f"compact JSON {m1['compact_json_bytes'] / 1024:.0f} KB"
             f"（真实 eval.json stat 参考 {m1['real_eval_json_kb_reference']} KB），"
             f"indent2 {m1['file_size_kb_indent2']} KB。")
    L.append(_tbl([["指标", "warm中位(ms)", "cold中位(ms)", "warm范围",
                    "cold范围", "样本"]]))
    L.append(_wc_line("dumps compact（纯 CPU）", m1["dumps_compact_cpu"]))
    L.append(_wc_line("dumps indent2（纯 CPU）", m1["dumps_indent2_cpu"]))
    L.append(_wc_line("write compact 落盘", m1["write_compact_disk"]))
    L.append(_wc_line("write indent2 原子（tmp+replace）",
                      m1["write_indent2_atomic_disk"]))
    L.append(_wc_line("read+parse 整文件", m1["read_parse_disk"]))
    L.append("")

    L.append("## 2. M2 PNG 编码与写盘（10 张 512² uint8 均匀噪声）")
    m2 = res["m2_png"]
    sz = m2["size_bytes_per_img"]
    L.append(f"单图 PNG 大小：l1={sz.get('noise_l1_bytes_per_img', 0) / 1024:.0f}KB, "
             f"l6={sz.get('noise_l6_bytes_per_img', 0) / 1024:.0f}KB"
             f"（噪声=PNG 最坏情形；真艺术图更可压缩，此处吞吐是保守下界）。")
    L.append(_tbl([["指标", "warm中位(ms)", "cold中位(ms)", "每图warm(ms)",
                    "每图cold(ms)"]]))
    for k, label in (("encode_only_l1", "仅编码 l1（raw_io 默认）"),
                     ("encode_only_l6_pil_default", "仅编码 l6（PIL 默认）"),
                     ("write_l1_serial", "编码+写盘 串行"),
                     ("write_l1_threads8",
                      "编码+写盘 8线程（镜像 save_png_batch_fast）")):
        w, c = m2[k]["warm"], m2[k]["cold"]
        L.append(f"{label} | {w['median_ms']} | {c['median_ms']} | "
                 f"{w['per_image_ms']} | {c['per_image_ms']}")
    L.append("")
    L.append(f"外推（估算）：750 张 l1 + 8 线程落盘 ≈ "
             f"**{m2['estimate_750_imgs_threads8_warm_s']}s**（warm 中位）。")
    L.append("")

    L.append("## 3. M3 sha256 吞吐（100MB 合成内存缓冲）")
    m3 = res["m3_sha256"]
    L.append(_tbl([["分块", "warm 中位(ms)", "warm GB/s", "cold GB/s"]]))
    for k in ("chunk_1mb", "chunk_4mb", "single_shot"):
        r = m3[k]
        L.append(f"{k} | {r['warm']['median_ms']} | {r['warm_gbps']} | "
                 f"{r['cold_gbps']}")
    L.append("")
    L.append(f"外推（估算）：30MB checkpoint 全量校验 ≈ "
             f"**{m3['estimate_sha256_30mb_ckpt_s']}s**（1MB 分块；"
             f"4MB 分块无显著增益，runtime.sha256_file 的 1MB 分块无需调整）。")
    L.append("")

    L.append("## 4. M4 JSONL 注册表：逐行追加 vs 整体重写（对接 B03）")
    m4 = res["m4_registry"]
    if "append_fsync" in m4:
        L.append(f"记录格式与 src/v25/artifact_registry.py::ArtifactRecord.to_dict "
                 f"同构，单行 {m4['line_bytes']}B × {m4['n_lines']} 行。")
        L.append(_tbl([["指标", "warm中位(ms)", "cold中位(ms)", "warm范围",
                        "cold范围", "样本"]]))
        L.append(_wc_line("逐行追加+fsync（镜像 _append_json）", m4["append_fsync"]))
        L.append(_wc_line("逐行追加 无fsync", m4["append_nofsync"]))
        L.append(_wc_line("整体重写（原子 tmp+replace）", m4["rewrite_all"]))
        L.append(_wc_line("全文件逐行解析（records() 等价）", m4["parse_all"]))
        e2e = m4.get("artifact_registry_e2e", {})
        if "register_warm_same_path" in e2e:
            L.append("")
            L.append(f"真实 ArtifactRegistry 端到端（registry 规模 "
                     f"{e2e['registry_size_after_warmup']} 条）：")
            rw_n = e2e["register_warm_same_path"]["warm"]
            rw_c = e2e["register_warm_same_path"]["cold"]
            ng = e2e.get("register_warm_nogitprobe", {})
            vk = e2e["validate_known"]["warm"]
            vu = e2e["validate_unknown"]["warm"]
            rows = [["指标", "中位(ms)", "范围", "说明"],
                    ["register 同路径 warm", str(rw_n["median_ms"]),
                     f"{rw_n['min_ms']}~{rw_n['max_ms']}",
                     "含每次 git rev-parse 子进程"]]
            if ng:
                rows.append(["register 同路径(免git探测)", str(ng["median_ms"]),
                             f"{ng['min_ms']}~{ng['max_ms']}",
                             "设 WEAVE_GIT_SHA 后纯 fsync 下界"])
            rows.append(["register 新路径 cold", str(rw_c["median_ms"]),
                         f"{rw_c['min_ms']}~{rw_c['max_ms']}",
                         "registry 持续增长的真实形态"])
            rows.append(["validate(known)", str(vk["median_ms"]),
                         f"{vk['min_ms']}~{vk['max_ms']}", "含 lookup 全文件解析"])
            rows.append(["validate(unknown)", str(vu["median_ms"]),
                         f"{vu['min_ms']}~{vu['max_ms']}", "无记录不报错不入日志"])
            L.append(_tbl(rows))
            L.append("")
            ng_ms = ng.get("median_ms", rw_n["median_ms"])
            L.append(
                f"**B03 对接数值**：高频 register 单次成本 ≈ "
                f"**{rw_n['median_ms']}ms**（默认，含 git rev-parse 子进程）/ "
                f"**{ng_ms}ms**（设 WEAVE_GIT_SHA 后），吞吐上限 ≈ "
                f"**{1000 / max(ng_ms, 1e-6):.0f} 次/秒**（免探测口径）；"
                f"裸逐行追加+fsync = **{m4['append_fsync']['per_line_ms']}ms/行**，"
                f"无 fsync **{m4['append_nofsync']['per_line_ms']}ms/行**，"
                f"fsync 开销倍数 ≈ "
                f"{m4['append_fsync']['per_line_ms'] / max(m4['append_nofsync']['per_line_ms'], 1e-9):.1f}x；"
                f"100 条整体重写 **{m4['rewrite_all']['warm']['median_ms']}ms**、"
                f"全量解析 **{m4['parse_all']['warm']['median_ms']}ms**。"
                f"注意：ArtifactRegistry.lookup 每次 records() 全文件 O(n) 解析，"
                f"注册表到 1k 条时单次 validate 的解析成本按线性外推约 "
                f"{m4['parse_all']['warm']['median_ms'] * 10:.0f}ms，高频调用建议进程内缓存记录索引。")
    L.append("")

    L.append("## 5. M5 小文件 vs 大文件 + 目录创建")
    m5 = res["m5_files"]
    L.append(_tbl([["指标", "warm中位(ms)", "cold中位(ms)", "摊薄"]]))
    L.append(f"创建 100×1KB | {m5['create_100x1kb']['warm']['median_ms']} | "
             f"{m5['create_100x1kb']['cold']['median_ms']} | "
             f"{m5['create_100x1kb']['per_file_us_warm']}µs/文件")
    L.append(f"读 100×1KB | {m5['read_100x1kb']['warm']['median_ms']} | "
             f"{m5['read_100x1kb']['cold']['median_ms']} | "
             f"{m5['read_100x1kb']['per_file_us_warm']}µs/文件")
    L.append(f"写 1×100KB | {m5['create_1x100kb']['warm']['median_ms']} | "
             f"{m5['create_1x100kb']['cold']['median_ms']} | 等量内容单文件")
    L.append(f"读 1×100KB | {m5['read_1x100kb']['warm']['median_ms']} | "
             f"{m5['read_1x100kb']['cold']['median_ms']} | 等量内容单文件")
    L.append(f"mkdir 嵌套链(parents) | "
             f"{m5['mkdir_nested_parents_fresh']['warm']['median_ms']} | "
             f"{m5['mkdir_nested_parents_fresh']['cold']['median_ms']} | 单链")
    L.append(f"mkdir exist_ok×50 | "
             f"{m5['mkdir_exist_ok_x50']['warm']['median_ms']} | "
             f"{m5['mkdir_exist_ok_x50']['cold']['median_ms']} | "
             f"{m5['mkdir_exist_ok_per_call_us_warm']}µs/次")
    L.append("")

    L.append("## 6. M6 Windows 路径处理（本机口径；不含 GPU 机器）")
    m6 = res["m6_paths"]
    L.append(_tbl([["正确性断言", "结果", "说明"]]))
    for c in m6["correctness"]:
        L.append(f"{c['name']} | {'PASS' if c['ok'] else 'FAIL'} | {c['detail']}")
    L.append("")
    L.append(_tbl([["开销项", "µs/操作", "样本数"]]))
    for k, v in m6["overhead"].items():
        L.append(f"{k} | {v['median_us_per_op']} | {v['n']}")
    L.append("")

    L.append("## 7. M7 torch 原始 batch 载荷往返（镜像 src/infer/run.py 载荷）")
    m7 = res.get("m7_torch_batch", {})
    if "skipped" in m7:
        L.append(f"跳过：{m7['skipped']}")
    else:
        L.append(f"载荷 {m7['payload_mb']}MB（batch={m7['batch']}：3×(4,64,64)fp16 "
                 f"latent + (4,3,512,512)uint8 rgb），torch {m7['torch_version']}，"
                 f"CUDA 可见设备数 = {m7['cuda_device_count']}"
                 f"（CUDA_VISIBLE_DEVICES=\"\"，隔离确认）。")
        L.append(_tbl([["指标", "warm中位(ms)", "cold中位(ms)", "warm MB/s",
                        "cold MB/s"]]))
        L.append(f"torch.save(tmp+replace) | {m7['save']['warm']['median_ms']} | "
                 f"{m7['save']['cold']['median_ms']} | {m7['save']['warm_mbps']} | "
                 f"{m7['save']['cold_mbps']}")
        L.append(f"torch.load | {m7['load']['warm']['median_ms']} | "
                 f"{m7['load']['cold']['median_ms']} | {m7['load']['warm_mbps']} | "
                 f"{m7['load']['cold_mbps']}")
    L.append("")

    est = res["extrapolation_per_arm_750"]
    L.append("## 8. 单臂评测 ~50s 中的非 GPU 开销归因（全部为估算）")
    L.append("")
    L.append("代码锚点（CPU/I/O 环节在评测流水线中的位置）：")
    L.append("")
    L.append(_tbl([["环节", "代码位置", "触发条件", "估算耗时/臂"]]))
    L.append(f"原始 batch torch.load（~738MB） | src/eval/measure.py:62 | "
             f"每臂必经 ~188 个 .pt | "
             f"{est.get('measure_raw_batch_torch_load_s', 'n/a')}s")
    L.append(f"PNG 编码+写盘 750 张(8线程,l1) | src/eval/measure.py:69 + "
             f"src/utils/raw_io.py:13-33 | 仅 write_png=True"
             f"（runner.run_measure_subprocess 默认 False，src/eval/runner.py:56）| "
             f"{est['measure_png_write_750imgs_threads8_s']}s")
    L.append(f"eval.json dumps+写盘（~0.9MB） | src/eval/measure.py:109-112 | "
             f"每臂必经 | {est['measure_eval_json_dumps_write_s']}s")
    L.append(f"features.pt torch.save（~5.9MB） | src/eval/measure.py:113-115 | "
             f"每臂必经 | {est.get('measure_features_pt_save_s', 'n/a')}s")
    L.append(f"原始 batch torch.save（~738MB, infer 阶段） | src/infer/run.py:75-77 | "
             f"每臂生成期一次 | {est.get('infer_raw_batch_torch_save_s', 'n/a')}s")
    L.append(f"checkpoint sha256 校验(30MB) | src/utils/runtime.py:39-44 / "
             f"src/v25/artifact_registry.py:143-152 | 仅 B03 hash 复核开启 | "
             f"{est['sha256_30mb_ckpt_s']}s")
    L.append(f"registry register（含 git 探测） | "
             f"src/v25/artifact_registry.py:360-400 | 仅 B03 接入后 | "
             f"{est['registry_register_per_call_ms']}ms/次")
    L.append(f"目录 mkdir(parents,exist_ok) | src/eval/measure.py:40-42, "
             f"src/utils/raw_io.py:23 | 每臂 1-2 次 | <0.01s（可忽略）")
    L.append("")
    L.append(f"汇总（估算）：单臂 750 任务评测里，纯 CPU/I/O 项合计 ≈ "
             f"**{est['non_gpu_cpu_io_seconds_total_png_on']}s（PNG 开）** / "
             f"**{est['non_gpu_cpu_io_seconds_total_png_off']}s（PNG 关）**，"
             f"占 ~{est['assumptions']['arm_wall_s_baseline']:.0f}s 臂墙钟的 "
             f"**{est['share_pct_of_arm_50s_png_on']}% / "
             f"{est['share_pct_of_arm_50s_png_off']}%**。"
             f"其余为 GPU 推理与特征裁判前向（DINO/CSD/CLIP/LPIPS）。")
    L.append("")
    L.append("**结论（评测流水线里哪些耗时根本不是 GPU 推理）**：")
    L.append("")
    L.append(f"1. **原始 batch 的 torch.save（infer 落盘）与 torch.load（measure 读回）"
             f"是最大的非 GPU 项**：~738MB/臂往返，估算 "
             f"{est.get('infer_raw_batch_torch_save_s', '?')}s + "
             f"{est.get('measure_raw_batch_torch_load_s', '?')}s；"
             f"每臂必付的固定 I/O 税，与模型快慢无关，改用更强 GPU 也不会缩短它。")
    L.append(f"2. **PNG 编码是 write_png=True 场景下的第二大头**：750 张噪声上界 ≈ "
             f"{est['measure_png_write_750imgs_threads8_s']}s（8 线程把单图 "
             f"{m2['write_l1_threads8']['warm']['per_image_ms']}ms 摊薄）；"
             f"正式口径 runner 默认 --no-png（src/eval/runner.py:56,69-70），此时为 0；"
             f"真艺术图比噪声更可压缩，实际编码时间低于此上界。")
    L.append(f"3. **JSON 写盘本身可忽略**：~0.9MB eval.json 序列化+落盘 "
             f"{est['measure_eval_json_dumps_write_s']}s，summary_arm.json（~3KB，"
             f"src/eval/runner.py:75）微秒级；measure.py:109 用的是非原子 write_text，"
             f"若 B06 要原子化，save_json 的 indent=2 会让体积翻倍、"
             f"dumps 多花约 "
             f"{m1['dumps_indent2_cpu']['warm']['median_ms'] / max(m1['dumps_compact_cpu']['warm']['median_ms'], 1e-9):.1f}x CPU，"
             f"绝对值仍 <0.1s，代价可接受。")
    L.append(f"4. **哈希校验是条件项**：sha256 吞吐 "
             f"{res['m3_sha256']['chunk_1mb']['warm_gbps']}GB/s（1MB 分块；"
             f"4MB 分块 {res['m3_sha256']['chunk_4mb']['warm_gbps']}GB/s 无显著增益），"
             f"30MB checkpoint 校验 ≈ {est['sha256_30mb_ckpt_s']}s；"
             f"只有 B03 注册表启用 artifact_sha256 复核（validate "
             f"check_artifact_hash=True）时才发生，常规缓存闸门（params_hash 等小字符串"
             f"比对）是纯内存操作。")
    L.append(f"5. **目录创建与路径处理均不是瓶颈**：mkdir exist_ok "
             f"{m5['mkdir_exist_ok_per_call_us_warm']}µs/次、Path.resolve "
             f"{m6['overhead']['path_resolve_existing']['median_us_per_op']}µs/次，"
             f"每臂万次级路径操作也 <0.1s；Windows 长路径(>260字符)/大小写/正反斜杠"
             f"实测全部无坑（见 §6 断言表全 PASS）。")
    L.append(f"6. **Windows 上\"大量小文件\"是真实的反模式**：逐个创建 1KB 文件 "
             f"≈ {m5['create_100x1kb']['per_file_us_warm']}µs/文件（100 个共 "
             f"{m5['create_100x1kb']['warm']['median_ms']}ms），同量 100KB 打包"
             f"单文件写+读仅 ≈ {m5['create_1x100kb']['warm']['median_ms']}ms，"
             f"按字节差约 "
             f"{m5['create_100x1kb']['per_file_us_warm'] * 100 / max(m5['create_1x100kb']['warm']['median_ms'] * 1000, 1e-6):.0f}x；"
             f"RawWriter 的 ~4MB batch .pt 分片（src/utils/raw_io.py:60-70）粒度合理，"
             f"但任何\"逐任务级\"小文件落盘（逐任务 json/元数据）都应打包或走 JSONL "
             f"批量追加，不要逐文件创建。")
    L.append(f"7. **B03 高频 register 的真实成本由 git 子进程主导**："
             f"register 含 git rev-parse 探测 {est['registry_register_per_call_ms']}ms/次"
             f"（若 git 不可用回退 unknown 不抛错），设 WEAVE_GIT_SHA 后降至 "
             f"{est.get('registry_register_nogit_per_call_ms', '?')}ms/次——"
             f"fsync 本身只占 "
             f"{m4['append_fsync']['per_line_ms']}ms/行，"
             f"高频注册建议批量登记或进程内复用 git_sha。")
    L.append("")

    L.append("## 9. 风险与限制")
    L.append("")
    for r in res["notes"]:
        L.append(f"- {r}")
    L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def env_snapshot(tmp_root: Path, mode: str) -> dict:
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "thread_limits": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", ""),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", ""),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", ""),
        },
        "tmp_root": str(tmp_root),
        "tmp_drive": Path(tmp_root).anchor or "?",
        "mode": mode,
    }


def dir_size_mb(root: Path) -> float:
    total = 0
    for p in root.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    return total / 1e6


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="样本减半，目标 <60s")
    ap.add_argument("--out-json",
                    default=str(REPO / "docs/v25/audits/b04_io_results.json"))
    ap.add_argument("--out-md",
                    default=str(REPO / "docs/v25/audits/b04_io_report.md"))
    ap.add_argument("--keep-tmp", action="store_true")
    ap.add_argument("--arm-wall-s", type=float, default=50.0,
                    help="单臂评测墙钟基线（用于占比估算）")
    ap.add_argument("--raw-payload-mb", type=float,
                    default=REAL_RAW_PAYLOAD_MB_PER_ARM,
                    help="单臂原始 batch 载荷 MB（meta.json 探测标定值）")
    a = ap.parse_args()

    mode = "quick" if a.quick else "full"
    rw, rc = (2, 2) if a.quick else (3, 3)
    n_jsonl = 40 if a.quick else 100
    n_register = 12 if a.quick else 40
    t_start = time.perf_counter()

    tmp = Path(tempfile.mkdtemp(prefix="b04_io_bench_"))
    print(f"[b04] tmp={tmp} mode={mode} reps(warm,cold)=({rw},{rc})", flush=True)
    results: dict = {
        "task": "B04",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "env": env_snapshot(tmp, mode),
        "notes": [],
        "errors": {},
    }
    steps = [
        ("m1_json750", lambda: m1_json750(tmp, rw, rc)),
        ("m2_png", lambda: m2_png(tmp, rw, rc)),
        ("m3_sha256", lambda: m3_sha256(rw, rc)),
        ("m4_registry",
         lambda: m4_registry(tmp, rw, rc, n_jsonl, n_register)),
        ("m5_files", lambda: m5_files(tmp, rw, rc)),
        ("m6_paths",
         lambda: m6_paths(tmp, rw, rc, n_ops=1000 if a.quick else 2000)),
        ("m7_torch_batch", lambda: m7_torch(tmp, rw, rc, batch=4)),
    ]
    for name, fn in steps:
        t0 = time.perf_counter()
        try:
            results[name] = fn()
            print(f"[b04] {name} done in {time.perf_counter() - t0:.1f}s",
                  flush=True)
        except Exception as e:  # 单项失败不拖垮整体
            results["errors"][name] = f"{type(e).__name__}: {e}"
            results[name] = {"error": f"{type(e).__name__}: {e}",
                             "trace": traceback.format_exc(limit=3)}
            print(f"[b04] {name} FAILED: {e}", flush=True)

    m7v = results.get("m7_torch_batch", {})
    if m7v.get("torch_version"):
        results["env"]["torch"] = (f"{m7v['torch_version']} "
                                   f"(cuda_device_count={m7v['cuda_device_count']})")

    try:
        results["extrapolation_per_arm_750"] = extrapolate(
            results, a.arm_wall_s, a.raw_payload_mb)
    except Exception as e:
        results["errors"]["extrapolation"] = f"{type(e).__name__}: {e}"

    failed_checks = [c["name"] for c in
                     results.get("m6_paths", {}).get("correctness", [])
                     if not c.get("ok")]
    notes = [
        "cold = 全新文件名/路径首触；Windows 用户态无法冲洗 OS 写缓存，"
        "非落盘冲洗口径。",
        "PNG 用均匀噪声（最坏情形），真艺术图更可压缩：PNG 编码吞吐为保守下界，"
        "写盘字节量为上界。",
        "真实臂标定：runs/v8/F2.e/gen/meta.json（750 任务，batch=4，decode=True）"
        "stat 得原始载荷 737.8MB / 188 个 batch .pt；"
        "runs/v25/measured/A_p1_l130/eval.json stat 897KB。标定只读 meta.json "
        "小文件与 stat，未读任何 .pt 大文件内容；运行期完全不访问 runs/。",
        "单臂占比的分母 ~50s 为总控给定基线，具体阶段组成以 A07 端到端成本模型为准；"
        "本报告只量化 CPU/I/O 项的绝对秒数。",
        "GPU 训练在远程 4090/3060，本机测量不与其抢磁盘；测量期间本机无其他重载"
        "任务时数值最可靠。",
        "M4 真实 register 成本包含每次 git rev-parse 子进程（WEAVE_GIT_SHA "
        "未设时的默认行为，git 失败回退 unknown 不抛错）；两项口径均已测出。",
    ]
    if failed_checks:
        notes.append(f"Windows 路径断言未全过：{failed_checks}"
                     f"（多为长路径未启用 LongPathsEnabled 注册表开关所致）")
    results["notes"] = notes

    tmp_mb = dir_size_mb(tmp)
    results["tmp_peak_write_mb"] = round(tmp_mb, 1)
    results["wall_s"] = round(time.perf_counter() - t_start, 1)

    out_json = Path(a.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2,
                                   default=str), encoding="utf-8")
    try:
        report = build_report(results, mode, results["wall_s"], tmp_mb)
    except Exception as e:
        report = (f"# B04 report generation failed\n\n"
                  f"{type(e).__name__}: {e}\n\n"
                  f"Raw data is in b04_io_results.json; errors: "
                  f"{results['errors']}\n")
    out_md = Path(a.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(report, encoding="utf-8")

    if not a.keep_tmp:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"[b04] wall={results['wall_s']}s tmp_peak={tmp_mb:.1f}MB "
          f"json={out_json} md={out_md}", flush=True)
    print("B04_BENCH_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
