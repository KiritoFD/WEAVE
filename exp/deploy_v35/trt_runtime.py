"""TensorRT 10.7 FP16 / Mixed-INT8 builder and execution helper for CLEAVE V35."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np
import tensorrt as trt
import torch


class _TensorEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    """Stream calibration tensors directly from a single preloaded CUDA/CPU tensor."""

    def __init__(self, samples: torch.Tensor, cache_path: Path, batch_size: int) -> None:
        super().__init__()
        self.samples = samples.contiguous().float().cpu()
        self.cache_path = cache_path
        self.batch_size = int(batch_size)
        self.idx = 0
        self.dev_buf = torch.empty((self.batch_size, *self.samples.shape[1:]), device="cuda", dtype=torch.float32)

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_batch(self, names: list[str]) -> list[int] | None:
        if self.idx + self.batch_size > self.samples.shape[0]:
            return None
        batch = self.samples[self.idx : self.idx + self.batch_size]
        self.idx += self.batch_size
        self.dev_buf.copy_(batch)
        return [int(self.dev_buf.data_ptr())]

    def read_calibration_cache(self) -> bytes | None:
        return self.cache_path.read_bytes() if self.cache_path.exists() else None

    def write_calibration_cache(self, cache: bytes) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_bytes(bytes(cache))


class TrtRunner:
    """Zero-copy GPU execution wrapper around a fixed-shape serialized TensorRT 10 engine."""

    def __init__(self, plan_path: Path, input_names: list[str], output_names: list[str], out_dtypes: dict[str, torch.dtype] | None = None) -> None:
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.rt = trt.Runtime(self.logger)
        self.engine = self.rt.deserialize_cuda_engine(plan_path.read_bytes())
        self.ctx = self.engine.create_execution_context()
        self.input_names = input_names
        self.output_names = output_names
        self.outputs: dict[str, torch.Tensor] = {}
        out_dtypes = out_dtypes or {}
        for name in output_names:
            shape = tuple(self.ctx.get_tensor_shape(name))
            dt = out_dtypes.get(name, torch.from_numpy(np.empty((), dtype=trt.nptype(self.engine.get_tensor_dtype(name)))).dtype)
            self.outputs[name] = torch.empty(shape, device="cuda", dtype=dt)

    def __call__(self, *inputs: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        assert len(inputs) == len(self.input_names)
        keep: list[torch.Tensor] = []
        for name, t in zip(self.input_names, inputs):
            dtype = torch.from_numpy(np.empty((), dtype=trt.nptype(self.engine.get_tensor_dtype(name)))).dtype
            t_c = t.to(dtype=dtype).contiguous()
            if tuple(t_c.shape) != tuple(self.ctx.get_tensor_shape(name)):
                raise ValueError(f'TRT input shape mismatch: {name} {tuple(t_c.shape)}')
            keep.append(t_c)
            self.ctx.set_tensor_address(name, t_c.data_ptr())
        for name in self.output_names:
            self.ctx.set_tensor_address(name, self.outputs[name].data_ptr())
        stream = torch.cuda.current_stream().cuda_stream
        ok = self.ctx.execute_async_v3(stream)
        assert ok, "TensorRT execute_async_v3 failed"
        if len(self.output_names) == 1:
            return self.outputs[self.output_names[0]].clone()
        return tuple(self.outputs[n].clone() for n in self.output_names)


def export_onnx_clean(
    module: torch.nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    onnx_path: Path,
    input_names: list[str],
    output_names: list[str],
    opset: int = 17,
) -> Path:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    if not onnx_path.exists():
        module.eval()
        with torch.no_grad():
            torch.onnx.export(
                module,
                example_inputs,
                str(onnx_path),
                input_names=input_names,
                output_names=output_names,
                opset_version=opset,
                dynamo=False,
            )
    return onnx_path


def build_trt_engine(
    onnx_path: Path,
    plan_path: Path,
    *,
    precision: str = "fp16",  # "fp16" | "int8_full" | "int8_mixed"
    calib_samples: torch.Tensor | None = None,
    calib_batch: int = 4,
    calib_cache: Path | None = None,
    workspace_gb: float = 2.5,
) -> tuple[Path, float]:
    if plan_path.exists():
        return plan_path, 0.0
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    ok = parser.parse_from_file(str(onnx_path))
    if not ok:
        errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError(f"ONNX parse failed for {onnx_path}: {errs}")

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))

    if precision in ("int8_full", "int8_mixed"):
        assert calib_samples is not None and calib_cache is not None
        config.set_flag(trt.BuilderFlag.INT8)
        calibrator = _TensorEntropyCalibrator(calib_samples, calib_cache, batch_size=calib_batch)
        config.int8_calibrator = calibrator
        if precision == "int8_mixed":
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            n_layers = network.num_layers
            conv_indices = [i for i in range(n_layers) if network.get_layer(i).type == trt.LayerType.CONVOLUTION]
            for i in range(n_layers):
                layer = network.get_layer(i)
                ltype = layer.type
                # Keep GroupNorm/Normalization and first/last layers in FP16; never touch Int32/Int64 shape/constant layers
                is_norm = ltype == trt.LayerType.NORMALIZATION
                is_edge_conv = bool(conv_indices) and i in (conv_indices[0], conv_indices[-1])
                if not (is_norm or is_edge_conv):
                    continue
                if any(layer.get_input(ii).dtype in (trt.DataType.INT32, trt.DataType.INT64, trt.DataType.BOOL) for ii in range(layer.num_inputs) if layer.get_input(ii) is not None):
                    continue
                layer.precision = trt.DataType.HALF
                for oi in range(layer.num_outputs):
                    out_t = layer.get_output(oi)
                    if out_t is not None and out_t.dtype in (trt.DataType.INT8, trt.DataType.FLOAT, trt.DataType.HALF):
                        layer.set_output_type(oi, trt.DataType.HALF)

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(f"TRT build failed for {onnx_path} ({precision})")
    plan_path.write_bytes(bytes(plan))
    return plan_path, time.time() - t0


def bench_gpu_callable(fn: Callable[[], torch.Tensor], batch_size: int, warmup: int = 12, iters: int = 60) -> float:
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        _ = fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) / float(batch_size))
    return float(np.percentile(times, 50))
