"""Rewrite the exported fp32 decoder ONNX into a uniform fp16 graph -- CPU only.

Why not ``onnxconverter_common``: it leaves Half/Float mixtures (e.g. attention's
``Div`` gets a Half constant but a Float ``Sqrt``).  onnxruntime silently upcasts,
TensorRT's strict type check rejects the model.  Here we rewrite *all* float
tensors (initializers, Constant node values, value_info, IO) to fp16 uniformly,
keeping shape/scale parameters (Shape/Resize roi/scales/sizes) in fp32 as ONNX
requires.

Usage:
  python tools/trt/convert_onnx_fp16.py --sizes 64,256,512
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

SHAPE_OPS = {"Resize", "Upsample", "Shape", "ConstantOfShape", "Expand", "Tile", "Range"}


def infer(model: onnx.ModelProto) -> onnx.ModelProto:
    try:
        return onnx.shape_inference.infer_shapes(model, strict_mode=False)
    except Exception:  # noqa: BLE001
        return model


def param_slots(model: onnx.ModelProto) -> set[str]:
    """Values used as Resize roi/scales/sizes (must stay fp32)."""
    keep: set[str] = set()
    for node in model.graph.node:
        if node.op_type in SHAPE_OPS:
            keep.update(node.input[1:])
    return keep


def to_fp16(model: onnx.ModelProto) -> tuple[onnx.ModelProto, dict]:
    graph = model.graph
    keep = param_slots(model)

    init_done = 0
    for i, init in enumerate(graph.initializer):
        if init.name in keep or init.data_type != TensorProto.FLOAT:
            continue
        arr = numpy_helper.to_array(init).astype(np.float16)
        graph.initializer[i].CopyFrom(numpy_helper.from_array(arr, init.name))
        init_done += 1

    const_done = 0
    for node in graph.node:
        if node.op_type != "Constant":
            continue
        if node.output and node.output[0] in keep:
            continue
        for attr in node.attribute:
            if attr.name == "value" and attr.type == onnx.AttributeProto.TENSOR \
                    and attr.t.data_type == TensorProto.FLOAT:
                arr = numpy_helper.to_array(attr.t).astype(np.float16)
                attr.t.CopyFrom(numpy_helper.from_array(arr, attr.t.name))
                const_done += 1

    io_done = 0
    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        if vi.type.tensor_type.elem_type == TensorProto.FLOAT:
            vi.type.tensor_type.elem_type = TensorProto.FLOAT16
            io_done += 1

    cast_done = 0
    for node in graph.node:
        if node.op_type != "Cast":
            continue
        for attr in node.attribute:
            if attr.name == "to":
                if attr.i in (TensorProto.FLOAT, TensorProto.DOUBLE):
                    attr.i = TensorProto.FLOAT16
                    cast_done += 1
                break

    # dtype attribute on ops such as EyeLike / RandomNormal
    for node in graph.node:
        for attr in node.attribute:
            if attr.name == "dtype" and attr.i == TensorProto.FLOAT:
                attr.i = TensorProto.FLOAT16

    return model, {"initializers": init_done, "constants": const_done,
                   "value_infos": io_done, "casts": cast_done,
                   "kept_fp32_params": len(keep)}


def validate(path: Path, raw: bytes) -> dict:
    import onnxruntime as ort
    out = {}
    try:
        onnx.checker.check_model(onnx.load(str(path)), full_check=True)
        out["checker"] = "ok"
    except Exception as exc:  # noqa: BLE001
        out["checker"] = repr(exc)[:200]
    try:
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        out["ort_load"] = "ok"
        out["io"] = [i.type for i in sess.get_inputs() + sess.get_outputs()]
    except Exception as exc:  # noqa: BLE001
        out["ort_load"] = repr(exc)[:200]
    try:
        import tensorrt as trt
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        net = builder.create_network(int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        parser = trt.OnnxParser(net, logger)
        out["trt_parse"] = "ok" if parser.parse(raw) else str(parser.get_error(0))[:200]
        del builder, net, parser
    except Exception as exc:  # noqa: BLE001
        out["trt_parse"] = repr(exc)[:200]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx_dir", default="/mnt/i/Latent_Style/WEAVE/runs/trt/onnx")
    ap.add_argument("--sizes", default="64,256,512")
    ap.add_argument("--suffix", default="fp16")
    ap.add_argument("--check_trt", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.onnx_dir)
    report = []
    for size in [int(x) for x in args.sizes.split(",") if x.strip()]:
        src = out_dir / f"vae_decoder_{size}.onnx"
        dst = out_dir / f"vae_decoder_{size}_{args.suffix}.onnx"
        model, stats = to_fp16(infer(onnx.load(str(src))))
        onnx.save(model, str(dst))
        row = {"size": size, "file": str(dst),
               "mb": round(dst.stat().st_size / 1024 / 1024, 1)}
        row.update(stats)
        row.update(validate(dst, dst.read_bytes()) if args.check_trt else {})
        report.append(row)
        print(json.dumps(row), flush=True)
    (out_dir / f"fp16_report_{args.suffix}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

