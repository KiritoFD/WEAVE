"""Arm -> src.infer spec mapping.

Every V8 arm is expressed as one of the registered inference variants, so the
inference code never needs a V8-specific branch. Adding an arm means adding a
row here, not editing src/infer.
"""
from __future__ import annotations

from pathlib import Path

from src.infer import InferSpec

REPO = Path(__file__).resolve().parents[2]
Q2A = REPO / "runs/v6/Q2a_embed_style/Q2a_embed_style.pt"
B4 = REPO / "runs/v6/P1_B4_distill50/P1_B4_distill50.pt"
B3 = REPO / "runs/v6/P1_B3_distill20/P1_B3_distill20.pt"
TRAIN = REPO / "runs/v8/train"

STARTS = {"Q2a": Q2A, "B4": B4, "B3": B3}


def _start(arm) -> Path:
    return STARTS.get(arm.get("start_ckpt") or arm.get("start", "Q2a"), Q2A)


def infer_spec_for(arm: dict) -> InferSpec:
    """Map an arm dict to a registered inference variant."""
    kind = arm["kind"]
    aid = arm["id"]

    if kind == "contract":
        return InferSpec(ckpt=B4, kind="direct", extra={"arm": aid})

    if kind == "control":
        ck = arm.get("control_ckpt")
        return InferSpec(
            ckpt=_start(arm), kind="control", steps=arm.get("steps", 2),
            step_rms=arm.get("step_rms", 0.10), rho=arm.get("rho", 0.30),
            extra={"spec": arm.get("spec", "hier_palette"),
                   "lambda_ratio": arm.get("lambda_ratio", 0.0),
                   "control_ckpt": ck, "arm": aid})

    if kind == "oracle":
        return InferSpec(ckpt=_start(arm), kind="oracle",
                         extra={"oracle_steps": arm.get("steps", 1),
                                "oracle_rms": arm.get("rho", 0.25), "arm": aid})

    if kind == "r1_steps":
        return InferSpec(ckpt=_start(arm), kind="r1_steps",
                         steps=arm.get("steps", 4),
                         step_rms=arm.get("step_rms", 0.08),
                         extra={"topk_frac": arm.get("topk_frac", 0.0), "arm": aid})

    if kind == "supported":
        return InferSpec(ckpt=_start(arm), kind="supported",
                         steps=arm.get("steps", 1), rho=arm.get("rho", 0.25),
                         extra={"lambda_ratio": arm.get("lambda_ratio", 0.10),
                                "critic_ckpt": arm.get("critic_ckpt"),
                                "local_support": arm.get("local_support", False),
                                "arm": aid})

    if kind == "spd_edit":
        return InferSpec(ckpt=_start(arm), kind="control",   # replaced below
                         extra={"arm": aid})

    raise KeyError(f"no inference mapping for arm kind {kind!r} ({aid})")
