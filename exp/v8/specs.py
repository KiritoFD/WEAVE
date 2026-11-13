"""Arm -> src.infer spec mapping.

Every V8 arm is expressed as one of the registered inference variants, so the
inference code never needs a V8-specific branch. Adding an arm means adding a
row here, not editing src/infer.

Checkpoint references inside an arm dict are ARM IDS ("A3", "C3", "L4"), not
paths. `_resolve_ckpt` turns them into the latest step file under
runs/v8/train/<arm>/; inference variants must therefore receive real paths in
`extra` and are responsible for loading them — a name that never reaches a
loader is a silent no-op (that is exactly how A3.eval once evaluated the
untrained zero control).
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


def _resolve_ckpt(arm_id: str) -> Path:
    """Latest step_*.pt for a trained arm, e.g. 'A3' -> runs/v8/train/A3/step_03000.pt."""
    d = TRAIN / arm_id
    if not d.is_dir():
        raise FileNotFoundError(f"no trained checkpoint dir for arm {arm_id!r}: {d}")
    steps = sorted(d.glob("step_*.pt"))
    if not steps:
        raise FileNotFoundError(f"no step_*.pt under {d}")
    return steps[-1]


def infer_spec_for(arm: dict) -> InferSpec:
    """Map an arm dict to a registered inference variant."""
    kind = arm["kind"]
    aid = arm["id"]

    def ckpt_of(key):
        v = arm.get(key)
        return str(_resolve_ckpt(v)) if v else None

    if kind == "contract":
        return InferSpec(ckpt=B4, kind="direct", extra={"arm": aid})

    if kind == "control":
        return InferSpec(
            ckpt=_start(arm), kind="control", steps=arm.get("steps", 2),
            step_rms=arm.get("step_rms", 0.10), rho=arm.get("rho", 0.30),
            extra={"spec": arm.get("spec", "hier_palette"),
                   "lambda_ratio": arm.get("lambda_ratio", 0.0),
                   "control_ckpt": ckpt_of("control_ckpt"),
                   "critic_ckpt": ckpt_of("critic_ckpt"),
                   "arm": aid})

    if kind == "oracle":
        return InferSpec(ckpt=_start(arm), kind="oracle",
                         extra={"oracle_steps": arm.get("steps", 1),
                                "oracle_rms": arm.get("rho", 0.25),
                                "critic_ckpt": ckpt_of("critic_ckpt"),
                                "arm": aid})

    if kind == "r1_steps":
        return InferSpec(ckpt=_start(arm), kind="r1_steps",
                         steps=arm.get("steps", 4),
                         step_rms=arm.get("step_rms", 0.08),
                         extra={"topk_frac": arm.get("topk_frac", 0.0),
                                "critic_ckpt": ckpt_of("critic_ckpt"),
                                "arm": aid})

    if kind == "supported":
        return InferSpec(ckpt=_start(arm), kind="supported",
                         steps=arm.get("steps", 1), rho=arm.get("rho", 0.25),
                         extra={"lambda_ratio": arm.get("lambda_ratio", 0.10),
                                "local_support": arm.get("local_support", False),
                                "critic_ckpt": ckpt_of("critic_ckpt"),
                                "arm": aid})

    if kind == "spd_edit":
        return InferSpec(
            ckpt=_start(arm), kind="spd_edit",
            steps=arm.get("steps", 2), step_rms=arm.get("step_rms", 0.08),
            rho=arm.get("rho", 0.25),
            extra={"spec": arm.get("spec", "hier_palette"),
                   "spd_ckpt": ckpt_of("spd_ckpt"),
                   "control_ckpt": ckpt_of("control_ckpt"),
                   "critic_ckpt": ckpt_of("critic_ckpt"),
                   "arm": aid})

    if kind == "spd_latent":
        return InferSpec(
            ckpt=_start(arm), kind="spd_latent",
            steps=arm.get("steps", 2), step_rms=arm.get("step_rms", 0.08),
            rho=arm.get("rho", 0.25),
            extra={"spd_ckpt": ckpt_of("spd_ckpt"),
                   "critic_ckpt": ckpt_of("critic_ckpt"),
                   "eta_search": bool(arm.get("eta_search", False)),
                   "budget_eps": arm.get("budget_eps", 0.15),
                   "arm": aid})

    if kind == "actuated":
        return InferSpec(
            ckpt=_start(arm), kind="actuated",
            steps=arm.get("steps", 0), step_rms=arm.get("step_rms", 0.08),
            rho=arm.get("rho", 0.25),
            extra={"act_ckpt": ckpt_of("act_ckpt"),
                   "critic_ckpt": ckpt_of("critic_ckpt"),
                   "eta": arm.get("eta", 0.05),
                   "arm": aid})

    if kind == "student":
        return InferSpec(ckpt=Path(ckpt_of("student_ckpt") or arm["ckpt"]),
                         kind="student",
                         extra={"start_ckpt": str(_start(arm)), "arm": aid})

    raise KeyError(f"no inference mapping for arm kind {kind!r} ({aid})")
