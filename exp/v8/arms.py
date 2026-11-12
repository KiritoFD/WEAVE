"""Fixed V8 arm matrix (24h plan section 6).

Arm ids are prefixed by route so nothing can silently collide with an earlier
round's names. `train` arms carry an update count; `infer` arms carry the
optimisation settings they run at evaluation time.
"""
from __future__ import annotations

# ---------------------------------------------------------------- P0 contract
P0 = [{"id": "contract", "kind": "contract"}]

# ---------------------------------------------------------------- A: GCE
# A0/A1/A2 are zero-training u-space optimisation at 1/2/4 steps.
A_INFER = [
    {"id": "A0", "kind": "control", "start": "Q2a", "spec": "global",
     "steps": 1, "step_rms": 0.10, "rho": 0.30},
    {"id": "A0.s2", "kind": "control", "start": "Q2a", "spec": "global",
     "steps": 2, "step_rms": 0.10, "rho": 0.30},
    {"id": "A0.s4", "kind": "control", "start": "Q2a", "spec": "global",
     "steps": 4, "step_rms": 0.08, "rho": 0.30},
    {"id": "A1", "kind": "control", "start": "Q2a", "spec": "hier",
     "steps": 1, "step_rms": 0.10, "rho": 0.30},
    {"id": "A1.s2", "kind": "control", "start": "Q2a", "spec": "hier",
     "steps": 2, "step_rms": 0.10, "rho": 0.30},
    {"id": "A1.s4", "kind": "control", "start": "Q2a", "spec": "hier",
     "steps": 4, "step_rms": 0.08, "rho": 0.30},
    {"id": "A2", "kind": "control", "start": "Q2a", "spec": "hier_palette",
     "steps": 1, "step_rms": 0.10, "rho": 0.30},
    {"id": "A2.s2", "kind": "control", "start": "Q2a", "spec": "hier_palette",
     "steps": 2, "step_rms": 0.10, "rho": 0.30},
    {"id": "A2.s4", "kind": "control", "start": "Q2a", "spec": "hier_palette",
     "steps": 4, "step_rms": 0.08, "rho": 0.30},
    {"id": "A2b", "kind": "control", "start": "B4", "spec": "hier_palette",
     "steps": 2, "step_rms": 0.10, "rho": 0.30},
]
A_TRAIN = [
    {"id": "A3", "kind": "train_control", "start": "Q2a", "spec": "hier_palette",
     "updates": 3000, "seed": 42, "checkpoints": [0, 1500, 3000]},
    {"id": "A4", "kind": "train_control", "start": "Q2a", "spec": "hier_palette_local",
     "updates": 3000, "seed": 42, "checkpoints": [0, 1500, 3000]},
]
A_EVAL_AFTER = [
    {"id": "A3.eval", "kind": "control", "start": "Q2a", "control_ckpt": "A3",
     "steps": 2, "step_rms": 0.10, "rho": 0.30},
    {"id": "A4.eval", "kind": "control", "start": "Q2a", "control_ckpt": "A4",
     "steps": 2, "step_rms": 0.10, "rho": 0.30},
]

# ---------------------------------------------------------------- B: LGG
B_INFER = [
    {"id": "L0.r08", "kind": "oracle", "start": "Q2a", "start_mode": "Q2a",
     "steps": 1, "rho": 0.08},
    {"id": "L0.r15", "kind": "oracle", "start": "Q2a", "start_mode": "Q2a",
     "steps": 1, "rho": 0.15},
    {"id": "L0.r25", "kind": "oracle", "start": "Q2a", "start_mode": "Q2a",
     "steps": 1, "rho": 0.25},
    {"id": "L1.b4.r25", "kind": "oracle", "start": "B4", "start_mode": "B4",
     "steps": 1, "rho": 0.25},
    {"id": "L1.b4.r15", "kind": "oracle", "start": "B4", "start_mode": "B4",
     "steps": 1, "rho": 0.15},
    {"id": "L1.topk", "kind": "r1_steps", "start": "B4", "topk_frac": 0.25,
     "steps": 4, "step_rms": 0.08},
]
B_TRAIN = [
    {"id": "L2", "kind": "train_spd", "space": "latent", "start": "B4",
     "rank": 0, "steps_k": 2, "updates": 2000, "seed": 42,
     "checkpoints": [0, 1000, 2000]},
    {"id": "L3", "kind": "train_spd", "space": "latent", "start": "B4",
     "rank": 2, "steps_k": 4, "updates": 2000, "seed": 42,
     "checkpoints": [0, 1000, 2000]},
    {"id": "L4", "kind": "train_spd", "space": "control", "start": "Q2a",
     "control_ckpt": "A3", "steps_k": 2, "updates": 2000, "seed": 42,
     "checkpoints": [0, 1000, 2000]},
]

# ---------------------------------------------------------------- C: RSDE
C_INFER = [
    {"id": "C0", "kind": "oracle", "start": "Q2a", "lambda_ratio": 0.0,
     "steps": 1, "rho": 0.25},
    {"id": "C1.l010", "kind": "supported", "start": "Q2a", "lambda_ratio": 0.10,
     "steps": 1, "rho": 0.25},
    {"id": "C1.l030", "kind": "supported", "start": "Q2a", "lambda_ratio": 0.30,
     "steps": 1, "rho": 0.25},
    {"id": "C2", "kind": "supported", "start": "Q2a", "lambda_ratio": 0.10,
     "local_support": True, "steps": 1, "rho": 0.25},
    {"id": "C4", "kind": "supported", "start": "Q2a", "lambda_ratio": 0.10,
     "critic_ckpt": "C3", "steps": 1, "rho": 0.25},
]
C_TRAIN = [
    {"id": "C3", "kind": "train_critic", "updates": 2500, "seed": 42,
     "checkpoints": [0, 1250, 2500]},
]

# ---------------------------------------------------------------- I: joint
I_ARMS = [
    {"id": "I0", "kind": "control", "start": "Q2a", "control_ckpt": "A3",
     "steps": 2, "step_rms": 0.10, "rho": 0.30, "lambda_ratio": 0.0},
    {"id": "I1", "kind": "control", "start": "Q2a", "control_ckpt": "A3",
     "steps": 2, "step_rms": 0.10, "rho": 0.30, "lambda_ratio": 0.10},
    {"id": "I2", "kind": "spd_edit", "start": "Q2a", "control_ckpt": "A3",
     "spd_ckpt": "L4", "lambda_ratio": 0.0},
    {"id": "I3", "kind": "spd_edit", "start": "Q2a", "control_ckpt": "A3",
     "spd_ckpt": "L4", "lambda_ratio": 0.10},
]

GROUPS = {
    "P0": P0,
    "A_infer": A_INFER,
    "A_train": A_TRAIN,
    "A_eval": A_EVAL_AFTER,
    "B_infer": B_INFER,
    "B_train": B_TRAIN,
    "C_infer": C_INFER,
    "C_train": C_TRAIN,
    "I": I_ARMS,
}


def all_arms():
    return [(g, a) for g, arms in GROUPS.items() for a in arms]


def total_updates():
    return sum(a.get("updates", 0) for _, a in all_arms())


if __name__ == "__main__":
    print({k: len(v) for k, v in GROUPS.items()})
    print("infer arms:", sum(len(v) for k, v in GROUPS.items() if "train" not in k))
    print("train arms:", sum(len(v) for k, v in GROUPS.items() if "train" in k))
    print("total updates:", total_updates())
