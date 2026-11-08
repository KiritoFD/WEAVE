"""The 40-arm experiment table from docs/bridgev10.md section 3.

Each arm is a dict of `train5` CLI flags plus bookkeeping. `implemented=False`
marks arms whose mechanism is not yet available in this codebase, so the suite
driver skips them instead of silently running something else.

Groups
  A  01-04   batch / sampling            (4)
  B  05-10   probability-flow parameterisation (6)
  C  11-15   bridge & EDM mechanisms     (5)
  D  16-23   direct target-distribution supervision (8)  [needs style_loss]
  E  24-31   online alpha-DSBM           (8)  [needs online_bridge]
  F  32-35   shared Gaussian pivot       (4)  [needs pivot]
  G  36-38   five-style shared structure (3)  [36/37/38 partly ready]
  H  39-40   OT minimal validation       (2)
"""
from __future__ import annotations

BASE = dict(
    objective="bidir", edm=1, heads_out=1, style_cond="decomposed", target="drift",
    time_sampling="uniform", sampler="perm", ot_ratio=0.0, ot_cost="rgb",
    eps=0.04, batch=1024, steps=800, width=256, depth=6, heads=4, patch=2,
    warmup=100, lr=2e-4, ema_decay=0.999,
)


def _arm(aid: int, group: str, note: str, implemented: bool = True, **over):
    cfg = dict(BASE)
    cfg.update(over)
    cfg["experiment_id"] = f"{aid:02d}"
    cfg["group"] = group
    cfg["note"] = note
    cfg["implemented"] = implemented
    return cfg


ARMS = [
    # ---------------- A: batch / sampling ----------------
    _arm(1, "A", "B256 3200upd small-batch reference", batch=256, steps=3200),
    _arm(2, "A", "B512 1600upd throughput vs gradient variance", batch=512, steps=1600),
    _arm(3, "A", "B1024 800upd with-replacement sampling", batch=1024, steps=800,
         sampler="replacement"),
    _arm(4, "A", "B1024 800upd balanced permutation (default control)",
         batch=1024, steps=800, sampler="perm"),

    # ---------------- B: parameterisation ----------------
    _arm(5, "B", "single direct velocity with re-derived EDM", objective="single_velocity"),
    _arm(6, "B", "05 without EDM", objective="single_velocity", edm=0),
    _arm(7, "B", "shared trunk, independent forward/reverse heads", heads_out=2),
    _arm(8, "B", "both directions supervised on every bridge sample", objective="bidir_joint"),
    _arm(9, "B", "ten pair-ID embeddings instead of decomposed style embeddings",
         style_cond="pair"),
    _arm(10, "B", "current+score two-head parameterisation", objective="score_current"),

    # ---------------- C: bridge / EDM mechanisms ----------------
    _arm(11, "C", "deterministic CFM, eps=0", objective="single_velocity", eps=0.0),
    _arm(12, "C", "wider stochastic path eps=0.16", eps=0.16),
    _arm(13, "C", "32x32 channel-covariance EDM", implemented=False),
    _arm(14, "C", "predict endpoint expectation, convert to drift", target="endpoint"),
    _arm(15, "C", "stratified time sampling with importance weights",
         time_sampling="stratified"),

    # ---------------- D: direct target-distribution supervision ----------------
    _arm(16, "D", "bidir + learned-representation global SWD", implemented=False),
    _arm(17, "D", "single velocity + learned global SWD", implemented=False),
    _arm(18, "D", "bidir + fixed-latent-statistic global SWD", implemented=False),
    _arm(19, "D", "single velocity + fixed global SWD", implemented=False),
    _arm(20, "D", "bidir + learned local-patch SWD", implemented=False),
    _arm(21, "D", "single velocity + learned local SWD", implemented=False),
    _arm(22, "D", "bidir + global and local SWD", implemented=False),
    _arm(23, "D", "single velocity + global and local SWD", implemented=False),

    # ---------------- E: online alpha-DSBM ----------------
    _arm(24, "E", "bidirectional SDE, EMA teacher, 4 steps", implemented=False),
    _arm(25, "E", "24 with 8-step SDE", implemented=False),
    _arm(26, "E", "24 with raw teacher", implemented=False),
    _arm(27, "E", "one-directional online update", implemented=False),
    _arm(28, "E", "50% independent + 50% online endpoints", implemented=False),
    _arm(29, "E", "deterministic ODE endpoints instead of SDE (reflow control)",
         implemented=False),
    _arm(30, "E", "replay buffer of stale endpoints", implemented=False),
    _arm(31, "E", "dynamic EDM cross covariance during online updates", implemented=False),

    # ---------------- F: shared Gaussian pivot ----------------
    _arm(32, "F", "shared Gaussian->art deterministic CFM", implemented=False),
    _arm(33, "F", "Gaussian->art bidirectional Brownian bridge + EDM", implemented=False),
    _arm(34, "F", "33 + target distribution supervision", implemented=False),
    _arm(35, "F", "33 + online alpha-DSBM", implemented=False),

    # ---------------- G: five-style shared structure ----------------
    _arm(36, "G", "three-style path composition consistency", implemented=False),
    _arm(37, "G", "shared embedding + pair low-rank adapters", pair_rank=8),
    _arm(38, "G", "depthwise local conv residual in DiT blocks", local_conv=True),

    # ---------------- H: OT ----------------
    _arm(39, "H", "five-style bridge + RGB mini-batch OT", ot_ratio=0.5, ot_cost="rgb"),
    _arm(40, "H", "five-style bridge + DINO semantic OT", ot_ratio=0.5, ot_cost="dino"),
]

BY_ID = {a["experiment_id"]: a for a in ARMS}


def get(aid) -> dict:
    key = f"{int(aid):02d}"
    if key not in BY_ID:
        raise KeyError(f"no arm {key}")
    return BY_ID[key]


def select(spec: str) -> list:
    """'all' | 'A' | '04,05,10' | '01-05' | 'impl'."""
    if spec == "all":
        return [a for a in ARMS if a["implemented"]]
    if spec == "impl":
        return [a for a in ARMS if a["implemented"]]
    if spec in "ABCDEFGH" and len(spec) == 1:
        return [a for a in ARMS if a["group"] == spec and a["implemented"]]
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(get(i) for i in range(int(lo), int(hi) + 1))
        else:
            out.append(get(part))
    return [a for a in out if a["implemented"]]
