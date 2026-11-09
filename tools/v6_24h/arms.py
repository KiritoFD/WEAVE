"""V6 24h plan - arm table.

Every arm is a single explicit change from the A0 recipe. `args` holds only the deltas;
the runner merges them onto BASE, so no arm can silently drift on a shared setting.

A0 recipe (reference, already trained):
  mode=direct canon=c1_gentle base_dim=48 steps=800 microbatch=80 accum=3 lr=3e-4
  coherence_beta=0 w_rec=1.0 w_style_dist=0.25 w_cross=0.35 w_canon_inv=1.5 seed=42
  -> 1,881,792 params, eval200 dT +0.0323 / DINO-C 0.9167 / LPIPS 0.3658
"""
from __future__ import annotations

BASE = {
    "mode": "direct", "canon_mode": "c1_gentle", "base_dim": 48, "steps": 800,
    "microbatch": 80, "accum": 3, "lr": 3e-4, "coherence_beta": 0.0,
    "w_rec": 1.0, "w_style_dist": 0.25, "w_cross": 0.35, "w_canon_inv": 1.5,
    "seed": 42,
}

# group order follows the plan: subtraction I, subtraction II, addition.
ARMS = [
    # ---------------- 减法 I: structure & necessity (plan section 4) ----------------
    dict(id="N1_no_style", group="N", note="Style-AdaLN and PaletteHead both pinned to id 0; "
         "losses still use the true target",
         args={"no_style": True}),
    dict(id="N2_no_canon", group="N", note="c_in = z (no canonicalization); L_canon_inv unchanged",
         args={"canon_mode": "none"}),
    dict(id="N3_dim32", group="N", note="base_dim 48 -> 32 (~0.85M)",
         args={"base_dim": 32}),
    dict(id="N4_steps400", group="N", note="training budget 800 -> 400, own LR cycle",
         args={"steps": 400}),
    dict(id="N5_inv_off", group="N", note="lam_c 0.525 -> 0, lam_s fixed 0.35 (decoupled form)",
         args={"lambda_cross": 0.35, "lambda_canon": 0.0}),
    # N6_b1_off was REMOVED from the queue after P0 measurement: the B1 branch is loaded
    # but numerically irrelevant (its term is 0.000697 of a 1.451 style loss = 0.048%, and
    # the style-branch gradient is bit-identical with it disabled: 1.691895 vs 1.691895).
    # Spending 42 min to confirm a measured no-op is not justified; the evidence is
    # recorded in docs/v6/V6_24H_P0_AUDIT.md and Negative_Results instead.
    dict(id="N7_dim24", group="N", note="minimal width probe (only if N3 passes)",
         args={"base_dim": 24}, conditional="dim24"),
    dict(id="N7_dim40", group="N", note="width probe (if N3 fails)",
         args={"base_dim": 40}, conditional="dim40"),

    # ---------------- 减法 II: losses & data (plan section 5) ----------------
    dict(id="L1_lowC", group="L", note="lam_s 0.35, lam_c 0.240 (A0-style lam_s, weaker content)",
         args={"lambda_cross": 0.35, "lambda_canon": 0.240}),
    dict(id="L2_highS", group="L", note="lam_s 1.20, lam_c 0.525 (stronger style only)",
         args={"lambda_cross": 1.20, "lambda_canon": 0.525}),
    dict(id="L3_core_off", group="L", note="Core64 sampling 50% -> 0%; reference stats still Core64",
         args={"core_sampler_off": True}),
    dict(id="L4_ref_full", group="L", note="reference stats from 1024/style instead of Core64",
         args={"ref_full": True}),
    dict(id="L5_same_off", group="L", note="lam_same 0.25 -> 0, L_rec unchanged",
         args={"no_same_style": True}),

    # ---------------- 加法: spatial gating (plan section 6) ----------------
    dict(id="S1_no_inject_64", group="S", note="disable 64x64 content injection, keep stem",
         args={"no_inject_64": True}),
    dict(id="S2_no_inject_32", group="S", note="disable 32x32 content injection",
         args={"no_inject_32": True}),
    dict(id="S3_no_inject_16", group="S", note="disable 16x16 content injection",
         args={"no_inject_16": True}),
]

# priority within a group: the plan's core questions first
PRIORITY = ["N1_no_style", "N2_no_canon", "N3_dim32", "N5_inv_off", "N4_steps400",
            "L1_lowC", "L2_highS", "L5_same_off", "L3_core_off",
            "L4_ref_full", "S1_no_inject_64", "S2_no_inject_32", "S3_no_inject_16",
            "N7_dim24", "N7_dim40"]


def merged(arm: dict) -> dict:
    """BASE + this arm's deltas, as a flat dict of train_eval_v6 CLI keys."""
    cfg = dict(BASE)
    cfg.update(arm.get("args", {}))
    return cfg


def to_cli(cfg: dict) -> list:
    """Translate a config dict into train_eval_v6 CLI tokens."""
    out = []
    for k, v in cfg.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                out.append(flag)
        else:
            out += [flag, str(v)]
    return out


def ordered(ids=None):
    """Arms in priority order, optionally restricted to `ids`."""
    by_id = {a["id"]: a for a in ARMS}
    seq = [by_id[i] for i in PRIORITY if i in by_id]
    if ids:
        seq = [a for a in seq if a["id"] in ids]
    return seq
