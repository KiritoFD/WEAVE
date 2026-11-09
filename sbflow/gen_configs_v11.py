"""Generate the V11 P1 arm configs (new IDs, no reuse of the V10 exp numbers).

Four arms, all forked from the SAME shared prefix `exp01/model_equiv3000.pt` and run for
1000 additional updates at the fixed 4000-equivalent LR cycle (no warmup restart):

  d1  learned-encoder Global+Local SWD   -> does direct style-distribution supervision work?
  d2  fixed-latent   Global+Local SWD    -> is a learned style representation necessary?
  e1  bidirectional SDE4 + EMA teacher   -> does the real online bridge update help?
  e2  bidirectional ODE4 re-pairing      -> SDE vs deterministic reflow control

The plain-bridge control at 4000 equivalent steps already exists and is NOT retrained:
`runs/blog_sbflow/suite40/exp01/model_equiv4000.pt` (the bidir objective was unaffected
by every V11 fix, so it remains a valid control).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "configs" / "v11"
OUT.mkdir(parents=True, exist_ok=True)

SUITE = "runs/blog_sbflow/suite40"
PREFIX = f"{SUITE}/exp01/model_equiv3000.pt"
STYLE_ENC = f"{SUITE}/shared_style_encoder.pt"

COMMON = dict(
    steps=1000, batch=512, patch=8, seed=42, width=256, depth=6, heads=4,
    warmup=200, lr=0.0002, ema_decay=0.999, min_t=0.002, eps=0.04,
    objective="bidir", edm=1, heads_out=1, style_cond="decomposed",
    target="drift", time_sampling="uniform", strata=8, sampler="perm",
    score_weight=1.0, ot_ratio=0.0, ot_cost="rgb", ot_reg=0.05, ot_tol=1e-6,
    local_conv=False, pair_rank=0, style_film=False, cycle_weight=0.0,
    monge_ot=False,
    dist_loss="none", style_enc="learned", lambda_dist=0.1,
    style_enc_epochs=60, style_enc_ckpt="", ode_steps_train=2, dist_nproj=64,
    ode_microbatch=64, grad_microbatch=64,
    online="none", online_steps=4, online_direction="both", online_mix=0.0,
    online_replay=False, online_raw_teacher=False, online_dynamic_cov=False,
    online_warmup=0, online_refresh_interval=0,
    triangle_weight=0.0,
    mixed_batch=True, equal_weight_stats=True, pivot=False,
    log_interval=50, save_interval=1000,
    equiv_batch=512, equiv_steps=4000, save_equiv="", full_equiv=0,
    lr_cycle_steps=4000,
    resume=PREFIX,
    vae="stabilityai/sd-vae-ft-ema", scaling_factor=0.18215,
    latent_shape=[4, 64, 64], dataset_styles=5, dataset_images=15545,
)

ARMS = {
    "d1": dict(note="learned-encoder Global+Local SWD, both directions, five targets "
                    "equal weight (fixes V10 defect 2 and 4)",
               dist_loss="both", style_enc="learned", style_enc_ckpt=STYLE_ENC),
    "d2": dict(note="fixed-latent Global+Local SWD control for d1: is a learned style "
                    "representation necessary?",
               dist_loss="both", style_enc="fixed"),
    "e1": dict(note="faithful alpha-DSBM: direction-specific loss, SDE4, EMA teacher "
                    "(fixes V10 defect 3)",
               online="sde", online_steps=4),
    "e2": dict(note="deterministic reflow control for e1: bidirectional ODE4 endpoints",
               online="ode", online_steps=4),
}

for aid, spec in ARMS.items():
    cfg = dict(COMMON)
    cfg.update(spec)
    cfg["experiment_id"] = aid
    cfg["output"] = f"runs/blog_sbflow/v11/{aid}"
    cfg["control"] = f"{SUITE}/exp01"
    cfg["fork_from"] = "exp01@3000"
    cfg["p1"] = True
    (OUT / f"{aid}.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"  {aid}: {spec['note'][:64]}")

# ---- P3: five-style conditional generation from a shared Gaussian prior ----
# Trains F_s: N(0,I) -> p_s for every style, which answers "can the shared model
# generate genuine Rococo / Ukiyo-e / Early Renaissance at all?" without any transport.
# Must start from scratch: the pair structure is {(s, pivot)} not the ten style pairs.
P3 = dict(COMMON)
P3.update(
    pivot=True, steps=4000, batch=512, patch=8,
    dist_loss="none", style_enc="learned", style_enc_ckpt="", online="none",
    grad_microbatch=0, ode_microbatch=64,
    save_interval=1000, save_equiv="1000,2000,3000,4000", full_equiv=4000,
    experiment_id="p3", output="runs/blog_sbflow/v11/p3",
    note="five-style conditional generation from a shared Gaussian prior "
         "(F_s: N(0,I) -> p_s); no cross-domain transport",
    p3=True,
)
P3.pop("resume", None)
(OUT / "p3.json").write_text(json.dumps(P3, ensure_ascii=False, indent=2, sort_keys=True),
                             encoding="utf-8")
print("  p3: five-style conditional generation from a shared Gaussian prior")
print(f"wrote {len(ARMS) + 1} configs to {OUT}")
