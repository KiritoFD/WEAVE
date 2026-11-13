"""Generate configs/sbflow/exp01..exp40.json for the 12h execution book."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "configs" / "sbflow"
OUT.mkdir(parents=True, exist_ok=True)

COMMON = dict(
    experiment_id="", output="", steps=4000, batch=512, patch=8,
    seed=42, width=256, depth=6, heads=4, warmup=200, lr=0.0002,
    ema_decay=0.999, min_t=0.002, eps=0.04,
    objective="bidir", edm=1, heads_out=1, style_cond="decomposed",
    target="drift", time_sampling="uniform", strata=8, sampler="perm",
    score_weight=1.0, ot_ratio=0.0, ot_cost="rgb", ot_reg=0.05, ot_tol=1e-6,
    local_conv=False, pair_rank=0, style_film=False, cycle_weight=0.0,
    dist_loss="none", style_enc="learned", lambda_dist=0.1,
    style_enc_epochs=60, style_enc_ckpt="", ode_steps_train=2, dist_nproj=64,
    ode_microbatch=64, grad_microbatch=0,
    online="none", online_steps=4, online_direction="both", online_mix=0.0,
    online_replay=False, online_raw_teacher=False, online_dynamic_cov=False,
    online_warmup=0, online_refresh_interval=0, triangle_weight=0.0,
    mixed_batch=True, equal_weight_stats=True, pivot=False,
    log_interval=50, save_interval=100000,
    equiv_steps=4000, equiv_batch=512,
    save_equiv="1000,2000,3000,4000", full_equiv=3000,
    lr_cycle_steps=4000,
    vae="stabilityai/sd-vae-ft-ema", scaling_factor=0.18215,
    latent_shape=[4, 64, 64], dataset_styles=5, dataset_images=15545,
    quick200=True, full750=False, priority=False, fork_from=None,
)
CORE = {"01", "04", "08", "10", "24", "27", "32", "35", "37"}
STYLE_ENC = "runs/blog_sbflow/suite40/shared_style_encoder.pt"
LEARNED = {"22", "23", "24", "25", "26", "37"}
HEAVY = {"19", "20", "21", "22", "23", "24", "25", "26", "37",
         "27", "28", "29", "30", "31", "32", "33", "36", "39"}


def arm(i, group, note, **kw):
    cfg = dict(COMMON)
    cfg.update(kw)
    cfg.update(id=f"{i:02d}", group=group, note=note,
               experiment_id=f"exp{i:02d}", priority=(f"{i:02d}" in CORE))
    if f"{i:02d}" in HEAVY:
        cfg["grad_microbatch"] = 64
    return cfg


def fork(i, group, note, src, **kw):
    base = dict(steps=1000, resume="@FORK@", fork_from=f"exp{src:02d}",
                lr_cycle_steps=4000, equiv_steps=4000)
    base.update(kw)
    return arm(i, group, note, **base)


ARMS = [
    # A: batch / token resolution / shared training
    arm(1, "A", "patch8 B512 mixed; unified main baseline and shared prefix source"),
    arm(2, "A", "patch8 B1024 mixed; large batch efficiency", steps=2000, batch=1024),
    arm(3, "A", "patch8 B256 mixed; small batch gradient noise", steps=8000, batch=256),
    arm(4, "A", "patch4 B256 mixed; token 64->256 for Rococo/ER/Ukiyo-e",
        steps=8000, batch=256, patch=4),
    arm(5, "A", "patch8 B512 single-pair; isolated contribution of mixed batch",
        single_pair=True, mixed_batch=False),
    arm(6, "A", "patch8 B512 frequency-weighted global stats; isolated equal-weight effect",
        frequency_stats=True, equal_weight_stats=False),
    # B: bridge mathematics and prediction targets
    arm(7, "B", "bidir_joint: both directions on the same bridge sample",
        objective="bidir_joint"),
    arm(8, "B", "direct single velocity with independently derived EDM",
        objective="single_velocity"),
    arm(9, "B", "single velocity without EDM", objective="single_velocity", edm=0),
    arm(10, "B", "current + score two-head decomposition", objective="score_current"),
    arm(11, "B", "endpoint conditional expectation prediction", target="endpoint"),
    arm(12, "B", "stratified time sampling with importance weights",
        time_sampling="stratified"),
    arm(13, "B", "Brownian bridge eps=0.16", eps=0.16),
    # C: five-style conditioning and spatial modelling
    arm(14, "C", "decomposed style embedding -> pair-ID condition", style_cond="pair"),
    arm(15, "C", "shared trunk + pair LoRA rank4", pair_rank=4),
    arm(16, "C", "per-layer Style-FiLM modulation", style_film=True),
    arm(17, "C", "DiT depthwise local conv residual", local_conv=True),
    arm(18, "C", "patch4 + per-layer Style-FiLM (paired control with exp04)",
        steps=8000, batch=256, patch=4, style_film=True),
    # D: direct target-distribution supervision (fork from prefixes)
    fork(19, "D", "bidir + fixed latent Global-SWD", 1,
         dist_loss="global", style_enc="fixed"),
    fork(20, "D", "bidir + fixed latent Local-Patch SWD", 1,
         dist_loss="local", style_enc="fixed"),
    fork(21, "D", "bidir + fixed Global+Local SWD", 1,
         dist_loss="both", style_enc="fixed"),
    fork(22, "D", "bidir + learned representation Global-SWD", 1,
         dist_loss="global", style_enc="learned", style_enc_ckpt=STYLE_ENC),
    fork(23, "D", "bidir + learned representation Local-SWD", 1,
         dist_loss="local", style_enc="learned", style_enc_ckpt=STYLE_ENC),
    fork(24, "D", "bidir + learned Global+Local SWD (core)", 1,
         dist_loss="both", style_enc="learned", style_enc_ckpt=STYLE_ENC),
    fork(25, "D", "single velocity + learned Global+Local SWD; fork exp08", 8,
         objective="single_velocity", dist_loss="both", style_enc="learned",
         style_enc_ckpt=STYLE_ENC),
    fork(26, "D", "endpoint prediction + learned Global+Local SWD; fork exp11", 11,
         target="endpoint", dist_loss="both", style_enc="learned",
         style_enc_ckpt=STYLE_ENC),
    # E: online Schrodinger Bridge (fork exp01 3000 -> 4000)
    fork(27, "E", "SDE4 + EMA teacher + bidirectional update (core)", 1,
         online="sde", online_steps=4),
    fork(28, "E", "SDE8 + EMA teacher", 1, online="sde", online_steps=8),
    fork(29, "E", "SDE4 + raw teacher", 1,
         online="sde", online_steps=4, online_raw_teacher=True),
    fork(30, "E", "SDE4 + 50% original bridge anchoring", 1,
         online="sde", online_steps=4, online_mix=0.5),
    fork(31, "E", "SDE4 + dynamic coupling covariance EDM", 1,
         online="sde", online_steps=4, online_dynamic_cov=True),
    fork(32, "E", "ODE4 deterministic reflow control (core)", 1,
         online="ode", online_steps=4),
    fork(33, "E", "SDE4 + cached endpoints refreshed every 100 steps (approximate)", 1,
         online="sde", online_steps=4, online_refresh_interval=100),
    # F: alternate transport structures
    arm(34, "F", "Gaussian Pivot + one-step conditional map (CFM surrogate, no ODE)",
        pivot=True, objective="pivot_cfm"),
    arm(35, "F", "Gaussian Pivot + Brownian bridge + EDM (core)", pivot=True),
    fork(36, "F", "exp35 prefix 3000 + online SDE 1000", 35,
         pivot=True, online="sde", online_steps=4),
    arm(37, "F", "one-step Monge map + learned Global+Local SWD (core)",
        objective="monge", dist_loss="both", style_enc="learned",
        style_enc_ckpt=STYLE_ENC),
    arm(38, "F", "one-step Monge + weak move regularisation",
        objective="monge", cycle_weight=0.01),
    fork(39, "F", "five-style triangle path consistency from exp01 3000", 1,
         triangle_weight=0.1),
    # G: OT validation
    arm(40, "G", "mixed batch + RGB mini-batch OT ratio 0.5; aligned with exp01",
        ot_ratio=0.5, ot_cost="rgb"),
]

assert len(ARMS) == 40, len(ARMS)
ids = [a["id"] for a in ARMS]
assert ids == [f"{i:02d}" for i in range(1, 41)], ids

for cfg in ARMS:
    p = OUT / f"exp{cfg['id']}.json"
    p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2, sort_keys=True),
                 encoding="utf-8")
print(f"wrote {len(ARMS)} configs to {OUT}")
for a in ARMS:
    print(f"  {a['id']} {a['group']} {a['objective']:16s} b={a['batch']:4d} "
          f"patch={a['patch']} steps={a['steps']:4d} fork={a['fork_from']} "
          f"core={a['priority']}")
