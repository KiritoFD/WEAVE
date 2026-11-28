"""CPU-only stage-1 recipe/profile contracts; no model/GPU construction."""
import copy
import subprocess
import sys

import pytest

from exp.stage1_recipe.config import defaults
from exp.stage1_recipe.training.oneshot_p import Session, arm_recipe, task_ratios
from exp.stage1_recipe.training.profile import summarize


def test_p_config_schema_and_adaptive_arm():
    cfg = defaults()
    cfg["training"]["p"]["ratios_stage"] = "p1"
    p0 = arm_recipe(cfg, "P0")
    assert p0["effective_batch"] == 96
    assert p0["warmup"] == 100
    assert p0["clip"] == 1.
    assert p0["y16_perturb"] == .40
    assert p0["reserve_seconds"] == 60
    assert task_ratios(p0) == (.30, .40, .15, .15)
    assert p0["lr_p"] == 1e-4 and p0["lr_style"] == 1.5e-4
    assert arm_recipe(cfg, "P1")["lr_p"] == 5e-5
    with pytest.raises(ValueError, match="requires resolved"):
        arm_recipe(cfg, "P2")
    cfg["arms"]["P2"].update(lr_p=5e-5)
    p2 = arm_recipe(cfg, "P2")
    assert p2["lr_style"] == pytest.approx(7.5e-5)
    assert p2["q_noise"] == .025
    assert task_ratios(p2) == task_ratios(p0)


def test_p_task_ratios_none_fallback_and_config_guard():
    cfg = defaults()
    cfg["training"]["p"].update(task_ratios=None, ratios_stage="p2")
    assert task_ratios(arm_recipe(cfg,"P0")) == (.10,.30,.30,.30)
    cfg["training"]["p"]["task_ratios"] = [.2,.3,.2,.3]
    assert task_ratios(arm_recipe(cfg,"P0")) == (.2,.3,.2,.3)
    cfg["training"]["p"]["y16_dropout"] = .2
    with pytest.raises(ValueError, match="y16_perturb"):
        arm_recipe(cfg,"P0")


def test_profile_accepts_e_micro_and_slowest_arm_budget():
    rows = [dict(arm=a,micro=1,checkpointing=True,stable=True,
                 seconds_per_update_p95=s,samples_per_s=14/s)
            for a,s in (("E0",2.),("E1",3.),("E2",4.))]
    out = summarize(rows,("E0","E1","E2"),safety_factor=.90)
    assert out["micro"] == 1 and out["checkpointing"]
    assert out["common_updates"] == 1012
    assert out["status"] == "ready"
    slow = copy.deepcopy(rows)
    slow[-1]["seconds_per_update_p95"] = 10.
    assert summarize(slow,("E0","E1","E2"))["status"] == "reduce_arms_required"
    assert summarize([], ("E0","E1","E2"))["status"] == "no_stable_configuration"


def test_resume_rejects_changed_micro_before_restoring_weights():
    session = Session.__new__(Session)
    session.micro = 4
    with pytest.raises(ValueError, match="Resume micro changed"):
        session.restore({"micro":8})


def test_p_import_help_and_gpu_guard_are_lazy(monkeypatch):
    monkeypatch.delenv("WEAVE_STAGE1_GPU_ALLOWED",raising=False)
    with pytest.raises(RuntimeError,match="GPU execution disabled"):
        Session({}, {},micro=4,checkpointing=False)
    code = ("import sys; from exp.stage1_recipe.training import oneshot_p,profile; "
            "assert 'torch' not in sys.modules; "
            "oneshot_p.parser().format_help();profile.parser().format_help()")
    subprocess.run([sys.executable,"-c",code],check=True)


def test_e_cpu_seed_slicing_and_warmup_contract():
    from exp.stage1_recipe.training.e_objectives import deterministic_seed, slices
    from exp.stage1_recipe.training.reference_e import warmup_factor
    assert deterministic_seed(1,2,3) == deterministic_seed(1,2,3)
    assert deterministic_seed(1,2,3) != deterministic_seed(1,3,3)
    assert [(s.start,s.stop) for s in slices(6,4)] == [(0,4),(4,6)]
    assert warmup_factor(1) == .01
    assert warmup_factor(100) == warmup_factor(250) == 1.
