"""Few CPU regressions for accidental GPU launch, windows locking and fair jobs."""
import json
import os
import sys

from exp.stage1_recipe import queue
from exp.stage1_recipe.config import defaults, validate


def test_default_queue_never_queries_card_or_starts_workers(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["queue"])
    def forbidden(*args, **kwargs):
        raise AssertionError("Default plan must not execute anything")
    monkeypatch.setattr(queue.subprocess, "run", forbidden)
    monkeypatch.setattr(queue.subprocess, "Popen", forbidden)
    queue.main()
    result = json.loads(capsys.readouterr().out)
    assert result["mode"] == "plan_only_no_gpu_query"
    assert sum(result["stages"].values()) == 16 * 3600


def test_lock_rejects_live_owner_without_signalling_process(tmp_path):
    import pytest
    with queue.lock_window(tmp_path):
        assert queue.pid_alive(os.getpid())
        with pytest.raises(RuntimeError, match="Another queue"):
            with queue.lock_window(tmp_path, clear_stale=True):
                pass
    assert not (tmp_path / "queue.lock").exists()


def test_generated_job_keeps_correct_branch_and_complete_formal_size():
    cfg = defaults()
    validate(cfg)
    q = queue.Queue.__new__(queue.Queue)
    q.config = cfg
    jobs = q.jobs("E1", "e", e="weights/e.pt", e_ema=".995", split="core750")
    assert [x["lambda"] for x in jobs] == [.4, .6, .8, 1.]
    assert all(x["n"] == 0 and "tau" not in x for x in jobs)
    jobs = q.jobs("P0", "p", p="weights/p.pt", p_ema=".999", split="artist_ood")
    assert [x["tau"] for x in jobs] == [.35, .5, .65, .8]
    assert all(x["n"] == 0 and "lambda" not in x for x in jobs)
