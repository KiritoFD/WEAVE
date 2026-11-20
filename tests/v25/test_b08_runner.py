"""B08 CPU tests: Windows/Linux 跨平台运行器、五状态判别与恢复建议 (tools/v25_cpu/b08_host_runner.py).

全 CPU / tmp_path / 纯 stdlib；不触碰 CUDA/ssh/真实 runs/；无硬编码绝对路径。
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import json
from pathlib import Path

import pytest

from src.v25.result_writer import (
    STATE_DONE,
    STATE_FAILED,
    STATE_INCOMPLETE,
    STATE_RUNNING,
    STATE_STARTED,
    ResultWriter,
    atomic_save_json,
)
from tools.v25_cpu.b08_host_runner import (
    SCHTASKS_NOT_RUN,
    SCHTASKS_RUNNING,
    JobSpec,
    inspect_job_state,
    main as b08_main,
    make_bracket_regex,
    parse_exit_code,
    recommend_recovery,
    render_job_bundle,
    render_safe_pkill_cmd,
    write_job_bundle,
)


def _fake_meta(n_tasks: int = 750, batch: int = 25) -> dict:
    batches = []
    for off in range(0, n_tasks, batch):
        ids = [f"task_{off + i:04d}" for i in range(min(batch, n_tasks - off))]
        batches.append({"offset": off, "file": f"batch_{off:05d}.pt", "ids": ids})
    return {
        "kind": "v25",
        "arm": "test_arm",
        "n_tasks": n_tasks,
        "decoder": "S10",
        "batches": batches,
    }


# --------------------------------------------------------------------------- #
# JobSpec 校验与默认环境
# --------------------------------------------------------------------------- #
class TestJobSpec:
    def test_linux_defaults(self):
        spec = JobSpec(
            job_id="v25r2",
            host_os="linux",
            module="exp.v25.r2_train_ablation",
            args=["--pool", "../pseudo_gt_24h/orig_mean_fp16"],
        )
        assert spec.gpu_queue == "4090"
        assert spec.workdir_rel == "."
        assert spec.log_rel == "runs/v25/v25r2.log"
        assert spec.err_rel == "runs/v25/v25r2.err.log"
        assert spec.env["OMP_NUM_THREADS"] == "2"
        assert spec.env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
        assert spec.env["HF_HUB_OFFLINE"] == "1"
        assert spec.env["TRANSFORMERS_OFFLINE"] == "1"

    def test_windows_defaults_and_no_expandable_segments(self):
        spec = JobSpec(
            job_id="v25r3",
            host_os="windows",
            module="exp.v25.r3_oneshot_minimal",
        )
        assert spec.gpu_queue == "3060"
        assert spec.env["OMP_NUM_THREADS"] == "2"
        assert spec.env["HF_HUB_OFFLINE"] == "1"
        assert spec.env["TRANSFORMERS_OFFLINE"] == "1"
        assert spec.env["PYTHONUTF8"] == "1"
        assert "PYTORCH_CUDA_ALLOC_CONF" not in spec.env

        with pytest.raises(ValueError, match="expandable_segments"):
            JobSpec(
                job_id="v25r3_bad",
                host_os="windows",
                module="exp.v25.r3_oneshot_minimal",
                env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
            )

    def test_rejects_absolute_paths_and_bad_ids(self):
        # 动态拼接盘符/绝对前缀，避免静态护栏误报
        win_abs = "G" + ":" + "\\WEAVE\\runs"
        posix_abs = "/" + "workspace" + "/WEAVE"
        with pytest.raises(ValueError, match="relative"):
            JobSpec(job_id="j1", host_os="windows", module="exp.v25.a", workdir_rel=win_abs)
        with pytest.raises(ValueError, match="relative"):
            JobSpec(job_id="j2", host_os="linux", module="exp.v25.a", log_rel=posix_abs)
        with pytest.raises(ValueError, match="job_id"):
            JobSpec(job_id="bad id with space", host_os="linux", module="exp.v25.a")


# --------------------------------------------------------------------------- #
# Linux (.sh + tmux) 与 Windows (.bat + schtasks) 包装脚本生成
# --------------------------------------------------------------------------- #
class TestScriptRendering:
    def test_linux_sh_and_tmux_generation(self, tmp_path):
        spec = JobSpec(
            job_id="v25_mech_a",
            host_os="linux",
            module="exp.v25.mechanism_a",
            args=["--arm", "all", "--updates", "1000"],
            log_rel="runs/v25/mech_a.log",
            err_rel="runs/v25/mech_a.err.log",
        )
        sh_path, bundle = write_job_bundle(spec, tmp_path, script_rel="runs/v25/jobs/v25_mech_a_launch.sh")
        assert sh_path.is_file()
        raw_bytes = sh_path.read_bytes()
        assert b"\r\n" not in raw_bytes, "Linux .sh must use LF line endings"
        text = raw_bytes.decode("utf-8")

        assert text.startswith("#!/usr/bin/env bash\n")
        assert "export OMP_NUM_THREADS=2" in text
        assert "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in text
        assert "export HF_HUB_OFFLINE=1" in text
        assert "export TRANSFORMERS_OFFLINE=1" in text
        assert "python -u -m exp.v25.mechanism_a --arm all --updates 1000" in text

        # 防自杀 [x] 括号技巧检查：不允许裸 pkill -f exp.v25...
        assert "[e]xp.v25.mechanism_a" in text
        assert "pkill -f exp." not in text
        assert "[e]xp.v25.mechanism_a" in bundle.commands["pkill_safe_cmd"]
        assert make_bracket_regex("exp.v25.foo") == "[e]xp.v25.foo"
        assert render_safe_pkill_cmd("exp.v25.foo") == "pkill -f '[e]xp.v25.foo'"

        # tmux 命令不内联复杂多级引号 python 命令，仅执行 bash <launch.sh>
        tmux_cmd = bundle.commands["tmux_cmd"]
        assert tmux_cmd.startswith("tmux new-session -d -s v25_mech_a ")
        assert "bash runs/v25/jobs/v25_mech_a_launch.sh" in tmux_cmd
        assert "python -u" not in tmux_cmd

    def test_windows_bat_and_schtasks_generation(self, tmp_path):
        spec = JobSpec(
            job_id="v25r3_main",
            host_os="windows",
            module="exp.v25.r3_oneshot_minimal",
            args=["--pool", "../pseudo_gt_24h/orig_mean_fp16", "--note", "run one"],
            log_rel="runs/v25/v25_r3.log",
            err_rel="runs/v25/v25_r3.log",
        )
        bat_path, bundle = write_job_bundle(
            spec,
            tmp_path,
            script_rel="runs/v25/jobs/v25r3_main_launch.bat",
        )
        assert bat_path.is_file()
        raw_bytes = bat_path.read_bytes()
        assert b"\r\n" in raw_bytes, "Windows .bat must use CRLF line endings"
        text = raw_bytes.decode("utf-8")

        assert text.startswith("@echo off\r\n")
        assert 'cd /d "%~dp0"' in text
        assert 'set "OMP_NUM_THREADS=2"' in text
        assert 'set "HF_HUB_OFFLINE=1"' in text
        assert 'set "TRANSFORMERS_OFFLINE=1"' in text
        assert 'set "PYTHONUTF8=1"' in text
        assert "expandable_segments" not in text
        assert 'if not exist "runs\\v25" mkdir "runs\\v25"' in text
        assert (
            'python -u -m exp.v25.r3_oneshot_minimal --pool ../pseudo_gt_24h/orig_mean_fp16 --note "run one" '
            '>> "runs\\v25\\v25_r3.log" 2>&1'
        ) in text

        create_cmd = bundle.commands["schtasks_create_cmd"]
        run_cmd = bundle.commands["schtasks_run_cmd"]
        assert "/Create" in create_cmd and "/F" in create_cmd
        assert "/TN v25r3_main" in create_cmd
        assert "/TR runs\\v25\\jobs\\v25r3_main_launch.bat" in create_cmd
        # 禁止内联 python 参数到 /TR，杜绝 exit=2 引号转义 bug
        assert "python" not in create_cmd
        assert '""' not in create_cmd
        assert run_cmd == "schtasks /Run /TN v25r3_main"


# --------------------------------------------------------------------------- #
# 五状态机 inspect_job_state & ResultWriter 集成
# --------------------------------------------------------------------------- #
class TestInspectJobState:
    def test_exit_2_maps_to_failed_schtasks_error(self):
        rep = inspect_job_state(log_text="", err_text="", task_exit_code=2, host_os="windows")
        assert rep.state == STATE_FAILED
        assert rep.failure_kind == "schtasks_path_or_quote_error"
        assert rep["task_exit_code"] == 2

    def test_httpx_and_oom_map_to_failed(self):
        rep_net = inspect_job_state(
            log_text="loading model...\n",
            err_text="httpx.ConnectError: [Errno 11001] getaddrinfo failed\n",
            task_exit_code=1,
        )
        assert rep_net.state == STATE_FAILED
        assert rep_net.failure_kind == "hf_network_error"

        rep_oom = inspect_job_state(
            log_text="step 100/1000\ntorch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB",
            err_text="",
            task_exit_code=None,
        )
        assert rep_oom.state == STATE_FAILED
        assert rep_oom.failure_kind == "cuda_oom"

    def test_hex_exit_code_and_schtasks_running_and_not_run(self):
        assert parse_exit_code("0x41301") == SCHTASKS_RUNNING
        assert parse_exit_code("0x41303") == SCHTASKS_NOT_RUN

        rep_run = inspect_job_state(log_text="epoch 1/5\n", task_exit_code="267009", host_os="windows")
        assert rep_run.state == STATE_RUNNING

        rep_start = inspect_job_state(log_text="", task_exit_code=SCHTASKS_NOT_RUN, host_os="windows")
        assert rep_start.state == STATE_STARTED

        rep_crash = inspect_job_state(log_text="", task_exit_code="0xC0000409", host_os="windows")
        assert rep_crash.state == STATE_FAILED
        assert rep_crash.task_exit_code == 0xC0000409

    def test_done_with_valid_meta_and_result_writer(self, tmp_path):
        arm_dir = tmp_path / "runs" / "v25" / "raw" / "arm_ok"
        meas_dir = tmp_path / "runs" / "v25" / "measured" / "arm_ok"
        meas_dir.mkdir(parents=True)

        rw = ResultWriter(arm_dir, expected_tasks=750)
        rw.start(cmd="python -m exp.v25.r1")
        rw.running(progress="375/750")
        meta = _fake_meta(750)
        atomic_save_json(meta, arm_dir / "meta.json")
        rw.done(meta=meta)
        atomic_save_json({"rows": [], "n_total": 750, "n_cross": 600, "n_same": 150}, meas_dir / "eval.json")
        atomic_save_json({"arm": "arm_ok"}, meas_dir / "summary_v25.json")

        rep = inspect_job_state(
            log_text="finished 750/750\n",
            err_text="",
            task_exit_code=0,
            result_state_dir=arm_dir,
        )
        assert rep.state == STATE_DONE
        assert rep.result_writer_state == STATE_DONE
        assert rep.meta_status == "valid"

    def test_corrupt_status_json_recovers_via_jsonl_replay(self, tmp_path):
        arm_dir = tmp_path / "arm_replay"
        rw = ResultWriter(arm_dir, expected_tasks=750)
        rw.start(cmd="test")
        meta = _fake_meta(750)
        atomic_save_json(meta, arm_dir / "meta.json")
        rw.done(meta=meta)

        # 破坏 status.json 缓存，应回放 status_log.jsonl 恢复 done
        (arm_dir / "status.json").write_text('{"state": "do', encoding="utf-8")
        rep = inspect_job_state(log_text="ok", task_exit_code=0, result_state_dir=arm_dir)
        assert rep.state == STATE_DONE
        assert rep.result_writer_state == STATE_DONE

    def test_incomplete_cases_meta_bad_orphan_stale_and_h1_h2(self, tmp_path):
        # 1) meta 只有 500 个任务 → 即使 exit=0 且 status 写 done，也判 incomplete
        arm_bad = tmp_path / "arm_bad_count"
        arm_bad.mkdir()
        atomic_save_json(_fake_meta(500), arm_bad / "meta.json")
        atomic_save_json({"state": "done"}, arm_bad / "status.json")
        rep_bad = inspect_job_state(log_text="done?", task_exit_code=0, result_state_dir=arm_bad)
        assert rep_bad.state == STATE_INCOMPLETE
        assert rep_bad.failure_kind == "raw_meta_incomplete_count"

        # 2) orphan batch_*.pt 无 meta.json
        arm_orphan = tmp_path / "arm_orphan"
        arm_orphan.mkdir()
        (arm_orphan / "batch_00000.pt").write_bytes(b"stub")
        rep_orp = inspect_job_state(log_text="starting...", task_exit_code=None, result_state_dir=arm_orphan)
        assert rep_orp.state == STATE_INCOMPLETE
        assert rep_orp.failure_kind == "orphan_raw"

        # 3) ResultWriter running 但日志停更 / 进程已死
        arm_stale = tmp_path / "arm_stale"
        rw = ResultWriter(arm_stale, expected_tasks=750)
        rw.start(cmd="test")
        rw.running(progress="100/750")
        rep_stale = inspect_job_state(
            log_text="100/750",
            task_exit_code=None,
            result_state_dir=arm_stale,
            log_age_min=45.0,
            stale_threshold_min=15.0,
            process_alive=False,
        )
        assert rep_stale.state == STATE_INCOMPLETE
        assert rep_stale.log_stale is True

        # 4) B06 H1 窗口：raw/arm_h1 meta 完整但 measured/arm_h1 无 summary
        runs_root = tmp_path / "runs_h"
        raw_h1 = runs_root / "raw" / "arm_h1"
        meas_h1 = runs_root / "measured" / "arm_h1"
        raw_h1.mkdir(parents=True)
        meas_h1.mkdir(parents=True)
        atomic_save_json(_fake_meta(750), raw_h1 / "meta.json")
        rep_h1 = inspect_job_state(log_text="raw finished", task_exit_code=0, result_state_dir=raw_h1)
        assert rep_h1.state == STATE_INCOMPLETE
        assert rep_h1.window == "H1"

        # 5) B06 H2 窗口：measured/arm_h1/eval.json 半写损坏
        (meas_h1 / "eval.json").write_text('{"rows": [1, 2', encoding="utf-8")
        rep_h2 = inspect_job_state(log_text="measuring...", task_exit_code=0, result_state_dir=raw_h1)
        assert rep_h2.state == STATE_INCOMPLETE
        assert rep_h2.window == "H2"


# --------------------------------------------------------------------------- #
# recommend_recovery 与 CLI
# --------------------------------------------------------------------------- #
class TestRecommendRecoveryAndCli:
    def test_recovery_plans_cover_h1_h2_and_failures(self, tmp_path):
        plan_done = recommend_recovery(STATE_DONE)
        assert plan_done.should_rerun is False
        assert plan_done.action == "none"

        plan_run = recommend_recovery(STATE_RUNNING, host_os="windows")
        assert plan_run.should_rerun is False
        assert plan_run.action == "monitor"

        plan_h1 = recommend_recovery({"state": STATE_INCOMPLETE, "window": "H1", "failure_kind": "h1_summary_missing"})
        assert plan_h1.should_rerun is True
        assert plan_h1.action == "resume_measure_only"
        assert any("meta.json" in c for c in plan_h1.pre_checks)
        assert any(".pt" in c for c in plan_h1.pre_checks)

        plan_h2 = recommend_recovery({"state": STATE_INCOMPLETE, "window": "H2", "failure_kind": "h2_corrupt_eval_json"})
        assert plan_h2.should_rerun is True
        assert plan_h2.action == "clean_corrupt_and_rerun"
        assert any("eval.json" in s for s in plan_h2.steps)

        plan_exit2 = recommend_recovery(STATE_FAILED, failure_kind="schtasks_path_or_quote_error")
        assert plan_exit2.should_rerun is True
        assert any("exit=2" in s for s in plan_exit2.steps)

        plan_net = recommend_recovery(STATE_FAILED, failure_kind="hf_network_error")
        assert any("HF_HUB_OFFLINE=1" in s for s in plan_net.steps)

    def test_cli_render_and_inspect(self, tmp_path, capsys):
        rc1 = b08_main([
            "render",
            "--job-id", "v25_cli_test",
            "--host-os", "windows",
            "--module", "exp.v25.r3_oneshot_minimal",
            "--out-dir", str(tmp_path),
        ])
        assert rc1 == 0
        out1 = json.loads(capsys.readouterr().out)
        assert Path(out1["written_path"]).is_file()
        assert out1["commands"]["schtasks_run_cmd"] == "schtasks /Run /TN v25_cli_test"

        log_f = tmp_path / "job.log"
        err_f = tmp_path / "job.err"
        log_f.write_text("running...\n", encoding="utf-8")
        err_f.write_text("httpx.ConnectError: offline\n", encoding="utf-8")
        rc2 = b08_main([
            "inspect",
            "--log-file", str(log_f),
            "--err-file", str(err_f),
            "--exit-code", "1",
            "--host-os", "windows",
        ])
        assert rc2 == 0
        out2 = json.loads(capsys.readouterr().out)
        assert out2["inspection"]["state"] == STATE_FAILED
        assert out2["inspection"]["failure_kind"] == "hf_network_error"
        assert out2["recovery"]["should_rerun"] is True
