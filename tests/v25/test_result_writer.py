"""B06 CPU tests: 结果 Writer、任务状态与恢复检查 (src/v25/result_writer.py).

验收项 (agent2.md §五 B06 / 任务卡):
- atomic_save_json: 正常往返 / 中断模拟 (replace 失败不伤原文件) / 替换语义 / Windows 路径 tmp+replace 兼容;
- ResultWriter 状态机: started→done / started→failed 的 JSONL 事件记录;
- task_id 完整性闸: meta 未过 750 唯一 id 校验时 done() 被拒绝、降级 incomplete (验收项 5);
- 损坏 JSON 读取容错: 半写 status.json / 撕裂 JSONL 尾行;
- scan_status: 半写窗口 (meta 完整但 summary 缺失 / orphan raw) 分诊。

全 CPU / tmp_path / 纯 stdlib; 不触碰仓库真实 runs/ 缓存。
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import json
import time

import pytest

from src.v25.result_writer import (
    LOG_FILENAME,
    STATE_DONE,
    STATE_FAILED,
    STATE_INCOMPLETE,
    STATE_RUNNING,
    STATE_STARTED,
    STATE_UNKNOWN,
    STATUS_FILENAME,
    ResultWriter,
    append_jsonl,
    atomic_save_json,
    read_json_tolerant,
    read_state,
    scan_status,
    validate_raw_meta_dict,
)


def _fake_meta(n_tasks: int = 750, batch: int = 25) -> dict:
    """镜像 RawWriter meta.json 结构 (src/utils/raw_io.py:78-86): batches[] 含 ids。"""
    batches = []
    for off in range(0, n_tasks, batch):
        ids = [f"task_{off + i:04d}" for i in range(min(batch, n_tasks - off))]
        batches.append({"offset": off, "file": f"batch_{off:05d}.pt", "ids": ids})
    return {"kind": "v25", "arm": "test_arm", "n_tasks": n_tasks,
            "decoder": "S10", "batches": batches}


# --------------------------------------------------------------------------- #
# atomic_save_json
# --------------------------------------------------------------------------- #
class TestAtomicSaveJson:
    def test_normal_roundtrip(self, tmp_path):
        p = tmp_path / "nested" / "deep" / "meta.json"
        obj = {"a": 1, "b": ["x", "y"], "c": {"τ": "中文"}, "d": 0.18215}
        atomic_save_json(obj, p)
        assert json.loads(p.read_text(encoding="utf-8")) == obj
        # 无残留 tmp
        assert not list(p.parent.glob("*.tmp*"))

    def test_nan_rejected_and_original_untouched(self, tmp_path):
        p = tmp_path / "m.json"
        atomic_save_json({"v": 1}, p)
        with pytest.raises(ValueError):
            atomic_save_json({"v": float("nan")}, p)
        assert json.loads(p.read_text(encoding="utf-8")) == {"v": 1}

    def test_replace_semantics(self, tmp_path):
        p = tmp_path / "results.json"
        atomic_save_json({"results": [1, 2]}, p)
        atomic_save_json({"results": [3]}, p)  # 第二次整体替换（先到者被覆盖）
        assert json.loads(p.read_text(encoding="utf-8")) == {"results": [3]}
        assert not list(p.parent.glob("*.tmp*"))

    def test_crash_midwrite_keeps_original(self, tmp_path, monkeypatch):
        """崩溃模拟：os.replace 失败（断电/kill 于替换瞬间）→ 原文件完好，tmp 残留。"""
        p = tmp_path / "arm" / "meta.json"
        good = {"ok": True, "gen": 1}
        atomic_save_json(good, p)

        import src.v25.result_writer as rw_mod

        def _boom(src, dst):
            raise OSError("simulated crash during replace")
        monkeypatch.setattr(rw_mod.os, "replace", _boom)
        with pytest.raises(OSError):
            atomic_save_json({"ok": True, "gen": 2, "torn": "half written"}, p)
        monkeypatch.undo()

        # 原文件仍是旧完整内容（读到的绝不是半写新内容）
        assert json.loads(p.read_text(encoding="utf-8")) == good
        # tmp 残留（真实 kill 无法清理）；下一次成功写自然复用同 pid tmp 名并清掉
        assert list(p.parent.glob("*.tmp*")), "replace 失败应留下 tmp 模拟崩溃现场"
        atomic_save_json({"ok": True, "gen": 2}, p)
        assert json.loads(p.read_text(encoding="utf-8")) == {"ok": True, "gen": 2}
        assert not list(p.parent.glob("*.tmp*"))

    def test_torn_tmp_file_does_not_affect_original(self, tmp_path):
        """写一半的 tmp 文件不影响原文件；读取方拿到的永远是完整 JSON 或 None。"""
        p = tmp_path / "summary.json"
        atomic_save_json({"state": "done"}, p)
        torn = p.with_name(p.name + ".tmp999999")
        torn.write_text('{"state": "do', encoding="utf-8")  # 模拟半写 tmp
        assert read_json_tolerant(p) == {"state": "done"}
        assert read_json_tolerant(torn) is None  # tmp 不是读对象；宽容读返回 None
        atomic_save_json({"state": "done", "v": 2}, p)  # 不受垃圾 tmp 干扰
        assert read_json_tolerant(p) == {"state": "done", "v": 2}

    def test_windows_path_compat(self, tmp_path):
        """Windows 兼容：嵌套目录 / 空格 / 中文 / 大小写不敏感重开 / 目标已存在时覆盖。"""
        p = tmp_path / "runs v25" / "raw" / "测试臂_01" / "status.json"
        atomic_save_json({"state": "started"}, p)
        assert read_json_tolerant(p) == {"state": "started"}
        # 目标已存在 → os.replace 覆盖成功
        atomic_save_json({"state": "done"}, p)
        assert read_json_tolerant(p) == {"state": "done"}
        # Windows 大小写不敏感路径重开等价
        alt = p.with_name(p.name.upper())
        assert read_json_tolerant(alt) == {"state": "done"}
        assert not list(p.parent.glob("*.tmp*"))

    def test_default_str_and_compact_default(self, tmp_path):
        """default=str 兼容 Path 等对象（与 runtime.save_json 约定一致）；默认 compact。"""
        from pathlib import Path as _P
        p = tmp_path / "cfg.json"
        atomic_save_json({"out": _P("runs") / "v25"}, p)
        raw = p.read_text(encoding="utf-8")
        # Path 经 default=str 序列化（JSON 内反斜杠转义）；compact 单行无换行
        assert "v25" in raw and "runs" in raw and "\n" not in raw
        atomic_save_json({"out": "x"}, p, indent=2)
        assert "\n" in p.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# 宽容读取 / JSONL
# --------------------------------------------------------------------------- #
class TestTolerantReadAndJsonl:
    def test_read_missing_and_corrupt_returns_none(self, tmp_path):
        assert read_json_tolerant(tmp_path / "nope.json") is None
        p = tmp_path / "torn.json"
        p.write_text('{"n_total": 750, "rows": [{"task_id": "x"', encoding="utf-8")  # 半写
        assert read_json_tolerant(p) is None
        p.write_text("", encoding="utf-8")
        assert read_json_tolerant(p) is None

    def test_read_bom_file(self, tmp_path):
        p = tmp_path / "bom.json"
        p.write_text(json.dumps({"k": 1}), encoding="utf-8-sig")
        assert read_json_tolerant(p) == {"k": 1}

    def test_append_jsonl_and_tolerant_replay(self, tmp_path):
        log = tmp_path / LOG_FILENAME
        append_jsonl(log, {"event": "a"})
        append_jsonl(log, {"event": "b"})
        with open(log, "a", encoding="utf-8") as fh:
            fh.write('{"event": "tor')  # 模拟崩溃撕裂的最后一行（无换行）
        from src.v25.result_writer import _iter_valid_jsonl
        assert [e["event"] for e in _iter_valid_jsonl(log)] == ["a", "b"]

    def test_validate_raw_meta_dict(self):
        assert validate_raw_meta_dict(_fake_meta(750), 750)
        assert not validate_raw_meta_dict(_fake_meta(749), 750)      # 缺任务
        m = _fake_meta(750)
        m["batches"][1]["ids"][0] = m["batches"][0]["ids"][0]        # 重复 id
        assert not validate_raw_meta_dict(m, 750)
        assert not validate_raw_meta_dict({}, 750)
        assert not validate_raw_meta_dict(None, 750)
        assert not validate_raw_meta_dict({"batches": [{"ids": "notalist"}]}, 750)


# --------------------------------------------------------------------------- #
# ResultWriter 状态机
# --------------------------------------------------------------------------- #
class TestResultWriterStateMachine:
    def test_started_to_done_jsonl(self, tmp_path):
        d = tmp_path / "raw" / "armA"
        d.mkdir(parents=True)
        rw = ResultWriter(d, expected_tasks=750)
        assert rw.current()["state"] == STATE_UNKNOWN

        ev_start = rw.start(cmd="python -m exp.v25.foo --arm armA")
        assert ev_start["prev_state"] == STATE_UNKNOWN and ev_start["state"] == STATE_STARTED
        ev_run = rw.running(progress="300/750")
        assert ev_run["state"] == STATE_RUNNING and ev_run["progress"] == "300/750"
        ev_done = rw.done(meta=_fake_meta(750), wall_sec=12.5)
        assert ev_done["state"] == STATE_DONE and ev_done["meta_validated"] is True

        # JSONL 事件链完整（append-only 历史）
        hist = rw.history()
        assert [e["state"] for e in hist] == [STATE_STARTED, STATE_RUNNING, STATE_DONE]
        assert all(e["task_id"] == "armA" for e in hist)
        # status.json 是最新状态缓存
        cur = rw.current()
        assert cur["state"] == STATE_DONE and cur["source"] == STATUS_FILENAME
        assert cur["wall_sec"] == 12.5

    def test_started_to_failed_jsonl(self, tmp_path):
        d = tmp_path / "raw" / "armB"
        d.mkdir(parents=True)
        rw = ResultWriter(d)
        rw.start()
        ev = rw.failed("RuntimeError: CUDA OOM at batch 12", stage="generate")
        assert ev["state"] == STATE_FAILED and "CUDA OOM" in ev["reason"]
        assert [e["state"] for e in rw.history()] == [STATE_STARTED, STATE_FAILED]
        assert rw.current()["state"] == STATE_FAILED

    def test_done_refuses_incomplete_meta(self, tmp_path):
        """验收项 5: meta 未过唯一 id 闸 → 拒绝记 done，降级 incomplete。"""
        d = tmp_path / "raw" / "armC"
        d.mkdir(parents=True)
        rw = ResultWriter(d)
        rw.start()
        ev = rw.done(meta=_fake_meta(749))  # 少 1 个任务
        assert ev["state"] == STATE_INCOMPLETE
        assert "completeness" in ev["reason"]
        assert ev["meta_validated"] is False
        assert rw.current()["state"] == STATE_INCOMPLETE
        # JSONL 里不存在 done 事件 —— "失败仍记 done" 被结构性堵住
        assert all(e["state"] != STATE_DONE for e in rw.history())

    def test_done_refuses_duplicate_ids(self, tmp_path):
        m = _fake_meta(750)
        m["batches"][2]["ids"][0] = m["batches"][0]["ids"][0]
        rw = ResultWriter(tmp_path / "raw" / "armD"); rw.start()
        assert rw.done(meta=m)["state"] == STATE_INCOMPLETE

    def test_done_auto_loads_meta_from_dir(self, tmp_path):
        """meta=None 时自动读取目录内 meta.json 校验（对接 RawWriter.finish 产物）。"""
        d = tmp_path / "raw" / "armE"
        d.mkdir(parents=True)
        atomic_save_json(_fake_meta(750), d / "meta.json")
        rw = ResultWriter(d)
        rw.start()
        assert rw.done()["state"] == STATE_DONE and rw.current()["meta_validated"] is True
        # 半写 meta.json（模拟 RawWriter 之外的写法崩溃）→ 同样拒绝 done
        d2 = tmp_path / "raw" / "armF"
        d2.mkdir(parents=True)
        (d2 / "meta.json").write_text('{"batches": [{"ids": ["task_0000"', encoding="utf-8")
        rw2 = ResultWriter(d2); rw2.start()
        assert rw2.done()["state"] == STATE_INCOMPLETE

    def test_done_without_meta_records_unvalidated(self, tmp_path):
        """无 meta.json 的非评测任务（纯训练臂）→ done 但如实记录 meta_validated=null。"""
        d = tmp_path / "runs" / "train_arm"
        d.mkdir(parents=True)
        rw = ResultWriter(d)
        rw.start()
        ev = rw.done(updates=1500)
        assert ev["state"] == STATE_DONE and ev["meta_validated"] is None

    def test_corrupt_status_json_falls_back_to_log_replay(self, tmp_path):
        """status.json 半写损坏 → 从 JSONL 回放恢复状态（验收项 4 容错）。"""
        d = tmp_path / "raw" / "armG"
        d.mkdir(parents=True)
        rw = ResultWriter(d)
        rw.start()
        rw.done(meta=_fake_meta(750))
        (d / STATUS_FILENAME).write_text('{"state": "don', encoding="utf-8")  # 半写
        cur = rw.current()
        assert cur["state"] == STATE_DONE
        assert cur["source"] == LOG_FILENAME
        # 无 status.json（刚创建目录就被 kill）同样回放
        (d / STATUS_FILENAME).unlink()
        assert rw.current()["state"] == STATE_DONE

    def test_torn_jsonl_tail_still_recovers_state(self, tmp_path):
        d = tmp_path / "raw" / "armH"
        d.mkdir(parents=True)
        rw = ResultWriter(d)
        rw.start()
        rw.done(meta=_fake_meta(750))
        with open(d / LOG_FILENAME, "a", encoding="utf-8") as fh:
            fh.write('{"state": "run')  # 崩溃撕裂尾行
        (d / STATUS_FILENAME).write_text('{"task', encoding="utf-8")  # 同时半写 status.json
        cur = rw.current()
        assert cur["state"] == STATE_DONE  # 最后一条有效事件仍是 done
        assert cur["source"] == LOG_FILENAME

    def test_explicit_incomplete(self, tmp_path):
        d = tmp_path / "raw" / "armI"
        d.mkdir(parents=True)
        rw = ResultWriter(d)
        rw.start()
        rw.incomplete("measure crashed: eval.json torn")
        assert rw.current()["state"] == STATE_INCOMPLETE

    def test_read_state_module_fn(self, tmp_path):
        d = tmp_path / "raw" / "armJ"
        d.mkdir(parents=True)
        assert read_state(d)["state"] == STATE_UNKNOWN
        ResultWriter(d).start()
        assert read_state(d)["state"] == STATE_STARTED
        assert read_state(d)["task_id"] == "armJ"

    def test_custom_log_path(self, tmp_path):
        d = tmp_path / "raw" / "armK"
        d.mkdir(parents=True)
        log = tmp_path / "central" / "queue.jsonl"
        rw = ResultWriter(d, log_path=log)
        rw.start()
        rw.done(meta=_fake_meta(750))
        assert log.is_file() and read_json_tolerant(d / LOG_FILENAME) is None
        assert rw.current()["state"] == STATE_DONE


# --------------------------------------------------------------------------- #
# scan_status: 队列健康分诊
# --------------------------------------------------------------------------- #
class TestScanStatus:
    def _make_paired_root(self, tmp_path):
        root = tmp_path / "runs" / "v25"
        (root / "raw").mkdir(parents=True)
        (root / "measured").mkdir()
        return root

    def test_paired_layout_full_ok_arm(self, tmp_path):
        root = self._make_paired_root(tmp_path)
        arm = root / "raw" / "A_ok"
        arm.mkdir()
        atomic_save_json(_fake_meta(750), arm / "meta.json")
        rw = ResultWriter(arm)
        rw.start()
        rw.done(meta=_fake_meta(750))
        (root / "measured" / "A_ok").mkdir(parents=True)
        atomic_save_json({"cross600": {}}, root / "measured" / "A_ok" / "summary_arm.json")

        rows = scan_status(root)
        assert len(rows) == 1
        r = rows[0]
        assert r["arm"] == "A_ok" and r["state"] == STATE_DONE
        assert r["raw_meta"] == "valid" and r["eval_json"] is False and r["summary"] is True
        assert r["flags"] == []

    def test_h1_window_meta_valid_summary_missing(self, tmp_path):
        """半写窗口 H1: meta 完整但 measured 摘要缺失 → 只需补跑 measure。"""
        root = self._make_paired_root(tmp_path)
        arm = root / "raw" / "B_nosum"
        arm.mkdir()
        atomic_save_json(_fake_meta(750), arm / "meta.json")
        rows = scan_status(root)
        assert rows[0]["flags"] == ["summary_missing"]
        assert rows[0]["raw_meta"] == "valid"

    def test_orphan_raw_generation_killed(self, tmp_path):
        """生成中途 kill: 有 batch 分片但无 meta.json → orphan_raw（整臂重生成）。"""
        root = self._make_paired_root(tmp_path)
        arm = root / "raw" / "C_orphan"
        arm.mkdir()
        (arm / "batch_00000.pt").write_bytes(b"x" * 8)
        rows = scan_status(root)
        assert rows[0]["flags"] == ["orphan_raw"]
        assert rows[0]["raw_meta"] == "missing"

    def test_done_but_meta_bad(self, tmp_path):
        """历史脏状态: 曾记 done（当时无 meta 凭证），随后 meta 被截短 → 双 flag。"""
        root = self._make_paired_root(tmp_path)
        arm = root / "raw" / "D_bad"
        arm.mkdir()
        rw = ResultWriter(arm)
        rw.start()
        rw.done(updates=10)  # 非 750 评测口径，meta_validated=null 记 done
        atomic_save_json(_fake_meta(500), arm / "meta.json")  # 之后 meta 被截短/重生成
        rows = scan_status(root)
        assert "raw_meta_incomplete_count" in rows[0]["flags"]
        assert "done_but_meta_bad" in rows[0]["flags"]

    def test_flat_layout_and_empty_root(self, tmp_path):
        flat = tmp_path / "runs" / "mech"
        (flat / "armX").mkdir(parents=True)
        atomic_save_json(_fake_meta(750), flat / "armX" / "meta.json")
        rows = scan_status(flat)
        assert len(rows) == 1 and rows[0]["arm"] == "armX"
        assert rows[0]["state"] == STATE_UNKNOWN  # 平铺布局无 measured 对照 → 不误报
        assert rows[0]["flags"] == []
        assert scan_status(tmp_path / "runs" / "empty_root") == []

    def test_cli_main_json_and_table(self, tmp_path, capsys):
        root = self._make_paired_root(tmp_path)
        arm = root / "raw" / "A_ok"
        arm.mkdir()
        atomic_save_json(_fake_meta(750), arm / "meta.json")
        rc_scan = scan_status(root)

        import src.v25.result_writer as m
        rc = m.main([str(root), "--json"])
        out = json.loads(capsys.readouterr().out)
        assert rc == 0 and out["n_arms"] == 1 and out["rows"][0]["arm"] == "A_ok"
        rc2 = m.main([str(root)])
        text = capsys.readouterr().out
        assert rc2 == 0 and "A_ok" in text and "1 arms" in text
        assert rc_scan[0]["age_min"] is not None

    def test_stale_flag_via_cli(self, tmp_path, capsys):
        root = self._make_paired_root(tmp_path)
        arm = root / "raw" / "A_stale"
        arm.mkdir()
        atomic_save_json(_fake_meta(750), arm / "meta.json")
        old = time.time() - 3600
        os.utime(arm / "meta.json", (old, old))
        import src.v25.result_writer as m
        m.main([str(root), "--stale-min", "30"])
        text = capsys.readouterr().out
        assert "stale" in text
