r"""V26-L 4090 长队列 (36h) — 两级任务 + 三态路由 + 预算追踪 + 诚实完成判定.

docs/1120/08_v26_long_plan.md §四. 与已完成的 queue_k.py 完全独立 (不覆盖).

机制要点 (用户 2026-11-21 指令, 防止提前宣布完成):
- 一级 = L0–L6 科学必做; 二级 = 预算填充任务池 (第二/三种子、训练长度、数据规模、
  路线均衡、SWD K/尺度、参考敏感性、Decoder 追加训练);
- 模块未实现的任务标记 blocked 并周期重试 (Agent B 在队列运行期间持续交付新模块,
  同步后自动拾取);
- 状态 JSON 实时更新: science_tasks_done / budget_remaining / gpu_compute_seconds /
  wallclock_elapsed / pending_replications / blocked_tasks / valid_results_count;
- 每任务三态路由 (success_next / failure_next / inconclusive_next), 失败不断链;
- 完成判定 = 必做任务全完成 ∧ 可执行备用任务处理 ∧ 预算窗口结束 (或明确资源阻断);
  提前完成须报告剩余预算与原因.

用法:  tmux: python -m exp.v26.gpu4090.queue_k_long
"""
from __future__ import annotations

import importlib
import importlib.util
import time
import traceback
from types import ModuleType

from exp.v26 import OUT, ROOT, sanitize_cuda_env, save_json, seed_all, utc

STATUS_PATH = OUT / "gpu4090" / "queue_long_status.json"
BUDGET_HOURS = 36.0
BLOCKED_RETRY_SEC = 900  # 15 min
POOL_TASKS = [
    "pool_seed23", "pool_seed45", "pool_len3000", "pool_len6000", "pool_len9000",
    "pool_datascale25", "pool_datascale50", "pool_routebalance",
    "pool_swd_kgrid", "pool_ref_sensitivity", "pool_decoder_extend",
]

STATUS: dict = {
    "plan": "V26-L 36h long queue (docs/1120/08_v26_long_plan.md)",
    "started_utc": None,
    "budget_hours": BUDGET_HOURS,
    "science_tasks_done": [],
    "valid_results_count": 0,
    "pending_replications": [],
    "blocked_tasks": [],
    "gpu_compute_seconds": 0.0,
    "wallclock_elapsed_h": 0.0,
    "budget_remaining_h": BUDGET_HOURS,
    "task_history": [],
    "finished": False,
    "finish_reason": None,
}


def _save_status() -> None:
    STATUS["wallclock_elapsed_h"] = round((time.time() - STATUS["_t0"]) / 3600.0, 3)
    STATUS["budget_remaining_h"] = round(max(0.0, BUDGET_HOURS - STATUS["wallclock_elapsed_h"]), 3)
    STATUS["science_tasks_done"] = sorted(set(STATUS["science_tasks_done"]))
    save_json(STATUS_PATH, {k: v for k, v in STATUS.items() if not k.startswith("_")})


def _budget_left() -> float:
    return BUDGET_HOURS - (time.time() - STATUS["_t0"]) / 3600.0


# ----------------------------------------------------- 任务模块发现 ----

def _module_ready(name: str) -> ModuleType | None:
    """模块文件已同步到 4090 → 可导入; 否则 None (blocked)."""
    try:
        return importlib.import_module(name)
    except Exception:
        return None


# ----------------------------------------------------- 任务实现包装 ----

def _t_lpc() -> str:
    """前置修正: K2-A 更正重跑 (margin bug + AUC) + K2-C 真图正对照."""
    from exp.v26.k2_style_subject_audit import k2a_discriminability, save_k2a_results
    from exp.v26.k2c_positive_control import run_positive_control

    bundle = _shared_bundle()
    a = k2a_discriminability(bundle, device="cuda")
    save_k2a_results(a)
    run_positive_control(_POOL, device="cuda")
    return "done"


def _t_l0() -> str:
    from exp.v26.gpu4090.l0_crossstyle_causal import run_l0
    run_l0(_POOL, device="cuda")
    return "done"


def _t_l1() -> str:
    from exp.v26.gpu4090.l1_flow_arbitrary_ref import run_l1
    return run_l1(_POOL, device="cuda")


def _t_l2() -> str:
    from exp.v26.gpu4090.l2_decoder_matched import run_l2
    return run_l2(_POOL, device="cuda")


def _t_l3() -> str:
    from exp.v26.gpu4090.l3_swd_branch_ablation import run_l3
    return run_l3(_POOL, device="cuda")


def _t_l4() -> str:
    from exp.v26.gpu4090.l4_rollout_bound import run_l4
    return run_l4(_POOL, device="cuda")


def _t_l5() -> str:
    from exp.v26.gpu4090.l5_fewstep_q32 import run_l5
    return run_l5(_POOL, device="cuda")


def _t_l6() -> str:
    from exp.v26.gpu4090.l6_closeout import run_l6
    return run_l6(_POOL, device="cuda")


# 一级科学必做任务: (id, fn, required, success_next, failure_next, inconclusive_next)
SCIENCE_TASKS = [
    ("L-PC", _t_lpc, True, "L0", "L0", "L0"),
    ("L0", _t_l0, True, "L1", "L1", "L1"),
    ("L1", _t_l1, True, "L2", "L2", "L2"),
    ("L2", _t_l2, True, "L3", "L6", "L3"),
    ("L3", _t_l3, True, "L4", "L4", "L4"),
    ("L4", _t_l4, True, "L5", "L5", "L5"),
    ("L5", _t_l5, True, "L6", "L6", "L6"),
    ("L6", _t_l6, True, None, None, None),
]

# 二级预算填充池 (模块后补; blocked 时跳过并在下轮重试)
POOL_IMPLS: dict[str, str] = {
    "pool_seed23": "exp.v26.gpu4090.pool_tasks:run_seed23",
    "pool_seed45": "exp.v26.gpu4090.pool_tasks:run_seed45",
    "pool_len3000": "exp.v26.gpu4090.pool_tasks:run_len3000",
    "pool_len6000": "exp.v26.gpu4090.pool_tasks:run_len6000",
    "pool_len9000": "exp.v26.gpu4090.pool_tasks:run_len9000",
    "pool_datascale25": "exp.v26.gpu4090.pool_tasks:run_datascale25",
    "pool_datascale50": "exp.v26.gpu4090.pool_tasks:run_datascale50",
    "pool_routebalance": "exp.v26.gpu4090.pool_tasks:run_routebalance",
    "pool_swd_kgrid": "exp.v26.gpu4090.pool_tasks:run_swd_kgrid",
    "pool_ref_sensitivity": "exp.v26.gpu4090.pool_tasks:run_ref_sensitivity",
    "pool_decoder_extend": "exp.v26.gpu4090.pool_tasks:run_decoder_extend",
}

_POOL: str = ""
_shared_bundle_cache = None


def _shared_bundle():
    global _shared_bundle_cache
    if _shared_bundle_cache is None:
        from exp.v19.fiber_duality import build_or_load_v19_foundation
        _shared_bundle_cache = build_or_load_v19_foundation(_POOL, device="cuda")
    return _shared_bundle_cache


def _run_task(tid: str, fn, required: bool) -> str:
    """执行单任务并返回三态结果 (done / failed / inconclusive / blocked)."""
    print(f"\n[QUEUE-L] ===== {tid} start {utc()} =====", flush=True)
    t0 = time.time()
    try:
        outcome = fn()
    except Exception:
        print(f"[QUEUE-L] ===== {tid} FAILED =====\n{traceback.format_exc()}", flush=True)
        STATUS["gpu_compute_seconds"] += time.time() - t0
        return "failed"
    STATUS["gpu_compute_seconds"] += time.time() - t0
    outcome = str(outcome).lower() if outcome is not None else "done"
    return outcome if outcome in ("done", "failed", "inconclusive", "blocked") else "done"


def _find_route(tid: str, outcome: str) -> str | None:
    for t_id, _fn, _req, s_next, f_next, i_next in SCIENCE_TASKS:
        if t_id == tid:
            return {"done": s_next, "failed": f_next, "inconclusive": i_next}[outcome]
    return None


def main():
    global _POOL
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    _POOL = args.pool

    sanitize_cuda_env()
    seed_all(20261185)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gpu4090").mkdir(parents=True, exist_ok=True)

    STATUS["_t0"] = time.time()
    STATUS["started_utc"] = utc()
    _save_status()
    print(f"[QUEUE-L] V26-L 36h long queue started {utc()} (budget {BUDGET_HOURS}h)", flush=True)

    # ---- 一级: 科学必做任务 (顺序 + 三态路由 + blocked 重试) ----
    idx = 0
    while idx < len(SCIENCE_TASKS):
        if _budget_left() <= 0.0:
            STATUS["finish_reason"] = "budget_exhausted_mid_science"
            break
        tid, fn, required, _, _, _ = SCIENCE_TASKS[idx]
        # 模块未同步 → blocked, 等待 Agent B 交付 (周期重试直到预算耗尽)
        mod_name = {
            "L-PC": None, "L0": "exp.v26.gpu4090.l0_crossstyle_causal",
            "L1": "exp.v26.gpu4090.l1_flow_arbitrary_ref",
            "L2": "exp.v26.gpu4090.l2_decoder_matched",
            "L3": "exp.v26.gpu4090.l3_swd_branch_ablation",
            "L4": "exp.v26.gpu4090.l4_rollout_bound",
            "L5": "exp.v26.gpu4090.l5_fewstep_q32",
            "L6": "exp.v26.gpu4090.l6_closeout",
        }.get(tid)
        if mod_name and importlib.util.find_spec(mod_name) is None:
            if tid not in STATUS["blocked_tasks"]:
                STATUS["blocked_tasks"].append(tid)
            print(f"[QUEUE-L] {tid} BLOCKED (module not synced) — retry in {BLOCKED_RETRY_SEC}s "
                  f"(budget left {_budget_left():.2f}h)", flush=True)
            if _budget_left() <= 0.0:
                STATUS["finish_reason"] = "budget_exhausted_with_blocked_tasks"
                break
            _save_status()
            time.sleep(BLOCKED_RETRY_SEC)
            continue
        if tid in STATUS["blocked_tasks"]:
            STATUS["blocked_tasks"].remove(tid)

        outcome = _run_task(tid, fn, required)
        entry = {"task": tid, "outcome": outcome, "utc": utc(),
                 "wallclock_h": round((time.time() - STATUS["_t0"]) / 3600.0, 3)}
        STATUS["task_history"].append(entry)
        if outcome == "done":
            STATUS["science_tasks_done"].append(tid)
            STATUS["valid_results_count"] += 1
        nxt = _find_route(tid, outcome)
        _save_status()
        if nxt is None:
            break
        # 找到下一任务在 SCIENCE_TASKS 中的位置 (路由可能跳过)
        names = [t[0] for t in SCIENCE_TASKS]
        idx = names.index(nxt) if nxt in names else idx + 1

    # ---- 二级: 预算填充任务池 (模块未交付 → 周期重试直至预算耗尽, 不提前 finish) ----
    pending_pool = list(POOL_TASKS)
    while pending_pool and _budget_left() > 0.05 and not STATUS.get("hard_block"):
        progressed = False
        for pool_id in list(pending_pool):
            if _budget_left() <= 0.05:
                break
            target = POOL_IMPLS.get(pool_id, "")
            mod_path, fn_name = target.split(":")
            mod = _module_ready(mod_path)
            if mod is None:
                if f"{pool_id} (module pending)" not in STATUS["blocked_tasks"]:
                    STATUS["blocked_tasks"].append(f"{pool_id} (module pending)")
                _save_status()
                continue
            print(f"\n[QUEUE-L] ===== {pool_id} start {utc()} =====", flush=True)
            t0 = time.time()
            try:
                getattr(mod, fn_name)(_POOL, device="cuda")
                STATUS["task_history"].append({"task": pool_id, "outcome": "done", "utc": utc()})
                STATUS["valid_results_count"] += 1
            except Exception:
                STATUS["task_history"].append({"task": pool_id, "outcome": "failed", "utc": utc()})
                print(f"[QUEUE-L] {pool_id} FAILED\n{traceback.format_exc()}", flush=True)
            STATUS["gpu_compute_seconds"] += time.time() - t0
            _save_status()

    # ---- 完成判定 (诚实口径): budget 耗尽才允许 finish ----
    all_done = all(t[0] in STATUS["science_tasks_done"] for t in SCIENCE_TASKS if t[2])
    blocked = list(STATUS["blocked_tasks"])
    STATUS["finished"] = True
    if all_done and not blocked:
        STATUS["finish_reason"] = (
            "science_complete_and_budget_exhausted" if _budget_left() <= 0.05
            else "science_complete_early_budget_remaining")
    elif blocked:
        STATUS["finish_reason"] = "blocked_tasks_remaining (modules not delivered in window)"
    _save_status()
    print(f"\n[QUEUE-L] ===== QUEUE FINISHED {utc()} =====", flush=True)
    print(f"[QUEUE-L] science_done={STATUS['science_tasks_done']} "
          f"blocked={blocked} valid_results={STATUS['valid_results_count']} "
          f"gpu_seconds={STATUS['gpu_compute_seconds']:.0f} "
          f"budget_left={STATUS['budget_remaining_h']}h "
          f"finish_reason={STATUS['finish_reason']}", flush=True)


if __name__ == "__main__":
    main()
