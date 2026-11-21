r"""V26-L 4090 长队列 (36h) — 两级任务 + 三态路由 + 预算追踪 + 诚实完成判定.

docs/1120/08_v26_long_plan.md §四. 与已完成的 queue_k.py 完全独立 (不覆盖).

机制要点 (用户 2026-11-21 指令, 防止提前宣布完成):
- 一级 = L0–L6 科学必做; 二级 = 预算填充任务池 (第二/三种子、训练长度、数据规模、
  路线均衡、SWD K/尺度、参考敏感性、Decoder 追加训练);
- **子进程隔离**: 每个科学阶段 / 池任务在独立进程中运行 (v2 修复) — 单进程长队列
  会跨阶段累积 GPU 显存 (实测 L1→L2 间累积 ~21GB 导致 L2 OOM), 子进程退出即
  完整释放; 与 src.eval.runner 裁判隔离同一设计原则;
- 模块未实现的任务标记 blocked 并周期重试 (Agent B 在队列运行期间持续交付新模块,
  同步后自动拾取);
- 池任务失败上限 (v2 修复): 连续失败 2 次即永久淘汰 — 否则 KeyError 类代码错误
  会以 ~3s/次 空转烧光全部预算;
- 状态 JSON 实时更新: science_tasks_done / budget_remaining / gpu_compute_seconds /
  wallclock_elapsed / pending_replications / blocked_tasks / valid_results_count;
- 每任务三态路由 (success_next / failure_next / inconclusive_next), 失败不断链;
- 完成判定 = 必做任务全完成 ∧ 可执行备用任务处理 ∧ 预算窗口结束 (或明确资源阻断);
  提前完成须报告剩余预算与原因.

用法:  tmux: python -m exp.v26.gpu4090.queue_k_long
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import time

from exp.v26 import OUT, ROOT, save_json, seed_all, utc

STATUS_PATH = OUT / "gpu4090" / "queue_long_status.json"
BUDGET_HOURS = 36.0
BLOCKED_RETRY_SEC = 900  # 15 min
POOL_FAIL_CAP = 2        # 连续失败 2 次永久淘汰 (防 KeyError 空转烧预算)
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


# ----------------------------------------------------- 子进程执行 ----

def _spawn(mod: str, pool: str, device: str, extra: list[str] | None = None) -> int:
    """独立进程运行一个阶段模块 (退出即释放全部 GPU 显存)."""
    cmd = [sys.executable, "-u", "-m", mod, "--pool", pool, "--device", device]
    if extra:
        cmd += extra
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    r = subprocess.run(cmd, cwd=str(ROOT), env=env)
    return r.returncode


# ----------------------------------------------------- 任务模块发现 ----

def _module_ready(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


# 一级科学必做任务: (id, 子模块列表, required, success_next, failure_next, inconclusive_next)
SCIENCE_TASKS = [
    ("L-PC", ["exp.v26.gpu4090.l_pc"], True, "L0", "L0", "L0"),
    ("L0", ["exp.v26.gpu4090.l0_crossstyle_causal"], True, "L1", "L1", "L1"),
    ("L1", ["exp.v26.gpu4090.l1_flow_arbitrary_ref"], True, "L2", "L2", "L2"),
    ("L2", ["exp.v26.gpu4090.l2_decoder_matched"], True, "L3", "L3", "L3"),
    ("L3", ["exp.v26.gpu4090.l3_swd_branch_ablation"], True, "EXP2", "EXP2", "EXP2"),
    ("EXP2", ["exp.v26.gpu4090.exp2_locality_probes"], True, "L4", "L4", "L4"),
    ("L4", ["exp.v26.gpu4090.l4_rollout_bound"], True, "L5", "L5", "L5"),
    ("L5", ["exp.v26.gpu4090.l5_fewstep_q32"], True, "L6", "L6", "L6"),
    ("L6", ["exp.v26.gpu4090.l6_closeout"], True, None, None, None),
]

# 二级预算填充池 (统一由 pool_tasks 子进程按任务名执行; blocked 时跳过并在下轮重试)
POOL_MODULE = "exp.v26.gpu4090.pool_tasks"

_POOL: str = ""


def _run_task(tid: str, mods: list[str], required: bool) -> str:
    """执行单科学任务 (逐子模块顺序子进程) 并返回三态结果."""
    print(f"\n[QUEUE-L] ===== {tid} start {utc()} =====", flush=True)
    t0 = time.time()
    for m in mods:
        if not _module_ready(m):
            print(f"[QUEUE-L] {tid} BLOCKED (module {m} not synced)", flush=True)
            STATUS["gpu_compute_seconds"] += time.time() - t0
            return "blocked"
        rc = _spawn(m, _POOL, "cuda")
        if rc != 0:
            print(f"[QUEUE-L] ===== {tid} FAILED ({m} rc={rc}) =====", flush=True)
            STATUS["gpu_compute_seconds"] += time.time() - t0
            return "failed"
    STATUS["gpu_compute_seconds"] += time.time() - t0
    return "done"


def _find_route(tid: str, outcome: str) -> str | None:
    for t_id, _m, _req, s_next, f_next, i_next in SCIENCE_TASKS:
        if t_id == tid:
            return {"done": s_next, "failed": f_next, "inconclusive": i_next,
                    "blocked": None}[outcome]
    return None


def main():
    global _POOL
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    _POOL = args.pool

    seed_all(20261185)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gpu4090").mkdir(parents=True, exist_ok=True)

    STATUS["_t0"] = time.time()
    STATUS["started_utc"] = utc()
    _save_status()
    print(f"[QUEUE-L] V26-L 36h long queue started {utc()} (budget {BUDGET_HOURS}h, "
          f"subprocess-isolated v2)", flush=True)

    # ---- 一级: 科学必做任务 (顺序 + 三态路由 + blocked 重试) ----
    idx = 0
    while idx < len(SCIENCE_TASKS):
        if _budget_left() <= 0.0:
            STATUS["finish_reason"] = "budget_exhausted_mid_science"
            break
        tid, mods, required, _, _, _ = SCIENCE_TASKS[idx]
        if any(not _module_ready(m) for m in mods):
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

        outcome = _run_task(tid, mods, required)
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

    # ---- 二级: 预算填充任务池 (每任务独立子进程; 连续失败 2 次永久淘汰) ----
    pending_pool = list(POOL_TASKS)
    fail_counts: dict[str, int] = {}
    while pending_pool and _budget_left() > 0.05:
        progressed = False
        for pool_id in list(pending_pool):
            if _budget_left() <= 0.05:
                break
            if not _module_ready(POOL_MODULE):
                if f"{pool_id} (module pending)" not in STATUS["blocked_tasks"]:
                    STATUS["blocked_tasks"].append(f"{pool_id} (module pending)")
                _save_status()
                continue
            print(f"\n[QUEUE-L] ===== {pool_id} start {utc()} =====", flush=True)
            t0 = time.time()
            rc = _spawn(POOL_MODULE, _POOL, "cuda", extra=[pool_id])
            STATUS["gpu_compute_seconds"] += time.time() - t0
            if rc == 0:
                STATUS["task_history"].append({"task": pool_id, "outcome": "done", "utc": utc()})
                STATUS["valid_results_count"] += 1
                pending_pool.remove(pool_id)
                progressed = True
            else:
                fail_counts[pool_id] = fail_counts.get(pool_id, 0) + 1
                STATUS["task_history"].append(
                    {"task": pool_id, "outcome": f"failed({fail_counts[pool_id]}/{POOL_FAIL_CAP})",
                     "utc": utc()})
                print(f"[QUEUE-L] {pool_id} FAILED (rc={rc}, "
                      f"{fail_counts[pool_id]}/{POOL_FAIL_CAP})", flush=True)
                if fail_counts[pool_id] >= POOL_FAIL_CAP:
                    pending_pool.remove(pool_id)  # 永久淘汰, 防止空转烧预算
            _save_status()
        if not progressed:
            # 全部剩余任务本轮都没成功 → 等待模块同步窗口, 不热转
            if STATUS["blocked_tasks"] or fail_counts:
                time.sleep(60)
            else:
                break

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
    else:
        STATUS["finish_reason"] = "science_incomplete (failed tasks routed through, budget remains)"
    _save_status()
    print(f"\n[QUEUE-L] ===== QUEUE FINISHED {utc()} =====", flush=True)
    print(f"[QUEUE-L] science_done={STATUS['science_tasks_done']} "
          f"blocked={blocked} valid_results={STATUS['valid_results_count']} "
          f"gpu_seconds={STATUS['gpu_compute_seconds']:.0f} "
          f"budget_left={STATUS['budget_remaining_h']}h "
          f"finish_reason={STATUS['finish_reason']}", flush=True)


if __name__ == "__main__":
    main()
