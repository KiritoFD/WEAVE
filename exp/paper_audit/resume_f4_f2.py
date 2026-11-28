"""Bounded serial completion of F4 and F2 only; wait hourly for a free 4090."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .run_4090_audit import _run_py, _wait_health
from src.utils.runtime import REPO, save_json, utc

ROOT = REPO / "runs/paper_audit"
DEADLINE = 1795884300.0  # Original audit deadline, never extended by retries.


def idle():
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    mem, util = [int(s.strip()) for s in r.stdout.strip().splitlines()[0].split(",")]
    # Existing host/context allocation is about 1810 MiB. Never kill other jobs.
    return mem < 2500 and util < 5, {"memory_mib": mem, "utilization": util}


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / "resume_f4_f2.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_p = ROOT / "resume_f4_f2_status.json"
    status = {"pid": os.getpid(), "started_utc": utc(), "deadline_epoch": DEADLINE,
              "requested_stages": ["F4", "F2"], "F5": "cancelled_by_user", "stages": {}}
    def persist(phase):
        status.update(phase=phase, updated_utc=utc())
        save_json(status_p, status)
    persist("waiting_for_gpu")
    while time.time() < DEADLINE - 3600:
        free, gpu = idle()
        status["last_gpu_check"] = gpu
        if free:
            break
        print(f"[resume] GPU busy {gpu}; next check in 3600s", flush=True)
        persist("waiting_for_gpu")
        time.sleep(min(3600, max(1, DEADLINE - time.time() - 3600)))
    else:
        persist("blocked_gpu_busy_before_deadline")
        return

    # Each stage is an isolated process; F4 frees GPU allocations before VLM load.
    persist("running_f4")
    try:
        status["stages"]["F4"] = _run_py("exp.paper_audit.f4_decoder_encoder_asymmetry", [],
                                          timeout_sec=max(30, DEADLINE - time.time() - 1800))
    except Exception as exc:
        status["stages"]["F4"] = {"returncode": -1, "error": str(exc)}
    persist("f4_finished")
    free, gpu = idle()
    if not free or time.time() > DEADLINE - 1200:
        status["last_gpu_check"] = gpu
        persist("f2_blocked_gpu_or_deadline")
        return
    workspace = Path(os.environ.get("WEAVE_REMOTE_WORKSPACE", str(Path.home() / "Workspace")))
    llama = workspace / "llama.cpp/build/bin/llama-server"
    model = workspace / "hf_cache/qwen38/Qwen3.8-27B-UD-Q4_K_S.gguf"
    mmproj = model.parent / "mmproj-F16.gguf"
    cmd = [str(llama), "-m", str(model), "-mm", str(mmproj), "--host", "127.0.0.1",
           "--port", "8089", "-ngl", "99", "-fa", "on", "-c", "8192", "--reasoning", "off",
           "--reasoning-budget", "0", "--no-reasoning-preserve", "-np", "1"]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(filter(None, [str(llama.parent), env.get("CUDA_LIB_DIR"), env.get("LD_LIBRARY_PATH")]))
    with (ROOT / "llama_f2_completion.log").open("ab") as log:
        srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        status["server_pid"] = srv.pid
        try:
            persist("starting_f2_server")
            if not _wait_health("http://127.0.0.1:8089", 180):
                raise RuntimeError("VLM server not healthy")
            persist("running_f2_balanced")
            status["stages"]["F2"] = _run_py("exp.paper_audit.vlm_pairs", [],
                                               timeout_sec=max(30, DEADLINE - time.time() - 60))
        except Exception as exc:
            status["stages"]["F2"] = {"returncode": -1, "error": str(exc)}
        finally:
            srv.terminate()
            try:
                srv.wait(20)
            except subprocess.TimeoutExpired:
                srv.kill()
                srv.wait()
    persist("complete" if all(v["returncode"] == 0 for v in status["stages"].values()) else "failed_stage")


if __name__ == "__main__":
    main()
