"""Serial 4090 Paper Audit Runner (F1, F3, F4, F5, F6, Stiefel-WikiArt, and F2 Qwen3.8-27B VLM Judge).

Enforces strict isolated-process execution per stage and a hard UTC epoch deadline so no GPU task
overruns the allocated compute window. Zero training / zero new learnable parameters.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from src.utils.runtime import REPO, save_json, utc


AUDIT_ROOT = REPO / "runs/paper_audit"


def _run_py(module: str, args: list[str], *, timeout_sec: float) -> dict[str, Any]:
    cmd = [sys.executable, "-u", "-m", module, *args]
    t0 = time.monotonic()
    print(f"[4090 audit] START {' '.join(cmd)} (timeout={timeout_sec:.0f}s)", flush=True)
    proc = subprocess.run(cmd, timeout=max(30.0, timeout_sec))
    dt = time.monotonic() - t0
    print(f"[4090 audit] DONE {module} rc={proc.returncode} in {dt:.1f}s", flush=True)
    return {"module": module, "args": args, "returncode": proc.returncode, "seconds": round(dt, 2)}


def _wait_download_if_needed(model_path: Path, min_bytes: int = 14_500_000_000, max_wait_s: float = 600.0) -> bool:
    t0 = time.monotonic()
    while time.monotonic() - t0 < max_wait_s:
        if model_path.exists() and model_path.stat().st_size >= min_bytes:
            # Check no curl still writing to it
            chk = subprocess.run(["pgrep", "-f", model_path.name], capture_output=True, text=True)
            if chk.returncode != 0:
                return True
        time.sleep(5.0)
    return model_path.exists() and model_path.stat().st_size >= min_bytes


def _wait_health(url: str, max_wait_s: float = 180.0) -> bool:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    t0 = time.monotonic()
    while time.monotonic() - t0 < max_wait_s:
        try:
            with opener.open(f"{url}/health", timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2.0)
    return False


def _calibrate_vlm(server_url: str) -> dict[str, Any]:
    from exp.paper_audit.f2_vlm_eval import F2_DIR, PAIRWISE_PROMPT, RUBRIC_PROMPT, _chat_vlm, _find_cand_path, _img_b64_uri, _parse_json_reply
    from src.utils.runtime import read_json

    man = read_json(F2_DIR / "f2_manifest.json")
    t0_task = man["tasks"][0]
    idx = int(t0_task["idx"])
    img_root = F2_DIR / "images"
    src_u = _img_b64_uri(img_root / "source" / f"{idx:04d}.png")
    ref_u = _img_b64_uri(img_root / "ref" / f"{idx:04d}.png")
    cands = [(_m, _find_cand_path(_m, idx)) for _m in ("cleave_pe_s085", "stylegallery", "stytr2")]
    cands = [(m, p) for m, p in cands if p is not None]

    lat_s = []
    parsed_ok = 0
    for m, p in cands:
        t0 = time.monotonic()
        rep = _chat_vlm(server_url, RUBRIC_PROMPT, [src_u, ref_u, _img_b64_uri(p)])
        dt = time.monotonic() - t0
        lat_s.append(dt)
        if _parse_json_reply(rep) is not None:
            parsed_ok += 1

    if len(cands) >= 2:
        t0 = time.monotonic()
        rep_pw = _chat_vlm(server_url, PAIRWISE_PROMPT, [src_u, ref_u, _img_b64_uri(cands[0][1]), _img_b64_uri(cands[1][1])], max_tokens=160)
        lat_s.append(time.monotonic() - t0)
        if _parse_json_reply(rep_pw) is not None:
            parsed_ok += 1

    mean_sec = float(sum(lat_s) / max(1, len(lat_s)))
    cal = {
        "n_smoke_calls": len(lat_s),
        "parsed_ok": parsed_ok,
        "mean_sec_per_call": round(mean_sec, 3),
        "latencies_sec": [round(x, 3) for x in lat_s],
    }
    save_json(F2_DIR / "vlm_calibration.json", cal)
    print(f"[F2 calibrate] {json.dumps(cal)}", flush=True)
    return cal


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deadline-epoch", type=float, required=True)
    ap.add_argument("--llama-bin", default=os.environ.get("LLAMA_SERVER_BIN", "../llama.cpp/build/bin/llama-server"))
    ap.add_argument("--model-path", default=os.environ.get("QWEN_GGUF_PATH", "../hf_cache/qwen38/Qwen3.8-27B-UD-Q4_K_S.gguf"))
    ap.add_argument("--mmproj-path", default=os.environ.get("QWEN_MMPROJ_PATH", "../hf_cache/qwen38/mmproj-F16.gguf"))
    ap.add_argument("--port", type=int, default=8089)
    args = ap.parse_args()

    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    status_p = AUDIT_ROOT / "audit_4090_status.json"
    status: dict[str, Any] = {"started_utc": utc(), "deadline_epoch": args.deadline_epoch, "stages": {}}

    def _rem() -> float:
        return args.deadline_epoch - time.time()

    phase_a = [
        ("f1_artist_holdout", "exp.paper_audit.f1_artist_holdout", ["--stage", "all"]),
        ("f3_instance_control", "exp.paper_audit.f3_instance_control", []),
        ("f4_asymmetry", "exp.paper_audit.f4_decoder_encoder_asymmetry", []),
        # F5 cancelled by user on 2026-11-28; never include in resumed batches.
        ("f6_p2a", "exp.paper_audit.f6_photo_to_art", []),
        ("stiefel_wikiart_2150", "exp.paper_audit.stiefel_wikiart_audit", []),
        ("f2_prepare", "exp.paper_audit.f2_vlm_eval", ["--stage", "prepare", "--n-per-target", "15"]),
    ]

    for name, mod, mod_args in phase_a:
        rem = _rem()
        if rem < 120:
            status["stages"][name] = {"skipped": "near_deadline"}
            break
        try:
            res = _run_py(mod, mod_args, timeout_sec=rem - 60)
            status["stages"][name] = res
        except Exception as exc:
            status["stages"][name] = {"error": str(exc)}
            print(f"[4090 audit] WARN {name} failed: {exc}", flush=True)
        save_json(status_p, status)

    # Phase B: Launch Qwen3.8 llama-server on GPU0 cleanly after all PyTorch phases have exited
    llama_bin = Path(args.llama_bin)
    model_p = Path(args.model_path)
    mmproj_p = Path(args.mmproj_path)
    rem = _rem()
    if rem > 300 and llama_bin.exists():
        if _wait_download_if_needed(model_p, max_wait_s=min(600.0, rem - 240.0)) and mmproj_p.exists():
            env = os.environ.copy()
            bin_dir = str(llama_bin.parent)
            extra_lib = ":".join([
                bin_dir,
                env.get("CUDA_LIB_DIR", ""),
                env.get("LD_LIBRARY_PATH", ""),
            ]).strip(":")
            env["LD_LIBRARY_PATH"] = extra_lib
            log_p = AUDIT_ROOT / "llama_server.log"
            srv_cmd = [
                str(llama_bin),
                "-m", str(model_p),
                "-mm", str(mmproj_p),
                "--host", "127.0.0.1",
                "--port", str(args.port),
                "-ngl", "99",
                "-fa", "on",
                "-c", "8192",
                "--reasoning", "off",
                "--reasoning-budget", "0",
                "--no-reasoning-preserve",
                "-np", "1",
            ]
            print("[4090 audit] Launching llama-server for F2...", flush=True)
            with log_p.open("wb") as logf:
                srv = subprocess.Popen(srv_cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
                try:
                    url = f"http://127.0.0.1:{args.port}"
                    if _wait_health(url, max_wait_s=180.0):
                        cal = _calibrate_vlm(url)
                        status["stages"]["f2_calibrate"] = cal
                        save_json(status_p, status)

                        sec_call = max(0.8, float(cal["mean_sec_per_call"]))
                        rem_judge = max(60.0, _rem() - 180.0)
                        max_calls = int(rem_judge / sec_call)
                        # Reserve ~240 calls for 40 pairwise comparisons (40 * 6 pairs), remainder / 8 methods = max_tasks
                        n_pw = min(40, max(10, (max_calls // 6) // 3))
                        max_tasks = min(75, max(15, (max_calls - n_pw * 6) // 8))
                        print(f"[F2 budget] rem={rem_judge:.0f}s rate={sec_call:.2f}s/call -> max_tasks={max_tasks}, n_pairwise={n_pw}", flush=True)
                        res_j = _run_py(
                            "exp.paper_audit.f2_vlm_eval",
                            ["--stage", "judge", "--server-url", url, "--max-tasks", str(max_tasks), "--n-pairwise", str(n_pw)],
                            timeout_sec=max(60.0, _rem() - 60.0),
                        )
                        status["stages"]["f2_judge"] = res_j
                    else:
                        status["stages"]["f2_judge"] = {"error": "llama-server healthcheck timed out"}
                finally:
                    srv.terminate()
                    try:
                        srv.wait(timeout=15)
                    except Exception:
                        srv.kill()
        else:
            status["stages"]["f2_judge"] = {"error": f"Model weights incomplete: {model_p}"}
    else:
        status["stages"]["f2_judge"] = {"skipped": f"rem={rem:.0f}s or missing {llama_bin}"}

    status["finished_utc"] = utc()
    save_json(status_p, status)
    print(f"[4090 audit ALL DONE] saved {status_p}", flush=True)


if __name__ == "__main__":
    main()
