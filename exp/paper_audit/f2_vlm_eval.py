"""F2: Independent Blind VLM Visual Audit via Local Qwen3.8-27B (llama-server Q4_K_S + mmproj-F16).

Two-phase clean execution (never mixes PyTorch VRAM with 17GB llama-server):
  --stage prepare : Extracts exact S10B-decoded RGB PNGs for CLEAVE-E (l075), CLEAVE-P/E (s085, s100),
                    and CLEAVE-One (t50) plus Source & Core-5 Reference PNGs on frozen stratified tasks.
  --stage judge   : Queries running llama-server OpenAI-compatible API (blind randomized order, zero
                    method names in prompts, strict JSON rubric + head-to-head position-swapped pairs).
"""
from __future__ import annotations

import argparse
import base64
import gc
import io
import json
import random
import re
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch

from exp.v18 import tasks_and_sources
from exp.v20.slf_flow import project_latent_y16_lcn
from exp.v27_single.bare_q32 import load_cache
from exp.v32_nodino.build_cnx_assets import build_or_load_cnx_assets
from exp.v32_nodino.cnx_layout import match_cnx_k8_submode
from exp.v33_semantic_control.models import integrate_v33_trajectory
from exp.v35_final import build_pcore_startpoint
from exp.v35_final.common import load_pcore_v35, load_v32_ecore_and_priors, make_v35_schedule
from exp.v36_art_foundation import CACHE_DIR as V36_CACHE_DIR
from exp.v36_art_foundation.eval_v36 import load_v36_models, reconstruct_reference_a3, sample_v36_ode
from src.utils import decode_s10, load_s10_decoder
from src.utils.protocol import EVAL_SEED, STYLES
from src.utils.runtime import REPO, read_json, sanitize_cuda_env, save_json, utc


F2_DIR = REPO / "runs/paper_audit/f2_vlm_eval"
EXT_DIR = REPO / "runs/paper_audit/f2_ext_assets"
V36_P2_CKPT = REPO / "runs/v36_art_foundation/checkpoints/v36_main_p2_final.pt"
P8_REPORT_PATH = REPO / "runs/v35_final/p8_final_report.json"

METHODS_ORDER = (
    "cleave_e_l075",
    "cleave_pe_s085",
    "cleave_pe_s100",
    "cleave_one_t50",
    "stylegallery",
    "stytr2",
    "aespa",
    "seedream",
)

RUBRIC_PROMPT = (
    "You are an impartial art-historian and visual quality critic evaluating style transfer.\n"
    "You are given 3 images in order:\n"
    "  Image 1: Content Source painting.\n"
    "  Image 2: Style Reference painting.\n"
    "  Image 3: Candidate Stylized Output.\n"
    "Score Image 3 strictly from 1 to 5 (integer) on each independent axis:\n"
    "  content_preservation (1=unrecognizable structure, 5=faithful scene/object geometry & layout from Image 1)\n"
    "  style_faithfulness (1=unchanged/wrong style, 5=strong artistic brushwork, stroke rhythm, tone & palette of Image 2)\n"
    "  artistic_repainting (1=merely a flat global color/contrast filter over Photo/Image 1 edges, 5=authentic painterly re-synthesis & texture)\n"
    "  artifact_cleanliness (1=severe blotches, noise, tiling or broken forms, 5=clean coherent painting)\n"
    "  no_layout_leakage (1=objects/shapes from Image 2 wrongly pasted into Image 3, 5=zero spatial layout copied from Image 2)\n"
    "Output ONLY valid compact JSON with keys: "
    '{"content_preservation":int,"style_faithfulness":int,"artistic_repainting":int,"artifact_cleanliness":int,"no_layout_leakage":int,"brief_note":"short phrase"}'
)

PAIRWISE_PROMPT = (
    "You are an impartial art critic comparing two stylized outputs on the same Content and Style Reference.\n"
    "Images in order:\n"
    "  Image 1: Content Source.\n"
    "  Image 2: Style Reference.\n"
    "  Image 3: Candidate X.\n"
    "  Image 4: Candidate Y.\n"
    "Which candidate achieves better overall painterly style transfer WITHOUT collapsing content or pasting Reference shapes?\n"
    'Output ONLY JSON: {"winner":"X" or "Y" or "TIE","style_winner":"X" or "Y" or "TIE","content_winner":"X" or "Y" or "TIE","reason":"short phrase"}'
)


def _select_stratified_cross_indices(n_per_target: int = 15) -> list[int]:
    """Select stratified cross-style task indices (balanced 15 x 5 = 75 cross tasks) with valid external images."""
    tasks = read_json(REPO / "eval_protocol/tasks_750.json")["tasks"]
    # Require valid external files if EXT_DIR is populated
    by_tgt: dict[int, list[int]] = {s: [] for s in range(5)}
    for i, t in enumerate(tasks):
        if int(t["source_style_id"]) == int(t["target_style_id"]):
            continue
        if (EXT_DIR / "seedream").exists():
            ok = all((EXT_DIR / m / f"{i:04d}.jpg").exists() or (EXT_DIR / m / f"{i:04d}.png").exists() for m in ("stylegallery", "stytr2", "aespa", "seedream"))
            if not ok:
                continue
        by_tgt[int(t["target_style_id"])].append(i)

    picked: list[int] = []
    for s in range(5):
        cands = by_tgt[s]
        step = max(1, len(cands) // n_per_target)
        sel = cands[::step][:n_per_target]
        picked.extend(sel)
    return sorted(picked)


@torch.no_grad()
def prepare_images(n_per_target: int = 15, device: str = "cuda") -> dict[str, Any]:
    F2_DIR.mkdir(parents=True, exist_ok=True)
    img_root = F2_DIR / "images"
    manifest_p = F2_DIR / "f2_manifest.json"
    if manifest_p.exists():
        return read_json(manifest_p)

    indices = _select_stratified_cross_indices(n_per_target)
    tasks_all, z_src_all, _ = tasks_and_sources()
    tasks_sel = [tasks_all[i] for i in indices]
    idx_t = torch.tensor(indices, dtype=torch.long)

    # Save Source & Target Reference PNGs (512px)
    for dname in ("source", "ref", "cleave_e_l075", "cleave_pe_s085", "cleave_pe_s100", "cleave_one_t50"):
        (img_root / dname).mkdir(parents=True, exist_ok=True)

    for tidx, t in zip(indices, tasks_sel):
        with Image.open(REPO / t["source_rel_path"]) as im:
            im.convert("RGB").resize((512, 512), Image.Resampling.BICUBIC).save(img_root / "source" / f"{tidx:04d}.png")

    ea = torch.load(V36_CACHE_DIR / "v36_eval_assets.pt", map_location="cpu", weights_only=False)
    dec = load_s10_decoder(device)
    z_refs = ea["refs150"]["z_ref"]
    for tidx, t in zip(indices, tasks_sel):
        r_idx = int(t["target_style_id"]) * 30 + (tidx % 30)
        rg = decode_s10(dec, z_refs[r_idx:r_idx + 1].to(device).float(), batch=1, output_cpu=True)[0].clamp(0, 1)
        Image.fromarray((rg.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(img_root / "ref" / f"{tidx:04d}.png")

    # Prepare features for CLEAVE-E, CLEAVE-P/E, CLEAVE-One
    z_src = z_src_all[idx_t].to(device).float()
    eps = torch.randn(z_src_all.shape, generator=torch.Generator().manual_seed(EVAL_SEED))[idx_t].to(device).float()
    v18 = load_cache("runs/v18/cache/ccf_pool_cache.pt")
    ts = load_cache("runs/v18/cache/test750_struct_cache.pt")
    assets = build_or_load_cnx_assets(device=device)
    q16 = assets["stiefel"]["test750"]["q32"][idx_t].to(device).float()
    cg = assets["stiefel"]["test750"]["cnx_gray"][idx_t].to(device).float()
    y16 = project_latent_y16_lcn(z_src, v18["luma_proj"]["w"].to(device).float(), v18["luma_proj"]["b"].to(device).float())
    c64 = ts["c_64"][idx_t].to(device).float()
    tid = torch.tensor([t["target_style_id"] for t in tasks_sel], device=device, dtype=torch.long)
    u_st = match_cnx_k8_submode(cg, tid, assets["bank"])

    ecore, raw_stats, priors_ec, priors_a = load_v32_ecore_and_priors(device=device)
    p8 = read_json(P8_REPORT_PATH)
    fcfg = p8["final_schedule_cfg"]
    ckpt_p = Path(p8["locked_ckpt"]) if Path(p8["locked_ckpt"]).exists() else (REPO / "runs/v34_frontier/stage_c/C2_44m_q32/C2_44m_q32_final.pt")
    pcore, _ = load_pcore_v35(ckpt_p, weight_key=p8["locked_weight"], device=device)

    def _save_z(name: str, zt: torch.Tensor) -> None:
        rgb = decode_s10(dec, zt.to(device).float(), batch=16, output_cpu=True).clamp(0.0, 1.0)
        for j, tidx in enumerate(indices):
            Image.fromarray((rgb[j].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(img_root / name / f"{tidx:04d}.png")

    # 1. CLEAVE-E (l075)
    ze_l075 = []
    for s in range(0, len(indices), 16):
        sl = slice(s, min(len(indices), s + 16))
        ze_l075.append(ecore(z_src[sl], c64[sl], q16[sl], tid[sl], torch.full((sl.stop - sl.start,), 0.75, device=device), raw_stats, priors_ec, noise=eps[sl]))
    _save_z("cleave_e_l075", torch.cat(ze_l075, 0))

    # 2. CLEAVE-P/E (s085, s100)
    for sv, lbl in ((0.85, "cleave_pe_s085"), (1.00, "cleave_pe_s100")):
        kw = make_v35_schedule(sv, ws_mul=fcfg["ws_mul"], wc_mul=fcfg["wc_mul"], dtau0=fcfg["dtau0"])
        lam_d = float(kw.pop("draft_lambda", 0.75))
        outs = []
        for s in range(0, len(indices), 16):
            sl = slice(s, min(len(indices), s + 16))
            ze = ecore(z_src[sl], c64[sl], q16[sl], tid[sl], torch.full((sl.stop - sl.start,), lam_d, device=device), raw_stats, priors_ec, noise=eps[sl])
            ze_s = build_pcore_startpoint(ze, tid[sl], priors_a, alpha_y=fcfg["alpha_y"], alpha_c=fcfg["alpha_c"])
            outs.append(integrate_v33_trajectory(pcore, ze_s, eps[sl], y16[sl], q16[sl], tid[sl], u_st[sl], steps=int(fcfg["steps"]), solver="dpm2m", **kw))
        _save_z(lbl, torch.cat(outs, 0))

    del ecore, pcore
    gc.collect()
    torch.cuda.empty_cache()

    # 3. CLEAVE-One (t50)
    pcore_v36, style_v36, priors_v36 = load_v36_models(V36_P2_CKPT, device)
    e_refs = style_v36(ea["refs150"]["stat_1561"].to(device).float()).cpu()
    r_idx_all = torch.tensor([int(t["target_style_id"]) * 30 + (tidx % 30) for tidx, t in zip(indices, tasks_sel)], dtype=torch.long)
    outs_one = []
    for s in range(0, len(indices), 16):
        sl = slice(s, min(len(indices), s + 16))
        ri = r_idx_all[sl]
        zs = reconstruct_reference_a3(z_src[sl], z_refs[ri].to(device).float(), priors_v36, mode="I2")
        outs_one.append(sample_v36_ode(pcore_v36, zs, y16[sl], q16[sl], e_refs[ri].to(device), eps[sl], tau0=0.50, steps=6, wj=3.8, wc=1.6, device=device))
    _save_z("cleave_one_t50", torch.cat(outs_one, 0))

    del pcore_v36, style_v36, dec
    gc.collect()
    torch.cuda.empty_cache()

    man = {
        "utc": utc(),
        "indices": indices,
        "tasks": [{"idx": tidx, "task_id": t["task_id"], "src_style": t["source_style"], "tgt_style": t["target_style"]} for tidx, t in zip(indices, tasks_sel)],
    }
    save_json(manifest_p, man)
    print(f"[F2 prepare] {len(indices)} stratified cross tasks prepared at {img_root}", flush=True)
    return man


def _img_b64_uri(path: Path, resize_px: int = 384) -> str:
    with Image.open(path) as im:
        im_r = im.convert("RGB").resize((resize_px, resize_px), Image.Resampling.BICUBIC)
        buf = io.BytesIO()
        im_r.save(buf, format="JPEG", quality=92)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _find_cand_path(method: str, tidx: int) -> Path | None:
    for root in (F2_DIR / "images", EXT_DIR):
        for ext in (".png", ".jpg"):
            p = root / method / f"{tidx:04d}{ext}"
            if p.exists():
                return p
    return None


def _chat_vlm(server_url: str, prompt: str, img_uris: list[str], *, max_tokens: int = 200) -> str:
    content = [{"type": "image_url", "image_url": {"url": u}} for u in img_uris]
    content.append({"type": "text", "text": prompt})
    body = {
        "model": "qwen3.8-27b",
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(
        f"{server_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with opener.open(req, timeout=180) as resp:
        res = json.loads(resp.read().decode("utf-8"))
    return res["choices"][0]["message"]["content"]


def _parse_json_reply(txt: str) -> dict[str, Any] | None:
    cleaned = re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL).strip()
    m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def run_judge(server_url: str, *, max_tasks: int = 75, n_pairwise: int = 40, seed: int = 20261128) -> dict[str, Any]:
    man = read_json(F2_DIR / "f2_manifest.json")
    tasks = man["tasks"][:max_tasks]
    img_root = F2_DIR / "images"

    # Build shuffled blind single-candidate triplet queue (no method label exposed!)
    rng = random.Random(seed)
    queue: list[tuple[int, str, str, Path, Path, Path]] = []
    for t in tasks:
        tidx = int(t["idx"])
        src_p = img_root / "source" / f"{tidx:04d}.png"
        ref_p = img_root / "ref" / f"{tidx:04d}.png"
        for meth in METHODS_ORDER:
            cp = _find_cand_path(meth, tidx)
            if cp is not None:
                queue.append((tidx, t["tgt_style"], meth, src_p, ref_p, cp))
    rng.shuffle(queue)

    triplets_p = F2_DIR / "vlm_triplet_scores.jsonl"
    done_keys: set[tuple[int, str]] = set()
    rows: list[dict[str, Any]] = []
    if triplets_p.exists():
        with triplets_p.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done_keys.add((int(r["idx"]), r["method"]))
                rows.append(r)

    t0 = time.monotonic()
    with triplets_p.open("a", encoding="utf-8") as f_out:
        for k, (tidx, tgt_s, meth, src_p, ref_p, cp) in enumerate(queue):
            if (tidx, meth) in done_keys:
                continue
            reply = _chat_vlm(server_url, RUBRIC_PROMPT, [_img_b64_uri(src_p), _img_b64_uri(ref_p), _img_b64_uri(cp)])
            parsed = _parse_json_reply(reply)
            if parsed is None:
                continue
            rec = {
                "idx": tidx,
                "tgt_style": tgt_s,
                "method": meth,
                "scores": {
                    ax: int(parsed[ax])
                    for ax in ("content_preservation", "style_faithfulness", "artistic_repainting", "artifact_cleanliness", "no_layout_leakage")
                    if ax in parsed
                },
                "note": str(parsed.get("brief_note", "")),
            }
            if len(rec["scores"]) == 5:
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f_out.flush()
                rows.append(rec)
            if (k + 1) % 25 == 0:
                dt = time.monotonic() - t0
                print(f"[F2 judge] {k + 1}/{len(queue)} calls elapsed={dt:.1f}s ({len(rows)} valid)", flush=True)

    # Head-to-head position-swapped pairwise comparisons vs CLEAVE-P/E (s085)
    pair_specs = [
        ("cleave_pe_s085", "cleave_e_l075"),
        ("cleave_pe_s085", "cleave_one_t50"),
        ("cleave_pe_s085", "stylegallery"),
        ("cleave_pe_s085", "stytr2"),
        ("cleave_pe_s085", "aespa"),
        ("cleave_pe_s085", "seedream"),
    ]
    pairs_p = F2_DIR / "vlm_pairwise.jsonl"
    done_pairs: set[tuple[int, str, str]] = set()
    pair_rows: list[dict[str, Any]] = []
    if pairs_p.exists():
        with pairs_p.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done_pairs.add((int(r["idx"]), r["m_a"], r["m_b"]))
                pair_rows.append(r)

    with pairs_p.open("a", encoding="utf-8") as f_pw:
        for t in tasks[:n_pairwise]:
            tidx = int(t["idx"])
            src_u = _img_b64_uri(img_root / "source" / f"{tidx:04d}.png")
            ref_u = _img_b64_uri(img_root / "ref" / f"{tidx:04d}.png")
            for m_a, m_b in pair_specs:
                if (tidx, m_a, m_b) in done_pairs:
                    continue
                pa, pb = _find_cand_path(m_a, tidx), _find_cand_path(m_b, tidx)
                if pa is None or pb is None:
                    continue
                swap = (rng.random() < 0.5)
                x_m, y_m = (m_b, m_a) if swap else (m_a, m_b)
                x_p, y_p = (pb, pa) if swap else (pa, pb)
                reply = _chat_vlm(server_url, PAIRWISE_PROMPT, [src_u, ref_u, _img_b64_uri(x_p), _img_b64_uri(y_p)], max_tokens=160)
                parsed = _parse_json_reply(reply)
                if not parsed:
                    continue
                w_raw = str(parsed.get("winner", "TIE")).upper()
                winner = x_m if w_raw == "X" else (y_m if w_raw == "Y" else "TIE")
                rec = {"idx": tidx, "m_a": m_a, "m_b": m_b, "swapped": swap, "winner": winner, "raw": parsed}
                f_pw.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f_pw.flush()
                pair_rows.append(rec)

    # Aggregate summary with 95% bootstrap CI
    axes = ("content_preservation", "style_faithfulness", "artistic_repainting", "artifact_cleanliness", "no_layout_leakage")
    by_m: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_m.setdefault(r["method"], []).append(r)

    rng_b = np.random.default_rng(20261128)
    method_stats: dict[str, Any] = {}
    for m, mrows in by_m.items():
        ax_out: dict[str, Any] = {}
        for ax in axes:
            vals = np.array([r["scores"][ax] for r in mrows], dtype=np.float64)
            boots = vals[rng_b.integers(0, len(vals), size=(1000, len(vals)))].mean(axis=1)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            ax_out[ax] = {"mean": round(float(vals.mean()), 3), "ci95_lo": round(float(lo), 3), "ci95_hi": round(float(hi), 3)}
        method_stats[m] = {"n": len(mrows), "axes": ax_out}

    pw_stats: dict[str, Any] = {}
    for m_a, m_b in pair_specs:
        sub = [r for r in pair_rows if r["m_a"] == m_a and r["m_b"] == m_b]
        if not sub:
            continue
        wa = sum(1 for r in sub if r["winner"] == m_a)
        wb = sum(1 for r in sub if r["winner"] == m_b)
        wt = sum(1 for r in sub if r["winner"] == "TIE")
        pw_stats[f"{m_a}_vs_{m_b}"] = {"n": len(sub), "win_a": wa, "win_b": wb, "tie": wt, "win_rate_a_excl_tie": round(wa / max(1, wa + wb), 4)}

    summary = {"utc": utc(), "model": "Qwen/Qwen3.8-27B-UD-Q4_K_S", "methods": method_stats, "pairwise": pw_stats}
    save_json(F2_DIR / "f2_summary.json", summary)
    print(f"[F2 DONE] saved {F2_DIR / 'f2_summary.json'}", flush=True)
    return summary


def main() -> None:
    sanitize_cuda_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["prepare", "judge"])
    ap.add_argument("--n-per-target", type=int, default=15)
    ap.add_argument("--max-tasks", type=int, default=75)
    ap.add_argument("--n-pairwise", type=int, default=40)
    ap.add_argument("--server-url", default="http://127.0.0.1:8089")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.stage == "prepare":
        prepare_images(a.n_per_target, a.device)
    else:
        run_judge(a.server_url, max_tasks=a.max_tasks, n_pairwise=a.n_pairwise)


if __name__ == "__main__":
    main()
