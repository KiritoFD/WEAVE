"""CLEAVE V35 Plan Stages P0 (Freeze & Integrity Audit), P1 (E-Core 7-Arm Causal Ablation), and P2 (D0/D1/D2 Decoder Ablation).

Zero-DINO AST hygiene strictly enforced.
"""
from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from exp.v18 import tasks_and_sources
from exp.v20.slf_flow import project_latent_y16_lcn
from exp.v20.submode_patchswd import load_convnext_tiny_offline
from exp.v27_single.bare_q32 import load_cache
from exp.v32_nodino.build_cnx_assets import build_or_load_cnx_assets
from exp.v32_nodino.cnx_layout import match_cnx_k8_submode
from exp.v33_semantic_control.models import integrate_v33_trajectory
from exp.v35_final import (
    C2_V34_CKPT,
    D0_FULL_VAE_DIR,
    D1_FULL_FT_CKPT,
    OUT,
    S10B_CKPT,
    STYLES,
    RawWriterV35,
    build_pcore_startpoint,
    measure_and_summarize_batch_v35,
    paired_ci_vs_baseline,
)
from exp.v35_final.common import (
    eval_pcore_test750,
    load_pcore_v35,
    load_v32_ecore_and_priors,
    make_v35_schedule,
)
from src.eval.holdout import build_or_load_holdout_spec
from src.eval.metrics import paired_cluster_ci
from src.utils import auto_decode_batch, auto_infer_batch, decode_s10, load_s10_decoder
from src.utils.protocol import EVAL_SEED, canonical_pid, pick_wrong_cross_tid
from src.utils.runtime import read_json, sanitize_cuda_env, save_json, utc
from src.utils.structure import dwt2_haar, rgb_to_gray


def _artist_key(stem: str) -> str:
    """Extract normalized artist identifier from WikiArt stem `<Style>__<artist>_<work>`."""
    body = stem.split("__", 1)[-1]
    return body.split("_", 1)[0].lower()


def _extra_cross_rates(m_dir: Path, *, alt_target_tid: torch.Tensor | None = None) -> dict[str, float]:
    """Read pred_eq_src (Source-Lock Rate) and optional draft-style follow rate from eval.json."""
    ev = read_json(m_dir / "eval.json")
    rows = [r for r in ev["rows"] if r["is_cross"]]
    n = max(1, len(rows))
    pred_src = sum(int(int(r["pred_class"]) == int(r["src_style_id"])) for r in rows) / n
    out = {"pred_eq_src": round(pred_src, 4)}
    if alt_target_tid is not None:
        all_rows = ev["rows"]
        alt_hits = [
            int(int(r["pred_class"]) == int(alt_target_tid[i].item()))
            for i, r in enumerate(all_rows)
            if r["is_cross"]
        ]
        out["pred_eq_wrong_draft"] = round(sum(alt_hits) / max(1, len(alt_hits)), 4)
    return out


# ---------------------------------------------------------------------------
# P0: Freeze Final Baseline & Integrity Audit
# ---------------------------------------------------------------------------

def run_p0(*, device: str = "cuda") -> dict[str, Any]:
    rep_p = OUT / "p0_freeze_audit.json"
    if rep_p.exists():
        print(f"[Stage P0] Reusing completed {rep_p}", flush=True)
        return read_json(rep_p)

    t0 = time.monotonic()
    # 1. Inspect C2 checkpoint schema (optimizer / EMA completeness)
    sv_c2 = torch.load(C2_V34_CKPT, map_location="cpu", weights_only=False)
    ckpt_audit = {
        "c2_ckpt": str(C2_V34_CKPT),
        "keys": list(sv_c2.keys()),
        "step": int(sv_c2.get("step", -1)),
        "has_ema": "ema" in sv_c2,
        "has_online": "online" in sv_c2,
        "has_optimizer": "opt" in sv_c2,
        "note": "C2_44m_q32_final.pt saves EMA weights only; P5 continuation re-warms AdamW moments for 100 steps.",
        "cfg": sv_c2.get("cfg", {}),
    }

    # 2. Verify V34 s150 vs tau0=0 configuration difference
    sched_s150 = make_v35_schedule(1.50)
    sched_tau0 = {"tau0": 0.0, "ws": 4.5, "wc": 1.8, "wsp": 1.35, "rescale_phi": 0.40, "gamma": 0.35, "lscale": 0.72, "draft_lambda": 0.75}

    # 3. Verify wrong-target counterfactual alters generation conditions cleanly
    tasks, _, _ = tasks_and_sources()
    sid_750 = torch.tensor([t["source_style_id"] for t in tasks], dtype=torch.long)
    tid_750 = torch.tensor([t["target_style_id"] for t in tasks], dtype=torch.long)
    tid_w = pick_wrong_cross_tid(sid_750, tid_750)
    wrong_audit = {
        "n_tasks": len(tasks),
        "all_wrong_differ_from_src": bool((tid_w != sid_750).all().item()),
        "all_wrong_differ_from_tgt": bool((tid_w != tid_750).all().item()),
    }

    # 4. Audit Holdout source/ref/artist vs train_4400 & test750 isolation
    v18 = load_cache("runs/v18/cache/ccf_pool_cache.pt")
    ho_spec = build_or_load_holdout_spec(device=device)
    tr_ids = [str(x) for x in v18["train"]["ids"]]
    ho_ids = [str(x) for x in v18["holdout"]["ids"]]
    tr_pids = {canonical_pid(x) for x in tr_ids}
    ho_pids = {canonical_pid(x) for x in ho_ids}
    ts_pids = {canonical_pid(t["source_stem"]) for t in tasks}
    ho_src_idx = set(ho_spec["src_idx"].tolist())
    ho_ref_idx = set(ho_spec["ref_idx"].tolist())

    tr_artists = {_artist_key(x) for x in tr_ids}
    ho_artists = {_artist_key(x) for x in ho_ids}
    ho_src_stems = {str(ho_ids[i]) for i in ho_src_idx}
    ho_ref_stems = {str(ho_ids[i]) for i in ho_ref_idx}

    holdout_audit = {
        "n_train": len(tr_ids),
        "n_holdout_total": len(ho_ids),
        "n_holdout_src": len(ho_src_idx),
        "n_holdout_ref": len(ho_ref_idx),
        "pid_overlap_ho_train": len(ho_pids & tr_pids),
        "pid_overlap_ho_test750": len(ho_pids & ts_pids),
        "src_ref_idx_overlap": len(ho_src_idx & ho_ref_idx),
        "artist_overlap_ho_train": len(ho_artists & tr_artists),
        "artist_overlap_ho_src_ref": len({_artist_key(s) for s in ho_src_stems} & {_artist_key(s) for s in ho_ref_stems}),
        "note": "Zero painting-level overlap across train_4400 / test750 / holdout-src / holdout-ref; holdout used previously in V34 verification so P6 calibration strictly uses train_calib_300 instead.",
    }

    # 5. Baseline reproduction on 44M C2 + A3 across s085/s100/s150 + tau0=0 (+ wrong)
    pcore, _ = load_pcore_v35(C2_V34_CKPT, device=device)
    levels = eval_pcore_test750(
        "P0_BaseA3", pcore,
        knob_list=(0.85, 1.00, 1.50),
        include_tau0=True, eval_wrong=True,
        alpha_y=1.0, alpha_c=1.0,
        device=device,
    )
    del pcore
    torch.cuda.empty_cache()

    for k in list(levels.keys()):
        levels[k].update(_extra_cross_rates(OUT / "measured" / f"P0_BaseA3_{k}"))

    rep = {
        "updated_utc": utc(),
        "wall_sec": round(time.monotonic() - t0, 2),
        "stage": "P0_freeze_and_audit",
        "ckpt_audit": ckpt_audit,
        "schedule_diff": {"s150": sched_s150, "tau0_pure_noise": sched_tau0},
        "wrong_target_audit": wrong_audit,
        "holdout_isolation_audit": holdout_audit,
        "repro_levels": levels,
    }
    save_json(rep_p, rep)
    print(f"[Stage P0 DONE] holdout_pid_overlap={holdout_audit['pid_overlap_ho_train']} s150_tau0={sched_s150['tau0']} vs tau0=0.0 | wall={rep['wall_sec']}s", flush=True)
    return rep


# ---------------------------------------------------------------------------
# P1: E-Core 7-Arm Causal Ablation (Zero-Training, Freeze 44M P-Core)
# ---------------------------------------------------------------------------

P1_ARM_SPECS: dict[str, dict[str, Any]] = {
    "P1_A0_src":          {"startpoint_mode": "raw_src",        "wrong_draft": False, "wrong_pcore": False, "label": "Source z_src (no E-Core)"},
    "P1_A1_src_chroma":   {"startpoint_mode": "raw_src_chroma", "wrong_draft": False, "wrong_pcore": False, "label": "Analytic Chroma-Aligned z_src"},
    "P1_A2_ecore":        {"startpoint_mode": "ecore",          "wrong_draft": False, "wrong_pcore": False, "label": "E-Core Draft (A0)"},
    "P1_A3_ecore_a3":     {"startpoint_mode": "ecore_a3",       "wrong_draft": False, "wrong_pcore": False, "label": "E-Core Draft + A3"},
    "P1_A4_td_wp":        {"startpoint_mode": "ecore_a3",       "wrong_draft": False, "wrong_pcore": True,  "label": "Correct Draft+A3 + Wrong P-Core"},
    "P1_A5_wd_tp":        {"startpoint_mode": "ecore_a3",       "wrong_draft": True,  "wrong_pcore": False, "label": "Wrong Draft+A3 + Correct P-Core"},
    "P1_A6_wd_wp":        {"startpoint_mode": "ecore_a3",       "wrong_draft": True,  "wrong_pcore": True,  "label": "Wrong Draft+A3 + Wrong P-Core"},
}


def run_p1(*, device: str = "cuda") -> dict[str, Any]:
    rep_p = OUT / "p1_ecore_report.json"
    if rep_p.exists():
        print(f"[Stage P1] Reusing completed {rep_p}", flush=True)
        return read_json(rep_p)

    t0 = time.monotonic()
    tasks, _, _ = tasks_and_sources()
    sid_750 = torch.tensor([t["source_style_id"] for t in tasks], dtype=torch.long)
    tid_750 = torch.tensor([t["target_style_id"] for t in tasks], dtype=torch.long)
    tid_wrong = pick_wrong_cross_tid(sid_750, tid_750)

    pcore, _ = load_pcore_v35(C2_V34_CKPT, device=device)
    shared_drafts: dict[tuple[float, bool], torch.Tensor] = {}
    results: dict[str, dict[str, Any]] = {}

    for arm_tag, cfg in P1_ARM_SPECS.items():
        lv = eval_pcore_test750(
            arm_tag, pcore,
            knob_list=(0.85, 1.00),
            eval_wrong=False,
            alpha_y=1.0, alpha_c=1.0,
            startpoint_mode=cfg["startpoint_mode"],
            wrong_draft=cfg["wrong_draft"],
            wrong_pcore=cfg["wrong_pcore"],
            shared_drafts=shared_drafts,
            device=device,
        )
        for k in ("s085", "s100"):
            lv[k].update(_extra_cross_rates(OUT / "measured" / f"{arm_tag}_{k}", alt_target_tid=tid_wrong))
        results[arm_tag] = {"label": cfg["label"], **lv}

    del pcore
    torch.cuda.empty_cache()

    def _delta(a: str, b: str, k: str) -> dict[str, float]:
        ma, mb = results[a][k], results[b][k]
        return {
            "d_dino_c": round(ma["dino_c"] - mb["dino_c"], 4),
            "d_dino_s": round(ma["dino_s"] - mb["dino_s"], 4),
            "d_top1_pp": round((ma["csd_top1"] - mb["csd_top1"]) * 100.0, 2),
            "d_src_lock_pp": round((ma["pred_eq_src"] - mb["pred_eq_src"]) * 100.0, 2),
        }

    attribution: dict[str, Any] = {}
    cis: dict[str, Any] = {}
    for k in ("s085", "s100"):
        attribution[k] = {
            "A2_minus_A1_ecore_beyond_chroma": _delta("P1_A2_ecore", "P1_A1_src_chroma", k),
            "A3_minus_A2_analytic_a3": _delta("P1_A3_ecore_a3", "P1_A2_ecore", k),
            "A1_minus_A0_raw_chroma": _delta("P1_A1_src_chroma", "P1_A0_src", k),
            "2x2_pcore_effect_on_target_draft_A3_minus_A4": _delta("P1_A3_ecore_a3", "P1_A4_td_wp", k),
            "2x2_draft_effect_with_wrong_pcore_A4_minus_A6": _delta("P1_A4_td_wp", "P1_A6_wd_wp", k),
            "2x2_draft_effect_with_target_pcore_A3_minus_A5": _delta("P1_A3_ecore_a3", "P1_A5_wd_tp", k),
        }
        cis[f"A2_vs_A1_{k}"] = paired_ci_vs_baseline(OUT / f"measured/P1_A2_ecore_{k}/eval.json", OUT / f"measured/P1_A1_src_chroma_{k}/eval.json")
        cis[f"A3_vs_A2_{k}"] = paired_ci_vs_baseline(OUT / f"measured/P1_A3_ecore_a3_{k}/eval.json", OUT / f"measured/P1_A2_ecore_{k}/eval.json")

    rep = {
        "updated_utc": utc(),
        "wall_sec": round(time.monotonic() - t0, 2),
        "stage": "P1_ecore_causal_ablation",
        "arms": results,
        "attribution": attribution,
        "paired_ci": cis,
    }
    save_json(rep_p, rep)
    print(f"[Stage P1 DONE] s085 A2-A1 dTop1={attribution['s085']['A2_minus_A1_ecore_beyond_chroma']['d_top1_pp']:+.2f}pp | A3-A2 dS={attribution['s085']['A3_minus_A2_analytic_a3']['d_dino_s']:+.4f}", flush=True)
    return rep


# ---------------------------------------------------------------------------
# P2: Decoder Final Contribution (D0 Full Frozen / D1 WikiArt FT / D2 S10B)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _generate_cached_latents(device: str = "cuda") -> dict[str, Path]:
    """Cache exact 750 final latents for E-Core (s075, s100) and P-Core C2+A3 (s085, s150)."""
    lat_dir = OUT / "p2/cached_latents"
    lat_dir.mkdir(parents=True, exist_ok=True)
    keys = ("ecore_s075", "ecore_s100", "pcore_s085", "pcore_s150")
    out_paths = {k: lat_dir / f"{k}.pt" for k in keys}
    if all(p.exists() for p in out_paths.values()):
        return out_paths

    tasks, z_src, _ = tasks_and_sources()
    tid_750 = torch.tensor([t["target_style_id"] for t in tasks], dtype=torch.long)
    eps_750 = torch.randn(z_src.shape, generator=torch.Generator().manual_seed(EVAL_SEED))
    v18 = load_cache("runs/v18/cache/ccf_pool_cache.pt")
    ts = load_cache("runs/v18/cache/test750_struct_cache.pt")
    assets = build_or_load_cnx_assets(device=device)
    stiefel, cnx_bank = assets["stiefel"], assets["bank"]
    q16_750 = stiefel["test750"]["q32"].float()
    cnx_gray_750 = stiefel["test750"]["cnx_gray"].float()
    y16_750 = project_latent_y16_lcn(z_src.float(), v18["luma_proj"]["w"].float(), v18["luma_proj"]["b"].float())
    c64_750 = ts["c_64"].float()

    ecore, raw_stats, priors_ec, priors_a = load_v32_ecore_and_priors(device=device)
    pcore, _ = load_pcore_v35(C2_V34_CKPT, device=device)
    infer_bs = auto_infer_batch(device)

    def _run_ecore(lam: float) -> torch.Tensor:
        chunks = []
        for st in range(0, len(tasks), infer_bs):
            sl = slice(st, min(st + infer_bs, len(tasks)))
            ze = ecore(
                z_src[sl].to(device).float(), c64_750[sl].to(device).float(), q16_750[sl].to(device).float(),
                tid_750[sl].to(device), torch.full((sl.stop - sl.start,), lam, device=device),
                raw_stats, priors_ec, noise=eps_750[sl].to(device).float(),
            )
            chunks.append(ze.cpu().half())
        return torch.cat(chunks, 0)

    def _run_pcore(s_val: float) -> torch.Tensor:
        kw = make_v35_schedule(s_val)
        lam = float(kw.pop("draft_lambda", 0.75))
        ze_all = _run_ecore(lam)
        chunks = []
        for st in range(0, len(tasks), infer_bs):
            sl = slice(st, min(st + infer_bs, len(tasks)))
            ti = tid_750[sl].to(device)
            u_st = match_cnx_k8_submode(cnx_gray_750[sl].to(device).float(), ti, cnx_bank)
            ze_start = build_pcore_startpoint(ze_all[sl].to(device).float(), ti, priors_a, alpha_y=1.0, alpha_c=1.0)
            zo = integrate_v33_trajectory(
                pcore, ze_start, eps_750[sl].to(device).float(),
                y16_750[sl].to(device).float(), q16_750[sl].to(device).float(),
                ti, u_st, steps=6, solver="dpm2m", **kw,
            )
            chunks.append(zo.cpu().half())
        return torch.cat(chunks, 0)

    ids = [t["task_id"] for t in tasks]
    saves = {
        "ecore_s075": _run_ecore(0.65),
        "ecore_s100": _run_ecore(0.90),
        "pcore_s085": _run_pcore(0.85),
        "pcore_s150": _run_pcore(1.50),
    }
    for k, zt in saves.items():
        torch.save({"ids": ids, "z": zt}, out_paths[k])

    del ecore, pcore
    gc.collect()
    torch.cuda.empty_cache()
    return out_paths


@torch.no_grad()
def _measure_texture_stats(rgb01: torch.Tensor, cnx_s1: torch.nn.Module, cnx_s2: torch.nn.Module, im_mean: torch.Tensor, im_std: torch.Tensor) -> dict[str, float]:
    """Compute Haar 2D band energy and ConvNeXt Stage-1/2 spatial texture std on a batch of RGB images."""
    g = rgb_to_gray(rgb01.float())
    ll, lh, hl, hh = dwt2_haar(g)
    xn = (rgb01.float() - im_mean) / im_std
    f1 = cnx_s1(xn)
    f2 = cnx_s2(f1)
    return {
        "haar_ll": float(ll.abs().mean().item()),
        "haar_lh": float(lh.abs().mean().item()),
        "haar_hl": float(hl.abs().mean().item()),
        "haar_hh": float(hh.abs().mean().item()),
        "cnx_s1_std": float(f1.std(dim=(2, 3), unbiased=False).mean().item()),
        "cnx_s2_std": float(f2.std(dim=(2, 3), unbiased=False).mean().item()),
    }


def _load_autoencoder_decoder(ckpt_override: Path | None, device: str):
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(str(D0_FULL_VAE_DIR), local_files_only=True).to(device).eval()
    if ckpt_override is not None:
        vae.load_state_dict(torch.load(ckpt_override, map_location=device, weights_only=False))
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


def run_p2(*, device: str = "cuda") -> dict[str, Any]:
    rep_p = OUT / "p2_decoder_report.json"
    if rep_p.exists():
        print(f"[Stage P2] Reusing completed {rep_p}", flush=True)
        return read_json(rep_p)

    sanitize_cuda_env()
    t0 = time.monotonic()
    lat_paths = _generate_cached_latents(device=device)
    tasks, _, _ = tasks_and_sources()

    # Load ConvNeXt stages once for Haar/texture retraction metrics
    cnx = load_convnext_tiny_offline(device)
    cnx_s1 = cnx[:2].eval().requires_grad_(False)
    cnx_s2 = cnx[2:4].eval().requires_grad_(False)
    im_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    im_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    dec_specs = [
        ("D0_sdvae", "full", None, D0_FULL_VAE_DIR.exists()),
        ("D1_wikiart_ft", "full", D1_FULL_FT_CKPT, D1_FULL_FT_CKPT.exists()),
        ("D2_s10b", "s10", S10B_CKPT, S10B_CKPT.exists()),
    ]
    dec_params: dict[str, int] = {}
    dec_ms: dict[str, float] = {}
    tex_stats: dict[str, dict[str, float]] = {}
    arms_to_measure: list[str] = []
    arm_meta: list[tuple[str, str, str]] = []

    for d_tag, kind, ckpt_p, available in dec_specs:
        if not available:
            print(f"[Stage P2 WARNING] Missing decoder asset for {d_tag}: {ckpt_p}", flush=True)
            continue
        if kind == "full":
            dec_obj = _load_autoencoder_decoder(ckpt_p, device=device)
            dec_params[d_tag] = sum(p.numel() for n, p in dec_obj.named_parameters() if n.startswith(("decoder", "post_quant_conv")))
            dec_bs = 16
            def _dec_fn(z: torch.Tensor, _dec=dec_obj, _bs=dec_bs, _dev=device) -> torch.Tensor:
                outs = []
                for i in range(0, len(z), _bs):
                    out = _dec.decode(z[i:i + _bs].float().to(_dev) / 0.18215).sample
                    outs.append((out / 2.0 + 0.5).clamp(0.0, 1.0))
                return torch.cat(outs, 0)
        else:
            dec_obj = load_s10_decoder(device)
            dec_params[d_tag] = sum(p.numel() for p in dec_obj.parameters())
            bs_d = auto_decode_batch(device)
            def _dec_fn(z: torch.Tensor, _dec=dec_obj, _bs=bs_d, _dev=device) -> torch.Tensor:
                return decode_s10(_dec, z.to(_dev).float(), batch=_bs, output_cpu=False).clamp(0.0, 1.0)

        for l_tag, lp in lat_paths.items():
            arm = f"P2_{l_tag}_{d_tag}"
            arms_to_measure.append(arm)
            arm_meta.append((l_tag, d_tag, arm))
            payload = torch.load(lp, map_location="cpu", weights_only=False)
            z_all = payload["z"]
            if not (OUT / "measured" / arm / "summary_v34.json").exists():
                writer = RawWriterV35(arm, dec=None, device=device)
                accum_t: dict[str, float] = {}
                n_chunks = 0
                torch.cuda.synchronize()
                td_0 = time.perf_counter()
                chunk_bs = 64
                for st in range(0, len(tasks), chunk_bs):
                    sl = slice(st, min(st + chunk_bs, len(tasks)))
                    rgb = _dec_fn(z_all[sl])
                    writer.write_batch(z_all[sl], tasks[sl], rgb_override=rgb)
                    ts_batch = _measure_texture_stats(rgb, cnx_s1, cnx_s2, im_mean, im_std)
                    for tk, tv in ts_batch.items():
                        accum_t[tk] = accum_t.get(tk, 0.0) + tv
                    n_chunks += 1
                torch.cuda.synchronize()
                if l_tag == "pcore_s085":
                    dec_ms[d_tag] = round((time.perf_counter() - td_0) * 1000.0 / len(tasks), 3)
                tex_stats[arm] = {tk: round(tv / max(1, n_chunks), 6) for tk, tv in accum_t.items()}
                writer.finalize(arm=arm, decoder=d_tag, latent_source=l_tag)

        del dec_obj
        gc.collect()
        torch.cuda.empty_cache()

    del cnx, cnx_s1, cnx_s2
    torch.cuda.empty_cache()

    tex_p = OUT / "p2/texture_stats.json"
    if tex_stats:
        prev = read_json(tex_p) if tex_p.exists() else {}
        prev.update(tex_stats)
        save_json(tex_p, prev)
    saved_tex = read_json(tex_p) if tex_p.exists() else {}

    # Batch referee evaluation for all P2 decoded outputs
    measured = measure_and_summarize_batch_v35(arms_to_measure, device=device)

    for _, _, arm in arm_meta:
        m_dir = OUT / "measured" / arm
        measured[arm].update(_extra_cross_rates(m_dir))
        ev_sum = read_json(m_dir / "summary_v34.json")["cross600"]
        measured[arm]["csd_margin"] = round(float(ev_sum["csd_margin"]), 5)
        if arm in saved_tex:
            measured[arm]["texture"] = saved_tex[arm]

    table: dict[str, dict[str, Any]] = {}
    paired_cis: dict[str, Any] = {}
    for l_tag in lat_paths:
        table[l_tag] = {}
        for d_tag, _, _, avail in dec_specs:
            if avail:
                arm = f"P2_{l_tag}_{d_tag}"
                table[l_tag][d_tag] = measured[arm]
        if all(avail for _, _, _, avail in dec_specs):
            r_d2 = read_json(OUT / f"measured/P2_{l_tag}_D2_s10b/eval.json")["rows"]
            for base_d in ("D0_sdvae", "D1_wikiart_ft"):
                r_b = read_json(OUT / f"measured/P2_{l_tag}_{base_d}/eval.json")["rows"]
                paired_cis[f"{l_tag}_D2_minus_{base_d}"] = paired_cluster_ci(r_d2, r_b)

    rep = {
        "updated_utc": utc(),
        "wall_sec": round(time.monotonic() - t0, 2),
        "stage": "P2_decoder_ablation",
        "decoders_available": {d: bool(av) for d, _, _, av in dec_specs},
        "decoder_params": dec_params,
        "decode_ms_per_img_4090": dec_ms,
        "table": table,
        "paired_ci": paired_cis,
    }
    save_json(rep_p, rep)
    print(f"[Stage P2 DONE] params={dec_params} decode_ms={dec_ms} in {rep['wall_sec']}s", flush=True)
    return rep
