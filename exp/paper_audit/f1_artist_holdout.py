"""F1: True Artist-Holdout-750 (Zero Artist Overlap with V35 train_4400, V36 Seen-Train, test750, and references_v1).

Constructs from V36 curation_manifest['artist_ood'] (already disjoint from all 76,319 V36 Seen-Train paintings)
after filtering out any overlap with V35 train_4400, test750, and references_v1, then runs in clean subprocesses:
  - CLEAVE-E (1-NFE closed-set E-Core, lam in {0.50, 0.75, 1.00})
  - CLEAVE-P/E (closed-set V35 P8 Final, s in {0.60, 0.85, 1.00, 1.50} + s=1.00 wrong target)
  - CLEAVE-One (open-ref V36 P2 50k + Reference-A3 I2, tau0 in {0.35, 0.50, 0.65} + tau0=0.50 wrong target)
Also evaluates available unseen-artist Core-5 paintings (Impressionism/Ukiyo_e/Rococo/Early_Renaissance) for
direct Seen-Movement Unseen-Artist Same vs Cross verification.
Zero training / zero new learnable parameters.
"""
from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from exp.v18 import tasks_and_sources
from exp.v20.slf_flow import project_latent_y16_lcn
from exp.v25.r14_e2e_benchmark import load_a0_encoder
from exp.v27_single.bare_q32 import load_cache
from exp.v32_nodino.build_cnx_assets import build_or_load_cnx_assets
from exp.v32_nodino.cnx_layout import ConvNeXtTinyStage3Extractor, extract_cnx_q32, match_cnx_k8_submode
from exp.v33_semantic_control.models import integrate_v33_trajectory
from exp.v35_cleave_one.style_features import ConvNeXtExplicitAndPatchBackbone
from exp.v35_final import C2_V34_CKPT, build_pcore_startpoint, summarize_matched_c
from exp.v35_final.common import load_pcore_v35, load_v32_ecore_and_priors, make_v35_schedule
from exp.v35_final.p0_p2_evals import _artist_key
from exp.v36_art_foundation import CACHE_DIR as V36_CACHE_DIR
from exp.v36_art_foundation.eval_v36 import load_v36_models, reconstruct_reference_a3, sample_v36_ode
from src.eval.metrics import paired_cluster_ci
from src.utils import FeatureStack, auto_decode_batch, auto_infer_batch, auto_referee_batch, decode_s10, load_s10_decoder
from src.utils.protocol import EVAL_SEED, STYLES, canonical_pid, pick_wrong_cross_tid
from src.utils.raw_io import RawWriter
from src.utils.runtime import REPO, read_json, sanitize_cuda_env, save_json, utc
from src.utils.structure import rgb_to_structure_64


F1_DIR = REPO / "runs/paper_audit/f1_artist_holdout"
SPEC_PATH = F1_DIR / "artist_holdout_750_spec.pt"
V36_P2_CKPT = REPO / "runs/v36_art_foundation/checkpoints/v36_main_p2_final.pt"
P8_REPORT_PATH = REPO / "runs/v35_final/p8_final_report.json"


def _norm_artist(name: str) -> str:
    return " ".join(str(name).strip().lower().replace("-", " ").replace("_", " ").split())


def _norm_slug(name: str) -> str:
    return _norm_artist(name).replace(" ", "-")


@torch.no_grad()
def build_or_load_artist_holdout_spec(*, device: str = "cuda") -> dict[str, Any]:
    if SPEC_PATH.exists():
        return torch.load(SPEC_PATH, map_location="cpu", weights_only=False)
    F1_DIR.mkdir(parents=True, exist_ok=True)

    v18 = load_cache("runs/v18/cache/ccf_pool_cache.pt")
    tr4400_ids = [str(x) for x in v18["train"]["ids"]]
    tr4400_arts = {_norm_slug(_artist_key(x)) for x in tr4400_ids}
    tr4400_pids = {canonical_pid(x) for x in tr4400_ids}

    cm = read_json(V36_CACHE_DIR / "curation_manifest.json")
    v36_train_arts = {_norm_slug(str(rec.get("artist", ""))) for rec in cm.get("seen_train", []) if rec.get("artist")}
    if not v36_train_arts:
        raise RuntimeError("curation_manifest.json missing seen_train artists")

    tasks750, _, _ = tasks_and_sources()
    refs_json = read_json(REPO / "eval_protocol/references_v1.json")["references"]
    banned_arts = set(tr4400_arts) | set(v36_train_arts)
    banned_arts |= {_norm_slug(_artist_key(t["source_stem"])) for t in tasks750}
    banned_arts |= {_norm_slug(_artist_key(r["stem"])) for rows in refs_json.values() for r in rows}
    banned_pids = set(tr4400_pids) | {canonical_pid(t["source_stem"]) for t in tasks750}
    banned_pids |= {canonical_pid(r["stem"]) for rows in refs_json.values() for r in rows}

    # Filter artist_ood for 0 artist & pid overlap, high-res existing files
    by_artist: dict[str, list[dict[str, Any]]] = {}
    for rec in cm["artist_ood"]:
        art_slug = _norm_slug(rec.get("artist", ""))
        if not art_slug or art_slug in banned_arts:
            continue
        p = Path(rec["path"])
        if not p.exists():
            continue
        pid = canonical_pid(p.stem)
        if pid in banned_pids:
            continue
        by_artist.setdefault(art_slug, []).append({**rec, "art_slug": art_slug})

    # Select 150 sources balanced round-robin across all eligible unseen artists
    rng = np.random.default_rng(20261128)
    artists_sorted = sorted(by_artist.keys())
    for a in artists_sorted:
        idxs = rng.permutation(len(by_artist[a])).tolist()
        by_artist[a] = [by_artist[a][i] for i in idxs]

    selected_src: list[dict[str, Any]] = []
    ptr = 0
    while len(selected_src) < 150:
        added = False
        for a in artists_sorted:
            if ptr < len(by_artist[a]):
                selected_src.append(by_artist[a][ptr])
                added = True
                if len(selected_src) == 150:
                    break
        if not added:
            break
        ptr += 1

    if len(selected_src) != 150:
        raise RuntimeError(f"Only found {len(selected_src)} eligible unseen-artist sources")

    sel_arts = {r["art_slug"] for r in selected_src}
    overlap_tr4400 = sel_arts & tr4400_arts
    overlap_v36 = sel_arts & v36_train_arts
    if overlap_tr4400 or overlap_v36:
        raise RuntimeError(f"Artist leakage detected! tr4400={overlap_tr4400} v36={overlap_v36}")

    # Load 150 RGBs (512x512 bicubic), encode A0 + C64 + Y16 + Q32 + FeatureStack
    rgb_list = []
    for rec in selected_src:
        with Image.open(rec["path"]) as im:
            arr = np.asarray(im.convert("RGB").resize((512, 512), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
        rgb_list.append(torch.from_numpy(arr).permute(2, 0, 1))
    rgb_src = torch.stack(rgb_list, 0)
    src_u8 = (rgb_src * 255.0).round().clamp(0, 255).to(torch.uint8)

    a0 = load_a0_encoder(device).float()
    cnx_ext = ConvNeXtTinyStage3Extractor(device=device).eval()
    st_ck = torch.load(REPO / "runs/v32_nodino/cache/cnx_stiefel_cache.pt", map_location="cpu", weights_only=False)
    w_stiefel = st_ck["W_stiefel_cnx"].to(device).float()
    w_luma = v18["luma_proj"]["w"].to(device).float()
    b_luma = v18["luma_proj"]["b"].to(device).float()

    from exp.v21.stiefel_ref_cache import stiefel_polar_quotient
    z_list, c64_list, y16_list, q32_list, cg_list = [], [], [], [], []
    for st in range(0, 150, 16):
        xb = rgb_src[st:st + 16].to(device)
        zs = (a0.encode(xb * 2.0 - 1.0).latent_dist.mean * 0.18215).float()
        c64 = rgb_to_structure_64(xb)
        y16 = project_latent_y16_lcn(zs, w_luma, b_luma)
        cg = cnx_ext.extract_gray_layout(xb)
        q32 = stiefel_polar_quotient(cg, w_stiefel)
        z_list.append(zs.cpu())
        c64_list.append(c64.cpu())
        y16_list.append(y16.cpu())
        q32_list.append(q32.cpu())
        cg_list.append(cg.cpu())

    z_src = torch.cat(z_list, 0)
    c64_src = torch.cat(c64_list, 0)
    y16_src = torch.cat(y16_list, 0)
    q32_src = torch.cat(q32_list, 0)
    cg_src = torch.cat(cg_list, 0)
    del a0, cnx_ext
    gc.collect()
    torch.cuda.empty_cache()

    # Load frozen Core-5 reference features & V36 150 refs
    ea = torch.load(V36_CACHE_DIR / "v36_eval_assets.pt", map_location="cpu", weights_only=False)
    refs150 = ea["refs150"]
    dec = load_s10_decoder(device)
    ref_u8_list = []
    for st in range(0, len(refs150["z_ref"]), 24):
        rg = decode_s10(dec, refs150["z_ref"][st:st + 24].to(device).float(), batch=24, output_cpu=True).clamp(0.0, 1.0)
        ref_u8_list.append((rg * 255.0).round().to(torch.uint8))
    ref_u8 = torch.cat(ref_u8_list, 0)
    del dec
    gc.collect()
    torch.cuda.empty_cache()

    stack = FeatureStack(device)
    proto = stack.csd_prototypes
    mb = auto_referee_batch(device)

    src_imgs = [Image.fromarray(a) for a in src_u8.permute(0, 2, 3, 1).numpy()]
    sf = stack.extract(src_imgs, batch=mb)
    ref_imgs = [Image.fromarray(a) for a in ref_u8.permute(0, 2, 3, 1).numpy()]
    rf = stack.extract(ref_imgs, batch=mb)
    del stack
    gc.collect()
    torch.cuda.empty_cache()

    ref_dino = [rf["dino"][s * 30:(s + 1) * 30].clone().cpu() for s in range(5)]
    ref_clip = torch.stack([rf["clip"][s * 30:(s + 1) * 30].mean(dim=0) for s in range(5)], dim=0)
    ref_clip = (ref_clip / ref_clip.norm(dim=-1, keepdim=True).clamp_min(1e-8)).cpu()

    tasks = []
    for k, rec in enumerate(selected_src):
        stem = f"art_ho_{k:03d}__{rec['art_slug']}__{Path(rec['path']).stem}"
        for ts in range(5):
            tid_int = k * 5 + ts
            tasks.append({
                "task_id": f"f1_{tid_int:04d}",
                "source_pos": k,
                "source_stem": stem,
                "source_artist": rec["art_slug"],
                "source_style": rec["style"],
                "target_style": STYLES[ts],
                "source_style_id": 5,  # strictly non-Core-5 -> all 750 are unseen-artist cross-style
                "target_style_id": ts,
            })

    audit_info = {
        "n_sources": len(selected_src),
        "n_tasks": len(tasks),
        "n_unique_source_artists": len(sel_arts),
        "unique_source_movements": sorted({r["style"] for r in selected_src}),
        "overlap_train4400_artists": len(overlap_tr4400),
        "overlap_v36_seen_artists": len(overlap_v36),
        "overlap_test750_artists": len(sel_arts & {_norm_slug(_artist_key(t["source_stem"])) for t in tasks750}),
        "overlap_refs_artists": len(sel_arts & {_norm_slug(_artist_key(r["stem"])) for rows in refs_json.values() for r in rows}),
        "overlap_pids": len({canonical_pid(Path(r["path"]).stem) for r in selected_src} & banned_pids),
    }

    payload = {
        "audit": audit_info,
        "selected_records": selected_src,
        "tasks": tasks,
        "src_u8": src_u8,
        "z_src": z_src,
        "c64_src": c64_src,
        "y16_src": y16_src,
        "q32_src": q32_src,
        "cg_src": cg_src,
        "src_dino": sf["dino"].cpu(),
        "src_csd": torch.as_tensor(sf["csd"]).cpu(),
        "ref_dino": ref_dino,
        "ref_clip": ref_clip,
        "csd_proto": torch.as_tensor(np.asarray(proto)).cpu(),
        "refs150_z": refs150["z_ref"],
        "refs150_stat": refs150["stat_1561"],
    }
    torch.save(payload, SPEC_PATH)
    save_json(F1_DIR / "artist_holdout_audit.json", audit_info)
    print(f"[F1 Spec Built] {json.dumps(audit_info)}", flush=True)
    return payload


@torch.no_grad()
def generate_f1_arms(device: str = "cuda") -> list[tuple[Path, Path, str]]:
    spec = build_or_load_artist_holdout_spec(device=device)
    tasks = spec["tasks"]
    n_t = len(tasks)
    src_pos = torch.tensor([t["source_pos"] for t in tasks], dtype=torch.long)
    tid_750 = torch.tensor([t["target_style_id"] for t in tasks], dtype=torch.long)
    sid_dummy = torch.tensor([t["target_style_id"] for t in tasks], dtype=torch.long)
    tid_wrong = (tid_750 + 2) % 5  # deterministic wrong target movement

    z_src = spec["z_src"][src_pos]
    c64 = spec["c64_src"][src_pos]
    y16 = spec["y16_src"][src_pos]
    q32 = spec["q32_src"][src_pos]
    cg = spec["cg_src"][src_pos]
    eps = torch.randn(z_src.shape, generator=torch.Generator().manual_seed(EVAL_SEED))

    raw_root = F1_DIR / "raw"
    meas_root = F1_DIR / "measured"
    bs = auto_infer_batch(device)
    dec = load_s10_decoder(device)

    all_arms: list[tuple[Path, Path, str]] = []

    def _run_and_write(arm_name: str, fn_batch):
        raw_p = raw_root / arm_name
        meas_p = meas_root / arm_name
        all_arms.append((raw_p, meas_p, arm_name))
        if (raw_p / "meta.json").exists() or (meas_p / "eval.json").exists():
            return
        writer = RawWriter(arm_name, raw_root, kind="f1_artist_holdout", expected_tasks=n_t)
        dec_bs = auto_decode_batch(device)
        for st in range(0, n_t, bs):
            ed = min(n_t, st + bs)
            zo = fn_batch(slice(st, ed))
            rgb = decode_s10(dec, zo, batch=dec_bs, output_cpu=True).clamp(0.0, 1.0)
            writer.append([t["task_id"] for t in tasks[st:ed]], rgb)
        writer.finish(arm=arm_name, utc=utc())

    # 1. CLEAVE-E (ECore_NoDINO_final.pt)
    ecore, raw_stats, priors_ec, priors_a = load_v32_ecore_and_priors(device=device)
    for lam in (0.50, 0.75, 1.00):
        arm = f"F1_CLEAVE_E_l{int(round(lam * 100)):03d}"
        _run_and_write(
            arm,
            lambda sl, _l=lam: ecore(
                z_src[sl].to(device).float(), c64[sl].to(device).float(), q32[sl].to(device).float(),
                tid_750[sl].to(device), torch.full((sl.stop - sl.start,), _l, device=device),
                raw_stats, priors_ec, noise=eps[sl].to(device).float(),
            ),
        )

    # 2. CLEAVE-P/E (P8 locked C2_44m_q32_final.pt)
    p8 = read_json(P8_REPORT_PATH)
    fcfg = p8["final_schedule_cfg"]
    pcore, _ = load_pcore_v35(REPO / P8_REPORT_PATH.parent.parent / "v34_frontier/stage_c/C2_44m_q32/C2_44m_q32_final.pt" if not Path(p8["locked_ckpt"]).exists() else Path(p8["locked_ckpt"]), weight_key=p8["locked_weight"], device=device)
    assets = build_or_load_cnx_assets(device=device)
    for s_val, is_w, tag in [(0.60, False, "s060"), (0.85, False, "s085"), (1.00, False, "s100"), (1.50, False, "s150"), (1.00, True, "s100_wrong")]:
        arm = f"F1_CLEAVE_PE_{tag}"
        kw = make_v35_schedule(s_val, ws_mul=fcfg["ws_mul"], wc_mul=fcfg["wc_mul"], dtau0=fcfg["dtau0"])
        lam_d = float(kw.pop("draft_lambda", 0.75))
        tl = tid_wrong if is_w else tid_750

        def _pe_fn(sl, _kw=dict(kw), _lam=lam_d, _tl=tl):
            ti = _tl[sl].to(device)
            ze = ecore(
                z_src[sl].to(device).float(), c64[sl].to(device).float(), q32[sl].to(device).float(),
                ti, torch.full((len(ti),), _lam, device=device), raw_stats, priors_ec, noise=eps[sl].to(device).float(),
            )
            ze_s = build_pcore_startpoint(ze, ti, priors_a, alpha_y=fcfg["alpha_y"], alpha_c=fcfg["alpha_c"])
            u_st = match_cnx_k8_submode(cg[sl].to(device).float(), ti, assets["bank"])
            return integrate_v33_trajectory(
                pcore, ze_s, eps[sl].to(device).float(),
                y16[sl].to(device).float(), q32[sl].to(device).float(),
                ti, u_st, steps=int(fcfg["steps"]), solver="dpm2m", **_kw,
            )

        _run_and_write(arm, _pe_fn)

    del ecore, pcore
    gc.collect()
    torch.cuda.empty_cache()

    # 3. CLEAVE-One (V36 P2 50k + Reference-A3 I2)
    pcore_v36, style_v36, priors_v36 = load_v36_models(V36_P2_CKPT, device)
    e_refs = style_v36(spec["refs150_stat"].to(device).float()).cpu()
    z_refs = spec["refs150_z"]
    for t0, is_w, tag in [(0.35, False, "t35"), (0.50, False, "t50"), (0.65, False, "t65"), (0.50, True, "t50_wrong")]:
        arm = f"F1_CLEAVE_One_{tag}"
        tl = tid_wrong if is_w else tid_750

        def _one_fn(sl, _t0=t0, _tl=tl):
            r_idx = torch.tensor([int(_tl[i].item()) * 30 + (i % 30) for i in range(sl.start, sl.stop)], dtype=torch.long)
            zs = reconstruct_reference_a3(z_src[sl].to(device).float(), z_refs[r_idx].to(device).float(), priors_v36, mode="I2")
            return sample_v36_ode(
                pcore_v36, zs, y16[sl].to(device).float(), q32[sl].to(device).float(),
                e_refs[r_idx].to(device), eps[sl].to(device).float(),
                tau0=_t0, steps=6, wj=3.8, wc=1.6, device=device,
            )

        _run_and_write(arm, _one_fn)

    del pcore_v36, style_v36, dec
    gc.collect()
    torch.cuda.empty_cache()
    return all_arms


@torch.no_grad()
def measure_f1_worker(device: str = "cuda") -> None:
    spec = torch.load(SPEC_PATH, map_location="cpu", weights_only=False)
    tasks = spec["tasks"]
    by_id = {t["task_id"]: t for t in tasks}
    stack = FeatureStack(device)
    proto = np.asarray(stack.csd_prototypes)
    rd = [r.to(device) for r in spec["ref_dino"]]
    rc = spec["ref_clip"].to(device)
    src_u8 = spec["src_u8"]
    src_dino = spec["src_dino"].numpy()
    src_csd = spec["src_csd"].numpy()
    mb = auto_referee_batch(device)

    for raw_p in sorted((F1_DIR / "raw").iterdir()):
        if not raw_p.is_dir() or not (raw_p / "meta.json").exists():
            continue
        meas_p = F1_DIR / "measured" / raw_p.name
        if (meas_p / "eval.json").exists():
            continue
        meas_p.mkdir(parents=True, exist_ok=True)
        meta = read_json(raw_p / "meta.json")
        rows = []
        for entry in meta["batches"]:
            pl = torch.load(raw_p / entry["file"], map_location="cpu", weights_only=False)
            group = [by_id[i] for i in pl["ids"]]
            u8 = pl["rgb_u8"]
            imgs = [Image.fromarray(a) for a in u8.permute(0, 2, 3, 1).numpy()]
            fo = stack.extract(imgs, batch=mb)
            df, cf, clf = fo["dino"], fo["csd"], fo["clip"]
            scores = torch.stack([(df @ p.T).max(1).values for p in rd], 1).cpu().numpy()
            cs = cf @ proto.T
            clips = (clf @ rc.T).cpu().numpy()
            raw_rgb = torch.stack([src_u8[t["source_pos"]] for t in group])
            # FeatureStack normalizes generated uint8 only; references must be RGB01.
            lr = stack.lpips_vs(u8, raw_rgb.float().div(255.0), batch=mb).cpu().tolist()
            for i, t in enumerate(group):
                ts = int(t["target_style_id"])
                spos = int(t["source_pos"])
                p_cls = int(cs[i].argmax())
                rows.append({
                    "task_id": t["task_id"],
                    "src_stem": t["source_stem"],
                    "src_artist": t["source_artist"],
                    "src_style": t["source_style"],
                    "tgt_style": t["target_style"],
                    "src_style_id": 5,
                    "tgt_style_id": ts,
                    "is_cross": True,
                    "pred_class": p_cls,
                    "style_scores_5": scores[i].tolist(),
                    "clip_scores_5": clips[i].tolist(),
                    "csd_scores_5": cs[i].tolist(),
                    "dino_s": float(scores[i, ts]),
                    "dino_c": float(df[i].cpu().numpy() @ src_dino[spos]),
                    "lpips": float(lr[i]),
                    "target_top1": int(p_cls == ts),
                    "target_margin": float(cs[i, ts] - np.delete(cs[i], ts).max()),
                    "dT": float(cs[i, ts] - src_csd[spos] @ proto[ts]),
                })
        save_json(meas_p / "eval.json", {"n_total": len(rows), "n_cross": len(rows), "n_same": 0, "rows": rows})
        print(f"[F1 measured] {raw_p.name}: n={len(rows)}", flush=True)


def _summarize_arm(rows: list[dict[str, Any]]) -> dict[str, Any]:
    dino_s = float(np.mean([r["dino_s"] for r in rows]))
    clip_s = float(np.mean([r["clip_scores_5"][r["tgt_style_id"]] for r in rows]))
    per_tgt = {}
    for sid, st in enumerate(STYLES):
        sub = [r for r in rows if int(r["tgt_style_id"]) == sid]
        ds = float(np.mean([r["dino_s"] for r in sub]))
        cs = float(np.mean([r["clip_scores_5"][sid] for r in sub]))
        per_tgt[st] = {
            "n": len(sub),
            "csd_top1": round(float(np.mean([r["target_top1"] for r in sub])), 4),
            "dino_c": round(float(np.mean([r["dino_c"] for r in sub])), 4),
            "dino_s": round(ds, 4),
            "clip_s": round(cs, 4),
            "style_affinity": round(0.5 * (ds + cs), 4),
            "lpips": round(float(np.mean([r["lpips"] for r in sub])), 4),
        }
    return {
        "n": len(rows),
        "cross750": {
            "csd_top1": round(float(np.mean([r["target_top1"] for r in rows])), 4),
            "csd_margin": round(float(np.mean([r["target_margin"] for r in rows])), 5),
            "dino_c": round(float(np.mean([r["dino_c"] for r in rows])), 4),
            "dino_s": round(dino_s, 4),
            "clip_s": round(clip_s, 4),
            "style_affinity": round(0.5 * (dino_s + clip_s), 4),
            "lpips": round(float(np.mean([r["lpips"] for r in rows])), 4),
        },
        "per_target_150": per_tgt,
    }


def finalize_f1() -> dict[str, Any]:
    spec_audit = read_json(F1_DIR / "artist_holdout_audit.json")
    arms_out: dict[str, Any] = {}
    for mp in sorted((F1_DIR / "measured").iterdir()):
        ev_p = mp / "eval.json"
        if ev_p.exists():
            rows = read_json(ev_p)["rows"]
            arms_out[mp.name] = _summarize_arm(rows)

    # Compare vs test750 baselines if local paths exist
    pe_rows = {k: arms_out[f"F1_CLEAVE_PE_{k}"]["cross750"] for k in ("s060", "s085", "s100", "s150") if f"F1_CLEAVE_PE_{k}" in arms_out}
    e_rows = {k: arms_out[f"F1_CLEAVE_E_{k}"]["cross750"] for k in ("l050", "l075", "l100") if f"F1_CLEAVE_E_{k}" in arms_out}
    one_rows = {k: arms_out[f"F1_CLEAVE_One_{k}"]["cross750"] for k in ("t35", "t50", "t65") if f"F1_CLEAVE_One_{k}" in arms_out}

    matched = {
        "cleave_e": summarize_matched_c(e_rows, c_grid=(0.61, 0.70, 0.75)) if len(e_rows) >= 2 else {},
        "cleave_pe": summarize_matched_c(pe_rows, c_grid=(0.50, 0.61, 0.70, 0.75)) if len(pe_rows) >= 2 else {},
        "cleave_one": summarize_matched_c(one_rows, c_grid=(0.50, 0.61, 0.70, 0.75)) if len(one_rows) >= 2 else {},
    }

    # Bootstrap CI across 150 unseen-artist sources for key operating points
    boot = {}
    if "F1_CLEAVE_PE_s085" in arms_out and "F1_CLEAVE_One_t50" in arms_out:
        r_pe = read_json(F1_DIR / "measured/F1_CLEAVE_PE_s085/eval.json")["rows"]
        r_one = read_json(F1_DIR / "measured/F1_CLEAVE_One_t50/eval.json")["rows"]
        r_e = read_json(F1_DIR / "measured/F1_CLEAVE_E_l075/eval.json")["rows"]
        boot["PE_s085_minus_E_l075"] = paired_cluster_ci(r_pe, r_e)
        boot["PE_s085_minus_One_t50"] = paired_cluster_ci(r_pe, r_one)
    if "F1_CLEAVE_PE_s100_wrong" in arms_out:
        r_ok = read_json(F1_DIR / "measured/F1_CLEAVE_PE_s100/eval.json")["rows"]
        r_wr = read_json(F1_DIR / "measured/F1_CLEAVE_PE_s100_wrong/eval.json")["rows"]
        boot["PE_s100_causal"] = paired_cluster_ci(r_ok, r_wr)
    if "F1_CLEAVE_One_t50_wrong" in arms_out:
        r_ok = read_json(F1_DIR / "measured/F1_CLEAVE_One_t50/eval.json")["rows"]
        r_wr = read_json(F1_DIR / "measured/F1_CLEAVE_One_t50_wrong/eval.json")["rows"]
        boot["One_t50_causal"] = paired_cluster_ci(r_ok, r_wr)

    rep = {"utc": utc(), "audit": spec_audit, "arms": arms_out, "matched_c": matched, "paired_ci": boot}
    save_json(F1_DIR / "f1_summary.json", rep)
    print(f"[F1 DONE] saved {F1_DIR / 'f1_summary.json'}", flush=True)
    return rep


def main() -> None:
    sanitize_cuda_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "gen", "measure", "finalize"])
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.stage in ("all", "gen"):
        generate_f1_arms(a.device)
    if a.stage == "all":
        subprocess.run([sys.executable, "-u", "-m", "exp.paper_audit.f1_artist_holdout", "--stage", "measure", "--device", a.device], check=True)
    elif a.stage == "measure":
        measure_f1_worker(a.device)
    if a.stage in ("all", "finalize"):
        finalize_f1()


if __name__ == "__main__":
    main()
