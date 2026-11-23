"""Phase 1: 150-task Gated Horse Race across T1 (Multi-Expert), T2 (Zero-Legacy), T3 (Reverse Pilot), T4 (FGW Gap-Filler)."""
from __future__ import annotations

import argparse
import gc
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from exp.v18 import STYLES
from exp.v20.slf_flow import coarse_y16_lcn
from exp.v20.submode_patchswd import build_or_load_core_knn_bank
from exp.v21.stiefel_ref_cache import build_or_load_v21_cache
from exp.v27_mainline.part_bank import build_or_load_part_bank
from exp.v27_single.unified.teacher_gate import task_plan
from exp.v28_unified import OUT, ROOT, SEED_PLAN, assert_vram_budget, atomic_save
from exp.v28_unified.data_map import load_rgb_512_tensor, resolve_train_rgb_paths
from exp.v28_unified.referee_free import RefereeFreeSelector
from exp.v28_unified.t3_reverse import build_t3_pool
from exp.v28_unified.teacher_routes import (
    T1MultiExpertTeacher,
    T2ZeroLegacyOptimizer,
    T4RegionalGapFiller,
)
from src.utils import FeatureStack, decode_s10, load_s10_decoder, reference_bank
from src.utils._local_eval_refs import csd_prototypes
from src.utils.runtime import sanitize_cuda_env, save_json, seed_all, utc


def _to_pils(rgb: torch.Tensor) -> list[Image.Image]:
    u8 = (rgb.detach().cpu().float().clamp(0.0, 1.0) * 255.0).round().byte().permute(0, 2, 3, 1).numpy()
    return [Image.fromarray(arr) for arr in u8]


@torch.inference_mode()
def evaluate_150_referee(
    dec_fp16,
    paths_4400: list,
    src_idx: torch.Tensor,
    tid: torch.Tensor,
    arms_latents: dict[str, torch.Tensor],
    device: str = "cuda",
) -> dict:
    """Evaluate 150-task teacher endpoints with frozen referee (CSD, DINO-C vs orig & S10, DINO-S, LPIPS, CSLS k=10)."""
    # Decode all latents first, then release decoder before loading FeatureStack
    rgb_orig = torch.cat([load_rgb_512_tensor([paths_4400[int(k)] for k in sl.tolist()]) for sl in src_idx.split(25)], 0)
    decoded = {}
    for name, z in arms_latents.items():
        decoded[name] = decode_s10(dec_fp16, z.to(device).float(), batch=12, output_cpu=True).clamp(0.0, 1.0)

    stack = FeatureStack(device)
    proto = stack.csd_prototypes
    rbank = reference_bank(device)
    rd = rbank["dino"]
    rc = rbank["clip"]

    # CSLS k=10 background over the 150 unique test sources
    src_feats = torch.load(ROOT / "runs/v6_r2/cache/tasks750_source_features.pt", map_location="cpu", weights_only=False)
    tasks_750 = __import__("json").loads((ROOT / "eval_protocol/tasks_750.json").read_text(encoding="utf-8"))["tasks"]
    stems = sorted({t["source_stem"] for t in tasks_750})
    bg_sim = np.stack([src_feats["raw_csd"][s] for s in stems]) @ proto.T
    r_proto = np.sort(bg_sim, axis=0)[-10:].mean(0)

    f_orig = stack.extract(_to_pils(rgb_orig), batch=16)
    u8_orig = (rgb_orig * 255.0).round().byte()
    tid_np = tid.numpy()

    out = {}
    for name, rgb in decoded.items():
        feats = stack.extract(_to_pils(rgb), batch=16)
        csd_scores = feats["csd"] @ proto.T
        pred = csd_scores.argmax(axis=1)
        hit = pred == tid_np
        csls_pred = (2.0 * csd_scores - r_proto).argmax(axis=1)
        csls_hit = csls_pred == tid_np
        dino_s_mat = torch.stack([(feats["dino"].to(device) @ p.T).max(1).values for p in rd], dim=1).cpu().numpy()
        dino_s = float(dino_s_mat[np.arange(len(tid_np)), tid_np].mean())
        clip_mat = (feats["clip"].to(device) @ rc.T).cpu().numpy()
        clip_s = float(clip_mat[np.arange(len(tid_np)), tid_np].mean())
        dino_c_orig = float(F.cosine_similarity(feats["dino"].float(), f_orig["dino"].float()).mean())
        u8 = (rgb * 255.0).round().byte()
        lpips_orig = float(stack.lpips_vs(u8, u8_orig, batch=16).mean())
        out[name] = {
            "csd_top1": round(float(hit.mean()), 4),
            "csls_k10": round(float(csls_hit.mean()), 4),
            "er_hits_30": int(hit[tid_np == 0].sum()),
            "uki_hits_30": int(hit[tid_np == 4].sum()),
            "er_rate": round(float(hit[tid_np == 0].mean()), 4) if (tid_np == 0).any() else None,
            "uki_rate": round(float(hit[tid_np == 4].mean()), 4) if (tid_np == 4).any() else None,
            "per_target": {
                STYLES[s]: round(float(hit[tid_np == s].mean()), 4) if (tid_np == s).any() else None
                for s in range(5)
            },
            "dino_s": round(dino_s, 4),
            "clip_s": round(clip_s, 4),
            "dino_c_vs_orig": round(dino_c_orig, 4),
            "lpips_vs_orig": round(lpips_orig, 4),
        }
    del stack
    gc.collect()
    torch.cuda.empty_cache()
    return out


def run_phase1(pool_dir: str, device: str = "cuda") -> dict:
    sanitize_cuda_env()
    seed_all(SEED_PLAN)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()

    pack = build_or_load_v21_cache(pool_dir, device=device)
    bundle, c21 = pack["bundle"], pack["c21"]
    bank = build_or_load_core_knn_bank(bundle, k_sub=8, core_per_style=180, device=device)
    pbank_dict = build_or_load_part_bank(pool_dir, device=device)

    tr = bundle["v18_cache"]["train"]
    paths_4400 = resolve_train_rgb_paths(tr["ids"], tr["sid"], require_full=True)
    _, src_idx, tid = task_plan(tr)
    eps = torch.randn((len(src_idx), 4, 64, 64), generator=torch.Generator().manual_seed(SEED_PLAN + 1)).half()

    dec_fp16 = load_s10_decoder(device)
    dec_bf16 = load_s10_decoder(device).to(torch.bfloat16).eval()
    for p in dec_bf16.parameters():
        p.requires_grad_(False)

    selector = RefereeFreeSelector(bundle, bank, dec_bf16, device=device)
    rgb_src = torch.cat([load_rgb_512_tensor([paths_4400[int(k)] for k in sl.tolist()]) for sl in src_idx.split(25)], 0)

    # ---- Route T1: 5-expert 3-tier generation + referee-free selection ----
    t_t1 = time.monotonic()
    t1 = T1MultiExpertTeacher(bundle, c21, bank, dec_bf16, device=device)
    cands = t1.generate_candidates(
        tr["z"][src_idx],
        coarse_y16_lcn(tr["y_64"][src_idx].float()),
        tr["dino_16"][src_idx],
        c21["train"]["q32"][src_idx],
        tr["c_64"][src_idx],
        bundle["fnd"]["tr_fib"][src_idx],
        tid,
        eps,
        src_idx=src_idx,
        batch=16,
    )
    t1_sel = {"hiC": [], "mid": [], "hiS": [], "hiS_gate_pass": [], "hiS_pick": []}
    for st in range(0, len(src_idx), 16):
        sl = slice(st, min(st + 16, len(src_idx)))
        res = selector.select_tiers(
            tr["z"][src_idx[sl]], rgb_src[sl].to(device), tid[sl].to(device), [c[sl] for c in cands]
        )
        for k in t1_sel:
            t1_sel[k].append(res[k])
    t1_sel = {k: torch.cat(v, 0) for k, v in t1_sel.items()}
    t1_elapsed = time.monotonic() - t_t1
    print(f"[Phase 1 T1] 150 tasks x 9 experts in {t1_elapsed:.1f}s | gate_pass hiS={float(t1_sel['hiS_gate_pass'].float().mean()):.3f}", flush=True)

    # ---- Route T4: Offline Partial-FGW + Part-Bary-SWD gap-filler on failed/hard ER & Uki ----
    t_t4 = time.monotonic()
    t4 = T4RegionalGapFiller(bundle, bank, dec_bf16, pbank_dict, device=device)
    is_hard_style = (tid == 0) | (tid == 4)
    # Trigger on failed gate OR worst 20% style score among ER/Uki
    trigger_mask = is_hard_style & (~t1_sel["hiS_gate_pass"])
    if int(trigger_mask.sum()) == 0:
        # Ensure at least worst 15% of ER/Uki are tested in Phase 1
        hard_idx = is_hard_style.nonzero().flatten()
        scores_hard = selector.score_candidates(
            tr["z"][src_idx[hard_idx]], rgb_src[hard_idx].to(device), tid[hard_idx].to(device), [t1_sel["hiS"][hard_idx]]
        )["style_dist"].squeeze(1).cpu()
        n_trig = max(2, int(round(0.20 * len(hard_idx))))
        trigger_mask[hard_idx[scores_hard.topk(n_trig).indices]] = True

    trig_idx = trigger_mask.nonzero().flatten()
    z_t1_t4_his = t1_sel["hiS"].clone()
    if len(trig_idx) > 0:
        best_c_trig = selector.match_gray_exemplar(rgb_src[trig_idx].to(device), tid[trig_idx].to(device))
        z_refined = t4.refine_batch(
            t1_sel["hiS"][trig_idx],
            tr["z"][src_idx[trig_idx]],
            rgb_src[trig_idx],
            c21["train"]["q32"][src_idx[trig_idx]],
            c21["train"]["q32"],
            tid[trig_idx].to(device),
            best_c_trig,
            steps=15,
        )
        # Re-check structural gate vs T1 hiS
        sc_after = selector.select_tiers(
            tr["z"][src_idx[trig_idx]], rgb_src[trig_idx].to(device), tid[trig_idx].to(device),
            [t1_sel["hiS"][trig_idx], z_refined],
        )
        z_t1_t4_his[trig_idx] = sc_after["hiS"]
        t4_unsolved = int((~sc_after["hiS_gate_pass"]).sum())
    else:
        t4_unsolved = 0
    t4_elapsed = time.monotonic() - t_t4
    print(f"[Phase 1 T4] triggered={len(trig_idx)}/{int(is_hard_style.sum())}, unsolved={t4_unsolved}, time={t4_elapsed:.1f}s", flush=True)

    # ---- Route T2: Zero-legacy single differentiable optimizer (steps 3/6/10) ----
    t_t2 = time.monotonic()
    t2 = T2ZeroLegacyOptimizer(bundle, selector, dec_bf16, device=device)
    t2_snaps = {3: [], 6: [], 10: []}
    for st in range(0, len(src_idx), 16):
        sl = slice(st, min(st + 16, len(src_idx)))
        sn = t2.optimize_batch(tr["z"][src_idx[sl]], rgb_src[sl], tid[sl], total_steps=10, snap_steps=(3, 6, 10))
        for k in (3, 6, 10):
            t2_snaps[k].append(sn[k])
    t2_snaps = {k: torch.cat(v, 0) for k, v in t2_snaps.items()}
    t2_elapsed = time.monotonic() - t_t2
    print(f"[Phase 1 T2] 150 tasks x 10 steps in {t2_elapsed:.1f}s", flush=True)

    # ---- Route T3: Pilot 150 (75 ER + 75 Uki) reversed-source quality check ----
    t_t3 = time.monotonic()
    t3_pilot = build_t3_pool(
        bundle, c21, bank, t1, selector, dec_bf16,
        per_style_pool=120, keep_per_style=75,
        out_path=OUT / "phase1_t3_pilot150.pt", device=device,
    )
    t3_elapsed = time.monotonic() - t_t3

    # Save endpoints for reuse in Phase 2/inspection
    atomic_save({
        "src_idx": src_idx, "tid": tid, "eps": eps,
        "t1_hiC": t1_sel["hiC"], "t1_mid": t1_sel["mid"], "t1_hiS": t1_sel["hiS"],
        "t1_t4_hiS": z_t1_t4_his,
        "t2_hiC": t2_snaps[3], "t2_mid": t2_snaps[6], "t2_hiS": t2_snaps[10],
    }, OUT / "phase1_endpoints_150.pt")

    del t1, t2, t4, selector, dec_bf16
    gc.collect()
    torch.cuda.empty_cache()

    arms_to_eval = {
        "T1_hiC": t1_sel["hiC"],
        "T1_mid": t1_sel["mid"],
        "T1_hiS": t1_sel["hiS"],
        "T1_T4_hiS": z_t1_t4_his,
        "T2_hiC_s3": t2_snaps[3],
        "T2_mid_s6": t2_snaps[6],
        "T2_hiS_s10": t2_snaps[10],
        # Individual key experts at hiS/hiC for ablation clarity
        "Expert_Peak1500_hiS": cands[4],
        "Expert_Q32_e2full": cands[1],
        "Expert_V19_S10VGG_hiC": cands[7],
        "Expert_V19_S10VGG_str": cands[9],
    }
    eval_150 = evaluate_150_referee(dec_fp16, paths_4400, src_idx, tid, arms_to_eval, device=device)

    # Also evaluate T3 pilot real vs fwd on the 150 reversed tasks
    # Note: for T3, evaluate against z_syn using FeatureStack
    t3_eval = evaluate_150_referee(
        dec_fp16, paths_4400, t3_pilot["real_idx"], t3_pilot["tid"],
        {"T3_real_GT": t3_pilot["z_real"], "T3_fwd_teacher": t3_pilot["z_fwd"]},
        device=device,
    )

    del dec_fp16
    gc.collect()
    torch.cuda.empty_cache()

    def _check_gate(hic_row: dict, his_row: dict) -> dict:
        c = {
            "his_top1_ge_62": his_row["csd_top1"] >= 0.62,
            "his_er_ge_38": his_row["er_rate"] >= 0.38,
            "his_uki_ge_38": his_row["uki_rate"] >= 0.38,
            "hic_dino_c_ge_64": hic_row["dino_c_vs_orig"] >= 0.64,
        }
        c["passed"] = all(c.values())
        return c

    vram_gb = assert_vram_budget(device=device, tag="phase1")
    report = {
        "updated_utc": utc(),
        "elapsed_s": round(time.monotonic() - t0, 1),
        "peak_vram_gb": round(vram_gb, 3),
        "timings_150_s": {
            "T1": round(t1_elapsed, 1),
            "T4": round(t4_elapsed, 1),
            "T2": round(t2_elapsed, 1),
            "T3_pilot": round(t3_elapsed, 1),
        },
        "t1_pick_histogram_hiS": [int((t1_sel["hiS_pick"] == i).sum()) for i in range(len(T1MultiExpertTeacher.CANDIDATE_NAMES))],
        "t4_stats": {
            "triggered": int(len(trig_idx)),
            "hard_total": int(is_hard_style.sum()),
            "unsolved_after_t4": t4_unsolved,
            "unsolved_rate": round(t4_unsolved / max(1, int(is_hard_style.sum())), 4),
        },
        "t3_pilot_geom": {
            "n": len(t3_pilot["tid"]),
            "mean_gray_sim_err": round(float(t3_pilot["gray_sim_err"].mean()), 4),
            "mean_lcn_err": round(float(t3_pilot["lcn_err"].mean()), 4),
        },
        "arms_150": eval_150,
        "t3_pilot_150": t3_eval,
        "gates": {
            "T1": _check_gate(eval_150["T1_hiC"], eval_150["T1_hiS"]),
            "T1_T4": _check_gate(eval_150["T1_hiC"], eval_150["T1_T4_hiS"]),
            "T2": _check_gate(eval_150["T2_hiC_s3"], eval_150["T2_hiS_s10"]),
        },
    }
    save_json(OUT / "phase1_race.json", report)
    print(f"[Phase 1] Gates: {report['gates']}", flush=True)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    run_phase1(args.pool, device=args.device)


if __name__ == "__main__":
    main()
