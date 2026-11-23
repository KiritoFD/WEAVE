"""T3: In-domain reversed real-painting pairs (ER/Uki GT on output side, Imp/Min/Roc synthetic sources)."""
from __future__ import annotations

import gc
import time

import torch
import torch.nn.functional as F

from exp.v18 import STYLES
from exp.v19.fiber_duality import extract_color_fiber_27d, extract_dino_rgb_moments_768d
from exp.v20.slf_flow import coarse_y16_lcn
from exp.v21.stiefel_ref_cache import stiefel_polar_quotient
from exp.v27.icme_step2_closedset_accel import D5_CFG
from exp.v27_mainline.dir1_reflow import _integrate_q32_seg
from exp.v28_unified import OUT, SEED_PLAN, atomic_save, recipe_hash
from exp.v28_unified.data_map import load_rgb_512_tensor, resolve_train_rgb_paths
from exp.v28_unified.referee_free import RefereeFreeSelector, latent_luma_lcn_dist
from exp.v28_unified.teacher_routes import T1MultiExpertTeacher
from src.utils import decode_s10
from src.utils._local_eval_refs import load_dino_model
from src.utils.runtime import bf16_autocast, read_json, save_json, seed_all
from src.utils.structure import extract_whitened_dino_layout, rgb_to_gray, rgb_to_structure_64


@torch.no_grad()
def build_t3_pool(
    bundle: dict,
    c21: dict,
    bank: dict,
    t1: T1MultiExpertTeacher,
    selector: RefereeFreeSelector,
    dec_bf16,
    *,
    per_style_pool: int = 880,
    keep_per_style: int = 600,
    out_path=None,
    device: str = "cuda",
) -> dict:
    """Build curated in-domain reversed real-painting pool Pool C (600 ER + 600 Uki = 1,200 pairs) + causal controls."""
    out_path = out_path or (OUT / "pools/pool_C.pt")
    if out_path.exists():
        cached = torch.load(out_path, map_location="cpu", weights_only=False)
        if cached.get("keep_per_style") == keep_per_style and len(cached["tid"]) == 2 * keep_per_style:
            print(f"[T3] Reusing {out_path} ({len(cached['tid'])} pairs)", flush=True)
            return cached

    seed_all(SEED_PLAN + 300)
    tr = bundle["v18_cache"]["train"]
    paths_4400 = resolve_train_rgb_paths(tr["ids"], tr["sid"], require_full=True)
    shard_dir = OUT / "pools/shards_pool_C"
    shard_dir.mkdir(parents=True, exist_ok=True)

    # Select up to per_style_pool real ER (0) and real Uki (4) paintings from train_4400
    gen = torch.Generator().manual_seed(SEED_PLAN + 301)
    cand_indices = []
    for sid in (0, 4):
        ids = (tr["sid"] == sid).nonzero().flatten()
        perm = ids[torch.randperm(len(ids), generator=gen)[:per_style_pool]]
        cand_indices.append(perm)
    all_idx = torch.cat(cand_indices, dim=0)
    N_all = len(all_idx)
    # Assign synthetic intermediate style in {1: Imp, 2: Min, 3: Roc}
    s_syn_all = torch.tensor([1 + (i % 3) for i in range(N_all)], dtype=torch.long)

    r_hash = recipe_hash({"per_style_pool": per_style_pool, "keep_per_style": keep_per_style, "seed": SEED_PLAN})
    hi_c = dict(D5_CFG, tau_min=0.48, tau_hi=0.78, ws_fig=5.2)
    hi_s = dict(D5_CFG, tau_min=0.30, tau_hi=0.68, ws_fig=5.5, wc_fig=1.65,
                wc_abs=0.70, wsp_fig_late=3.6, wsp_abs=5.2)

    # Stage 1: Generate z_syn and score geometric alignment against original real painting
    syn_chunks, gsim_chunks, lcn_chunks = [], [], []
    eps_gen = torch.Generator().manual_seed(SEED_PLAN + 302)
    for sh_i, sl_idx in enumerate(all_idx.split(256)):
        sp = shard_dir / f"syn_{sh_i:03d}.pt"
        sl_start = sh_i * 256
        sl_end = sl_start + len(sl_idx)
        s_syn_b = s_syn_all[sl_start:sl_end]
        if sp.exists():
            pl = torch.load(sp, map_location="cpu", weights_only=False)
            if pl.get("recipe_hash") == r_hash and torch.equal(pl["real_idx"], sl_idx):
                syn_chunks.append(pl["z_syn"])
                gsim_chunks.append(pl["gray_sim_err"])
                lcn_chunks.append(pl["lcn_err"])
                continue
        eps_b = torch.randn((len(sl_idx), 4, 64, 64), generator=eps_gen).half()
        z_syn_b, gs_b, lc_b = [], [], []
        for sub in range(0, len(sl_idx), 16):
            sub_idx = sl_idx[sub:sub + 16]
            ti_syn = s_syn_b[sub:sub + 16].to(device)
            zs = tr["z"][sub_idx].to(device).float()
            yb = coarse_y16_lcn(tr["y_64"][sub_idx].to(device).float())
            db = tr["dino_16"][sub_idx].to(device).float()
            qb = c21["train"]["q32"][sub_idx].to(device).float()
            cb = tr["c_64"][sub_idx].to(device).float()
            fb = bundle["fnd"]["tr_fib"][sub_idx].to(device).float()
            ep = eps_b[sub:sub + 16].to(device).float()

            anc = t1.compute_p1_anchor(zs, db, cb, fb, ti_syn, src_idx=sub_idx)
            z_syn = _integrate_q32_seg(t1.q32_flow, anc, ep, yb, db, qb, ti_syn, bank,
                                       steps=6, full_steps=set(), cfg_kw=hi_c)
            # Evaluate structural preservation vs REAL original RGB!
            rgb_real = load_rgb_512_tensor([paths_4400[int(k)] for k in sub_idx.tolist()]).to(device)
            rgb_syn = decode_s10(dec_bf16, z_syn, batch=8, dtype=torch.bfloat16, output_cpu=False).clamp(0.0, 1.0)
            with bf16_autocast(device):
                r42_r = selector.vgg.forward_gray_r42(rgb_real)
                r42_s = selector.vgg.forward_gray_r42(rgb_syn)
            a16 = F.normalize(F.adaptive_avg_pool2d(r42_r.float(), 16).flatten(2), dim=1)
            b16 = F.normalize(F.adaptive_avg_pool2d(r42_s.float(), 16).flatten(2), dim=1)
            sim_r = torch.bmm(a16.transpose(1, 2), a16)
            sim_s = torch.bmm(b16.transpose(1, 2), b16)
            gs = F.smooth_l1_loss(sim_s, sim_r, beta=0.05, reduction="none").mean(dim=(1, 2))
            lc = latent_luma_lcn_dist(z_syn, zs, selector.w_luma, selector.b_luma)
            z_syn_b.append(z_syn.cpu().half())
            gs_b.append(gs.cpu())
            lc_b.append(lc.cpu())

        pl = {
            "recipe_hash": r_hash,
            "real_idx": sl_idx,
            "z_syn": torch.cat(z_syn_b, 0),
            "gray_sim_err": torch.cat(gs_b, 0),
            "lcn_err": torch.cat(lc_b, 0),
        }
        atomic_save(pl, sp)
        syn_chunks.append(pl["z_syn"])
        gsim_chunks.append(pl["gray_sim_err"])
        lcn_chunks.append(pl["lcn_err"])
        print(f"[T3] Synthetic source shard {sh_i}: {sl_end}/{N_all}", flush=True)

    z_syn_all = torch.cat(syn_chunks, 0)
    gsim_all = torch.cat(gsim_chunks, 0)
    lcn_all = torch.cat(lcn_chunks, 0)
    # Combined geometric distortion score (lower = better geometry alignment)
    geom_score = gsim_all + 0.08 * lcn_all
    real_sid_all = tr["sid"][all_idx]

    kept_pos = []
    for sid in (0, 4):
        m = (real_sid_all == sid).nonzero().flatten()
        best_local = geom_score[m].topk(min(keep_per_style, len(m)), largest=False).indices
        kept_pos.append(m[best_local])
    kept = torch.cat(kept_pos, dim=0)

    kept_real_idx = all_idx[kept]
    kept_tid = tr["sid"][kept_real_idx].long()
    kept_z_real = tr["z"][kept_real_idx].half()
    kept_z_syn = z_syn_all[kept]
    kept_gsim = gsim_all[kept]
    kept_lcn = lcn_all[kept]

    # Stage 2: Extract student conditioning (c_64, q_32, y_16, d_16, fib) for kept z_syn + forward teacher control z_fwd
    dino = load_dino_model(device=device)
    W_stiefel = c21["W_stiefel"]
    c64_l, q32_l, d16_l, y64_l, fib_l, fwd_l = [], [], [], [], [], []
    eps_fwd_gen = torch.Generator().manual_seed(SEED_PLAN + 303)
    for st in range(0, len(kept), 16):
        sl = slice(st, min(st + 16, len(kept)))
        zs = kept_z_syn[sl].to(device).float()
        ti = kept_tid[sl].to(device)
        rgb = decode_s10(dec_bf16, zs, batch=8, dtype=torch.bfloat16, output_cpu=False).clamp(0.0, 1.0)
        cb = rgb_to_structure_64(rgb).float()
        y64b = F.avg_pool2d(rgb_to_gray(rgb), kernel_size=8, stride=8)
        db = extract_whitened_dino_layout(dino, rgb, batch=16, device=device).to(device).float()
        qb = stiefel_polar_quotient(db, W_stiefel).to(device).float()
        col27 = extract_color_fiber_27d(rgb)
        dino768 = extract_dino_rgb_moments_768d(dino, rgb, batch=16, device=device).to(device)
        fb = torch.cat([col27, dino768], dim=-1)

        # Forward teacher control z_fwd from z_syn targeting kept_tid (ER or Uki)
        anc = t1.compute_p1_anchor(zs, db, cb, fb, ti, src_idx=None)
        ep = torch.randn(zs.shape, generator=eps_fwd_gen).to(device).float()
        z_fwd = _integrate_q32_seg(
            t1.q32_flow, anc, ep, coarse_y16_lcn(y64b), db, qb, ti, bank,
            steps=6, full_steps={0, 1}, cfg_kw=hi_s,
        )
        c64_l.append(cb.cpu().half())
        q32_l.append(qb.cpu().half())
        d16_l.append(db.cpu().half())
        y64_l.append(y64b.cpu().half())
        fib_l.append(fb.cpu().half())
        fwd_l.append(z_fwd.cpu().half())

    del dino
    gc.collect()
    torch.cuda.empty_cache()

    # Causal Control 2: Derangement shuffle of kept_z_real within the same target style
    shuf_perm = torch.arange(len(kept_tid))
    for sid in (0, 4):
        m = (kept_tid == sid).nonzero().flatten()
        # Deterministic cyclic derangement after random perm
        p_gen = torch.Generator().manual_seed(SEED_PLAN + 304 + sid)
        order = m[torch.randperm(len(m), generator=p_gen)]
        shuf_perm[order] = order.roll(1)

    payload = {
        "recipe_hash": r_hash,
        "keep_per_style": keep_per_style,
        "real_idx": kept_real_idx,
        "tid": kept_tid,
        "z_syn": kept_z_syn,
        "c_64": torch.cat(c64_l, 0),
        "q_32": torch.cat(q32_l, 0),
        "d_16": torch.cat(d16_l, 0),
        "y_64": torch.cat(y64_l, 0),
        "fib_syn": torch.cat(fib_l, 0),
        "z_real": kept_z_real,
        "z_fwd": torch.cat(fwd_l, 0),
        "z_shuf": kept_z_real[shuf_perm],
        "gray_sim_err": kept_gsim,
        "lcn_err": kept_lcn,
    }
    atomic_save(payload, out_path)
    print(f"[T3] Saved Pool C -> {out_path} ({len(kept_tid)} pairs, "
          f"mean gray_sim_err={float(kept_gsim.mean()):.4f}, lcn_err={float(kept_lcn.mean()):.4f})", flush=True)
    return payload
