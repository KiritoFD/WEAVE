"""Phase 2: Atomic 256-sharded construction of Pool A (T1+T4), Pool B (T2), and Pool C (T3) + 300-sample audit."""
from __future__ import annotations

import argparse
import gc
import time

import torch

from exp.v20.slf_flow import coarse_y16_lcn
from exp.v20.submode_patchswd import build_or_load_core_knn_bank
from exp.v21.stiefel_ref_cache import build_or_load_v21_cache
from exp.v27_mainline.part_bank import build_or_load_part_bank
from exp.v27_single.unified.teacher_gate import task_plan
from exp.v28_unified import OUT, ROOT, SEED_PLAN, assert_vram_budget, atomic_save, recipe_hash
from exp.v28_unified.data_map import load_rgb_512_tensor, resolve_train_rgb_paths
from exp.v28_unified.phase1_race import evaluate_150_referee
from exp.v28_unified.referee_free import RefereeFreeSelector
from exp.v28_unified.t3_reverse import build_t3_pool
from exp.v28_unified.teacher_routes import (
    T1MultiExpertTeacher,
    T2ZeroLegacyOptimizer,
    T4RegionalGapFiller,
)
from src.utils import load_s10_decoder
from src.utils.runtime import sanitize_cuda_env, save_json, seed_all, utc


def build_pool_A(
    bundle: dict,
    c21: dict,
    bank: dict,
    pbank_dict: dict,
    paths_4400: list,
    dec_bf16,
    *,
    shard_size: int = 256,
    device: str = "cuda",
) -> dict:
    """Build 7,500-pair Pool A (T1 3-tier + T4 gap-fill on ER/Uki failures + seeded random baseline for T_rand)."""
    out_p = OUT / "pools/pool_A.pt"
    r_hash = recipe_hash({"pool": "A_v1", "experts": T1MultiExpertTeacher.CANDIDATE_NAMES, "seed": SEED_PLAN})
    if out_p.exists():
        ck = torch.load(out_p, map_location="cpu", weights_only=False)
        if ck.get("recipe_hash") == r_hash and len(ck["src"]) == 7500:
            print(f"[Phase 2 Pool A] Reusing {out_p}", flush=True)
            return ck

    tr = bundle["v18_cache"]["train"]
    selected, _, _ = task_plan(tr)
    src = selected.repeat_interleave(5)
    tid = torch.arange(5).repeat(len(selected))
    same = tr["sid"][src] == tid
    cross_ids = (~same).nonzero().flatten()

    selector = RefereeFreeSelector(bundle, bank, dec_bf16, device=device)
    t1 = T1MultiExpertTeacher(bundle, c21, bank, dec_bf16, device=device)
    t4 = T4RegionalGapFiller(bundle, bank, dec_bf16, pbank_dict, device=device)

    shard_dir = OUT / "pools/shards_pool_A"
    shard_dir.mkdir(parents=True, exist_ok=True)
    y16_all = coarse_y16_lcn(tr["y_64"].float())
    shards = []
    t0 = time.monotonic()

    for sh_i, ids in enumerate(cross_ids.split(shard_size)):
        sp = shard_dir / f"shard_{sh_i:03d}.pt"
        if sp.exists():
            pl = torch.load(sp, map_location="cpu", weights_only=False)
            if pl.get("recipe_hash") == r_hash and torch.equal(pl["ids"], ids):
                shards.append(pl)
                continue

        gen = torch.Generator().manual_seed(SEED_PLAN + 1000 + sh_i)
        eps = torch.randn((len(ids), 4, 64, 64), generator=gen).half()
        s_idx = src[ids]
        t_idx = tid[ids]

        cands = t1.generate_candidates(
            tr["z"][s_idx],
            y16_all[s_idx],
            tr["dino_16"][s_idx],
            c21["train"]["q32"][s_idx],
            tr["c_64"][s_idx],
            bundle["fnd"]["tr_fib"][s_idx],
            t_idx,
            eps,
            src_idx=s_idx,
            batch=32,
        )

        # Deterministic random candidate choice per task for Phase 4 T_rand control
        r_gen = torch.Generator().manual_seed(SEED_PLAN + 2000 + sh_i)
        r_pick_hic = torch.randint(0, len(cands), (len(ids),), generator=r_gen)
        r_pick_mid = torch.randint(0, len(cands), (len(ids),), generator=r_gen)
        r_pick_his = torch.randint(0, len(cands), (len(ids),), generator=r_gen)
        cand_stack = torch.stack(cands, dim=1)  # (N, 9, 4, 64, 64)
        ar = torch.arange(len(ids))
        rand_hic = cand_stack[ar, r_pick_hic]
        rand_mid = cand_stack[ar, r_pick_mid]
        rand_his = cand_stack[ar, r_pick_his]

        rgb_src = torch.cat([load_rgb_512_tensor([paths_4400[int(k)] for k in sl.tolist()]) for sl in s_idx.split(32)], 0)
        sel_out = {"hiC": [], "mid": [], "hiS": [], "hiS_gate_pass": []}
        for st in range(0, len(ids), 32):
            sl = slice(st, min(st + 32, len(ids)))
            res = selector.select_tiers(
                tr["z"][s_idx[sl]], rgb_src[sl].to(device), t_idx[sl].to(device), [c[sl] for c in cands]
            )
            for k in sel_out:
                sel_out[k].append(res[k])
        sel_out = {k: torch.cat(v, 0) for k, v in sel_out.items()}

        # T4 trigger on ER/Uki tasks failing hiS structural gate
        is_hard = (t_idx == 0) | (t_idx == 4)
        trig = is_hard & (~sel_out["hiS_gate_pass"])
        trig_local = trig.nonzero().flatten()
        n_unsolved = 0
        if len(trig_local) > 0:
            best_c_tr = selector.match_gray_exemplar(rgb_src[trig_local].to(device), t_idx[trig_local].to(device))
            z_ref = t4.refine_batch(
                sel_out["hiS"][trig_local],
                tr["z"][s_idx[trig_local]],
                rgb_src[trig_local],
                c21["train"]["q32"][s_idx[trig_local]],
                c21["train"]["q32"],
                t_idx[trig_local].to(device),
                best_c_tr,
                steps=15,
            )
            recheck = selector.select_tiers(
                tr["z"][s_idx[trig_local]], rgb_src[trig_local].to(device), t_idx[trig_local].to(device),
                [sel_out["hiS"][trig_local], z_ref],
            )
            sel_out["hiS"][trig_local] = recheck["hiS"]
            n_unsolved = int((~recheck["hiS_gate_pass"]).sum())

        pl = {
            "recipe_hash": r_hash,
            "ids": ids,
            "hiC": sel_out["hiC"],
            "mid": sel_out["mid"],
            "hiS": sel_out["hiS"],
            "rand_hiC": rand_hic,
            "rand_mid": rand_mid,
            "rand_hiS": rand_his,
            "t4_triggered": int(len(trig_local)),
            "t4_hard_total": int(is_hard.sum()),
            "t4_unsolved": n_unsolved,
        }
        atomic_save(pl, sp)
        shards.append(pl)
        print(f"[Pool A] shard {sh_i+1}: {min((sh_i+1)*shard_size, len(cross_ids))}/{len(cross_ids)} "
              f"(t4_trig={pl['t4_triggered']}, elapsed={time.monotonic()-t0:.0f}s)", flush=True)

    fields = {k: tr["z"][src].half().clone() for k in ("hiC", "mid", "hiS", "rand_hiC", "rand_mid", "rand_hiS")}
    t4_trig_sum, t4_hard_sum, t4_unsolved_sum = 0, 0, 0
    for pl in shards:
        ids = pl["ids"]
        for k in fields:
            fields[k][ids] = pl[k]
        t4_trig_sum += pl["t4_triggered"]
        t4_hard_sum += pl["t4_hard_total"]
        t4_unsolved_sum += pl["t4_unsolved"]

    pool_a = {
        "recipe_hash": r_hash,
        "src": src,
        "tid": tid,
        "same": same,
        "t4_stats": {
            "triggered": t4_trig_sum,
            "hard_total": t4_hard_sum,
            "unsolved": t4_unsolved_sum,
            "unsolved_rate": round(t4_unsolved_sum / max(1, t4_hard_sum), 4),
        },
        **fields,
    }
    atomic_save(pool_a, out_p)
    del t1, t4, selector
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[Pool A] COMPLETE -> {out_p} (t4_stats={pool_a['t4_stats']})", flush=True)
    return pool_a


def build_pool_B(
    bundle: dict,
    bank: dict,
    paths_4400: list,
    dec_bf16,
    *,
    shard_size: int = 256,
    device: str = "cuda",
) -> dict:
    """Build 7,500-pair Pool B using zero-legacy T2 optimizer (3/6/10 steps)."""
    out_p = OUT / "pools/pool_B.pt"
    r_hash = recipe_hash({"pool": "B_v1", "steps": [3, 6, 10], "seed": SEED_PLAN})
    if out_p.exists():
        ck = torch.load(out_p, map_location="cpu", weights_only=False)
        if ck.get("recipe_hash") == r_hash and len(ck["src"]) == 7500:
            print(f"[Phase 2 Pool B] Reusing {out_p}", flush=True)
            return ck

    tr = bundle["v18_cache"]["train"]
    selected, _, _ = task_plan(tr)
    src = selected.repeat_interleave(5)
    tid = torch.arange(5).repeat(len(selected))
    same = tr["sid"][src] == tid
    cross_ids = (~same).nonzero().flatten()

    selector = RefereeFreeSelector(bundle, bank, dec_bf16, device=device)
    t2 = T2ZeroLegacyOptimizer(bundle, selector, dec_bf16, device=device)
    shard_dir = OUT / "pools/shards_pool_B"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shards = []
    t0 = time.monotonic()

    for sh_i, ids in enumerate(cross_ids.split(shard_size)):
        sp = shard_dir / f"shard_{sh_i:03d}.pt"
        if sp.exists():
            pl = torch.load(sp, map_location="cpu", weights_only=False)
            if pl.get("recipe_hash") == r_hash and torch.equal(pl["ids"], ids):
                shards.append(pl)
                continue
        s_idx = src[ids]
        t_idx = tid[ids]
        rgb_src = torch.cat([load_rgb_512_tensor([paths_4400[int(k)] for k in sl.tolist()]) for sl in s_idx.split(32)], 0)
        snaps = {3: [], 6: [], 10: []}
        for st in range(0, len(ids), 32):
            sl = slice(st, min(st + 32, len(ids)))
            sn = t2.optimize_batch(tr["z"][s_idx[sl]], rgb_src[sl], t_idx[sl], total_steps=10, snap_steps=(3, 6, 10))
            for k in (3, 6, 10):
                snaps[k].append(sn[k])
        pl = {
            "recipe_hash": r_hash,
            "ids": ids,
            "hiC": torch.cat(snaps[3], 0),
            "mid": torch.cat(snaps[6], 0),
            "hiS": torch.cat(snaps[10], 0),
        }
        atomic_save(pl, sp)
        shards.append(pl)
        print(f"[Pool B] shard {sh_i+1}: {min((sh_i+1)*shard_size, len(cross_ids))}/{len(cross_ids)} "
              f"(elapsed={time.monotonic()-t0:.0f}s)", flush=True)

    fields = {k: tr["z"][src].half().clone() for k in ("hiC", "mid", "hiS")}
    for pl in shards:
        for k in fields:
            fields[k][pl["ids"]] = pl[k]

    pool_b = {"recipe_hash": r_hash, "src": src, "tid": tid, "same": same, **fields}
    atomic_save(pool_b, out_p)
    del t2, selector
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[Pool B] COMPLETE -> {out_p}", flush=True)
    return pool_b


def audit_pools_300(pool_a: dict, pool_b: dict, pool_c: dict, paths_4400: list, device: str = "cuda") -> dict:
    """Audit 300 balanced cross samples (disjoint from Phase 1 150 probe) across Pool A, B, and C."""
    dec_fp16 = load_s10_decoder(device)
    gen = torch.Generator().manual_seed(SEED_PLAN + 999)

    cross_mask = ~pool_a["same"]
    # Pick 60 per target style = 300 cross samples
    picks = []
    for s in range(5):
        cand = (cross_mask & (pool_a["tid"] == s)).nonzero().flatten()
        perm = cand[torch.randperm(len(cand), generator=gen)[:60]]
        picks.append(perm)
    idx_300 = torch.cat(picks, dim=0)

    rep_ab = evaluate_150_referee(
        dec_fp16, paths_4400, pool_a["src"][idx_300], pool_a["tid"][idx_300],
        {
            "PoolA_hiC": pool_a["hiC"][idx_300],
            "PoolA_mid": pool_a["mid"][idx_300],
            "PoolA_hiS": pool_a["hiS"][idx_300],
            "PoolA_rand_hiS": pool_a["rand_hiS"][idx_300],
            "PoolB_hiC": pool_b["hiC"][idx_300],
            "PoolB_mid": pool_b["mid"][idx_300],
            "PoolB_hiS": pool_b["hiS"][idx_300],
        },
        device=device,
    )

    # 300 samples (150 ER + 150 Uki) from Pool C
    c_picks = []
    for sid in (0, 4):
        m = (pool_c["tid"] == sid).nonzero().flatten()
        c_picks.append(m[torch.randperm(len(m), generator=gen)[:150]])
    c_idx = torch.cat(c_picks, dim=0)
    rep_c = evaluate_150_referee(
        dec_fp16, paths_4400, pool_c["real_idx"][c_idx], pool_c["tid"][c_idx],
        {
            "PoolC_real_GT": pool_c["z_real"][c_idx],
            "PoolC_fwd_teacher": pool_c["z_fwd"][c_idx],
            "PoolC_shuf_real": pool_c["z_shuf"][c_idx],
        },
        device=device,
    )
    del dec_fp16
    gc.collect()
    torch.cuda.empty_cache()
    audit = {"updated_utc": utc(), "n_audit": 300, "pool_ab": rep_ab, "pool_c": rep_c, "t4_stats": pool_a["t4_stats"]}
    save_json(OUT / "phase2_pool_audit_300.json", audit)
    return audit


def run_phase2(pool_dir: str, device: str = "cuda", only: str = "all") -> dict:
    sanitize_cuda_env()
    seed_all(SEED_PLAN + 1)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()

    pack = build_or_load_v21_cache(pool_dir, device=device)
    bundle, c21 = pack["bundle"], pack["c21"]
    bank = build_or_load_core_knn_bank(bundle, k_sub=8, core_per_style=180, device=device)
    pbank_dict = build_or_load_part_bank(pool_dir, device=device)

    tr = bundle["v18_cache"]["train"]
    paths_4400 = resolve_train_rgb_paths(tr["ids"], tr["sid"], require_full=True)

    dec_bf16 = load_s10_decoder(device).to(torch.bfloat16).eval()
    for p in dec_bf16.parameters():
        p.requires_grad_(False)

    targets = set(only.split(",")) if only != "all" else {"A", "B", "C"}
    pool_a = build_pool_A(bundle, c21, bank, pbank_dict, paths_4400, dec_bf16, device=device) if "A" in targets else torch.load(OUT / "pools/pool_A.pt", map_location="cpu", weights_only=False)
    pool_b = build_pool_B(bundle, bank, paths_4400, dec_bf16, device=device) if "B" in targets else torch.load(OUT / "pools/pool_B.pt", map_location="cpu", weights_only=False)

    if "C" in targets:
        selector = RefereeFreeSelector(bundle, bank, dec_bf16, device=device)
        t1 = T1MultiExpertTeacher(bundle, c21, bank, dec_bf16, device=device)
        pool_c = build_t3_pool(bundle, c21, bank, t1, selector, dec_bf16,
                               per_style_pool=880, keep_per_style=600,
                               out_path=OUT / "pools/pool_C.pt", device=device)
        del selector, t1
    else:
        pool_c = torch.load(OUT / "pools/pool_C.pt", map_location="cpu", weights_only=False)

    del dec_bf16
    gc.collect()
    torch.cuda.empty_cache()

    audit = audit_pools_300(pool_a, pool_b, pool_c, paths_4400, device=device)
    vram_gb = assert_vram_budget(device=device, tag="phase2")
    audit["elapsed_s"] = round(time.monotonic() - t0, 1)
    audit["peak_vram_gb"] = round(vram_gb, 3)
    save_json(OUT / "phase2_pool_audit_300.json", audit)
    return audit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--only", default="all")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    run_phase2(args.pool, device=args.device, only=args.only)


if __name__ == "__main__":
    main()
