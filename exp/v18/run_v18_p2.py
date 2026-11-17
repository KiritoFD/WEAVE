"""WEAVE V18 Phase 2 Sequencer (4090 Acceleration: Track A Push, CCF-v2 Cross-Hybrid, Latency & PNG Export).

Usage:
  python -m exp.v18.run_v18_p2 --all --pool ../pseudo_gt_24h/orig_mean_fp16
"""
from __future__ import annotations

import argparse
import copy
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from . import EVAL_SEED, OUT, ROOT, STYLES, RawWriterV18, autocast, read_json, run_measure_process, save_json, seed_all, tasks_and_sources
from .ccf_model import CCFUNet, generate_ccf_arm_750, update_ema
from .ld4_plus import CalibratedLD4Plus, FastDINOStudent, run_ccf_plus_ld4
from .structure import (
    _train_linear_mlp_probe,
    augment_structure,
    local_contrast_norm,
    project_latent_lcn,
    rank_normalize_structure,
)
from .style_bank import build_or_load_ccf_cache

P2_ARMS = [
    "V18_A1c_LD4plus_e24",
    "V18_A1c_LD4plus_wrong",
    "V18_A1d_LD4plus_e30",
    "V18_A2b_FastStudent_pro",
    "V18_A2b_FastStudent_pro_wrong",
    "V18_B5_CCFv2_t0_w25",
    "V18_B6_CCFv2_t0_w35",
    "V18_B6_CCFv2_wrong",
    "V18_B7_CCFv2_t025_w30",
    "V18_C2_CCFv2_LD4plus",
]


def _pick_wrong_cross_tid(sid: torch.Tensor, tid: torch.Tensor) -> torch.Tensor:
    """Pick deterministic wrong target style != tid and != sid (for cross tasks)."""
    cand = (tid + 1) % 5
    clash = (cand == sid) & (sid != tid)
    return torch.where(clash, (tid + 2) % 5, cand)


def run_gate0b_audit(cache_dict: dict, device: str = "cuda") -> dict:
    from tools.v6.canonicalize import extract_band_stats_features

    tr, ho = cache_dict["train"], cache_dict["holdout"]
    with torch.no_grad():
        c_tr = rank_normalize_structure(tr["c_64"].cuda(), keep_coarse=True)
        c_ho = rank_normalize_structure(ho["c_64"].cuda(), keep_coarse=True)
        f_tr = extract_band_stats_features(c_tr).cpu()
        f_ho = extract_band_stats_features(c_ho).cpu()
    acc = _train_linear_mlp_probe(f_tr, tr["sid"], f_ho, ho["sid"])
    lcn_coarse_true = local_contrast_norm(ho["y_64"].float(), kernel=13, eps=0.04)
    corr = float(F.cosine_similarity(c_ho[:, 1:2].cpu().flatten(1), lcn_coarse_true.flatten(1), dim=1).mean().item())
    rep = {
        "c_rank_coarse_probe_5way": round(acc, 4),
        "c_rank_coarse_layout_cosine": round(corr, 4),
    }
    save_json(OUT / "gate0b_leakage_audit.json", rep)
    print("[V18 Gate0b] Rank-Normalized Coarse Structure Audit:", rep, flush=True)
    return rep


def run_track_a_p2(cache_dict: dict, test_struct: dict, device: str = "cuda"):
    from src.utils import decode_s10, load_s10_decoder
    from tools.v6.canonicalize import extract_band_stats_features

    op = CalibratedLD4Plus(device=device)
    op.calibrate_on_holdout(cache_dict["holdout"]["z"], cache_dict["holdout"]["sid"])
    tasks, z_src_750, n2c_cache = tasks_and_sources()
    z0_750 = n2c_cache["z0"].float()
    zn2_750 = n2c_cache["z_n2"].float()
    dec = load_s10_decoder(device)

    # 1. LD4+ push: A1c (e24 + wrong control) & A1d (e30)
    a1_configs = [
        ("V18_A1c_LD4plus_e24", False, 0.24, 0.28, 4, 0.45, 0.45, 0.70, 0.50),
        ("V18_A1c_LD4plus_wrong", True, 0.24, 0.28, 4, 0.45, 0.45, 0.70, 0.50),
        ("V18_A1d_LD4plus_e30", False, 0.30, 0.34, 4, 0.55, 0.50, 0.90, 0.40),
    ]
    for arm_tag, is_wrong, eps_v, rho_v, k_v, mfrac, alpha_ts, wr1, wlse in a1_configs:
        if (OUT / f"raw/{arm_tag}/meta.json").exists():
            continue
        wr = RawWriterV18(arm_tag)
        t0 = time.time()
        for st in range(0, 750, 16):
            sl = slice(st, min(st + 16, 750))
            grp = tasks[sl]
            sid = torch.tensor([t["source_style_id"] for t in grp], device=device)
            tid = torch.tensor([t["target_style_id"] for t in grp], device=device)
            zs = z_src_750[sl].to(device)
            if is_wrong:
                used_tid = _pick_wrong_cross_tid(sid, tid)
                _, z0, zn = op.run_b4_and_n2c(zs, used_tid)
            else:
                used_tid = tid
                z0 = z0_750[sl].to(device)
                zn = zn2_750[sl].to(device)
            z_out = op.refine(
                zn, z0, zs, sid, used_tid,
                eps=eps_v, rho=rho_v, k_steps=k_v, mask_frac=mfrac,
                src_push_alpha=alpha_ts, w_r1=wr1, w_lse=wlse,
            )
            with torch.no_grad():
                z_out[sid == tid] = zs[sid == tid]
                rgb = decode_s10(dec, z_out, batch=8)
            wr.append([t["task_id"] for t in grp], rgb, z_out)
        wr.finish(eps=eps_v, rho=rho_v, k_steps=k_v, wrong_label=is_wrong, wall_sec=round(time.time() - t0, 1))
        print(f"[V18 P2 TrackA] {arm_tag} generated in {time.time()-t0:.1f}s", flush=True)

    # 2. Build 5,000 clean balanced B4 -> LD4+ pairs (200 src/style * 5 targets) for BOTH A2b & CCF-v2
    pairs_path = OUT / "cache/b4_ld4plus_pairs_5k.pt"
    if not pairs_path.exists():
        t0 = time.time()
        tr = cache_dict["train"]
        g_p = torch.Generator().manual_seed(20261119)
        sel_idx = torch.cat([
            (tr["sid"] == s).nonzero(as_tuple=True)[0][torch.randperm(int((tr["sid"] == s).sum()), generator=g_p)[:200]]
            for s in range(5)
        ])
        z_s = tr["z"][sel_idx].repeat_interleave(5, dim=0)
        sid_p = tr["sid"][sel_idx].repeat_interleave(5, dim=0)
        tid_p = torch.arange(5).repeat(len(sel_idx))

        z_b4_l, c_can_l, delta_l, z_end_l = [], [], [], []
        for st in range(0, len(z_s), 50):
            sl = slice(st, min(st + 50, len(z_s)))
            zb = z_s[sl].to(device)
            sb = sid_p[sl].to(device)
            tb = tid_p[sl].to(device)
            c_can, z0_b, zn2_b = op.run_b4_and_n2c(zb, tb)
            z_end = op.refine(
                zn2_b, z0_b, zb, sb, tb,
                eps=0.22, rho=0.26, k_steps=4, mask_frac=0.45,
                src_push_alpha=0.45, w_r1=0.70, w_lse=0.50,
            )
            with torch.no_grad():
                z_end[sb == tb] = zb[sb == tb]
            z_b4_l.append(z0_b.cpu().half())
            c_can_l.append(c_can.cpu().half())
            delta_l.append((z_end - z0_b).cpu().half())
            z_end_l.append(z_end.cpu().half())
            if (st // 50) % 20 == 0:
                print(f"[V18 P2 Pairs5K] {st+len(zb)}/{len(z_s)} ({time.time()-t0:.0f}s)", flush=True)

        # Assign nearest training sub-mode prototype to each z_end for CCF-v2 supervision
        bank = cache_dict["style_bank"]
        f_mean = bank["feat_mean"].to(device)
        f_std = bank["feat_std"].to(device)
        centers = bank["submode_centers"].to(device)  # (5, K, 48)
        z_end_cat = torch.cat(z_end_l, dim=0)
        protos = []
        with torch.no_grad():
            for st in range(0, len(z_end_cat), 256):
                ze = z_end_cat[st:st + 256].to(device).float()
                tb = tid_p[st:st + 256].to(device)
                f = extract_band_stats_features(ze)
                fn = F.normalize((f - f_mean) / f_std, dim=-1)
                c_b = centers[tb]  # (B, K, 48)
                sim = torch.bmm(c_b, fn.unsqueeze(-1)).squeeze(-1)
                best_k = sim.argmax(dim=1)
                protos.append(c_b[torch.arange(len(tb), device=device), best_k].cpu().half())

        torch.save(
            {
                "sel_idx": sel_idx,
                "sid": sid_p,
                "tid": tid_p,
                "z_b4": torch.cat(z_b4_l, dim=0),
                "c_canon": torch.cat(c_can_l, dim=0),
                "delta": torch.cat(delta_l, dim=0),
                "z_end": z_end_cat,
                "end_proto": torch.cat(protos, dim=0),
            },
            pairs_path,
        )
        print(f"[V18 P2 Pairs5K] Saved 5,000 pairs in {time.time()-t0:.1f}s", flush=True)

    # 3. Train FastDINOStudent-Pro (width=112, blocks=5, 10,000 steps)
    student_ckpt = OUT / "a2_student/fast_student_pro.pt"
    if not student_ckpt.exists():
        student_ckpt.parent.mkdir(parents=True, exist_ok=True)
        pairs = torch.load(pairs_path, map_location="cpu", weights_only=False)
        tr = cache_dict["train"]
        sel_idx = pairs["sel_idx"]
        c64_src = tr["c_64"][sel_idx]
        dino_src = tr["dino_16"][sel_idx]

        seed_all(20261119)
        net = FastDINOStudent(width=112, blocks=5).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=0.01)
        updates = 10000
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=5e-4, total_steps=updates, pct_start=0.06)
        gen = torch.Generator().manual_seed(20261119)
        cross_idx = (pairs["sid"] != pairs["tid"]).nonzero(as_tuple=True)[0]
        t0 = time.time()
        net.train()
        for step in range(1, updates + 1):
            pick = cross_idx[torch.randint(len(cross_idx), (64,), generator=gen)]
            src_row = pick // 5
            zb4 = pairs["z_b4"][pick].to(device).float()
            cc = pairs["c_canon"][pick].to(device).float()
            dt_gt = pairs["delta"][pick].to(device).float()
            sb = pairs["sid"][pick].to(device).long()
            tb = pairs["tid"][pick].to(device).long()
            c64 = c64_src[src_row].to(device).float()
            d16 = dino_src[src_row].to(device).float()

            with autocast(device):
                pred = net(zb4, cc, c64, d16, sb, tb).float()
            mse = F.mse_loss(pred, dt_gt)
            valid = dt_gt.flatten(1).norm(dim=1) > 1e-4
            cos_loss = (1.0 - F.cosine_similarity(pred[valid].flatten(1), dt_gt[valid].flatten(1), dim=1)).mean() if valid.any() else 0.0
            lf_pen = F.avg_pool2d(pred - dt_gt, 4).square().mean()
            loss = mse + 0.15 * cos_loss + 0.10 * lf_pen
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            if step == 1 or step % 1000 == 0 or step == updates:
                with torch.no_grad():
                    cos_v = float(F.cosine_similarity(pred[valid].flatten(1), dt_gt[valid].flatten(1), dim=1).mean().item())
                    rms_r = float(pred.flatten(1).norm(dim=1).mean() / dt_gt.flatten(1).norm(dim=1).clamp_min(1e-6).mean())
                print(f"[V18 StudentPro] {step:5d}/{updates} mse={mse.item():.5f} cos={cos_v:.4f} rmsR={rms_r:.3f} {time.time()-t0:.0f}s", flush=True)

        torch.save({"model": net.cpu().state_dict(), "width": 112, "blocks": 5, "n_params": net.n_params()}, student_ckpt)
        del net, opt
        torch.cuda.empty_cache()

    st_ck = torch.load(student_ckpt, map_location="cpu", weights_only=False)
    student = FastDINOStudent(width=st_ck["width"], blocks=st_ck["blocks"]).to(device).eval()
    student.load_state_dict(st_ck["model"])
    for arm_tag, is_wrong in [("V18_A2b_FastStudent_pro", False), ("V18_A2b_FastStudent_pro_wrong", True)]:
        if (OUT / f"raw/{arm_tag}/meta.json").exists():
            continue
        wr = RawWriterV18(arm_tag)
        t0 = time.time()
        with torch.no_grad():
            for st in range(0, 750, 32):
                sl = slice(st, min(st + 32, 750))
                grp = tasks[sl]
                zs = z_src_750[sl].to(device)
                sid = torch.tensor([t["source_style_id"] for t in grp], device=device)
                tid = torch.tensor([t["target_style_id"] for t in grp], device=device)
                used_tid = _pick_wrong_cross_tid(sid, tid) if is_wrong else tid
                c_can = op.canon(zs, mode="c1_gentle", augment=False)
                z_b4 = op.b4(c_can, 0.0, c_can, used_tid)
                c64 = test_struct["c_64"][sl].to(device).float()
                d16 = test_struct["dino_16"][sl].to(device).float()
                with autocast(device):
                    delta = student(z_b4, c_can, c64, d16, sid, used_tid).float()
                z_final = torch.where((sid == tid).view(-1, 1, 1, 1), zs, z_b4 + delta)
                rgb = decode_s10(dec, z_final, batch=8)
                wr.append([t["task_id"] for t in grp], rgb, z_final)
        wr.finish(wrong_label=is_wrong, pure_feedforward=True, params=st_ck["n_params"], wall_sec=round(time.time() - t0, 1))
        print(f"[V18 StudentPro Eval] {arm_tag} in {time.time()-t0:.1f}s", flush=True)

    del student, op, dec
    torch.cuda.empty_cache()


def train_ccf_v2(cache_dict: dict, updates: int = 6000, micro: int = 128, lr: float = 3e-4, device: str = "cuda") -> Path:
    out_dir = OUT / "ccf_v2"
    ckpt_final = out_dir / "ccf_final.pt"
    if ckpt_final.exists():
        print(f"[V18 CCFv2] Reusing {ckpt_final}", flush=True)
        return ckpt_final

    out_dir.mkdir(parents=True, exist_ok=True)
    seed_all(20261119)

    tr = cache_dict["train"]
    bank = cache_dict["style_bank"]
    w_luma = cache_dict["luma_proj"]["w"].to(device).float()
    b_luma = cache_dict["luma_proj"]["b"].to(device).float()

    pairs = torch.load(OUT / "cache/b4_ld4plus_pairs_5k.pt", map_location="cpu", weights_only=False)
    sel_idx = pairs["sel_idx"]

    # Precompute rank-normalized coarse structure (strips 59.3% stroke/histogram leakage!)
    c_rank_tr = rank_normalize_structure(tr["c_64"].float(), keep_coarse=True).half()
    y_coarse_tr = local_contrast_norm(tr["y_64"].float(), kernel=13, eps=0.04).half()

    c_rank_src = c_rank_tr[sel_idx]
    dino_src = tr["dino_16"][sel_idx]
    y_coarse_src = y_coarse_tr[sel_idx]

    cross_mask = pairs["sid"] != pairs["tid"]
    cross_buckets = [((pairs["tid"] == s) & cross_mask).nonzero(as_tuple=True)[0] for s in range(5)]
    self_buckets = [(tr["sid"] == s).nonzero(as_tuple=True)[0] for s in range(5)]

    mean_protos = bank["mean_protos"].float()
    sample_protos = bank["sample_proto"].float()

    model = CCFUNet(base_dim=128).to(device)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=updates, pct_start=0.06)
    gen = torch.Generator().manual_seed(20261119)

    n_cross = int(round(micro * 0.55))
    n_self = micro - n_cross
    curves = []
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    model.train()

    for step in range(1, updates + 1):
        # 55% balanced cross-style pairs + 45% balanced clean self-paired paintings
        cls_c = torch.randperm(n_cross, generator=gen) % 5
        idx_c = torch.tensor([int(cross_buckets[s][int(torch.randint(len(cross_buckets[s]), (1,), generator=gen))]) for s in cls_c.tolist()])
        src_row = idx_c // 5

        cls_s = torch.randperm(n_self, generator=gen) % 5
        idx_s = torch.tensor([int(self_buckets[s][int(torch.randint(len(self_buckets[s]), (1,), generator=gen))]) for s in cls_s.tolist()])

        z_real = torch.cat([pairs["z_end"][idx_c].float(), tr["z"][idx_s].float()], dim=0).to(device)
        c_raw = torch.cat([c_rank_src[src_row].float(), c_rank_tr[idx_s].float()], dim=0).to(device)
        c_in = augment_structure(c_raw, gen=gen)
        d16 = torch.cat([dino_src[src_row].float(), tr["dino_16"][idx_s].float()], dim=0).to(device)
        tid = torch.cat([pairs["tid"][idx_c].long(), tr["sid"][idx_s].long()], dim=0).to(device)
        y_gt = torch.cat([y_coarse_src[src_row].float(), y_coarse_tr[idx_s].float()], dim=0).to(device)

        proto_raw = torch.cat([pairs["end_proto"][idx_c].float(), sample_protos[idx_s].float()], dim=0)
        u_mode = torch.rand(micro, generator=gen)
        proto = torch.where((u_mode < 0.60).unsqueeze(1), proto_raw, mean_protos[tid.cpu()]).to(device)
        drop_style = (u_mode >= 0.85).to(device)
        drop_content = (torch.rand(micro, generator=gen) < 0.10).to(device)

        tau = torch.rand(micro, generator=gen).to(device)
        eps = torch.randn(z_real.shape, generator=gen).to(device)
        ta = tau.view(-1, 1, 1, 1)
        z_tau = (1.0 - ta) * eps + ta * z_real
        v_star = z_real - eps

        with autocast(device):
            v_pred = model(z_tau, c_in, d16, tid, proto, tau, drop_style=drop_style, drop_content=drop_content).float()

        loss_fm = F.mse_loss(v_pred, v_star)
        valid_st = (tau >= 0.25) & (~drop_content)
        if valid_st.any():
            z1_hat = z_tau[valid_st] + (1.0 - ta[valid_st]) * v_pred[valid_st]
            lcn_hat = project_latent_lcn(z1_hat, w_luma, b_luma, kernel=13, eps=0.04)
            loss_st = F.l1_loss(lcn_hat, y_gt[valid_st])
        else:
            loss_st = torch.zeros((), device=device)

        loss = loss_fm + 0.08 * loss_st
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        if not math.isfinite(gn):
            raise FloatingPointError(f"Non-finite grad at CCFv2 step {step}")
        opt.step()
        sched.step()
        update_ema(ema, model, decay=min(0.998, (1.0 + step) / (10.0 + step)))

        if step == 1 or step % 300 == 0 or step == updates:
            rec = {
                "step": step,
                "loss_fm": round(float(loss_fm.item()), 5),
                "loss_struct": round(float(loss_st.item()), 5),
                "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                "sec": round(time.time() - t0, 1),
            }
            curves.append(rec)
            print(f"[V18 CCFv2] {step:4d}/{updates} fm={rec['loss_fm']:.4f} st={rec['loss_struct']:.4f} peak={rec['peak_gb']}GB {rec['sec']:.0f}s", flush=True)

        if step % 2000 == 0 or step == updates:
            save_json(out_dir / "train_curves.json", curves)

    torch.save(
        {
            "ema": ema.cpu().state_dict(),
            "model": model.cpu().state_dict(),
            "base_dim": 128,
            "updates": updates,
            "n_params": model.n_params(),
            "style_bank": {
                "mean_protos": bank["mean_protos"],
                "vivid_protos": bank["vivid_protos"],
                "submode_centers": bank["submode_centers"],
            },
        },
        ckpt_final,
    )
    del model, ema, opt
    torch.cuda.empty_cache()
    return ckpt_final


def run_ccfv2_gen(test_struct: dict, device: str = "cuda"):
    ckpt = OUT / "ccf_v2/ccf_final.pt"
    test_v2 = {
        "c_64": rank_normalize_structure(test_struct["c_64"].float(), keep_coarse=True).half(),
        "dino_16": test_struct["dino_16"],
    }
    for arm_name, ws, tau0, wrong in [
        ("V18_B5_CCFv2_t0_w25", 2.5, 0.0, False),
        ("V18_B6_CCFv2_t0_w35", 3.5, 0.0, False),
        ("V18_B6_CCFv2_wrong", 3.5, 0.0, True),
        ("V18_B7_CCFv2_t025_w30", 3.0, 0.25, False),
    ]:
        generate_ccf_arm_750(
            arm_name, ckpt, test_v2,
            w_style=ws, w_content=1.0, proto_mode="mean",
            wrong_label=wrong, steps=24, tau0=tau0, device=device,
        )


def run_bench_and_export(cache_dict: dict, test_struct: dict, device: str = "cuda"):
    from src.utils import decode_s10, load_s10_decoder

    dec = load_s10_decoder(device)
    op = CalibratedLD4Plus(device=device)
    op.calibrate_on_holdout(cache_dict["holdout"]["z"], cache_dict["holdout"]["sid"])

    def _bench(fn, n_warm=3, n_run=20, bs=1):
        for _ in range(n_warm):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_run):
            fn()
        torch.cuda.synchronize()
        return round((time.perf_counter() - t0) * 1000.0 / (n_run * bs), 3)

    lat = {"gpu": torch.cuda.get_device_name(0), "ms_per_img": {}}
    z1 = torch.randn(1, 4, 64, 64, device=device)
    c64_1 = test_struct["c_64"][:1].to(device).float()
    d16_1 = test_struct["dino_16"][:1].to(device).float()
    s1 = torch.tensor([0], device=device)
    t1 = torch.tensor([1], device=device)

    with torch.no_grad():
        lat["ms_per_img"]["s10_decode_b1"] = _bench(lambda: decode_s10(dec, z1, batch=1), bs=1)
        lat["ms_per_img"]["b4_plus_s10_b1"] = _bench(
            lambda: decode_s10(dec, op.b4(op.canon(z1, mode="c1_gentle", augment=False), 0.0, op.canon(z1, mode="c1_gentle", augment=False), t1), batch=1),
            bs=1,
        )

    for tag, ck_rel in [("A2_FastStudent_044M", "a2_student/fast_student.pt"), ("A2b_FastStudent_pro_105M", "a2_student/fast_student_pro.pt")]:
        ckp = OUT / ck_rel
        if ckp.exists():
            ck = torch.load(ckp, map_location="cpu", weights_only=False)
            stu = FastDINOStudent(width=ck["width"], blocks=ck["blocks"]).to(device).eval()
            stu.load_state_dict(ck["model"])
            with torch.no_grad():
                def _step_stu():
                    cc = op.canon(z1, mode="c1_gentle", augment=False)
                    zb4 = op.b4(cc, 0.0, cc, t1)
                    with autocast(device):
                        dt = stu(zb4, cc, c64_1, d16_1, s1, t1).float()
                    return decode_s10(dec, zb4 + dt, batch=1)
                lat["ms_per_img"][f"{tag}_b1"] = _bench(_step_stu, bs=1)
            del stu

    # Record wall-clock 750 inference latencies directly from meta.json
    for arm_dir in sorted((OUT / "raw").glob("*/meta.json")):
        m = read_json(arm_dir)
        if "wall_sec" in m:
            lat["ms_per_img"][f"750_wall_{arm_dir.parent.name}"] = round(m["wall_sec"] * 1000.0 / 750.0, 2)

    save_json(OUT / "latency_4090.json", lat)
    print("[V18 Bench] Latency summary:", lat["ms_per_img"], flush=True)

    # Export 750 PNGs for key flagship arms
    export_root = OUT / "png_export"
    for arm in ["V18_A1c_LD4plus_e24", "V18_A2b_FastStudent_pro", "V18_B6_CCFv2_t0_w35", "V18_C2_CCFv2_LD4plus"]:
        raw_dir = OUT / "raw" / arm
        meta_p = raw_dir / "meta.json"
        if not meta_p.exists():
            continue
        dst = export_root / arm
        if dst.exists() and len(list(dst.glob("*.png"))) == 750:
            continue
        dst.mkdir(parents=True, exist_ok=True)
        meta = read_json(meta_p)
        for b in meta["batches"]:
            bd = torch.load(raw_dir / b["file"], map_location="cpu", weights_only=False)
            u8 = bd["rgb_u8"]  # (B, 3, H, W)
            for idx, tid in enumerate(b["ids"]):
                arr = u8[idx].permute(1, 2, 0).numpy()
                Image.fromarray(arr).save(dst / f"{tid.replace('/', '__')}.png")
        print(f"[V18 Export] Saved 750 PNGs -> {dst}", flush=True)

    del dec, op
    torch.cuda.empty_cache()


def build_p2_report():
    from .run_v18 import ALL_EVAL_ARMS, paired_cluster_ci

    all_arms = list(ALL_EVAL_ARMS) + P2_ARMS
    summaries = {}
    for arm in all_arms:
        sp = OUT / "measured" / arm / "summary_v18.json"
        if sp.exists():
            summaries[arm] = read_json(sp)

    paired = {}
    for a, b in [
        ("V18_A1b_LD4plus_e18", "V18_A0_LD4_repro"),
        ("V18_A1c_LD4plus_e24", "V18_A1c_LD4plus_wrong"),
        ("V18_A1c_LD4plus_e24", "V18_A0_LD4_repro"),
        ("V18_A2_FastStudent", "V18_A2_FastStudent_wrong"),
        ("V18_A2b_FastStudent_pro", "V18_A2b_FastStudent_pro_wrong"),
        ("V18_B6_CCFv2_t0_w35", "V18_B6_CCFv2_wrong"),
        ("V18_C2_CCFv2_LD4plus", "V18_B6_CCFv2_t0_w35"),
    ]:
        if a in summaries and b in summaries:
            paired[f"{a}__vs__{b}"] = paired_cluster_ci(a, b)

    save_json(OUT / "v18_p2_master_summary.json", {"summaries": summaries, "paired_ci95": paired})
    lines = [
        "# WEAVE V18 (Phase 1 + Phase 2) 实测总表",
        "",
        "| 实验臂 | Cross600 Top-1 ↑ | Margin ↑ | DINO-S ↑ | Affinity ↑ | DINO-C ↑ | LPIPS ↓ | 源锁定率 ↓ | ER / Imp / Min / Roc / Uki (%) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for arm, s in summaries.items():
        c = s["cross600"]
        pt = s["per_target_cross120"]
        p_str = " / ".join(f"{pt[st]['csd_top1']*100:.1f}" for st in STYLES)
        lines.append(
            f"| {arm} | **{c['csd_top1']*100:.2f}%** | {c['csd_margin']:+.4f} | "
            f"{c['dino_s']:.4f} | {c['style_affinity']:.4f} | {c['dino_c']:.4f} | "
            f"{c['lpips']:.4f} | {c['pred_eq_src']*100:.1f}% | {p_str} |"
        )
    (OUT / "V18_RESULTS_AUTO.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("V18_P2_REPORT_DONE", flush=True)


def run_stage(stage: str, pool_dir: str, ccf_updates: int = 6000, ccf_micro: int = 128, device: str = "cuda"):
    if stage == "gate0b_leakage":
        cache_dict, _ = build_or_load_ccf_cache(pool_dir, device=device)
        run_gate0b_audit(cache_dict, device=device)
    elif stage == "track_a_p2_gen":
        cache_dict, test_struct = build_or_load_ccf_cache(pool_dir, device=device)
        run_track_a_p2(cache_dict, test_struct, device=device)
    elif stage == "ccfv2_train":
        cache_dict, _ = build_or_load_ccf_cache(pool_dir, device=device)
        train_ccf_v2(cache_dict, updates=ccf_updates, micro=ccf_micro, device=device)
    elif stage == "ccfv2_gen":
        _, test_struct = build_or_load_ccf_cache(pool_dir, device=device)
        run_ccfv2_gen(test_struct, device=device)
    elif stage == "track_c2_gen":
        cache_dict, _ = build_or_load_ccf_cache(pool_dir, device=device)
        run_ccf_plus_ld4("V18_B6_CCFv2_t0_w35", "V18_C2_CCFv2_LD4plus", cache_dict, device=device)
    elif stage == "bench_and_export":
        cache_dict, test_struct = build_or_load_ccf_cache(pool_dir, device=device)
        run_bench_and_export(cache_dict, test_struct, device=device)
    elif stage.startswith("measure:"):
        run_measure_process(stage.split(":", 1)[1], device=device)
    elif stage == "report_p2":
        build_p2_report()
    else:
        raise ValueError(f"Unknown P2 stage: {stage}")


def run_all_p2(pool_dir: str, ccf_updates: int = 6000, ccf_micro: int = 128):
    OUT.mkdir(parents=True, exist_ok=True)
    status_path = OUT / "pipeline_status_p2.json"
    status = read_json(status_path) if status_path.exists() else {}

    stages = [
        "gate0b_leakage",
        "track_a_p2_gen",
        "measure:V18_A1c_LD4plus_e24",
        "measure:V18_A1c_LD4plus_wrong",
        "measure:V18_A1d_LD4plus_e30",
        "measure:V18_A2b_FastStudent_pro",
        "measure:V18_A2b_FastStudent_pro_wrong",
        "ccfv2_train",
        "ccfv2_gen",
        "measure:V18_B5_CCFv2_t0_w25",
        "measure:V18_B6_CCFv2_t0_w35",
        "measure:V18_B6_CCFv2_wrong",
        "measure:V18_B7_CCFv2_t025_w30",
        "track_c2_gen",
        "measure:V18_C2_CCFv2_LD4plus",
        "bench_and_export",
        "report_p2",
    ]
    t_pipe = time.time()
    for stg in stages:
        if status.get(stg, {}).get("state") == "done":
            print(f"[V18 P2] SKIP {stg}", flush=True)
            continue
        print(f"\n[V18 P2] === START {stg} ===", flush=True)
        t0 = time.time()
        status[stg] = {"state": "running"}
        save_json(status_path, status)
        cmd = [
            sys.executable, "-u", "-m", "exp.v18.run_v18_p2",
            "--stage", stg, "--pool", pool_dir,
            "--ccf-updates", str(ccf_updates), "--ccf-micro", str(ccf_micro),
        ]
        r = subprocess.run(cmd, cwd=ROOT)
        elapsed = round((time.time() - t0) / 60.0, 2)
        if r.returncode != 0:
            status[stg] = {"state": f"failed:{r.returncode}", "minutes": elapsed}
            save_json(status_path, status)
            raise RuntimeError(f"P2 stage {stg} failed with rc={r.returncode}")
        status[stg] = {"state": "done", "minutes": elapsed}
        save_json(status_path, status)
        print(f"[V18 P2] === DONE {stg} ({elapsed:.2f} min, total {(time.time()-t_pipe)/60:.1f} min) ===", flush=True)
    print("V18_P2_PIPELINE_COMPLETE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--ccf-updates", type=int, default=6000)
    ap.add_argument("--ccf-micro", type=int, default=128)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--stage", default="")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.stage:
        run_stage(a.stage, a.pool, ccf_updates=a.ccf_updates, ccf_micro=a.ccf_micro, device=a.device)
    elif a.all:
        run_all_p2(a.pool, ccf_updates=a.ccf_updates, ccf_micro=a.ccf_micro)
    else:
        ap.error("Specify --all or --stage <name>")


if __name__ == "__main__":
    main()
