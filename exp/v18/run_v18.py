"""WEAVE V18 Master Sequencer (Strict Process-per-Stage VRAM Isolation).

Usage on 3060:
  python -m exp.v18.run_v18 --canary --pool ../pseudo_gt_24h/orig_mean_fp16
  python -m exp.v18.run_v18 --all --pool ../pseudo_gt_24h/orig_mean_fp16
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import OUT, ROOT, STYLES, read_json, run_measure_process, save_json, utc

DEFAULT_POOL = str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16")

CCF_ARMS = [
    ("V18_B1_CCF_w20", 2.0, 1.0, "mean", False),
    ("V18_B2_CCF_w35", 3.5, 1.0, "mean", False),
    ("V18_B3_CCF_submode_w30", 3.0, 1.0, "vivid", False),
    ("V18_B3_CCF_submode_wrong", 3.0, 1.0, "vivid", True),
    ("V18_B4_CCF_dual_w30_c15", 3.0, 1.5, "vivid", False),
]

ALL_EVAL_ARMS = [
    "V18_A0_LD4_repro",
    "V18_A1_LD4plus_e10",
    "V18_A1b_LD4plus_e18",
    "V18_A2_FastStudent",
    "V18_A2_FastStudent_wrong",
    "V18_B1_CCF_w20",
    "V18_B2_CCF_w35",
    "V18_B3_CCF_submode_w30",
    "V18_B3_CCF_submode_wrong",
    "V18_B4_CCF_dual_w30_c15",
    "V18_C1_CCF_LD4plus",
]


def run_canary(pool_dir: str, device: str = "cuda"):
    """Fast end-to-end verification of all V18 modules, gradients, identities, and VRAM."""
    from .ccf_model import CCFUNet, sample_ccf_batch
    from .ld4_plus import CalibratedLD4Plus, FastDINOStudent
    from .structure import augment_structure, extract_whitened_dino_layout, rgb_to_structure_64
    from src.utils import decode_s10, load_s10_decoder
    from src.utils._local_eval_refs import load_dino_model

    print("[V18 Canary] 1/4 Testing S10 + Structure + Whitened DINO...", flush=True)
    dec = load_s10_decoder(device)
    dino = load_dino_model(device=device)
    z_dummy = (torch.randn(2, 4, 64, 64, device=device) * 0.5)
    rgb = decode_s10(dec, z_dummy, batch=2).to(device).clamp(0.0, 1.0)
    c64 = rgb_to_structure_64(rgb)
    d16 = extract_whitened_dino_layout(dino, rgb, batch=2, device=device).to(device)
    assert c64.shape == (2, 4, 64, 64) and torch.isfinite(c64).all()
    assert d16.shape == (2, 384, 16, 16) and torch.isfinite(d16).all()
    assert float(d16.mean(dim=(2, 3)).abs().max()) < 1e-4, "Whitened DINO global mean must be 0!"
    del dec, dino
    torch.cuda.empty_cache()

    print("[V18 Canary] 2/4 Testing CCFUNet forward/backward + Dual-CFG sample...", flush=True)
    ccf = CCFUNet(base_dim=128).to(device)
    sid = torch.tensor([0, 4], device=device)
    proto = torch.randn(2, 48, device=device)
    tau = torch.tensor([0.2, 0.7], device=device)
    c64_aug = augment_structure(c64)
    v = ccf(z_dummy, c64_aug, d16, sid, proto, tau)
    loss = v.square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in ccf.parameters() if p.requires_grad)
    with torch.no_grad():
        z_samp = sample_ccf_batch(ccf.eval(), c64, d16, sid, proto, torch.randn_like(z_dummy), steps=4, w_style=3.0, w_content=1.5)
    assert z_samp.shape == (2, 4, 64, 64) and torch.isfinite(z_samp).all()
    print(f"  CCFUNet params={ccf.n_params()/1e6:.2f}M peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)
    del ccf
    torch.cuda.empty_cache()

    print("[V18 Canary] 3/4 Testing FastDINOStudent zero-init & same-style exact identity...", flush=True)
    stu = FastDINOStudent(width=80, blocks=4).to(device)
    d0 = stu(z_dummy, z_dummy, c64, d16, torch.tensor([0, 2], device=device), torch.tensor([1, 2], device=device))
    assert float(d0.abs().max()) == 0.0, "Zero-init identity failed!"
    # Perturb out conv weights to verify same-style mask still enforces exact 0 on sample 1 (sid==tid==2)
    with torch.no_grad():
        stu.out.weight.normal_(0, 0.01)
    d1 = stu(z_dummy, z_dummy, c64, d16, torch.tensor([0, 2], device=device), torch.tensor([1, 2], device=device))
    assert float(d1[0].abs().max()) > 0.0 and float(d1[1].abs().max()) == 0.0, "Same-style exact zero failed!"
    d1.square().mean().backward()
    del stu
    torch.cuda.empty_cache()

    print("[V18 Canary] 4/4 Testing CalibratedLD4Plus 1-step refine...", flush=True)
    op = CalibratedLD4Plus(device=device)
    c_can, z0_b, zn2_b = op.run_b4_and_n2c(z_dummy, torch.tensor([1, 4], device=device))
    z_ref = op.refine(zn2_b, z0_b, z_dummy, torch.tensor([0, 4], device=device), torch.tensor([1, 4], device=device), k_steps=1)
    assert torch.isfinite(z_ref).all() and torch.equal(z_ref[1], zn2_b[1])
    del op
    torch.cuda.empty_cache()
    print("V18_CANARY_ALL_OK", flush=True)


def run_stage(stage: str, pool_dir: str, ccf_updates: int = 8000,
              ccf_micro: int = 64, ccf_accum: int = 2, device: str = "cuda"):
    from .ccf_model import generate_ccf_arm_750, train_ccf
    from .ld4_plus import run_ccf_plus_ld4, run_track_a_and_amortize
    from .structure import audit_leakage
    from .style_bank import build_or_load_ccf_cache

    if stage == "cache_and_gate0":
        cache_dict, _ = build_or_load_ccf_cache(pool_dir, device=device)
        audit_leakage(cache_dict)
    elif stage == "track_a_gen":
        cache_dict, test_struct = build_or_load_ccf_cache(pool_dir, device=device)
        run_track_a_and_amortize(cache_dict, test_struct, device=device)
    elif stage == "ccf_train":
        cache_dict, _ = build_or_load_ccf_cache(pool_dir, device=device)
        train_ccf(cache_dict, updates=ccf_updates, micro=ccf_micro, accum=ccf_accum, device=device)
    elif stage == "ccf_gen":
        _, test_struct = build_or_load_ccf_cache(pool_dir, device=device)
        ckpt = OUT / "ccf_main/ccf_final.pt"
        for arm_name, ws, wc, pmode, is_wrong in CCF_ARMS:
            generate_ccf_arm_750(
                arm_name, ckpt, test_struct,
                w_style=ws, w_content=wc, proto_mode=pmode,
                wrong_label=is_wrong, device=device,
            )
    elif stage == "track_c_gen":
        cache_dict, _ = build_or_load_ccf_cache(pool_dir, device=device)
        run_ccf_plus_ld4("V18_B3_CCF_submode_w30", "V18_C1_CCF_LD4plus", cache_dict, device=device)
    elif stage.startswith("measure:"):
        arm_name = stage.split(":", 1)[1]
        run_measure_process(arm_name, device=device)
    elif stage == "report":
        build_v18_report()
    else:
        raise ValueError(f"Unknown stage: {stage}")


def paired_cluster_ci(arm_a: str, arm_b: str) -> dict:
    da = read_json(OUT / "measured" / arm_a / "eval.json")["rows"]
    db = {r["task_id"]: r for r in read_json(OUT / "measured" / arm_b / "eval.json")["rows"]}
    rows = [r for r in da if r["is_cross"]]
    stems = sorted({r["src_stem"] for r in rows})
    metrics = ["csd_top1", "target_margin", "dino_s", "dino_c", "lpips"]
    key_map = {"csd_top1": "target_top1"}
    mat = np.array([
        [
            sum(
                float(r[key_map.get(m, m)]) - float(db[r["task_id"]][key_map.get(m, m)])
                for r in rows if r["src_stem"] == s
            ) / 4.0
            for m in metrics
        ]
        for s in stems
    ])
    counts = np.random.default_rng(20261118).multinomial(len(stems), np.ones(len(stems)) / len(stems), size=2000)
    dist = counts @ mat / len(stems)
    return {
        m: {
            "delta": round(float(mat[:, i].mean()), 5),
            "ci95": [round(float(x), 5) for x in np.quantile(dist[:, i], [0.025, 0.975])],
        }
        for i, m in enumerate(metrics)
    }


def build_v18_report():
    summaries = {}
    for arm in ALL_EVAL_ARMS:
        sp = OUT / "measured" / arm / "summary_v18.json"
        if sp.exists():
            summaries[arm] = read_json(sp)

    paired = {}
    for a, b in [
        ("V18_A1_LD4plus_e10", "V18_A0_LD4_repro"),
        ("V18_A1b_LD4plus_e18", "V18_A0_LD4_repro"),
        ("V18_A2_FastStudent", "V18_A2_FastStudent_wrong"),
        ("V18_B3_CCF_submode_w30", "V18_B3_CCF_submode_wrong"),
        ("V18_C1_CCF_LD4plus", "V18_B3_CCF_submode_w30"),
    ]:
        if a in summaries and b in summaries:
            paired[f"{a}__vs__{b}"] = paired_cluster_ci(a, b)

    gate0 = read_json(OUT / "gate0_leakage_audit.json") if (OUT / "gate0_leakage_audit.json").exists() else None
    master = {"utc": utc(), "gate0": gate0, "summaries": summaries, "paired_ci95": paired}
    save_json(OUT / "v18_master_summary.json", master)

    lines = [
        "# WEAVE V18 突破实验实测结果汇总",
        "",
        "| 实验臂 | Cross600 Top-1 ↑ | Margin ↑ | DINO-S ↑ | Affinity ↑ | DINO-C ↑ | LPIPS ↓ | 源锁定率 ↓ | ER / Imp / Min / Roc / Uki Top-1 (%) |",
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
    print("V18_REPORT_DONE", flush=True)


def run_all_subprocess_pipeline(pool_dir: str, ccf_updates: int = 5000,
                                  ccf_micro: int = 64, ccf_accum: int = 2,
                                  skip_track_a: bool = False):
    OUT.mkdir(parents=True, exist_ok=True)
    status_path = OUT / "pipeline_status.json"
    status = read_json(status_path) if status_path.exists() else {}

    stages = ["cache_and_gate0"]
    if not skip_track_a:
        stages += [
            "track_a_gen",
            "measure:V18_A0_LD4_repro",
            "measure:V18_A1_LD4plus_e10",
            "measure:V18_A1b_LD4plus_e18",
            "measure:V18_A2_FastStudent",
            "measure:V18_A2_FastStudent_wrong",
        ]
    stages += [
        "ccf_train",
        "ccf_gen",
        "measure:V18_B1_CCF_w20",
        "measure:V18_B2_CCF_w35",
        "measure:V18_B3_CCF_submode_w30",
        "measure:V18_B3_CCF_submode_wrong",
        "measure:V18_B4_CCF_dual_w30_c15",
        "track_c_gen",
        "measure:V18_C1_CCF_LD4plus",
        "report",
    ]

    t_pipe = time.time()
    for stg in stages:
        if status.get(stg, {}).get("state") == "done":
            print(f"[V18 Pipe] SKIP {stg} (already done)", flush=True)
            continue
        print(f"\n[V18 Pipe] === START {stg} ({utc()}) ===", flush=True)
        t0 = time.time()
        status[stg] = {"state": "running", "utc_start": utc()}
        save_json(status_path, status)

        cmd = [
            sys.executable, "-u", "-m", "exp.v18.run_v18",
            "--stage", stg,
            "--pool", pool_dir,
            "--ccf-updates", str(ccf_updates),
            "--ccf-micro", str(ccf_micro),
            "--ccf-accum", str(ccf_accum),
        ]
        r = subprocess.run(cmd, cwd=ROOT)
        elapsed = round((time.time() - t0) / 60.0, 2)
        if r.returncode != 0:
            status[stg] = {"state": f"failed:{r.returncode}", "minutes": elapsed}
            save_json(status_path, status)
            raise RuntimeError(f"Stage {stg} failed with code {r.returncode}")

        status[stg] = {"state": "done", "minutes": elapsed, "utc_end": utc()}
        save_json(status_path, status)
        print(f"[V18 Pipe] === DONE {stg} ({elapsed:.2f} min, total {(time.time()-t_pipe)/60:.1f} min) ===", flush=True)

    print("V18_PIPELINE_COMPLETE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=DEFAULT_POOL)
    ap.add_argument("--ccf-updates", type=int, default=5000)
    ap.add_argument("--ccf-micro", type=int, default=64)
    ap.add_argument("--ccf-accum", type=int, default=2)
    ap.add_argument("--skip-track-a", action="store_true")
    ap.add_argument("--canary", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--stage", default="")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    if a.canary:
        run_canary(a.pool, device=a.device)
    elif a.stage:
        run_stage(a.stage, a.pool, ccf_updates=a.ccf_updates,
                  ccf_micro=a.ccf_micro, ccf_accum=a.ccf_accum, device=a.device)
    elif a.all:
        run_all_subprocess_pipeline(a.pool, ccf_updates=a.ccf_updates,
                                    ccf_micro=a.ccf_micro, ccf_accum=a.ccf_accum,
                                    skip_track_a=a.skip_track_a)
    else:
        ap.error("Specify --canary, --stage <name>, or --all")


if __name__ == "__main__":
    main()
