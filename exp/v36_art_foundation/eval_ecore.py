"""Evaluate V36 Open-Ref Reference E-Core (1-NFE Standalone & Paired w/ Frozen V36 45M P-Core) on full test750."""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch

from exp.v18 import tasks_and_sources
from exp.v20.slf_flow import coarse_y16_lcn
from exp.v29_biglittle import RawWriterV29, compute_dino_top5_mean
from exp.v32_nodino.cnx_layout import ConvNeXtTinyStage3Extractor, extract_cnx_q32
from exp.v35_cleave_one.prepare_data import _encode_rgb_scaled, _load_sd_vae_encoder
from exp.v35_cleave_one.style_features import ConvNeXtExplicitAndPatchBackbone
from exp.v36_art_foundation import CACHE_DIR, FIG_DIR, MEAS_DIR, ROOT
from exp.v36_art_foundation.ecore_model import ReferenceECoreNet
from exp.v36_art_foundation.eval_v36 import (
    _compute_v36_csls_k10,
    load_v36_models,
    reconstruct_reference_a3,
    sample_v36_ode,
)
from src.eval.runner import run_measure_batch_subprocess
from src.utils import decode_s10, load_s10_decoder
from src.utils.protocol import EVAL_SEED, pick_wrong_cross_tid
from src.utils.runtime import bf16_autocast, read_json, sanitize_cuda_env, save_json, utc


def load_reference_ecore(ecore_ckpt: Path, device: str = "cuda") -> tuple[ReferenceECoreNet, dict[str, torch.Tensor]]:
    priors = torch.load(CACHE_DIR / "v36_seen_train_priors.pt", map_location="cpu", weights_only=False)
    priors_gpu = {k: v.to(device).float() if torch.is_tensor(v) else v for k, v in priors.items()}
    ck = torch.load(ecore_ckpt, map_location="cpu", weights_only=False)
    ecore = ReferenceECoreNet(priors["stat_mean"], priors["stat_std"]).to(device).eval()
    ecore.load_state_dict(ck["ecore_ema"], strict=True)
    ecore.requires_grad_(False)
    return ecore, priors_gpu


@torch.no_grad()
def _generate_ecore_raw(
    arm_tag: str,
    *,
    ecore: ReferenceECoreNet,
    priors: dict[str, torch.Tensor],
    z_src_750: torch.Tensor,
    q32_750: torch.Tensor,
    y16_750: torch.Tensor,
    z_refs: torch.Tensor,
    st_refs: torch.Tensor,
    idx_ref: torch.Tensor,
    tasks: list[dict],
    dec,
    lam: float,
    pcore=None,
    style_enc=None,
    tau0: float | None = None,
    wj: float = 3.8,
    wc: float = 1.6,
    steps: int = 6,
    use_a3_post: bool = False,
    wrong: bool = False,
    device: str = "cuda",
) -> tuple[Path, Path]:
    raw_dir = ROOT / "runs/v36_art_foundation/raw_ecore" / arm_tag
    m_dir = MEAS_DIR / "ecore" / arm_tag
    if (m_dir / "summary_v36.json").exists() or (m_dir / "eval.json").exists():
        return raw_dir, m_dir

    raw_dir.mkdir(parents=True, exist_ok=True)
    writer = RawWriterV29(arm_tag, dec=dec, device=device)
    writer.path = raw_dir
    eps_ec = torch.randn(z_src_750.shape, generator=torch.Generator().manual_seed(EVAL_SEED + 17))
    eps_pc = torch.randn(z_src_750.shape, generator=torch.Generator().manual_seed(EVAL_SEED))

    bs = 24
    for st in range(0, len(tasks), bs):
        ed = min(st + bs, len(tasks))
        sl = slice(st, ed)
        ri = idx_ref[sl]
        zc = z_src_750[sl].to(device).float()
        q32 = q32_750[sl].to(device).float()
        zr = z_refs[ri].to(device).float()
        sr = st_refs[ri].to(device).float()
        lb = torch.full((ed - st,), float(lam), device=device)
        z_draft = ecore(zc, zr, q32, sr, lb, priors, noise=eps_ec[sl].to(device).float())

        if pcore is not None and style_enc is not None and tau0 is not None:
            z_in = reconstruct_reference_a3(z_draft, zr, priors, mode="I2") if use_a3_post else z_draft
            es = style_enc(sr)
            zo = sample_v36_ode(
                pcore, z_in, y16_750[sl].to(device).float(), q32, es, eps_pc[sl].to(device).float(),
                tau0=float(tau0), steps=steps, wj=wj, wc=wc, device=device,
            )
        else:
            zo = z_draft
        writer.write_batch(zo, tasks[sl])
    writer.finalize(arm=arm_tag, lambda_val=lam, wrong=wrong)
    return raw_dir, m_dir


@torch.no_grad()
def _save_ecore_visual_grid(
    ecore: ReferenceECoreNet,
    pcore,
    style_enc,
    priors: dict[str, torch.Tensor],
    *,
    tag: str = "v36_ecore",
    lam: float = 1.00,
    tau0: float = 0.65,
    wj: float = 3.8,
    wc: float = 1.6,
    device: str = "cuda",
) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    dec = load_s10_decoder(device)
    tasks, z_src_750, _ = tasks_and_sources()
    ea = torch.load(CACHE_DIR / "v36_eval_assets.pt", map_location="cpu", weights_only=False)
    q32_750, y16_750, refs_d = ea["q32_750"], ea["y16_750"], ea["refs150"]
    z_refs, st_refs = refs_d["z_ref"].to(device).float(), refs_d["stat_1561"].to(device).float()

    src_picks = [0, 150, 300, 450, 600]
    ref_picks = [0, 30, 60, 90, 120]
    cell, pad = 160, 8
    cols = 1 + len(ref_picks)

    eps_ec = torch.randn(len(ref_picks), 4, 64, 64, device=device, generator=torch.Generator(device=device).manual_seed(EVAL_SEED + 17))
    eps_pc = torch.randn(len(ref_picks), 4, 64, 64, device=device, generator=torch.Generator(device=device).manual_seed(EVAL_SEED))
    lb = torch.full((len(ref_picks),), float(lam), device=device)

    for mode_name, use_pcore in [("1NFE_ecore", False), ("dual_ecore_pcore", True)]:
        canvas = Image.new("RGB", (cols * (cell + pad) + pad, len(src_picks) * (cell + pad) + 32), (18, 18, 22))
        draw = ImageDraw.Draw(canvas)
        title = (
            f"V36 Ref-ECore (1-NFE, lam={lam}) | Core-5 Unseen"
            if not use_pcore else
            f"V36 Open-Ref Dual-Core (ECore lam={lam} -> PCore tau0={tau0}, wj={wj}) | Core-5 Unseen"
        )
        draw.text((10, 8), title, fill=(235, 235, 240))
        for ri, si in enumerate(src_picks):
            y0 = 32 + ri * (cell + pad)
            src_rgb = decode_s10(dec, z_src_750[si:si + 1].to(device).float(), batch=1)[0]
            canvas.paste(Image.fromarray((src_rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).resize((cell, cell), Image.Resampling.BILINEAR), (pad, y0))

            zc = z_src_750[si:si + 1].to(device).float().repeat(len(ref_picks), 1, 1, 1)
            y16 = y16_750[si:si + 1].to(device).float().repeat(len(ref_picks), 1, 1, 1)
            q32 = q32_750[si:si + 1].to(device).float().repeat(len(ref_picks), 1, 1, 1)
            zr, sr = z_refs[ref_picks], st_refs[ref_picks]
            zd = ecore(zc, zr, q32, sr, lb, priors, noise=eps_ec)
            if use_pcore:
                zo = sample_v36_ode(pcore, zd, y16, q32, style_enc(sr), eps_pc, tau0=tau0, steps=6, wj=wj, wc=wc, device=device)
            else:
                zo = zd
            outs = decode_s10(dec, zo, batch=len(ref_picks))
            for ci in range(len(ref_picks)):
                canvas.paste(Image.fromarray((outs[ci].permute(1, 2, 0).numpy() * 255).astype(np.uint8)).resize((cell, cell), Image.Resampling.BILINEAR), (pad + (ci + 1) * (cell + pad), y0))
        canvas.save(FIG_DIR / f"{tag}_{mode_name}_core5_grid.png")
    del dec
    return FIG_DIR / f"{tag}_dual_ecore_pcore_core5_grid.png"


@torch.no_grad()
def benchmark_ecore_speed(ecore_ckpt: Path, pcore_ckpt: Path, device: str = "cuda") -> dict[str, Any]:
    ecore, priors = load_reference_ecore(ecore_ckpt, device)
    pcore, style_enc, _ = load_v36_models(pcore_ckpt, device)
    dec = load_s10_decoder(device)
    vae = _load_sd_vae_encoder(device)
    cnx_gray = ConvNeXtTinyStage3Extractor(device=device)
    cnx_all = ConvNeXtExplicitAndPatchBackbone(device=device)
    w_stiefel = priors["W_stiefel_cnx"]
    w_l, b_l = priors["luma_w"], priors["luma_b"]

    def _bench(B: int, use_pcore: bool, rep: int = 8) -> tuple[float, float]:
        src = torch.rand(B, 3, 512, 512, device=device)
        ref = torch.rand(B, 3, 512, 512, device=device)
        eps1 = torch.randn(B, 4, 64, 64, device=device)
        eps2 = torch.randn(B, 4, 64, 64, device=device)
        lb = torch.ones(B, device=device)

        def once():
            zs = _encode_rgb_scaled(vae, src)
            zr = _encode_rgb_scaled(vae, ref)
            with bf16_autocast(device):
                sr = cnx_all.extract_stat_1561(ref)
                cg = cnx_gray.extract_gray_layout(src)
            q32 = extract_cnx_q32(cg.float(), w_stiefel)
            zd = ecore(zs, zr, q32, sr, lb, priors, noise=eps1)
            if use_pcore:
                y16 = coarse_y16_lcn(((zs.float() * w_l).sum(1, keepdim=True) + b_l).clamp(0.0, 1.0))
                zo = sample_v36_ode(pcore, zd, y16, q32, style_enc(sr.float()), eps2, tau0=0.65, steps=6, wj=3.8, wc=1.6, device=device)
            else:
                zo = zd
            _ = decode_s10(dec, zo, batch=B, output_cpu=True)

        for _ in range(3):
            once()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(rep):
            once()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / rep
        return round(B / dt, 2), round(dt * 1000.0, 2)

    b1_e, ms1_e = _bench(1, False, 10)
    b8_e, ms8_e = _bench(8, False, 8)
    b1_d, ms1_d = _bench(1, True, 10)
    b8_d, ms8_d = _bench(8, True, 8)
    out = {
        "gpu": torch.cuda.get_device_name(0),
        "ecore_1nfe": {"b1_ips": b1_e, "b1_ms": ms1_e, "b8_ips": b8_e, "b8_ms_per_img": round(ms8_e / 8.0, 2)},
        "dual_ecore_pcore": {"b1_ips": b1_d, "b1_ms": ms1_d, "b8_ips": b8_d, "b8_ms_per_img": round(ms8_d / 8.0, 2)},
    }
    del ecore, pcore, style_enc, dec, vae, cnx_gray, cnx_all
    gc.collect()
    torch.cuda.empty_cache()
    return out


def evaluate_ecore_and_dual(
    ecore_ckpt: Path,
    pcore_ckpt: Path,
    *,
    tag: str = "v36_ecore",
    wj: float = 3.8,
    wc: float = 1.6,
    device: str = "cuda",
) -> dict[str, Any]:
    sanitize_cuda_env()
    MEAS_DIR.mkdir(parents=True, exist_ok=True)
    tasks, z_src_750, _ = tasks_and_sources()
    ea = torch.load(CACHE_DIR / "v36_eval_assets.pt", map_location="cpu", weights_only=False)
    q32_750, y16_750, refs_d = ea["q32_750"], ea["y16_750"], ea["refs150"]
    z_refs, st_refs = refs_d["z_ref"], refs_d["stat_1561"]

    sid_750 = torch.tensor([t["source_style_id"] for t in tasks], dtype=torch.long)
    tid_750 = torch.tensor([t["target_style_id"] for t in tasks], dtype=torch.long)
    tid_wrong = pick_wrong_cross_tid(sid_750, tid_750)
    idx_ok = torch.tensor([int(tid_750[i].item()) * 30 + (i % 30) for i in range(len(tasks))], dtype=torch.long)
    idx_wr = torch.tensor([int(tid_wrong[i].item()) * 30 + (i % 30) for i in range(len(tasks))], dtype=torch.long)

    ecore, priors = load_reference_ecore(ecore_ckpt, device)
    pcore, style_enc, _ = load_v36_models(pcore_ckpt, device)
    dec = load_s10_decoder(device)

    _save_ecore_visual_grid(ecore, pcore, style_enc, priors, tag=tag, lam=1.00, tau0=0.65, wj=wj, wc=wc, device=device)

    pending: list[tuple[Path, Path, str]] = []
    metas: dict[str, dict[str, Any]] = {}
    causal_pairs: list[tuple[str, str]] = []

    def _queue(arm_tag: str, *, lam: float, use_pc: bool, tau0: float | None, use_a3: bool, wrong: bool, group: str):
        r_d, m_d = _generate_ecore_raw(
            arm_tag, ecore=ecore, priors=priors, z_src_750=z_src_750, q32_750=q32_750, y16_750=y16_750,
            z_refs=z_refs, st_refs=st_refs, idx_ref=(idx_wr if wrong else idx_ok), tasks=tasks, dec=dec,
            lam=lam, pcore=(pcore if use_pc else None), style_enc=(style_enc if use_pc else None),
            tau0=tau0, wj=wj, wc=wc, steps=6, use_a3_post=use_a3, wrong=wrong, device=device,
        )
        pending.append((r_d, m_d, arm_tag))
        metas[arm_tag] = {"group": group, "lam": lam, "tau0": tau0, "use_pcore": use_pc, "use_a3_post": use_a3, "wrong": wrong}

    # 1. Standalone 1-NFE Reference E-Core sweep
    for lv in (0.40, 0.60, 0.75, 1.00):
        lt = f"{tag}_standalone_l{int(round(lv * 100)):03d}"
        _queue(lt, lam=lv, use_pc=False, tau0=None, use_a3=False, wrong=False, group="standalone")
        if lv in (0.75, 1.00):
            wt = f"{lt}_wrong"
            _queue(wt, lam=lv, use_pc=False, tau0=None, use_a3=False, wrong=True, group="standalone")
            causal_pairs.append((lt, wt))

    # 2. Full Open-Ref Dual-Core (Reference E-Core -> Frozen V36 45M P-Core)
    dual_specs = [
        (0.75, 0.65, False),
        (1.00, 0.50, False),
        (1.00, 0.65, False),
        (1.00, 0.75, False),
        (1.00, 0.65, True),
    ]
    for lv, t0, a3p in dual_specs:
        suf = "_a3" if a3p else ""
        dt = f"{tag}_dual_l{int(round(lv * 100)):03d}_t{int(round(t0 * 100)):02d}{suf}"
        wt = f"{dt}_wrong"
        _queue(dt, lam=lv, use_pc=True, tau0=t0, use_a3=a3p, wrong=False, group="dual")
        _queue(wt, lam=lv, use_pc=True, tau0=t0, use_a3=a3p, wrong=True, group="dual")
        causal_pairs.append((dt, wt))

    del ecore, pcore, style_enc, dec
    gc.collect()
    torch.cuda.empty_cache()

    need = [(r, m, n) for r, m, n in pending if not (m / "summary_v36.json").exists()]
    if need:
        run_measure_batch_subprocess(need, device=device, write_png=False)

    rows_map: dict[str, dict[str, Any]] = {}
    for _, m_dir, arm_tag in pending:
        sum_p = m_dir / "summary_v36.json"
        if sum_p.exists():
            rows_map[arm_tag] = read_json(sum_p)
        else:
            sa = read_json(m_dir / "summary_arm.json")
            ev = read_json(m_dir / "eval.json")
            sc, ss = sa["cross600"], sa["same150"]
            out = {
                **metas[arm_tag],
                "arm": arm_tag,
                "cross600": {
                    "csd_top1": float(sc["csd_top1"]),
                    "csd_margin": float(sc["csd_margin"]),
                    "dino_c": float(sc["dino_c"]),
                    "dino_s": float(sc["dino_s"]),
                    "dino_s_top5": compute_dino_top5_mean(m_dir),
                    "csls_k10": _compute_v36_csls_k10(ev["rows"]),
                    "clip_s": float(sc["clip_s"]),
                    "style_affinity": float(sc["style_affinity"]),
                    "lpips": float(sc["lpips"]),
                },
                "same150": {"csd_top1": float(ss["csd_top1"]), "dino_c": float(ss["dino_c"])},
            }
            save_json(sum_p, out)
            rows_map[arm_tag] = out
            for bf in (_ for _ in (ROOT / "runs/v36_art_foundation/raw_ecore" / arm_tag).glob("batch_*.pt")):
                bf.unlink()

    for ok_tag, wr_tag in causal_pairs:
        ro, rw = rows_map[ok_tag], rows_map[wr_tag]
        ro["wrong_ctrl_top1"] = rw["cross600"]["csd_top1"]
        ro["causal_delta_pp"] = round((ro["cross600"]["csd_top1"] - rw["cross600"]["csd_top1"]) * 100.0, 2)

    standalone_rows = [r for r in rows_map.values() if r["group"] == "standalone" and not r["wrong"]]
    dual_rows = [r for r in rows_map.values() if r["group"] == "dual" and not r["wrong"]]

    for r in standalone_rows + dual_rows:
        c = r["cross600"]
        print(
            f"[V36 ECore Eval] {r['arm']:32s} | Top1={c['csd_top1']*100:5.2f}% D-C={c['dino_c']:.4f} "
            f"D-S={c['dino_s']:.4f} Causal={r.get('causal_delta_pp', 0.0):+5.2f}pp",
            flush=True,
        )

    speed = benchmark_ecore_speed(ecore_ckpt, pcore_ckpt, device=device)
    rep = {
        "utc": utc(),
        "ecore_ckpt": str(ecore_ckpt),
        "pcore_ckpt": str(pcore_ckpt),
        "standalone": standalone_rows,
        "dual": dual_rows,
        "speed_4090": speed,
    }
    save_json(MEAS_DIR / f"{tag}_full_report.json", rep)
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ecore-ckpt", required=True)
    ap.add_argument("--pcore-ckpt", required=True)
    ap.add_argument("--tag", default="v36_ecore")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    evaluate_ecore_and_dual(Path(a.ecore_ckpt), Path(a.pcore_ckpt), tag=a.tag, device=a.device)


if __name__ == "__main__":
    main()
