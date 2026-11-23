"""Run the prescribed pretraining teacher gate for PRISM-1-Unified."""


import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from exp.v18 import STYLES, canonical_pid
from exp.v19.aniso_teacher import AnisotropicS10Teacher
from exp.v20.slf_flow import coarse_y16_lcn
from exp.v20.submode_patchswd import build_or_load_core_knn_bank
from exp.v21.stiefel_ref_cache import build_or_load_v21_cache
from exp.v22.t4 import T4_CONFIG
from exp.v27.icme_step2_closedset_accel import D5_CFG
from exp.v27.icme_step3b_q32_flagship import _integrate_q32_seg
from exp.v27.icme_step3_q32_vs_dino16 import make_bottleneck_net
from exp.v27_mainline import OUT, Q32_FLAGSHIP_CKPT, ROOT

from src.utils import decode_s10, load_s10_decoder
from src.utils._local_eval_refs import csd_model, csd_prototypes, encode_dino, load_dino_model
from src.utils.runtime import read_json, save_json

SEED = 20261126
DEST = OUT / "unified_teacher_gate"


def task_plan(train):
    gen = torch.Generator().manual_seed(SEED)
    selected = []
    for cls in range(5):
        ids = (train["sid"] == cls).nonzero().flatten()
        selected.append(ids[torch.randperm(len(ids), generator=gen)[:300]])
    selected = torch.cat(selected)
    src = selected.repeat_interleave(5)
    tid = torch.arange(5).repeat(len(selected))
    cross = train["sid"][src] != tid
    probe = []
    for cls in range(5):
        options = (cross & (tid == cls)).nonzero().flatten()
        probe.append(options[torch.randperm(len(options), generator=gen)[:30]])
    return selected, src[torch.cat(probe)], tid[torch.cat(probe)]


def load_teachers(bundle):
    anchor_model = AnisotropicS10Teacher(bundle, device="cuda")
    flow = make_bottleneck_net(32).cuda().eval().requires_grad_(False)
    flow.load_state_dict(torch.load(Q32_FLAGSHIP_CKPT, map_location="cpu", weights_only=False)["ema"])
    return anchor_model, flow


def make_endpoints(bundle, c21, bank, src, tid, *, teachers=None, eps=None, batch=8):
    data = bundle["v18_cache"]["train"]
    cfg = T4_CONFIG
    anchor_model, flow = teachers if teachers is not None else load_teachers(bundle)
    if eps is None:
        gen = torch.Generator().manual_seed(SEED + 1)
        eps = torch.randn((len(src), 4, 64, 64), generator=gen).half()
    results = {"hiC": [], "hiS": []}
    hi_c = dict(D5_CFG, tau_min=.48, tau_hi=.78, ws_fig=5.2)
    hi_s = dict(D5_CFG, tau_min=.30, tau_hi=.68, ws_fig=5.5, wc_fig=1.65,
                wc_abs=.70, wsp_fig_late=3.6, wsp_abs=5.2)
    for st in range(0, len(src), batch):
        sl = slice(st, st+batch)
        idx, labels = src[sl], tid[sl].cuda()
        z_src = data["z"][idx].cuda().float()
        with torch.no_grad():
            _, z0, n2 = anchor_model.ld4.run_b4_and_n2c(z_src, labels)
        anchor = anchor_model.solve_batch(
            n2, z0, z_src, bundle["fnd"]["tr_fib"][idx].cuda().float(), labels,
            mode=cfg["mode"], alpha_chroma=cfg["alpha_c"], m_steps=cfg["m_steps"],
            rho_y_edge=cfg["rho_y_edge"], rho_y_low=cfg["rho_y_low"],
            rho_c=cfg["rho_c"], lcn_budget=cfg["lcn_budget"], etas=tuple(cfg["etas"]),
            same_mask=torch.zeros_like(labels, dtype=torch.bool),
        ).detach()
        y = coarse_y16_lcn(data["y_64"][idx].cuda().float())
        d = data["dino_16"][idx].cuda().float()
        q = c21["train"]["q32"][idx].cuda().float()
        for name, full, params in (("hiC", set(), hi_c), ("hiS", {0, 1}, hi_s)):
            z = _integrate_q32_seg(flow, anchor, eps[sl].cuda().float(), y, d, q, labels, bank,
                                   steps=6, full_steps=full, cfg_kw=params)
            results[name].append(z.cpu().half())
        if (st // batch) % 8 == 0:
            print(f"TEACHER {min(st+batch,len(src))}/{len(src)} chunk", flush=True)
    return {"eps": eps, "src": src, "tid": tid, **{k: torch.cat(v) for k, v in results.items()}}


def pils(dec, latent):
    rgb = decode_s10(dec, latent.cuda().float(), batch=8, output_cpu=True)
    imgs = (rgb * 255).round().byte().permute(0, 2, 3, 1).numpy()
    return [Image.fromarray(arr) for arr in imgs]


@torch.inference_mode()
def referee(data, endpoints):
    dec = load_s10_decoder("cuda")
    hi_s = pils(dec, endpoints["hiS"])
    hi_c = pils(dec, endpoints["hiC"])
    labels = endpoints["tid"].numpy()
    csd, _, _ = csd_model.load_csd("cuda")
    features = csd_model.feats_pils(csd, hi_s, batch=8, device="cuda")
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    scores = features @ csd_prototypes().T
    hit = scores.argmax(axis=1) == labels
    del csd
    torch.cuda.empty_cache()

    dino = load_dino_model(device="cuda")
    content = pils(dec, data["z"][endpoints["src"]])
    dc, _ = encode_dino(dino, hi_c, batch=8, device="cuda")
    ds, _ = encode_dino(dino, content, batch=8, device="cuda")
    dino_c = F.cosine_similarity(dc.float(), ds.float())

    paths = {(i, canonical_pid(p.name)): p
             for i, cls in enumerate(STYLES)
             for p in sorted((ROOT.parent / "train_512clean" / cls).iterdir())
             if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
    found, originals = [], []
    for row, i in enumerate(endpoints["src"].tolist()):
        p = paths.get((int(data["sid"][i]), canonical_pid(data["ids"][i])))
        if p is not None:
            found.append(row)
            with Image.open(p) as img:
                originals.append(img.convert("RGB"))
    d_raw, _ = encode_dino(dino, originals, batch=8, device="cuda")
    raw = F.cosine_similarity(dc[found].float(), d_raw.float()) if found else None

    sums = {cls: {"hits": int(hit[labels == i].sum()), "n": int((labels == i).sum())}
            for i, cls in enumerate(STYLES)}
    return {
        "his_top1": float(hit.mean()),
        "his_per_target": sums,
        "hic_dino_c_s10_source": float(dino_c.mean()),
        "hic_dino_c_original_subset": float(raw.mean()) if raw is not None else None,
        "original_source_n": len(found),
        "per_task": [{"sid": int(data["sid"][i]), "tid": int(labels[j]), "source_id": data["ids"][i],
                      "hit": bool(hit[j]), "hic_dino_c_s10_source": float(dino_c[j])}
                     for j, i in enumerate(endpoints["src"].tolist())],
    }


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    pack = build_or_load_v21_cache(str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    bundle, c21 = pack["bundle"], pack["c21"]
    bank = build_or_load_core_knn_bank(bundle, k_sub=8, core_per_style=180, device="cuda")
    data = bundle["v18_cache"]["train"]
    selected, src, tid = task_plan(data)
    path = DEST / "pilot_endpoints.pt"
    if path.exists():
        endpoints = torch.load(path, map_location="cpu", weights_only=False)
    else:
        endpoints = make_endpoints(bundle, c21, bank, src, tid)
        torch.save(endpoints, path)
    report = referee(data, endpoints)
    report.update(seed=SEED, planned_sources=len(selected), planned_tasks=5*len(selected),
                  gate_tasks=len(src), anchor="V19_T4_CONFIG (offline teacher only)",
                  checkpoint=Q32_FLAGSHIP_CKPT.name, student_updates=0)
    conditions = {
        "his_top1_at_least_54_percent": report["his_top1"] >= .54,
        "er_at_least_25_percent": report["his_per_target"][STYLES[0]]["hits"] >= 8,
        "uki_at_least_25_percent": report["his_per_target"][STYLES[4]]["hits"] >= 8,
        "hic_dino_c_at_least_062": report["hic_dino_c_s10_source"] >= .62,
    }
    report["gate"] = conditions
    report["passed"] = all(conditions.values())
    save_json(DEST / "results.json", report)
    print({k: v for k, v in report.items() if k != "per_task"}, flush=True)
    print("PASS" if report["passed"] else "STOP: TEACHER GATE FAILED", flush=True)


if __name__ == "__main__":
    main()
