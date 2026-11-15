"""V15 arm-level eval: sample + S10-decode the 750 tasks, save raw payloads.

The 3060 only GENERATES: partial-noise editing z_tau0 = (1-t0)*eps + t0*z_src
integrated to tau=1 (target-style conditioning), decoded via S10, stored as
measure()-compatible batch payloads:
  runs/v15/eval/raw/<arm>/t0_<level>/batch_XXXXX.pt  {"ids", "rgb_u8", "z_final"}
  ... + meta.json in src.eval.measure format.
The authoritative referee (DINO+CSD+CLIP+LPIPS) runs LOCALLY via
src.eval.measure on these dirs.  Quick DINO scoring here is provisional only.

eps fixed per (task, level) across arms (seed 20261115+level*100).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "runs/v15"
EVAL = RUNS / "eval"
RAW = EVAL / "raw"
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
K_STEPS = 20
BATCH = 64


def load_refs(device):
    from src.utils import reference_bank
    refs = json.loads((REPO / "eval_protocol/references_v1.json")
                      .read_text(encoding="utf-8"))["references"]
    bank = reference_bank(device)
    rd = [b.to(device) for b in bank["dino"]]           # per-style (N, D)
    ref_stems = {s: [r["stem"] for r in refs[s]] for s in STYLES}
    return rd, ref_stems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="0,0.4,0.7")
    ap.add_argument("--arms", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dino-quick", action="store_true",
                    help="also run provisional DINO scoring on the 3060")
    a = ap.parse_args()
    dev = a.device
    levels = [float(x) for x in a.levels.split(",")]

    from src.utils import decode_s10, load_s10_decoder
    from PIL import Image
    from .model import V15UNet

    tasks = json.loads((REPO / "eval_protocol/tasks_750.json")
                       .read_text(encoding="utf-8"))["tasks"]
    cache = torch.load(RUNS.parent / "v9_local_4070/cache/n2c_z.pt",
                       map_location="cpu", weights_only=False)
    z_src_all = cache["z_src"]
    assert len(tasks) == len(z_src_all), (len(tasks), len(z_src_all))

    if a.dino_quick:
        from src.utils._local_eval_refs import encode_dino, load_dino_model
        src = torch.load(REPO / "runs/v6_r2/cache/tasks750_source_features.pt",
                         map_location="cpu", weights_only=False)
        src_dino = {s: torch.as_tensor(v, dtype=torch.float32)
                    for s, v in src["raw_dino"].items()}
        rd, ref_stems = load_refs(dev)
        dino_model = load_dino_model(device=dev)
    s10 = load_s10_decoder(dev)
    RAW.mkdir(parents=True, exist_ok=True)

    arm_dirs = sorted(
        list(RUNS.glob("*/final.pt")) + list(RUNS.glob("scaling/*/final.pt")))
    if a.arms:
        keep = set(a.arms.split(","))
        arm_dirs = [p for p in arm_dirs if p.parent.name in keep]
    arm_dirs = [p.parent for p in arm_dirs]
    print(f"eval arms: {[p.name for p in arm_dirs]} levels={levels}", flush=True)

    eps_by_level = {t0: torch.randn(len(tasks), 4, 64, 64,
                                    generator=torch.Generator().manual_seed(
                                        20261115 + int(t0 * 100)))
                    for t0 in levels}

    summary = {}
    for adir in arm_dirs:
        ck = torch.load(adir / "final.pt", map_location="cpu",
                        weights_only=False)
        cfg = ck["config"]
        model = V15UNet(base_dim=int(cfg["base_dim"]),
                        use_attention=bool(cfg["use_attention"]),
                        style_mode=str(cfg.get("style_mode", "adaln"))).to(dev)
        model.load_state_dict(ck["model"])
        model.eval()
        tag = adir.name
        rows = []
        t_sample = t_decode = t_save = 0.0
        t0w = time.time()
        with torch.inference_mode():
            for lev in levels:
                ldir = RAW / tag / f"t0_{lev}"
                ldir.mkdir(parents=True, exist_ok=True)
                meta_batches = []
                eps = eps_by_level[lev]
                for bi in range(0, len(tasks), BATCH):
                    idx = list(range(bi, min(bi + BATCH, len(tasks))))
                    group = [tasks[i] for i in idx]
                    z = ((1 - lev) * eps[idx] + lev * z_src_all[idx]).to(dev)
                    sid = torch.tensor([t["target_style_id"] for t in group],
                                       device=dev)
                    span = (1 - lev) / K_STEPS
                    ts = time.time()
                    for j in range(K_STEPS):
                        t = lev + span * j
                        tt = torch.full((len(idx),), t, device=dev)
                        with torch.autocast("cuda", dtype=torch.bfloat16,
                                            enabled=dev == "cuda"):
                            v = model(z, sid, tt).float()
                        z = z + v * span
                    torch.cuda.synchronize()
                    t_sample += time.time() - ts
                    ts = time.time()
                    rgb = decode_s10(s10, z, batch=8)
                    u8 = rgb.mul(255).round().clamp_(0, 255).byte().cpu()
                    torch.cuda.synchronize()
                    t_decode += time.time() - ts
                    ts = time.time()
                    payload = {"ids": [t["task_id"] for t in group],
                               "rgb_u8": u8,
                               "z_final": z.detach().half().cpu()}
                    name = f"batch_{bi:05d}.pt"
                    tmp = ldir / (name + ".tmp")
                    torch.save(payload, tmp)
                    tmp.replace(ldir / name)
                    t_save += time.time() - ts
                    meta_batches.append({"offset": bi, "ids": payload["ids"],
                                         "file": name})
                    if a.dino_quick:
                        imgs = [Image.fromarray(x) for x in
                                u8.permute(0, 2, 3, 1).numpy()]
                        df, _ = encode_dino(dino_model, imgs, batch=16,
                                            device=dev)
                        df = torch.nn.functional.normalize(df.float(), dim=-1)
                        gen = df.cpu().numpy()
                        sd = torch.stack([src_dino[t["source_stem"]]
                                          for t in group])
                        dc = (gen * sd.numpy()).sum(-1)
                        sc = np.stack([(torch.as_tensor(gen).to(dev) @ p.T
                                        ).cpu().numpy() for p in rd], axis=1)
                        for i, t in enumerate(group):
                            stems = ref_stems[t["target_style"]]
                            sim = sc[i, t["target_style_id"]].copy()
                            for r, stem in enumerate(stems):
                                if stem == t["source_stem"]:
                                    sim[r] = -1e9
                            rows.append({
                                "level": lev, "task_id": t["task_id"],
                                "tgt": int(t["target_style_id"]),
                                "cross": int(t["source_style_id"] !=
                                             t["target_style_id"]),
                                "dino_s": float(sim.max()),
                                "dino_c": float(dc[i]),
                                "style_top1": int(int(np.argmax(
                                    sc[i].max(axis=1))) ==
                                    int(t["target_style_id"]))})
                    print(f"  {tag} L{lev} {bi+len(idx)}/{len(tasks)} "
                          f"{time.time()-t0w:.0f}s", flush=True)
                (ldir / "meta.json").write_text(json.dumps(
                    {"kind": "v15_probe", "arm": tag, "level": lev,
                     "k_steps": K_STEPS, "n_tasks": len(tasks),
                     "batches": meta_batches}, indent=1))
        timing = {}
        tpath = EVAL / "timing.json"
        if tpath.exists():
            timing = json.loads(tpath.read_text())
        timing[tag] = {"n_images": len(levels) * len(tasks),
                       "sample_s": round(t_sample, 1),
                       "decode_s": round(t_decode, 1),
                       "save_s": round(t_save, 1),
                       "wall_s": round(time.time() - t0w, 1),
                       "n_params_m": round(
                           sum(p.numel() for p in model.parameters()) / 1e6, 2)}
        tpath.write_text(json.dumps(timing, indent=1))
        print(f"{tag} timing: unet(sample)={t_sample:.0f}s "
              f"vae(decode)={t_decode:.0f}s io(save)={t_save:.0f}s "
              f"wall={time.time()-t0w:.0f}s", flush=True)
        if a.dino_quick:
            rec = {}
            for lev in levels:
                rr = [r for r in rows if r["level"] == lev]
                cr = [r for r in rr if r["cross"]]
                rec[f"t0={lev}"] = {
                    "dino_s": round(float(np.mean([r["dino_s"] for r in rr])), 4),
                    "dino_s_cross": round(float(np.mean([r["dino_s"] for r in cr])), 4),
                    "dino_c": round(float(np.mean([r["dino_c"] for r in rr])), 4),
                    "style_top1": round(float(np.mean([r["style_top1"] for r in rr])), 4),
                    "style_top1_cross": round(float(np.mean(
                        [r["style_top1"] for r in cr])), 4),
                    "n": len(rr)}
            (EVAL / f"{tag}.json").write_text(json.dumps(
                {"rows": rows, "k_steps": K_STEPS, "levels": rec}, indent=1))
            summary[tag] = rec
            print(f"{tag}: " + " | ".join(
                f"L{k}: s={v['dino_s']}(x{v['dino_s_cross']}) "
                f"c={v['dino_c']} top1={v['style_top1']}"
                for k, v in rec.items()), flush=True)
        del model
        torch.cuda.empty_cache()
    (EVAL / "summary.json").write_text(json.dumps(summary, indent=1))
    print("EVAL_ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
