"""V15 arm-level DINO eval: partial-noise style editing on the 750 tasks.

For each arm: z_tau0 = (1-t0)*eps + t0*z_src, integrate tau0 -> 1 conditioned
on the task's target style (k Euler steps), decode via S10, score with the
frozen DINO referee (CSD skipped on the 3060):
  dino_s = max over target-style reference bank (source-self refs masked)
  dino_c = cosine(gen_dino, source raw_dino)
Levels t0 (data fraction at ODE start): 0.0 = pure style generation,
0.4 / 0.7 = local-editing regime.  eps fixed per (task, level) across arms.

Run on the 3060: python -m exp.v15.eval_arms [--levels 0,0.4,0.7]
Output: runs/v15/eval/<arm>.json + eval/summary.json
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
    a = ap.parse_args()
    dev = a.device
    levels = [float(x) for x in a.levels.split(",")]

    from src.utils import decode_s10, load_s10_decoder
    from src.utils._local_eval_refs import encode_dino, load_dino_model
    from PIL import Image
    from .model import V15UNet

    tasks = json.loads((REPO / "eval_protocol/tasks_750.json")
                       .read_text(encoding="utf-8"))["tasks"]
    src = torch.load(REPO / "runs/v6_r2/cache/tasks750_source_features.pt",
                     map_location="cpu", weights_only=False)
    cache = torch.load(RUNS.parent / "v9_local_4070/cache/n2c_z.pt",
                       map_location="cpu", weights_only=False)
    z_src_all = cache["z_src"]                            # (750, 4, 64, 64)
    assert len(tasks) == len(z_src_all), (len(tasks), len(z_src_all))
    src_dino = {s: torch.as_tensor(v, dtype=torch.float32)
                for s, v in src["raw_dino"].items()}
    rd, ref_stems = load_refs(dev)
    dino_model = load_dino_model(device=dev)
    s10 = load_s10_decoder(dev)
    EVAL.mkdir(parents=True, exist_ok=True)

    arm_dirs = ([RUNS / t for t in a.arms.split(",")] if a.arms else
                sorted(p for p in RUNS.iterdir() if p.is_dir() and
                       (p / "final.pt").exists()))
    print(f"eval arms: {[p.name for p in arm_dirs]} levels={levels}", flush=True)

    # fixed noise per (task, level), identical across arms
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
        t0w = time.time()
        with torch.inference_mode():
            for lev in levels:
                eps = eps_by_level[lev]
                for bi in range(0, len(tasks), BATCH):
                    idx = list(range(bi, min(bi + BATCH, len(tasks))))
                    group = [tasks[i] for i in idx]
                    z = ((1 - lev) * eps[idx] + lev * z_src_all[idx]).to(dev)
                    sid = torch.tensor([t["target_style_id"] for t in group],
                                       device=dev)
                    span = (1 - lev) / K_STEPS
                    for j in range(K_STEPS):
                        t = lev + span * j
                        tt = torch.full((len(idx),), t, device=dev)
                        with torch.autocast("cuda", dtype=torch.bfloat16,
                                            enabled=dev == "cuda"):
                            v = model(z, sid, tt).float()
                        z = z + v * span
                    rgb = decode_s10(s10, z, batch=8)
                    u8 = rgb.mul(255).round().clamp_(0, 255).byte()
                    imgs = [Image.fromarray(x) for x in
                            u8.permute(0, 2, 3, 1).cpu().numpy()]
                    df, _ = encode_dino(dino_model, imgs, batch=16, device=dev)
                    df = torch.nn.functional.normalize(df.float(), dim=-1)
                    gen = df.cpu()
                    sd = torch.stack([src_dino[t["source_stem"]] for t in group])
                    dc = (gen * sd).sum(-1).numpy()
                    sc = np.stack([(gen.to(dev) @ p.T).cpu().numpy()
                                   for p in rd], axis=1)     # (B, 5, N)
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
                    print(f"  {tag} L{lev} {len(rows)} "
                          f"{time.time()-t0w:.0f}s", flush=True)
        rec = {}
        for lev in levels:
            rr = [r for r in rows if r["level"] == lev]
            cr = [r for r in rr if r["cross"]]
            sr = [r for r in rr if not r["cross"]]
            rec[f"t0={lev}"] = {
                "dino_s": round(float(np.mean([r["dino_s"] for r in rr])), 4),
                "dino_s_cross": round(float(np.mean([r["dino_s"] for r in cr])), 4),
                "dino_c": round(float(np.mean([r["dino_c"] for r in rr])), 4),
                "style_top1": round(float(np.mean([r["style_top1"] for r in rr])), 4),
                "n": len(rr), "n_cross": len(cr)}
        (EVAL / f"{tag}.json").write_text(json.dumps(
            {"rows_n": len(rows), "k_steps": K_STEPS, "levels": rec},
            indent=1))
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
