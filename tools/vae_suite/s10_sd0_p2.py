"""P2: downstream acceptance of the old-S10 SD0 adaptation.

Uses the PRECOMPUTED Q2a+Oracle1 latents (750 tasks, WEAVE scaled SD0 space)
from runs/sd0_s10/q2a_oracle_images/oracle_latents_scaled.pt, so B/C/D see
IDENTICAL latents: the only difference is the post_quant_conv + decoder.

  B = SD0 latent + A0 PQC          + old S10
  C = SD0 latent + calibrated PQC  + old S10
  D = SD0 latent + SD0 PQC         + official D0   (already generated, reused)
  A = A0 latent  + A0 PQC          + old S10       (reference only; needs A0 encode)

Then scores every variant under the docs/baseline paper_rgb_v1 protocol by
reusing score_baseline_protocol.py's machinery (DINO-S / CLIP-S / Affinity /
DINO-C / LPIPS / CSD Top-1), cross600 + same150.

Run:
  python -m tools.vae_suite.s10_sd0_p2 --arms Oracle1,B4 --phase gen
  python -m tools.vae_suite.s10_sd0_p2 --arms Oracle1,B4 --phase score
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path("/root/Workspace/WEAVE_VAE")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "12h"))

from tools.vae_suite.s10_sd0_adapt import (A0_DIR, SD0_DIR, S10_CK, OUT,
                                           build_models, make_pqc, WIDTHS)
from tools.vae_suite.decoder_prune import StudentDecoder, build_decoder

QDIR = ROOT / "runs/sd0_s10/q2a_oracle_images"
TASKS = ROOT / "deliver/testset/protocol/tasks_750.json"
LAT = QDIR / "oracle_latents_scaled.pt"
P2 = OUT / "p2"          # default: Oracle1 arm
P2.mkdir(parents=True, exist_ok=True)
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]


def build_variants(dev):
    a0, sd0, d, pa = build_models(dev)
    wa = pa["weight"].float().to(dev).view(4, 4)
    ba = pa["bias"].float().to(dev)
    cc = torch.load(OUT / "calibrated_pqc.pt", map_location="cpu", weights_only=False)
    wc = cc["post_quant_conv"]["weight"].float().to(dev).view(4, 4)
    bc = cc["post_quant_conv"]["bias"].float().to(dev)
    out = {
        "B": StudentDecoder(make_pqc(wa, ba, dev), d).to(dev).eval(),
        "C": StudentDecoder(make_pqc(wc, bc, dev), d).to(dev).eval(),
        "D": lambda z: sd0.decode(z).sample,
    }
    return out, sd0


def to_uint8(y):
    return (y.clamp(-1, 1).add(1).mul(127.5).round().clamp(0, 255).byte()
            .permute(1, 2, 0).cpu().numpy())


def gen(dev, bs=8):
    spec = json.loads(TASKS.read_text(encoding="utf-8"))
    tasks = spec["tasks"]
    zmap = torch.load(LAT, map_location="cpu", weights_only=False)
    print(f"[P2.gen] {len(tasks)} tasks, {len(zmap)} latents", flush=True)
    # sanity: every task has a latent
    missing = [t["task_id"] for t in tasks if t["task_id"] not in zmap]
    assert not missing, f"{len(missing)} tasks missing latents"

    variants, sd0 = build_variants(dev)
    sf = float(sd0.config.scaling_factor)
    for name, dec in variants.items():
        vdir = P2 / name
        (vdir / "cross").mkdir(parents=True, exist_ok=True)
        (vdir / "same").mkdir(parents=True, exist_ok=True)
        manifest = []
        with torch.no_grad():
            for i in range(0, len(tasks), bs):
                grp = tasks[i:i + bs]
                z = torch.stack([zmap[t["task_id"]] for t in grp]).to(dev) / sf
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    y = dec(z).float()
                for j, t in enumerate(grp):
                    sub = "cross" if t["is_cross"] else "same"
                    fn = t["task_id"].replace("/", "__") + ".png"
                    Image.fromarray(to_uint8(y[j])).save(vdir / sub / fn)
                    manifest.append({
                        "task_id": t["task_id"], "file": f"{sub}/{fn}",
                        "is_cross": t["is_cross"], "source_stem": t["source_stem"],
                        "source_style": t["source_style"],
                        "source_style_id": t["source_style_id"],
                        "target_style": t["target_style"],
                        "target_style_id": t["target_style_id"],
                    })
                del z, y
                torch.cuda.empty_cache()
        (vdir / "manifest.json").write_text(json.dumps({
            "variant": name, "n": len(manifest),
            "n_cross": sum(m["is_cross"] for m in manifest),
            "n_same": sum(not m["is_cross"] for m in manifest),
            "rows": manifest}, indent=2), encoding="utf-8")
        print(f"[P2.gen] {name}: {len(manifest)} PNGs", flush=True)
    print("[P2.gen] done", flush=True)


def score(dev, workdir=None, src_dir_override=None, tag="p2"):
    """Score B/C/D with the exact paper_rgb_v1 protocol."""
    global P2
    if workdir is not None:
        P2 = Path(workdir)
    def dino_cls(model, pils, dev, bs=16):
        from tools.pseudo_gt.matching import encode_dino
        import torch.nn.functional as F
        out = []
        with torch.no_grad():
            for i in range(0, len(pils), bs):
                cls, _ = encode_dino(model, pils[i:i + bs], batch=bs, device=dev)
                out.append(F.normalize(cls.float(), dim=-1).cpu().numpy())
        return np.concatenate(out)

    import csd_model, lpips
    from transformers import AutoModel, CLIPModel, CLIPProcessor
    from tools.pseudo_gt.matching import encode_dino, load_dino_model
    from tools.vae_suite.score_baseline_protocol import (csd_feats, clip_cls,
                                                         cluster_boot_idx,
                                                         paired_ci)
    REFS = ROOT / "deliver/vae_ranked_eval/vae_style_reference_subset.pt"
    HF = ROOT / "eval_cache/hf/hub"
    refs = torch.load(REFS, map_location="cpu", weights_only=False)
    ref_dino = [torch.nn.functional.normalize(
        refs["styles"][s]["dino_cls"].float(), dim=-1).numpy() for s in STYLES]
    proto_clip = torch.nn.functional.normalize(torch.stack(
        [refs["clip_prototypes"][s].float() for s in STYLES]), dim=-1).numpy()
    cp = torch.load(ROOT / "deliver/testset/refs/csd_proto.pt",
                    map_location="cpu", weights_only=False)
    proto_csd = np.stack([np.asarray(cp[s], np.float32) for s in STYLES])
    proto_csd /= np.linalg.norm(proto_csd, axis=1, keepdims=True)

    dino = load_dino_model(device=dev)
    clipm = CLIPModel.from_pretrained("openai/clip-vit-base-patch32",
                                      local_files_only=True,
                                      cache_dir=str(HF)).to(dev).eval()
    clipp = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32",
                                          local_files_only=True,
                                          cache_dir=str(HF))
    csd, _, _ = csd_model.load_csd(dev); csd = csd.float().eval()
    lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    for m in (clipm, csd, lp):
        for p in m.parameters():
            p.requires_grad = False

    # source round-trip images come from the sdvae set (official D0) -- the
    # paper_rgb_v1 reference. For B/C the same RGB source is used.
    src_dir = Path(src_dir_override) if src_dir_override else QDIR / "sources"
    stems = sorted(p.stem for p in src_dir.glob("*.png"))
    src_idx = {s: i for i, s in enumerate(stems)}
    src_pils = [Image.open(src_dir / f"{s}.png").convert("RGB") for s in stems]
    src_t = [torch.from_numpy(np.asarray(p, np.uint8)).permute(2, 0, 1)
             .float().div_(127.5).sub_(1.0) for p in src_pils]
    Ds = dino_cls(dino, src_pils, dev)
    Cs = clip_cls(clipm, clipp, src_pils, dev)
    Fs = csd_feats(csd, src_pils, dev)

    results, per_task = {}, {}
    for v in ("B", "C", "D") if workdir is None else ("A",):
        mf = json.loads((P2 / v / "manifest.json").read_text())
        rows = mf["rows"]
        pils = [Image.open(P2 / v / r["file"]).convert("RGB") for r in rows]
        gt = [torch.from_numpy(np.asarray(p, np.uint8)).permute(2, 0, 1)
              .float().div_(127.5).sub_(1.0) for p in pils]
        Dg = dino_cls(dino, pils, dev)
        Cg = clip_cls(clipm, clipp, pils, dev)
        Fg = csd_feats(csd, pils, dev)
        ds = np.stack([(Dg @ r.T).max(axis=1) for r in ref_dino], axis=1)
        cs = Cg @ proto_clip.T
        lpv = []
        with torch.no_grad():
            for i in range(0, len(rows), 8):
                g = torch.stack(gt[i:i + 8]).to(dev)
                s = torch.stack([src_t[src_idx[rows[j]["source_stem"]]]
                                 for j in range(i, min(i + 8, len(rows)))]).to(dev)
                lpv.append(lp(g, s).flatten().cpu().numpy())
        lpv = np.concatenate(lpv)
        recs = []
        for j, r in enumerate(rows):
            ti = r["target_style_id"]
            si = src_idx[r["source_stem"]]
            sc = Fg[j] @ proto_csd.T
            dn, cl = float(ds[j, ti]), float(cs[j, ti])
            recs.append({"task_id": r["task_id"], "is_cross": r["is_cross"],
                         "source_stem": r["source_stem"],
                         "target_style": STYLES[ti],
                         "dino_s": dn, "clip_s": cl,
                         "style_affinity": (dn + cl) / 2,
                         "dino_c": float(Dg[j] @ Ds[si]),
                         "lpips": float(lpv[j]),
                         "csd_top1": int(int(np.argmax(sc)) == ti),
                         "csd_margin": float(sc[ti] - np.max(np.delete(sc, ti)))})
        per_task[v] = recs

        def agg(rs):
            g = lambda k: float(np.nanmean([x[k] for x in rs]))
            return {k: round(g(k), 5) for k in
                    ("dino_s", "clip_s", "style_affinity", "dino_c", "lpips",
                     "csd_top1", "csd_margin")}
        cross = [x for x in recs if x["is_cross"]]
        results[v] = {"cross600": agg(cross),
                      "same150": agg([x for x in recs if not x["is_cross"]])}
        c = results[v]["cross600"]
        print(f"  {v}: affinity={c['style_affinity']:.4f} dino_s={c['dino_s']:.4f} "
              f"clip_s={c['clip_s']:.4f} dino_c={c['dino_c']:.4f} "
              f"lpips={c['lpips']:.4f} top1={c['csd_top1']:.4f}", flush=True)
        del Dg, Cg, Fg, pils, gt
        torch.cuda.empty_cache()

    # paired bootstrap C vs B (the calibration must beat direct compatibility)
    boot = {}
    if "B" not in per_task:
        boot = {}
        out = {"phase": "P2", "protocol": "paper_rgb_v1 cross600",
               "variants": results, "paired_bootstrap_vs_B": boot,
               "per_task_records": {k: len(v) for k, v in per_task.items()},
               "note": "A reference arm scored alone (no B in same workdir)"}
        (P2 / "scores.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        (P2 / "per_task.json").write_text(json.dumps(per_task), encoding="utf-8")
        print(f"[P2.score] saved {P2 / 'scores.json'} (A only)", flush=True)
        return
    base = {x["task_id"]: x for x in per_task["B"] if x["is_cross"]}
    common = sorted(base)
    clusters = np.array([base[k]["source_stem"] for k in common])
    for v in ("C", "D"):
        other = {x["task_id"]: x for x in per_task[v] if x["is_cross"]}
        for m in ("style_affinity", "dino_s", "clip_s", "dino_c", "lpips",
                  "csd_top1", "csd_margin"):
            boot[f"{v}_vs_B|{m}"] = paired_ci(
                [other[k][m] for k in common], [base[k][m] for k in common],
                clusters, seed=0)

    out = {"phase": "P2", "protocol": "paper_rgb_v1 cross600",
           "variants": results, "paired_bootstrap_vs_B": boot,
           "p0": json.loads((OUT / "p0.json").read_text()),
           "p1": json.loads((OUT / "p1.json").read_text()),
           "per_task_records": {k: len(v) for k, v in per_task.items()}}
    (P2 / "scores.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (P2 / "per_task.json").write_text(json.dumps(per_task), encoding="utf-8")
    print(f"[P2.score] saved {P2 / 'scores.json'}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["gen", "score", "both"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dir", default="")
    ap.add_argument("--src-dir", default="")
    args = ap.parse_args()
    if args.phase in ("gen", "both"):
        gen(args.device)
    if args.phase in ("score", "both"):
        score(args.device, workdir=args.dir or None,
              src_dir_override=args.src_dir or None)


if __name__ == "__main__":
    main()
