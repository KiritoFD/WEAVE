"""Capacity ceiling: CAN the 4x64x64 latent express a style change at all?

Zero training. Interpolate between a source latent z_a and a target-style latent
z_b, decode, and classify with the CSD nearest-centroid judge:

    z(t) = (1-t) * z_a + t * z_b,   t = 0, .25, .5, .75, 1.0

Reading the curves:
  * monotone rise to the target class  -> the latent DOES carry style along this
    path; the bottleneck is FINDING the direction without the target image, so an
    explicit style code / FiLM is worth building.
  * flat until t=1                     -> style is not linearly accessible from the
    latent at all; a style head on the same representation will not help either,
    and the problem sits in the VAE representation or the decoder.

Also reported: the same sweep in the other direction, so a judge bias toward one
class cannot fake a trend.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/root/Workspace/WEAVE_VAE")
sys.path.insert(0, "/root/Workspace/WEAVE_VAE/tools/12h")

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL

from tools.vae_style_vae.data import MANIFEST, StyleImageDS, load_split

A0_DIR = "/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae"
OUT = Path("/root/Workspace/WEAVE_VAE/runs/vae_style_vae")
NM = [0.48145466, 0.4578275, 0.40821073]
NS = [0.26862954, 0.26130258, 0.27577711]
T5 = ["Impressionism", "Realism", "Romanticism", "Expressionism",
      "Post-Impressionism"]


def csd_feat(csd, img, dev):
    nm = torch.tensor(NM, device=dev).view(1, 3, 1, 1)
    ns = torch.tensor(NS, device=dev).view(1, 3, 1, 1)
    x01 = (img.clamp(-1, 1) + 1) / 2
    x224 = F.interpolate(x01, (224, 224), mode="bicubic", align_corners=False)
    _, _, s = csd((x224 - nm) / ns)
    return F.normalize(s.float(), dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-src", type=int, default=20)
    ap.add_argument("--n-ref", type=int, default=32)
    ap.add_argument("--out", default=str(OUT / "capacity_ceiling.json"))
    args = ap.parse_args()
    dev = "cuda"

    man, tr, va = load_split(MANIFEST)
    ds = StyleImageDS(va, man)
    by = defaultdict(list)
    for i in range(len(ds)):
        by[ds.recs[ds.indices[i]]["style_id"]].append(i)
    present = [s for s in T5 if s in man["style_to_id"] and
               len(by[man["style_to_id"][s]]) >= 20]
    ids = {s: man["style_to_id"][s] for s in present}
    print(f"[Data] {present}", flush=True)

    vae = AutoencoderKL.from_pretrained(A0_DIR, torch_dtype=torch.float32).to(dev).eval()
    import csd_model
    csd, _, _ = csd_model.load_csd(dev)
    csd = csd.float().eval()
    for p in csd.parameters():
        p.requires_grad = False

    src, ref = {}, {}
    for s in present:
        lst = by[ids[s]]
        src[s] = lst[:min(args.n_src, len(lst) // 2)]
        ref[s] = lst[len(src[s]):len(src[s]) + args.n_ref]
    # centroids from refs, judge accuracy checked on held-out
    cent = {}
    for s in present:
        with torch.no_grad():
            x = torch.stack([ds[j]["x"] for j in ref[s]]).to(dev)
            cent[s] = F.normalize(csd_feat(csd, x, dev).mean(0, keepdim=True), dim=-1)
            del x
            torch.cuda.empty_cache()
    CENT = torch.cat([cent[s] for s in present])

    used = set()
    for s in present:
        used |= set(src[s]) | set(ref[s])
    corr = tot = 0
    for i, s in enumerate(present):
        held = [j for j in by[ids[s]] if j not in used][:40]
        if not held:
            continue
        with torch.no_grad():
            x = torch.stack([ds[j]["x"] for j in held]).to(dev)
            pr = (csd_feat(csd, x, dev) @ CENT.T).argmax(-1)
        corr += int((pr == i).sum()); tot += len(x)
        del x
        torch.cuda.empty_cache()
    print(f"[Judge] held-out acc = {corr/max(tot,1)*100:.1f}% "
          f"(chance {100/len(present):.1f}%)", flush=True)

    def z_of(s, j):
        with torch.no_grad():
            x = ds[j]["x"].unsqueeze(0).to(dev)
            z = vae.encode(x).latent_dist.mean.float()
            del x
            torch.cuda.empty_cache()
            return z

    ts = [0.0, 0.25, 0.5, 0.75, 1.0]
    rows = []
    for si, s in enumerate(present):
        for ti, t in enumerate(present):
            per_t = {tt: {"top1": 0, "prob": [], "n": 0} for tt in ts}
            for j in src[s]:
                za = z_of(s, j)
                # target latent: mean of a few target-style refs (stable, not one image)
                # use a SINGLE real target latent: the mean of same-style latents
                # decodes to something the judge cannot identify (measured: t=1
                # same-style Top-1 collapses to chance), so averaging destroys style
                zb = z_of(t, ref[t][0])
                for tt in ts:
                    z = (1 - tt) * za + tt * zb
                    with torch.no_grad(), torch.autocast(device_type="cuda",
                                                        dtype=torch.bfloat16):
                        y = vae.decode(z).sample.float()
                    f = csd_feat(csd, y, dev)
                    pr = (f @ CENT.T)
                    per_t[tt]["top1"] += int((pr.argmax(-1) == ti).sum())
                    per_t[tt]["prob"].append(float(pr[0, ti]))
                    per_t[tt]["n"] += 1
                    del z, y
                    torch.cuda.empty_cache()
                del za, zb
            for tt in ts:
                d = per_t[tt]
                rows.append({"src": s, "tgt": t, "t": tt,
                             "top1": round(d["top1"] / max(d["n"], 1), 4),
                             "prob": round(float(np.mean(d["prob"])), 4)})
        print(f"  {s:<22} done", flush=True)

    cross = [r for r in rows if r["src"] != r["tgt"]]
    same = [r for r in rows if r["src"] == r["tgt"]]
    print("\n" + "=" * 78)
    print("CAPACITY CEILING: style Top-1 vs interpolation t")
    print("=" * 78)
    print(f"{'t':>6}{'cross Top-1':>14}{'cross prob':>13}"
          f"{'same Top-1':>13}{'same prob':>12}")
    summary = {}
    for tt in ts:
        c = [r for r in cross if r["t"] == tt]
        sm = [r for r in same if r["t"] == tt]
        ct = float(np.mean([r["top1"] for r in c])) if c else float("nan")
        cp = float(np.mean([r["prob"] for r in c])) if c else float("nan")
        st = float(np.mean([r["top1"] for r in sm])) if sm else float("nan")
        sp = float(np.mean([r["prob"] for r in sm])) if sm else float("nan")
        print(f"{tt:>6.2f}{ct*100:>13.1f}%{cp:>13.3f}{st*100:>12.1f}%{sp:>12.3f}")
        summary[str(tt)] = {"cross_top1": round(ct, 4), "cross_prob": round(cp, 4),
                            "same_top1": round(st, 4), "same_prob": round(sp, 4)}

    print(f"\nchance = {100/len(present):.1f}%")
    base = summary["0.0"]["cross_top1"]
    full = summary["1.0"]["cross_top1"]
    print(f"cross Top-1: t=0 {base*100:.1f}% -> t=1 {full*100:.1f}% "
          f"(delta {(full-base)*100:+.1f} pts)")
    if full - base > 0.15:
        verdict = ("MONOTONE / LARGE: the latent CAN express style change along "
                   "the z_a->z_b path; the bottleneck is finding the direction "
                   "without the target image -> build the explicit style code.")
    elif full - base > 0.05:
        verdict = ("PARTIAL: some style is linearly accessible but weak; an "
                   "explicit style code may help, expect limited headroom.")
    else:
        verdict = ("FLAT: style is not linearly accessible from the latent even "
                   "with the target latent in hand -> a style head on the same "
                   "representation will not help; the VAE representation is the "
                   "bottleneck.")
    print(f"VERDICT: {verdict}")

    Path(args.out).write_text(json.dumps({"judge_acc": round(corr / max(tot, 1), 4),
                                          "summary": summary, "verdict": verdict,
                                          "rows": rows}, indent=2))
    print(f"[Saved] {args.out}")


if __name__ == "__main__":
    main()
