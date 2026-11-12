"""Four-part evaluation for the style-friendly VAE.

1. 107-class style linear probe on the latent (same protocol as the A0 baseline
   10.51%, so the numbers are directly comparable).
2. Reconstruction: L1 / LPIPS@512 / CSD-keep on the frozen 2,048 holdout.
3. Cross-content style swap over 750 tasks (150 sources x 5 targets):
   CSD nearest-centroid Top-1 and target-probability change, with a same-norm
   random control, plus DINO-C / LPIPS content preservation.
   Absolute target cosine is NOT used: it is confounded (a random perturbation
   already raises CSD toward some styles).
4. Per-style breakdown.

Usage: python -m tools.vae_style_vae.eval --ckpt runs/vae_style_vae/A_joint_best.pt
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
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL

from tools.vae_style_vae.data import MANIFEST, StyleImageDS, load_split
from tools.vae_style_vae.model import StyleVAE, style_content, style_swap
from tools.vae_suite.decoder_prune import build_decoder

A0_DIR = "/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae"
S10_CKPT = "/root/Workspace/WEAVE_VAE/runs_vae_prune_placeholder"
OUT = Path("/root/Workspace/WEAVE_VAE/runs/vae_style_vae")
NM = [0.48145466, 0.4578275, 0.40821073]
NS = [0.26862954, 0.26130258, 0.27577711]

TARGET5 = ["Impressionism", "Realism", "Romanticism", "Expressionism",
           "Post-Impressionism"]


def build(ckpt_path, dev, a0_decoder=False):
    vae = AutoencoderKL.from_pretrained(A0_DIR, torch_dtype=torch.float32).to(dev)
    import torch as _t
    if a0_decoder:
        dec = vae.decoder                      # true A0 baseline
    else:
        ck = _t.load("/root/Workspace/WEAVE_VAE/runs/vae_decoder_prune/s10_p5_lowlr/decoder.pt",
                     map_location="cpu", weights_only=False)
        dec = build_decoder(ck["meta"]["widths_forward"])
        dec.load_state_dict(ck["state_dict"])
    model = StyleVAE(vae, dec.to(dev)).to(dev)
    if ckpt_path and Path(ckpt_path).exists():
        s = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.encoder.load_state_dict(s["encoder"])
        model.decoder.load_state_dict(s["decoder"])
        print(f"[Load] {ckpt_path} meta={s.get('meta')}", flush=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def csd_feat(csd, img, dev):
    nm = torch.tensor(NM, device=dev).view(1, 3, 1, 1)
    ns = torch.tensor(NS, device=dev).view(1, 3, 1, 1)
    x01 = (img.clamp(-1, 1) + 1) / 2
    x224 = F.interpolate(x01, (224, 224), mode="bicubic", align_corners=False)
    _, _, s = csd((x224 - nm) / ns)
    return F.normalize(s.float(), dim=-1)


def dino_feat(m, img, dev):
    nm = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    ns = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    x01 = (img.clamp(-1, 1) + 1) / 2
    x224 = F.interpolate(x01, (224, 224), mode="bicubic", align_corners=False)
    o = m(pixel_values=(x224 - nm) / ns)
    return F.normalize(o.last_hidden_state[:, 0].float(), dim=-1)


@torch.no_grad()
def encode_all(model, ds, idxs, dev, bs=8):
    Z = []
    for i in range(0, len(idxs), bs):
        x = torch.stack([ds[j]["x"] for j in idxs[i:i + bs]]).to(dev)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            z = model.encode(x).mean.float()
        Z.append(z)
        del x
        torch.cuda.empty_cache()
    return torch.cat(Z)


def fit_linear(Xtr, ytr, Xte, yte, n_cls, dev, epochs=30, bs=2048, lr=1e-3):
    mu = Xtr.mean(0, keepdim=True)
    sd = Xtr.std(0, keepdim=True) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    clf = nn.Linear(Xtr.shape[1], n_cls).to(dev)
    nn.init.zeros_(clf.weight); nn.init.zeros_(clf.bias)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=1e-4)
    n = len(Xtr)
    for _ in range(epochs):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            k = perm[i:i + bs]
            loss = F.cross_entropy(clf(Xtr[k]), ytr[k])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    with torch.no_grad():
        acc = float((clf(Xte).argmax(-1) == yte).float().mean())
        tr = float((clf(Xtr[:8000]).argmax(-1) == ytr[:8000]).float().mean())
    return acc, tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(OUT / "A_joint_best.pt"))
    ap.add_argument("--a0-decoder", action="store_true",
                    help="use A0's own decoder (the true baseline)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--probe-per-style", type=int, default=200)
    ap.add_argument("--top-styles", type=int, default=20,
                    help="probe only the N best-populated styles; 107 classes with a handful of samples each is noise, and the baseline must use the same protocol")
    ap.add_argument("--n-src", type=int, default=30)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    dev = "cuda"
    model = build(args.ckpt, dev, args.a0_decoder)

    man, tr_recs, val_recs = load_split(MANIFEST)
    name_of = {v: k for k, v in man["style_to_id"].items()}
    ds_tr = StyleImageDS(tr_recs, man)
    ds_val = StyleImageDS(val_recs, man)

    import lpips
    lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    for p in lp.parameters():
        p.requires_grad = False
    import csd_model
    csd, _, _ = csd_model.load_csd(dev)
    csd = csd.float().eval()
    for p in csd.parameters():
        p.requires_grad = False
    from transformers import AutoModel
    dino = AutoModel.from_pretrained("facebook/dinov2-base").to(dev).eval()
    for p in dino.parameters():
        p.requires_grad = False

    res = {}

    # ---------------- 1. 107-class linear probe ----------------
    print("\n" + "=" * 84)
    print("1. STYLE LINEAR PROBE ON LATENT (A0 baseline 10.51%)")
    print("=" * 84, flush=True)
    by_style = defaultdict(list)
    for i in range(len(ds_tr)):
        by_style[ds_tr.recs[ds_tr.indices[i]]["style_id"]].append(i)
    rng = np.random.default_rng(0)
    keep, ys, rows = [], [], []
    # top-N by population, so every class has enough samples for a stable estimate
    ranked = sorted(by_style.items(), key=lambda kv: -len(kv[1]))
    styles = [s for s, v in ranked[:args.top_styles] if len(v) >= 20]
    print("  styles: " + ", ".join(f"{name_of[s]}({len(by_style[s])})" for s in styles),
          flush=True)
    for s in styles:
        lst = by_style[s]
        pick = lst if len(lst) <= args.probe_per_style else \
            rng.choice(lst, args.probe_per_style, replace=False).tolist()
        for p in pick:
            rows.append(int(p)); ys.append(s)
    sid = {s: i for i, s in enumerate(sorted(set(ys)))}
    y = torch.tensor([sid[s] for s in ys], device=dev)
    Z = encode_all(model, ds_tr, rows, dev)
    tr_i, te_i = [], []
    ynp = np.array([sid[s] for s in ys])
    for s in sorted(set(ys)):
        m = np.where(ynp == sid[s])[0]
        rng.shuffle(m)
        cut = max(2, int(len(m) * 0.8))
        tr_i += m[:cut].tolist(); te_i += m[cut:].tolist()
    tr_i = torch.tensor(tr_i, device=dev); te_i = torch.tensor(te_i, device=dev)
    Zf = Z.flatten(1)   # Linear needs (N, 4*64*64); Z stays 4-D for style_content
    acc, tr_acc = fit_linear(Zf[tr_i], y[tr_i], Zf[te_i], y[te_i], len(sid), dev)
    print(f"  {len(sid)} styles, train={len(tr_i)} test={len(te_i)}, "
          f"chance={100/len(sid):.2f}%", flush=True)
    print(f"  latent probe test acc = {acc*100:.2f}%  (train {tr_acc*100:.2f}%)",
          flush=True)

    # Decompose: is the gain coming from the means (mu) or the scales (log sigma)?
    # A model can make style "separable" purely by inflating per-channel means,
    # which is the failure mode the plan warns about. Reporting the split makes
    # that visible instead of hidden behind a single number.
    S8 = torch.cat([style_content(Z[i:i + 4096])[0] for i in range(0, len(Z), 4096)])
    parts = {}
    for nm_, cols in (("mu_only", slice(0, 4)), ("logsigma_only", slice(4, 8)),
                      ("s_full", slice(0, 8))):
        a, _ = fit_linear(S8[tr_i][:, cols], y[tr_i], S8[te_i][:, cols],
                          y[te_i], len(sid), dev)
        parts[nm_] = round(a * 100, 2)
        print(f"  {nm_:<14} test acc = {a*100:.2f}%", flush=True)
    res["probe"] = {"n_styles": len(sid), "chance_pct": round(100 / len(sid), 2),
                    "test_pct": round(acc * 100, 2),
                    "train_pct": round(tr_acc * 100, 2),
                    "s8_split": parts,
                    "s8_abs_mean": round(float(S8.abs().mean()), 4),
                    "mu_abs_mean": round(float(S8[:, :4].abs().mean()), 4),
                    "logsigma_abs_mean": round(float(S8[:, 4:].abs().mean()), 4)}
    del Z
    torch.cuda.empty_cache()

    # ---------------- 2. reconstruction ----------------
    print("\n" + "=" * 84)
    print("2. RECONSTRUCTION on the frozen holdout")
    print("=" * 84, flush=True)
    l1s, lps, csds = [], [], []
    n = min(128, len(ds_val))
    with torch.no_grad():
        for i in range(0, n, args.bs):
            x = torch.stack([ds_val[j]["x"] for j in range(i, min(i + args.bs, n))]).to(dev)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                y = model.decode(model.encode(x).mean).float()
            l1s.append(float((y - x).abs().mean()))
            lps.append(float(lp(y, x).mean()))
            csds.append(float((csd_feat(csd, y, dev) * csd_feat(csd, x, dev)).sum(-1).mean()))
            del x, y
            torch.cuda.empty_cache()
    res["recon"] = {"l1": round(float(np.mean(l1s)), 5),
                    "lpips": round(float(np.mean(lps)), 5),
                    "csd": round(float(np.mean(csds)), 5), "n": n}
    print(f"  {json.dumps(res['recon'])}", flush=True)

    # ---------------- 3. 750-task style swap ----------------
    print("\n" + "=" * 84)
    print("3. CROSS-CONTENT STYLE SWAP (5 styles x 5 targets)")
    print("=" * 84, flush=True)
    present = [s for s in TARGET5 if s in man["style_to_id"]]
    ids = {s: man["style_to_id"][s] for s in present}
    by_val = defaultdict(list)
    for i in range(len(ds_val)):
        by_val[ds_val.recs[ds_val.indices[i]]["style_id"]].append(i)
    src, ref = {}, {}
    for s in present:
        lst = by_val[ids[s]]
        if len(lst) < 8:
            print(f"  [skip] {s} has only {len(lst)} holdout images", flush=True)
            continue
        src[s] = lst[:min(args.n_src, len(lst))]
        ref[s] = lst[len(src[s]):len(src[s]) + 32] or lst[:32]
    present = [s for s in present if s in src]
    print(f"  styles={present} sources={sum(len(v) for v in src.values())}", flush=True)

    cent = {}
    for s in present:
        with torch.no_grad():
            x = torch.stack([ds_val[j]["x"] for j in ref[s]]).to(dev)
            cent[s] = F.normalize(csd_feat(csd, x, dev).mean(0, keepdim=True), dim=-1)
            del x
            torch.cuda.empty_cache()
    CENT = torch.cat([cent[s] for s in present])
    # reference latents per style, computed ONCE (was re-encoded per source: 30x waste)
    ZREF = {}
    for s in present:
        with torch.no_grad():
            x = torch.stack([ds_val[j]["x"] for j in ref[s][:4]]).to(dev)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                ZREF[s] = model.encode(x).mean.float().mean(0, keepdim=True)
            del x
            torch.cuda.empty_cache()

    def classify(img):
        f = csd_feat(csd, img, dev)
        return (f @ CENT.T).argmax(-1), (f @ CENT.T)

    # sanity: classifier accuracy on real holdout images
    # held-out judge accuracy: images never used as centroids, and not sources
    used = set()
    for s in present:
        used |= set(src[s]) | set(ref[s])
    held = {s: [j for j in by_val[ids[s]] if j not in used][:40] for s in present}
    correct = tot = 0
    for i, s in enumerate(present):
        if not held[s]:
            continue
        with torch.no_grad():
            x = torch.stack([ds_val[j]["x"] for j in held[s]]).to(dev)
            pred, _ = classify(x)
        correct += int((pred == i).sum()); tot += len(x)
        del x
        torch.cuda.empty_cache()
    judge_acc = correct / max(tot, 1)
    print(f"  [judge] HELD-OUT nearest-centroid acc = {judge_acc*100:.1f}% "
          f"(chance {100/len(present):.1f}%, n={tot})", flush=True)

    rows3 = []
    # Loop over SOURCES in the outer loop: y0 = D(z_a) and its classification do
    # not depend on the target style, so they are computed once per source
    # instead of once per (source, target) task.
    for s in present:
        si = present.index(s)
        per_t = {t: {"top1": 0, "n": 0, "dprob": [], "dino_c": [], "lpips": []}
                 for t in present}
        for j in src[s]:
            xa = ds_val[j]["x"].unsqueeze(0).to(dev)
            with torch.no_grad(), torch.autocast(device_type="cuda",
                                                dtype=torch.bfloat16):
                za = model.encode(xa).mean.float()
                y0 = model.decode(za).float()
            _, prob0 = classify(y0)
            f_dino_a = dino_feat(dino, xa, dev)
            for ti, t in enumerate(present):
                with torch.no_grad(), torch.autocast(device_type="cuda",
                                                    dtype=torch.bfloat16):
                    z_sw = style_swap(za, ZREF[t])
                    y_sw = model.decode(z_sw).float()
                pred, prob = classify(y_sw)
                d = per_t[t]
                d["top1"] += int((pred == ti).sum()); d["n"] += 1
                d["dprob"].append(float(prob[0, ti] - prob0[0, ti]))
                d["dino_c"].append(float((dino_feat(dino, y_sw, dev) *
                                          f_dino_a).sum(-1).mean()))
                d["lpips"].append(float(lp(y_sw, xa).mean()))
                del z_sw, y_sw
                torch.cuda.empty_cache()
            del xa, za, y0
            torch.cuda.empty_cache()
        for t in present:
            d = per_t[t]
            rows3.append({"src": s, "tgt": t, "same": s == t,
                          "top1": round(d["top1"] / max(d["n"], 1), 4),
                          "dprob": round(float(np.mean(d["dprob"])), 5),
                          "dino_c": round(float(np.mean(d["dino_c"])), 5),
                          "lpips": round(float(np.mean(d["lpips"])), 5)})
        print(f"  {s:<22} done", flush=True)

    cross = [r for r in rows3 if not r["same"]]
    same = [r for r in rows3 if r["same"]]
    res["swap"] = {
        "n_cross": len(cross), "n_same": len(same),
        "cross_top1": round(float(np.mean([r["top1"] for r in cross])), 4),
        "cross_dprob": round(float(np.mean([r["dprob"] for r in cross])), 5),
        "cross_dino_c": round(float(np.mean([r["dino_c"] for r in cross])), 5),
        "cross_lpips": round(float(np.mean([r["lpips"] for r in cross])), 5),
        "same_top1": round(float(np.mean([r["top1"] for r in same])), 4) if same else None,
        "chance_top1": round(1 / len(present), 4),
        "judge_heldout_acc": round(judge_acc, 4),
        "rows": rows3,
    }
    print("\n" + "-" * 84)
    print(f"  CROSS  n={len(cross)}  Top-1={res['swap']['cross_top1']*100:.1f}% "
          f"(chance {100/len(present):.1f}%)  dprob={res['swap']['cross_dprob']:+.4f}  "
          f"DINO-C={res['swap']['cross_dino_c']:.4f}  LPIPS={res['swap']['cross_lpips']:.4f}",
          flush=True)
    if same:
        print(f"  SAME   n={len(same)}  Top-1={res['swap']['same_top1']*100:.1f}%",
              flush=True)

    print("\n" + "=" * 84)
    print("PER-STYLE TARGET TOP-1 (rows=source, cols=target)")
    print("=" * 84)
    print("src -> tgt".ljust(24) + "".join(f"{t[:10]:>12}" for t in present))
    for s in present:
        line = f"{s[:22]:<24}"
        for t in present:
            r = [x for x in rows3 if x["src"] == s and x["tgt"] == t]
            line += f"{r[0]['top1']*100:>11.0f}%" if r else f"{'-':>12}"
        print(line)

    outp = args.out or str(OUT / f"eval_{args.tag or Path(args.ckpt).stem}.json")
    Path(outp).write_text(json.dumps(res, indent=2))
    print(f"\n[Saved] {outp}")


if __name__ == "__main__":
    main()
