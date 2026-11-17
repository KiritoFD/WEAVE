"""Apply V11b-style RGB wavelet flow to measure()-compatible raw payloads.

Reads runs/.../t0_<lev>/batch_*.pt {"ids","rgb_u8"}, edits ONLY the RGB H2
band via edit_bands (SDEdit warm start + edge mask + RMS clip), writes a new
payload dir measure() can score. --wrong-style feeds (sid+2)%5.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from exp.v11b.bands import dwt2, idwt2, sobel_mag
from exp.v11b.sample_flow import edit_bands, load_flow

REPO = Path(__file__).resolve().parents[2]
TASKS = json.loads((REPO / "eval_protocol/tasks_750.json")
                   .read_text(encoding="utf-8"))["tasks"]
T2I = {t["task_id"]: i for i, t in enumerate(TASKS)}
SIGMAS = json.loads((REPO / "docs/v11b/P0_PROFILE.json")
                    .read_text())["sigmas"]
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="dir with batch_*.pt + meta.json")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=56)
    ap.add_argument("--band", default="H2", choices=["H1", "H2", "both"])
    ap.add_argument("--tau0", type=float, default=0.6)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--edge-decay", type=float, default=0.5)
    ap.add_argument("--rms-clip", type=float, default=1.5)
    ap.add_argument("--wrong-style", action="store_true")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device
    raw, out = Path(a.raw), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model = load_flow(Path(a.ckpt), device=dev, width=a.width)

    meta = json.loads((raw / "meta.json").read_text())
    meta_batches = []
    n = 0
    t0 = time.time()
    files = sorted(raw.glob("batch_*.pt"))
    out_i = 0
    for p in files:
        d = torch.load(p, map_location="cpu", weights_only=False)
        ids, u8 = d["ids"], d["rgb_u8"]
        ys_all = []
        for s in range(0, len(ids), a.batch):
            sl = slice(s, min(s + a.batch, len(ids)))
            g_ids = ids[sl]
            x = u8[sl].float().to(dev).permute(0, 2, 3, 1) / 255.0
            x = x.permute(0, 3, 1, 2).contiguous()          # (B,3,512,512)
            dd = dwt2(x)
            bb = {"LL2": dd["LL2"], "H2": dd["H2"], "H1": dd["H1"],
                  "LL1": dd["LL1"],
                  "edge1": sobel_mag(dd["LL1"].mean(1, keepdim=True)),
                  "edge2": sobel_mag(dd["LL2"].mean(1, keepdim=True))}
            # per-sample style id (batch may mix styles)
            sids = []
            for tid_ in g_ids:
                t = TASKS[T2I[tid_]]
                sid = t["target_style_id"]
                if a.wrong_style:
                    sid = (sid + 2) % 5
                sids.append(sid)
            bands = ["H1", "H2"] if a.band == "both" else [a.band]
            for b in bands:
                for sid in set(sids):
                    idxs = torch.tensor(
                        [i for i, s in enumerate(sids) if s == sid],
                        device=dev)
                    if len(idxs) == 0:
                        continue
                    one = {k: (v[idxs] if torch.is_tensor(v) else v)
                           for k, v in bb.items()}
                    one[b] = edit_bands(model, one, sid, SIGMAS, band=b,
                                        tau0=a.tau0, K=a.K, cfg_w=a.cfg,
                                        alpha=a.alpha,
                                        edge_decay=a.edge_decay,
                                        rms_clip_mult=a.rms_clip,
                                        device=dev)
                    bb[b][idxs] = one[b]
            y = idwt2(bb["LL2"], bb["H2"], bb["H1"]).clamp(0, 1)
            y8 = y.mul(255).round().clamp(0, 255).byte().cpu()
            ys_all.append(y8)                              # (b,3,H,W)
            n += len(g_ids)
            if a.limit and n >= a.limit:
                break
            print(f"  {n} {time.time()-t0:.0f}s", flush=True)
        if ys_all:
            yy = torch.cat(ys_all, 0)
            ii = ids[:yy.shape[0]]
            gp = out / f"batch_{out_i * a.batch:05d}.pt"
            torch.save({"ids": ii, "rgb_u8": yy}, gp)
            out_i += 1
        if a.limit and n >= a.limit:
            break
    # rebuild meta in measure() format
    mb = []
    off = 0
    for p in sorted(out.glob("batch_*.pt")):
        d = torch.load(p, map_location="cpu", weights_only=False)
        mb.append({"offset": off, "ids": d["ids"], "file": p.name})
        off += len(d["ids"])
    (out / "meta.json").write_text(json.dumps(
        {"kind": "v15_h2", "n_tasks": off, "batches": mb}, indent=1))
    print(f"H2_DONE {off} imgs {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
