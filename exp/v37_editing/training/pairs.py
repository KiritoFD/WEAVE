"""Fixed reverse real pairs, strictly Seen-Train, with proxy-derived Q32.

No evaluator is imported or used to accept/filter pairs. A small fixed contact
sheet must be inspected before training; selection is metadata-only.
"""
import argparse
import json
import time
from collections import defaultdict
import numpy as np
import torch
from PIL import Image, ImageDraw
from exp.v20.slf_flow import coarse_y16_lcn
from exp.v32_nodino.cnx_layout import ConvNeXtTinyStage3Extractor, extract_cnx_q32
from exp.v36_art_foundation import CACHE_DIR, ROOT
from exp.v36_art_foundation.eval_v36 import load_v36_models, reconstruct_reference_a3, sample_v36_ode
from exp.v37_editing import OUT, SEED, atomic_json, read_manifest
from src.utils import decode_s10, load_s10_decoder
from src.utils.runtime import bf16_autocast


def make_index(n=10000):
    folder = OUT / "pairs"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "index.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    meta = torch.load(CACHE_DIR / "seen_train_index.pt", weights_only=False)
    manifest = json.loads((CACHE_DIR / "curation_manifest.json").read_text())
    records = manifest["seen_train"]
    valid = meta["valid_idx"].numpy()
    by_style, by_artist_style = defaultdict(list), defaultdict(list)
    for i in valid:
        r = records[i]
        by_style[r["style"]].append(int(i))
        by_artist_style[(r["artist"], r["style"])].append(int(i))
    rng = np.random.default_rng(SEED + 300)
    targets = rng.choice(valid, size=n, replace=False)
    refs, others, same_artist = [], [], []
    for i in targets:
        r = records[i]
        candidates = [j for j in by_artist_style[(r["artist"], r["style"])] if j != i]
        same_artist.append(bool(candidates))
        if not candidates:
            candidates = [j for j in by_style[r["style"]] if j != i]
        assert candidates, "Need independent work with the target style"
        refs.append(rng.choice(candidates, size=2, replace=len(candidates) < 2))
        other_style = rng.choice([s for s in by_style if s != r["style"]])
        others.append(rng.choice(by_style[other_style]))
    result = {"target": targets, "target_refs": np.array(refs), "other_ref": np.array(others),
              "same_artist": np.array(same_artist), "seed": SEED + 300,
              "n": n, "teacher": json.loads((OUT / "teacher_selection.json").read_text()),
              "tau": .65, "wj": 3.8, "wc": 1.6, "steps": 6,
              "pair_selection": "fixed metadata-only, no evaluator filtering"}
    assert np.all(result["target_refs"] != targets[:, None])
    assert len(set(targets.tolist())) == n
    torch.save(result, path)
    atomic_json(folder / "provenance.json", {k: v for k, v in result.items()
                if k not in ("target", "target_refs", "other_ref", "same_artist")} |
                {"independent_reference_same_artist_fraction": float(np.mean(same_artist)),
                 "all_target_references_same_movement": True, "all_other_refs_different_movement": True,
                 "all_sources_seen_train": True})
    return result


@torch.no_grad()
def generate(preview=False):
    index = make_index()
    folder = OUT / "pairs"
    state_path = folder / "state.json"
    done = json.loads(state_path.read_text())["done"] if state_path.exists() else 0
    n = index["n"]
    end = min(24, n) if preview else n
    if done >= end:
        return
    mode = "r+" if (folder / "proxy_z.npy").exists() else "w+"
    pz = np.lib.format.open_memmap(folder / "proxy_z.npy", mode=mode, dtype=np.float16, shape=(n, 4, 64, 64))
    pq = np.lib.format.open_memmap(folder / "proxy_q.npy", mode=mode, dtype=np.float16, shape=(n, 32, 16, 16))
    meta = torch.load(CACHE_DIR / "seen_train_index.pt", weights_only=False)
    arrays = {k: np.load(ROOT / meta[k + "_rel"], mmap_mode="r") for k in ("z_gt", "q32", "y16", "stat")}
    tensor = lambda k, ids: torch.from_numpy(np.array(arrays[k][ids])).cuda().float()
    p, se, priors = load_v36_models(index["teacher"]["path"])
    dec, cnx = load_s10_decoder("cuda"), ConvNeXtTinyStage3Extractor("cuda")
    w = priors["W_stiefel_cnx"].cuda().float()
    started = time.perf_counter()
    start_done = done
    qdiff = []
    for s in range(done, end, 24):
        if time.time() > read_manifest()["eval_reserved_from_epoch"] - 60:
            raise TimeoutError("Pair generation reached training deadline")
        e = min(s + 24, end)
        ids, refs = index["target"][s:e], index["other_ref"][s:e]
        z, zr, q, y = tensor("z_gt", ids), tensor("z_gt", refs), tensor("q32", ids), tensor("y16", ids)
        es = se(tensor("stat", refs))
        eps = torch.stack([torch.randn((4, 64, 64), generator=torch.Generator().manual_seed(SEED + 100000 + i))
                           for i in range(s, e)]).cuda()
        start = reconstruct_reference_a3(z, zr, priors)
        proxy = sample_v36_ode(p, start, y, q, es, eps, tau0=index["tau"], steps=6, wj=3.8, wc=1.6)
        rgb = decode_s10(dec, proxy, batch=24, output_cpu=False)
        with bf16_autocast("cuda"):
            layout = cnx.extract_gray_layout(rgb)
        qp = extract_cnx_q32(layout.float(), w)
        pz[s:e], pq[s:e] = proxy.cpu().half().numpy(), qp.cpu().half().numpy()
        qdiff.append(float((qp - q).square().mean().sqrt()))
        pz.flush(); pq.flush()
        atomic_json(state_path, {"done": e, "n": n, "complete": e == n,
                    "samples_per_s": (e - start_done) / (time.perf_counter() - started),
                    "q_recomputed_from_proxy": True, "q_rms_change_last_batch": qdiff[-1],
                    "fixed_tau": index["tau"], "nfe": 6})
        if s == 0:
            save_sheet(dec, z[:12], zr[:12], proxy[:12], tensor("z_gt", index["target_refs"][:12, 0]), folder)
        if e % 240 == 0 or e == end:
            print(f"PAIRS {e}/{n} | {(e-start_done)/(time.perf_counter()-started):.1f} pairs/s | Q-diff {qdiff[-1]:.4f}", flush=True)


@torch.no_grad()
def save_sheet(dec, target, wrong_ref, proxy, target_ref, folder):
    columns = [decode_s10(dec, z, batch=12, output_cpu=True) for z in (target, wrong_ref, proxy, target_ref)]
    # Fixed first 12; no score-based cherry-picking. Keep center crops for texture comparison.
    for page in range(3):
        canvas = Image.new("RGB", (6 * 192, 4 * 216), "white")
        draw = ImageDraw.Draw(canvas)
        for row in range(4):
            i = page * 4 + row
            ims = [Image.fromarray((c[i].permute(1, 2, 0).numpy().clip(0, 1)*255).astype(np.uint8)) for c in columns]
            imgs = ims + [ims[0].crop((192, 192, 320, 320)), ims[2].crop((192, 192, 320, 320))]
            for j, im in enumerate(imgs):
                canvas.paste(im.resize((192, 192)), (j * 192, row * 216 + 24))
                draw.text((j * 192 + 4, row * 216 + 4), ["real target", "other-style ref", "proxy source", "independent target ref", "target crop", "proxy crop"][j], fill="black")
        canvas.save(folder / f"fixed_preview_{page}.jpg", quality=92)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true")
    a = ap.parse_args()
    generate(a.preview)
