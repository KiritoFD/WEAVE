"""Staged variant of tools/eval_images_manifest.py for memory-tight hosts.

FeatureStack loads CSD-ViT-L + DINO + CLIP + LPIPS in one process; on this
host that repeatedly dies (commit-charge AV / CUDA OOM) because ~3GB of
models + activations compete with a nearly-full Windows commit limit.

This script runs ONE model family per process and stores raw normalized
features per task; a later `finalize` step reassembles eval_rows.jsonl /
eval.json with byte-identical math to eval_images_manifest.py.

Stages:
  csd    -> out/stage_csd.pt   {task_ids, feats[N,768]}
  dino   -> out/stage_dino.pt  {task_ids, feats[N,384]}
  clip   -> out/stage_clip.pt  {task_ids, feats[N,512]}
  lpips  -> out/stage_lpips.pt {task_ids, lpips[N]}

Usage:
  python tools/_eval_staged.py run --stage csd --images DIR --out DIR --device cuda
  python tools/_eval_staged.py finalize --out DIR
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

_orig_load = torch.load


def _patched_load(f, *args, **kwargs):
    # mmap only helps for large checkpoints; small resume files must stay
    # unmapped or torch.save() cannot overwrite them (WinError 1224).
    use_mmap = (isinstance(f, (str, Path))
                and kwargs.get("mmap") is None
                and Path(f).stat().st_size > 64 * 1024 * 1024)
    if use_mmap:
        kwargs["mmap"] = True
    try:
        return _orig_load(f, *args, **kwargs)
    except RuntimeError as exc:
        if use_mmap and "mmap" in str(exc).lower():
            kwargs.pop("mmap", None)
            return _orig_load(f, *args, **kwargs)
        raise


torch.load = _patched_load

from src.eval.measure import STYLES  # noqa: E402
from src.eval.metrics import enrich_rows, summarize_rows  # noqa: E402

IMAGE_SIZE = 512
CHUNK = 16  # smaller chunks: less activation memory per stage


def safe_id(index: int, task_id: str) -> str:
    return f"{index:04d}_" + hashlib.sha256(task_id.encode()).hexdigest()[:10]


def _get_ref():
    try:
        from tools.v6_next import r2_eval as ref
    except Exception:
        from src.utils import _local_eval_refs as ref
    return ref


def _load_clip(device):
    from transformers import CLIPModel, CLIPProcessor
    import paths
    cache = str(paths.eval_cache_root("hf") / "hub")
    clip = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32", local_files_only=True,
        cache_dir=cache).to(device).eval()
    proc = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32", local_files_only=True, cache_dir=cache)
    return clip, proc


def build_pending(args):
    with args.manifest.open(encoding="utf-8", newline="") as fp:
        rows_in = list(csv.DictReader(fp))
    if args.limit:
        rows_in = rows_in[:args.limit]
    tasks = json.loads((REPO / "eval_protocol/tasks_750.json")
                       .read_text(encoding="utf-8"))["tasks"]
    by_id = {t["task_id"]: t for t in tasks}
    by_id_nfc = {unicodedata.normalize("NFC", k): v for k, v in by_id.items()}

    def _lookup(i, task_id):
        t = by_id.get(task_id) or by_id_nfc.get(
            unicodedata.normalize("NFC", task_id))
        return t if t is not None else tasks[i]

    pending, missing = [], []
    for i, row in enumerate(rows_in):
        t = _lookup(i, row["task_id"])
        png = args.images / f"{safe_id(i, t['task_id'])}.png"
        if not png.is_file():
            alt = args.images / f"{safe_id(i, row['task_id'])}.png"
            if alt.is_file():
                png = alt
        (pending if png.is_file() else missing).append(
            (i, row, t, png) if png.is_file() else (i, row["task_id"]))
    return pending, missing, rows_in


def _resume_state(out: Path, stage: str):
    path = out / f"stage_{stage}.pt"
    done: dict[str, int] = {}
    feats = []
    lp = []
    if path.is_file():
        z = torch.load(path, map_location="cpu", weights_only=False)
        ids = z["task_ids"]
        for j, tid in enumerate(ids):
            done[tid] = j
        if "feats" in z:
            feats = [z["feats"][j] for j in range(len(ids))]
        if "lpips" in z:
            lp = list(z["lpips"])
        # stored order == task order of first run; rebuild aligned lists
        return z
    return None


def cmd_run(args) -> None:
    torch.set_num_threads(4)  # fewer OMP buffers under tight commit
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) \
        else "cpu"
    pending, missing, _ = build_pending(args)
    if missing:
        print(f"[warn] {len(missing)} missing; first {missing[0]}", flush=True)

    # resume: keep already-computed task_ids, drop them from pending
    stage_path = out / f"stage_{args.stage}.pt"
    done_ids = set()
    stored = {"task_ids": [], "order": []}
    if stage_path.is_file():
        z = torch.load(stage_path, map_location="cpu", weights_only=False)
        stored = z
        done_ids = set(z["task_ids"])
        print(f"[resume] {len(done_ids)} done", flush=True)
    pending = [p for p in pending if p[2]["task_id"] not in done_ids]

    ref = _get_ref()
    t0 = time.time()

    if args.stage == "csd":
        model, _, _ = ref.csd_model.load_csd(device)
        model.eval()

        def compute(imgs):
            f = ref.csd_model.feats_pils(model, imgs, batch=4, device=device)
            f = f / np.linalg.norm(f, axis=1, keepdims=True)
            return torch.tensor(f, dtype=torch.float32)
    elif args.stage == "dino":
        model = ref.load_dino_model(device=device)
        from tools.legacy_utils.compute_dino_metrics import DINO_TRANSFORM

        def compute(imgs):
            # CPU-safe replica of ref.encode_dino's CLS branch
            # (encode_dino hard-asserts CUDA; features are identical).
            outs = []
            for i in range(0, len(imgs), 8):
                x = torch.stack([DINO_TRANSFORM(im.convert("RGB"))
                                 for im in imgs[i:i + 8]])
                with torch.inference_mode():
                    h = model(x.to(device)).last_hidden_state
                outs.append(F.normalize(h[:, 0].float(), dim=-1).cpu())
            return torch.cat(outs)
    elif args.stage == "clip":
        clip, proc = _load_clip(device)

        def compute(imgs):
            inputs = proc(images=imgs, return_tensors="pt").to(device)
            o = clip.get_image_features(**inputs)
            o = o if torch.is_tensor(o) else o.pooler_output
            return F.normalize(o.float(), dim=-1).cpu()
    elif args.stage == "lpips":
        import lpips as lpips_mod
        lp = lpips_mod.LPIPS(net="alex", verbose=False).to(device).eval()
        src = torch.load(REPO / "runs/v6_r2/cache/tasks750_source_features.pt",
                         map_location="cpu", weights_only=False)

        def compute(imgs):
            u8 = torch.stack([
                torch.from_numpy(np.asarray(im)).permute(2, 0, 1)
                for im in imgs]).to(device)
            raw = torch.stack([src["raw_rgb"][t["source_stem"]]
                               for t in group]).to(device)
            return lp(ref.to_lpips_input(u8.float().div(255)),
                      ref.to_lpips_input(raw)).flatten().cpu()
    else:
        raise SystemExit(f"bad stage {args.stage}")

    # storage containers aligned with stored["task_ids"] + new appends
    ids = list(stored.get("task_ids", []))
    if "feats" in stored:
        prev = stored["feats"]
        rows_out = [prev[j] for j in range(prev.shape[0])]
    else:
        rows_out = []
    for c0 in range(0, len(pending), args.chunk):
        chunk = pending[c0:c0 + args.chunk]
        imgs = []
        group = []
        for i, row, t, png in chunk:
            im = Image.open(png).convert("RGB")
            if im.size != (IMAGE_SIZE, IMAGE_SIZE):
                im = im.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
            imgs.append(im)
            group.append(t)
        feats_chunk = compute(imgs)  # lpips closure uses `group`
        for j, t in enumerate(group):
            ids.append(t["task_id"])
            rows_out.append(feats_chunk[j])
        # flush every chunk: tiny tensors, cheap insurance
        torch.save({"task_ids": ids,
                    "feats": torch.stack(
                        [r if torch.is_tensor(r) else torch.tensor(r)
                         for r in rows_out])}, stage_path)
        print(f"  {args.stage} {len(ids)}/{len(ids)+len(pending)-c0-len(chunk)} "
              f"... {time.time()-t0:.0f}s", flush=True)
    print(f"STAGE_{args.stage.upper()}_DONE {len(ids)}")


def cmd_finalize(args) -> None:
    out = args.out.resolve()
    stages = {}
    for s in ("csd", "dino", "clip", "lpips"):
        p = out / f"stage_{s}.pt"
        if not p.is_file():
            raise SystemExit(f"missing {p} — run stage {s} first")
        stages[s] = torch.load(p, map_location="cpu", weights_only=False)
    tids = stages["csd"]["task_ids"]
    assert all(stages[s]["task_ids"] == tids for s in stages), \
        "stage task_id order mismatch"

    tasks = json.loads((REPO / "eval_protocol/tasks_750.json")
                       .read_text(encoding="utf-8"))["tasks"]
    by_id = {t["task_id"]: t for t in tasks}
    src = torch.load(REPO / "runs/v6_r2/cache/tasks750_source_features.pt",
                     map_location="cpu", weights_only=False)
    from src.utils import reference_bank
    bank = reference_bank("cpu")
    proto = _get_ref().csd_prototypes()
    rd, rc = bank["dino"], bank["clip"]

    cf = stages["csd"]["feats"]
    df = stages["dino"]["feats"]
    clf = stages["clip"]["feats"]
    lp = stages["lpips"]["feats"].reshape(-1).float()

    scores_by_style = []
    for style, p in zip(STYLES, rd):
        sim = df @ p.T
        ref_manifest = json.loads(
            (REPO / "eval_protocol/references_v1.json")
            .read_text(encoding="utf-8"))["references"]
        stems = [r["stem"] for r in ref_manifest[style]]
        for i, tid in enumerate(tids):
            ss = by_id[tid]["source_stem"]
            if ss in stems:
                sim[i, stems.index(ss)] = -1e9
        scores_by_style.append(sim.max(1).values)
    scores = torch.stack(scores_by_style, 1).numpy()
    cs = cf @ proto.T
    clips = clf @ rc.T

    rows = []
    for i, tid in enumerate(tids):
        t = by_id[tid]
        ss, ts = t["source_style_id"], t["target_style_id"]
        cs_i = cs[i].numpy()
        rows.append({
            "task_id": t["task_id"], "src_stem": t["source_stem"],
            "src_style": t["source_style"], "tgt_style": t["target_style"],
            "src_style_id": int(ss), "tgt_style_id": int(ts),
            "is_cross": ss != ts,
            "style_scores_5": scores[i].tolist(),
            "clip_scores_5": clips[i].tolist(),
            "csd_scores_5": cs_i.tolist(),
            "dino_s": float(scores[i, ts]),
            "dino_c": float(df[i] @ src["raw_dino"][t["source_stem"]]),
            "lpips": float(lp[i]),
            "target_top1": int(cs_i.argmax() == ts),
            "target_margin": float(cs_i[ts] - np.delete(cs_i, ts).max()),
            "dT": float(cs_i[ts]
                        - src["raw_csd"][t["source_stem"]] @ proto[ts]),
        })
    rows = enrich_rows(rows, styles=STYLES)
    full = len(rows) == 750
    summary = summarize_rows(rows, styles=STYLES, require_full=full)
    (out / "eval_rows.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8")
    (out / "eval.json").write_text(json.dumps({
        "n_total": len(rows), "n_cross": sum(r["is_cross"] for r in rows),
        "n_same": sum(not r["is_cross"] for r in rows),
        "n_missing": 750 - len(rows), "rows": rows, "summary": summary,
        "staged": True,
    }, ensure_ascii=False), encoding="utf-8")
    torch.save({"task_ids": tids, "dino": df, "csd": cf, "clip": clf},
               out / "features.pt")
    print(json.dumps(summary, indent=1)[:2000])
    print("FINALIZE_DONE")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--stage", required=True,
                   choices=["csd", "dino", "clip", "lpips"])
    r.add_argument("--manifest", required=True, type=Path)
    r.add_argument("--images", required=True, type=Path)
    r.add_argument("--out", required=True, type=Path)
    r.add_argument("--device", default="cuda")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--chunk", type=int, default=CHUNK)
    f = sub.add_parser("finalize")
    f.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    if args.cmd == "run":
        cmd_run(args)
    else:
        cmd_finalize(args)


if __name__ == "__main__":
    main()
