"""Process-isolated referee evaluator for train_4400 internal calibration split (train_calib_300).

Strictly disjoint from tasks_750, references_v1, and holdout_300 so Stage P2 hyperparameter
calibration never peeks at official test750 or unseen holdout_750.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .metrics import enrich_rows, summarize_rows
from src.utils.protocol import STYLES
from src.utils.raw_io import save_png_batch_fast
from src.utils.runtime import REPO, auto_referee_batch, run_isolated_python, save_json

CALIB_SPEC_PATH = REPO / "runs/v35_final/calib/train_calib_300_spec.pt"


@torch.no_grad()
def build_or_load_calib_spec(*, device: str = "cuda", cache_path: Path = CALIB_SPEC_PATH) -> dict[str, Any]:
    """Select 60 core train_4400 sources (12/style -> 300 tasks) and 60 disjoint core train_4400 refs."""
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    from src.utils import FeatureStack, decode_s10, load_s10_decoder

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    v18 = torch.load(REPO / "runs/v18/cache/ccf_pool_cache.pt", map_location="cpu", weights_only=False)
    tr = v18["train"]
    cnx_bank = torch.load(REPO / "runs/v32_nodino/cache/k8_cnx417_bank.pt", map_location="cpu", weights_only=False)
    core_cls = [ci.cpu().long() for ci in cnx_bank["core_idx_per_style"]]

    src_list, ref_list = [], []
    for s in range(5):
        ci = core_cls[s]
        if len(ci) < 24:
            raise ValueError(f"Need >= 24 core train items for style {STYLES[s]}, got {len(ci)}")
        # Deterministically spread across core_cls[s]
        step = max(1, len(ci) // 24)
        picked = ci[::step][:24].tolist()
        src_list.extend(picked[:12])
        ref_list.extend(picked[12:24])

    src_idx = torch.tensor(src_list, dtype=torch.long)
    ref_idx = torch.tensor(ref_list, dtype=torch.long)

    dec = load_s10_decoder(device).to(torch.bfloat16).eval().requires_grad_(False)
    stack = FeatureStack(device)
    proto = stack.csd_prototypes

    def _extract(indices: torch.Tensor) -> dict[str, Any]:
        u8_l, df_l, cf_l, clf_l = [], [], [], []
        for st in range(0, len(indices), 16):
            sl = indices[st:st + 16]
            rgb = decode_s10(dec, tr["z"][sl].to(device).float(), dtype=torch.bfloat16, batch=len(sl), output_cpu=False).clamp(0.0, 1.0)
            u8 = (rgb * 255.0).round().to(torch.uint8).cpu()
            imgs = [Image.fromarray(a) for a in u8.permute(0, 2, 3, 1).numpy()]
            fo = stack.extract(imgs, batch=8)
            u8_l.append(u8)
            df_l.append(fo["dino"].cpu())
            cf_l.append(torch.from_numpy(fo["csd"]).cpu())
            clf_l.append(fo["clip"].cpu())
        return {
            "u8": torch.cat(u8_l, 0),
            "dino": torch.cat(df_l, 0),
            "csd": torch.cat(cf_l, 0),
            "clip": torch.cat(clf_l, 0),
        }

    src_feats = _extract(src_idx)
    ref_feats = _extract(ref_idx)

    ref_dino = [ref_feats["dino"][s * 12:(s + 1) * 12].clone() for s in range(5)]
    ref_clip = torch.stack([ref_feats["clip"][s * 12:(s + 1) * 12].mean(dim=0) for s in range(5)], dim=0)
    ref_clip = ref_clip / ref_clip.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    sid = tr["sid"].long()
    tasks = []
    for k, orig_i in enumerate(src_idx.tolist()):
        ss = int(sid[orig_i].item())
        stem = f"calib_{k:03d}_{STYLES[ss][:3]}_{orig_i}"
        for ts in range(5):
            tid_int = k * 5 + ts
            tasks.append({
                "task_id": f"cal_{tid_int:04d}",
                "source_pos": k,
                "orig_train_idx": int(orig_i),
                "source_stem": stem,
                "source_style": STYLES[ss],
                "target_style": STYLES[ts],
                "source_style_id": ss,
                "target_style_id": ts,
            })

    del dec, stack
    gc.collect()
    torch.cuda.empty_cache()

    payload = {
        "tasks": tasks,
        "src_idx": src_idx,
        "ref_idx": ref_idx,
        "src_u8": src_feats["u8"],
        "src_dino": src_feats["dino"],
        "src_csd": src_feats["csd"],
        "ref_dino": ref_dino,
        "ref_clip": ref_clip,
        "csd_proto": torch.as_tensor(np.asarray(proto)).cpu(),
    }
    torch.save(payload, cache_path)
    return payload


def summarize_calib_arm(arm_name: str, meas_dir: Path) -> dict[str, Any]:
    d = json.loads((meas_dir / "eval.json").read_text(encoding="utf-8"))
    rows = d["rows"]
    cross_rows = [r for r in rows if r["is_cross"]]
    same_rows = [r for r in rows if not r["is_cross"]]

    def _stats(rs: list[dict[str, Any]]) -> dict[str, Any]:
        clip_s = float(np.mean([r["clip_scores_5"][r["tgt_style_id"]] for r in rs]))
        dino_s = float(np.mean([r["dino_s"] for r in rs]))
        return {
            "n": len(rs),
            "dino_c": round(float(np.mean([r["dino_c"] for r in rs])), 4),
            "dino_s": round(dino_s, 4),
            "csd_top1": round(float(np.mean([r["target_top1"] for r in rs])), 4),
            "clip_s": round(clip_s, 4),
            "style_affinity": round(0.5 * (dino_s + clip_s), 4),
            "lpips": round(float(np.mean([r["lpips"] for r in rs])), 4),
        }

    return {"arm": arm_name, "cross240": _stats(cross_rows), "same60": _stats(same_rows)}


def measure_calib(raw_dir: Path, out: Path, *, spec_path: Path = CALIB_SPEC_PATH, device: str = "cuda", micro_batch: int | None = None) -> list[dict[str, Any]]:
    return measure_calib_batch([(raw_dir, out)], spec_path=spec_path, device=device, micro_batch=micro_batch)[0]


def measure_calib_batch(
    pairs: list[tuple[Path, Path]],
    *,
    spec_path: Path = CALIB_SPEC_PATH,
    device: str = "cuda",
    micro_batch: int | None = None,
) -> list[list[dict]]:
    """Measure multiple train_calib_300 raw outputs in a single FeatureStack load."""
    from src.utils import FeatureStack

    mb = int(micro_batch) if micro_batch is not None else auto_referee_batch(device)
    spec = build_or_load_calib_spec(device=device, cache_path=spec_path)
    tasks = spec["tasks"]
    by_id = {t["task_id"]: t for t in tasks}
    stack = FeatureStack(device)
    proto_np = np.asarray(stack.csd_prototypes)
    rd = [r.to(device) for r in spec["ref_dino"]]
    rc = spec["ref_clip"].to(device)
    src_u8 = spec["src_u8"]
    src_dino = spec["src_dino"].numpy()
    src_csd = spec["src_csd"].numpy()

    all_rows: list[list[dict]] = []
    for raw_dir, out in pairs:
        raw_dir, out = Path(raw_dir), Path(out)
        out.mkdir(parents=True, exist_ok=True)
        meta = json.loads((raw_dir / "meta.json").read_text(encoding="utf-8"))
        rows = []
        t0 = time.time()
        for entry in meta["batches"]:
            payload = torch.load(raw_dir / entry["file"], map_location="cpu", weights_only=False)
            group = [by_id[i] for i in payload["ids"]]
            u8 = payload["rgb_u8"]
            imgs = [Image.fromarray(a) for a in u8.permute(0, 2, 3, 1).numpy()]
            fo = stack.extract(imgs, batch=mb)
            df, cf, clf = fo["dino"], fo["csd"], fo["clip"]
            scores = torch.stack([(df @ p.T).max(1).values for p in rd], 1).cpu().numpy()
            cs = cf @ proto_np.T
            clips = (clf @ rc.T).cpu().numpy()
            raw_rgb = torch.stack([src_u8[t["source_pos"]] for t in group])
            lr = stack.lpips_vs(u8, raw_rgb, batch=mb).cpu().tolist()
            for i, t in enumerate(group):
                ss, ts = int(t["source_style_id"]), int(t["target_style_id"])
                spos = int(t["source_pos"])
                p_cls = int(cs[i].argmax())
                rows.append({
                    "task_id": t["task_id"],
                    "src_stem": t["source_stem"],
                    "src_style": t["source_style"],
                    "tgt_style": t["target_style"],
                    "src_style_id": ss,
                    "tgt_style_id": ts,
                    "is_cross": ss != ts,
                    "pred_class": p_cls,
                    "style_scores_5": scores[i].tolist(),
                    "clip_scores_5": clips[i].tolist(),
                    "csd_scores_5": cs[i].tolist(),
                    "dino_s": float(scores[i, ts]),
                    "dino_c": float(df[i].cpu().numpy() @ src_dino[spos]),
                    "lpips": float(lr[i]),
                    "target_top1": int(p_cls == ts),
                    "target_margin": float(cs[i, ts] - np.delete(cs[i], ts).max()),
                    "dT": float(cs[i, ts] - src_csd[spos] @ proto_np[ts]),
                })
            print(f"  calib measure [{raw_dir.name}] {len(rows)} {time.time() - t0:.1f}s", flush=True)
        rows = enrich_rows(rows, styles=STYLES, require_full=False, expected_sources=60)
        summary = summarize_rows(rows, styles=STYLES, require_full=False)
        (out / "eval.json").write_text(json.dumps({
            "n_total": len(rows), "n_cross": sum(r["is_cross"] for r in rows),
            "n_same": sum(not r["is_cross"] for r in rows), "rows": rows, "summary": summary,
        }, ensure_ascii=False), encoding="utf-8")
        all_rows.append(rows)
    return all_rows


def run_calib_measure_batch_subprocess(
    items: list[tuple[Path | str, Path | str, str | None]],
    *,
    device: str = "cuda",
    micro_batch: int | None = None,
    force: bool = False,
) -> dict[str, dict[str, Any]]:
    mb = int(micro_batch) if micro_batch is not None else auto_referee_batch(device)
    pending: list[str] = []
    norm: list[tuple[Path, Path, str]] = []
    for raw_dir, meas_dir, arm_name in items:
        raw_p, meas_p = Path(raw_dir), Path(meas_dir)
        name = arm_name or raw_p.name
        norm.append((raw_p, meas_p, name))
        if force or not (meas_p / "eval.json").exists():
            pending.append(f"{raw_p}::{meas_p}")
    if pending:
        rc = run_isolated_python("src.eval.calib", ["--pairs", *pending, "--device", device, "--micro-batch", str(mb)])
        if rc != 0:
            raise RuntimeError(f"src.eval.calib --pairs failed (rc={rc})")
    out: dict[str, dict[str, Any]] = {}
    for _, meas_p, name in norm:
        summary = summarize_calib_arm(name, meas_p)
        save_json(meas_p / "summary_calib.json", summary)
        out[name] = summary
    return out


def run_calib_measure_subprocess(raw_dir: Path | str, meas_dir: Path | str, *, arm_name: str | None = None, device: str = "cuda", micro_batch: int | None = None, force: bool = False) -> dict[str, Any]:
    raw_p, meas_p = Path(raw_dir), Path(meas_dir)
    name = arm_name or raw_p.name
    return run_calib_measure_batch_subprocess([(raw_p, meas_p, name)], device=device, micro_batch=micro_batch, force=force)[name]


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure train_calib_300 outputs")
    ap.add_argument("--raw")
    ap.add_argument("--out")
    ap.add_argument("--pairs", nargs="+")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--micro-batch", type=int, default=0)
    args = ap.parse_args()
    mb = args.micro_batch if args.micro_batch > 0 else None
    if args.pairs:
        pairs = [(Path(s.split("::", 1)[0]), Path(s.split("::", 1)[1])) for s in args.pairs]
        res = measure_calib_batch(pairs, device=args.device, micro_batch=mb)
        print(f"CALIB_BATCH_DONE {len(res)} arms", flush=True)
    else:
        if not args.raw or not args.out:
            ap.error("Either --pairs or --raw/--out required")
        rows = measure_calib(Path(args.raw), Path(args.out), device=args.device, micro_batch=mb or auto_referee_batch(args.device))
        print(f"CALIB_EVAL_DONE {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
