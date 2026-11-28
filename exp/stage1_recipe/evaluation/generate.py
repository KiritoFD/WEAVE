"""Explicit GPU workers: latent generation and isolated referee scoring.

The queue writes immutable jobs. Import/help is CPU-only; actual workers require
WEAVE_STAGE1_GPU_ALLOWED=1. No training/test-set selection shares a process.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")

from exp.stage1_recipe.config import ROOT, read, resolve, write


def require_gpu_window() -> None:
    if os.environ.get("WEAVE_STAGE1_GPU_ALLOWED") != "1":
        raise RuntimeError("GPU worker disabled. Use the serial queue only after the card is free.")


def fingerprint(job: dict) -> str:
    return hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()


@lru_cache(maxsize=64)
def _file_digest(path: str, size: int, mtime_ns: int) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_digest(path: Path) -> str:
    stat = path.stat()
    return _file_digest(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


def require_canonical_cache(config: dict, root: Path | None = None) -> Path:
    root = ROOT if root is None else root
    value = Path(config["paths"]["cache_root"])
    actual = (value if value.is_absolute() else root / value).resolve()
    if actual != (root / "runs/v36_art_foundation/cache").resolve():
        raise ValueError("Stage 1 requires canonical V36 cache (junctions are resolved)")
    return actual


def generation_provenance(config: dict, job: dict, pairing: dict, seed: int, root: Path | None = None) -> dict:
    root = ROOT if root is None else root
    cache = require_canonical_cache(config, root)
    path_for = lambda value: (Path(value) if Path(value).is_absolute() else root / value)
    checkpoints = {key: file_digest(path_for(job[key])) for key in ("e_checkpoint", "p_checkpoint")
                   if job.get(key)}
    names = ["v36_seen_train_priors.pt", "v36_caches_ready.json", "curation_manifest.json"]
    if job["split"] == "core750":
        names.append("v36_eval_assets.pt")
    assets = {name: file_digest(cache / name) for name in names}
    implementations = ["exp/v36_art_foundation/ecore_model.py", "exp/v36_art_foundation/models.py",
                       "exp/v36_art_foundation/eval_v36.py", "src/eval/measure.py", "src/utils/features.py"]
    metadata = {"schema": "stage1_generation_v2", "job": job, "checkpoints_sha256": checkpoints,
                "evaluation": config["evaluation"], "p_noise_seed": seed, "e_noise_seed": seed + 17,
                "pairing": pairing, "cache_root": config["paths"]["cache_root"],
                "cache_metadata_sha256": assets,
                "implementation_sha256": file_digest(Path(__file__)),
                "model_and_referee_sources_sha256": {p: file_digest(root / p) for p in implementations if (root / p).is_file()},
                "decoder_protocol": "S10 scaled latent; no Same150 bypass"}
    return {**metadata, "signature": fingerprint(metadata)}


def validate_provenance(expected: dict, actual: dict | None) -> None:
    if not actual or actual.get("signature") != expected["signature"] or fingerprint(
            {k: v for k, v in actual.items() if k != "signature"}) != actual.get("signature"):
        raise ValueError("Stale evaluation artifact: checkpoint/config/noise/pairing provenance differs; "
                         "use a new output directory rather than reusing unmatched cached results")


def _state(ck: dict, key: str, ema: str | None):
    """Accept historical V36 checkpoints and stage-one full-state checkpoints."""
    if ema == "online":
        name = "online" if key == "ecore" and "online" in ck else key
        if name not in ck:
            raise KeyError(f"No online {key} in checkpoint")
        return ck[name]
    if ema not in (None, "ema"):
        shadows = ck.get("emas", {})
        shadow_key = next((name for name in shadows if float(name) == float(ema)), None)
        if shadow_key is None:
            if key + "_ema" in ck and ck.get("ema_decay") is not None and float(ck["ema_decay"]) == float(ema):
                return ck[key + "_ema"]
            raise KeyError(f"Requested EMA {ema} is absent; available: {list(shadows)}")
        shadow = shadows[shadow_key]
        return shadow.get("ecore", shadow) if key == "ecore" else shadow[key]
    if key + "_ema" in ck:
        return ck[key + "_ema"]
    if ema == "ema" or ck.get("emas"):
        raise KeyError(f"Explicit EMA decay or online required for {key}")
    name = "online" if key == "ecore" and "online" in ck else key
    if name not in ck:
        raise KeyError(f"No {key} weights in checkpoint")
    return ck[name]


def load_pack(config: dict, job: dict, root: Path = ROOT):
    """Correct-reference pack plus complete, target-independent pairing provenance."""
    require_canonical_cache(config, root)
    if job["split"] == "core750":
        import exp.v36_art_foundation.diag_v36 as diag
        old_cache = diag.CACHE_DIR
        diag.CACHE_DIR = require_canonical_cache(config, root)
        try:
            tasks, z, q, y, refs, idx, wrong_idx = diag._load_diag_assets()
        finally:
            diag.CACHE_DIR = old_cache
        if len(tasks) != 750 or len({t["task_id"] for t in tasks}) != 750:
            raise ValueError("Full unique core750 task set required")
        pack = {"zc": z, "q": q, "y": y, "zr": refs["z_ref"][idx],
                "sr": refs["stat_1561"][idx], "zw": refs["z_ref"][wrong_idx],
                "sw": refs["stat_1561"][wrong_idx]}
        pairing = {"task_ids": [t["task_id"] for t in tasks],
                   "source_ids": [t["source_stem"] for t in tasks],
                   "target_style_ids": [int(t["target_style_id"]) for t in tasks],
                   "reference_ids": idx.tolist(), "wrong_reference_ids": wrong_idx.tolist()}
    else:
        from exp.v37_editing.evaluation import common
        from exp.v36_art_foundation import CACHE_DIR as OLD_CACHE_DIR
        import exp.v36_art_foundation as v36
        old_cache, old_root = v36.CACHE_DIR, v36.ROOT
        v36.CACHE_DIR, v36.ROOT = require_canonical_cache(config, root), root
        try:
            pack = common.split_pack(job["split"], int(job.get("n", 0)))
        finally:
            v36.CACHE_DIR, v36.ROOT = old_cache, old_root
        tasks = None
        pairing = {"task_ids": [f"{job['split']}:{i}" for i in pack["source_ids"]],
                   "source_ids": list(pack["source_ids"]), "reference_ids": list(pack["ref_ids"]),
                   "wrong_reference_ids": list(pack["wrong_ids"])}
    n = len(pack["zc"])
    expected = {"core750": 750, "artist_ood": 1000, "movement_ood": 500}
    if job["split"] in expected and n != expected[job["split"]]:
        raise ValueError(f"Full {job['split']} required, got {n}")
    if any(len(v) != n for v in pairing.values()):
        raise ValueError("Split pairing IDs are not aligned")
    return pack, tasks, pairing


def load_models(config: dict, job: dict, root: Path = ROOT):
    import torch
    from exp.v36_art_foundation.models import FoundationPCoreNet, StyleMLPEncoder
    from exp.v36_art_foundation.ecore_model import ReferenceECoreNet
    cache = require_canonical_cache(config, root)
    priors = torch.load(cache / "v36_seen_train_priors.pt",
                        map_location="cpu", weights_only=False)
    family = job["family"]
    pcore = style = ecore = None
    if family in ("p", "ep"):
        ppath = Path(job["p_checkpoint"]); ppath = ppath if ppath.is_absolute() else root / ppath
        ck = torch.load(ppath, map_location="cpu", weights_only=False)
        pcore = FoundationPCoreNet(base_dim=208, style_dim=256, cond_dim=384, use_gamma=True,
                                  normalize_style=True, per_layer_adagn=True, use_checkpoint=False)
        pcore.load_state_dict(_state(ck, "pcore", job.get("p_ema")), strict=True)
        style = StyleMLPEncoder(1561, 512, 256, priors["stat_mean"], priors["stat_std"])
        style.load_state_dict(_state(ck, "style_enc", job.get("p_ema")), strict=True)
        del ck
        pcore, style = pcore.cuda().eval(), style.cuda().eval()
    if family in ("e", "ep"):
        epath = Path(job["e_checkpoint"]); epath = epath if epath.is_absolute() else root / epath
        ck = torch.load(epath, map_location="cpu", weights_only=False)
        ecore = ReferenceECoreNet(priors["stat_mean"], priors["stat_std"])
        ecore.load_state_dict(_state(ck, "ecore", job.get("e_ema")), strict=True)
        del ck
        ecore = ecore.cuda().eval()
    priors = {k: v.cuda().float() if torch.is_tensor(v) else v for k, v in priors.items()}
    return pcore, style, ecore, priors


def generate(config: dict, job: dict, output: Path, deadline: float, root: Path = ROOT) -> None:
    require_gpu_window()
    import torch
    from exp.v36_art_foundation.eval_v36 import reconstruct_reference_a3, sample_v36_ode
    from src.utils.protocol import EVAL_SEED
    from src.utils.runtime import bf16_autocast
    cfg = config["evaluation"]
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "job.json"
    if manifest.exists() and fingerprint(read(manifest)) != fingerprint(job):
        raise ValueError(f"Output already belongs to another frozen job: {output}")
    write(manifest, job)
    target = output / "latents.pt"
    pack, tasks, pairing = load_pack(config, job, root)
    seed = int(cfg.get("seed", EVAL_SEED))
    provenance = generation_provenance(config, job, pairing, seed, root)
    old_provenance = output / "provenance.json"
    if old_provenance.exists():
        validate_provenance(provenance, read(old_provenance))
    elif any((output / name).exists() for name in ("raw", "measured", "metrics.json")):
        raise ValueError("Output contains unversioned scoring artifacts; choose a new directory")
    if target.exists():
        existing = torch.load(target, map_location="cpu", weights_only=False)
        validate_provenance(provenance, existing.get("provenance"))
        return
    write(output / "provenance.json", provenance)
    pcore, style, ecore, priors = load_models(config, job, root)
    if job.get("wrong"):
        pack["zr"], pack["sr"] = pack["zw"], pack["sw"]
    n = len(pack["zc"])
    eps_e = torch.randn((n, 4, 64, 64), generator=torch.Generator().manual_seed(seed + 17))
    eps_p = torch.randn((n, 4, 64, 64), generator=torch.Generator().manual_seed(seed))
    batch = int(cfg["batch"])
    zs = []
    with torch.inference_mode():
        for start in range(0, n, batch):
            if time.time() >= deadline - 15 or (output.parent / "STOP").exists():
                raise TimeoutError("Generation did not finish the full task set; no partial acceptance")
            sl = slice(start, min(start + batch, n))
            zc, q, y, zr, sr = [pack[k][sl].cuda().float() for k in ("zc", "q", "y", "zr", "sr")]
            with bf16_autocast("cuda"):
                if ecore is not None:
                    draft = ecore(zc, zr, q, sr, torch.full((len(zc),), job.get("lambda", 1.), device="cuda"),
                                  priors, noise=eps_e[sl].cuda(), alpha_y=job.get("alpha_y", 1.),
                                  alpha_c=job.get("alpha_c", 1.))
                else:
                    draft = reconstruct_reference_a3(zc, zr, priors, alpha_y=job.get("alpha_y", 1.),
                                                      alpha_c=job.get("alpha_c", 1.))
                if pcore is not None:
                    zo = sample_v36_ode(pcore, draft, y, q, style(sr), eps_p[sl].cuda(),
                                        tau0=job["tau"], steps=6, wj=job.get("wj", cfg["wj"]),
                                        wc=job.get("wc", cfg["wc"]))
                else:
                    zo = draft
            zs.append(zo.cpu().half())
            write(output / "progress.json", {"completed": min(start + batch, n), "total": n,
                                              "updated_epoch": time.time(), "status": "generating"})
    tmp = target.with_suffix(".tmp")
    torch.save({"z": torch.cat(zs), "job": job, "tasks": tasks,
                "source_ids": pairing["source_ids"], "pairing": pairing, "provenance": provenance}, tmp)
    tmp.replace(target)
    write(output / "progress.json", {"completed": n, "total": n, "updated_epoch": time.time(),
                                      "status": "generated", "job_sha256": fingerprint(job)})


def score(config: dict, job: dict, output: Path, deadline: float, root: Path = ROOT) -> None:
    """New process, model-free except frozen referees/decoder."""
    require_gpu_window()
    import torch
    import numpy as np
    from PIL import Image
    from src.utils import FeatureStack, decode_s10, load_s10_decoder
    from src.utils.protocol import EVAL_SEED
    saved = torch.load(output / "latents.pt", map_location="cpu", weights_only=False)
    if fingerprint(job) != fingerprint(saved["job"]):
        raise ValueError("Scoring job does not match saved generation")
    pack, tasks, pairing = load_pack(config, job, root)
    provenance = generation_provenance(config, job, pairing, int(config["evaluation"].get("seed", EVAL_SEED)), root)
    validate_provenance(provenance, saved.get("provenance"))
    if saved.get("pairing") != pairing or len(saved["z"]) != len(pairing["task_ids"]):
        raise ValueError("Saved latents do not match current task/reference identities")
    if (output / "metrics.json").exists():
        validate_provenance(provenance, read(output / "metrics.json").get("stage1_provenance"))
        return
    if job["split"] == "core750":
        from exp.v29_biglittle import RawWriterV29
        from src.eval.runner import run_measure_subprocess
        # This worker writes raw RGB only. Referee is a separate subprocess.
        dec = load_s10_decoder("cuda")
        writer = RawWriterV29(job["name"], dec=dec, device="cuda")
        writer.path = output / "raw"
        writer.path.mkdir(parents=True, exist_ok=True)
        for st in range(0, 750, config["evaluation"]["batch"]):
            if time.time() >= deadline - 15:
                raise TimeoutError("Full core750 decode deadline")
            writer.write_batch(saved["z"][st:st + config["evaluation"]["batch"]].cuda().float(),
                               saved["tasks"][st:st + config["evaluation"]["batch"]])
        writer.finalize(arm=job["name"], no_same_bypass=True)
        del writer, dec
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        run_measure_subprocess(output / "raw", output / "measured", arm_name=job["name"],
                               write_png=False, micro_batch=config["evaluation"]["batch"])
        metrics = read(output / "measured/eval.json")
        assert (metrics["n_total"], metrics["n_cross"], metrics["n_same"]) == (750, 600, 150)
        expected = {tid: i for i, tid in enumerate(pairing["task_ids"])}
        if len(metrics["rows"]) != 750 or {r["task_id"] for r in metrics["rows"]} != set(expected):
            raise ValueError("Referee output task identities differ from generation")
        for row in metrics["rows"]:
            i = expected[row["task_id"]]
            if row["src_stem"] != pairing["source_ids"][i] or int(row["tgt_style_id"]) != pairing["target_style_ids"][i]:
                raise ValueError("Referee source/target mapping changed")
            row.update(reference_id=str(pairing["reference_ids"][i]),
                       wrong_reference_id=str(pairing["wrong_reference_ids"][i]))
        metrics["stage1_provenance"] = provenance
        write(output / "metrics.json", metrics)
        return
    dec, stack = load_s10_decoder("cuda"), FeatureStack("cuda")
    def extract(z):
        chunks = {k: [] for k in ("dino", "csd", "clip")}
        for st in range(0, len(z), config["evaluation"]["batch"]):
            if time.time() >= deadline - 15:
                raise TimeoutError("Referee deadline; no incomplete score accepted")
            rgb = decode_s10(dec, z[st:st + config["evaluation"]["batch"]].cuda().float(), output_cpu=True)
            imgs = [Image.fromarray((x.permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)) for x in rgb]
            ff = stack.extract(imgs, batch=config["evaluation"]["batch"])
            for k in chunks:
                chunks[k].append(np.asarray(ff[k].cpu()) if torch.is_tensor(ff[k]) else np.asarray(ff[k]))
        return {k: np.concatenate(v) for k, v in chunks.items()}
    output_f = extract(saved["z"])
    source_f, ref_f, wrong_f = [extract(pack[k]) for k in ("zc", "zr", "zw")]
    dot = lambda k, ref: (output_f[k] * ref[k]).sum(-1)
    vals = {"dino_c": dot("dino", source_f), "dino_s": dot("dino", ref_f),
            "clip_s": dot("clip", ref_f), "csd_2way_follow": (dot("csd", ref_f) > dot("csd", wrong_f)).astype(float)}
    rows = [{"source_id": str(sid), "task_id": pairing["task_ids"][i],
             "reference_id": str(pairing["reference_ids"][i]),
             "wrong_reference_id": str(pairing["wrong_reference_ids"][i]),
             **{k: float(v[i]) for k, v in vals.items()}}
            for i, sid in enumerate(pack["source_ids"])]
    write(output / "metrics.json", {"job": job, "n": len(rows), "rows": rows,
                                    "means": {k: float(v.mean()) for k, v in vals.items()},
                                    "scope": "instance-reference; not Core5 classification",
                                    "stage1_provenance": provenance})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("generate", "score"))
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--job", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--deadline-epoch", required=True, type=float)
    a = p.parse_args()
    config, job = read(a.config), read(a.job)
    if time.time() >= a.deadline_epoch:
        raise TimeoutError("Stage deadline has already elapsed")
    root = ROOT
    output = a.output if a.output.is_absolute() else root / a.output
    (generate if a.mode == "generate" else score)(config, job, output, a.deadline_epoch, root)


if __name__ == "__main__":
    main()
