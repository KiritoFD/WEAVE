"""Fixed Seen-Dev same-palette physical-response probes, never a quality gate.

The explicit GPU worker generates A/B/wrong-reference outputs and uses only
deterministic Haar/Sobel/RGB diagnostics. No DINO/CSD/CLIP/LPIPS referee is loaded.
Pair selection uses cached real references, not edited/recoloured reference images.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from exp.stage1_recipe.config import ROOT, read, resolve, write


def choose_pairs(stat, records, n=64, palette_quantile=.25):
    """Deterministic, output-blind near-palette/high-Haar reference pairs.

    Oklab coordinates are cache[0:3] means and [3:6] std; Haar is [27:33].
    Distances use Seen-Dev per-coordinate std so luminance does not dominate
    merely by its units. Near means the lowest within-movement distance quartile,
    and high-Haar means the largest differences among those eligible real pairs.
    Actual raw/standardized distances and selection threshold are retained.
    """
    stat = np.asarray(stat, dtype=np.float64)
    if len(stat) != len(records) or stat.ndim != 2 or stat.shape[1] != 1561:
        raise ValueError("Expected aligned Seen-Dev records and N x 1561 cached descriptors")
    if not np.isfinite(stat[:, :33]).all():
        raise ValueError("Nonfinite reference palette/Haar descriptors")
    palette_scale = np.maximum(stat[:, :6].std(0), 1e-6)
    haar_scale = np.maximum(stat[:, 27:33].std(0), 1e-6)
    by_style = {}
    for i, row in enumerate(records):
        by_style.setdefault(row["style"], []).append(i)
    eligible, thresholds = [], {}
    for style, indices in sorted(by_style.items()):
        candidates = []
        for p, a in enumerate(indices):
            for b in indices[p + 1:]:
                dp = stat[a, :6] - stat[b, :6]
                dh = stat[a, 27:33] - stat[b, 27:33]
                candidates.append({"ref_a": int(a), "ref_b": int(b), "style": style,
                    "palette_distance": float(np.linalg.norm(dp / palette_scale)),
                    "palette_raw_l2": float(np.linalg.norm(dp)),
                    "haar_distance": float(np.linalg.norm(dh / haar_scale)),
                    "palette_a_oklab_mean_std": stat[a, :6].tolist(),
                    "palette_b_oklab_mean_std": stat[b, :6].tolist()})
        if not candidates:
            continue
        threshold = float(np.quantile([r["palette_distance"] for r in candidates], palette_quantile))
        thresholds[style] = threshold
        eligible.extend([{**r, "within_style_palette_threshold": threshold} for r in candidates
                         if r["palette_distance"] <= threshold])
    eligible.sort(key=lambda r: (-r["haar_distance"], r["palette_distance"], r["style"], r["ref_a"], r["ref_b"]))
    if len(eligible) < n:
        raise ValueError(f"Only {len(eligible)} real near-palette same-movement pairs; require {n}, no OOD fallback")
    return eligible[:n], {"source": "seen_dev_real_cached_stat1561", "n": n,
        "near_palette_definition": "within-movement lowest palette-distance quartile",
        "palette_quantile": palette_quantile, "eligible_pair_count": len(eligible),
        "rank": "largest standardized Haar6 distance, then palette distance and IDs",
        "pair_selection_uses_outputs": False, "reference_recolouring": False,
        "palette_dimensions": [0, 6], "haar_dimensions": [27, 33],
        "palette_scale": palette_scale.tolist(), "haar_scale": haar_scale.tolist(),
        "thresholds_by_movement": thresholds}


def direction_cosine(actual, expected, eps=1e-10):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    norms = np.linalg.norm(a), np.linalg.norm(b)
    return None if min(norms) <= eps else float(np.dot(a, b) / (norms[0] * norms[1]))


def _gray(rgb):
    return .299 * rgb[:, :1] + .587 * rgb[:, 1:2] + .114 * rgb[:, 2:3]


def physical_features(rgb):
    """Return per-image grayscale signatures and raw spatial content arrays."""
    import torch
    import torch.nn.functional as F
    from src.utils.structure import dwt2_haar
    g = _gray(rgb.float()).clamp(0, 1)
    g = F.interpolate(g, (256, 256), mode="bilinear", align_corners=False)
    norm = (g - g.mean((2, 3), keepdim=True)) / g.std((2, 3), keepdim=True, unbiased=False).clamp_min(.02)
    lo, *hi = dwt2_haar(norm)
    _, *mid = dwt2_haar(lo)
    haar = torch.cat([v.square().mean((2, 3)).clamp_min(1e-8).log() for v in hi + mid], 1)
    kx = norm.new_tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3) / 4
    gx, gy = F.conv2d(norm, kx, padding=1), F.conv2d(norm, kx.transpose(-1, -2), padding=1)
    mag = (gx.square() + gy.square() + 1e-8).sqrt()
    ang = torch.atan2(gy, gx) % torch.pi
    bins = (ang / torch.pi * 8).long().clamp(0, 7).flatten(1)
    hist = norm.new_zeros(len(norm), 8)
    hist.scatter_add_(1, bins, mag.flatten(1))
    hist /= hist.sum(1, keepdim=True).clamp_min(1e-8)
    sobel = torch.cat([hist, mag.mean((2, 3)), mag.std((2, 3), unbiased=False)], 1)
    return {"haar": haar, "sobel": sobel,
            "gray64": F.interpolate(g, (64, 64), mode="area"), "gray256": g,
            "gray_unit64": F.interpolate(norm, (64, 64), mode="area")}


def response_row(source, ref_a, ref_b, wrong_ref, out_a, out_b, out_wrong):
    import torch
    from src.utils.structure import dwt2_haar, extract_color_fiber_27d
    tensors = [source, ref_a, ref_b, wrong_ref, out_a, out_b, out_wrong]
    f = [physical_features(x) for x in tensors]
    sf, af, bf, wf, oa, ob, ow = f
    result = {"output_ab_rgb_mae": float((out_a - out_b).abs().mean()),
              "output_ab_gray_mae": float((_gray(out_a) - _gray(out_b)).abs().mean()),
              "output_a_vs_wrong_rgb_mae": float((out_a - out_wrong).abs().mean()),
              "output_a_vs_wrong_gray_mae": float((_gray(out_a) - _gray(out_wrong)).abs().mean()),
              "decoded_reference_palette_l2": float((extract_color_fiber_27d(ref_a)[:, :6] -
                                                      extract_color_fiber_27d(ref_b)[:, :6]).norm())}
    for key in ("haar", "sobel"):
        to_numpy = lambda x: x.detach().float().cpu().numpy().reshape(-1)
        expected, actual = to_numpy(bf[key] - af[key]), to_numpy(ob[key] - oa[key])
        follow = .5 * (float((oa[key] - af[key]).square().mean() < (oa[key] - bf[key]).square().mean())
                       + float((ob[key] - bf[key]).square().mean() < (ob[key] - af[key]).square().mean()))
        result[key] = {"reference_delta": expected.tolist(), "output_delta": actual.tolist(),
                       "reference_delta_norm": float(np.linalg.norm(expected)),
                       "output_delta_norm": float(np.linalg.norm(actual)),
                       "direction_cosine": direction_cosine(actual, expected),
                       "physical_signature_2way_follow": follow,
                       "correct_a_distance": float((oa[key] - af[key]).square().mean().sqrt()),
                       "wrong_condition_to_a_distance": float((ow[key] - af[key]).square().mean().sqrt()),
                       "wrong_ref_distance": float((ow[key] - wf[key]).square().mean().sqrt())}
    for label, output, feat in (("a", out_a, oa), ("b", out_b, ob), ("wrong", out_wrong, ow)):
        residual = _gray(output - source)
        low, *high = dwt2_haar(residual)
        ll, *middle = dwt2_haar(low)
        result[label] = {"gray_content_l1_64": float((feat["gray64"] - sf["gray64"]).abs().mean()),
                         "gray_contrast_normalized_content_l1_64": float((feat["gray_unit64"] - sf["gray_unit64"]).abs().mean()),
                         "rgb_residual_rms": float((output - source).square().mean().sqrt()),
                         "gray_residual_bands_rms": [float(v.square().mean().sqrt()) for v in [ll, *middle, *high]],
                         "band_order": ["LL2", "LH2", "HL2", "HH2", "LH1", "HL1", "HH1"]}
    return result


def save_sheet(samples, output):
    """Fixed first eight pair IDs; no generated-score selection or recolouring."""
    from PIL import Image, ImageDraw
    columns = ["source", "ref A", "ref B", "output A", "output B", "wrong ref", "output wrong"]
    size, label_h = 160, 24
    rgb = Image.new("RGB", (len(columns) * size, len(samples) * (size + label_h)), "white")
    crop = Image.new("RGB", rgb.size, "white")
    dr, dc = ImageDraw.Draw(rgb), ImageDraw.Draw(crop)
    for row, sample in enumerate(samples):
        for col, (title, tensor) in enumerate(zip(columns, sample)):
            arr = (tensor.detach().float().cpu().squeeze(0).permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
            im = Image.fromarray(arr)
            x, y = col * size, row * (size + label_h)
            dr.text((x + 4, y + 4), f"{row:02d} {title}", fill="black")
            dc.text((x + 4, y + 4), f"{row:02d} {title} gray", fill="black")
            rgb.paste(im.resize((size, size), Image.Resampling.BILINEAR), (x, y + label_h))
            cx, cy = im.width // 2, im.height // 2
            gray_crop = im.convert("L").crop((cx - 64, cy - 64, cx + 64, cy + 64))
            crop.paste(gray_crop.convert("RGB").resize((size, size)), (x, y + label_h))
    rgb.save(output / "fixed8_rgb.png")
    crop.save(output / "fixed8_gray_crop.png")


def run(config, job, output, deadline, root=ROOT):
    from exp.stage1_recipe.evaluation.generate import (
        file_digest, fingerprint, generation_provenance, load_models, load_pack,
        require_gpu_window, validate_provenance)
    require_gpu_window()
    import torch
    from exp.v36_art_foundation.eval_v36 import reconstruct_reference_a3, sample_v36_ode
    from src.utils import decode_s10, load_s10_decoder
    from src.utils.protocol import EVAL_SEED
    from src.utils.runtime import bf16_autocast

    if job.get("split", "seen_dev") != "seen_dev":
        raise ValueError("Physical probe pair selection is Seen-Dev only; no Core/OOD filtering")
    probe_job = {**job, "split": "seen_dev", "n": 0}
    pack, _, _ = load_pack(config, probe_job, root)
    cache = Path(config["paths"]["cache_root"])
    if not cache.is_absolute(): cache = root / cache
    ready, manifest = read(cache / "v36_caches_ready.json"), read(cache / "curation_manifest.json")
    info, records = ready["dev_info"], manifest["seen_dev"]
    arrays = {k: np.load(root / info["paths"][k], mmap_mode="r") for k in ("z_gt", "stat")}
    pairs, method = choose_pairs(arrays["stat"], records)
    seed = int(config["evaluation"].get("seed", EVAL_SEED))
    wrong_rng = np.random.default_rng(seed + 31)
    available = list(enumerate(pack["source_ids"]))
    selected = []
    used = set()
    for pair in pairs:
        found = next(((pos, int(sid)) for pos, sid in available if int(sid) not in used and
                      records[int(sid)]["style"] != pair["style"]), None)
        if found is None:
            raise ValueError("Not enough distinct Seen-Dev cross-movement sources")
        pos, sid = found
        used.add(sid)
        wrongs = [i for i, rec in enumerate(records) if rec["style"] not in
                  (records[sid]["style"], pair["style"]) and i not in (pair["ref_a"], pair["ref_b"])]
        if not wrongs:
            raise ValueError("No third-movement wrong reference")
        wrong = int(wrong_rng.choice(wrongs))
        selected.append({**pair, "source_id": sid, "pack_position": pos, "wrong_reference_id": int(wrong)})
    pairing = {"task_ids": [f"physical64:{i}" for i in range(64)],
               "source_ids": [r["source_id"] for r in selected],
               "reference_ids": [r["ref_a"] for r in selected],
               "reference_b_ids": [r["ref_b"] for r in selected],
               "wrong_reference_ids": [r["wrong_reference_id"] for r in selected]}
    generation = generation_provenance(config, probe_job, pairing, seed, root)
    meta = {"generation": generation, "pair_selection": method, "pairs": selected,
            "implementation_sha256": file_digest(Path(__file__)),
            "ecore_implementation_sha256": file_digest(root / "exp/v36_art_foundation/ecore_model.py"),
            "decoder_sha256": file_digest(root / "deliver/checkpoints/runs/vae_decoder_prune/s10_p5_lowlr/decoder.pt"),
            "scope": "physical_response_only; not style quality or adoption evidence"}
    provenance = {**meta, "signature": fingerprint(meta)}
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "probe.json"
    if result_path.exists():
        existing = read(result_path)
        validate_provenance(provenance, existing.get("provenance"))
        if not all((output / p).exists() for p in existing["samples"]):
            raise ValueError("Existing probe lacks its fixed PNG evidence")
        return
    p, style, ec, priors = load_models(config, job)
    dec = load_s10_decoder("cuda")
    eps_e = torch.randn((64, 4, 64, 64), generator=torch.Generator().manual_seed(seed + 17))
    eps_p = torch.randn((64, 4, 64, 64), generator=torch.Generator().manual_seed(seed))
    rows, samples = [], []
    alpha_y, alpha_c = job.get("alpha_y", 1.), job.get("alpha_c", 1.)
    with torch.inference_mode():
        for i, pair in enumerate(selected):
            if time.time() >= deadline - 20 or (output.parent / "STOP").exists():
                raise TimeoutError("Physical probe incomplete; no partial result passes review")
            pos = pair["pack_position"]
            zc, q, y = [pack[k][pos:pos + 1].cuda().float() for k in ("zc", "q", "y")]
            outputs, refs = [], []
            for ref_id in (pair["ref_a"], pair["ref_b"], pair["wrong_reference_id"]):
                zr = torch.from_numpy(np.array(arrays["z_gt"][ref_id:ref_id + 1])).cuda().float()
                sr = torch.from_numpy(np.array(arrays["stat"][ref_id:ref_id + 1])).cuda().float()
                with bf16_autocast("cuda"):
                    draft = ec(zc, zr, q, sr, torch.full((1,), job.get("lambda", 1.), device="cuda"),
                               priors, noise=eps_e[i:i + 1].cuda(), alpha_y=alpha_y, alpha_c=alpha_c) if ec else (
                               reconstruct_reference_a3(zc, zr, priors, alpha_y=alpha_y, alpha_c=alpha_c))
                    zo = sample_v36_ode(p, draft, y, q, style(sr), eps_p[i:i + 1].cuda(), tau0=job.get("tau", .5),
                                       steps=6, wj=job.get("wj", config["evaluation"]["wj"]),
                                       wc=job.get("wc", config["evaluation"]["wc"])) if p else draft
                outputs.append(decode_s10(dec, zo, batch=1, output_cpu=False))
                refs.append(decode_s10(dec, zr, batch=1, output_cpu=False))
            source = decode_s10(dec, zc, batch=1, output_cpu=False)
            row = response_row(source, refs[0], refs[1], refs[2], *outputs)
            rows.append({"pair": i, **pair, "diagnostic": row})
            if i < 8:
                samples.append([x.cpu() for x in (source, refs[0], refs[1], outputs[0], outputs[1], refs[2], outputs[2])])
            write(output / "progress.json", {"status": "physical_probing", "done": i + 1, "total": 64,
                                              "updated_epoch": time.time()})
    save_sheet(samples, output)
    mean = lambda key: float(np.mean([r["diagnostic"][key] for r in rows]))
    summary = {"output_ab_rgb_mae": mean("output_ab_rgb_mae"),
               "output_ab_gray_mae": mean("output_ab_gray_mae"),
               "output_a_vs_wrong_rgb_mae": mean("output_a_vs_wrong_rgb_mae"),
               "output_a_vs_wrong_gray_mae": mean("output_a_vs_wrong_gray_mae"),
               "decoded_reference_palette_l2": mean("decoded_reference_palette_l2")}
    for key in ("haar", "sobel"):
        valid = [r["diagnostic"][key]["direction_cosine"] for r in rows
                 if r["diagnostic"][key]["direction_cosine"] is not None]
        summary[key] = {"direction_cosine": float(np.mean(valid)) if valid else None,
                        "direction_valid_pairs": len(valid),
                        "physical_signature_2way_follow": float(np.mean([r["diagnostic"][key]["physical_signature_2way_follow"] for r in rows]))}
    write(result_path, {"status": "complete", "n": 64, "job": job, "seed": seed,
          "provenance": provenance, "pairs": rows, "summary": summary,
          "samples": ["fixed8_rgb.png", "fixed8_gray_crop.png"],
          "metric_scope": "Unlearned grayscale Haar/Sobel/RGB response; not CSD/DINO or perceptual quality",
          "does_not_establish": ["brushwork correctness", "quality improvement", "generalization improvement"],
          "automatic_adoption": False, "requires_side_by_side_human_review": True})
    write(output / "review_template.json", {"gray_style_improved": None, "color_matched_improved": None,
          "no_new_artifacts": None, "reviewer": None, "evidence": [], "notes": "Compare candidate and baseline fixed8 sheets at the same content level; response alone is insufficient.",
          "probe_signature": provenance["signature"], "baseline_probe_signature": None})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--deadline-epoch", required=True, type=float)
    args = parser.parse_args()
    if time.time() >= args.deadline_epoch:
        raise TimeoutError("Physical-probe deadline already elapsed")
    root = ROOT
    output = args.output if args.output.is_absolute() else root / args.output
    run(read(args.config), read(args.job), output, args.deadline_epoch, root)


if __name__ == "__main__":
    main()
