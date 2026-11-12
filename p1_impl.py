"""P1 implementation to be appended into tools/v6_next/vae_gap_probe.py."""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
NM = [0.48145466, 0.4578275, 0.40821073]
NS = [0.26862954, 0.26130258, 0.27577711]


def rank_direction(critic, z: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """Unit-RMS direction that increases the target margin."""
    zr = z.detach().clone().requires_grad_(True)
    q = critic(zr.float()).float()
    oh = F.one_hot(tgt, len(STYLES)).bool()
    m = (q.gather(1, tgt[:, None]).squeeze(1)
         - torch.logsumexp(q.masked_fill(oh, torch.finfo(q.dtype).min), dim=1))
    m.sum().backward()
    g = zr.grad.detach()
    D = g[0].numel()
    gn = g.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
    return g / (gn / (D ** 0.5) + 1e-12)


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


def run_p1(args):
    ROOT = Path(__file__).resolve().parents[2]
    OUT = ROOT / "deliver" / "vae_gap_results.json"
    TASKS = ROOT / "eval_protocol" / "tasks_750.json"
    dev = getattr(args, "device", "cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 108)
    print("P1: LATENT EDITING-DIRECTION GEOMETRY DIAGNOSIS (100 BALANCED TASKS, GPU, NO TRAINING)")
    print("=" * 108, flush=True)

    # 1. Sample 100 balanced tasks (5 per each of the 20 cross-style pairs)
    spec = json.loads(TASKS.read_text(encoding="utf-8"))
    tasks = spec["tasks"]
    cross_tasks = [t for t in tasks if t.get("is_cross")]
    by_pair = defaultdict(list)
    for t in cross_tasks:
        by_pair[(t["source_style"], t["target_style"])].append(t)

    n_per_pair = getattr(args, "n_per_pair", 5)
    balanced_100 = []
    for pair in sorted(by_pair.keys()):
        balanced_100.extend(by_pair[pair][:n_per_pair])
    print(f"[P1.1] Selected {len(balanced_100)} balanced cross tasks across {len(by_pair)} style pairs.", flush=True)

    # 2. Check and load models
    from tools.v6.models_v6 import C3UNet
    from tools.v6.canonicalize import ContentCanonicalizer
    from tools.v6_next.probe_style import LatentStyleNet
    from diffusers import AutoencoderKL
    import lpips
    from transformers import AutoModel
    sys.path.insert(0, str(ROOT / "tools" / "12h"))
    import csd_model

    # Q2a
    q2a_p = ROOT / "runs/v6/Q2a_embed_style/Q2a_embed_style.pt"
    if not q2a_p.exists():
        q2a_p = ROOT / "deliver/checkpoints/runs/v6/Q2a_embed_style/Q2a_embed_style.pt"
    ck_q2a = torch.load(q2a_p, map_location="cpu", weights_only=False)
    gen_q2a = C3UNet(base_dim=16, num_styles=5, mode="direct").to(dev).eval()
    gen_q2a.load_state_dict(ck_q2a["model"])

    # B4
    b4_p = ROOT / "runs/v6/P1_B4_distill50/P1_B4_distill50.pt"
    if not b4_p.exists():
        b4_p = ROOT / "deliver/checkpoints/runs/v6/P1_B4_distill50/P1_B4_distill50.pt"
    ck_b4 = torch.load(b4_p, map_location="cpu", weights_only=False)
    gen_b4 = C3UNet(base_dim=16, num_styles=5, mode="direct").to(dev).eval()
    gen_b4.load_state_dict(ck_b4["model"])

    # R1 Critic
    r1_p = ROOT / "runs/v6_next/probes/r1_r1.pt"
    if not r1_p.exists():
        r1_p = ROOT / "deliver/checkpoints/runs/v6_next/probes/r1_r1.pt"
    ck_r1 = torch.load(r1_p, map_location="cpu", weights_only=False)
    critic = LatentStyleNet(width=64).to(dev).eval()
    critic.load_state_dict(ck_r1["model"])

    # Canonicalizer
    canon_p = ROOT / "runs/v6/cache/canonicalizer_stats.pt"
    c_stats = torch.load(canon_p, map_location="cpu", weights_only=False)
    canon = ContentCanonicalizer(c_stats).to(dev).eval()

    # Pre-cached latents
    lat_p = ROOT / "deliver/testset/refs/eval_src_latents.pt"
    src_lats = torch.load(lat_p, map_location="cpu", weights_only=False)

    # VAE, CSD, DINO, LPIPS
    sdvae_p = Path("/root/Workspace/xy/DiT/data/pretrained/sd-vae-ft-ema")
    if not sdvae_p.exists():
        sdvae_p = ROOT / "runs_vae/A0/final_diffusers_vae"
    vae = AutoencoderKL.from_pretrained(str(sdvae_p), torch_dtype=torch.float32).to(dev).eval()

    csd, _, _ = csd_model.load_csd(dev)
    csd = csd.float().eval()
    for p in csd.parameters():
        p.requires_grad = False

    proto_dict = torch.load(ROOT / "deliver/testset/refs/csd_proto.pt", map_location="cpu", weights_only=False)
    proto = np.stack([proto_dict[st] if isinstance(proto_dict[st], np.ndarray) else proto_dict[st].numpy() for st in STYLES])
    proto = proto / np.linalg.norm(proto, axis=1, keepdims=True)

    dino = AutoModel.from_pretrained("facebook/dinov2-base", local_files_only=True).to(dev).eval()
    lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    for p in lp.parameters():
        p.requires_grad = False

    print("[P1.2] Loaded Q2a, B4, R1, Canonicalizer, VAE, CSD, DINOv2, LPIPS.", flush=True)

    # 3. Task execution loop
    rms_budget = getattr(args, "rms", 0.25)
    alphas = [0.25, 0.50, 0.75, 1.00]
    eps_list = [0.05, 0.10, 0.25]

    records = []
    print(f"[P1.3] Running geometry diagnosis across {len(balanced_100)} tasks...", flush=True)

    batch_size = getattr(args, "batch_size", 4)
    for b_idx in range(0, len(balanced_100), batch_size):
        chunk = balanced_100[b_idx: b_idx + batch_size]
        z_src_list = [src_lats[t["source_stem"]].float() for t in chunk]
        z_src = torch.stack(z_src_list).to(dev)
        tg = torch.tensor([t["target_style_id"] for t in chunk], device=dev, dtype=torch.long)
        sg = torch.tensor([t["source_style_id"] for t in chunk], device=dev, dtype=torch.long)

        # 3.1 Q2a and B4 forward
        with torch.inference_mode():
            c = canon(z_src, mode="c1_gentle", augment=False)
            z0 = gen_q2a(c, 0.0, c, tg)
            zb = gen_b4(c, 0.0, c, tg)

        dz_b = zb - z0

        # 3.2 Oracle 1-step gradient
        g0 = rank_direction(critic, z0, tg)
        dz_o = rms_budget * g0

        # Geometric metrics
        cos_align = F.cosine_similarity(dz_b.flatten(1), dz_o.flatten(1), dim=1).cpu().numpy()
        D = dz_o[0].numel()
        rms_b = (dz_b.flatten(1).pow(2).mean(dim=1)).sqrt().cpu().numpy()
        rms_o = (dz_o.flatten(1).pow(2).mean(dim=1)).sqrt().cpu().numpy()
        rms_ratio = rms_b / (rms_o + 1e-12)

        # 3.3 Oracle gradient local stability (perturb z0 with unit RMS noise)
        stab_dict = {eps: [] for eps in eps_list}
        rng = torch.Generator(device=dev).manual_seed(42 + b_idx)
        for eps in eps_list:
            noise = torch.randn(z0.shape, device=dev, generator=rng)
            noise_rms = noise.flatten(1).norm(dim=1).view(-1, 1, 1, 1) / (D ** 0.5)
            noise = noise / (noise_rms + 1e-12)
            z_pert = z0 + eps * noise
            g_pert = rank_direction(critic, z_pert, tg)
            cos_stab = F.cosine_similarity(g0.flatten(1), g_pert.flatten(1), dim=1).cpu().numpy()
            stab_dict[eps] = cos_stab

        # 3.4 Decoded images and scoring
        # Evaluate z_src, z0, zb, and z(alpha)
        sf = 0.18215
        variants = {"z0": z0, "zb": zb}
        for a in alphas:
            variants[f"zo_{a:.2f}"] = z0 + a * dz_o

        with torch.inference_mode():
            x_src = vae.decode(z_src / sf).sample.float()
            x01_src = (x_src.clamp(-1, 1) + 1) / 2
            x224_src = F.interpolate(x01_src, (224, 224), mode="bicubic", align_corners=False)
            nm_d = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
            ns_d = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
            fd_src = F.normalize(dino(pixel_values=(x224_src - nm_d) / ns_d).last_hidden_state[:, 0].float(), dim=-1)

            scores_by_var = {}
            for v_name, z_var in variants.items():
                y_var = vae.decode(z_var / sf).sample.float()
                # CSD
                f_csd = csd_feat(csd, y_var, dev).cpu().numpy()
                sc = f_csd @ proto.T
                # DINO-C
                x01_var = (y_var.clamp(-1, 1) + 1) / 2
                x224_var = F.interpolate(x01_var, (224, 224), mode="bicubic", align_corners=False)
                fd_var = F.normalize(dino(pixel_values=(x224_var - nm_d) / ns_d).last_hidden_state[:, 0].float(), dim=-1)
                dinoc = (fd_var * fd_src).sum(dim=-1).cpu().numpy()
                # LPIPS
                l_lp = lp(y_var.clamp(-1, 1), x_src.clamp(-1, 1)).flatten().cpu().numpy()

                top1_arr = (sc.argmax(axis=1) == tg.cpu().numpy()).astype(float)
                margins = [float(sc[k, tg[k].item()] - np.delete(sc[k], tg[k].item()).max()) for k in range(len(chunk))]
                scores_by_var[v_name] = {
                    "top1": top1_arr, "margin": np.array(margins), "dino_c": dinoc, "lpips": l_lp
                }

        for i, t in enumerate(chunk):
            rec = {
                "task_id": t["task_id"], "src_stem": t["source_stem"],
                "src_style": t["source_style"], "tgt_style": t["target_style"],
                "cos_align": float(cos_align[i]), "rms_ratio": float(rms_ratio[i]),
                "rms_b": float(rms_b[i]), "rms_o": float(rms_o[i]),
                "stab": {f"eps_{eps}": float(stab_dict[eps][i]) for eps in eps_list},
                "scores": {v: {metric: float(scores_by_var[v][metric][i]) for metric in ["top1", "margin", "dino_c", "lpips"]}
                           for v in scores_by_var}
            }
            records.append(rec)

        if (b_idx // batch_size + 1) % 5 == 0 or b_idx + batch_size >= len(balanced_100):
            print(f"  Processed {min(b_idx + batch_size, len(balanced_100))}/{len(balanced_100)} tasks", flush=True)

    # 4. Aggregations
    mean_cos = float(np.mean([r["cos_align"] for r in records]))
    median_cos = float(np.median([r["cos_align"] for r in records]))
    p10_cos = float(np.percentile([r["cos_align"] for r in records], 10))
    frac_cos_pos = float(np.mean([r["cos_align"] > 0 for r in records]))

    mean_rms_b = float(np.mean([r["rms_b"] for r in records]))
    mean_rms_o = float(np.mean([r["rms_o"] for r in records]))
    mean_rms_ratio = float(np.mean([r["rms_ratio"] for r in records]))

    stab_summary = {}
    for eps in eps_list:
        v = [r["stab"][f"eps_{eps}"] for r in records]
        stab_summary[f"eps_{eps}"] = {
            "mean_cos": round(float(np.mean(v)), 4),
            "median_cos": round(float(np.median(v)), 4),
            "p10_cos": round(float(np.percentile(v, 10)), 4)
        }

    # Summary table of decoded scores across variants
    variant_names = ["z0", "zb"] + [f"zo_{a:.2f}" for a in alphas]
    decoded_summary = {}
    for v in variant_names:
        top1_m = float(np.mean([r["scores"][v]["top1"] for r in records]))
        margin_m = float(np.mean([r["scores"][v]["margin"] for r in records]))
        dinoc_m = float(np.mean([r["scores"][v]["dino_c"] for r in records]))
        lpips_m = float(np.mean([r["scores"][v]["lpips"] for r in records]))
        decoded_summary[v] = {
            "top1": round(top1_m, 4), "margin": round(margin_m, 4),
            "dino_c": round(dinoc_m, 4), "lpips": round(lpips_m, 4)
        }

    # 5. Equal-content-cost matching:
    # Find the Oracle alpha whose DINO-C is closest to B4's DINO-C
    b4_dinoc = decoded_summary["zb"]["dino_c"]
    b4_top1 = decoded_summary["zb"]["top1"]
    closest_alpha_var = min([f"zo_{a:.2f}" for a in alphas],
                            key=lambda v: abs(decoded_summary[v]["dino_c"] - b4_dinoc))
    closest_alpha_top1 = decoded_summary[closest_alpha_var]["top1"]
    closest_alpha_dinoc = decoded_summary[closest_alpha_var]["dino_c"]

    # Linear interpolation between adjacent alphas for exact DINO-C matching
    alpha_vals = [a for a in alphas]
    alpha_dinocs = [decoded_summary[f"zo_{a:.2f}"]["dino_c"] for a in alphas]
    alpha_top1s = [decoded_summary[f"zo_{a:.2f}"]["top1"] for a in alphas]

    matched_oracle_top1 = None
    for k in range(len(alpha_vals) - 1):
        d1, d2 = alpha_dinocs[k], alpha_dinocs[k+1]
        t1, t2 = alpha_top1s[k], alpha_top1s[k+1]
        if (d1 <= b4_dinoc <= d2) or (d2 <= b4_dinoc <= d1):
            ratio = (b4_dinoc - d1) / (d2 - d1 + 1e-12)
            matched_oracle_top1 = t1 + ratio * (t2 - t1)
            break
    if matched_oracle_top1 is None:
        matched_oracle_top1 = closest_alpha_top1

    # 6. Verdict determination
    if mean_cos >= 0.50 and abs(b4_top1 - matched_oracle_top1) <= 0.05:
        verdict = ("B4 IS MAINLY SHORTENING ORACLE STEP SIZE AND SITS ON THE SAME PARETO FRONTIER: "
                   f"cos(dz_B, dz_O)={mean_cos:+.3f} is high, RMS ratio={mean_rms_ratio:.2f}, "
                   f"and at matched DINO-C ({b4_dinoc:.3f}), B4 Top-1 ({b4_top1*100:.1f}%) matches Oracle "
                   f"({matched_oracle_top1*100:.1f}%). Direction is correct. TEMPORARILY DO NOT CHANGE VAE; "
                   "existing supervision has strong amortization headroom.")
    elif mean_cos < 0.30:
        verdict = ("B4 DIRECTION SEVERELY DEVIATED FROM ORACLE: "
                   f"cos(dz_B, dz_O)={mean_cos:+.3f} is poor. Prioritize modifying WEAVE distillation supervision, "
                   "not yet attributable to VAE.")
    elif stab_summary["eps_0.10"]["mean_cos"] < 0.40:
        verdict = ("ORACLE GRADIENT VOLATILE UNDER LOCAL PERTURBATION: "
                   f"stability at eps=0.10 is {stab_summary['eps_0.10']['mean_cos']:.3f}. "
                   "Teacher gradient landscape is rough, making regression hard; worth studying Encoder editing geometry.")
    else:
        verdict = f"P1 Completed: cos={mean_cos:+.3f}, RMS ratio={mean_rms_ratio:.2f}, matched Top-1={matched_oracle_top1*100:.1f}% vs B4={b4_top1*100:.1f}%."

    # Per-target style breakdown of alignment and RMS
    per_target = {}
    for st in STYLES:
        sub = [r for r in records if r["tgt_style"] == st]
        per_target[st] = {
            "n": len(sub),
            "cos_align": round(float(np.mean([r["cos_align"] for r in sub])), 4),
            "rms_ratio": round(float(np.mean([r["rms_ratio"] for r in sub])), 4),
            "b4_top1": round(float(np.mean([r["scores"]["zb"]["top1"] for r in sub])), 4),
            "oracle_1_top1": round(float(np.mean([r["scores"]["zo_1.00"]["top1"] for r in sub])), 4),
            "b4_dino_c": round(float(np.mean([r["scores"]["zb"]["dino_c"] for r in sub])), 4),
            "oracle_1_dino_c": round(float(np.mean([r["scores"]["zo_1.00"]["dino_c"] for r in sub])), 4)
        }

    p1_results = {
        "n_tasks": len(records),
        "alignment": {
            "mean_cos": round(mean_cos, 4),
            "median_cos": round(median_cos, 4),
            "p10_cos": round(p10_cos, 4),
            "frac_positive": round(frac_cos_pos, 4),
            "mean_rms_b4": round(mean_rms_b, 4),
            "mean_rms_oracle": round(mean_rms_o, 4),
            "mean_rms_ratio": round(mean_rms_ratio, 4),
        },
        "gradient_stability": stab_summary,
        "decoded_variants": decoded_summary,
        "equal_content_matched": {
            "b4_dino_c": round(b4_dinoc, 4),
            "b4_top1": round(b4_top1, 4),
            "closest_alpha_variant": closest_alpha_var,
            "closest_alpha_top1": round(closest_alpha_top1, 4),
            "matched_oracle_top1": round(matched_oracle_top1, 4),
            "gap_b4_minus_oracle": round(b4_top1 - matched_oracle_top1, 4)
        },
        "per_target": per_target,
        "verdict": verdict
    }

    # Print summary tables
    print("\n" + "=" * 92)
    print("P1 SUMMARY RESULTS")
    print("=" * 92)
    print(f"1. Direction Alignment cos(dz_B4, dz_Oracle):")
    print(f"   Mean: {mean_cos:+.4f} | Median: {median_cos:+.4f} | P10: {p10_cos:+.4f} | Frac > 0: {frac_cos_pos*100:.1f}%")
    print(f"\n2. Step Size RMS Ratio RMS(dz_B4) / RMS(dz_Oracle):")
    print(f"   RMS(B4) = {mean_rms_b:.4f} vs RMS(Oracle) = {mean_rms_o:.4f} -> Ratio = {mean_rms_ratio:.3f} (B4 takes ~{mean_rms_ratio*100:.1f}% of teacher step)")
    print(f"\n3. Oracle Gradient Stability cos(g0, g_perturbed):")
    for eps, s in stab_summary.items():
        print(f"   {eps}: Mean = {s['mean_cos']:+.4f} | Median = {s['median_cos']:+.4f} | P10 = {s['p10_cos']:+.4f}")
    print(f"\n4. Decoded Evaluation & Content Cost Trade-Off:")
    print(f"   {'Variant':<12}{'Top-1':>8}{'Margin':>9}{'DINO-C':>9}{'LPIPS':>8}")
    for v, s in decoded_summary.items():
        print(f"   {v:<12}{s['top1']*100:7.1f}%{s['margin']:+9.4f}{s['dino_c']:9.4f}{s['lpips']:8.4f}")
    print(f"\n5. Equal-Content-Cost Control:")
    print(f"   B4 (DINO-C = {b4_dinoc:.4f}): Top-1 = {b4_top1*100:.1f}%")
    print(f"   Oracle at matched DINO-C (~{b4_dinoc:.4f}): Top-1 = {matched_oracle_top1*100:.1f}% (gap = {(b4_top1 - matched_oracle_top1)*100:+.1f} pp)")
    print(f"\n6. Per-Target Style Breakdown:")
    print(f"   {'Target':<20}{'cos_align':>10}{'RMS_ratio':>11}{'B4 Top-1':>10}{'Or1 Top-1':>11}{'B4 DINO-C':>11}")
    for st, s in per_target.items():
        print(f"   {st:<20}{s['cos_align']:>+10.4f}{s['rms_ratio']:>11.3f}{s['b4_top1']*100:>9.1f}%{s['oracle_1_top1']*100:>10.1f}%{s['b4_dino_c']:>11.4f}")
    print(f"\nVERDICT: {verdict}")

    # Update output JSON
    existing = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    if "analyses" not in existing:
        existing["analyses"] = {}
    existing["analyses"]["p1_geometry"] = p1_results
    OUT.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[Saved updated results] {OUT}", flush=True)
    return p1_results
