import io, json
from pathlib import Path

ROOT = Path('.')
p = ROOT / 'tools/v6_next/vae_gap_probe.py'
s = io.open(p, encoding='utf-8').read()

anchor = '''    OUT.parent.mkdir(parents=True, exist_ok=True)'''
new = '''    # ---- F: P1 status + existing geometry evidence ----
    # P1 needs Q2a / R1 / B4 checkpoints. They are absent locally AND on the remote
    # box (deliver/ carries results only; runs/v6_next does not exist). Recorded here
    # so the gap is explicit rather than silently skipped.
    print("\\n" + "=" * 108)
    print("P0.8  P1 STATUS AND EXISTING GEOMETRY EVIDENCE")
    print("=" * 108)
    need = ["runs/v6/Q2a_embed_style/Q2a_embed_style.pt",
            "runs/v6_next/probes/r1_r1.pt",
            "runs/v6/P1_B4_distill50/P1_B4_distill50.pt"]
    missing = [w for w in need if not (ROOT / w).exists()]
    print(f"  P1 required weights present: {len(need) - len(missing)}/{len(need)}")
    for w in missing:
        print(f"    MISSING  {w}")
    p1 = {"status": "blocked" if missing else "ready", "missing": missing,
          "would_measure": ["cos(dz_B, dz_O)", "RMS(dz_B)/RMS(dz_O)",
                            "cos(dz_B, grad R1)", "oracle gradient local stability",
                            "decoded CSD / DINO-C"],
          "note": ("The decision-relevant part of P1 (equal-content-cost comparison) "
                   "was instead obtained causally from the oracle's own strength "
                   "slope, see P0.7.")}

    # L-series latent-direction geometry already on disk (different model family:
    # L11/L8 direct latent editors, NOT B4). Reported for context only.
    lgeo = {}
    for f in ["direction_L11.json", "direction_L8.json"]:
        fp = FULL / f
        if not fp.exists():
            continue
        d = json.loads(fp.read_text(encoding="utf-8"))
        lgeo[f] = {k: d.get(k) for k in
                   ["ckpt", "n", "cos_align_mean", "cos_align_p10", "m_target_mean",
                    "m_source_mean", "max_cos_mean", "knn_target_mean",
                    "tgt_acquired", "src_retained", "near_not_crossed"]}
    if lgeo:
        print("  existing latent-direction geometry (L-series, NOT B4):")
        for f, v in lgeo.items():
            print(f"    {f}: n={v['n']} cos_align={v['cos_align_mean']:+.4f} "
                  f"(p10 {v['cos_align_p10']:+.4f})  tgt_acquired="
                  f"{v['tgt_acquired']:.3f}  src_retained={v['src_retained']:.3f}  "
                  f"near_not_crossed={v['near_not_crossed']:.3f}")
        print("    -> the L-series edit direction is nearly orthogonal to the target "
              "(cos ~0.14) and crosses the style boundary in only 3% of tasks; "
              "B4 is a different family and must not inherit this verdict.")
    p1["existing_l_series_geometry"] = lgeo
    res["analyses"]["p1_status"] = p1

    OUT.parent.mkdir(parents=True, exist_ok=True)'''

assert anchor in s
s = s.replace(anchor, new, 1)
io.open(p, 'w', encoding='utf-8').write(s)
print("added P0.8")
