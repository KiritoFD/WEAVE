"""Patch r2_eval.py to add 5-style DINO score vectors + target specificity.

Adds to each row (backward-compatible: existing fields untouched):
  style_scores_5 : max cosine to each of the 5 styles' 30 reference CLS feats
  dino_s_specificity : S(z,t) - mean_{k!=t} S(z,k)

Usage (on 3060, from repo root): python3 /tmp/patch_r2eval.py
"""
import io
from pathlib import Path

P = Path("/mnt/i/Latent_Style/WEAVE/tools/v6_next/r2_eval.py")
s = io.open(P, encoding="utf-8").read()

# 1) after dino_s_gain computation, compute 5-style scores + specificity
old = """                dino_s_raw = float(sims_src.max())
                dino_s_gain = float(dino_s - dino_s_raw)

                rows.append({"""
new = """                dino_s_raw = float(sims_src.max())
                dino_s_gain = float(dino_s - dino_s_raw)

                # --- V7R_NEXT: 5-style DINO score vector + target specificity ---
                # S(z,k) = max cosine to style k's 30 reference CLS features.
                # G(z,t) = S(z,t) - mean_{k!=t} S(z,k). Answers whether a
                # DINO-S gain is target-specific or just "closer to all styles".
                style_scores_5 = []
                for _sid, _sname in enumerate(STYLES):
                    _sims = (dg[i] @ ref_dino[_sname].T).cpu().numpy()
                    # mask self-reference for same-style tasks, as above
                    if (
                        not is_cross
                        and ref_stems.get(_sname)
                        and k in ref_stems[_sname]
                    ):
                        _mi = ref_stems[_sname].index(k)
                        if _mi < len(_sims):
                            _sims[_mi] = -1e9
                    style_scores_5.append(float(_sims.max()))
                _other = [style_scores_5[j] for j in range(5) if j != tid]
                dino_s_specificity = float(
                    style_scores_5[tid] - sum(_other) / 4.0)

                rows.append({"""
assert old in s, "compute anchor not found"
s = s.replace(old, new)

# 2) add the two fields into the row dict
old2 = """                    "dino_s": dino_s,
                    "dino_s_raw": dino_s_raw,
                    "dino_s_gain": dino_s_gain,"""
new2 = """                    "dino_s": dino_s,
                    "dino_s_raw": dino_s_raw,
                    "dino_s_gain": dino_s_gain,
                    "style_scores_5": style_scores_5,
                    "dino_s_specificity": dino_s_specificity,"""
assert old2 in s, "row anchor not found"
s = s.replace(old2, new2)

io.open(P, "w", encoding="utf-8").write(s)
print("patched r2_eval.py: style_scores_5 + dino_s_specificity per row")
