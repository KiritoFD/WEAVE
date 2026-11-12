"""Analyse the Style-Rank collapse: trajectory of sim_p/sim_n, |h|, KL."""
import json
from pathlib import Path

P = Path("/root/Workspace/WEAVE_VAE/runs/vae_style_rank20/train_metrics.jsonl")
rows, vals, stale, drift = [], [], [], []
for line in P.read_text(errors="ignore").splitlines():
    try:
        d = json.loads(line)
    except Exception:
        continue
    if "upd" in d and "sim_pos" in d:
        rows.append(d)
    elif "val" in d:
        vals.append(d["val"])
    elif "stale" in d:
        stale.append(d["stale"])
    elif "drift" in d:
        drift.append(d["drift"])

print(f"training points: {len(rows)}  (upd {rows[0]['upd']}..{rows[-1]['upd']})")
print(f"\n{'upd':>5}{'ep':>6}{'rec':>8}{'kl':>7}{'rank':>7}{'ce':>7}"
      f"{'sim_p':>8}{'sim_n':>8}{'marg':>8}{'racc':>7}{'|h|':>7}{'mt':>6}")
print("-" * 86)
for r in rows:
    if r["upd"] % 50 == 0 or r["upd"] <= 20:
        print(f"{r['upd']:>5}{r.get('epoch',0):>6.2f}{r['rec']:>8.4f}{r['kl']:>7.2f}"
              f"{r['rank']:>7.4f}{r['ce']:>7.3f}{r['sim_pos']:>8.4f}{r['sim_neg']:>8.4f}"
              f"{r['sim_pos']-r['sim_neg']:>8.4f}{r['rank_acc']:>7.2f}{r['h_norm']:>7.1f}"
              f"{r.get('match_lt005',0):>6.2f}")

print("\n=== validation ===")
for v in vals:
    print(f"  upd {v['upd']:>5} ep{v['epoch']:.2f} | CM={v['content_matched']} "
          f"Cross={v['cross_artist']} Hard={v['same_genre_hard']} "
          f"LPIPS={v['lpips']} SSIM={v['ssim']}")

print("\n=== staleness ===")
for s in stale:
    print(f"  upd {s['upd']:>5}: d_stale={s['d_stale']:.5f} "
          f"|ds|={s['d_sim_to_anchor']:.5f} = {s['ratio_vs_margin']*100:.1f}% of margin")

print("\n=== gradient drift ===")
for d in drift:
    print(f"  upd {d['upd']:>5}: rank/base={d['ratio_rank_base']:.3f} "
          f"ce/base={d['ratio_ce_base']:.3f} cos={d['cos_rank_base']:+.3f}")

# collapse onset detection
print("\n=== collapse onset (first upd where sim_p > 0.5) ===")
for r in rows:
    if r["sim_pos"] > 0.5:
        print(f"  first at upd {r['upd']} (ep {r.get('epoch',0):.2f}): "
              f"sim_p={r['sim_pos']:.4f} sim_n={r['sim_neg']:.4f} |h|={r['h_norm']:.1f} "
              f"kl={r['kl']:.2f} racc={r['rank_acc']:.2f}")
        break
else:
    print("  not observed")

print("\n=== |h| and KL trajectory (every 100) ===")
for r in rows:
    if r["upd"] % 100 == 0:
        print(f"  upd {r['upd']:>5}: |h|={r['h_norm']:>6.1f} kl={r['kl']:>6.2f} "
              f"sim_p={r['sim_pos']:>7.4f} sim_n={r['sim_neg']:>7.4f}")
