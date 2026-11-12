"""Patch r2_eval.py: add --gen v7r_next dispatch (idempotent).

Keeps all existing branches untouched. New optional arg --v7r-next-arm.
"""
import io
from pathlib import Path

P = Path("/mnt/i/Latent_Style/WEAVE/tools/v6_next/r2_eval.py")
s = io.open(P, encoding="utf-8").read()
if "v7r_next" in s:
    print("already patched")
else:
    old = """    elif args.gen == "v7r":
        # V7-R arms (P1 probes / E-objectives / P3 correctors / N predictors)
        from tools.v7r.arms import build_v7r_pack

        return build_v7r_pack(args, device)
"""
    new = old + """
    elif args.gen == "v7r_next":
        # WEAVE-CPT / V7R_NEXT arms (constrained CPT editor / distilled student)
        from tools.v7r_next.gen_v7r_next import build_pack

        return build_pack(args, device)
"""
    assert old in s, "gen==v7r anchor not found"
    s = s.replace(old, new)

    old2 = """    if kind == "v7r":
        from tools.v7r.arms import run_v7r_edit

        return run_v7r_edit(gen_pack, z_batch, sid_batch, tid_batch, task_batch)
"""
    new2 = old2 + """
    if kind == "v7r_next":
        from tools.v7r_next.gen_v7r_next import run_edit

        return run_edit(gen_pack, z_batch, sid_batch, tid_batch, task_batch)
"""
    assert old2 in s, "kind==v7r anchor not found"
    s = s.replace(old2, new2)

    old3 = '    ap.add_argument("--v7r-arm", default="R0")\n'
    new3 = old3 + '    ap.add_argument("--v7r-next-arm", default="T2")\n'
    assert old3 in s, "v7r-arm arg anchor not found"
    s = s.replace(old3, new3)
    io.open(P, "w", encoding="utf-8").write(s)
    print("patched r2_eval.py: --gen v7r_next + --v7r-next-arm")
