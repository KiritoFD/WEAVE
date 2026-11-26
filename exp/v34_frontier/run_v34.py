"""CLEAVE V34 Unified CLI Runner: Smoke Test, Stage A, Stage B, Stage C, and Stage D."""
from __future__ import annotations

import argparse

from exp.v34_frontier import OUT
from exp.v34_frontier.stage_a_startpoint import run_stage_a
from exp.v34_frontier.stage_b_cond import run_stage_b
from exp.v34_frontier.stage_c_capacity import run_stage_c
from exp.v34_frontier.stage_d_merge import run_stage_d
from src.utils.runtime import save_json, utc


def run_smoke(device: str = "cuda") -> dict:
    """Fast 10-step GPU smoke verification covering Stages A, B2, and 44M C3."""
    from exp.v34_frontier.train_common import build_or_load_style_instances, train_pcore_v34

    build_or_load_style_instances(device=device)
    b2 = train_pcore_v34(
        "smoke_B2_inst_tex1101",
        OUT / "smoke/B2",
        base_dim=144,
        cond_dim=320,
        style_mode="inst_1101",
        use_adapter=False,
        updates=100,
        smoke_steps=10,
        device=device,
    )
    c3 = train_pcore_v34(
        "smoke_C3_44m_s2",
        OUT / "smoke/C3",
        base_dim=208,
        cond_dim=384,
        style_mode="k8",
        use_adapter=True,
        updates=100,
        smoke_steps=10,
        device=device,
    )
    rep = {"updated_utc": utc(), "smoke_B2": b2, "smoke_C3": c3}
    save_json(OUT / "v34_smoke_report.json", rep)
    print(f"[V34 Smoke Done] B2 VRAM={b2['peak_vram_gb']:.2f}GB | C3 VRAM={c3['peak_vram_gb']:.2f}GB", flush=True)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["smoke", "a", "b", "c", "d", "all"], required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--c-updates", type=int, default=3500)
    ap.add_argument("--b-updates", type=int, default=3500)
    ap.add_argument("--d-updates", type=int, default=4500)
    args = ap.parse_args()

    if args.stage == "smoke":
        run_smoke(device=args.device)
        return
    if args.stage in ("a", "all"):
        run_stage_a(device=args.device)
    if args.stage in ("b", "all"):
        run_stage_b(updates=args.b_updates, device=args.device)
    if args.stage in ("c", "all"):
        run_stage_c(updates=args.c_updates, device=args.device)
    if args.stage in ("d", "all"):
        run_stage_d(updates=args.d_updates, device=args.device)


if __name__ == "__main__":
    main()
