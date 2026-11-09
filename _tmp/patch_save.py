import pathlib

p = pathlib.Path("sbflow/train5.py")
s = p.read_text(encoding="utf-8")
start = s.index("    def _equiv(step: int) -> int:")
end = s.index("    def sample_pair_batch(a, b, k):", start)
new = '''    def _equiv(step: int) -> int:
        """Equivalent exposure in patch8 / B512 update units."""
        return round(step * args.batch / max(1, args.equiv_batch))

    def save(step, tag: str = ""):
        samp = []
        for s in samplers:
            samp.append({"perm": s.perm.clone() if hasattr(s, "perm") else None,
                         "pos": getattr(s, "pos", None)})
        payload = {
            "model": model.state_dict(), "ema": ema.state_dict(),
            "optimizer": opt.state_dict(), "scheduler": sched_lr.state_dict(), "step": step,
            "equiv_step": _equiv(step),
            "style_stats": data.stats.state_dict(), "styles": data.styles,
            "model_kwargs": dict(channels=channels, image_size=H, patch=args.patch,
                                 width=args.width, depth=args.depth, heads=args.heads,
                                 num_styles=data.num_styles, style_cond=args.style_cond,
                                 heads_out=(2 if args.objective == "score_current" else args.heads_out),
                                 two_head=("score_current" if args.objective == "score_current"
                                           else "drift"),
                                 local_conv=args.local_conv, pair_rank=args.pair_rank,
                                 style_film=args.style_film),
            "objective": args.objective, "config": vars(args),
            "score_scale": s_scale.cpu(),
            "ae": data.ae, "ae_scaling_factor": data.scaling_factor,
            "rng": {"g_data": g_data.get_state(), "g_noise": g_noise.get_state(),
                    "g_time": g_time.get_state(), "g_ot": g_ot.get_state(),
                    "g_sched": g_sched.get_state(), "g_mix": g_mix.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": (torch.cuda.get_rng_state_all() if device == "cuda" else None)},
            "samplers": samp,
            "pair_sched": [list(x) for x in sched],
        }
        name = (f"model_{tag}.pt" if tag else f"model_step{step}.pt")
        torch.save(payload, out / name)
        # Also expose a stable equivalent-step alias for fork resolution.
        eq = f"equiv{_equiv(step)}"
        alias = out / f"model_{eq}.pt"
        if tag != eq and not alias.exists():
            try:
                torch.save(payload, alias)
            except Exception:
                pass

'''
p.write_text(s[:start] + new + s[end:], encoding="utf-8")
print("rewrote save()")
