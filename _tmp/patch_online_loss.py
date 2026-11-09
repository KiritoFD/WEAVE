import pathlib

p = pathlib.Path("sbflow/online_bridge.py")
s = p.read_text(encoding="utf-8")
old_start = s.index("    if replay is not None:\n        x1_hat = replay.sample")
old_end = s.index("\n    # optional anchoring on the original independent coupling (exp 28)", old_start)
replacement = '''    if use_replay and replay is not None:
        x1_hat = replay.sample(a, b, "fwd", len(z0), z0.device, gen)
        x0_hat = replay.sample(a, b, "rev", len(z1), z1.device, gen)
        if x1_hat is None:
            x1_hat = sampler(teacher, z0, a, b, st, eps, steps, t_floor, gen)
            replay.push(a, b, "fwd", x1_hat)
        if x0_hat is None:
            x0_hat = sde_reverse(teacher, z1, a, b, st, eps, steps, t_floor, gen)
            replay.push(a, b, "rev", x0_hat)
    else:
        x1_hat = sampler(teacher, z0, a, b, st, eps, steps, t_floor, gen)
        x0_hat = sde_reverse(teacher, z1, a, b, st, eps, steps, t_floor, gen)
'''
s = s[:old_start] + replacement + s[old_end:]
p.write_text(s, encoding="utf-8")
print("restored online_loss replay branch")
