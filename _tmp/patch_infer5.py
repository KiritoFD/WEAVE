import pathlib
import re

p = pathlib.Path("sbflow/infer5.py")
s = p.read_text(encoding="utf-8")

# scale every raw network output by residual_scale so 0 gives the pure analytic EDM drift
s = s.replace(
    "        raw = model((z - mx) * ci, tt, torch.ones_like(sa), sa, sb).float()",
    "        raw = model((z - mx) * ci, tt, torch.ones_like(sa), sa, sb).float() * rs")
s = s.replace(
    "        pv, ps = model((z - mx) * ci, tt, torch.ones_like(sa), sa, sb, return_both=True)",
    "        pv, ps = model((z - mx) * ci, tt, torch.ones_like(sa), sa, sb, return_both=True)\n"
    "        pv = pv * rs")
s = s.replace(
    "        raw_f = model((z - mx) * ci, tt, torch.ones_like(sa), sa, sb).float()",
    "        raw_f = model((z - mx) * ci, tt, torch.ones_like(sa), sa, sb).float() * rs")
s = s.replace(
    "        raw_b = model((z - mxr) * cir, 1 - tt, torch.zeros_like(sa), sa, sb).float()",
    "        raw_b = model((z - mxr) * cir, 1 - tt, torch.zeros_like(sa), sa, sb).float() * rs")
s = s.replace(
    "    raw_f = model((z - mxf) * cif, tt, torch.ones_like(sa), sa, sb).float()",
    "    raw_f = model((z - mxf) * cif, tt, torch.ones_like(sa), sa, sb).float() * rs")
s = s.replace(
    "    raw_b = model(xb, 1 - tt, torch.zeros_like(sa), sa, sb).float()",
    "    raw_b = model(xb, 1 - tt, torch.zeros_like(sa), sa, sb).float() * rs")

# thread the extra kwargs through the solver
s = s.replace(
    """def solve_transfer(model, z, src_style: int, tgt_style: int, stats: StyleStats,
                   eps: float, steps: int = 4, method: str = "heun",
                   t_floor: float = 0.002, objective: str = "bidir",
                   reverse_override: Optional[bool] = None) -> torch.Tensor:""",
    """def solve_transfer(model, z, src_style: int, tgt_style: int, stats: StyleStats,
                   eps: float, steps: int = 4, method: str = "heun",
                   t_floor: float = 0.002, objective: str = "bidir",
                   reverse_override: Optional[bool] = None,
                   cond_a: Optional[int] = None, cond_b: Optional[int] = None,
                   residual_scale: float = 1.0) -> torch.Tensor:""")
s = s.replace(
    "        k0 = probability_flow_velocity(model, out, t0, a, b, stats, eps, t_floor, objective)",
    "        k0 = probability_flow_velocity(model, out, t0, a, b, stats, eps, t_floor, objective,\n"
    "                                      cond_a, cond_b, residual_scale)")
s = s.replace(
    """            k1 = probability_flow_velocity(model, out + h * k0, t1, a, b, stats, eps,
                                           t_floor, objective)""",
    """            k1 = probability_flow_velocity(model, out + h * k0, t1, a, b, stats, eps,
                                           t_floor, objective, cond_a, cond_b,
                                           residual_scale)""")
p.write_text(s, encoding="utf-8")
print("patched infer5")
