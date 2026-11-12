"""V7 unit tests (CPU, tiny tensors) — run by the runner's --dry-run preflight.

Covers: relative-parameterization identity, energy losses (shape/finite/grad),
TopK per-sample mask counts, T0/T1 loop parity with apply_gated_edit, RMS
budget, gradient-balance audit, trace writer, paired-cluster bootstrap sanity,
and P4.3 gate logic.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.v7.v7_common import (  # noqa: E402
    STYLES, audit_grad_balance, model_relative, pick_trace_tasks, rms_normalize,
    s1_loss, s1_loss_per_sample, s2_loss, s2_loss_per_sample,
)

PASS, FAIL = [], []


def check(name, cond, info=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {info}", flush=True)


def fake_s1_targets(device="cpu") -> dict:
    torch.manual_seed(0)
    styles = {}
    for s in STYLES:
        ll = torch.randn(8, 4, 32, 32)
        lh = torch.randn(8, 4, 32, 32)
        flat = ll.reshape(8, 4, -1)
        flat = flat - flat.mean(dim=2, keepdim=True)
        cov = torch.bmm(flat, flat.transpose(1, 2)) / 1023
        pat_ll = torch.nn.functional.unfold(ll, kernel_size=4, stride=4).permute(0, 2, 1).reshape(-1, 64)
        proj_ll, _ = torch.sort(pat_ll @ torch.randn(64, 64).t(), dim=0)
        hf = torch.cat([lh, lh, lh], dim=1)
        pat_hf = torch.nn.functional.unfold(hf, kernel_size=4, stride=4).permute(0, 2, 1).reshape(-1, 192)
        proj_hf, _ = torch.sort(pat_hf @ torch.randn(64, 192).t(), dim=0)
        styles[s] = {
            "ll_mu": ll.mean(dim=(2, 3)).mean(dim=0), "ll_sd": ll.std(dim=(2, 3)).mean(dim=0).clamp(min=1e-4),
            "lh_mu": lh.mean(dim=(2, 3)).mean(dim=0), "lh_sd": lh.std(dim=(2, 3)).mean(dim=0).clamp(min=1e-4),
            "hl_mu": lh.mean(dim=(2, 3)).mean(dim=0), "hl_sd": lh.std(dim=(2, 3)).mean(dim=0).clamp(min=1e-4),
            "hh_mu": lh.mean(dim=(2, 3)).mean(dim=0), "hh_sd": lh.std(dim=(2, 3)).mean(dim=0).clamp(min=1e-4),
            "ll2_mu": ll.mean(dim=(2, 3)).mean(dim=0), "ll2_sd": ll.std(dim=(2, 3)).mean(dim=0).clamp(min=1e-4),
            "ll_cov": cov.mean(dim=0), "ll_cov_sqrt": cov.mean(dim=0),
            "ll_cov_invsqrt": cov.mean(dim=0),
            "qgrid_ll": proj_ll[:: max(1, proj_ll.shape[0] // 128)][:128],
            "qgrid_hf": proj_hf[:: max(1, proj_hf.shape[0] // 128)][:128],
        }
    return {"R_ll": torch.randn(64, 64), "R_hf": torch.randn(64, 192), "styles": styles}


def main() -> int:
    torch.manual_seed(0)
    device = "cpu"

    print("[1] relative parameterization identity", flush=True)
    from tools.v6.models_v6 import C3UNet

    m = C3UNet(base_dim=8, num_styles=5, mode="direct").eval()
    for p in m.parameters():
        p.requires_grad_(False)
    canon = None  # identity canon via mode none
    from tools.v6.canonicalize import ContentCanonicalizer

    canon = ContentCanonicalizer(None)
    z = torch.randn(3, 4, 64, 64)
    sid = torch.tensor([1, 2, 3])
    z_rel = model_relative(m, z, canon, "none", sid, sid)
    check("relative identity (s==t -> zs)", torch.allclose(z_rel, z, atol=1e-6))
    tid = torch.tensor([0, 1, 2])
    z_rel2 = model_relative(m, z, canon, "none", sid, tid)
    check("relative shape preserved", z_rel2.shape == z.shape)

    print("[2] S1 energy: finite, per-sample, differentiable", flush=True)
    tg = fake_s1_targets()
    zs = torch.randn(5, 4, 64, 64, requires_grad=True)
    ids = torch.tensor([0, 1, 2, 3, 4])
    l_batch = s1_loss(zs, ids, tg)
    l_ps = s1_loss_per_sample(zs, ids, tg)
    check("s1 batch scalar", l_batch.dim() == 0 and torch.isfinite(l_batch))
    check("s1 per-sample vector", l_ps.shape == (5,) and bool(torch.isfinite(l_ps).all()))
    l_ps.sum().backward()
    check("s1 gradient flows", zs.grad is not None
          and bool(torch.isfinite(zs.grad).all()))

    print("[3] S2 energy: stats + loss differentiable", flush=True)
    from tools.v6_next.probe_style import LatentStyleNet
    from tools.v7.v7_common import build_s2_style_stats, s2_stats_of
    critic = LatentStyleNet(width=8).eval()
    for p in critic.parameters():
        p.requires_grad_(False)
    ext = _TinyExt(critic)
    z2 = torch.randn(4, 4, 64, 64)
    st = s2_stats_of(ext(z2)[1]["b64"])
    check("s2 stats shapes", st["mu"].shape == (4, 8) and st["corr"].shape == (4, 8, 8))
    lat = {s: torch.randn(12, 4, 64, 64) for s in STYLES}
    stats = build_s2_style_stats(ext, lat, per_style=12, device="cpu", batch=6)
    z2g = z2.clone().requires_grad_(True)
    l2 = s2_loss(z2g, torch.tensor([0, 1, 2, 3]), stats, ext)
    l2.backward()
    check("s2 loss differentiable", torch.isfinite(l2)
          and z2g.grad is not None and bool(torch.isfinite(z2g.grad).all()))
    lp = s2_loss_per_sample(z2.detach(), torch.tensor([0, 1, 2, 3]), stats, ext)
    check("s2 per-sample vector", lp.shape == (4,) and bool(torch.isfinite(lp).all()))

    print("[4] TopK mask: per-sample position count", flush=True)
    from tools.v6_next.w_gating import topk_spatial_mask, apply_gated_edit
    g = torch.randn(3, 4, 64, 64)
    masked = topk_spatial_mask(g, 0.25)
    positions = (masked != 0).any(dim=1)  # collapse channels -> (B,H,W)
    counts = positions.flatten(1).sum(dim=1)
    check("topk25 == 1024 positions per sample",
          bool((counts == 1024).all()), f"counts={counts.tolist()}")

    print("[5] T0/T1 loop parity with apply_gated_edit", flush=True)
    from tools.v7.p2_corrector import _r1_gated_loop
    pack = {"critic": critic, "trace_task_ids": set(),
            "trace_state": {"steps": {}, "displacement": {}, "final_z": {}},
            "grad_norms": []}
    z0 = torch.randn(2, 4, 64, 64)
    tid = torch.tensor([0, 3])
    z_v7 = _r1_gated_loop(z0.clone(), tid, pack, "T0", 4, 0.08, None)
    z_ref = apply_gated_edit(z0.clone(), z0.clone(), tid, critic=critic,
                             canon=None, canon_mode="none", mode="full",
                             steps=4, per_step_rms=0.08)
    check("T0 == apply_gated_edit(full)", torch.allclose(z_v7, z_ref, atol=1e-6),
          f"maxdiff={float((z_v7-z_ref).abs().max()):.2e}")

    print("[6] energy loop RMS budget", flush=True)
    from tools.v7.p2_corrector import _energy_loop
    pack2 = {"arm": "T4", "ckpt_path": "test", "start_mode": "direct",
             "critic": critic, "feat_ext": ext,
             "canon": canon, "canon_mode": "none", "energy_name": "S1_sd",
             "energy_stats": tg,
             "content_weight": 0.5, "anchor_weight": 0.5, "r1_weight": 0.25,
             "struct_weight": 0.0, "steps": 4, "per_step_rms": 0.08,
             "trace_task_ids": set(),
             "trace_state": {"steps": {}, "displacement": {}, "final_z": {}},
             "grad_norms": []}
    z_src = torch.randn(2, 4, 64, 64)
    z_out = _energy_loop(z0.clone(), z_src, tid, pack2, 4, 0.08, None)
    disp = (z_out - z0).flatten(1).pow(2).mean(dim=1).sqrt()
    check("energy loop displacement <= 4*0.08*1.05",
          bool((disp <= 4 * 0.08 * 1.05 + 1e-6).all()), f"disp={disp.tolist()}")
    audit = audit_grad_balance(pack2["grad_norms"][0])
    check("grad audit structure", "max_over_min" in audit
          and "imbalanced" in audit)

    print("[7] trace finish writes files", flush=True)
    from tools.v7.p2_corrector import finish_v7_trace
    with tempfile.TemporaryDirectory() as td:
        pack2["trace_task_ids"] = set()
        finish_v7_trace(pack2, Path(td))
        for f in ["optim_trace.json", "grad_norm.json", "run_meta.json"]:
            check(f"written {f}", (Path(td) / f).exists())

    print("[8] trace task picking", flush=True)
    tasks = [{"task_id": f"t{i}", "is_cross": True,
              "target_style": STYLES[i % 5]} for i in range(100)]
    tr = pick_trace_tasks(tasks, 4)
    check("20 trace tasks, 4/style", len(tr) == 20)

    print("[9] paired cluster bootstrap sanity", flush=True)
    from tools.v6_24h.stats import paired_cluster_ci
    rng = np.random.default_rng(0)
    a = rng.normal(0.5, 0.1, 200)
    b = a + 0.02
    clusters = [f"c{i // 4}" for i in range(200)]
    p = paired_cluster_ci(a.tolist(), b.tolist(), clusters, n_boot=500)
    check("paired CI excludes zero for +0.02 shift", p["excludes_zero"],
          f"ci={p['ci']}")

    print("[10] P4.3 gate logic", flush=True)
    from tools.v7.p5_report import gate_status
    e = {"cross600": {"top1": 0.3, "dino_s": 0.44, "dino_c": 0.82, "lpips": 0.46,
                      "minimalism_offtarget": 0.2, "dT": 0.1},
         "same150": {}, "rows": []}
    paired = {"target_top1": {"delta": 0.1, "ci": [0.05, 0.15]},
              "dino_s": {"delta": 0.03, "ci": [0.01, 0.05]},
              "dino_c": {"delta": 0.0, "ci": [0, 0]},
              "lpips": {"delta": 0.0, "ci": [0, 0]}}
    check("gate PASS on qualifying numbers", gate_status(e, paired) == "PASS")
    paired["dino_s"] = {"delta": 0.0185, "ci": [0.015, 0.022]}
    check("gate PARTIAL below +0.020", gate_status(e, paired) == "PARTIAL")

    print(f"\nTOTAL pass={len(PASS)} fail={len(FAIL)}", flush=True)
    if FAIL:
        print("FAILED:", FAIL, flush=True)
        return 1
    return 0


class _TinyExt:
    """R1FeatureExtractor-like wrapper for CPU tests (no hook caching needed)."""

    SCALES = ("b64", "b32", "b16")

    def __init__(self, critic):
        self.critic = critic

    def __call__(self, z):
        from tools.v6_next.w_distill import R1FeatureExtractor
        if not hasattr(self, "_ext"):
            self._ext = R1FeatureExtractor(self.critic)
        return self._ext(z)


if __name__ == "__main__":
    raise SystemExit(main())
