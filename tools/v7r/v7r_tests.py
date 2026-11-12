"""V7-R CPU unit tests (no GPU) — run by the runner's --dry-run preflight."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.v7.v7_common import STYLES  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, info=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {info}", flush=True)


def main() -> int:
    torch.manual_seed(0)
    dev = "cpu"

    print("[1] painter parsing", flush=True)
    from tools.v7r.v7r_common import painter_of
    check("painter parsed", painter_of(
        "wikiart__Early_Renaissance__Early_Renaissance__andrea-del-castagno_crucifixion")
        == "andrea-del-castagno")
    check("anon id -> empty", painter_of("12345") == "")

    print("[2] ResidualCorrector: zero-init => identity, param budget", flush=True)
    from tools.v7r.v7r_common import ResidualCorrector, corrector_output
    for use_source in (True, False):
        H = ResidualCorrector(base_dim=48, use_source=use_source)
        n_params = sum(p.numel() for p in H.parameters())
        check(f"params in 50K-250K (use_source={use_source})",
              50_000 <= n_params <= 250_000, f"n={n_params}")
        c = torch.randn(3, 4, 64, 64)
        z0 = torch.randn(3, 4, 64, 64)
        sid = torch.tensor([0, 1, 2])
        tid = torch.tensor([2, 3, 4])
        out = H(c, z0, sid, tid)
        check(f"zero-init output == z0 (use_source={use_source})",
              torch.allclose(out, z0, atol=1e-7))
    H2 = ResidualCorrector(base_dim=48, use_source=True)
    sid = torch.tensor([0, 1, 2])
    tid = torch.tensor([0, 1, 2])
    cross = sid != tid
    out = corrector_output(H2.eval(), z0, c, sid, tid, cross)
    check("same-style keeps z0", torch.allclose(out, z0, atol=1e-7))
    with torch.no_grad():
        Ht = ResidualCorrector(base_dim=48, use_source=True)
        torch.nn.init.normal_(Ht.out_conv.weight, std=0.02)
        out2 = Ht(c, z0, torch.tensor([0, 1, 2]), torch.tensor([2, 3, 4]))
    check("trained H moves cross-style outputs", not torch.allclose(out2, z0))

    print("[3] BudgetedEditor: descent reduces a quadratic, audit recorded", flush=True)
    from tools.v7r.v7r_common import BudgetedEditor, audit_summary
    target = torch.randn(1, 4, 64, 64)
    z0 = torch.zeros(1, 4, 64, 64)

    def obj(z):
        return ((z - target) ** 2).mean()

    class FakeCanon:
        def __call__(self, z, mode="c1_gentle", augment=False):
            return z

    ed = BudgetedEditor(z0, z0.clone(), torch.tensor([0]),
                        canon=FakeCanon(), canon_mode="c1_gentle",
                        objectives={"style": obj},
                        weights={"style": 1.0, "content": 0.0, "anchor": 0.0},
                        steps=4, per_step_rms=0.08)
    z = z0
    e0 = float(obj(z))
    for _ in range(4):
        z = ed.step(z)
    e1 = float(obj(z))
    check("objective decreased", e1 < e0, f"{e0:.3f}->{e1:.3f}")
    s = audit_summary(ed.audit)
    check("audit has post-scale norms and alphas",
          "post_scale_style" in s and "alpha_hist" in s
          and s["final_displacement_rms"] > 0)

    print("[4] patch energy: finite, differentiable, real < noise", flush=True)
    from tools.v7r.p1_patchbank import build_banks, bank_softnn_loss
    banks = build_banks(device="cpu")
    ids_own = torch.tensor([0, 1, 2, 3])
    from tools.v7r.v7r_common import SD_POOL
    d = torch.load(SD_POOL / "00_Early_Renaissance.pt", map_location="cpu",
                   weights_only=False)
    zr = d["latents"].float()[100:104]  # real train-split latents (same split as banks)
    e_own = bank_softnn_loss(zr, ids_own, banks, use_ll=True)
    check("energy finite", torch.isfinite(e_own), f"{float(e_own):.3f}")
    zg = zr.clone().requires_grad_(True)
    e = bank_softnn_loss(zg, ids_own, banks, use_ll=True)
    e.backward()
    check("patch energy differentiable", zg.grad is not None
          and bool(torch.isfinite(zg.grad).all()))
    e_noise = bank_softnn_loss(torch.randn(4, 4, 64, 64), ids_own, banks,
                               use_ll=True)
    check("real latents closer to own bank than unit noise",
          float(e_own) < float(e_noise),
          f"real={float(e_own):.2f} noise={float(e_noise):.2f}")

    print("[5] randomized_canon constraints", flush=True)
    from tools.v6.canonicalize import ContentCanonicalizer
    from tools.v7r.v7r_common import randomized_canon
    canon = ContentCanonicalizer(None)
    zz = torch.randn(2, 4, 64, 64)
    c1 = randomized_canon(zz, canon)
    check("randomized canon shape+finite", c1.shape == zz.shape
          and bool(torch.isfinite(c1).all()))
    from wavelet import dwt2_haar
    _, lh_o, _, _ = dwt2_haar(canon(zz, mode="c1_gentle", augment=False))
    _, lh_r, _, _ = dwt2_haar(c1)
    check("HF amplitude randomized within [0.6,1.4] band",
          float((lh_r.std() / lh_o.std().clamp(min=1e-6)).abs()) < 2.0)

    print("[6] arm configs complete", flush=True)
    from tools.v7r.arms import ARM_CONFIGS
    need = ([f"p1a_{o}_{v}" for o in ("S1", "S2", "R1")
             for v in ("fwd", "rev", "rand", "wrong")]
            + ["E0", "E1", "E2", "E3", "E4", "R0", "NPred", "R1c", "R2c",
               "R3c", "R4c", "R5c", "R1c_s1", "R1c_s2", "R3c_noC", "R3c_disp"])
    missing = [a for a in need if a not in ARM_CONFIGS]
    check("all planned arms registered", not missing, str(missing))
    n1 = ARM_CONFIGS["R3c"]
    check("R3c uses BEST_E resolution", n1["objective"] == "BEST_E")

    print("[7] N1/N2 differ only in use_source", flush=True)
    from tools.v7r.v7r_common import save_corrector, load_corrector
    with tempfile.TemporaryDirectory() as td:
        Ha = ResidualCorrector(base_dim=48, use_source=False)
        Hb = ResidualCorrector(base_dim=48, use_source=True)
        check("state dict keys identical except none",
              set(Ha.state_dict()) == set(Hb.state_dict()))
        p = Path(td) / "n.pt"
        save_corrector(p, Ha, {"name": "N1"}, 0)
        m, ck = load_corrector(p, "cpu")
        check("corrector roundtrip", not ck["use_source"]
              and torch.allclose(m.stem.weight, Ha.stem.weight))

    print("[8] E1 conditioning arms registered", flush=True)
    from tools.v7r.arms import ARM_CONFIGS as AC2
    for a in ["E1_A", "E1_B", "E1_C", "E1_D", "E1_R1src",
              "E3A_repr", "E3B_repr"]:
        check(f"arm {a} registered", a in AC2)
    check("E1_B swaps source", AC2["E1_B"]["src_mode"] == "wrong"
          and AC2["E1_B"]["tgt_mode"] == "correct")
    check("E1_D swaps target", AC2["E1_D"]["src_mode"] == "correct"
          and AC2["E1_D"]["tgt_mode"] == "wrong")

    print("[9] E2-C consistency loss math", flush=True)
    from tools.v7r.v7r_common import ResidualCorrector as RC
    Hc = RC(base_dim=48, use_source=True).eval()
    with torch.no_grad():
        torch.nn.init.normal_(Hc.out_conv.weight, std=0.05)
    c1 = torch.randn(2, 4, 64, 64)
    c2 = torch.randn(2, 4, 64, 64)
    z0 = torch.randn(2, 4, 64, 64)
    sid2 = torch.tensor([1, 2])
    tid2 = torch.tensor([3, 4])
    with torch.no_grad():
        cons_same = ((Hc(c1, z0, sid2, tid2) - Hc(c1, z0, sid2, tid2))
                     .square().flatten(1).mean())
        cons_diff = ((Hc(c1, z0, sid2, tid2) - Hc(c2, z0, sid2, tid2))
                     .square().flatten(1).mean())
    check("consistency(same view) == 0", float(cons_same) < 1e-12)
    check("consistency(diff views) > 0", float(cons_diff) > 0)

    print("[10] E3 representation head + supcon", flush=True)
    from tools.v7r.e3_repr import StyleReprHead, supcon_loss
    head = StyleReprHead(in_dim=192, dim=64)
    n_par = sum(pp.numel() for pp in head.parameters())
    check("head params in 50K-300K", 50_000 <= n_par <= 300_000, f"n={n_par}")
    emb = torch.randn(6, 192)
    e = head.embed(emb)
    check("embed normalized", torch.allclose(e.norm(dim=1),
                                             torch.ones(6), atol=1e-5))
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    painters = torch.tensor([1, 1, 2, 3, 4, 5])
    l0 = supcon_loss(e, labels, painters)
    check("supcon finite (has cross-painter positives)",
          torch.isfinite(l0) and float(l0) > 0)
    painters_same = torch.tensor([1, 1, 3, 3, 5, 5])  # per-style single painter
    l_none = supcon_loss(e, labels, painters_same)
    check("supcon == 0 without cross-painter positives", float(l_none) == 0.0)

    print("[11] eta accounting runs", flush=True)
    import subprocess
    r = subprocess.run([sys.executable, "-m", "tools.v7r.eta"],
                       capture_output=True, text=True, cwd=str(ROOT))
    check("eta module executes", r.returncode == 0, r.stdout[-120:])
    eta_p = ROOT / "runs" / "v7r" / "eta.json"
    if eta_p.exists():
        d = json.loads(eta_p.read_text(encoding="utf-8"))
        check("eta budget = 57600", d["BUDGET_TOTAL_SEC"] == 57600)
        check("eta remaining > 0 on a fresh run", d["ETA_REMAINING_SEC"] > 0)

    print(f"\nTOTAL pass={len(PASS)} fail={len(FAIL)}", flush=True)
    if FAIL:
        print("FAILED:", FAIL, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
