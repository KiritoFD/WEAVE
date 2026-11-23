"""Frozen six-step ICME Q32 evaluation on the original test750 with S10."""

import argparse
import hashlib
import time

import torch

from exp.v18 import tasks_and_sources
from exp.v20.run_v20 import get_test750_y16_and_d16
from exp.v27.icme_step2_closedset_accel import D5_CFG
from exp.v27.icme_step3_q32_vs_dino16 import make_bottleneck_net
from exp.v27_mainline import OUT, Q32_FLAGSHIP_CKPT, ROOT, RawWriterV27M, run_measure_v27m
from exp.v27_mainline.dir1_reflow import _integrate_q32_seg
from src.utils import load_s10_decoder
from src.utils.protocol import EVAL_SEED
from src.utils.runtime import read_json, save_json


def load_cache(relative):
    return torch.load(ROOT / relative, map_location="cpu", weights_only=False)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=24)
    args = parser.parse_args()
    device = "cuda"
    tasks, sources, _ = tasks_and_sources()
    ts = load_cache("runs/v18/cache/test750_struct_cache.pt")
    bundle = {"test_struct": ts}
    if "y_64" not in ts:
        bundle["v18_cache"] = load_cache("runs/v18/cache/ccf_pool_cache.pt")
    y, d = get_test750_y16_and_d16(bundle)
    q = load_cache("runs/v21/cache/v21_stiefel_ref_cache.pt")["test750"]["q32"]
    bank = load_cache("runs/v20/cache/core_knn_bank.pt")
    priors = load_cache("runs/v19/cache/v19_foundation.pt")["priors"]
    tid = torch.tensor([row["target_style_id"] for row in tasks])
    eps = torch.randn(sources.shape, generator=torch.Generator().manual_seed(EVAL_SEED))
    model = make_bottleneck_net(32).to(device).eval()
    model.load_state_dict(torch.load(Q32_FLAGSHIP_CKPT, map_location="cpu", weights_only=False)["ema"])
    dec = load_s10_decoder(device)
    analytic = (torch.einsum("ij,bjhw->bihw", priors["P_Y"].float(), sources.float())
                + priors["mu_c"][tid].float())
    e2full = dict(D5_CFG, ws_fig=5.5, wc_fig=1.65, wc_abs=.70, wsp_fig_late=3.6, wsp_abs=5.2)
    specs = [
        ("m1_csj", 0., set(), D5_CFG, None, False),
        ("m1_e2full", 0., {0, 1}, e2full, None, False),
        ("m2_wc1", 0., {0, 1}, e2full, 1., False),
        ("m2_wc2", 0., {0, 1}, e2full, 2., False),
        ("m2_wc3", 0., {0, 1}, e2full, 3., False),
        ("m3_analytic", .20, {0, 1}, e2full, None, False),
        ("m4_zero_vivid", 0., {0, 1}, e2full, None, True),
    ]
    result = {
        "checkpoint": Q32_FLAGSHIP_CKPT.name,
        "sha256": hashlib.sha256(Q32_FLAGSHIP_CKPT.read_bytes()).hexdigest(),
        "seed": EVAL_SEED, "weights_trained": False, "same_bypass": False,
        "arms": {},
    }
    output = OUT / "bare_q32" / "results.json"
    for label, tau, full, cfg_base, wc, zero in specs:
        t0 = time.monotonic()
        arm = f"bare_q32_{label}"
        print(f"START {arm}", flush=True)
        if not (OUT / "measured" / arm / "summary_v27m.json").is_file():
            wr = RawWriterV27M(arm, dec=dec, device=device)
            cfg = dict(cfg_base, tau_min=tau, tau_hi=tau)
            for start in range(0, len(tasks), args.batch):
                sl = slice(start, start + args.batch)
                inputs = [x[sl].to(device).float() for x in (eps, y, d, q)]
                anchor = analytic[sl].to(device) if tau else None
                z = _integrate_q32_seg(model, anchor, *inputs, tid[sl].to(device), bank,
                                       steps=6, full_steps=full, cfg_kw=cfg,
                                       wc_override=wc, zero_vivid=zero)
                wr.write_batch(z, tasks[sl])
            wr.finalize(arm=arm, tau0=tau, steps=6, same_bypass=False)
        row = run_measure_v27m(arm, device=device)
        measured = read_json(OUT / "measured" / arm / "eval.json")["rows"]
        sub = [r for r in measured if r["is_cross"] and r["tgt_style_id"] in (0, 4)]
        row["hard_target_hits"] = {
            name: [sum(r["target_top1"] for r in sub if r["tgt_style_id"] == i),
                   sum(r["tgt_style_id"] == i for r in sub)]
            for i, name in ((0, "ER"), (4, "Uki"))
        }
        row["er_uki_dino_c"] = sum(r["dino_c"] for r in sub) / len(sub)
        row.update(tau0=tau, physical_steps=6, nfe=24 + 4 * len(full),
                   full_cfg_steps=sorted(full), wc_full_override=wc, zero_vivid=zero,
                   elapsed_s=round(time.monotonic() - t0, 2))
        result["arms"][label] = row
        save_json(output, result)
    print(f"COMPLETE {output}", flush=True)


if __name__ == "__main__":
    main()
