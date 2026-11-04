"""S15 / S16: is the legacy FM supervision real, and does de-styling actually work?

S15  fixed content + fixed style condition; compare velocity labels built from
     multiple real targets: within-style randomness vs between-style signal.
S16  residual style left in b2_destyle.destyle_latent(x): probe accuracy and
     offline DINO-S of the de-styled decode.

Usage: python3 tools/audit/e_legacy.py [s15] [s16]
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import audit_core as C


def load_model():
    from diagnose_saturation import (_load_checkpoint, FlowMatchingObjective,
                                     build_model_from_config, _subbands, _velocity,
                                     load_experiment_config)
    cfg = load_experiment_config(str(ROOT / "configs/hf_oriented_internal_early_stop.json"))
    model = build_model_from_config(cfg.model, bridge_cfg=cfg.bridge).to("cuda")
    # NOTE: this config builds no HH subband heads (enable_hh_head is not a
    # ModelConfig field).  The checkpoint carries 19 unused hh tensors; load
    # them as unexpected and assert nothing expected is missing.
    sd = torch.load(str(ROOT / "archive/runs_submission/canonical_oriented_epoch4/epoch_0004.pt"),
                    map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    elif isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    res = model.load_state_dict(sd, strict=False)
    if res.missing_keys:
        raise RuntimeError(f"missing keys: {res.missing_keys[:5]} ({len(res.missing_keys)})")
    print(f"[ckpt] loaded, unexpected={len(res.unexpected_keys)} (hh heads unused)")
    model.eval()
    objective = FlowMatchingObjective(cfg)
    return model, objective, cfg, _subbands, _velocity


def _pairwise_cos(vs: list[torch.Tensor]) -> float:
    total, n = 0.0, 0
    for i in range(len(vs)):
        for j in range(i + 1, len(vs)):
            total += float((F.normalize(vs[i].flatten(), dim=0) * F.normalize(vs[j].flatten(), dim=0)).sum())
            n += 1
    return total / max(n, 1)


def run_s15() -> dict:
    model, objective, cfg, _subbands, _velocity = load_model()
    ds = C.load_dataset()
    split = C.make_split(1000, n_ref=800, n_query=200, seed=0)
    content = torch.stack([ds.latents[0][split.query[0]][:16]]).to("cuda")[0]
    sub = ["ll", "lh", "hl", "hh"]
    out: dict = {"content_style": C.STYLES[0], "n_content": 16, "per_target_style": {}}
    style_ref = ds.latents[1][split.ref[1]][:16].to("cuda")      # FIXED condition
    style_id_fixed = 1
    labels_by_style = {}
    with torch.no_grad():
        v = _velocity(model, content, 0.0, style_id_fixed, style_ref)
        for t in range(5):
            cands = ds.latents[t][split.ref[t]][:8].to("cuda")
            lab = objective._build_training_target(content[:8], cands, style_ref[:8])
            ls_, cs_ = _subbands(lab), _subbands(content[:8])
            u = [{k: ls_[k][i] - cs_[k][i] for k in sub} for i in range(lab.shape[0])]
            labels_by_style[t] = u
            cos_self = _pairwise_cos([x["ll"] for x in u])
            cos_v = [float((F.normalize(v["ll"].mean(0).flatten(), dim=0) *
                            F.normalize(x["ll"].flatten(), dim=0)).sum()) for x in u]
            out["per_target_style"][C.SHORT[t]] = {
                "label_pairwise_cos_within_ll": cos_self,
                "label_pairwise_cos_within_lh": _pairwise_cos([x["lh"] for x in u]),
                "cos_model_v_vs_label_ll_mean": float(sum(cos_v) / len(cos_v)),
                "cos_model_v_vs_label_ll_std": float(torch.tensor(cos_v).std()),
                "label_ll_norm": float(u[0]["ll"].pow(2).flatten(1).sum(1).sqrt().mean()),
            }

    # between-style vs within-style decomposition on the label LL code
    means = torch.stack([torch.stack([x["ll"].flatten() for x in labels_by_style[t]]).mean(0) for t in range(5)])
    within = sum(float(torch.stack([x["ll"].flatten() for x in labels_by_style[t]]).var(0).mean())
                 for t in range(5)) / 5
    between = float(means.var(0).mean())
    out["label_decomposition_ll"] = {
        "between_style_var": between,
        "within_style_var": within,
        "ratio_between_over_within": between / max(within, 1e-12),
    }
    # cross-style label cosine
    cross = 0.0
    nc = 0
    for a in range(5):
        for b in range(a + 1, 5):
            cross += float((F.normalize(means[a], dim=0) * F.normalize(means[b], dim=0)).sum())
            nc += 1
    out["cross_style_label_cos_mean"] = cross / max(nc, 1)
    print(f"[S15] between/within var={out['label_decomposition_ll']['ratio_between_over_within']:.3f} "
          f"cross-style label cos={out['cross_style_label_cos_mean']:.3f} "
          f"within-ll cos="
          f"{ {k: round(v['label_pairwise_cos_within_ll'], 3) for k, v in out['per_target_style'].items()} }")
    C.save_json("s15_legacy_fm.json", out)
    return out


def run_s16() -> dict:
    from b2_destyle import destyle_latent
    ds = C.load_dataset()
    split = C.make_split(1000, n_ref=800, n_query=200, seed=0)
    z_ref = torch.stack([ds.latents[s][split.ref[s]] for s in range(5)])
    x = torch.stack([ds.latents[s][split.query[s]][:16] for s in range(5)])   # [5,16,4,64,64]
    y = torch.arange(5).repeat_interleave(16)
    gen = torch.Generator().manual_seed(0)
    xd = torch.stack([destyle_latent(x[s], generator=gen) for s in range(5)])

    # residual style probe: train on original ref latents, test on de-styled
    f_ref = torch.cat([C.feat_subband(z_ref[s]) for s in range(5)]).float()
    y_ref = torch.arange(5).repeat_interleave(z_ref.shape[1])
    f_out = C.feat_subband(xd.reshape(-1, 4, 64, 64)).float()
    pred, _ = C.linear_probe(f_ref, y_ref, f_out, k=5, iters=300)
    acc = float((pred == y).float().mean())
    per_style = {C.SHORT[s]: float((pred[y == s] == s).float().mean()) for s in range(5)}

    # offline DINO-S after de-styling
    pools = torch.load(C.AUDIT_DIR / "cache" / "ref_pools.pt", map_location="cpu", weights_only=False)
    img_d = C.decode(xd.reshape(-1, 4, 64, 64))
    gd = C.dino_cls(C.quantize01(img_d))
    res = {}
    for s in range(5):
        m = y == s
        own = float((gd[m] @ pools["dino_cls"][s].T).max(1).values.mean())
        others = [float((gd[m] @ pools["dino_cls"][t].T).max(1).values.mean()) for t in range(5) if t != s]
        res[C.SHORT[s]] = {"dino_s_own_style": own, "dino_s_other_mean": sum(others) / 4,
                           "margin": own - sum(others) / 4}
    out = {
        "residual_style_probe_acc": acc,
        "residual_style_probe_per_style": per_style,
        "residual_dino_s": res,
        "mean_residual_margin": sum(v["margin"] for v in res.values()) / 5,
    }
    print(f"[S16] residual style probe acc={acc:.3f} (chance .2); "
          f"dino margin={out['mean_residual_margin']:.4f}")
    C.save_json("s16_destyle_residual.json", out)
    return out


def main() -> None:
    steps = [s.lower() for s in sys.argv[1:]] or ["s15", "s16"]
    if "s15" in steps:
        run_s15()
    if "s16" in steps:
        run_s16()


if __name__ == "__main__":
    main()
