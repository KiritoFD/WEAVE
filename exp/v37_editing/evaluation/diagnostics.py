"""Missing V37 diagnostics; reuse complete V36 Oracle and baseline measurements."""
import argparse
import gc
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from exp.v36_art_foundation import CACHE_DIR
from exp.v36_art_foundation.diag_v36 import _load_diag_assets, _generate_pending_arm, _flush_measure
from exp.v36_art_foundation.eval_v36 import load_v36_models, reconstruct_reference_a3, sample_v36_ode
from exp.v37_editing import OUT, SEED, atomic_json, read_manifest
from exp.v37_editing.evaluation.common import P_CKPTS, split_pack, save_latents, measure_embeddings
from src.utils import load_s10_decoder, decode_s10
from src.utils.runtime import bf16_autocast


@torch.no_grad()
def generate_seen():
    pack = split_pack("seen_dev", 300)
    folder = OUT / "diagnostics/seen"
    for ckname, ckpath in P_CKPTS.items():
        p, se, priors = load_v36_models(ckpath)
        zs = reconstruct_reference_a3(pack["zc"], pack["zr"], priors)
        es = se(pack["sr"].cuda()).cpu()
        for tau, qm in [(0., "correct"), (.35, "correct"), (.65, "correct"), (.35, "zero"), (.35, "shuffle")]:
            name = f"{ckname}_t{int(tau * 100):02d}_q{qm}"
            path = folder / f"{name}.pt"
            if path.exists():
                continue
            q = pack["q"] if qm == "correct" else torch.zeros_like(pack["q"]) if qm == "zero" else pack["q"].roll(1, 0)
            zz = []
            calls = [0]
            hook = p.register_forward_hook(lambda *_: calls.__setitem__(0, calls[0] + 1))
            for s in range(0, 300, 24):
                sl = slice(s, s + 24)
                zz.append(sample_v36_ode(p, zs[sl].cuda(), pack["y"][sl].cuda(), q[sl].cuda(), es[sl].cuda(),
                                        pack["eps"][sl].cuda(), tau0=tau, wj=3.8, wc=1.6).cpu())
            hook.remove()
            save_latents(path, torch.cat(zz), {"arm": name, "ckpt": ckname, "tau": tau, "q": qm,
                         "split": "seen_dev", "n": 300, "forward_calls": calls[0], "batches": 13,
                         "nfe_per_sample": calls[0] / 13, "cfg_branch_equivalents": 18})
            print("GENERATED " + name, flush=True)
        palette_response(p, se, priors, ckname)
        del p, se
        gc.collect(); torch.cuda.empty_cache()


@torch.no_grad()
def palette_response(p, se, priors, ckname):
    """Same-palette/different-texture controls chosen from input descriptors, not output judges."""
    ready = json.loads((CACHE_DIR / "v36_caches_ready.json").read_text())["dev_info"]
    from exp.v36_art_foundation import ROOT
    st = torch.from_numpy(np.array(np.load(ROOT / ready["paths"]["stat"], mmap_mode="r"))).float()
    norm = se.normalize_stats(st.cuda()).cpu()
    pal, tex = norm[:, :27], norm[:, 1497:]
    d = torch.cdist(pal, pal)
    d.fill_diagonal_(float("inf"))
    near = d.topk(20, largest=False).indices[:64]
    ri = torch.arange(64)
    ti = torch.stack([(tex[cand] - tex[i]).square().mean(1).argmax() for i, cand in enumerate(near)])
    rj = near[ri, ti]
    pack = split_pack("seen_dev", 64)
    ea, eb = se(st[ri].cuda()), se(st[rj].cuda())
    # Fixed identical A3 start for both conditions; excludes analytic palette transport.
    zs = reconstruct_reference_a3(pack["zc"], pack["zr"], priors)
    outputs = []
    for es in (ea, eb):
        out = []
        for s in range(0, 64, 16):
            sl = slice(s, s + 16)
            out.append(sample_v36_ode(p, zs[sl].cuda(), pack["y"][sl].cuda(), pack["q"][sl].cuda(),
                                     es[sl], pack["eps"][sl].cuda(), tau0=.35, wj=3.8, wc=1.6).cpu())
        outputs.append(torch.cat(out))
    # These are sensitivity diagnostics, not claims of brushwork quality.
    row = {"ckpt": ckname, "n": 64, "ref_a": ri.tolist(), "ref_b": rj.tolist(),
           "normalized_palette_rms": float((pal[ri] - pal[rj]).square().mean().sqrt()),
           "normalized_texture_rms": float((tex[ri] - tex[rj]).square().mean().sqrt()),
           "encoder_cosine_distance": float((1 - F.cosine_similarity(ea, eb)).mean()),
           "output_latent_rms": float((outputs[0] - outputs[1]).square().mean().sqrt()),
           "limitation": "Descriptor-selected references need visual brushwork review; response alone is not correctness"}
    path = OUT / "diagnostics" / f"palette_{ckname}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"outputs": outputs, "row": row}, path)
    atomic_json(path.with_suffix(".json"), row)


@torch.no_grad()
def core_quadrants():
    tasks, z, q, y, refs, ok, wrong = _load_diag_assets()
    p, se, priors = load_v36_models(P_CKPTS["p2_50k"])
    dec = load_s10_decoder("cuda")
    starts = {"correct": reconstruct_reference_a3(z.float(), refs["z_ref"][ok].float(), priors),
              "wrong": reconstruct_reference_a3(z.float(), refs["z_ref"][wrong].float(), priors)}
    es = se(refs["stat_1561"].cuda().float()).cpu()
    pending, metas = [], {}
    # Rerun full 2x2 with identical implementation/seeds; no checkpoint selection on Core5.
    for a3 in ("correct", "wrong"):
        for cond in ("correct", "wrong"):
            name = f"V37_D2_A3{a3}_P{cond}_t50"
            raw, meas = _generate_pending_arm(name, z_start_750=starts[a3], e_style_750=es[ok if cond == "correct" else wrong],
                q32_750=q, y16_750=y, tasks=tasks, dec=dec, pcore=p, tau0=.50)
            pending.append((raw, meas, name))
            metas[name] = {"diag": "V37_D2", "a3_ref": a3, "p_ref": cond, "tau0": .5, "nfe": 6}
    # Structural invariance test at tau=0, with actual network evaluation on 24 cases.
    vals = []
    for a3 in ("correct", "wrong"):
        vals.append(sample_v36_ode(p, starts[a3][:24].cuda(), y[:24].cuda().float(), q[:24].cuda().float(),
                    es[ok[:24]].cuda(), torch.randn((24, 4, 64, 64), generator=torch.Generator().manual_seed(SEED)).cuda(),
                    tau0=0., wj=3.8, wc=1.6).cpu())
    atomic_json(OUT / "diagnostics/tau0_invariance.json", {"n": 24, "max_abs": float((vals[0] - vals[1]).abs().max())})
    del p, se, dec
    gc.collect(); torch.cuda.empty_cache()
    rows = _flush_measure(pending, metas, "cuda")
    atomic_json(OUT / "diagnostics/core_quadrants.json", rows)


def select_teacher():
    rows = json.loads((OUT / "diagnostics/seen/summary.json").read_text())
    scores = {}
    for name in P_CKPTS:
        rr = sorted([r for r in rows if r["ckpt"] == name and r["q"] == "correct"], key=lambda r: r["dino_c"])
        cc, ss = [r["dino_c"] for r in rr], [r["dino_s"] for r in rr]
        if cc[0] <= .70 <= cc[-1]:
            scores[name] = float(np.interp(.70, cc, ss))
    assert scores, "No Seen-Dev bracket around C=.70"
    name = max(scores, key=scores.get)
    atomic_json(OUT / "teacher_selection.json", {"selected": name, "path": str(P_CKPTS[name]),
                "criterion": "Seen-Dev300 interpolated instance DINO-S at DINO-C=.70; predeclared",
                "scores": scores, "official_test_used": False})


@torch.no_grad()
def core_q_controls():
    tasks, z, q, y, refs, ok, wrong = _load_diag_assets()
    stems = [t["source_stem"] for t in tasks]
    # Shuffle content identities, not neighboring task rows (five targets share a source).
    unique = sorted(set(stems))
    first = {s: stems.index(s) for s in unique}
    replacement = {s: unique[(i + 1) % len(unique)] for i, s in enumerate(unique)}
    wrong_q_idx = torch.tensor([first[replacement[s]] for s in stems])
    assert all(stems[int(j)] != s for s, j in zip(stems, wrong_q_idx))
    dec = load_s10_decoder("cuda")
    pending, metas = [], {}
    for ckname, ckpath in P_CKPTS.items():
        p, se, priors = load_v36_models(ckpath)
        start = reconstruct_reference_a3(z.float(), refs["z_ref"][ok].float(), priors)
        es = se(refs["stat_1561"].cuda().float()).cpu()[ok]
        for mode in ("zero", "shuffle"):
            qq = torch.zeros_like(q) if mode == "zero" else q[wrong_q_idx]
            name = f"V37_D3_core_{ckname}_q{mode}_t35"
            raw, meas = _generate_pending_arm(name, z_start_750=start, e_style_750=es, q32_750=qq, y16_750=y,
                                              tasks=tasks, dec=dec, pcore=p, tau0=.35)
            pending.append((raw, meas, name))
            metas[name] = {"diag": "V37_D3_coreQ", "ckpt": ckname, "q": mode, "tau": .35, "nfe": 6}
        del p, se
        gc.collect(); torch.cuda.empty_cache()
    del dec
    gc.collect(); torch.cuda.empty_cache()
    atomic_json(OUT / "diagnostics/core_q_controls.json", _flush_measure(pending, metas, "cuda"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("seen", "measure", "core", "select"))
    a = ap.parse_args()
    if a.stage == "seen": generate_seen()
    elif a.stage == "measure": measure_embeddings(OUT / "diagnostics/seen")
    elif a.stage == "core": core_quadrants()
    else: select_teacher()
