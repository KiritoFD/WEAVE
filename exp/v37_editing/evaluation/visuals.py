"""Fixed final examples with grayscale crops; never used for checkpoint selection."""
import gc
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
from exp.v36_art_foundation.diag_v36 import _load_diag_assets
from exp.v36_art_foundation.eval_ecore import load_reference_ecore
from exp.v36_art_foundation.eval_v36 import load_v36_models, reconstruct_reference_a3, sample_v36_ode
from exp.v37_editing import OUT, SEED, atomic_json
from exp.v37_editing.evaluation.common import P_CKPTS
from src.utils import load_s10_decoder, decode_s10
from src.utils.protocol import EVAL_SEED


@torch.no_grad()
def fixed_examples():
    selection = json.loads((OUT / "matched_selection.json").read_text())["checkpoints"]
    selection.pop("initial")
    tasks, z, q, y, refs, ri, _ = _load_diag_assets()
    chosen = [i for i, t in enumerate(tasks) if t["source_style_id"] != t["target_style_id"]]
    chosen = np.random.default_rng(SEED + 404).choice(chosen, 8, replace=False).tolist()
    # Match the official test750 noises exactly before slicing.
    eps = torch.randn(z.shape, generator=torch.Generator().manual_seed(EVAL_SEED))[chosen].cuda()
    eps_e = torch.randn(z.shape, generator=torch.Generator().manual_seed(EVAL_SEED + 17))[chosen].cuda()
    zc, qc, yc = z[chosen].cuda().float(), q[chosen].cuda().float(), y[chosen].cuda().float()
    zr = refs["z_ref"][ri[chosen]].cuda().float()
    sr = refs["stat_1561"][ri[chosen]].cuda().float()
    p, se, priors = load_v36_models(P_CKPTS["p2_50k"])
    dec = load_s10_decoder("cuda")
    es = se(sr)
    a3 = reconstruct_reference_a3(zc, zr, priors)
    pb = sample_v36_ode(p, a3, yc, qc, es, eps, tau0=.65, wj=3.8, wc=1.6)
    latents = {"source": zc, "reference": zr, "P+A3": pb}
    for route, ck in selection.items():
        ec, pg = load_reference_ecore(Path(ck))
        draft = ec(zc, zr, qc, sr, torch.ones(8, device="cuda"), pg, noise=eps_e)
        latents[route] = draft
        latents[route + "+P"] = sample_v36_ode(p, draft, yc, qc, es, eps, tau0=.65, wj=3.8, wc=1.6)
        del ec
    rgb = {k: decode_s10(dec, v, batch=8, output_cpu=True) for k, v in latents.items()}
    folder = OUT / "figures"; folder.mkdir(exist_ok=True)
    names = list(rgb)
    for page in range(4):
        for grayscale in (False, True):
            canvas = Image.new("RGB", (len(names) * 192, 2 * 216), "white")
            draw = ImageDraw.Draw(canvas)
            for row in range(2):
                i = page * 2 + row
                for col, name in enumerate(names):
                    im = Image.fromarray((rgb[name][i].permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8))
                    if grayscale:
                        im = ImageOps.autocontrast(ImageOps.grayscale(im.crop((192,192,320,320)))).convert("RGB")
                    canvas.paste(im.resize((192,192)), (col*192,row*216+24))
                    draw.text((col*192+3,row*216+4),name,fill="black")
            canvas.save(folder / f"fixed_{page}_{'gray_crop' if grayscale else 'full'}.jpg", quality=93)
    atomic_json(folder / "provenance.json", {"task_ids": [tasks[i]["task_id"] for i in chosen], "indices": chosen,
                "seed": SEED + 404, "lambda": 1., "tau": .65, "checkpoints": selection,
                "grayscale_note": "Per-image autocontrast, fixed center crop; diagnostic only, not a quantitative metric"})
