"""V8 raw-vs-EMA comparison on the frozen 750 tasks (P0 review item)."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.v8.common import (  # noqa: E402
    STYLES, V8_ROOT, aggregate_rows, batch_tensors, build_source_latents, compute_batch_rows,
    ensure_dirs, load_eval_bundle, load_frozen_tasks, save_json, source_cache_for_frozen,
)
from tools.v8.models import VelocityUNet  # noqa: E402
from tools.v8.solvers import solve_euler  # noqa: E402


def main() -> None:
    ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tasks = load_frozen_tasks()
    models, ref_feats, vae, trt = load_eval_bundle(device)
    src_latents = build_source_latents(tasks, vae, device)
    source_cache = source_cache_for_frozen()

    out = {"protocol": {"tasks": "frozen 750", "nfes": [4], "weight_variants": ["raw", "ema"]}, "methods": {}}
    for tag in ["B0", "B2"]:
        ck_path = V8_ROOT / tag / "model.pt"
        if not ck_path.exists():
            print(f"skip missing {ck_path}")
            continue
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        for variant in ["raw", "ema"]:
            state = ck.get("model") if variant == "raw" else ck.get("ema")
            if not state:
                continue
            model = VelocityUNet(base_dim=48, num_styles=len(STYLES)).to(device).eval()
            model.load_state_dict(state)
            rows, l2 = [], []
            t0 = time.time()
            with torch.inference_mode():
                for i in range(0, len(tasks), 8):
                    chunk = tasks[i:i + 8]
                    z0, s, t = batch_tensors(chunk, src_latents, device)
                    z_gen = solve_euler(model, z0, s, t, 4)
                    l2b = (z_gen.float() - z0.float()).flatten(1).norm(p=2, dim=1).cpu().tolist()
                    l2.extend(l2b)
                    rows.extend(compute_batch_rows(z_gen, chunk, models, ref_feats, vae, trt, source_cache, latent_l2=l2b, nfe=4))
            cross = aggregate_rows(rows, only_cross=True)
            same = aggregate_rows([r for r in rows if not r["is_cross"]])
            cross["gen_wall_sec"] = round(time.time() - t0, 1)
            key = f"{tag}_{variant}"
            out["methods"][key] = {"cross": cross, "same": same, "rows": rows}
            print(f"  [{key}] dT={cross['mean_dT']:+.4f} Top-1={cross['Target_Top1']}% "
                  f"Margin={cross['Target_Margin']:+.4f} DINO-C={cross['DINO_C']:.4f} LPIPS={cross['LPIPS']:.4f} "
                  f"ukiyo_dT={cross['ukiyo_dT']:+.4f} styles+={cross['styles_improved']}/5", flush=True)

    save_json(out, V8_ROOT / "p0_raw_vs_ema_750.json")
    print(f"saved {V8_ROOT / 'p0_raw_vs_ema_750.json'}")


if __name__ == "__main__":
    main()
