import sys, json
from pathlib import Path
import torch
repo = Path('/mnt/i/Latent_Style/WEAVE')
sys.path.insert(0, str(repo))
import paths
root = paths.asset_root() / 'pseudo_gt_24h' / 'orig_mean_fp16'
out = {"asset_root": str(paths.asset_root()), "root": str(root), "root_exists": root.exists(), "files": {}}
for sid, s in enumerate(['Early_Renaissance','Impressionism','Minimalism','Rococo','Ukiyo_e']):
    p = root / f'{sid:02d}_{s}.pt'
    item = {"exists": p.exists(), "bytes": p.stat().st_size if p.exists() else 0}
    if p.exists():
        d = torch.load(p, map_location='cpu', weights_only=False)
        item["keys"] = list(d.keys()) if isinstance(d, dict) else str(type(d))
        if isinstance(d, dict) and 'latents' in d:
            x = d['latents']
            item.update({"latents_shape": list(x.shape), "latents_dtype": str(x.dtype),
                         "min": float(x.min()), "max": float(x.max()),
                         "mean": float(x.mean()), "std": float(x.std())})
        if isinstance(d, dict):
            item["scaling_factor"] = d.get('scaling_factor')
            item["vae"] = d.get('vae')
            for k in ['paths','names','ids','stems']:
                if k in d:
                    v = d[k]
                    item[k+"_type"] = str(type(v))
                    try: item[k+"_len"] = len(v)
                    except Exception: pass
    out['files'][s] = item
print(json.dumps(out, indent=2))
