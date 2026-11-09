import torch
from pathlib import Path
import sys
from pathlib import Path as P
repo = P('/mnt/i/Latent_Style/WEAVE')
sys.path.insert(0, str(repo))
import paths
asset = paths.asset_root()
root = asset / 'pseudo_gt_24h' / 'orig_mean_fp16'
print('asset_root', asset)
print('root', root, root.exists())
for sid, s in enumerate(['Early_Renaissance','Impressionism','Minimalism','Rococo','Ukiyo_e']):
    p = root / f'{sid:02d}_{s}.pt'
    print('\n', p.name, p.exists(), p.stat().st_size if p.exists() else None)
    if p.exists():
        d = torch.load(p, map_location='cpu', weights_only=False)
        print('keys', list(d.keys())[:10] if isinstance(d, dict) else type(d))
        if isinstance(d, dict) and 'latents' in d:
            x = d['latents']
            print('latents', tuple(x.shape), x.dtype, 'min/max/mean/std', float(x.min()), float(x.max()), float(x.mean()), float(x.std()))
        for k, v in (d.items() if isinstance(d, dict) else []):
            if k != 'latents':
                print('meta', k, type(v), v if not torch.is_tensor(v) else tuple(v.shape))
