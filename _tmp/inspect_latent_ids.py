import sys
from pathlib import Path
import torch
repo = Path('/mnt/i/Latent_Style/WEAVE')
sys.path.insert(0, str(repo))
from tools.v8.common import D5_LATENT_ROOT, STYLES

for sid, s in enumerate(STYLES):
    p = D5_LATENT_ROOT / f"{sid:02d}_{s}.pt"
    d = torch.load(p, map_location='cpu', weights_only=False)
    ids = d['ids']
    print(f'\n=== {s} : n={len(ids)} ===')
    for x in ids[:3]:
        print('  id:', repr(x))
    print('  files[:3]:', d.get('files', [])[:3])
    print('  subdir:', d.get('subdir'))
    print('  schema:', d.get('schema'))
