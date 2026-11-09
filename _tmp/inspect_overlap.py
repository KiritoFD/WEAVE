import sys
from pathlib import Path
repo = Path('/mnt/i/Latent_Style/WEAVE')
sys.path.insert(0, str(repo))
from tools.v8.common import STYLES, train_latent_stems

t = train_latent_stems()
root = repo / 'datasets' / 'distinct5_512' / 'train'
for s in STYLES:
    pool = t[s]
    sample_pool = sorted(pool)[:3]
    files = sorted((root / s).glob('*.jpg'))
    print(f'\n=== {s} ===')
    print('  pool size', len(pool), 'sample:', sample_pool)
    print('  file count', len(files))
    print('  file sample:', [p.stem for p in files[:3]])
    hits = sum(1 for p in files if p.stem.split('__', 1)[-1] in pool)
    print('  short-stem hits in pool:', hits)
