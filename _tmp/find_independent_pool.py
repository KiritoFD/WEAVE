import sys
from pathlib import Path
repo = Path('/mnt/i/Latent_Style/WEAVE')
sys.path.insert(0, str(repo))
from tools.v8.common import STYLES, train_latent_stems

t = train_latent_stems()
roots = {
    'distinct5_512/train': repo / 'datasets' / 'distinct5_512' / 'train',
    'wikiart_new_512': repo / 'datasets' / 'wikiart_new_512',
    'wikiart512_5style/train': repo / 'datasets' / 'wikiart512_5style' / 'train',
    'wikiart512_5style/test': repo / 'datasets' / 'wikiart512_5style' / 'test',
    'wikiart_random20_512/train': repo / 'datasets' / 'wikiart_random20_512' / 'train',
    'wikiart_stress_splits_512': repo / 'datasets' / 'wikiart_stress_splits_512',
    'r5': repo / 'datasets' / 'r5',
    'other5_512': repo / 'datasets' / 'other5_512',
}
for name, root in roots.items():
    if not root.exists():
        print(f'{name}: MISSING'); continue
    print(f'\n=== {name} ===')
    for s in STYLES:
        d = root / s
        if not d.exists():
            print(f'  {s}: no dir'); continue
        files = sorted(list(d.glob('*.jpg')) + list(d.glob('*.png')))
        pool = t[s]
        free = [p for p in files if p.stem.split('__',1)[-1] not in pool and p.stem not in pool]
        print(f'  {s}: total={len(files)} independent={len(free)}')
