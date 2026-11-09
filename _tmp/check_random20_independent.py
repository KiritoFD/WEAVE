import sys
from pathlib import Path
repo = Path('/mnt/i/Latent_Style/WEAVE')
sys.path.insert(0, str(repo))
from tools.v8.common import STYLES, train_latent_stems, test_source_stems

pools = train_latent_stems()
banned = test_source_stems()
banned_short = {b.split('__', 1)[-1] for b in banned}
root = repo / 'datasets' / 'wikiart_random20_512' / 'images' / 'test'

for s in STYLES:
    d = root / s
    files = sorted(list(d.glob('*.jpg')) + list(d.glob('*.png')))
    free = []
    for p in files:
        short = p.stem.split('__', 1)[-1]
        if short in pools[s] or p.stem in pools[s]:
            continue
        if short in banned_short:
            continue
        free.append(p)
    print(f'{s}: total={len(files)} fully_independent={len(free)}')
    for p in free[:2]:
        print('   ', p.name)
