import sys
from pathlib import Path
repo = Path('/mnt/i/Latent_Style/WEAVE')
sys.path.insert(0, str(repo))
from tools.v8.common import STYLES, train_latent_stems, test_source_stems

pools = train_latent_stems()
banned = test_source_stems()
banned_short = {b.split('__', 1)[-1] for b in banned}
root = repo / 'datasets' / 'wikiart_random20_512' / 'images' / 'test'

print('exists:', root.exists())
print('subdirs:', [p.name for p in root.iterdir()][:25] if root.exists() else None)

for s in STYLES:
    d = root / s
    if not d.exists():
        # try case-insensitive
        cands = [p for p in root.iterdir() if p.name.lower() == s.lower()]
        print(f'{s}: dir missing, candidates={[c.name for c in cands]}')
        if not cands:
            continue
        d = cands[0]
    files = sorted(list(d.glob('*.jpg')) + list(d.glob('*.png')))
    free = [p for p in files if p.stem.split('__', 1)[-1] not in pools[s] and p.stem.split('__', 1)[-1] not in banned_short]
    print(f'{s} [{d.name}]: total={len(files)} independent={len(free)}')
