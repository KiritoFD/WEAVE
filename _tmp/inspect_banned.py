import sys
from pathlib import Path
repo = Path('/mnt/i/Latent_Style/WEAVE')
sys.path.insert(0, str(repo))
from tools.v8.common import STYLES, train_latent_stems, test_source_stems

t = train_latent_stems()
banned = test_source_stems()
root = repo / 'datasets' / 'distinct5_512' / 'train'
s = 'Early_Renaissance'
pool = t[s]
files = sorted((root / s).glob('*.jpg'))
print('banned sample:', list(banned)[:3])
banned_short = {b.split('__',1)[-1] for b in banned}
print('banned short sample:', sorted(banned_short)[:3])
short_hits = sum(1 for p in files if p.stem.split('__',1)[-1] in banned_short)
print('files matched by banned short stems:', short_hits, '/', len(files))
for p in files[:5]:
    short = p.stem.split('__',1)[-1]
    print(' ', p.stem, '| short:', short, '| in banned_short:', short in banned_short, '| in pool:', p.stem in pool)
