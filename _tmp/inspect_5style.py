import sys, csv
from pathlib import Path
repo = Path('/mnt/i/Latent_Style/WEAVE')
sys.path.insert(0, str(repo))
from tools.v8.common import STYLES, train_latent_stems

pools = train_latent_stems()
for s in STYLES:
    print(s, 'train pool', len(pools[s]))

# wikiart512_5style/train: figure out its own style map
r = repo / 'datasets' / 'wikiart512_5style' / 'train'
print('\nwikiart512_5style/train subdirs:', [p.name for p in r.iterdir() if p.is_dir()])
imp = r / 'Impressionism'
files = sorted(imp.glob('*.jpg'))[:3]
print('sample files:', [f.name for f in files])
print('pool hits:', [f.stem.split('__',1)[-1] in pools['Impressionism'] for f in files])

# check 5style train more broadly
rng = repo / 'datasets' / 'wikiart_random20_512'
if rng.exists():
    print('\nwikiart_random20_512 subdirs:', [p.name for p in rng.iterdir()][:10])

# Data/all_data_info.csv based independent selection feasibility
data_csv = Path('/mnt/i/Latent_Style/Data/all_data_info.csv')
print('\nall_data_info exists:', data_csv.exists())
