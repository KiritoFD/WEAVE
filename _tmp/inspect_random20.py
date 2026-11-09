import sys
from pathlib import Path
repo = Path('/mnt/i/Latent_Style/WEAVE')
sys.path.insert(0, str(repo))
from tools.v8.common import STYLES, train_latent_stems

pools = train_latent_stems()
roots = [
    repo / 'datasets' / 'wikiart_random20_512',
    repo / 'datasets' / 'wikiart_new_512',
    repo / 'datasets' / 'style_data',
]
for root in roots:
    if not root.exists():
        print(root, 'MISSING'); continue
    print(f'\n=== {root.relative_to(repo)} ===')
    subdirs = [p for p in root.rglob('*') if p.is_dir()]
    shown = 0
    for d in subdirs:
        files = list(d.glob('*.jpg'))[:1]
        if not files:
            continue
        name = d.relative_to(root)
        print(f'  {str(name)[:60]:60s} n={len(list(d.glob("*.jpg")))} sample={files[0].name}')
        shown += 1
        if shown > 30:
            break
