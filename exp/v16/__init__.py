"""V16: capacity diagnosis, bounded LD4 distillation, RGB-to-latent projection."""
from pathlib import Path
import contextlib
import hashlib
import json
import os
import unicodedata
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/v16'
STYLES = ['Early_Renaissance', 'Impressionism', 'Minimalism', 'Rococo', 'Ukiyo_e']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=1, default=str,
                               allow_nan=False), encoding='utf-8')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def autocast(device='cuda'):
    return torch.autocast('cuda', dtype=torch.bfloat16) if str(device).startswith('cuda') else contextlib.nullcontext()


def load_base(path=None, device='cuda'):
    from exp.v15.model import V15UNet
    path = path or ROOT / 'runs/v15/W_dim96_conv_base/final.pt'
    ck = torch.load(path, map_location='cpu', weights_only=False)
    c = ck['config']
    m = V15UNet(base_dim=int(c['base_dim']),
                use_attention=bool(c.get('use_attention', False)),
                style_mode=c.get('style_mode', 'adaln'),
                n_experts=int(c.get('n_experts', 4))).to(device).eval()
    m.load_state_dict(ck['model'], strict=True)
    m.requires_grad_(False)
    return m, ck


@torch.no_grad()
def generate(model, source, target, noise, tau0=.7, steps=20):
    z = tau0 * source + (1 - tau0) * noise
    dt = (1 - tau0) / steps
    for k in range(steps):
        tau = torch.full((len(z),), tau0 + k * dt, device=z.device)
        with autocast(z.device):
            v = model(z, target, tau).float()
        z = z + dt * v
    return z


def pool_samples(pool, train_each=100, hold_each=20, seed=20261116):
    """Use original FM train/holdout split; exclude test AND reference stems."""
    from exp.v15.train import _h10
    tasks = read(ROOT / 'eval_protocol/tasks_750.json')['tasks']
    refs = read(ROOT / 'eval_protocol/references_v1.json')['references']
    banned = {t['source_stem'] for t in tasks}
    banned |= {r['stem'] for rows in refs.values() for r in rows}
    def canonical(s):
        return unicodedata.normalize('NFC', Path(str(s).replace('\\', '/')).stem.split('__')[-1]).casefold()
    banned = {canonical(x) for x in banned}
    result = {'train': [], 'holdout': []}
    manifest = {'seed': seed, 'excluded_test_and_reference_stems': len(banned),
                'painting_id_protocol': 'NFC last double-underscore component; deduplicate all copies before FM split', 'styles': {}}
    g = torch.Generator().manual_seed(seed)
    for sid, style in enumerate(STYLES):
        p = Path(pool) / f'{sid:02d}_{style}.pt'
        d = torch.load(p, map_location='cpu', weights_only=False)
        ids = [str(x) for x in d['ids']]
        groups = {}
        for i, value in enumerate(ids):
            if canonical(value) not in banned:
                groups.setdefault(canonical(value), []).append(i)
        # A work with any original-training copy belongs to train; only all-held-out
        # groups qualify as FM holdout. Pick a copy with that original split.
        eligible = [next((i for i in rows if _h10(ids[i]) < 9), rows[0]) for rows in groups.values()]
        manifest['styles'][style] = {'file': str(p), 'n_pool': len(ids), 'eligible': len(eligible)}
        for split, n, train in [('train', train_each, True), ('holdout', hold_each, False)]:
            candidates = [i for i in eligible if (_h10(ids[i]) < 9) == train]
            if len(candidates) < n:
                raise ValueError(f'{style}/{split}: {len(candidates)} < {n}')
            chosen = [candidates[j] for j in torch.randperm(len(candidates), generator=g)[:n].tolist()]
            for i in chosen:
                result[split].append((d['latents'][i].float(), sid, ids[i]))
            manifest['styles'][style][split + '_ids'] = [ids[i] for i in chosen]
    for split, rows in result.items():
        result[split] = {'z': torch.stack([x[0] for x in rows]),
                         'sid': torch.tensor([x[1] for x in rows]),
                         'ids': [x[2] for x in rows]}
    assert not set(result['train']['ids']) & set(result['holdout']['ids'])
    return result, manifest


def tasks_and_sources():
    tasks = read(ROOT / 'eval_protocol/tasks_750.json')['tasks']
    cache = torch.load(ROOT / 'runs/v9_local_4070/cache/n2c_z.pt',
                       map_location='cpu', weights_only=False)
    assert len(tasks) == 750 == len(cache['z_src'])
    if 'task_ids' in cache:
        assert [t['task_id'] for t in tasks] == cache['task_ids']
    return tasks, cache['z_src'].float()


class RawWriter:
    """Atomic batches, no PNG. Paths are unique to this experiment."""
    def __init__(self, name):
        self.path = OUT / 'raw' / name
        self.path.mkdir(parents=True, exist_ok=True)
        self.batches, self.n = [], 0

    def append(self, ids, rgb, z=None):
        name = f'batch_{self.n:05d}.pt'
        payload = {'ids': ids, 'rgb_u8': (rgb.detach().cpu().float() * 255).round().clamp(0, 255).byte()}
        if z is not None:
            payload['z_final'] = z.detach().cpu().half()
        temp = self.path / (name + '.tmp')
        torch.save(payload, temp)
        temp.replace(self.path / name)
        self.batches.append({'offset': self.n, 'file': name, 'ids': ids})
        self.n += len(ids)

    def finish(self, **extra):
        assert self.n == 750, self.n
        ids = [t for b in self.batches for t in b['ids']]
        assert len(set(ids)) == 750
        save(self.path / 'meta.json', {'kind': 'v16', 'n_tasks': self.n,
             'decoder': 'S10', 'batches': self.batches, **extra})
