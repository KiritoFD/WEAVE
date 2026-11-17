"""B: a V15-anchored LD4 teacher and a frozen-base residual student.

Teacher acceptance never gates student training. Frozen referees are not losses.
"""
import argparse
import math
import time
import torch
import torch.nn.functional as F
from . import ROOT, OUT, save, read, sha, autocast, load_base, generate, pool_samples
from . import tasks_and_sources, RawWriter
from src.model.edit_adapter import EditResidualAdapter


class Teacher:
    def __init__(self):
        from src.infer import load_critic
        from exp.v8.guidance import SPDHead
        from exp.v9_local.local_style import LocalStyleEnergy, BETA, MU
        from exp.v9_local.direction import DirectionComputer
        class PerSampleEnergy(LocalStyleEnergy):
            def loss(self, z, tid):
                et, en = self.energies(z, tid)
                return (et + BETA * F.softplus(MU + et - en)).mean(1)
        self.critic = load_critic('cuda')
        ck = torch.load(ROOT/'runs/v8/train/N2C/step_04000.pt', map_location='cpu', weights_only=False)
        self.head = SPDHead(rank=int(ck.get('rank', 0))).cuda().eval()
        self.head.load_state_dict(ck['head'])
        self.head.requires_grad_(False)
        self.energy = PerSampleEnergy(str(ROOT/'runs/v9_local_4070/bank/style_bank.pt'), 'cuda')
        self.direction = DirectionComputer(self.critic, self.head, self.energy, 'cuda')

    def __call__(self, base, source, sid, tid):
        from exp.v9_local.solver import solve_batch
        # New operator: reset anchor and trust budget around this V15 output.
        with torch.enable_grad():
            dirs = self.direction.directions(base, base, tid, 'mix')
        with torch.no_grad():
            # The old .5/1 normalized steps typically project to rho=.15, which
            # can exceed sqrt(eps)*RMS(base-source) at the new V15 working point.
            # Scale the shortlist to the actual per-image trust radius first.
            radius=(.10*(base-source).flatten(1).square().mean(1)).sqrt().clamp_max(.15)
            dirs={k:v*radius[:,None,None,None] for k,v in dirs.items()}
            mask = self.direction.region_mask(base, base, tid, frac=.25)
            out, info = solve_batch(base, base, source, sid, tid, [dirs], mask,
                                   self.critic, self.energy, eps=.10, k_steps=2, rho=.15)
            same = sid == tid
            out[same] = base[same]
            for i in same.nonzero(as_tuple=True)[0].tolist():
                info[i]['short'] = True
            for i in range(len(info)):
                info[i]['trust_radius']=float(radius[i])
            cost=(out-base).flatten(1).square().mean(1)/(base-source).flatten(1).square().mean(1).clamp_min(1e-8)
            if (cost>.100001).any():raise AssertionError('Teacher exceeded V15 content budget')
        return out.detach(), info


def prepare(args):
    path = OUT/'B/teacher_pairs.pt'
    if path.exists():
        return
    source, manifest = pool_samples(args.pool, args.train_each, args.hold_each)
    save(OUT/'B/data_manifest.json', manifest)
    model, _ = load_base()
    teacher = Teacher()
    from src.model import canonicalizer
    canon = canonicalizer('cuda')
    g = torch.Generator().manual_seed(args.seed)
    result = {}
    start = time.time()
    for split, d in source.items():
        # Every source visits all five target labels; same labels get zero teacher edit.
        src = d['z'].repeat_interleave(5, 0)
        sid = d['sid'].repeat_interleave(5)
        tid = torch.arange(5).repeat(len(d['z']))
        eps = torch.randn(src.shape, generator=g)
        arrays = {k: [] for k in ['base', 'content', 'delta']}
        infos = []
        for st in range(0, len(src), args.batch):
            sl = slice(st, st + args.batch)
            z, si, ti = src[sl].cuda(), sid[sl].cuda(), tid[sl].cuda()
            with torch.no_grad():
                base = generate(model, z, ti, eps[sl].cuda())
                content = canon(z, mode='c1_gentle', augment=False)
            corrected, report = teacher(base, z, si, ti)
            for key, value in [('base', base), ('content', content), ('delta', corrected-base)]:
                arrays[key].append(value.cpu().half())
            infos.extend(report)
            if st % (args.batch * 10) == 0:
                print(f'B pairs {split} {st+len(z)}/{len(src)} {time.time()-start:.0f}s', flush=True)
        result[split] = {k: torch.cat(v) for k, v in arrays.items()}
        result[split].update({'sid': sid, 'tid': tid, 'source_ids': d['ids'], 'infos': infos})
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, path)
    stats = {}
    for split, d in result.items():
        cross = d['sid'] != d['tid']
        rms = d['delta'].float().flatten(1).square().mean(1).sqrt()
        stats[split] = {'n': len(rms), 'cross_n': int(cross.sum()),
                        'delta_rms_cross': float(rms[cross].mean()),
                        'nonzero_cross_fraction': float((rms[cross] > 1e-7).float().mean()),
                        'accepted_cross_fraction': sum(i['accepted'] > 0 for i, c in zip(d['infos'], cross) if c) / int(cross.sum())}
    save(OUT/'B/teacher_pair_summary.json', {'splits': stats, 'wall_sec': time.time()-start,
         'anchor': 'V15 output, not the historical B4 anchor',
         'step_radius': 'min(.15,sqrt(.10)*RMS(V15-source)) before mask and eta shortlist',
         'local_energy': 'per-sample E; fixes historical batch-mean proxy broadcast',
         'source_manifest': str(OUT/'B/data_manifest.json')})


@torch.no_grad()
def fit_metrics(net, d, batch=16):
    sqerr = energy = dot = prednorm = targetnorm = 0.0
    n = 0
    for st in range(0, len(d['tid']), batch):
        sl = slice(st, st + batch)
        x, c, tid, target = [d[k][sl].cuda().float() if k != 'tid' else d[k][sl].cuda()
                              for k in ['base','content','tid','delta']]
        with autocast(): prediction = net(x, c, tid).float()
        cross = d['sid'][sl].cuda() != tid
        p, t = prediction[cross].flatten(1), target[cross].flatten(1)
        sqerr += float((p-t).square().sum()); energy += float(t.square().sum())
        dot += float((p*t).sum()); prednorm += float(p.square().sum()); targetnorm += float(t.square().sum())
        n += p.numel()
    return {'mse': sqerr/max(n, 1), 'teacher_energy': energy/max(n, 1),
            'relative_mse': sqerr/max(energy, 1e-20),
            'global_cosine': dot/max((prednorm*targetnorm)**.5, 1e-20),
            'prediction_rms': (prednorm/max(n, 1))**.5,
            'teacher_rms': (targetnorm/max(n, 1))**.5}


def train(args):
    path = OUT/'B/adapter.pt'
    if path.exists(): return
    d = torch.load(OUT/'B/teacher_pairs.pt', map_location='cpu', weights_only=False)
    torch.manual_seed(args.seed)
    net = EditResidualAdapter().cuda()
    n_params = sum(p.numel() for p in net.parameters())
    assert 200_000 <= n_params <= 500_000, n_params
    baseline = {s: fit_metrics(net, v) for s, v in d.items()}
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=.01)
    gen = torch.Generator().manual_seed(args.seed)
    curves, start = [], time.time()
    torch.cuda.reset_peak_memory_stats()
    net.train()
    for step in range(1, args.updates+1):
        opt.zero_grad(set_to_none=True)
        rate = 3e-4 * (.05 + .95 * .5 * (1 + math.cos(math.pi * (step-1)/args.updates)))
        for pg in opt.param_groups: pg['lr'] = rate
        for _ in range(4):
            idx = torch.randint(len(d['train']['tid']), (8,), generator=gen)
            base = d['train']['base'][idx].cuda().float()
            content = d['train']['content'][idx].cuda().float()
            target = d['train']['delta'][idx].cuda().float()
            tid = d['train']['tid'][idx].cuda()
            with autocast(): delta = net(base, content, tid).float()
            kd = (delta-target).square().mean()
            # Preserve the base's low-frequency structure; never regress to z_src.
            low = F.avg_pool2d(delta, 4).square().mean()
            reg = delta.square().mean()
            loss = kd + .1 * low + .01 * reg
            (loss/4).backward()
        grad = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
        if not torch.isfinite(grad): raise FloatingPointError(f'nonfinite grad at {step}')
        opt.step()
        if step == 1 or step % 100 == 0:
            row={'step': step, 'loss': float(loss), 'kd': float(kd),
                 'low_frequency_penalty': float(low), 'residual_penalty': float(reg),
                 'grad_norm': float(grad), 'lr': rate, 'wall_sec': time.time()-start}
            curves.append(row)
            print(f'B train {step}/{args.updates} kd={row["kd"]:.7f} {row["wall_sec"]:.0f}s', flush=True)
    net.eval()
    final = {s: fit_metrics(net, v) for s, v in d.items()}
    torch.save({'model': net.cpu().state_dict(), 'width': 64, 'blocks': 3,
                'updates': args.updates, 'seed': args.seed, 'n_params': n_params,
                'teacher_pairs_sha256': sha(OUT/'B/teacher_pairs.pt')}, path)
    save(OUT/'B/fit.json', {'zero_adapter': baseline, 'trained_adapter': final,
         'curves': curves, 'wall_sec': time.time()-start, 'n_params': n_params,
         'effective_batch': 32, 'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
         'teacher_quality_gate': False, 'note': 'Student is trained regardless of teacher downstream quality.'})


def evaluate(args):
    names = ['B_base', 'B_teacher', 'B_adapter', 'B_adapter_wrong']
    if all((OUT/'raw'/name/'meta.json').exists() for name in names): return
    tasks, source = tasks_and_sources()
    eps = torch.randn(source.shape, generator=torch.Generator().manual_seed(20261185))
    # Same tau0=.7 noise seed as V15: 20261115 + int(.7*100).
    model, _ = load_base()
    ck = torch.load(OUT/'B/adapter.pt', map_location='cpu', weights_only=False)
    net = EditResidualAdapter(ck['width'], ck['blocks']).cuda().eval()
    net.load_state_dict(ck['model']);net.requires_grad_(False)
    teacher = Teacher()
    from src.model import canonicalizer
    from src.utils import load_s10_decoder, decode_s10
    canon, decoder = canonicalizer('cuda'), load_s10_decoder('cuda')
    writers = {name: RawWriter(name) for name in names}
    allinfo, start = [], time.time()
    for st in range(0, 750, args.batch):
        sl = slice(st, st+args.batch); group=tasks[sl]
        z=source[sl].cuda()
        sid=torch.tensor([t['source_style_id'] for t in group], device='cuda')
        tid=torch.tensor([t['target_style_id'] for t in group], device='cuda')
        with torch.no_grad():
            base=generate(model,z,tid,eps[sl].cuda())
            c=canon(z,mode='c1_gentle',augment=False)
            with autocast():
                student=base+net(base,c,tid).float()
                wrong=base+net(base,c,(tid+2)%5).float()
        corrected, info=teacher(base,z,sid,tid)
        for name, value in zip(names,[base,corrected,student,wrong]):
            rgb=decode_s10(decoder,value,batch=1)
            writers[name].append([t['task_id'] for t in group],rgb,value)
        allinfo.extend(info)
        if st % (args.batch*10)==0:print(f'B eval {st+len(z)}/750 {time.time()-start:.0f}s',flush=True)
    for w in writers.values():w.finish(tau0=.7,steps=20,noise_seed=20261185)
    save(OUT/'B/eval_operator.json',{'infos':allinfo,'wall_sec':time.time()-start,
         'teacher_anchor':'V15 base','same_teacher':'forced no edit; student same task behavior is learned, not bypassed'})


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--phase',choices=['prepare','train','eval'],required=True)
    ap.add_argument('--pool',required=True)
    ap.add_argument('--train-each',type=int,default=100)
    ap.add_argument('--hold-each',type=int,default=20)
    ap.add_argument('--batch',type=int,default=8)
    ap.add_argument('--updates',type=int,default=1500)
    ap.add_argument('--seed',type=int,default=20261116)
    a=ap.parse_args()
    {'prepare':prepare,'train':train,'eval':evaluate}[a.phase](a)


if __name__=='__main__':main()
