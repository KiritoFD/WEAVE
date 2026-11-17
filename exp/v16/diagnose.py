"""A: matched FM error and all-five-label velocity response, without training."""
import argparse
import csv
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from . import ROOT, OUT, STYLES, read, save, sha, load_base, autocast, pool_samples


def energy(z):
    from exp.v11b.bands import dwt2
    return {k: float(v.square().mean()) for k, v in dwt2(z.float()).items() if k != 'LL1'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', required=True)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--per-style', type=int, default=8)
    ap.add_argument('--strict-audit', type=str, help='Audit JSON: replace FM holdout with works whose every pool copy is held out')
    a = ap.parse_args()
    dest = OUT / ('A_strict' if a.strict_audit else 'A'); dest.mkdir(parents=True, exist_ok=True)
    if a.strict_audit:
        manifest = read(OUT/'A/sample_manifest.json')
        audit = read(a.strict_audit)
        selected = audit['strict_original_fm_holdout_selected']
        splits = {'train': [], 'holdout': []}
        for sid, style in enumerate(STYLES):
            ids = selected[str(sid)]
            assert len(ids) == a.per_style
            d = torch.load(Path(a.pool) / f'{sid:02d}_{style}.pt', map_location='cpu', weights_only=False)
            lookup = {str(pid): i for i, pid in enumerate(d['ids'])}
            manifest['styles'][style]['holdout_ids'] = ids
            for split in splits:
                splits[split] += [(d['latents'][lookup[pid]].float(), sid, pid)
                                   for pid in manifest['styles'][style][split+'_ids']]
        for split, rows in splits.items():
            splits[split] = {'z': torch.stack([r[0] for r in rows]),
                             'sid': torch.tensor([r[1] for r in rows]), 'ids': [r[2] for r in rows]}
        manifest['holdout_protocol'] = 'NFC painting-stem groups; all copies in original FM holdout; no Test750 source or reference'
        manifest['training_protocol'] = 'Same original 40 diagnostic training works; one reference overlap retained'
        manifest['audit_sha256'] = sha(a.strict_audit)
    else:
        splits, manifest = pool_samples(a.pool, a.per_style, a.per_style)
    save(dest / 'sample_manifest.json', manifest)
    arms = [('W_dim96_conv_base', ROOT/'runs/v15/W_dim96_conv_base/final.pt')]
    arms += [(p.parent.name, p) for p in sorted((ROOT/'runs/v15/scaling').glob('*/final.pt'))]
    missing = [name for name in ['S7_dim512', 'B8k_dim96'] if not any(n == name for n, _ in arms)]
    rows, summaries = [], []
    time_grid = [i / 10 for i in range(11)]
    for name, path in arms:
        model, ck = load_base(path)
        t0 = time.time()
        for split, d in splits.items():
            eps = torch.randn(d['z'].shape, generator=torch.Generator().manual_seed(20261117))
            for tau in time_grid:
                values = {k: [] for k in ['fm_mse', 'wrong_fm_mse', 'style_response', 'v_rms']}
                for start in range(0, len(d['z']), a.batch):
                    sl = slice(start, start + a.batch)
                    real, sid, noise = d['z'][sl].cuda(), d['sid'][sl].cuda(), eps[sl].cuda()
                    zt = tau * real + (1 - tau) * noise
                    tt = torch.full((len(real),), tau, device='cuda')
                    with torch.no_grad(), autocast():
                        predictions = torch.stack([model(zt, torch.full_like(sid, s), tt).float() for s in range(5)], 1)
                    correct = predictions[torch.arange(len(real), device='cuda'), sid]
                    wrong = predictions[torch.arange(len(real), device='cuda'), (sid + 2) % 5]
                    target = real - noise
                    norm = predictions.flatten(2).square().mean(2).sqrt().mean(1)
                    diffs = torch.stack([(predictions[:, i] - predictions[:, j]).flatten(1).square().mean(1).sqrt()
                                         for i in range(5) for j in range(i + 1, 5)], 1).mean(1)
                    vals = {'fm_mse': (correct-target).flatten(1).square().mean(1),
                            'wrong_fm_mse': (wrong-target).flatten(1).square().mean(1),
                            'style_response': diffs / norm.clamp_min(1e-12), 'v_rms': norm}
                    for k, v in vals.items(): values[k] += v.cpu().tolist()
                row = {'arm': name, 'base_dim': ck['config']['base_dim'],
                       'updates': ck.get('opt_step'), 'split': split, 'tau': tau,
                       'n': len(d['z']), **{k: sum(v)/len(v) for k, v in values.items()}}
                rows.append(row)
                print(f'A {name} {split} t={tau:.1f} mse={row["fm_mse"]:.5f} R={row["style_response"]:.4f}', flush=True)
        # Frequencies and source/target CSD are read from existing full750 outputs.
        evidence = []
        for base in [ROOT/'runs/v15/eval_measure', ROOT/'runs/v15/eval_measure_scaling']:
            for p in base.glob(f'{name}_t0_0.7/eval.json'):
                evidence.append(str(p))
            for p in base.glob(f'**/{name}/t0_0.7/eval.json'):
                evidence.append(str(p))
            for p in base.glob(f'{name}*/t0_0.7/eval.json'):
                if str(p) not in evidence: evidence.append(str(p))
        raw = ROOT/'runs/v15/eval/raw'/name/'t0_0.7'
        freq = {'latent': [], 'rgb': []}
        if (raw/'meta.json').exists():
            meta = read(raw/'meta.json')
            assert meta['n_tasks'] == 750
            for ent in meta['batches']:
                p = torch.load(raw/ent['file'], map_location='cpu', weights_only=False)
                if 'z_final' in p: freq['latent'].append({'n': len(p['ids']), **energy(p['z_final'])})
                freq['rgb'].append({'n': len(p['ids']), **energy(p['rgb_u8'].float()/255)})
        measured=[]
        for p in evidence:
            old=read(p)
            rr=old.get('rows',[])
            if len(rr)!=750:
                measured.append({'path':p,'status':'not_full750','n':len(rr)})
                continue
            cross=[r for r in rr if r.get('is_cross',r.get('src_style')!=r.get('tgt_style'))]
            if len(cross)!=600:raise ValueError(f'Wrong cross count in {p}')
            stats={k:sum(float(r[k]) for r in cross)/600 for k in
                   ['dino_s','dino_c','lpips','target_top1','target_margin','dT'] if k in cross[0]}
            if 'csd_scores_5' in cross[0]:
                stats['csd_source_score']=sum(r['csd_scores_5'][r['src_style_id']] for r in cross)/600
                stats['csd_target_score']=sum(r['csd_scores_5'][r['tgt_style_id']] for r in cross)/600
            measured.append({'path':p,'sha256':sha(p),'cross600':stats})
        summaries.append({'arm': name, 'path': str(path), 'sha256': sha(path),
                          'updates': ck.get('opt_step'), 'config': ck['config'],
                          'wall_sec': time.time()-t0, 'frequency_batches': freq,
                          'existing_measurements': measured})
        del model; torch.cuda.empty_cache()
        save(dest/'progress.json', {'done': [s['arm'] for s in summaries]})
    with (dest/'curves.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save(dest/'results.json', {'rows': rows, 'arms': summaries, 'missing': missing,
        'note': 't=0/1 are endpoint diagnostics; R_style measures influence magnitude, not correctness. No new training.'})


if __name__ == '__main__': main()
