"""Combine ten independent two-style evaluation JSONs into one frozen Test600.

Each two-domain model handles the 60 ordered tasks for one unordered pair.
This is the blog-faithful alternative to adding an unreported 5-way head.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .eval_test600 import _mmd2

STYLES=['Early_Renaissance','Impressionism','Minimalism','Rococo','Ukiyo_e']


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--results',nargs='+',required=True,help='Ten style_scores.json results')
    p.add_argument('--repo-root',required=True)
    p.add_argument('--out',required=True)
    args=p.parse_args()
    if len(args.results)!=10:raise ValueError('Need exactly 10 unordered pair scores for 5 styles')
    allrows=[];pairs=set()
    for f in args.results:
        d=json.loads(Path(f).read_text(encoding='utf-8'))
        a=d['protocol']['source_style'];b=d['protocol']['target_style']
        pairs.add(frozenset((a,b)))
        allrows+=d['rows']
    if len(pairs)!=10:raise ValueError('10 unique unordered style pairs required')
    for variant in ('generated','dcae_noop'):
        rows=[r for r in allrows if r['variant']==variant]
        ids=[r['task_id'] for r in rows]
        if len(ids)!=600 or len(set(ids))!=600:
            raise AssertionError(f'Expected 600 distinct cross-style tasks for {variant}, got {len(set(ids))}/{len(ids)}')
    refs=torch.load(Path(args.repo_root)/'eval_protocol/cache/references_v1_features.pt',map_location='cpu',weights_only=False)
    gamma=float(refs['rbf_gamma'])
    out={'protocol':'WEAVE frozen Test600','n_cross':600,'models':args.results,'style_first':{}}
    for tag in ('generated','dcae_noop'):
        subset=[r for r in allrows if r['variant']==tag]
        out['style_first'][tag]={'overall':{},'per_target':{}}
        for label,rows in [('overall',subset)]+[(name,[r for r in subset if r['target']==name]) for name in STYLES]:
            res={k:float(np.mean([r[k] for r in rows])) for k in
                 ('csd_dT','csd_dS','target_margin','target_top1','lpips_to_input')}
            if label!='overall':
                generated=torch.tensor([r['csd_embedding'] for r in rows],dtype=torch.float32)
                real=refs['styles'][label]['csd']
                res['csd_mmd2_unbiased']=_mmd2(generated,real,gamma)
                out['style_first'][tag]['per_target'][label]=res
            else:out['style_first'][tag]['overall']=res
    Path(args.out).write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(out['style_first'],ensure_ascii=False,indent=2))

if __name__=='__main__':main()
