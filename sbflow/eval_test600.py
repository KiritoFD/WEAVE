"""Frozen WEAVE Test600 evaluator for the two-domain blog reproduction.

The blog describes a two-domain translator; this script evaluates only the
60 ordered Test600 transfers between --style-a and --style-b. Ten unordered
style-pair checkpoints are necessary to cover the full 5-style Test600.

Run `generate` and `score` as separate processes to avoid holding DCAE, DiT,
and WEAVE evaluator CSD/CLIP/DINO/LPIPS weights simultaneously on a 24G GPU.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch


def _tasks(repo_root, a, b):
    raw=json.loads((Path(repo_root)/'eval_protocol/tasks_750.json').read_text(encoding='utf-8'))
    tasks=[t for t in raw['tasks'] if t['source_style'] in (a,b) and
           t['target_style'] in (a,b) and t['source_style'] != t['target_style']]
    if not tasks:
        raise ValueError(f'No frozen cross-domain tasks for styles {a} and {b}')
    for t in tasks:
        if 'source_rel_path' not in t:
            t['source_rel_path']='data/test/'+t['source_style']+'/'+t['source_filename']
    return tasks


def _save_image(z,path):
    u8=((z.detach().float().clamp(-1,1)[0].permute(1,2,0).cpu().numpy()+1)*127.5).round().clip(0,255).astype(np.uint8)
    path.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(u8).save(path)


def generate(args):
    from .cache import image_tensor,load_dcae
    from .infer import load_bundle,solve_pfode
    repo=Path(args.repo_root).resolve()
    tasks=_tasks(repo,args.style_a,args.style_b)
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    ck,model,mom,mean,std=load_bundle(args.ckpt,device,args.weights)
    ae=load_dcae(ck['ae'],device)
    size=int(ck.get('image_size',ck['config'].get('image_size',512)))
    sf=float(ck['ae_scaling_factor'])
    manifest=[]
    import time as _time
    _ph={'io':0.0,'encode':0.0,'ode':0.0,'decode':0.0,'save':0.0}
    _t0=_time.perf_counter()
    with torch.inference_mode():
        for i in range(0,len(tasks),args.batch):
            group=tasks[i:i+args.batch]
            _a=_time.perf_counter()
            x=torch.stack([image_tensor(repo/t['source_rel_path'],size)*2-1 for t in group]).to(device)
            torch.cuda.synchronize();_b=_time.perf_counter();_ph['io']+=_b-_a
            with (torch.autocast('cuda',dtype=torch.bfloat16) if device=='cuda' else __import__('contextlib').nullcontext()):
                zae=ae.encode(x).latent.float()*sf
                torch.cuda.synchronize();_c=_time.perf_counter();_ph['encode']+=_c-_b
                z=(zae-mean[None,:,None,None])/std[None,:,None,None]
                # A->B and B->A have different physical drifts; group by direction.
                zout=torch.empty_like(z)
                for style,dir_name in [(args.style_a,'AtoB'),(args.style_b,'BtoA')]:
                    pos=[j for j,t in enumerate(group) if t['source_style']==style]
                    if pos:
                        zout[pos]=solve_pfode(model,z[pos].clone(),float(ck['config']['eps']),
                                              mom,args.steps,dir_name,args.t_floor,args.method)
                torch.cuda.synchronize();_d=_time.perf_counter();_ph['ode']+=_d-_c
                pred=ae.decode((zout*std[None,:,None,None]+mean[None,:,None,None])/sf).sample
                noop=ae.decode(zae/sf).sample
                torch.cuda.synchronize();_e=_time.perf_counter();_ph['decode']+=_e-_d
            for j,t in enumerate(group):
                key=hashlib.sha256(t['task_id'].encode('utf-8')).hexdigest()[:20]
                pgen=out/'generated'/f'{key}.png';pnoop=out/'dcae_noop'/f'{key}.png'
                _save_image(pred[j:j+1],pgen);_save_image(noop[j:j+1],pnoop)
                manifest.append({**t,'generated':str(pgen),'dcae_noop':str(pnoop)})
            torch.cuda.synchronize();_ph['save']+=_time.perf_counter()-_e
            if i%max(args.batch*10,1)==0:
                print(f'generated {i+len(group)}/{len(tasks)}',flush=True)
    _wall=_time.perf_counter()-_t0
    print('[eval-timing] generate total=%.1fs  io=%.1f encode=%.1f ode=%.1f decode=%.1f save=%.1f'
          % (_wall,_ph['io'],_ph['encode'],_ph['ode'],_ph['decode'],_ph['save']),flush=True)
    nfe=args.steps*(2 if args.method=='euler' else 4)
    m={'timing_generate_sec':{k:round(v,2) for k,v in _ph.items()},
       'timing_generate_total_sec':round(_wall,2),
       'model':args.ckpt,'weight':args.weights,'source_style':args.style_a,
       'target_style':args.style_b,'tasks':manifest,'solver':args.method,
       'solver_steps':args.steps,'network_nfe':nfe,
       'note':'Two-domain Test600 subset, real DCAE no-op baseline; no generated images in training.'}
    path=out/'manifest.json';path.write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
    print('saved',path,'n_tasks',len(tasks),'network_NFE',nfe)


def _mmd2(generated,real,gamma):
    """Unbiased two-sample MMD^2; finite-sample value may be negative."""
    x=torch.as_tensor(generated,dtype=torch.float32)
    y=torch.as_tensor(real,dtype=torch.float32)
    n,m=len(x),len(y)
    if n<2 or m<2:return float('nan')
    def k(a,b):return torch.exp(-gamma*torch.cdist(a,b).square())
    xx=k(x,x);yy=k(y,y)
    same=(xx.sum()-xx.diagonal().sum())/(n*(n-1))
    realv=(yy.sum()-yy.diagonal().sum())/(m*(m-1))
    cross=2*k(x,y).mean()
    return float(same+realv-cross)


def score(args):
    repo=Path(args.repo_root).resolve()
    sys.path.insert(0,str(repo))
    from tools.evaluation_v2.features import EvaluatorModels,build_reference_features_cache
    manifest=json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    tasks=manifest['tasks']
    device='cuda' if torch.cuda.is_available() else 'cpu'
    evaluator=EvaluatorModels(device)
    refs=build_reference_features_cache(evaluator)
    proto=evaluator.proto_csd
    if torch.is_tensor(proto):proto=proto.detach().float().cpu().numpy()
    else:proto=np.asarray(proto,dtype=np.float32)
    style_names=list(manifest.get('style_names',
        ['Early_Renaissance','Impressionism','Minimalism','Rococo','Ukiyo_e']))
    # Repo protocol/style order is fixed; the score component does not infer it.
    gamma=float(refs['rbf_gamma'])
    scores=[]
    import time as _time
    _ph={'io':0.0,'feat':0.0,'src_feat':0.0,'lpips':0.0}
    _t0=_time.perf_counter()
    for start in range(0,len(tasks),args.batch):
        batch=tasks[start:start+args.batch]
        _a=_time.perf_counter()
        imgs={tag:[Image.open(t[tag]).convert('RGB') for t in batch]
              for tag in ('generated','dcae_noop')}
        src=[Image.open(repo/t['source_rel_path']).convert('RGB').resize(imgs['generated'][0].size,Image.Resampling.BICUBIC)
             for t in batch]
        torch.cuda.synchronize();_b=_time.perf_counter();_ph['io']+=_b-_a
        feat={tag:evaluator.extract_image_features(pils) for tag,pils in imgs.items()}
        torch.cuda.synchronize();_c=_time.perf_counter();_ph['feat']+=_c-_b
        with torch.inference_mode():
            from_src=evaluator.extract_image_features(src)
            torch.cuda.synchronize();_ph['src_feat']+=_time.perf_counter()-_c
            for tag in ('generated','dcae_noop'):
                csd=feat[tag]['csd'].numpy()
                src_csd=from_src['csd'].numpy()
                values=csd@proto.T
                src_values=src_csd@proto.T
                for j,t in enumerate(batch):
                    tid=style_names.index(t['target_style'])
                    sid=style_names.index(t['source_style'])
                    other=np.delete(values[j],tid)
                    top=int(values[j].argmax()==tid)
                    # LPIPS only as a severe-destruction diagnostic.
                    a=torch.from_numpy(np.asarray(imgs[tag][j],dtype=np.float32).copy()).permute(2,0,1).to(device)/127.5-1
                    b=torch.from_numpy(np.asarray(src[j],dtype=np.float32).copy()).permute(2,0,1).to(device)/127.5-1
                    _l0=_time.perf_counter()
                    lps=float(evaluator.lpips(a[None],b[None]).mean().item())
                    _ph['lpips']+=_time.perf_counter()-_l0
                    scores.append({'task_id':t['task_id'],'source':t['source_style'],
                       'target':t['target_style'],'variant':tag,
                       'csd_dT':float(values[j,tid]-src_values[j,tid]),
                       'csd_dS':float(values[j,sid]-src_values[j,sid]),
                       'target_margin':float(values[j,tid]-other.max()),
                       'target_top1':top,'lpips_to_input':lps,
                       'csd_embedding':csd[j].tolist()})
        for pils in imgs.values():
            for im in pils:im.close()
        for im in src:im.close()
        print(f'scored {start+len(batch)}/{len(tasks)}',flush=True)
    _wall=_time.perf_counter()-_t0
    print('[eval-timing] score loop=%.1fs  io=%.1f feat=%.1f src_feat=%.1f lpips=%.1f'
          % (_wall,_ph['io'],_ph['feat'],_ph['src_feat'],_ph['lpips']),flush=True)
    agg={}
    for tag in ('generated','dcae_noop'):
        one=[x for x in scores if x['variant']==tag]
        agg[tag]={}
        for style in (manifest['source_style'],manifest['target_style']):
            vals=[v for v in one if v['target']==style]
            feat=np.asarray([v['csd_embedding'] for v in vals],dtype=np.float32)
            reference=refs['styles'][style]['csd']
            keys=('csd_dT','csd_dS','target_margin','target_top1','lpips_to_input')
            agg[tag][style]={k:float(np.mean([v[k] for v in vals])) for k in keys}
            agg[tag][style]['csd_mmd2_unbiased']=_mmd2(feat,reference,gamma)
        agg[tag]['cross_overall']={k:float(np.mean([v[k] for v in one]))
                                    for k in ('csd_dT','csd_dS','target_margin','target_top1','lpips_to_input')}
    output=Path(args.output or Path(args.manifest).with_name('style_scores.json'))
    output.write_text(json.dumps({'protocol':{'source':'WEAVE frozen Test600 (pair subset)',
          'n_tasks':len(tasks),'source_style':manifest['source_style'],
          'target_style':manifest['target_style'],'network_nfe':manifest['network_nfe'],
          'mmd_gamma':gamma,
          'timing_score_sec':{k:round(v,2) for k,v in _ph.items()},
          'timing_score_total_sec':round(_wall,2)},
          'aggregate':agg,'rows':scores},ensure_ascii=False,indent=2),encoding='utf-8')
    print('saved',output)
    print(json.dumps(agg,indent=2))


def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest='command',required=True)
    g=sub.add_parser('generate')
    g.add_argument('--repo-root',required=True)
    g.add_argument('--ckpt',required=True)
    g.add_argument('--style-a',required=True)
    g.add_argument('--style-b',required=True)
    g.add_argument('--output',required=True)
    g.add_argument('--weights',choices=('raw','ema'),default='raw')
    g.add_argument('--steps',type=int,default=4)
    g.add_argument('--method',choices=('euler','heun'),default='heun')
    g.add_argument('--batch',type=int,default=2)
    g.add_argument('--t-floor',type=float,default=.002)
    s=sub.add_parser('score')
    s.add_argument('--repo-root',required=True)
    s.add_argument('--manifest',required=True)
    s.add_argument('--output',default='')
    s.add_argument('--batch',type=int,default=4)
    args=p.parse_args()
    {'generate':generate,'score':score}[args.command](args)

if __name__=='__main__':main()
