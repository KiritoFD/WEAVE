"""Serial, resumable 3060 execution. Keep SSH foreground; no other jobs touched."""
import argparse
import csv
import datetime
import os
import subprocess
import sys
import time
import numpy as np
import torch
from . import ROOT, OUT, read, save, sha, tasks_and_sources


def preflight():
    required=['runs/v15/W_dim96_conv_base/final.pt',
      'runs/v9_local_4070/bank/style_bank.pt','runs/v9_local_4070/cache/n2c_z.pt',
      'runs/v8/train/N2C/step_04000.pt','runs/v6_next/probes/r1_r1.pt',
      'runs/v6/cache/canonicalizer_stats.pt',
      'runs/v6_r2/cache/tasks750_source_features.pt',
      'eval_protocol/cache/references_v1_features.pt',
      'runs/v11b/flow_v1_w56/ckpt_100.pt',
      'deliver/checkpoints/runs/vae_decoder_prune/s10_p5_lowlr/decoder.pt']
    for p in required:
        if not (ROOT/p).is_file():raise FileNotFoundError(ROOT/p)
    from src.utils import load_s10_decoder,decode_s10
    from src.model.edit_adapter import EditResidualAdapter
    tasks,zs=tasks_and_sources()
    assert sum(t['source_style_id']!=t['target_style_id'] for t in tasks)==600
    s10=load_s10_decoder('cuda');s10.requires_grad_(False)
    z=zs[:1].cuda().detach().requires_grad_(True)
    rgb=decode_s10(s10,z,batch=1,enable_grad=True,output_cpu=False)
    grad=torch.autograd.grad(rgb.square().mean()*1024,z)[0]/1024
    if not torch.isfinite(grad).all() or not grad.norm()>0:
        raise FloatingPointError('S10 differentiable decode canary failed')
    reference=decode_s10(s10,z.detach(),batch=1,output_cpu=False)
    assert torch.equal(reference,rgb.detach()),'Gradient mode changed forward decoding'
    torch.manual_seed(20261116)
    net=EditResidualAdapter().cuda()
    residual=net(z.detach(),z.detach(),torch.zeros(1,device='cuda',dtype=torch.long))
    assert residual.abs().max()==0,'Adapter must start as exact identity'
    from src.utils import FeatureStack,reference_bank
    from PIL import Image
    stack=FeatureStack('cuda');bank=reference_bank('cuda')
    img=Image.fromarray((reference[0].cpu().permute(1,2,0)*255).round().byte().numpy())
    features=stack.extract([img])
    for key,value in features.items():
        if not np.isfinite(np.asarray(value.cpu() if torch.is_tensor(value) else value)).all():
            raise FloatingPointError(f'Frozen referee canary {key}')
    code=['src/utils/vae.py','src/model/edit_adapter.py']+[str(p.relative_to(ROOT)).replace('\\','/') for p in (ROOT/'exp/v16').glob('*.py')]
    save(OUT/'preflight.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'python':sys.executable,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),
      'task_count':750,'cross_count':600,'s10_gradient_norm':float(grad.norm()),
      'referee_feature_shapes':{k:list(v.shape) for k,v in features.items()},
      's10_forward_equal':True,'adapter_params':sum(p.numel() for p in net.parameters()),
      'code_sha256':{p:sha(ROOT/p) for p in code},
      'assets':{p:{'bytes':(ROOT/p).stat().st_size} for p in required},
      's10_sha256':sha(ROOT/required[-1]),
      'task_sha256':sha(ROOT/'eval_protocol/tasks_750.json')})
    print('PREFLIGHT_OK real S10 gradient, exact forward, zero adapter, 750 tasks',flush=True)


def command(label,module,args):
    logs=OUT/'logs';logs.mkdir(parents=True,exist_ok=True)
    status=read(OUT/'status.json') if (OUT/'status.json').exists() else {}
    if status.get(label,{}).get('state')=='complete':
        print(f'SKIP {label}',flush=True);return
    start=time.time()
    status[label]={'state':'running','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   'args':[sys.executable,'-u','-m',module,*args]}
    save(OUT/'status.json',status)
    with (logs/(label+'.log')).open('a',encoding='utf-8') as log:
        proc=subprocess.Popen([sys.executable,'-u','-m',module,*args],cwd=ROOT,
                 stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
        for line in proc.stdout:
            log.write(line);log.flush();print(f'[{label}] {line}',end='',flush=True)
        code=proc.wait()
    status[label].update(state='complete' if code==0 else 'failed',exit_code=code,
                          wall_sec=time.time()-start)
    save(OUT/'status.json',status)
    if code:raise RuntimeError(f'{label} exited {code}; see {logs/(label+".log")}')


def paired(a,b):
    byid={r['task_id']:r for r in b}
    rows=[r for r in a if r['is_cross']]
    assert len(rows)==600
    stems=sorted(set(r['src_stem'] for r in rows));assert len(stems)==150
    metrics=['dino_s','dino_c','lpips','target_top1','target_margin']
    matrix=np.array([[sum(float(r[k])-float(byid[r['task_id']][k]) for r in rows if r['src_stem']==s)/4 for k in metrics] for s in stems])
    counts=np.random.default_rng(20261116).multinomial(len(stems),np.ones(len(stems))/len(stems),size=2000)
    dist=counts@matrix/len(stems)
    return {k:{'mean':float(matrix[:,i].mean()),'ci95':np.quantile(dist[:,i],[.025,.975]).tolist()}
            for i,k in enumerate(metrics)}


def report():
    from src.eval.metrics import summarize_rows
    measures={}
    for p in sorted((OUT/'measured').glob('*/eval.json')):
        d=read(p);assert d['n_total']==750 and d['n_cross']==600 and d['n_same']==150
        measures[p.parent.name]=d
    pairs={}
    for arm,base in [('B_teacher','B_base'),('B_adapter','B_base'),('B_adapter_wrong','B_adapter'),
                     ('C_h2_default','C_base'),('C_h2_strong','C_base'),
                     ('C_h2_strong_wrong','C_h2_strong'),
                     ('C_project_strong','C_h2_strong'),('C_project_strong','C_base')]:
        pairs[arm+'__vs__'+base]=paired(measures[arm]['rows'],measures[base]['rows'])
    simple=[]
    for arm,d in measures.items():
        for split,cross in [('cross600',True),('same150',False)]:
            rows=[r for r in d['rows'] if r['is_cross']==cross]
            means={k:float(np.mean([r[k] for r in rows])) for k in ['dino_s','dino_c','lpips','target_top1','target_margin','dT']}
            means['clip_s']=float(np.mean([r['clip_scores_5'][r['tgt_style_id']] for r in rows]))
            means['csd_source_score']=float(np.mean([r['csd_scores_5'][r['src_style_id']] for r in rows]))
            means['csd_target_score']=float(np.mean([r['csd_scores_5'][r['tgt_style_id']] for r in rows]))
            simple.append({'arm':arm,'split':split,'n':len(rows),**means})
    save(OUT/'summary.json',{'metrics':simple,'paired':pairs,'preflight':read(OUT/'preflight.json'),
         'A_original_path_id_holdout':read(OUT/'A/results.json'),
         'A_strict':read(OUT/'A_strict/results.json') if (OUT/'A_strict/results.json').exists() else None,
         'data_audit':read(OUT/'data_audit.json') if (OUT/'data_audit.json').exists() else None,
         'B_fit_original':read(OUT/'B/fit.json'),
         'B_teacher':read(OUT/'B/teacher_pair_summary.json'),'C':read(OUT/'C/results.json')['summary']})
    with (OUT/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(simple[0]));w.writeheader();w.writerows(simple)
    lines=['# V16 A/B/C 3060 实测结果','',
           '协议：完整test750；cross600主读数、same150单列；全部S10；无训练用Test750。',
           '解读限制：LD4在V15点重新锚定内容预算；不是复现旧LD4本身的35.17%。',
           'C是有限步非凸反演的实得吸收率，不是S10不可表达性的数学证明。','',
           '数据审计：Test750源零交集；B训练497画/500路径，参考重合12，留出参考重合3。',
           'B留出剔除2幅跨集重复后98源/392cross，CPU fp32 relMSE=.93502/cos=.25574。',
           'A初版路径留出5/40存在训练副本；严格画作留出见A_strict，未生成时为空。','',
           '| arm | subset | DINO-S | DINO-C | LPIPS | CSD Top1 | margin | CLIP-S |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in simple:
        lines.append('| '+r['arm']+' | '+r['split']+' | '+' | '.join(f'{r[k]:.5f}' for k in ['dino_s','dino_c','lpips','target_top1','target_margin','clip_s'])+' |')
    lines+=['','## 配对源图聚类95%CI（2000次，重复簇计数保留）','',
            '| comparison | metric | delta | lo | hi |','|---|---|---:|---:|---:|']
    for comparison,result in pairs.items():
        for metric,v in result.items():
            lines.append(f'| {comparison} | {metric} | {v["mean"]:+.6f} | {v["ci95"][0]:+.6f} | {v["ci95"][1]:+.6f} |')
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('V16_ALL_COMPLETE',flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--pool',required=True)
    ap.add_argument('--preflight-only',action='store_true')
    ap.add_argument('--projection-steps',type=int,default=40)
    a=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
    torch.set_num_threads(4)
    pf=read(OUT/'preflight.json') if (OUT/'preflight.json').exists() else None
    tested=['src/utils/vae.py','src/model/edit_adapter.py','exp/v16/__init__.py']
    reusable=pf is not None and all(pf['code_sha256'].get(p)==sha(ROOT/p) for p in tested)
    reusable=reusable and pf['task_sha256']==sha(ROOT/'eval_protocol/tasks_750.json')
    reusable=reusable and pf['s10_sha256']==sha(ROOT/'deliver/checkpoints/runs/vae_decoder_prune/s10_p5_lowlr/decoder.pt')
    if reusable and not a.preflight_only:
        print('REUSE_PREFLIGHT unchanged tested API, decoder and task hashes',flush=True)
    else:
        preflight()
    torch.cuda.empty_cache()
    if a.preflight_only:return
    pool=['--pool',a.pool]
    command('A_diagnose','exp.v16.diagnose',pool)
    for phase in ['prepare','train','eval']:
        command('B_'+phase,'exp.v16.distill',['--phase',phase,*pool])
    command('C_project','exp.v16.project_rgb',['--steps',str(a.projection_steps)])
    for raw in sorted((OUT/'raw').glob('*')):
        if not (raw/'meta.json').exists():continue
        command('measure_'+raw.name,'src.eval.measure',['--raw',str(raw),
                '--out',str(OUT/'measured'/raw.name),'--device','cuda','--no-png'])
    report()


if __name__=='__main__':main()
