"""C: finite-step S10 inversion of paired RGB-H2 edits (full test750).

This measures an achievable reconstruction, NOT a mathematical decoder ceiling.
Additional default/unregularized inversions remain partial numerical pilots.
"""
import argparse
import time
import torch
from . import ROOT, OUT, read, save, sha, RawWriter
from src.utils import load_s10_decoder, decode_s10
from exp.v11b.bands import dwt2, idwt2, sobel_mag, band_rms_clip
from exp.v11b.sample_flow import load_flow


@torch.no_grad()
def h2_target(flow, base, tid, eps, sigmas, strong=False, wrong=False):
    bands = dwt2(base)
    hbase, ll = bands['H2'], bands['LL2']
    edge = sobel_mag(ll)
    sigma = torch.tensor([sigmas[str(int(t))]['H2'] for t in tid], device=base.device).view(-1,1,1,1)
    tau0, steps, alpha, cfg = (.3, 8, 1., 1.) if strong else (.6, 4, .5, 1.)
    h = tau0*hbase + (1-tau0)*eps*sigma
    label = (tid+2)%5 if wrong else tid
    dt=(1-tau0)/steps
    for k in range(steps):
        tau=torch.full((len(base),),tau0+k*dt,device=base.device)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            vc=flow.forward_band('H2',h,ll,edge,label,tau,unconditional=False).float()
            if cfg != 1:
                vu=flow.forward_band('H2',h,ll,edge,label,tau,unconditional=True).float()
                vc=vu+cfg*(vc-vu)
        h=h+dt*vc
    edge_decay, clip = (.3,3.) if strong else (.5,1.5)
    mask=(1-edge_decay*edge/edge.amax((1,2,3),keepdim=True).clamp_min(1e-8)).clamp(0,1)
    delta=band_rms_clip((h-hbase)*mask,clip*hbase.flatten(1).square().mean(1).sqrt())
    target=idwt2(bands['LL2'],hbase+alpha*delta,bands['H1']).clamp(0,1)
    return target


def invert(decoder, z0, base, target, lam, steps=40, lr=.002):
    delta=torch.zeros_like(z0,requires_grad=True)
    opt=torch.optim.Adam([delta],lr=lr)
    with torch.no_grad():
        best_obj=(base-target).flatten(1).square().mean(1)
        best_z=z0.clone()
    curves=[]
    for step in range(steps+1):
        if step==0:
            rgb=base
        else:
            rgb=decode_s10(decoder,z0+delta,batch=len(z0),enable_grad=True,output_cpu=False)
        pix=(rgb-target).flatten(1).square().mean(1)
        penalty=delta.flatten(1).square().mean(1)
        obj=pix+lam*penalty
        with torch.no_grad():
            better=obj<best_obj
            best_z[better]=(z0+delta)[better]
            best_obj[better]=obj[better]
        if step in {0,10,20,steps}:
            curves.append({'step':step,'pixel_mse':pix.detach().cpu().tolist(),
                           'latent_mse':penalty.detach().cpu().tolist(),
                           'objective':obj.detach().cpu().tolist()})
        if step==steps:break
        opt.zero_grad(set_to_none=True)
        # At step0, base is detached: run the official differentiable path once.
        if step==0:
            rgb=decode_s10(decoder,z0+delta,batch=len(z0),enable_grad=True,output_cpu=False)
            obj=(rgb-target).flatten(1).square().mean(1)+lam*penalty
        (obj.mean()*1024).backward()
        if delta.grad is None or not torch.isfinite(delta.grad).all():
            raise FloatingPointError('S10 input gradient missing/nonfinite')
        delta.grad.div_(1024)
        opt.step()
    with torch.no_grad():
        best_rgb=decode_s10(decoder,best_z,batch=len(z0),output_cpu=False)
    return best_z.detach(),best_rgb.detach(),curves


@torch.no_grad()
def projection_metrics(base,target,pred,z,z0):
    dt=target-base;dp=pred-base;error=pred-target
    den=dt.flatten(1).square().mean(1)
    total=error.flatten(1).square().mean(1)
    bands_t,bands_p,bands_e=dwt2(dt),dwt2(dp),dwt2(error)
    rows=[]
    for i in range(len(base)):
        row={'target_mse':float(den[i]),'pixel_mse':float(total[i]),
             'absorbed_fraction_rgb':float(1-total[i]/den[i].clamp_min(1e-15)),
             'target_mae_255':float(dt[i].abs().mean()*255),
             'projection_mae_255':float(error[i].abs().mean()*255),
             'delta_z_rms':float((z[i]-z0[i]).square().mean().sqrt()),
             'delta_alignment':float(torch.nn.functional.cosine_similarity(dt[i].flatten(),dp[i].flatten(),dim=0))}
        for band in ['LL2','H2','H1']:
            d=float(bands_t[band][i].square().mean())
            e=float(bands_e[band][i].square().mean())
            row[band+'_target_energy']=d
            row[band+'_residual_energy']=e
            if band=='H2':row['absorbed_fraction_h2']=1-e/max(d,1e-15)
        rows.append(row)
    return rows


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--batch',type=int,default=4)
    ap.add_argument('--steps',type=int,default=40)
    ap.add_argument('--lr',type=float,default=.002)
    a=ap.parse_args()
    if (OUT/'C/results.json').exists():return
    save(OUT/'C/config.json',{'code_sha256':sha(__file__),'batch':a.batch,'memory_format':'contiguous',
         'steps':a.steps,'lr':a.lr,'lambdas':[.001],
         'full_inversion_cases':['strong'],'extra_inversions':'partial pilot only',
         's10_wrapper_sha256':sha(ROOT/'src/utils/vae.py'),
         'h2_checkpoint_sha256':sha(ROOT/'runs/v11b/flow_v1_w56/ckpt_100.pt')})
    tasks=read(ROOT/'eval_protocol/tasks_750.json')['tasks']
    base_dir=OUT/'raw/B_base'
    meta=read(base_dir/'meta.json')
    zs={}
    for ent in meta['batches']:
        d=torch.load(base_dir/ent['file'],map_location='cpu',weights_only=False)
        zs.update(zip(d['ids'],d['z_final']))
    sigmas=read(ROOT/'docs/v11b/P0_PROFILE.json')['sigmas']
    decoder=load_s10_decoder('cuda')
    decoder.requires_grad_(False)
    flow=load_flow(ROOT/'runs/v11b/flow_v1_w56/ckpt_100.pt',device='cuda',width=56)
    flow.requires_grad_(False)
    names=['C_base','C_h2_default','C_h2_strong','C_h2_strong_wrong','C_project_strong']
    writers={name:RawWriter(name) for name in names}
    chunks=OUT/'C/chunks';chunks.mkdir(parents=True,exist_ok=True)
    allrows=[];start=time.time()
    for st in range(0,750,a.batch):
        group=tasks[st:st+a.batch];ids=[t['task_id'] for t in group]
        fname=f'batch_{st:05d}.pt';record=chunks/f'{st:05d}.json'
        if record.exists() and all((w.path/fname).exists() for w in writers.values()):
            allrows+=[r for r in read(record)['rows'] if r['case']=='strong']
            for w in writers.values():
                w.batches.append({'offset':st,'file':fname,'ids':ids});w.n+=len(ids)
            continue
        z0=torch.stack([zs[t] for t in ids]).float().cuda()
        tid=torch.tensor([t['target_style_id'] for t in group],device='cuda')
        base=decode_s10(decoder,z0,batch=len(z0),output_cpu=False)
        # RNG is per-task, independent of batch partition; wrong only changes label.
        eps=torch.stack([torch.randn(9,128,128,generator=torch.Generator().manual_seed(20261116+st+j))
                         for j in range(len(ids))]).cuda()
        default=h2_target(flow,base,tid,eps,sigmas)
        strong=h2_target(flow,base,tid,eps,sigmas,strong=True)
        wrong=h2_target(flow,base,tid,eps,sigmas,strong=True,wrong=True)
        outputs={'C_base':(base,z0),'C_h2_default':(default,None),
                 'C_h2_strong':(strong,None),'C_h2_strong_wrong':(wrong,None)}
        rows=[];curves={}
        for label,target,lam,output_name in [('strong',strong,.001,'C_project_strong')]:
            zo,rgb,curve=invert(decoder,z0,base,target,lam,a.steps,a.lr)
            rec=projection_metrics(base,target,rgb,zo,z0)
            for t,row in zip(group,rec):
                row.update({'task_id':t['task_id'],'source_stem':t['source_stem'],
                            'is_cross':t['source_style_id']!=t['target_style_id'],
                            'case':label,'lambda':lam,'steps':a.steps})
            rows+=rec;curves[label]=curve
            if output_name:outputs[output_name]=(rgb,zo)
        for name,(rgb,z) in outputs.items():writers[name].append(ids,rgb,z)
        save(record,{'rows':rows,'curves':curves,'ids':ids})
        allrows+=rows
        save(OUT/'C/progress.json',{'n_done':st+len(ids),'n_total':750,
             'elapsed_sec':time.time()-start,'steps':a.steps})
        if st%(a.batch*5)==0:print(f'C invert {st+len(ids)}/750 {time.time()-start:.0f}s',flush=True)
    for w in writers.values():w.finish(base='redecode stored fp16 z, not previous RGB',
        h2_noise_seed=20261116,projection_steps=a.steps)
    summary={}
    for case in ['strong']:
        summary[case]={}
        for split,cross in [('cross600',True),('same150',False)]:
            rows=[r for r in allrows if r['case']==case and r['is_cross']==cross]
            assert len(rows)==(600 if cross else 150)
            numeric=[k for k,v in rows[0].items() if isinstance(v,(int,float)) and k not in ['is_cross','steps','lambda']]
            summary[case][split]={k:sum(r[k] for r in rows)/len(rows) for k in numeric}
            summary[case][split]['energy_weighted_absorbed_fraction_rgb']=1-sum(r['pixel_mse'] for r in rows)/max(sum(r['target_mse'] for r in rows),1e-20)
    save(OUT/'C/results.json',{'summary':summary,'rows':allrows,'wall_sec':time.time()-start,
       'steps':a.steps,'lr':a.lr,'lambdas':[.001],
       'extra_inversions':'default/no-reg are partial pilots, not full750 evidence',
       'target':'original RGB H2 flow, only H2 replaced; strong uses tau.3 K8 alpha1 edge.3 rms3',
       'caveat':'Finite-step nonconvex inversion is an achievable fit, not proof of decoder impossibility. Zero penalty control only reports pixel/band diagnostics.'})


if __name__=='__main__':main()
