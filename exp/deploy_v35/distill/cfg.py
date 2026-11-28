"""Single-branch continuous-guidance P-Core and true teacher-rollout supervision."""
from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from exp.deploy_v35.bundle import file_sha256, load_pcore_from_file
from exp.deploy_v35.distill.data import batches, capture_rng, configure, restore_rng, save_checkpoint, write_json
from exp.deploy_v35.pipeline import _precompute_dpm2m_schedule, make_optimized_pipeline
from exp.deploy_v35.torch_opt import patch_bottleneck_sdpa
from exp.v18.ccf_model import sinusoidal_time_emb
from exp.v29_biglittle.pcore_model import PCoreSLFUNet
from exp.v32_nodino.cnx_layout import match_cnx_k8_submode
from exp.v34_frontier.stage_a_startpoint import reconstruct_analytic_draft
from exp.v35_final.common import make_v35_schedule
from exp.deploy_v35 import P8_LOCKED_CFG


class GuidedPCore(nn.Module):
    """Teacher initialization; zero-init projection of ws/wc/wsp/phi/gamma/layout.

    Predicts the already merged and rescaled field in one forward. Conditional
    dropout and training-only Q32 jitter are intentionally disabled.
    """
    def __init__(self, backbone: PCoreSLFUNet):
        super().__init__()
        self.m = backbone
        self.guidance = nn.Linear(6, backbone.cond_mlp[0].out_features, bias=False)
        nn.init.zeros_(self.guidance.weight)
        patch_bottleneck_sdpa(self.m)

    def forward(self, z, y, q, tid, u, tau, control):
        m = self.m
        g = control[:, 4].view(-1, 1, 1, 1)
        ls = control[:, 5].view(-1, 1, 1, 1)
        qc = F.interpolate(F.avg_pool2d(q, 4), scale_factor=4., mode='nearest')
        qc = qc - qc.mean((2,3), keepdim=True)
        qeff = g*q+(1-g)*qc
        s16 = torch.cat([y*g*ls, m.dino_proj(qeff)*ls], 1)
        oh = F.one_hot(tid.clamp(0,4),5).float()
        flags = torch.stack([torch.zeros_like(tau), 1-control[:,5]], -1)
        vec = torch.cat([sinusoidal_time_emb(tau,64), oh, F.normalize(u,dim=-1)*2., flags,
                         sinusoidal_time_emb(control[:,4],16)], -1)
        scaled = control / control.new_tensor([4.,4.,4.,1.,1.,1.])
        cond = m.cond_mlp[3](m.cond_mlp[2](F.silu(m.cond_mlp[0](vec)+self.guidance(scaled))))
        h64 = m.in_conv(z)
        for block in m.enc64:
            h64 = block(h64,cond)
        h32 = m.down32(h64)
        for block in m.enc32:
            h32 = block(h32,cond)
        h16 = m.mid1(m.down16(h32), s16, cond)
        h16 = m.mid2(m.mid_attn(h16), s16, cond)
        u32 = m.skip32(torch.cat([m.up32(F.interpolate(h16,scale_factor=2.,mode='nearest')),h32],1))
        for block in m.dec32:
            u32 = block(u32,cond)
        u64 = m.skip64(torch.cat([m.up64(F.interpolate(u32,scale_factor=2.,mode='nearest')),h64],1))
        for block in m.dec64:
            u64 = block(u64,cond)
        return m.out_conv(F.silu(m.out_norm(u64))).float()


def combine(v4: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
    sc,s0,oc,oo = v4.float().chunk(4,0)
    ws,wc,wsp,phi = [control[:,i].view(-1,1,1,1) for i in range(4)]
    v = oo+wc*(oc-oo)+ws*(sc-oc)+wsp*(s0-oo)
    std_ref = sc.flatten(1).std(1).view(-1,1,1,1).clamp_min(1e-4)
    std = v.flatten(1).std(1).view(-1,1,1,1).clamp_min(1e-4)
    return phi*v*(std_ref/std)+(1-phi)*v


def integrate(velocity, ze, eps, *, tau0, ws, wc, wsp, rescale_phi, gamma, lscale, steps=6,
              startup='midpoint'):
    """Same midpoint startup and DPM-2M recurrence as the frozen production solver."""
    tau0 = float(tau0) if ze is not None and tau0 > 0 else 0.
    z = (1-tau0)*eps+tau0*ze if tau0 > 0 else eps.clone()
    previous = None
    for row in _precompute_dpm2m_schedule(steps,tau0,ws,wc,wsp,startup):
        def evaluate(zz, midpoint=False):
            key = 'm' if midpoint else '0'
            control = zz.new_tensor([row['ws'+key],row['wc'+key],row['wsp'+key],rescale_phi,gamma,lscale]).expand(len(zz),-1)
            tau = torch.full((len(zz),), row['tm' if midpoint else 't0'],device=zz.device)
            return velocity(zz,tau,control)
        v = evaluate(z)
        if row['need_midpoint']:
            z = z+row['dt']*evaluate(z+.5*row['dt']*v,True)
        elif previous is not None:
            z = z+row['dt']*(v+.5*row['ratio_prev']*(v-previous))
        else:
            z = z+row['dt']*v
        previous = v
    return z


def load_student(path: Path, device='cuda') -> GuidedPCore:
    ck = torch.load(path,map_location='cpu',weights_only=False)
    m = GuidedPCore(PCoreSLFUNet(**ck['model_cfg']))
    m.load_state_dict(ck['model'],strict=True)
    return m.to(device).eval()


@torch.inference_mode()
def cache_rollouts(config: dict, out: Path) -> None:
    configure(config['seed'])
    out.mkdir(parents=True,exist_ok=True)
    bundle, cache = Path(config['bundle']),Path(config['cache'])
    provenance = dict(pcore_sha256=file_sha256(bundle/'pcore.pt'), a0_cache=file_sha256(cache/'complete.json'),
                      n_train=config['cfg_sources'], n_calib=config['cfg_calib_sources'],steps=6,
                      knobs=[.6,.85,1.,1.5],teacher='optimized_fp16_four_branch_phi_midpoint_v1')
    if (out/'complete.json').exists():
        import json
        if json.loads((out/'complete.json').read_text())['provenance'] != provenance:
            raise ValueError('CFG rollout cache provenance mismatch')
        return
    pipe = make_optimized_pipeline(bundle, compile_mode='')
    counts = {}
    for split,limit in [('train',config['cfg_sources']),('calib',config['cfg_calib_sources'])]:
        dest = out/split
        dest.mkdir(exist_ok=True)
        consumed = 0
        for path in sorted((cache/split).glob('*.pt')):
            d = torch.load(path,map_location='cpu',weights_only=False)
            for begin in range(0,len(d['rgb']),config['cfg_micro_batch']):
                b = min(config['cfg_micro_batch'],limit-consumed,len(d['rgb'])-begin)
                if b <= 0:
                    break
                rgb = d['rgb'][begin:begin+b].cuda().float()/255.
                src = pipe.prepare_source(rgb)
                tid = (torch.arange(consumed,consumed+b,device='cuda')%5).long()
                u = match_cnx_k8_submode(src['g16'],tid,pipe.cnx_bank)
                eps = torch.randn(b,4,64,64,device='cuda')
                for knob in (.6,.85,1.,1.5):
                    target = dest/f'{consumed:06d}_{int(knob*100):03d}.pt'
                    if target.exists():
                        continue
                    kw = make_v35_schedule(knob,ws_mul=P8_LOCKED_CFG['ws_mul'],wc_mul=P8_LOCKED_CFG['wc_mul'],dtau0=P8_LOCKED_CFG['dtau0'])
                    lam = kw.pop('draft_lambda',.75)
                    ze = pipe._run_ecore(src['z_s'],src['c64'],src['q16'],tid,lam,eps)
                    ze = reconstruct_analytic_draft(ze,tid,pipe.priors_a,mode='A3')
                    zero = torch.zeros(b,dtype=torch.bool,device='cuda')
                    one = ~zero
                    ls = torch.full((b,),float(kw['lscale']),device='cuda')
                    args = pipe.fast_pcore.prepare_traj(src['y16'],src['q16'],tid,u,
                                                     torch.cat([zero,zero,one,one]),torch.cat([zero,one,zero,one]),
                                                     torch.cat([ls,ls*0,ls,ls*0]),float(kw['gamma']))
                    gc1,bc1,gc2,bc2,cb = args
                    rows = {k:[] for k in ('z','target','tau','control','y','q','tid','u')}
                    def velocity(z,t,control):
                        merged = combine(pipe.fast_pcore(z,t,cb,gc1,bc1,gc2,bc2),control)
                        values = (z,merged,t,control,src['y16'],src['q16'],tid,u)
                        for k,v in zip(rows,values):
                            rows[k].append(v.detach().cpu().long() if k=='tid' else v.detach().cpu().half())
                        return merged
                    integrate(velocity,ze,eps,**{k:float(kw[k]) for k in ('tau0','ws','wc','wsp','rescale_phi','gamma','lscale')})
                    save_checkpoint(target,{k:torch.cat(v) for k,v in rows.items()})
                consumed += b
                if consumed % 32 == 0:
                    print(f'[CFG cache {split}] sources={consumed}/{limit}',flush=True)
            if consumed >= limit:
                break
        counts[split] = consumed
        if consumed != limit:
            raise ValueError(f'Incomplete CFG {split} source count {consumed}/{limit}')
    write_json(out/'complete.json',dict(provenance=provenance,counts=counts,expected_calls_per_trajectory=7))


KEYS = ('z','target','tau','control','y','q','tid','u')


@torch.no_grad()
def validate(model, paths: list[Path], limit=128) -> dict:
    error,energy,cos,n=0.,0.,0.,0
    for p in paths:
        d=torch.load(p,map_location='cpu',weights_only=False)
        for start in range(0,len(d['z']),2):
            b={k:d[k][start:start+2].cuda().float() if k!='tid' else d[k][start:start+2].cuda().long() for k in KEYS}
            with torch.autocast('cuda',dtype=torch.bfloat16):
                pred=model(b['z'],b['y'],b['q'],b['tid'],b['u'],b['tau'],b['control'])
            if not torch.isfinite(pred).all():
                raise FloatingPointError('Non-finite CFG student')
            error+=(pred-b['target']).square().sum().item()
            energy+=b['target'].square().sum().item()
            cos+=F.cosine_similarity(pred.flatten(1),b['target'].flatten(1)).sum().item()
            n+=len(pred)
            if n>=limit:
                break
        if n>=limit:
            break
    return dict(n=n,velocity_relative_rmse=(error/max(energy,1e-12))**.5,velocity_cosine=cos/n)


def train(config: dict,out: Path) -> None:
    configure(config['seed'])
    out.mkdir(parents=True,exist_ok=True)
    bundle,cache=Path(config['bundle']),Path(config['cfg_cache'])
    cache_hash=file_sha256(cache/'complete.json')
    last=out/'last.pt'
    optimizer_state=None
    if last.exists():
        ck=torch.load(last,map_location='cpu',weights_only=False)
        if ck['cache_hash']!=cache_hash:
            raise ValueError('CFG resume provenance mismatch')
        model=load_student(last)
        model_cfg,step,elapsed,best=ck['model_cfg'],ck['step'],ck['elapsed'],ck['best']
        optimizer_state=ck['optimizer']
        restore_rng(ck['rng'])
    else:
        backbone,_=load_pcore_from_file(bundle/'pcore.pt',device='cpu')
        backbone.requires_grad_(True)
        model_cfg=dict(base_dim=backbone.in_conv.out_channels,u_dim=backbone.u_dim,
                       cond_dim=backbone.cond_mlp[0].out_features,use_gamma=True,normalize_u=True)
        model=GuidedPCore(backbone).cuda()
        step,elapsed,best=0,0.,float('inf')
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.get('cfg_lr',3e-5),weight_decay=1e-4)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    stream=batches(sorted((cache/'train').glob('*.pt')),KEYS,config['cfg_micro_batch'],config['seed']+step)
    calib=sorted((cache/'calib').glob('*.pt'))
    start=time.monotonic()
    write_json(out/'initial.json',validate(model,calib,config.get('calib_limit',128)))
    torch.cuda.reset_peak_memory_stats()
    for step in range(step+1,config['cfg_steps']+1):
        model.train()
        model.m.eval()
        optimizer.zero_grad(set_to_none=True)
        total=0.
        for _ in range(config['accum']):
            b=next(stream)
            b={k:v.long() if k=='tid' else v.float() for k,v in b.items()}
            with torch.autocast('cuda',dtype=torch.bfloat16):
                pred=model(b['z'],b['y'],b['q'],b['tid'],b['u'],b['tau'],b['control'])
                loss=F.smooth_l1_loss(pred,b['target'],beta=.1)+.1*F.mse_loss(pred,b['target'])
            if not torch.isfinite(loss):
                raise FloatingPointError(f'CFG non-finite loss at step={step}')
            (loss/config['accum']).backward()
            total+=loss.detach().item()/config['accum']
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        stop=elapsed+time.monotonic()-start>=config['cfg_seconds'] or step==config['cfg_steps']
        if step%50==0 or stop:
            print(f'[CFG] step={step} loss={total:.5f} elapsed={elapsed+time.monotonic()-start:.0f}s',flush=True)
        if step%300==0 or stop:
            metrics=validate(model,calib,config.get('calib_limit',128))
            if metrics['velocity_relative_rmse']<best:
                best=metrics['velocity_relative_rmse']
                save_checkpoint(out/'best.pt',dict(model=model.state_dict(),model_cfg=model_cfg,step=step,
                                                  metrics=metrics,cache_hash=cache_hash))
            write_json(out/'progress.json',dict(step=step,calibration=metrics,best_score=best))
        if step%100==0 or stop:
            save_checkpoint(last,dict(model=model.state_dict(),model_cfg=model_cfg,optimizer=optimizer.state_dict(),
                                     step=step,elapsed=elapsed+time.monotonic()-start,best=best,
                                     cache_hash=cache_hash,rng=capture_rng()))
        if stop:
            break
    write_json(out/'complete.json',dict(step=step,best_score=best,seconds=elapsed+time.monotonic()-start,
                                      params=sum(p.numel() for p in model.parameters()),
                                      peak_allocated_gb=torch.cuda.max_memory_allocated()/2**30,
                                      checkpoint_sha256=file_sha256(out/'best.pt'),
                                      quality_status='teacher-state regression only; P4 student rollout required'))


def install(pipe,path: Path) -> None:
    model=load_student(path).requires_grad_(False)
    pipe.fast_pcore=None
    pipe.pcore=None
    pipe.ensure_pcore_loaded=lambda: None
    def rollout(ze,eps,y,q,tid,u,**kw):
        def velocity(z,t,control):
            pipe.stats.pcore_batch_calls+=1
            pipe.stats.pcore_cond_evals+=1
            with torch.autocast('cuda',dtype=torch.bfloat16):
                return pipe._get_callable('guided_pcore',len(z),model)(z,y,q,tid,u,t,control)
        return integrate(velocity,ze,eps,**kw)
    pipe._integrate_zero_sync=rollout
