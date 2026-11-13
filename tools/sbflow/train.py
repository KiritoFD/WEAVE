"""Faithful-to-article TWO-DOMAIN bidirectional miniOT Brownian bridge matching.

No alpha-DSBM, no RGB/content/GAN/Gram/CLIP/style-classifier loss.
This script requires precomputed DCAE caches from tools.sbflow.cache.
"""
import argparse
import contextlib
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from .bridge import bridge_sample
from .network import BidirDiT
from .ot import paired_batch
from .preconditioning import MomentAccumulator,PairMoments,precondition_input,normalized_target


def cache_read(path):
    d=torch.load(path,map_location='cpu',weights_only=False)
    for k in ('latent','thumb','ae','scaling_factor','image_size'):assert k in d,k
    assert d['latent'].shape[0]==d['thumb'].shape[0]
    return d


def pick_pair(A,B,batch,ot_ratio,ot_reg,generator,ot_cost,ot_iters=200,ot_tol=0.0):
    dev=A['z'].device
    ia=torch.randint(len(A['z']),(batch,),device=dev,generator=generator)
    ib=torch.randint(len(B['z']),(batch,),device=dev,generator=generator)
    feat='dino' if ot_cost=='dino' else 'thumb'
    return paired_batch(A['z'][ia],B['z'][ib],A[feat][ia],B[feat][ib],
                        ot_ratio,ot_reg,generator,mode=ot_cost,ot_iters=ot_iters,ot_tol=ot_tol)


def estimate_moments(A,B,eps,batch,ot_ratio,ot_reg,n_batches,gen,ot_cost,ot_iters=200,ot_tol=0.0):
    mom=MomentAccumulator()
    for _ in range(n_batches):
        x,y,_=pick_pair(A,B,batch,ot_ratio,ot_reg,gen,ot_cost,ot_iters,ot_tol)
        mom.update(x,y)
    return mom.finalize()


def loss_batch(model,A,B,mom,epsilon,batch,ot_ratio,ot_reg,min_t,gen,loss_mode,ot_cost,want_info=True,
               ot_iters=200,ot_tol=0.0):
    z0,z1,info=pick_pair(A,B,batch,ot_ratio,ot_reg,gen,ot_cost,ot_iters,ot_tol)
    t=torch.rand(batch,device=z0.device,generator=gen)*(1-2*min_t)+min_t
    xi=torch.randn(z0.shape,device=z0.device,generator=gen,dtype=z0.dtype)
    xt,yplus,yminus=bridge_sample(z0,z1,t,epsilon,xi)
    half=batch//2
    xf,cf=precondition_input(xt[:half],t[:half],epsilon,mom)
    xb,cb=precondition_input(xt[half:],1-t[half:],epsilon,mom.reversed())
    y_true=torch.cat([normalized_target(xt[:half],yplus[:half],cf),
                      normalized_target(xt[half:],yminus[half:],cb)],0)
    x=torch.cat([xf,xb],0)
    tau=torch.cat([t[:half],1-t[half:]],0)
    d=torch.cat([torch.ones(half,device=z0.device,dtype=torch.long),
                 torch.zeros(batch-half,device=z0.device,dtype=torch.long)])
    y_pred=model(x,tau,d).float()
    if loss_mode=='normalized': loss=F.mse_loss(y_pred,y_true.float())
    else:
        # Physical drift MSE: multiply residual by its time-dependent c_out.
        scale=torch.cat([cf[-1],cb[-1]],dim=0)
        loss=((y_pred-y_true.float())*scale).square().mean()
    if not want_info:
        # Avoid per-microbatch .item() GPU syncs; only log intervals need scalars.
        return loss,info
    return loss,{'f_mse':(y_pred[:half]-y_true[:half]).square().mean().item(),
                 'b_mse':(y_pred[half:]-y_true[half:]).square().mean().item(),
                 **info}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source-cache',required=True)
    p.add_argument('--target-cache',required=True)
    p.add_argument('--output',default='runs/blog_sbflow')
    p.add_argument('--steps',type=int,default=50000)
    p.add_argument('--microbatch',type=int,default=16)
    p.add_argument('--accum',type=int,default=4)
    p.add_argument('--eps',type=float,default=0.04)
    p.add_argument('--ot-ratio',type=float,default=0.5)
    p.add_argument('--ot-reg',type=float,default=0.05)
    p.add_argument('--ot-cost',choices=['rgb','dino'],default='rgb')
    p.add_argument('--ot-iters',type=int,default=200,
                   help='Sinkhorn iteration cap (200 is the reference default)')
    p.add_argument('--ot-tol',type=float,default=0.0,
                   help='Early-stop marginal tolerance; >0 collapses ~200 iters to ~10')
    p.add_argument('--min-t',type=float,default=.002)
    p.add_argument('--moments-batches',type=int,default=64)
    p.add_argument('--lr',type=float,default=2e-4)
    p.add_argument('--ema-decay',type=float,default=.999)
    p.add_argument('--width',type=int,default=384)
    p.add_argument('--depth',type=int,default=12)
    p.add_argument('--heads',type=int,default=6)
    p.add_argument('--patch',type=int,default=2)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--log-interval',type=int,default=100)
    p.add_argument('--save-interval',type=int,default=5000)
    p.add_argument('--loss-mode',choices=['normalized','physical'],default='normalized')
    p.add_argument('--warmup',type=int,default=-1,
                   help='LR warmup steps; -1 keeps the legacy min(1000, steps//10)')
    p.add_argument('--resume',default='')
    args=p.parse_args()
    if args.eps<=0 or args.min_t<=0 or args.min_t>=.5:raise ValueError('Bad Brownian parameters')
    if args.microbatch%2:raise ValueError('Even microbatch required')
    if args.ot_ratio<0 or args.ot_ratio>1:raise ValueError('OT ratio must be 0..1')
    torch.manual_seed(args.seed)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    cache0=cache_read(args.source_cache);cache1=cache_read(args.target_cache)
    for k in ('ae','scaling_factor','image_size'):
        if cache0[k]!=cache1[k]:raise ValueError(f'Mismatched DCAE metadata: {k}')
    raw0=cache0['latent'].float();raw1=cache1['latent'].float()
    channels,h,w=raw0.shape[1:]
    if raw1.shape[1:]!=(channels,h,w) or h!=w:raise ValueError('latent shape mismatch')
    # A single affine transform for BOTH domains; never independently whiten
    # source and target, which could erase real domain/style differences.
    both=torch.cat([raw0,raw1],dim=0)
    lat_mean=both.mean((0,2,3)); lat_std=both.var((0,2,3),unbiased=False).clamp_min(1e-8).sqrt()
    del both
    A={'z':((raw0-lat_mean[None,:,None,None])/lat_std[None,:,None,None]).to(device),
       'thumb':cache0['thumb'].float().to(device)/255.}
    B={'z':((raw1-lat_mean[None,:,None,None])/lat_std[None,:,None,None]).to(device),
       'thumb':cache1['thumb'].float().to(device)/255.}
    del raw0,raw1
    if args.ot_cost=='dino':
        if 'dino' not in cache0 or 'dino' not in cache1:
            raise ValueError('DINO OT requested but cache missing `dino`; rerun cache.py with --include-dino')
        if cache0['dino_model']!=cache1['dino_model']:
            raise ValueError('Source/target DINO feature models disagree')
        A['dino']=cache0['dino'].float().to(device)
        B['dino']=cache1['dino'].float().to(device)
    gen=torch.Generator(device=device).manual_seed(args.seed*17+1)
    moments=estimate_moments(A,B,args.eps,args.microbatch,args.ot_ratio,args.ot_reg,
                            args.moments_batches,gen,args.ot_cost,args.ot_iters,args.ot_tol)
    model_kwargs={'channels':channels,'image_size':h,'patch':args.patch,'width':args.width,
                  'depth':args.depth,'heads':args.heads}
    model=BidirDiT(**model_kwargs).to(device)
    ema=BidirDiT(**model_kwargs).to(device)
    ema.load_state_dict(model.state_dict());ema.requires_grad_(False)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    # cosine with warmup is a documented reconstruction choice, not a blog setting
    warm=min(1000,args.steps//10) if args.warmup<0 else max(1,args.warmup)
    def lr_ratio(s):
        if s<warm:return max(.02,(s+1)/max(1,warm))
        v=(s-warm)/max(1,args.steps-warm)
        return .05+.95*.5*(1+math.cos(math.pi*min(1,v)))
    schedule=torch.optim.lr_scheduler.LambdaLR(opt,lr_lambda=lr_ratio)
    start_step=0
    if args.resume:
        old=torch.load(args.resume,map_location='cpu',weights_only=False)
        model.load_state_dict(old['model']);ema.load_state_dict(old['ema']);opt.load_state_dict(old['optimizer'])
        schedule.load_state_dict(old['scheduler']);start_step=old['step']
        moments=PairMoments.from_state_dict(old['moments']);lat_mean=old['latent_mean'];lat_std=old['latent_std']
        if old['config']['source_cache']!=args.source_cache or old['config']['target_cache']!=args.target_cache:
            raise ValueError('Cannot resume on a different data cache')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    npars=sum(q.numel() for q in model.parameters()); print('params:',npars, 'device:',device,flush=True)
    def save(step):
        torch.save({'model':model.state_dict(),'ema':ema.state_dict(),
                    'optimizer':opt.state_dict(),'scheduler':schedule.state_dict(),'step':step,
                    'latent_mean':lat_mean.cpu(),'latent_std':lat_std.cpu(),
                    'moments':moments.state_dict(),'model_kwargs':model_kwargs,
                    'ae':cache0['ae'],'ae_scaling_factor':cache0['scaling_factor'],
                    'config':vars(args)},output/f'model_step{step}.pt')
    prof={'ot':0.0,'fwd':0.0,'opt':0.0,'n':0}
    for step in range(start_step+1,args.steps+1):
        model.train(); opt.zero_grad(set_to_none=True)
        total=0.; rec={}
        want_info=(step%args.log_interval==0 or step==1)
        for accumulation in range(args.accum):
            context=(torch.autocast('cuda',dtype=torch.bfloat16) if device=='cuda'
                     else contextlib.nullcontext())
            with context:
                loss,info=loss_batch(model,A,B,moments,args.eps,args.microbatch,
                                     args.ot_ratio,args.ot_reg,args.min_t,gen,args.loss_mode,args.ot_cost,
                                     want_info=want_info,ot_iters=args.ot_iters,ot_tol=args.ot_tol)
            (loss/args.accum).backward()
            if want_info:
                total+=loss.item()/args.accum;rec=info
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step();schedule.step()
        with torch.no_grad():
            # EMA warm-up prevents bias from dominating early checkpoints.
            decay=min(args.ema_decay,(step+1)/(step+10))
            for pe,pv in zip(ema.parameters(),model.parameters()):
                pe.lerp_(pv,1-decay)
        if step%args.log_interval==0 or step==1:
            print(f'step={step}/{args.steps} loss={total:.4f} f={rec.get("f_mse",-1):.3f} '
                  f'b={rec.get("b_mse",-1):.3f} grad={float(grad):.3f} '
                  f'OT_row={rec.get("row_err",-1):.1e} LR={opt.param_groups[0]["lr"]:.2e}',flush=True)
        if step%args.save_interval==0 or step==args.steps:save(step)
    print('Saved to',output)

if __name__=='__main__':main()
