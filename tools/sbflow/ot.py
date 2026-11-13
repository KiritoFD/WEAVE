"""Pixel-thumbnail minibatch entropic OT, with unbiased row-conditional sampling.

Unlike a greedy without-replacement sampler, conditional sampling respects the
computed Sinkhorn coupling in expectation. No content loss is imposed.
"""
import math
import torch
import torch.nn.functional as F


def image_cost(thumb0,thumb1,mode='rgb'):
    if mode=='rgb':
        x=thumb0.float().flatten(1);y=thumb1.float().flatten(1)
        # Pixel values must be in 0..1; mean squared distance per feature.
        return torch.cdist(x,y,p=2).square()/x.shape[1]
    if mode=='dino':
        # The vectors MUST be pretrained DINO CLS embeddings from cache.py,
        # not flattened RGB thumbnails. Frozen features are NOT a style loss.
        if thumb0.ndim != 2 or thumb1.ndim != 2:
            raise ValueError('dino OT requires cached B,D semantic embeddings')
        x=F.normalize(thumb0.float(),dim=-1)
        y=F.normalize(thumb1.float(),dim=-1)
        return (1-x@y.T).clamp_min(0)
    raise ValueError(mode)


def sinkhorn(cost, reg=0.05, iters=200, tol=0.0, check_every=10):
    """Log-domain Sinkhorn with uniform marginals.

    tol>0 enables early stopping once both marginal errors fall below tol, which
    is the dominant training cost when the coupling is recomputed every microbatch.
    Default tol=0 preserves the exact original behaviour.
    """
    if reg<=0:raise ValueError('reg must be positive')
    c=cost.float()
    n,m=c.shape
    logk=-c/reg
    loga=-math.log(n);logb=-math.log(m)
    u=torch.zeros((n,1),device=c.device,dtype=c.dtype)
    v=torch.zeros((1,m),device=c.device,dtype=c.dtype)
    used=iters
    for it in range(iters):
        u=loga-torch.logsumexp(logk+v,dim=1,keepdim=True)
        v=logb-torch.logsumexp(logk+u,dim=0,keepdim=True)
        if tol>0 and (it+1)%check_every==0:
            pi=torch.exp(logk+u+v)
            if (pi.sum(1)-1/n).abs().max()<tol and (pi.sum(0)-1/m).abs().max()<tol:
                used=it+1
                break
    logpi=logk+u+v
    pi=torch.exp(logpi)
    err0=(pi.sum(1)-1/n).abs().max()
    err1=(pi.sum(0)-1/m).abs().max()
    return pi,{'row_err':err0.item(),'col_err':err1.item(),'mass':pi.sum().item(),'iters_used':used}


def choose_targets(cost,reg,generator=None,iters=200,tol=0.0):
    pi,info=sinkhorn(cost,reg,iters,tol)
    # One source per row. Duplicated target indices are allowed and correct.
    conditional=pi/pi.sum(dim=1,keepdim=True).clamp_min(1e-30)
    idx=torch.multinomial(conditional,1,replacement=True,generator=generator).squeeze(1)
    return idx,info


def paired_batch(z0,z1,thumb0,thumb1,ot_ratio,ot_reg,generator=None,mode='rgb',ot_iters=200,ot_tol=0.0):
    b=z0.shape[0]
    if z1.shape[0]!=b: raise ValueError('Equal batch sizes required')
    if ot_ratio<0 or ot_ratio>1:raise ValueError('Invalid OT ratio')
    if ot_ratio>0:
        cost=image_cost(thumb0,thumb1,mode)
        target_index,info=choose_targets(cost,ot_reg,generator,ot_iters,ot_tol)
    else:
        target_index=torch.arange(b,device=z0.device)
        info={}
    if ot_ratio<1:
        select=torch.rand(b,device=z0.device,generator=generator)<ot_ratio
        target_index=torch.where(select,target_index,torch.arange(b,device=z0.device))
    return z0,z1[target_index],info
