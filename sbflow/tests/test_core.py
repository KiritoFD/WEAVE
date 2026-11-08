import math
import torch
from sbflow.bridge import bridge_sample,exact_conditional_flow
from sbflow.preconditioning import (PairMoments,MomentAccumulator,
    independent_unit_coefficients,moment_coefficients,
    precondition_input,normalized_target,reconstruct_drift)
from sbflow.ot import sinkhorn,choose_targets,paired_batch
from sbflow.network import BidirDiT


def test_bridge():
    torch.manual_seed(0)
    x=torch.randn(4,2,3,3);y=torch.randn_like(x)
    t=torch.tensor([.15,.35,.66,.85]);xi=torch.randn_like(x)
    xt,bp,bm=bridge_sample(x,y,t,.04,xi)
    target=exact_conditional_flow(x,y,t,.04,xi)
    assert torch.allclose((bp-bm)/2,target,atol=1e-5,rtol=1e-5)
    # Reverse-time bridge uses exactly the same intermediate state.
    xr,yr,zminus=bridge_sample(y,x,1-t,.04,xi)
    assert torch.allclose(xr,xt,atol=1e-6)
    assert torch.allclose(bp,zminus,atol=1e-5)
    assert torch.allclose(bm,yr,atol=1e-5)


def test_edm_corrected_closed_form():
    ts=torch.tensor([.03,.2,.5,.8,.97])
    eps=.1
    z=torch.zeros(1)
    m=PairMoments(z,z,torch.ones(1),torch.ones(1),z)
    (_,_,ci,cs,co)=moment_coefficients(ts,eps,m)
    ci2,cs2,co2=independent_unit_coefficients(ts,eps)
    for a,b in ((ci.flatten(),ci2),(cs.flatten(),cs2),(co.flatten(),co2)):
        assert torch.allclose(a,b,atol=2e-6,rtol=2e-6),(a,b)


def test_edm_empirical_correlated():
    torch.manual_seed(0)
    n=30000
    x0=torch.randn(n,2,1,1)
    x1=.7*x0+.5*torch.randn_like(x0)+.2
    acc=MomentAccumulator()
    for st in range(0,n,1000):acc.update(x0[st:st+1000],x1[st:st+1000])
    mom=acc.finalize()
    for ts in (.1,.5,.9):
        t=torch.full((n,),ts)
        xt,yp,ym=bridge_sample(x0,x1,t,.1)
        for a,b,m in ((yp,t,mom),(ym,1-t,mom.reversed())):
            xin,c=precondition_input(xt,b,.1,m)
            y=normalized_target(xt,a,c)
            # By construction a centered linearly whitened residual.
            assert abs(y.mean().item())<.045,(ts,y.mean())
            assert abs(y.square().mean().item()-1)<.07,(ts,y.square().mean())
            assert abs(xin.mean().item())<.045
            assert abs(xin.square().mean().item()-1)<.07
            assert torch.allclose(reconstruct_drift(xt,y,c),a,atol=1e-5,rtol=1e-5)


def test_ot():
    c=torch.rand(16,32)
    pi,info=sinkhorn(c,reg=.04,iters=300)
    assert info['row_err']<1e-5 and info['col_err']<1e-5
    idx,_=choose_targets(c,.04)
    assert idx.shape==(16,)
    x=torch.randn(8,2,3,3);y=torch.randn_like(x)
    u=torch.rand(8,3,4,4);v=torch.rand_like(u)
    x2,y2,info=paired_batch(x,y,u,v,.5,.05)
    assert x2.shape==y2.shape==x.shape


def test_network():
    model=BidirDiT(channels=4,image_size=8,patch=2,width=48,depth=2,heads=4)
    x=torch.randn(4,4,8,8)
    t=torch.rand(4);direction=torch.tensor([0,1,1,0])
    out=model(x,t,direction)
    assert out.shape==x.shape
    assert torch.isfinite(out).all()
    assert out.abs().max()==0


if __name__=='__main__':
    for n,f in list(globals().items()):
        if n.startswith('test_') and callable(f):
            f();print('PASS',n)
