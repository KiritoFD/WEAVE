import torch
from tools.sbflow.network import BidirDiT
from tools.sbflow.preconditioning import PairMoments
from tools.sbflow.infer import ode_velocity,solve_pfode


def test_solver_smoke():
    m=BidirDiT(channels=2,image_size=4,patch=2,width=32,depth=1,heads=4)
    zeros=torch.zeros(2);ones=torch.ones(2)
    moment=PairMoments(zeros,zeros,ones,ones,zeros)
    z=torch.randn(3,2,4,4)
    t=torch.full((3,),.5)
    vel=ode_velocity(m,z,t,.04,moment)
    assert vel.shape==z.shape and torch.isfinite(vel).all()
    for method in ('euler','heun'):
        for direction in ('AtoB','BtoA'):
            y=solve_pfode(m,z.clone(),.04,moment,4,direction,method=method)
            assert y.shape==z.shape and torch.isfinite(y).all()
    print('PASS deterministic solver smoke')


def test_analytic_gaussian_transport():
    # For independent Gaussian endpoint marginals the optimal two drift
    # conditional expectations are exactly affine. The zero-network part of
    # the correct EDM residual representation therefore gives an oracle
    # probability-flow velocity, which MUST map one Gaussian to the other.
    class Zero(torch.nn.Module):
        def forward(self,x,t,direction):return torch.zeros_like(x)
    torch.manual_seed(8)
    m0,s0,m1,s1=.4,.7,1.8,1.3
    mom=PairMoments(torch.tensor([m0]),torch.tensor([m1]),
                    torch.tensor([s0*s0]),torch.tensor([s1*s1]),torch.tensor([0.]))
    z=m0+s0*torch.randn(10000,1,1,1)
    for dr,inp,mean,std in [
        ('AtoB',z,m1,s1),('BtoA',m1+s1*torch.randn_like(z),m0,s0)]:
        out=solve_pfode(Zero(),inp,.04,mom,32,direction=dr,t_floor=1e-6,method='heun')
        assert abs(float(out.mean())-mean)<.035,(dr,float(out.mean()),mean)
        assert abs(float(out.std())-std)<.035,(dr,float(out.std()),std)
    print('PASS analytic Gaussian bidirectional marginal transport')


if __name__=='__main__':
    test_solver_smoke()
    test_analytic_gaussian_transport()
