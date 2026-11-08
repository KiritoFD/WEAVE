"""EDM-like drift preconditioning derived from first/second moments.

Works for non-independent OT paired endpoints. Means, channel variances and
cross covariance are estimated for the *actual* endpoint coupling. All
statistics refer to globally and identically scaled source/target latents.
"""
from dataclasses import dataclass
import torch


@dataclass
class PairMoments:
    mu0: torch.Tensor
    mu1: torch.Tensor
    var0: torch.Tensor
    var1: torch.Tensor
    cov01: torch.Tensor

    def reversed(self):
        return PairMoments(self.mu1, self.mu0, self.var1, self.var0, self.cov01)

    def state_dict(self):
        return {k: getattr(self, k).detach().cpu() for k in ('mu0','mu1','var0','var1','cov01')}

    @classmethod
    def from_state_dict(cls, d):
        return cls(*(d[k] for k in ('mu0','mu1','var0','var1','cov01')))


class MomentAccumulator:
    def __init__(self):
        self.count=0
        self.a=self.b=self.aa=self.bb=self.ab=None

    @torch.no_grad()
    def update(self,x0,x1):
        if x0.shape != x1.shape or x0.ndim != 4:
            raise ValueError('Require equal B,C,H,W endpoints')
        x0=x0.detach().double();x1=x1.detach().double()
        axes=(0,2,3);n=x0.shape[0]*x0.shape[2]*x0.shape[3]
        parts=(x0.sum(axes),x1.sum(axes),(x0*x0).sum(axes),
               (x1*x1).sum(axes),(x0*x1).sum(axes))
        if self.a is None:
            self.a,self.b,self.aa,self.bb,self.ab=[v.cpu() for v in parts]
        else:
            for attr,p in zip(('a','b','aa','bb','ab'),parts):
                setattr(self,attr,getattr(self,attr)+p.cpu())
        self.count+=n

    def finalize(self):
        if self.count == 0: raise ValueError('No samples')
        a=self.a/self.count;b=self.b/self.count
        va=self.aa/self.count-a*a;vb=self.bb/self.count-b*b
        co=self.ab/self.count-a*b
        return PairMoments(a.float(),b.float(),va.float().clamp_min(1e-10),
                           vb.float().clamp_min(1e-10),co.float())


def independent_unit_coefficients(t, epsilon):
    """Correct scalar closed forms; appendix J simplified cskip, cout are erroneous."""
    if epsilon<=0: raise ValueError('epsilon must be positive')
    v=(1-t)**2+t*t+epsilon*t*(1-t)
    ci=torch.rsqrt(v)
    cs=((2-epsilon)*t-1)/v
    co2=((1-t)+epsilon*t)/((1-t)*v)
    return ci,cs,torch.sqrt(co2)


def moment_coefficients(t, epsilon, moments:PairMoments, tiny=1e-8):
    """Coefficients for locally forward drift, including nonzero endpoint means.

    t: (B,) physical time within the local direction, B,C,H,W latents.
    Returns (mu_x, mu_y, cin, cskip, cout) each B,C,1,1.
    """
    if epsilon<=0: raise ValueError('epsilon must be positive')
    if (t<=0).any() or (t>=1).any():raise ValueError('t must be in (0,1)')
    dev=t.device;dt=t.dtype
    def ch(v):return v.to(device=dev,dtype=dt).view(1,-1,1,1)
    a=(1-t).view(-1,1,1,1);b=t.view(-1,1,1,1)
    mu0=ch(moments.mu0);mu1=ch(moments.mu1)
    v0=ch(moments.var0);v1=ch(moments.var1);cov=ch(moments.cov01)
    mx=a*mu0+b*mu1
    my=mu1-mu0
    vx=(a*a*v0+b*b*v1+2*a*b*cov+epsilon*a*b).clamp_min(tiny)
    var_d=(v0+v1-2*cov).clamp_min(0)
    cyx=a*(cov-v0)+b*(v1-cov)-epsilon*b
    vy=var_d+epsilon*b/a
    ci=vx.rsqrt();cs=cyx/vx
    resvar=(vy-cyx*cyx/vx).clamp_min(tiny)
    co=resvar.sqrt()
    return mx,my,ci,cs,co


def precondition_input(x,t,epsilon,moments):
    mx,my,ci,cs,co=moment_coefficients(t,epsilon,moments)
    return (x-mx)*ci,(mx,my,ci,cs,co)


def reconstruct_drift(x, raw, coeffs):
    mx,my,ci,cs,co=coeffs
    return my+cs*(x-mx)+co*raw


def normalized_target(x,y,coeffs):
    mx,my,ci,cs,co=coeffs
    return (y-my-cs*(x-mx))/co
