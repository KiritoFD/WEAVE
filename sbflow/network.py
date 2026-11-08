"""Shared, direction-conditioned latent DiT. No style encoder/classifier/loss."""
import math
import torch
from torch import nn
import torch.nn.functional as F


def time_embed(t,dim):
    half=dim//2
    freq=torch.exp(-math.log(10000)*torch.arange(half,device=t.device,dtype=torch.float32)/max(half-1,1))
    u=2*math.pi*t.float().reshape(-1,1)*freq.reshape(1,-1)
    x=torch.cat([u.sin(),u.cos()],dim=-1)
    return F.pad(x,(0,dim-x.shape[1]))


def modulate(x,shift,scale):
    return x*(1+scale.unsqueeze(1))+shift.unsqueeze(1)


class AdaBlock(nn.Module):
    def __init__(self,width,heads):
        super().__init__()
        self.norm1=nn.LayerNorm(width,elementwise_affine=False,eps=1e-6)
        self.norm2=nn.LayerNorm(width,elementwise_affine=False,eps=1e-6)
        self.attn=nn.MultiheadAttention(width,heads,batch_first=True)
        self.mlp=nn.Sequential(nn.Linear(width,4*width),nn.GELU(),nn.Linear(4*width,width))
        self.ada=nn.Sequential(nn.SiLU(),nn.Linear(width,6*width))
        nn.init.zeros_(self.ada[1].weight);nn.init.zeros_(self.ada[1].bias)

    def forward(self,x,c):
        sa,sca,ga,sm,scm,gm=self.ada(c).chunk(6,dim=-1)
        z=modulate(self.norm1(x),sa,sca)
        z=self.attn(z,z,z,need_weights=False)[0]
        x=x+ga.unsqueeze(1)*z
        x=x+gm.unsqueeze(1)*self.mlp(modulate(self.norm2(x),sm,scm))
        return x


class BidirDiT(nn.Module):
    def __init__(self,channels=32,image_size=16,patch=2,width=384,depth=12,heads=6):
        super().__init__()
        if image_size%patch!=0:raise ValueError('image size must divide patch')
        self.channels=channels;self.image_size=image_size;self.patch=patch
        self.nt=(image_size//patch)**2
        self.proj=nn.Conv2d(channels,width,kernel_size=patch,stride=patch)
        self.pos=nn.Parameter(torch.zeros(1,self.nt,width))
        nn.init.normal_(self.pos,std=.02)
        self.time=nn.Sequential(nn.Linear(width,width),nn.SiLU(),nn.Linear(width,width))
        self.dir=nn.Embedding(2,width)
        self.blocks=nn.ModuleList([AdaBlock(width,heads) for _ in range(depth)])
        self.fin_norm=nn.LayerNorm(width,elementwise_affine=False,eps=1e-6)
        self.fin_ada=nn.Sequential(nn.SiLU(),nn.Linear(width,2*width))
        self.fin_out=nn.Linear(width,channels*patch*patch)
        nn.init.zeros_(self.fin_out.weight);nn.init.zeros_(self.fin_out.bias)

    def forward(self,x,t,direction):
        B,C,H,W=x.shape
        if C!=self.channels or H!=self.image_size or W!=self.image_size:
            raise ValueError(f'Expected B,{self.channels},{self.image_size},{self.image_size}, got {tuple(x.shape)}')
        t=t.flatten(); direction=direction.flatten().long()
        c=self.time(time_embed(t,self.pos.shape[-1]))+self.dir(direction)
        h=self.proj(x).flatten(2).transpose(1,2)+self.pos
        for block in self.blocks: h=block(h,c)
        shift,scale=self.fin_ada(c).chunk(2,dim=-1)
        out=self.fin_out(modulate(self.fin_norm(h),shift,scale))
        p=self.patch;side=self.image_size//p
        return out.reshape(B,side,side,p,p,C).permute(0,5,1,3,2,4).reshape(B,C,H,W)
