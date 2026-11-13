"""Probability flow ODE: 0.5*(forward drift - reverse-time drift).

Directional reverse A->B or B->A, with exact inverse latent standardization.
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from .cache import image_tensor,load_dcae
from .network import BidirDiT
from .preconditioning import PairMoments,precondition_input,reconstruct_drift


def ode_velocity(model,z,t_phys,epsilon,mom,direction='AtoB',t_floor=.002):
    """The probability flow ODE uses drifts at physically opposite times."""
    if direction not in ('AtoB','BtoA'):raise ValueError(direction)
    t_phys=t_phys.clamp(t_floor,1-t_floor)
    if direction=='AtoB':
        mp=mom;mm=mom.reversed();dp=1;dm=0
    else:
        mp=mom.reversed();mm=mom;dp=0;dm=1
    xp,cp=precondition_input(z,t_phys,epsilon,mp)
    xm,cm=precondition_input(z,1-t_phys,epsilon,mm)
    d1=torch.full_like(t_phys,dp,dtype=torch.long)
    d2=torch.full_like(t_phys,dm,dtype=torch.long)
    bp=reconstruct_drift(z,model(xp,t_phys,d1).float(),cp)
    bm=reconstruct_drift(z,model(xm,1-t_phys,d2).float(),cm)
    return (bp-bm)*.5


@torch.inference_mode()
def solve_pfode(model,z,epsilon,mom,n_steps=16,direction='AtoB',t_floor=.002,method='heun'):
    if n_steps<1:raise ValueError('Need >=1 steps')
    h=1/n_steps
    for i in range(n_steps):
        t0=torch.full((len(z),),i*h,device=z.device)
        k0=ode_velocity(model,z,t0,epsilon,mom,direction,t_floor)
        if method=='euler':z=z+h*k0
        elif method=='heun':
            t1=torch.full_like(t0,(i+1)*h)
            k1=ode_velocity(model,z+h*k0,t1,epsilon,mom,direction,t_floor)
            z=z+h*.5*(k0+k1)
        else:raise ValueError(method)
        if not torch.isfinite(z).all():
            raise FloatingPointError(f'Non-finite ODE state after step={i}; check drift and epsilon.')
    return z


def load_bundle(path,device,which='raw'):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    model=BidirDiT(**ck['model_kwargs']).to(device).eval()
    model.load_state_dict(ck['model'] if which=='raw' else ck['ema'])
    mom=PairMoments.from_state_dict(ck['moments'])
    mean=ck['latent_mean'].to(device)
    std=ck['latent_std'].to(device)
    return ck,model,mom,mean,std


def translate_tensor(ae,ck,model,mom,mean,std,x,steps,direction,method,t_floor):
    sf=float(ck['ae_scaling_factor'])
    z=(ae.encode(x).latent.float()*sf-mean[None,:,None,None])/std[None,:,None,None]
    z=solve_pfode(model,z,ck['config']['eps'],mom,steps,direction,t_floor,method)
    zx=(z*std[None,:,None,None]+mean[None,:,None,None])/sf
    im=ae.decode(zx).sample.float().clamp(-1,1)
    return im


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--ckpt',required=True)
    p.add_argument('--input',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=4)
    p.add_argument('--method',choices=['euler','heun'],default='heun')
    p.add_argument('--direction',choices=['AtoB','BtoA'],default='AtoB')
    p.add_argument('--weights',choices=['raw','ema'],default='raw')
    p.add_argument('--t-floor',type=float,default=.002)
    args=p.parse_args()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    ck,model,mom,mean,std=load_bundle(args.ckpt,device,args.weights)
    ae=load_dcae(ck['ae'],device)
    im=image_tensor(args.input,int(ck['config'].get('image_size',512))) if 'image_size' in ck['config'] else image_tensor(args.input,512)
    with torch.inference_mode():
        if device=='cuda':
            with torch.autocast('cuda',dtype=torch.bfloat16):
                out=translate_tensor(ae,ck,model,mom,mean,std,im[None].to(device)*2-1,
                                     args.steps,args.direction,args.method,args.t_floor)
        else:
            out=translate_tensor(ae,ck,model,mom,mean,std,im[None].to(device)*2-1,
                                 args.steps,args.direction,args.method,args.t_floor)
    arr=((out[0].permute(1,2,0).cpu().numpy()+1)*127.5).round().clip(0,255).astype('uint8')
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(arr).save(args.output)
    print('saved',args.output, 'network_NFE=',args.steps*(2 if args.method=='euler' else 4))

if __name__=='__main__':main()
