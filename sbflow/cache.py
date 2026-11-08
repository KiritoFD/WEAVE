"""Freeze DCAE image latents and pixel thumbnails used for the minibatch OT cost.

Usage: python -m sbflow.cache --images DATASET/Style --out runs/cache/Style.pt
Requires `diffusers` and DCAE weight download on the training machine.
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F


def load_dcae(model_name,device):
    from diffusers import AutoencoderDC
    ae=AutoencoderDC.from_pretrained(model_name).eval().to(device)
    for p in ae.parameters():p.requires_grad_(False)
    return ae


def image_tensor(p,size):
    with Image.open(p) as im:
        im=im.convert('RGB').resize((size,size),Image.Resampling.BICUBIC)
        t=torch.from_numpy(np.asarray(im, dtype=np.uint8).copy()).permute(2,0,1).float()/255.
    return t


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--images',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--ae',default='mit-han-lab/dc-ae-f32c32-mix-1.0-diffusers')
    ap.add_argument('--image-size',type=int,default=512)
    ap.add_argument('--batch',type=int,default=4)
    ap.add_argument('--device',default='cuda')
    ap.add_argument('--include-dino',action='store_true',help='Also cache real DINOv2 CLS features for optional semantic OT')
    ap.add_argument('--dino-model',default='facebook/dinov2-small')
    args=ap.parse_args()
    root=Path(args.images)
    files=sorted(p for p in root.rglob('*') if p.suffix.lower() in ('.png','.jpg','.jpeg','.webp'))
    if not files:raise ValueError(f'No images in {root}')
    ae=load_dcae(args.ae,args.device)
    latent_list=[]; thumb_list=[]; dino_list=[]
    if args.include_dino:
        from transformers import AutoImageProcessor,AutoModel
        dino_processor=AutoImageProcessor.from_pretrained(args.dino_model)
        dino_model=AutoModel.from_pretrained(args.dino_model).to(args.device).eval()
        for q in dino_model.parameters():q.requires_grad_(False)
    with torch.inference_mode():
        for start in range(0,len(files),args.batch):
            batch=torch.stack([image_tensor(p,args.image_size) for p in files[start:start+args.batch]])
            thumb=(F.interpolate(batch,size=(32,32),mode='area')*255).round().clamp(0,255).byte()
            x=(batch*2-1).to(args.device)
            if args.device.startswith('cuda'):
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    latent=ae.encode(x).latent
            else:latent=ae.encode(x).latent
            # Diffusers AutoencoderDC `.encode(...).latent`, NOT `.latent_dist.mean`.
            sf=float(ae.config.scaling_factor)
            latent_list.append((latent.float()*sf).half().cpu())
            thumb_list.append(thumb.cpu())
            if args.include_dino:
                with torch.inference_mode():
                    pils=[Image.open(p).convert('RGB') for p in files[start:start+args.batch]]
                    inp=dino_processor(images=pils,return_tensors='pt').to(args.device)
                    feats=dino_model(**inp).last_hidden_state[:,0].float()
                    feats=F.normalize(feats,dim=-1).half().cpu()
                    dino_list.append(feats)
                    for im in pils:im.close()
            if start%(args.batch*64)==0:print('cached',start,'/',len(files),flush=True)
    out={'latent':torch.cat(latent_list), 'thumb':torch.cat(thumb_list),
         'ids':[str(p.resolve()) for p in files], 'ae':args.ae,
         'scaling_factor':sf,'image_size':args.image_size,
         'pixel_input_range':'-1..1','thumbnail_range':'0..255'}
    if args.include_dino:
        out['dino']=torch.cat(dino_list)
        out['dino_model']=args.dino_model
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(out,path)
    print('saved',path,tuple(out['latent'].shape),flush=True)

if __name__=='__main__':main()
