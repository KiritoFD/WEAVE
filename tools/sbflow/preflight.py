"""Real DCAE roundtrip, cache reproducibility and frozen Test600 exclusion audit.

Run on the actual training GPU before the expensive model training. Unlike
synthetic unit tests, this checks the selected AE checkpoint and image data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from .cache import image_tensor,load_dcae
from .train import cache_read


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cache-a',required=True)
    ap.add_argument('--cache-b',required=True)
    ap.add_argument('--repo-root',default='')
    ap.add_argument('--n',type=int,default=4)
    args=ap.parse_args()
    aa=cache_read(args.cache_a);bb=cache_read(args.cache_b)
    assert aa['ae']==bb['ae'] and aa['scaling_factor']==bb['scaling_factor']
    assert aa['latent'].shape[1:]==bb['latent'].shape[1:]
    assert aa['image_size']==bb['image_size']
    print('DCAE cache shapes',aa['latent'].shape,bb['latent'].shape)
    paths=aa['ids']+bb['ids']
    if args.repo_root:
        p=Path(args.repo_root)/'eval_protocol/tasks_750.json'
        d=json.loads(p.read_text(encoding='utf-8'))
        test_sha={s.get('source_sha256') for s in d.get('sources',[]) if s.get('source_sha256')}
        test_path={(Path(args.repo_root)/s['source_rel_path']).resolve() for s in d.get('sources',[]) if 'source_rel_path' in s}
        overlaps=[x for x in paths if Path(x).resolve() in test_path or sha256(x) in test_sha]
        if overlaps:raise AssertionError(f'Test600 leakage: {overlaps[:5]}')
        print('frozen Test600 SHA/path audit PASS;',len(paths),'training images checked')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    ae=load_dcae(aa['ae'],device)
    sf=float(aa['scaling_factor'])
    for name,cache in (('A',aa),('B',bb)):
        picks=[0,len(cache['ids'])//2,len(cache['ids'])-1][:args.n]
        for i in picks:
            size=int(cache['image_size'])
            orig=image_tensor(cache['ids'][i],size)[None].to(device)*2-1
            with torch.inference_mode():
                with (torch.autocast('cuda',dtype=torch.bfloat16) if device=='cuda'
                      else __import__('contextlib').nullcontext()):
                    z=ae.encode(orig).latent.float()*sf
                    dec=ae.decode(cache['latent'][i:i+1].float().to(device)/sf).sample.float().clamp(-1,1)
            cache_error=float((z-cache['latent'][i:i+1].float().to(device)).square().mean())
            mse=float((dec-orig).square().mean())
            psnr=-10*__import__('math').log10(max(mse/4,1e-12))
            print(f'DCAE {name} idx={i} encoder_cache_MSE={cache_error:.7f} '
                  f'RGB_roundtrip_PSNR={psnr:.2f}dB')
            assert cache_error<.02,'Cached latents inconsistent with current AE/precision/preprocessing'
            assert torch.isfinite(dec).all()
    print('REAL DCAE preflight PASS (inspect reconstructions visually before proceeding)')

if __name__=='__main__':main()
