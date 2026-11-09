import pathlib

p = pathlib.Path("sbflow/eval_test600.py")
s = p.read_text(encoding="utf-8")

# --- generate: per-phase timing -------------------------------------------------
s = s.replace(
    """    manifest=[]
    with torch.inference_mode():
        for i in range(0,len(tasks),args.batch):
            group=tasks[i:i+args.batch]
            x=torch.stack([image_tensor(repo/t['source_rel_path'],size)*2-1 for t in group]).to(device)""",
    """    manifest=[]
    import time as _time
    _ph={'io':0.0,'encode':0.0,'ode':0.0,'decode':0.0,'save':0.0}
    _t0=_time.perf_counter()
    with torch.inference_mode():
        for i in range(0,len(tasks),args.batch):
            group=tasks[i:i+args.batch]
            _a=_time.perf_counter()
            x=torch.stack([image_tensor(repo/t['source_rel_path'],size)*2-1 for t in group]).to(device)
            torch.cuda.synchronize();_b=_time.perf_counter();_ph['io']+=_b-_a""",
)

s = s.replace(
    """                zae=ae.encode(x).latent.float()*sf
                z=(zae-mean[None,:,None,None])/std[None,:,None,None]""",
    """                zae=ae.encode(x).latent.float()*sf
                torch.cuda.synchronize();_c=_time.perf_counter();_ph['encode']+=_c-_b
                z=(zae-mean[None,:,None,None])/std[None,:,None,None]""",
)

s = s.replace(
    """                pred=ae.decode((zout*std[None,:,None,None]+mean[None,:,None,None])/sf).sample
                noop=ae.decode(zae/sf).sample""",
    """                torch.cuda.synchronize();_d=_time.perf_counter();_ph['ode']+=_d-_c
                pred=ae.decode((zout*std[None,:,None,None]+mean[None,:,None,None])/sf).sample
                noop=ae.decode(zae/sf).sample
                torch.cuda.synchronize();_e=_time.perf_counter();_ph['decode']+=_e-_d""",
)

s = s.replace(
    """            if i%max(args.batch*10,1)==0:
                print(f'generated {i+len(group)}/{len(tasks)}',flush=True)""",
    """            torch.cuda.synchronize();_ph['save']+=_time.perf_counter()-_e
            if i%max(args.batch*10,1)==0:
                print(f'generated {i+len(group)}/{len(tasks)}',flush=True)""",
)

s = s.replace(
    """    nfe=args.steps*(2 if args.method=='euler' else 4)
    m={'model':args.ckpt,""",
    """    _wall=_time.perf_counter()-_t0
    print('[eval-timing] generate total=%.1fs  io=%.1f encode=%.1f ode=%.1f decode=%.1f save=%.1f'
          % (_wall,_ph['io'],_ph['encode'],_ph['ode'],_ph['decode'],_ph['save']),flush=True)
    nfe=args.steps*(2 if args.method=='euler' else 4)
    m={'timing_generate_sec':{k:round(v,2) for k,v in _ph.items()},
       'timing_generate_total_sec':round(_wall,2),
       'model':args.ckpt,""",
)

p.write_text(s, encoding="utf-8")
print("patched eval generate timing")
