import pathlib

p = pathlib.Path("sbflow/eval_test600.py")
s = p.read_text(encoding="utf-8")

s = s.replace(
    """    scores=[]
    for start in range(0,len(tasks),args.batch):
        batch=tasks[start:start+args.batch]
        imgs={tag:[Image.open(t[tag]).convert('RGB') for t in batch]
              for tag in ('generated','dcae_noop')}""",
    """    scores=[]
    import time as _time
    _ph={'io':0.0,'feat':0.0,'src_feat':0.0,'lpips':0.0}
    _t0=_time.perf_counter()
    for start in range(0,len(tasks),args.batch):
        batch=tasks[start:start+args.batch]
        _a=_time.perf_counter()
        imgs={tag:[Image.open(t[tag]).convert('RGB') for t in batch]
              for tag in ('generated','dcae_noop')}""",
)

s = s.replace(
    """        feat={tag:evaluator.extract_image_features(pils) for tag,pils in imgs.items()}
        with torch.inference_mode():
            from_src=evaluator.extract_image_features(src)""",
    """        torch.cuda.synchronize();_b=_time.perf_counter();_ph['io']+=_b-_a
        feat={tag:evaluator.extract_image_features(pils) for tag,pils in imgs.items()}
        torch.cuda.synchronize();_c=_time.perf_counter();_ph['feat']+=_c-_b
        with torch.inference_mode():
            from_src=evaluator.extract_image_features(src)
            torch.cuda.synchronize();_ph['src_feat']+=_time.perf_counter()-_c""",
)

s = s.replace(
    """                    lps=float(evaluator.lpips(a[None],b[None]).mean().item())""",
    """                    _l0=_time.perf_counter()
                    lps=float(evaluator.lpips(a[None],b[None]).mean().item())
                    _ph['lpips']+=_time.perf_counter()-_l0""",
)

s = s.replace(
    """    if not scores:
        raise ValueError('No scores produced')""",
    """    _wall=_time.perf_counter()-_t0
    print('[eval-timing] score loop=%.1fs  io=%.1f feat=%.1f src_feat=%.1f lpips=%.1f'
          % (_wall,_ph['io'],_ph['feat'],_ph['src_feat'],_ph['lpips']),flush=True)
    if not scores:
        raise ValueError('No scores produced')""",
)

p.write_text(s, encoding="utf-8")
print("patched eval score timing")
