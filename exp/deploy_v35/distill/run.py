"""Single-GPU serial queue. Every training/generation/referee runs in a fresh process.

Example: python -m exp.deploy_v35.distill.run --train-root DATA --bundle BUNDLE
Use --smoke in a separate output directory before the full queue.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path


MODULE='exp.deploy_v35.distill.run'


def write(path: Path,data: dict):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(data,indent=2),encoding='utf-8')
    tmp.replace(path)


def recheck_probes(config: dict):
    """Repair incomplete optional probes between GPU stages, in fresh processes."""
    root, reports = Path(config['out']), Path(config['opt_out'])
    for stage, name in [('p0', 'p0_trt_report.json'), ('taesd_recheck', 'p1_report.json')]:
        report = reports/name
        marker = root/'stages'/f'{stage}_recheck.json'
        if marker.exists() or not report.exists():
            continue
        data = json.loads(report.read_text())
        needed = (any('channels' in row.get('error', '') for row in data.get('ecore_trt', {}).values())
                  if stage == 'p0' else data.get('taesd_probe', {}).get('status') == 'unavailable')
        if not needed:
            continue
        backup = report.with_name(report.stem+'_before_recheck.json')
        if not backup.exists():
            backup.write_bytes(report.read_bytes())
        print(f'[probe recheck] {stage}', flush=True)
        with (root/'logs'/f'{stage}_recheck.log').open('ab', buffering=0) as log:
            subprocess.run([sys.executable, '-u', '-m', MODULE, '--config', str(root/'config.json'),
                            '--worker', stage], stdout=log, stderr=subprocess.STDOUT, check=True)
        write(marker, dict(status='complete', report=str(report), finished=time.time()))


def worker(config: dict,stage: str):
    # Referee and compiler subprocesses must inherit the resolved venv priority,
    # including optional dependencies appended for the service account.
    if os.environ.get('WEAVE_DEPENDENCY_SITE'):
        os.environ['PYTHONPATH']=os.pathsep.join(p for p in sys.path if p)
    root=Path(config['out'])
    if stage in ('cache_encoder', 'finalize') and not config['smoke']:
        recheck_probes(config)
    if stage=='p0':
        from exp.deploy_v35.p0_p1_run import run_p0_trt_and_int8
        from exp.deploy_v35 import OPT_OUT_DIR
        report=run_p0_trt_and_int8(OPT_OUT_DIR)
        write(OPT_OUT_DIR/'p0_trt_report.json',report)
    elif stage=='p1':
        from exp.deploy_v35.p0_p1_run import run_p1_taesd_and_prune_design
        from exp.deploy_v35 import OPT_OUT_DIR
        report=run_p1_taesd_and_prune_design(OPT_OUT_DIR)
        write(OPT_OUT_DIR/'p1_report.json',report)
    elif stage=='taesd_recheck':
        from exp.deploy_v35.p0_p1_run import _run_taesd_probe
        from exp.deploy_v35 import OPT_OUT_DIR
        path=OPT_OUT_DIR/'p1_report.json'
        report=json.loads(path.read_text())
        try:
            report.update(_run_taesd_probe(OPT_OUT_DIR))
        except OSError as exc:
            report['taesd_probe']=dict(status='unavailable', reason=str(exc))
        write(path, report)
    elif stage=='cache_encoder':
        from .data import cache_encoder
        cache_encoder(config,Path(config['cache']))
    elif stage in ('enc20','enc10'):
        from .encoder import train
        train(config,stage,root/stage)
    elif stage=='select':
        from .acceptance import select_encoder
        select_encoder(config)
    elif stage=='cache_cfg':
        from .cfg import cache_rollouts
        cache_rollouts(config,Path(config['cfg_cache']))
    elif stage=='cfg':
        from .cfg import train
        train(config,root/'cfg')
    elif stage=='smoke':
        from .acceptance import smoke
        smoke(config)
    elif stage.startswith('generate:'):
        from .acceptance import generate
        _,profile,mode=stage.split(':')
        generate(config,profile,mode)
    elif stage.startswith('measure:'):
        from .acceptance import measure
        measure(config,stage.split(':')[1])
    elif stage.startswith('bench:'):
        from .acceptance import benchmark
        _,profile,mode,batch=stage.split(':')
        benchmark(config,profile,mode,int(batch))
    elif stage=='finalize':
        from .acceptance import finalize
        finalize(config)
    elif stage=='plot':
        subprocess.run([sys.executable,'docs/performance/plot_cleave_pareto.py'],check=True)
    else:
        raise ValueError(stage)


def stages(config: dict):
    if not config['smoke']:
        yield 'p0'
        yield 'p1'
    yield from ('cache_encoder','enc20','enc10','select','cache_cfg','cfg')
    if config['smoke']:
        yield 'smoke'
        return
    for profile in ('optimized','trt','encoder','cfg','combined'):
        modes=('p',) if profile in ('cfg','combined') else ('e','p')
        for mode in modes:
            for batch in (1,4):
                yield f'bench:{profile}:{mode}:{batch}'
        if profile=='optimized':
            continue  # Frozen complete test750 baseline is already available.
        for mode in modes:
            yield f'generate:{profile}:{mode}'
        yield f'measure:{profile}'
    yield 'finalize'
    yield 'plot'


def process_alive(pid: int) -> bool:
    if os.name != 'nt':
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        return True
    # On Windows os.kill(pid, 0) TERMINATES the target. Query its handle instead.
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        if ctypes.get_last_error() == 87:  # ERROR_INVALID_PARAMETER: no such PID
            return False
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        return code.value == 259  # STILL_ACTIVE
    finally:
        kernel.CloseHandle(handle)


def serial(config: dict,path: Path):
    root=Path(config['out'])
    plan=list(stages(config))
    lock=root/'queue.lock'
    try:
        fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:
        old=int(lock.read_text())
        if not process_alive(old):
            lock.unlink()
            fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        else:
            raise RuntimeError(f'Serial queue already active: pid={old}')
    os.write(fd,str(os.getpid()).encode())
    os.close(fd)
    env=os.environ.copy()
    env.update(PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8',HF_HUB_DOWNLOAD_TIMEOUT='30',HF_HUB_ETAG_TIMEOUT='15',
               WEAVE_DISTILL_TRAIN_ROOT=config['train_root'])
    try:
        for index,stage in enumerate(plan):
            safe=stage.replace(':','_')
            marker=root/'stages'/f'{safe}.json'
            if marker.exists():
                continue
            if 'combined' in stage:
                quality=[root/'p4'/f'{p}_quality.json' for p in ('trt','encoder','cfg')]
                if not all(p.exists() and json.loads(p.read_text())['all_knobs_pass'] for p in quality):
                    write(marker,dict(stage=stage,status='skipped',reason='Components did not all pass full CI gates'))
                    continue
            # Reuse completed P0/P1 evidence only, never infer completion from engines.
            if stage in ('p0','p1') and (Path(config['opt_out'])/f'{stage}_trt_report.json' if stage=='p0' else Path(config['opt_out'])/'p1_report.json').exists():
                write(marker,dict(stage=stage,status='reused_complete_report'))
                continue
            logs=root/'logs'
            logs.mkdir(exist_ok=True)
            write(root/'status.json',dict(stage=stage,index=index+1,total=len(plan),status='running',pid=os.getpid(),started=time.time()))
            print(f'[{index+1}/{len(plan)}] {stage}',flush=True)
            t0=time.monotonic()
            with (logs/f'{safe}.log').open('ab',buffering=0) as log:
                proc=subprocess.Popen([sys.executable,'-u','-m',MODULE,'--config',str(path),'--worker',stage],stdout=log,stderr=subprocess.STDOUT,env=env)
                rc=proc.wait()
            if rc:
                write(root/'status.json',dict(stage=stage,status='failed',returncode=rc,log=str(logs/f'{safe}.log')))
                raise RuntimeError(f'Stage {stage} failed rc={rc}; queue stopped, see log')
            write(marker,dict(stage=stage,status='complete',seconds=time.monotonic()-t0))
        write(root/'status.json',dict(status='complete',total=len(plan),finished=time.time()))
    finally:
        lock.unlink(missing_ok=True)


def main():
    # Hugging Face captures the endpoint at import time, before probe functions.
    os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
    # Task Scheduler's service account has no user-site dependencies. Append the
    # configured directory after the venv, so it cannot shadow pinned packages.
    dependency_site=os.environ.get('WEAVE_DEPENDENCY_SITE')
    if dependency_site:
        import site
        site.addsitedir(dependency_site)
    # Triton may be inherited from system site-packages by a venv. Its default
    # compiler lookup uses the venv's platlib, where the bundled TinyCC is absent.
    if os.name == 'nt' and not os.environ.get('CC'):
        spec = importlib.util.find_spec('triton')
        if spec and spec.origin:
            bundled_cc = Path(spec.origin).parent/'runtime'/'tcc'/'tcc.exe'
            if bundled_cc.is_file():
                os.environ['CC'] = str(bundled_cc)
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path)
    ap.add_argument('--worker')
    ap.add_argument('--train-root',type=Path)
    ap.add_argument('--bundle',type=Path,default=Path('runs/deploy_bundle_v35'))
    ap.add_argument('--out',type=Path,default=Path('runs/v35_3060_opt/distillation'))
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--prepare',action='store_true',help='Write configuration without starting GPU work')
    args=ap.parse_args()
    if args.config:
        config=json.loads(args.config.read_text())
        path=args.config.resolve()
    else:
        if args.train_root is None:
            ap.error('--train-root is required')
        if not args.train_root.is_dir():
            raise FileNotFoundError(args.train_root)
        root=args.out.resolve()
        root.mkdir(parents=True,exist_ok=True)
        config=dict(train_root=str(args.train_root.resolve()),bundle=str(args.bundle.resolve()),out=str(root),
                    opt_out=str(root.parent),cache=str(root/'encoder_cache'),cfg_cache=str(root/'cfg_cache'),seed=20261127,
                    n_train=8192,n_calib=128,micro_batch=1,cfg_micro_batch=2,accum=4,
                    encoder_steps=6000,encoder_seconds=7200,cfg_sources=512,cfg_calib_sources=64,
                    cfg_steps=12000,cfg_seconds=14400,eval_batch=4,referee_batch=4,smoke=args.smoke)
        if args.smoke:
            config.update(n_train=16,n_calib=4,accum=1,cfg_micro_batch=1,encoder_steps=2,cfg_steps=2,
                          encoder_seconds=90,cfg_seconds=90,cfg_sources=2,cfg_calib_sources=2,calib_limit=2)
        path=root/'config.json'
        if path.exists() and json.loads(path.read_text())!=config:
            raise ValueError('Existing output uses another config; pass --config to resume')
        write(path,config)
    if args.prepare:
        print(path)
    elif args.worker:
        worker(config,args.worker)
    else:
        serial(config,path)


if __name__=='__main__':
    main()
