"""Process-isolated parity, test750 quality gates and real RGB-to-RGB timings."""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import torch

from exp.deploy_v35 import E_GATE_KNOBS, P_BENCH_KNOBS, OPT_OUT_DIR
from exp.deploy_v35.bundle import file_sha256
from exp.deploy_v35.distill import encoder, cfg
from exp.deploy_v35.distill.data import write_json
from exp.deploy_v35.eval_gate import generate_test750_arm, paired_cluster_bootstrap
from exp.deploy_v35.parity import load_rgb01_from_rel, select_stratified_tasks, make_deterministic_eps
from exp.deploy_v35.pipeline import make_optimized_pipeline
from src.eval.runner import run_measure_batch_subprocess
from src.utils.runtime import REPO, read_json


def install_trt(pipe, batch: int, *, encoder_enabled=True) -> None:
    from exp.deploy_v35.trt_runtime import TrtRunner
    root = OPT_OUT_DIR/'trt_engines'
    a0 = TrtRunner(root/f'a0_b{batch}_fp16.plan',['rgb'],['z_s']) if encoder_enabled else None
    dec = TrtRunner(root/f's10_b{batch}_fp16.plan',['zo'],['rgb_out'])
    def padded(runner, x):
        n = len(x)
        if n > batch:
            raise ValueError('TRT batch exceeds fixed engine shape')
        xx = torch.cat([x,x[:1].expand(batch-n,*x.shape[1:])]) if n < batch else x
        return runner(xx.float())[:n]
    if a0 is not None:
        pipe.fast_a0 = pipe.a0 = None
        def encode(x):
            pipe.stats.a0_calls += 1
            return padded(a0,x)
        pipe._encode_a0 = encode
    pipe.fast_s10 = pipe.s10 = None
    def decode(z):
        pipe.stats.s10_calls += 1
        # Exported FastS10B includes scaling and one checkpoint PQC already.
        return padded(dec,z)
    pipe._decode_s10 = decode


def selected_encoder(config: dict) -> Path:
    selection = json.loads((Path(config['out'])/'selection.json').read_text())
    return Path(config['out'])/selection['encoder']/'best.pt'


def candidate(config: dict, profile: str, mode: str, batch: int, compile_mode='reduce-overhead'):
    uses_cfg = profile in ('cfg','combined') and mode=='p'
    pipe = make_optimized_pipeline(Path(config['bundle']),load_pcore=(mode=='p' and not uses_cfg),
                                   compile_mode=compile_mode,compile_batches=(batch,))
    if profile in ('encoder','combined'):
        encoder.install(pipe,selected_encoder(config))
    if uses_cfg:
        cfg.install(pipe,Path(config['out'])/'cfg'/'best.pt')
    if profile in ('trt','combined'):
        install_trt(pipe,batch,encoder_enabled=(profile=='trt'))
    return pipe


@torch.inference_mode()
def select_encoder(config: dict) -> None:
    out=Path(config['out'])
    reports={k:json.loads((out/k/'complete.json').read_text()) for k in ('enc20','enc10')}
    chosen=min(reports,key=lambda k:reports[k]['best_score'])
    write_json(out/'selection.json',dict(encoder=chosen,checkpoint_sha256=reports[chosen]['checkpoint_sha256'],
               rule='minimum held-out train-side latent-relative-RMSE + S10 MAE/255; no test750 selection',
               reports=reports))


@torch.inference_mode()
def smoke(config: dict) -> None:
    out=Path(config['out'])
    tasks=select_stratified_tasks(2)
    rgb=torch.stack([load_rgb01_from_rel(t['source_rel_path']) for t in tasks]).cuda()
    tid=torch.tensor([t['target_style_id'] for t in tasks],device='cuda')
    eps=torch.stack([make_deterministic_eps(t['task_id']) for t in tasks]).cuda()
    # Exercise both learned adapters, actual seven-call solver and endpoint identity.
    pipe=candidate(config,'encoder','e',2,compile_mode='')
    assert pipe.pcore is None and pipe.fast_pcore is None
    y=pipe.run_e_only(rgb,tid,.6,eps)
    assert torch.isfinite(y).all() and y.shape==rgb.shape
    assert torch.equal(pipe.run_e_only(rgb,tid,0.,eps),rgb)
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    pipe=candidate(config,'cfg','p',2,compile_mode='')
    pipe.stats.reset()
    y=pipe.run_e_to_p(rgb,tid,1.,eps)
    stats=pipe.stats.as_dict()
    assert stats['pcore_batch_calls']==7 and stats['pcore_cond_evals']==7, stats
    assert torch.isfinite(y).all() and y.shape==rgb.shape
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    pipe=candidate(config,'trt','e',4,compile_mode='')
    yt=pipe.run_e_only(rgb,tid,.6,eps)  # two-image tail padded into the fixed B4 engines
    assert torch.isfinite(yt).all() and yt.shape==rgb.shape
    write_json(out/'smoke.json',dict(passed=True,stats=stats,rgb_mean=y.mean().item(),
                                    trt_tail_shape=list(yt.shape),
                                    caveat='Two optimizer steps and two images verify wiring only, not quality'))


def arms(profile: str):
    if profile not in ('cfg','combined'):
        for knob in E_GATE_KNOBS:
            yield 'e',knob,f'e_l{round(knob*100):03d}'
    for knob in P_BENCH_KNOBS:
        yield 'p',knob,f'p_s{round(knob*100):03d}'


@torch.inference_mode()
def generate(config: dict, profile: str, mode: str) -> None:
    out=Path(config['out'])/'p4'
    batch=config['eval_batch']
    contract=dict(bundle_sha256=file_sha256(Path(config['bundle'])/'manifest.json'), profile=profile,
                  protocol_sha256=file_sha256(REPO/'eval_protocol/tasks_750.json'))
    if profile in ('encoder','combined'):
        contract['encoder_sha256']=file_sha256(selected_encoder(config))
    if profile in ('cfg','combined'):
        contract['cfg_sha256']=file_sha256(Path(config['out'])/'cfg'/'best.pt')
    contract_path=out/'contracts'/f'{profile}.json'
    if contract_path.exists() and json.loads(contract_path.read_text())!=contract:
        raise ValueError('Candidate changed after test750 was frozen; use a new run directory')
    write_json(contract_path,contract)
    requested=[(k,tag) for m,k,tag in arms(profile) if m==mode]
    if not requested:
        return
    pipe=candidate(config,profile,mode,batch)
    tj=read_json(REPO/'eval_protocol/tasks_750.json')
    src={r['source_rel_path']:load_rgb01_from_rel(r['source_rel_path']) for r in tj['sources']}
    for knob,tag in requested:
        name=f'{profile}_{tag}'
        generate_test750_arm(pipe,name,out/'raw',mode='e_only' if mode=='e' else 'e_to_p',
                             knob=knob,batch_size=batch,src_cpu_map=src)
    if mode=='p':
        generate_test750_arm(pipe,f'{profile}_p_s100_wrong',out/'raw',mode='e_to_p',knob=1.,
                             batch_size=batch,src_cpu_map=src,wrong_target=True)


def measure(config: dict, profile: str) -> None:
    """Called in a clean worker after every generator process has exited."""
    out=Path(config['out'])/'p4'
    legacy=OPT_OUT_DIR/'test750_gate'
    items=[]
    for _,_,tag in arms(profile):
        name=f'{profile}_{tag}'
        items.append((out/'raw'/name,out/'measured'/name,name))
        ref=f'optimized_{tag}'
        ref_eval=legacy/'measured'/ref/'eval.json'
        if not ref_eval.exists():
            raise FileNotFoundError(f'Missing frozen optimized baseline: {ref_eval}')
        base=json.loads(ref_eval.read_text())
        if (base['n_total'],base['n_cross'],base['n_same'])!=(750,600,150):
            raise ValueError('Baseline protocol counts mismatch')
    wrong=f'{profile}_p_s100_wrong'
    items.append((out/'raw'/wrong,out/'measured'/wrong,wrong))
    summaries=run_measure_batch_subprocess(items,write_png=False,micro_batch=config['referee_batch'])
    comparisons={}
    for _,_,tag in arms(profile):
        comparisons[tag]=paired_cluster_bootstrap(out/'measured'/f'{profile}_{tag}'/'eval.json',
                                                 legacy/'measured'/f'optimized_{tag}'/'eval.json')
    control=summaries.pop(wrong)
    write_json(out/f'{profile}_quality.json',dict(summaries=summaries,comparisons=comparisons,wrong_target_control=control,
                 all_knobs_pass=all(x['non_inferiority_pass'] for x in comparisons.values()),
                 baseline='frozen optimized test750: same seed, weights, schedule and source preprocessing',
                 eval_protocol_sha256=file_sha256(REPO/'eval_protocol/tasks_750.json')))


@torch.inference_mode()
def benchmark(config: dict, profile: str, mode: str, batch: int) -> None:
    out=Path(config['out'])/'p4'
    pipe=candidate(config,profile,mode,batch)
    tasks=select_stratified_tasks(max(batch,4))[:batch]
    cpu=torch.stack([load_rgb01_from_rel(t['source_rel_path']) for t in tasks]).pin_memory()
    output=torch.empty_like(cpu,pin_memory=True)
    tid=torch.tensor([t['target_style_id'] for t in tasks],device='cuda')
    eps=torch.stack([make_deterministic_eps(t['task_id']) for t in tasks]).cuda()
    def step():
        rgb=cpu.cuda(non_blocking=True)
        y=pipe.run_e_only(rgb,tid,.6,eps) if mode=='e' else pipe.run_e_to_p(rgb,tid,1.,eps)
        output.copy_(y,non_blocking=True)
    for _ in range(10):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times=[]
    start=time.monotonic()
    while len(times)<30 or time.monotonic()-start<10.:
        t0=time.perf_counter()
        step()
        torch.cuda.synchronize()
        times.append((time.perf_counter()-t0)*1000)
    times.sort()
    pipe.stats.reset()
    step()
    torch.cuda.synchronize()
    med=times[len(times)//2]
    write_json(out/'speed'/f'{profile}_{mode}_b{batch}.json',dict(profile=profile,mode=mode,batch=batch,
         p50_batch_ms=med,p95_batch_ms=times[int(.95*(len(times)-1))],ms_img=med/batch,ips=1000*batch/med,
         samples=len(times),peak_allocated_gb=torch.cuda.max_memory_allocated()/2**30,
         stats=pipe.stats.as_dict(),scope='warm resident host float32 RGB to host float32 RGB, H2D+D2H included; file IO excluded',
         hardware=torch.cuda.get_device_name()))


def finalize(config: dict) -> None:
    root=Path(config['out'])
    p4=root/'p4'
    reports={p.stem:json.loads(p.read_text()) for p in p4.glob('*_quality.json')}
    speed={p.stem:json.loads(p.read_text()) for p in (p4/'speed').glob('*.json')}
    points=[]
    for report in reports.values():
        for name,summary in report['summaries'].items():
            c=summary['cross600']
            profile,mode,knob=name.split('_')
            timing=speed.get(f'{profile}_{mode}_b4') if knob in ('l060','s100') else None
            points.append(dict(arm=name,dc=c['dino_c'],ds=c['dino_s'],t1=c['csd_top1']*100.,
                               n=600,protocol='Cross600+Same150/S10B',group='internal_deployment',
                               profile=profile,knob=knob,hw='RTX 3060 B4',label=name,
                               ips=timing['ips'] if timing else None,
                               non_inferiority_pass=report['comparisons'][f'{mode}_{knob}']['non_inferiority_pass'],
                               same=summary['same150']['csd_top1']*100.,lpips=c['lpips']))
    decision={}
    for profile in ('trt','encoder','cfg','combined'):
        rep=reports.get(profile+'_quality')
        if rep is None:
            continue
        mode='p' if profile in ('cfg','combined') else 'e'
        base=speed.get(f'optimized_{mode}_b4')
        current=speed.get(f'{profile}_{mode}_b4')
        speedup=base['ms_img']/current['ms_img'] if base and current else None
        decision[profile]=dict(quality_pass=rep['all_knobs_pass'],speedup=speedup,
                  accepted=rep['all_knobs_pass'] and speedup is not None and speedup>=1.15)
    result=dict(quality=reports,speed=speed,decision=decision,points=points,
                selection=json.loads((root/'selection.json').read_text()),
                status='complete; only accepted profiles may be promoted')
    write_json(root/'complete.json',result)
    write_json(REPO/'docs/performance/data/3060_distillation_summary.json',result)
    lines=['# A0 / CFG 串行蒸馏验收','',
           '完整 test750，Cross600 + Same150；按来源聚类 bootstrap 95% CI 判非劣。', '',
           '| 后端 | 质量门控 | B4 实测加速 | 采用 |','|---|---|---:|---|']
    for name,d in decision.items():
        ratio=f"{d['speedup']:.2f}x" if d['speedup'] else '未测'
        lines.append(f"| {name} | {d['quality_pass']} | {ratio} | {d['accepted']} |")
    lines.extend(['','完整指标和逐档 CI 见性能数据 3060_distillation_summary.json。',
                  '未通过者保留为实验产物；本流程不覆盖原 A0/P-Core 权重。'])
    (root/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
