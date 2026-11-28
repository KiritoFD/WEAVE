"""High-value guards for inherited coordinates, guidance and statistical acceptance."""
from types import SimpleNamespace

import torch
import torch.nn as nn

from exp.deploy_v35.a0_prune import build_encoder, prune_encoder
from exp.deploy_v35.distill.cfg import GuidedPCore, integrate
from exp.deploy_v35.eval_gate import paired_cluster_bootstrap
from exp.v29_biglittle.pcore_model import PCoreSLFUNet
from src.utils.runtime import save_json


def test_generation_reads_unicode_protocol_with_explicit_utf8(tmp_path, monkeypatch):
    from pathlib import Path
    from exp.deploy_v35.distill import acceptance
    rel = 'data/test/Impressionism/joaqu\u00edn-sorolla.jpg'
    protocol = tmp_path/'eval_protocol/tasks_750.json'
    save_json(protocol, dict(sources=[dict(source_rel_path=rel)], tasks=[]))
    save_json(tmp_path/'bundle/manifest.json', {})
    original = Path.read_text

    def gbk_default(path, encoding=None, errors=None):
        return original(path, encoding=encoding or 'gbk', errors=errors)

    monkeypatch.setattr(Path, 'read_text', gbk_default)
    monkeypatch.setattr(acceptance, 'REPO', tmp_path)
    monkeypatch.setattr(acceptance, 'candidate', lambda *args: object())
    loaded, generated = [], []
    monkeypatch.setattr(acceptance, 'load_rgb01_from_rel', lambda value: loaded.append(value))
    monkeypatch.setattr(acceptance, 'generate_test750_arm', lambda *args, **kw: generated.append(kw['src_cpu_map']))
    acceptance.generate(dict(out=str(tmp_path/'run'), bundle=str(tmp_path/'bundle'), eval_batch=4), 'trt', 'e')
    assert loaded == [rel]
    assert generated and all(list(src) == [rel] for src in generated)


def test_queue_liveness_check_does_not_terminate_worker():
    import subprocess
    import sys
    from exp.deploy_v35.distill.run import process_alive
    child = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read()'], stdin=subprocess.PIPE)
    try:
        assert process_alive(child.pid)
        assert child.poll() is None
        child.communicate(timeout=10)
        assert not process_alive(child.pid)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def test_full_width_prune_preserves_latent_coordinates():
    torch.set_num_threads(2)
    enc=build_encoder([128,256,512,512]).eval()
    qc=nn.Conv2d(8,8,1)
    original=SimpleNamespace(encoder=enc,quant_conv=qc)
    student,_=prune_encoder(original,[128,256,512,512])
    x=torch.rand(1,3,64,64)
    with torch.no_grad():
        expected=qc(enc(x*2-1))[:,:4]*.18215
        got=student(x)
    torch.testing.assert_close(got,expected,atol=1e-6,rtol=1e-5)


def test_guided_student_initializes_as_full_conditional_teacher_and_has_gradients():
    torch.set_num_threads(2)
    m=PCoreSLFUNet(base_dim=16,u_dim=417,cond_dim=32,use_gamma=True,normalize_u=True).eval()
    nn.init.normal_(m.out_conv.weight,std=.02)
    guided=GuidedPCore(m).eval()
    z,y,q=torch.randn(1,4,32,32),torch.randn(1,1,8,8),torch.randn(1,32,8,8)
    tid,u,t=torch.tensor([2]),torch.randn(1,417),torch.tensor([.7])
    control=torch.tensor([[1.,1.,0.,0.,.4,.8]])
    ref=m(z,y,q,tid,u,t,gamma=control[:,4],layout_scale=control[:,5])
    got=guided(z,y,q,tid,u,t,control)
    torch.testing.assert_close(got,ref,atol=1e-6,rtol=1e-5)
    (got-ref.detach()+.1).square().mean().backward()
    assert guided.guidance.weight.grad is not None
    assert torch.isfinite(guided.guidance.weight.grad).all()
    assert guided.guidance.weight.grad.abs().max()>0


def test_actual_nfe_and_endpoint_for_constant_velocity():
    calls=[]
    def velocity(z,t,control):
        calls.append((t.clone(),control.clone()))
        return torch.ones_like(z)
    eps=torch.zeros(1,4,8,8)
    ze=torch.ones_like(eps)*2
    out=integrate(velocity,ze,eps,tau0=.6,ws=3.,wc=2.,wsp=1.,rescale_phi=.3,gamma=.7,lscale=.8)
    torch.testing.assert_close(out,torch.ones_like(out)*1.6)
    assert len(calls)==7
    assert calls[0][0].item()<calls[1][0].item()<calls[2][0].item()


def test_point_estimate_cannot_replace_noninferiority_ci(tmp_path):
    ref,cand=[],[]
    for i in range(150):
        for t in range(5):
            row=dict(task_id=f'{i}/{t}',src_stem=f'{i}',tgt_style_id=t,is_cross=t!=i%5,
                     target_top1=1,dino_s=.5,dino_c=.6,lpips=.5,clip_scores_5=[.7]*5)
            ref.append(row)
            cand.append(dict(row,dino_c=.6+(.1 if i<75 else -.1)))
    save_json(tmp_path/'ref.json',dict(rows=ref))
    save_json(tmp_path/'cand.json',dict(rows=cand))
    result=paired_cluster_bootstrap(tmp_path/'cand.json',tmp_path/'ref.json')
    assert result['point_checks']['dino_c']
    assert not result['checks']['dino_c']
    assert result['status']=='inconclusive'
