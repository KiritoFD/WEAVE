"""Resume contract: 20 steps == 10 + resume + 10 (CPU, strict equality)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def tiny_train(net, opt, gen, n, out=None):
    losses = []
    for step in range(n):
        opt.zero_grad(set_to_none=True)
        z = torch.randn(2, 4, 16, 16, generator=gen)
        t = torch.randint(5, (2,), generator=gen)
        tau = torch.rand(2, generator=gen)
        loss = (net(z, t, tau) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    return losses


def test_resume_equivalence(tmp_path):
    from exp.v17.fm_residual import VelocityResidual
    from exp.v17.checkpoint import save_ckpt, load_ckpt

    def run(resume_at):
        torch.manual_seed(11)
        net = VelocityResidual(width=16)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
        gen = torch.Generator().manual_seed(42)
        losses = []
        if resume_at is not None:
            # first segment
            losses += tiny_train(net, opt, gen, resume_at)
            save_ckpt(tmp_path / 'ck.pt', model=net, optimizer=opt,
                      step=resume_at, gen_state=gen.get_state())
            net2 = VelocityResidual(width=16)
            opt2 = torch.optim.AdamW(net2.parameters(), lr=1e-3)
            ck = load_ckpt(tmp_path / 'ck.pt', model=net2, optimizer=opt2)
            losses += tiny_train(net2, opt2, gen, 20 - resume_at)
            return losses
        losses += tiny_train(net, opt, gen, 20)
        return losses

    full = run(None)
    split = run(10)
    assert full == split, (full[:3], split[:3])
