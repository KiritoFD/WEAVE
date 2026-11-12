import io

p = 'tools/v7r_next/audit.py'
s = io.open(p, encoding='utf-8').read()
anchor = '''def np_mean(x):
    import numpy as np
    return float(np.mean(x))
'''
new = '''def r1_margin_direction_check(results: dict) -> dict:
    """P0.1: does R1-fwd RAISE and R1-rev LOWER the R1 CLASSIFIER target margin?

    eval.json only stores CSD target_margin; the R1 gate must use the frozen
    critic directly. We reconstruct z0 from a small fixed source set, apply
    the same unit-RMS R1 gradient (fwd / rev / zero), and compare the critic's
    target margin. No decoding, no 750 eval -> seconds.
    """
    import torch
    import torch.nn.functional as F
    from tools.v7r.arms import make_style_objective
    from tools.v7r.v7r_common import rms_normalize
    from tools.v6.train_eval_v6 import STYLES
    dev = "cuda"
    critic = C.load_critic(dev)
    canon = C.load_canon(dev)
    b4 = C.load_c3unet(dev)
    # fixed small source set from the frozen TRAIN pool (not tasks_750)
    from tools.v7r.v7r_common import load_pool
    pool, _, _ = load_pool(device="cpu")
    C.set_seed()
    zs = torch.stack([pool[STYLES[s % 5]][s * 7] for s in range(10)]).to(dev)
    tid = torch.arange(10, device=dev) % 5
    with torch.no_grad():
        c = canon(zs.float(), mode="c1_gentle", augment=False)
        z0 = b4(c, 0.0, c, tid).detach()
    obj = make_style_objective("R1", {"critic": critic}, tid)

    def r1_margin(z):
        q = critic(z.float()).float()
        oh = F.one_hot(tid, 5).bool()
        m = q.gather(1, tid[:, None]).squeeze(1) - torch.logsumexp(
            q.masked_fill(oh, torch.finfo(q.dtype).min), dim=1)
        return float(m.mean())
    zr = z0.clone().requires_grad_(True)
    g = torch.autograd.grad(obj(zr), zr)[0]
    g = rms_normalize(g)
    rms = 0.25  # PROBE_RMS, matches the single-step probe budget
    m0 = r1_margin(z0)
    mf = r1_margin((z0 + rms * g).detach())
    mr = r1_margin((z0 - rms * g).detach())
    return {
        "r1_margin_B4": round(m0, 5),
        "r1_margin_fwd": round(mf, 5),
        "r1_margin_rev": round(mr, 5),
        "fwd_raises_r1_margin": bool(mf > m0),
        "rev_lowers_r1_margin": bool(mr < m0),
        "note": "R1 classifier target margin computed directly from frozen "
                "critic on 10 fixed TRAIN-pool latents; NOT the CSD margin "
                "stored in eval.json. Unit-RMS gradient, same probe budget.",
    }


'''
assert anchor in s
s = s.replace(anchor, new + anchor)
io.open(p, 'w', encoding='utf-8').write(s)
print('added r1_margin_direction_check')
