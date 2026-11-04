import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import audit_core as C
from b_grad import ref_pools, offline_metrics, perturb


def per_image_dino_s(img, pools):
    q = C.quantize01(img)
    gd = C.dino_cls(q)
    return torch.stack([(gd @ pools["dino_cls"][s].T).max(1).values for s in range(5)], 1)


ds = C.load_dataset()
split = C.make_split(1000, n_ref=800, n_query=200, seed=0)
z_q = torch.stack([ds.latents[s][split.query[s]] for s in range(5)]).to("cuda")
pools = ref_pools("cuda")
zc = z_q[:, :12].reshape(-1, 4, 64, 64)
src = torch.arange(5).repeat_interleave(12).to(zc.device)
img = C.decode(zc).to(zc.device)
d = per_image_dino_s(img, pools)
own = d.gather(1, src.cpu().view(-1, 1)).squeeze(1)
print("own per image", [round(float(x), 4) for x in own[:12]])
print("own mean by style", [round(float(own[s * 12:(s + 1) * 12].mean()), 4) for s in range(5)])
m = offline_metrics(img, img, pools, 1)
print("offline_metrics identity", {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()})
print("pool sizes", {k: tuple(v.shape) for k, v in pools["dino_cls"].items()})
print("img range", float(img.min()), float(img.max()))
