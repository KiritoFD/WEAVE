import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import audit_core as C
from b_grad import ref_pools, offline_metrics

ds = C.load_dataset()
split = C.make_split(1000, n_ref=800, n_query=200, seed=0)
z_q = torch.stack([ds.latents[s][split.query[s]] for s in range(5)]).to("cuda")
pools = ref_pools("cuda")
n = 8
z0 = z_q[:, :n].reshape(-1, 4, 64, 64)
img = C.decode(z0).to(z0.device)
col = {C.SHORT[s]: [] for s in range(5)}
for s in range(5):
    m = offline_metrics(img, img, pools, s)
    for k, v in m["dino_s_per_style"].items():
        col[k].append(v)
print("identity column means (40 mixed images):",
      {k: round(sum(v) / len(v), 4) for k, v in col.items()})
per = []
for s in range(5):
    zs = z0[s * n:(s + 1) * n]
    mi = C.decode(zs).to(zs.device)
    m = offline_metrics(mi, mi, pools, s)
    per.append((C.SHORT[s], round(m["dino_s_target"], 4)))
print("identity own-style sims:", per)
