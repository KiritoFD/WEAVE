import sys
sys.path.insert(0, ".")
import torch
from sbflow.network import BidirDiT

x = torch.randn(4, 32, 16, 16)
t = torch.rand(4)
d = torch.tensor([1, 0, 1, 0])
sa = torch.tensor([0, 1, 2, 3])
sb = torch.tensor([1, 4, 3, 4])

cases = [
    dict(style_cond="decomposed"),
    dict(style_cond="pair"),
    dict(style_cond="decomposed", heads_out=2),
    dict(style_cond="decomposed", local_conv=True, pair_rank=8),
    dict(style_cond="none"),
]
for kw in cases:
    m = BidirDiT(channels=32, image_size=16, patch=2, width=256, depth=6, heads=4,
                 num_styles=5, **kw)
    n = sum(p.numel() for p in m.parameters())
    o = m(x, t, d) if kw.get("style_cond") == "none" else m(x, t, d, sa, sb)
    print(f"{str(kw):62s} params={n:9d} out={tuple(o.shape)} finite={bool(torch.isfinite(o).all())}")

m = BidirDiT(channels=32, image_size=16, patch=2, width=256, depth=6, heads=4,
             num_styles=5, style_cond="decomposed")
ids = sorted({int(m._pair_id(torch.tensor([a]), torch.tensor([b]))[0])
              for a in range(5) for b in range(5) if a < b})
print("pair ids all:", ids)
print("pair id (1,4) vs (4,1):",
      int(m._pair_id(torch.tensor([1]), torch.tensor([4]))[0]),
      int(m._pair_id(torch.tensor([4]), torch.tensor([1]))[0]))

# zero-init behaviour: style conditioning must start inert
m.eval()
o1 = m(x, t, d, torch.tensor([0, 0, 0, 0]), torch.tensor([1, 1, 1, 1]))
o2 = m(x, t, d, torch.tensor([2, 2, 2, 2]), torch.tensor([4, 4, 4, 4]))
print("style-conditioned output identical at init (style_mix zero):", torch.allclose(o1, o2))
