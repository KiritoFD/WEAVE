import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import results_root

d = json.loads(results_root("cross_protocol", "all_summary.json").read_text(encoding="utf-8"))
order = [
    "identity", "tgt", "sdturbo", "stylealigned", "zstar", "styleshot", "styleid",
    "cut", "samst", "samam", "stytr2", "aespa", "seedream", "wct", "weave_oriented_e4", "weave",
]
for board in ["D5-512", "P2A-256"]:
    print("===", board)
    for k in order:
        if k not in d[board]:
            continue
        a = d[board][k]

        def f(x):
            v = a.get(x)
            return f"{v:.3f}" if isinstance(v, float) else "—"

        print(
            f"{a['display']:22s} "
            f"DS {f('cross_dino_style')} CS {f('cross_clip_style')} "
            f"LP {f('cross_content_lpips')} DC {f('cross_dino_content')}  "
            f"|| all DS {f('all_dino_style')} CS {f('all_clip_style')} "
            f"LP {f('all_content_lpips')} DC {f('all_dino_content')}"
        )
