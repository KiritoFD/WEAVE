import io

p = 'tools/v6_next/vae_gap_probe.py'
s = io.open(p, encoding='utf-8').read()

impl = io.open('p1_impl.py', encoding='utf-8').read()
# extract from rank_direction to end
p1_code = impl[impl.index('def rank_direction'):]

old = '''def run_p1(args):
    raise SystemExit("P1 not implemented yet; run --mode p0 first")'''

assert old in s, "old run_p1 not found"
s = s.replace(old, p1_code)

# add arguments to main argparse in vae_gap_probe.py
old_main = '''    ap.add_argument("--mode", default="p0", choices=["p0", "p1"])
    ap.add_argument("--teacher", default="Oracle_1step")
    ap.add_argument("--student", default="B4_lambda50")
    ap.add_argument("--boot", type=int, default=4000)'''

new_main = '''    ap.add_argument("--mode", default="p0", choices=["p0", "p1"])
    ap.add_argument("--teacher", default="Oracle_1step")
    ap.add_argument("--student", default="B4_lambda50")
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--n-per-pair", type=int, default=5, help="tasks per style pair for P1 (20 pairs * 5 = 100)")
    ap.add_argument("--rms", type=float, default=0.25, help="Oracle edit RMS budget")
    ap.add_argument("--batch-size", type=int, default=4, help="batch size for P1 evaluation")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")'''

assert old_main in s
s = s.replace(old_main, new_main)

# ensure torch is imported at top
if 'import torch' not in s:
    s = 'import torch\nimport torch.nn.functional as F\n' + s

io.open(p, 'w', encoding='utf-8').write(s)
print("Merged run_p1 into vae_gap_probe.py successfully.")
