import json

p0_path = "deliver/vae_gap_results_p0.json"  # we will save current local as p0 first if needed
main_path = "deliver/vae_gap_results.json"

# load remote p1 results
p1_d = json.load(open(main_path, encoding="utf-8"))

# now re-run p0 locally to get the clean combined dictionary
import subprocess, sys
p = subprocess.run([sys.executable, "tools/v6_next/vae_gap_probe.py", "--mode", "p0"], capture_output=True, text=True)
combined = json.load(open(main_path, encoding="utf-8"))

# merge p1 geometry into combined
combined["analyses"]["p1_geometry"] = p1_d["analyses"]["p1_geometry"]

with open(main_path, "w", encoding="utf-8") as f:
    json.dump(combined, f, indent=2, ensure_ascii=False)

print("Merged P0 and P1 into deliver/vae_gap_results.json successfully!")
print("Top keys:", list(combined.keys()))
print("Analyses keys:", list(combined["analyses"].keys()))
