import pathlib
import subprocess

head = subprocess.run(["git", "show", "HEAD:sbflow/style_loss.py"],
                      capture_output=True, text=True, encoding="utf-8").stdout

# Extract the two definitions I accidentally dropped.
i = head.index("def chunked_swd2(")
j = head.index("def supervised_contrastive(")
dropped = head[i:j].rstrip() + "\n\n\n"
print(f"recovered {len(dropped.splitlines())} lines")

p = pathlib.Path("sbflow/style_loss.py")
s = p.read_text(encoding="utf-8")
k = s.index("def supervised_contrastive(")
s = s[:k] + dropped + s[k:]
p.write_text(s, encoding="utf-8")
print("re-inserted chunked_swd2 and StyleEncoder")
