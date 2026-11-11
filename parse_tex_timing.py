import re, json

tex = open('aaai2027_v4/paper.tex', encoding='utf-8').read()
start = tex.index('\\begin{tabular}{@{}lcccccccccccccc@{}')
seg = tex[start:tex.index('\\bottomrule', start)]
BS = chr(92)
rows = []
for line in seg.splitlines():
    if '&' not in line or line.strip().startswith('%'):
        continue
    if any(k in line for k in ('cmidrule', 'toprule', 'Method', 'multicolumn',
                               'tabular', 'centering', 'setlength', 'midrule')):
        continue
    cells = [c.strip() for c in line.split('&')]
    if len(cells) < 15:
        continue
    name = cells[0]
    name = re.sub(BS + r'textbf\{([^}]*)\}', r'\1', name)
    name = re.sub(BS + r'[a-zA-Z]+\{([^}]*)\}', r'\1', name)
    name = name.replace('{', '').replace('}', '').strip()
    params = cells[13]
    ti = cells[14].split(BS)[0].strip().rstrip('/').strip()
    m = re.match(r'(.+?)/(.+)', ti)
    train, infer = (m.group(1).strip(), m.group(2).strip()) if m else ('', ti)
    rows.append({'method': name, 'params': params,
                 'train_min': train, 'infer_min': infer})
print(json.dumps(rows, ensure_ascii=False, indent=1))
