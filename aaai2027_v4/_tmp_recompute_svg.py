import re, math

svg = open('fig_page1_b.svg', encoding='utf-8').read()

# x axis (style) ticks: pixel x -> value
xticks = {28.339319: 0.50, 68.296741: 0.55, 108.254163: 0.60,
          148.211585: 0.65, 188.169007: 0.70, 228.126429: 0.75}
# y axis (ArtFID) ticks: pixel y -> value
yticks = {23.26048: 200, 52.017268: 250, 80.774056: 300,
          109.530844: 350, 138.287631: 400}


def lerp(tbl, px):
    ks = sorted(tbl)
    if px <= ks[0]:
        return tbl[ks[0]]
    if px >= ks[-1]:
        return tbl[ks[-1]]
    for a, b in zip(ks, ks[1:]):
        if a <= px <= b:
            t = (px - a) / (b - a)
            return tbl[a] + (tbl[b] - tbl[a]) * t


rows = []
for m in re.finditer(r'<use xlink:href="#m[0-9a-f]+"\s+x="([\d.]+)"\s+y="([\d.]+)"\s+style="fill:\s*#([0-9A-Fa-f]+)[^"]*"', svg):
    cx, cy, color = float(m.group(1)), float(m.group(2)), m.group(3)
    style = lerp(xticks, cx)
    artfid = lerp(yticks, cy)
    rows.append((color, style, artfid))
    print(f'  #{color}  x={cx:.2f} y={cy:.2f} -> style={style:.3f} artfid={artfid:.1f}')

print('markers found:', len(rows))
sv = [r[1] for r in rows]
av = [r[2] for r in rows]


def pearson(xs, ys):
    xm = sum(xs) / len(xs)
    ym = sum(ys) / len(ys)
    num = sum((a - xm) * (b - ym) for a, b in zip(xs, ys))
    xd = math.sqrt(sum((a - xm) ** 2 for a in xs))
    yd = math.sqrt(sum((b - ym) ** 2 for b in ys))
    return num / (xd * yd)


r = pearson(sv, [-v for v in av])
print(f'FROM SVG MARKERS: r(style, -ArtFID) = {r:.4f}   R^2 = {r*r:.4f}   n = {len(rows)}')
