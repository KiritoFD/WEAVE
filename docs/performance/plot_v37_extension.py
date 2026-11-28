"""V37 panels: keep diagnostic, official Core750, and independent holdouts distinct."""
from __future__ import annotations
import html
import json
import re
from pathlib import Path
from src.eval.pareto import frontier_indices


def collect(root, baseline):
    out = root / "runs/v37_editing"
    panels = {}
    def read(p):
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
    core = read(out / "official_core750.json")
    diag = read(out / "diagnostics/core_quadrants.json")
    if core or diag:
        points = [{"label": p["arm"], "group": "V36 P+A3", "c": p["dc"], "s": p["ds"], "t": p["t1"]} for p in baseline]
        for r in core:
            c = r["cross600"]
            assert c["n"] == 600 and r["same150"]["n"] == 150
            points.append({"label": r["arm"], "group": r["route"] + (" E" if r["tau"] is None else " E+P"),
                           "c": c["dino_c"], "s": c["dino_s"], "t": c["csd_top1"]*100})
        for r in diag:
            c = r["cross600"]
            points.append({"label": "A3 " + r["a3_ref"] + "/P " + r["p_ref"], "group": "D2 control (excluded from frontier)",
                           "c": c["dino_c"], "s": c["dino_s"], "t": c["csd_top1"]*100, "control": True})
        previous = read(root / "runs/v36_art_foundation/measured/v36_ecore_full_report.json")
        if previous:
            for kind in ("standalone", "dual"):
                for r in previous.get(kind, []):
                    if kind == "dual" and (r["lam"] != 1. or r["use_a3_post"]):
                        continue
                    c = r["cross600"]
                    points.append({"label": r["arm"], "group": "V36 E4.5k" + ("" if kind == "standalone" else "+P"),
                                   "c": c["dino_c"], "s": c["dino_s"], "t": c["csd_top1"] * 100})
        panels["Core750 / Cross600"] = {"points": points, "metric": "CSD Top-1 (%)", "note": "Each point measured on all 750 tasks; controls excluded from the official frontier. Lines connect observed points."}
        controls = read(out / "diagnostics/core_q_controls.json")
        if controls:
            panels["Core750 Q32 controls"] = {"points": [p for p in points if p["group"] == "V36 P+A3"] + [
                {"label": r["arm"], "group": r["ckpt"] + " Q " + r["q"], "control": True,
                 "c": r["cross600"]["dino_c"], "s": r["cross600"]["dino_s"],
                 "t": r["cross600"]["csd_top1"] * 100} for r in controls],
                "metric": "CSD Top-1 (%)", "note": "Q32 zero/shuffle controls excluded from official frontier; each arm uses all 750 tasks."}
    seen = read(out / "diagnostics/seen/summary.json")
    if seen:
        panels["Seen-Dev300 diagnostics"] = {"points": [{"label": r["arm"], "group": r["ckpt"] + (" Q control" if r["q"] != "correct" else ""),
                "c": r["dino_c"], "s": r["dino_s"], "t": r["csd_2way_follow"]*100, "control": r["q"] != "correct"} for r in seen],
                "metric": "CSD instance 2-way (%)", "note": "Instance-reference DINO-S; not directly comparable to Core5 style metrics. Q controls shown as crosses."}
    matched = read(out / "matched_seen/summary.json")
    if matched:
        panels["Seen-Dev300 E0/E1 comparison"] = {"points": [
            {"label": r["arm"], "group": r["route"] + (" E" if r["tau"] is None else " E+P"),
             "c": r["dino_c"], "s": r["dino_s"], "t": r["csd_2way_follow"] * 100} for r in matched],
            "metric": "CSD instance 2-way (%)", "note": "Matched 12902-step E0/E1; fixed Seen-Dev300; reference-instance metrics."}
    hold = read(out / "holdouts/summary.json")
    for split in ("artist_ood", "movement_ood"):
        rows = [r for r in hold if r["split"] == split]
        if rows:
            panels[split] = {"points": [{"label": r["arm"], "group": r["route"], "c": r["dino_c"], "s": r["dino_s"], "t": r["csd_2way_follow"]*100} for r in rows],
                             "metric": "CSD instance 2-way (%)", "note": f"Independent {split}, complete {rows[0]['n']} sources per point. No Core5 classifier accuracy."}
    for panel in panels.values():
        valid = [p for p in panel["points"] if not p.get("control")]
        panel["front_s"] = [valid[i] for i in frontier_indices([(p["c"], p["s"]) for p in valid])]
        panel["front_t"] = [valid[i] for i in frontier_indices([(p["c"], p["t"]) for p in valid])]
    return panels


def interactive(panels):
    data = json.dumps(panels, ensure_ascii=False).replace("</", "<\\/")
    return '''<style>
#v37panel {font:14px var(--font-sans,Arial);color:var(--foreground,#aaa);margin-top:24px}
#v37panel select {padding:7px;background:var(--secondary,#252525);color:var(--foreground,#ddd);border:1px solid var(--border,#777);border-radius:6px}
#v37panel svg {width:100%;height:400px;display:block} #v37panel p {line-height:1.45}
</style><div id="v37panel"><label>V37 <select id="v37select"></select></label><svg id="v37svg" viewBox="0 0 720 400" role="img" aria-label="V37 content style Pareto plots"></svg><p id="v37note"></p><p id="v37hover">Hover over a point for its measured values.</p></div><script>
(()=>{const panels=__DATA__,sel=document.getElementById('v37select'),svg=document.getElementById('v37svg');
const NS='http://www.w3.org/2000/svg';function el(n,a,t){const x=document.createElementNS(NS,n);Object.entries(a).forEach(([k,v])=>x.setAttribute(k,v));if(t)x.textContent=t;return x;}
Object.keys(panels).forEach(k=>{const o=document.createElement('option');o.textContent=k;sel.appendChild(o)});
function draw(){svg.innerHTML='';const d=panels[sel.value],pts=d.points,groups=[...new Set(pts.map(p=>p.group))],colors=['#38bdf8','#f59e0b','#34d399','#c084fc','#f472b6','#fb7185','#a3e635','#94a3b8'];
document.getElementById('v37note').textContent=d.note;
const x0=Math.floor((Math.min(...pts.map(p=>p.c))-.02)*10)/10,x1=Math.min(1,Math.max(...pts.map(p=>p.c))+.04);
['s','t'].forEach((key,i)=>{let left=50+i*355,top=35,w=290,h=225;let yy=pts.map(p=>p[key]),lo=key==='s'?Math.floor((Math.min(...yy)-.02)*20)/20:0,hi=key==='s'?Math.ceil((Math.max(...yy)+.02)*20)/20:100;
const sx=v=>left+(v-x0)/(x1-x0)*w,sy=v=>top+h-(v-lo)/(hi-lo)*h;
for(let j=0;j<=4;j++){let x=x0+(x1-x0)*j/4,y=lo+(hi-lo)*j/4;svg.appendChild(el('line',{x1:left,y1:sy(y),x2:left+w,y2:sy(y),stroke:'currentColor',opacity:.18}));svg.appendChild(el('text',{x:left-7,y:sy(y)+4,'text-anchor':'end',fill:'currentColor','font-size':10},y.toFixed(key==='s'?2:0)));svg.appendChild(el('text',{x:sx(x),y:top+h+18,'text-anchor':'middle',fill:'currentColor','font-size':10},x.toFixed(2)));}
svg.appendChild(el('text',{x:left+w/2,y:18,'text-anchor':'middle',fill:'currentColor','font-size':12},key==='s'?'DINO-C vs DINO-S':'DINO-C vs '+d.metric));
svg.appendChild(el('text',{x:left+w/2,y:top+h+36,'text-anchor':'middle',fill:'currentColor','font-size':11},'DINO-C ↑'));
const fr=d['front_'+key].slice().sort((a,b)=>a.c-b.c);svg.appendChild(el('path',{d:fr.map((p,j)=>(j?'L':'M')+sx(p.c)+','+sy(p[key])).join(' '),fill:'none',stroke:'currentColor','stroke-dasharray':'4 3','stroke-width':1,opacity:.8}));
groups.forEach((g,gi)=>{let pp=pts.filter(p=>p.group===g).sort((a,b)=>a.c-b.c),col=colors[gi%colors.length];if(!pp[0].control)svg.appendChild(el('path',{d:pp.map((p,j)=>(j?'L':'M')+sx(p.c)+','+sy(p[key])).join(' '),fill:'none',stroke:col,'stroke-width':1.4}));
pp.forEach(p=>{let dot=el(p.control?'path':'circle',p.control?{d:`M${sx(p.c)-4},${sy(p[key])-4}l8,8m-8,0l8,-8`,stroke:col,'stroke-width':2}:{cx:sx(p.c),cy:sy(p[key]),r:4,fill:col});dot.addEventListener('mouseenter',()=>document.getElementById('v37hover').textContent=p.label+' | C='+p.c.toFixed(4)+' S='+p.s.toFixed(4)+' '+d.metric+'='+p.t.toFixed(2));svg.appendChild(dot)});});});
groups.forEach((g,i)=>svg.appendChild(el('text',{x:20+(i%3)*235,y:315+Math.floor(i/3)*16,fill:colors[i%colors.length],'font-size':10},g)));}
sel.onchange=draw;draw();})();</script>'''.replace("__DATA__", data)


def append_v37_views(root: Path, fig_dir: Path, baseline):
    panels = collect(root, baseline)
    if not panels:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(panels), 2, figsize=(10, 3.6 * len(panels)), squeeze=False)
    for row, (title, d) in enumerate(panels.items()):
        groups = sorted({p["group"] for p in d["points"]})
        for col, key in enumerate(("s", "t")):
            ax = axes[row, col]
            for g in groups:
                pts = sorted([p for p in d["points"] if p["group"] == g], key=lambda p: p["c"])
                ctrl = pts[0].get("control", False)
                ax.plot([p["c"] for p in pts], [p[key] for p in pts], "x" if ctrl else "o-", label=g, markersize=4, linewidth=1)
            ff = sorted(d["front_"+key], key=lambda p: p["c"])
            ax.plot([p["c"] for p in ff], [p[key] for p in ff], "k--", linewidth=.7, alpha=.5)
            ax.set(xlabel="DINO-C", ylabel="DINO-S" if key == "s" else d["metric"], title=title)
            ax.grid(alpha=.2)
            if col == 0: ax.legend(fontsize=6)
    fig.tight_layout()
    path = fig_dir / "v37_pareto.svg"
    fig.savefig(path); plt.close(fig)
    fragment = interactive(panels)
    (fig_dir / "v37_pareto.html").write_text('<!doctype html><html><head><meta charset="utf-8"></head><body>' + fragment + '</body></html>', encoding="utf-8")
    latest = fig_dir / "cleave_pareto_latest.html"
    latest.write_text(latest.read_text(encoding="utf-8").replace("</body>", fragment + "</body>"), encoding="utf-8")
    # Append independent protocol panels to the current vector artifact without blending frontiers.
    svg = fig_dir / "cleave_pareto_latest.svg"
    text = svg.read_text(encoding="utf-8")
    ext = path.read_text(encoding="utf-8")
    vb = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', text)
    evb = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', ext)
    if vb and evb:
        w, h = map(float, vb.groups()); ew, eh = map(float, evb.groups()); scale = w/ew
        header = text[:text.index('>', text.index('<svg'))+1]
        newheader = re.sub(r'height="[^"]+"', f'height="{h+eh*scale}pt"', header)
        newheader = newheader.replace(vb.group(0), f'viewBox="0 0 {w} {h+eh*scale}"')
        text = newheader + text[len(header):]
        content = ext[ext.index('>', ext.index('<svg'))+1:ext.rindex('</svg>')]
        # Matplotlib defs ids must not collide with those in the existing figure.
        ids = re.findall(r'id="([^"]+)"', content)
        for id_ in sorted(ids, key=len, reverse=True):
            content = content.replace(f'"{id_}"', f'"v37_{id_}"').replace(f'#{id_}', f'#v37_{id_}')
        text = text.replace('</svg>', f'<g transform="translate(0 {h}) scale({scale})">{content}</g></svg>')
        svg.write_text(text, encoding="utf-8")
