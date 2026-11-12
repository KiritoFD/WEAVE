"""CPU-only paired analysis of style metrics and encoder/decoder factorials."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "deliver/vae_ranked_eval"
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
ARMS = ["Oracle1", "B4", "B3", "ScoreDiff_N24", "ScoreDiff_N12"]
VARIANTS = ["sdvae", "a0", "s10", "cedit"]
KEYS = ["target_top1", "target_margin", "dino_s", "clip_s", "dino_margin", "clip_margin",
        "dino_top1", "clip_top1", "dino_dT", "clip_dT", "dino_c", "lpips",
        "edge_energy", "normalized_edge", "spatial_variance"]
KEYS += [m+suffix for m in ['dino','clip'] for suffix in ['_allmean','_contrast']]


def mean(rows, key):
    return float(np.mean([r[key] for r in rows]))


def bootstrap(values, stems):
    keys = sorted(set(stems))
    x = np.array([np.mean([v for v,s in zip(values,stems) if s == k]) for k in keys])
    rng = np.random.default_rng(42)
    draws = rng.integers(0,len(keys),(2000,len(keys)))
    ci = np.percentile(x[draws].mean(1),[2.5,97.5]).tolist()
    return {"mean":float(x.mean()),"ci95":ci,"significant":bool(ci[0]>0 or ci[1]<0)}


def paired(rows, base, keys):
    b = {r['task_id']:r for r in base}
    return {k:bootstrap([r[k]-b[r['task_id']][k] for r in rows],
                        [r['source_stem'] for r in rows]) for k in keys}


def cross(rows):
    return [r for r in rows if r['is_cross']]


def summary(rows):
    return {k:mean(rows,k) for k in KEYS if k in rows[0]}


def add_recon_gains(rows, recon):
    by = {r['source_stem']:r for r in recon}
    for r in rows:
        ti = STYLES.index(r['target_style'])
        for m in ['dino','clip']:
            r[m+'_vs_recon'] = r[m+'_s'] - by[r['source_stem']][m+'_scores'][ti]
            scores=np.asarray(r[m+'_scores'])
            r[m+'_allmean']=float(scores.mean())
            r[m+'_contrast']=float(scores[ti]-np.delete(scores,ti).mean())


def factorial(data, variant):
    cells = {'00':data['results']['sdvae']['B4'],
             '11':data['results'][variant]['B4'],
             '10':data['hybrids'][f'E_{variant}__D_sdvae'],
             '01':data['hybrids'][f'E_sdvae__D_{variant}']}
    cells = {n:cross(r) for n,r in cells.items()}
    for rows in cells.values():
        for r in rows:
            ti=STYLES.index(r['target_style'])
            for m in ['dino','clip']:
                scores=np.asarray(r[m+'_scores'])
                r[m+'_allmean']=float(scores.mean())
                r[m+'_contrast']=float(scores[ti]-np.delete(scores,ti).mean())
    keys = ['target_top1','target_margin','dino_s','clip_s','dino_margin','clip_margin',
            'edge_energy','normalized_edge','dino_allmean','clip_allmean','dino_contrast','clip_contrast']
    lookup = {n:{r['task_id']:r for r in rows} for n,rows in cells.items()}
    ids = [r['task_id'] for r in cells['00']]
    stems = [r['source_stem'] for r in cells['00']]
    out = {'cells':{n:summary(r) for n,r in cells.items()},'effects':{}}
    for k in keys:
        a,b,c,d = [np.array([lookup[n][i][k] for i in ids]) for n in ['00','10','01','11']]
        enc = 0.5*((b-a)+(d-c))
        dec = 0.5*((c-a)+(d-b))
        assert np.allclose(enc+dec,d-a)
        out['effects'][k] = {
            'total':bootstrap(d-a,stems), 'encoder_shapley':bootstrap(enc,stems),
            'decoder_shapley':bootstrap(dec,stems), 'interaction':bootstrap(d-b-c+a,stems),
            'encoder_with_SD_decoder':bootstrap(b-a,stems),
            'decoder_with_SD_encoder':bootstrap(c-a,stems)}
    return out


def main():
    data = json.loads((OUT/'style_comparison.json').read_text(encoding='utf-8'))
    result = {'calibration':{},'cells':{},'factorials':{},'decoder_only':{},'agreement':{},
              'reconstruction':{v:summary(r) for v,r in data['reconstruction'].items()}}
    for name in ['dino','clip']:
        result['calibration'][name] = {'accuracy':mean(data['sources'],name+'_top1'),
            'per_style':{s:mean([r for r in data['sources'] if r['source_style']==s],name+'_top1') for s in STYLES}}
    if 'source_csd' in data:
        result['calibration']['csd'] = {'accuracy':mean(data['source_csd'],'target_top1')}
    for variant,cells in data['results'].items():
        result['cells'][variant] = {}
        for arm,rows in cells.items():
            add_recon_gains(rows,data['reconstruction'][variant])
            base = data['results']['sdvae'][arm]
            add_recon_gains(base,data['reconstruction']['sdvae'])
            cr,br = cross(rows),cross(base)
            keys = KEYS+['dino_vs_recon','clip_vs_recon']
            result['cells'][variant][arm] = {
                'cross600':summary(cr), 'same150':summary([r for r in rows if not r['is_cross']]),
                'delta_vs_sdvae':paired(cr,br,keys),
                'per_target':{s:{'means':summary([r for r in cr if r['target_style']==s]),
                                 'delta':paired([r for r in cr if r['target_style']==s],br,
                                                ['target_top1','dino_s','clip_s','dino_margin','clip_margin'])}
                              for s in STYLES}}
    for variant in ['a0','cedit']:
        if all(k in data['hybrids'] for k in [f'E_{variant}__D_sdvae',f'E_sdvae__D_{variant}']):
            result['factorials'][variant] = factorial(data,variant)
    for arm in ARMS:
        if arm in data['results'].get('s10',{}) and arm in data['results'].get('a0',{}):
            result['decoder_only'][arm] = paired(cross(data['results']['s10'][arm]),
                cross(data['results']['a0'][arm]),KEYS)
    for metric in ['dino_s','clip_s','dino_margin','clip_margin']:
        entries=[]
        for variant in ['a0','s10','cedit']:
            for arm,c in result['cells'].get(variant,{}).items():
                a=c['delta_vs_sdvae']['target_top1']; b=c['delta_vs_sdvae'][metric]
                entries.append({'variant':variant,'arm':arm,'csd_delta':a['mean'],
                    'metric_delta':b['mean'],'same_sign':bool(np.sign(a['mean'])==np.sign(b['mean'])),
                    'both_significant':bool(a['significant'] and b['significant'])})
        result['agreement'][metric]={'same_sign':sum(e['same_sign'] for e in entries),
                                    'n':len(entries),'entries':entries}
    (OUT/'attribution.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 换 VAE 的风格指标复核与归因','',
      '固定同一 750 任务；主表只用 600 cross，150 same 独立统计。无新增图片。',
      'DINO-S：目标画风 30 张固定参考图的最大 CLS 余弦；CLIP-S：目标画风固定图像特征均值原型的余弦。',
      'DINO/CLIP margin 与 Top-1 使用同样的五类参考分数，属于本轮补充诊断，不与 CSD Top-1 混称。',
      '所有差值对照本轮 sdvae；95% CI 按源图聚类配对 bootstrap（2000 次）。',
      'DINO/CLIP 并非专门的风格解耦模型；除目标相似度外，同时查看目标对其他风格的 margin，避免把整体相似度抬高当成目标风格改善。','',
      '置信区间是逐项 95% CI，未做多重比较校正；本轮属于固定候选的探索性配对复核。','',
      '## 真实测试图校准（150 张，参考图与测试图不相交）','',
      '| 判官 | 五类 Top-1 |','|---|---:|']
    for n,c in result['calibration'].items(): lines.append(f"| {n} | {c['accuracy']*100:.2f}% |")
    lines+=['','## Cross600 均值','',
      '| 模型 | VAE | CSD Top-1 | DINO-S | CLIP-S | DINO margin | CLIP margin | DINO-C | LPIPS |',
      '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        for v in VARIANTS:
            c=result['cells'].get(v,{}).get(arm)
            if c:
                m=c['cross600'];lines.append(f"| {arm} | {v} | {m['target_top1']*100:.2f}% | {m['dino_s']:.5f} | {m['clip_s']:.5f} | {m['dino_margin']:+.5f} | {m['clip_margin']:+.5f} | {m['dino_c']:.4f} | {m['lpips']:.4f} |")
    lines+=['','## 相对 sdvae 的配对变化','',
      '| 模型 | VAE | ΔCSD Top-1 (pp) | ΔDINO-S [95% CI] | ΔCLIP-S [95% CI] | ΔDINO margin | ΔCLIP margin |',
      '|---|---|---:|---|---|---:|---:|']
    def fmt(x):return f"{x['mean']:+.5f} [{x['ci95'][0]:+.5f},{x['ci95'][1]:+.5f}]"
    for arm in ARMS:
        for v in ['a0','s10','cedit']:
            c=result['cells'].get(v,{}).get(arm)
            if c:
                q=c['delta_vs_sdvae'];lines.append(f"| {arm} | {v} | {q['target_top1']['mean']*100:+.2f} | {fmt(q['dino_s'])} | {fmt(q['clip_s'])} | {q['dino_margin']['mean']:+.5f} | {q['clip_margin']['mean']:+.5f} |")
    lines+=['','## B4 的 Encoder / Decoder 交叉干预','',
      '对 A0、C_edit 分别建立 E_SD/E_V × D_SD/D_V 的 2×2 对照。混合搭配是刻意的诊断干预，不是推荐的部署方式。',
      f"同链路对角线复核：首批图像与既有 sdvae/B4 PNG 最大 uint8 误差 = {data.get('factorial_diagonal_max_u8_error','not recorded')}。",'',
      'Shapley 分配是该固定流水线内的两因素贡献分解；交互项单独报告，不能外推为模型的普遍性质。','',
      '| VAE | 指标 | 总变化 | Encoder 贡献 | Decoder 贡献 | 交互项 |',
      '|---|---|---:|---:|---:|---:|']
    for v,f in result['factorials'].items():
        for k in ['target_top1','target_margin','dino_s','clip_s','dino_margin','clip_margin']:
            q=f['effects'][k];scale=100 if k=='target_top1' else 1
            lines.append(f"| {v} | {k}{' (pp)' if scale==100 else ''} | {q['total']['mean']*scale:+.5f} | {q['encoder_shapley']['mean']*scale:+.5f} | {q['decoder_shapley']['mean']*scale:+.5f} | {q['interaction']['mean']*scale:+.5f} |")
    lines+=['','## A0 → S10：固定 Encoder 的 Decoder 对照','',
      '| 模型 | ΔCSD Top-1 (pp) | ΔDINO-S | ΔCLIP-S | 归一化相邻像素梯度能量变化 |',
      '|---|---:|---:|---:|---:|']
    for a,q in result['decoder_only'].items():lines.append(f"| {a} | {q['target_top1']['mean']*100:+.2f} | {q['dino_s']['mean']:+.5f} | {q['clip_s']['mean']:+.5f} | {q['normalized_edge']['mean']:+.5f} |")
    lines+=['','## 指标结论一致性','',
      '下表比较相对 sdvae 变化的符号，不等同于显著性检验。逐项置信区间及分目标结果在 attribution.json。','']
    for m,c in result['agreement'].items():lines.append(f"- {m} 与 CSD Top-1 的方向相同：{c['same_sign']}/{c['n']}。")
    lines+=['','## 整体相似度偏移与目标选择性','',
      'allmean 是对五类参考的平均相似度；contrast 是目标类减去其余四类平均。',
      '若目标相似度与 allmean 同幅上涨，但 margin/contrast 不涨，不能归因为目标画风更准确。','',
      '| 模型 | VAE | ΔDINO-S | ΔDINO allmean | ΔDINO contrast | ΔCLIP-S | ΔCLIP allmean | ΔCLIP contrast |',
      '|---|---|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        for v in ['a0','s10','cedit']:
            c=result['cells'].get(v,{}).get(arm)
            if c:
                q=c['delta_vs_sdvae']; vals=[q[k]['mean'] for k in ['dino_s','dino_allmean','dino_contrast','clip_s','clip_allmean','clip_contrast']]
                lines.append(f"| {arm} | {v} | "+' | '.join(f'{x:+.5f}' for x in vals)+' |')
    lines+=['','## 重建图的独立变化（150 张，目标为原图真实画风）','',
      '| VAE | DINO-S | CLIP-S | DINO Top-1 | CLIP Top-1 | 归一化梯度能量 |',
      '|---|---:|---:|---:|---:|---:|']
    for v,c in result['reconstruction'].items():lines.append(f"| {v} | {c['dino_s']:.5f} | {c['clip_s']:.5f} | {c['dino_top1']*100:.2f}% | {c['clip_top1']*100:.2f}% | {c['normalized_edge']:.5f} |")
    lines+=['','## 归因边界','',
      '- A0 → S10 的 Encoder 相同，因此该差异可归到 Decoder 替换；但不能只用相邻像素梯度统计证明是某一频段造成。',
      '- A0、C_edit 的 Encoder/Decoder 因果分解只在 B4 上实施，不能直接外推每个模型的贡献比例。',
      '- Decoder 贡献指冻结 WEAVE 输出经该 Decoder 的任务效果；它混合了 Decoder 的表征偏好及 WEAVE 输出与其训练 latent 分布的兼容性，不等同于 Decoder 独立重建能力差。',
      '- Shapley 是两个受控替换因素的效应分配，包含对称分摊的交互效应；它不是训练动态的机制证明。',
      '- 所有结果都冻结 WEAVE、R1、canonicalizer；没有重新训练 WEAVE 适配新 VAE，也没有人工盲评。',
      '- DINO/CLIP 的真实图分类校准说明此参考集在本测试域具有区分力，不证明特征只含风格或已排除题材混淆。',
      '- 当前证据不支持把 CSD Top-1 下降直接表述为所有意义上的画风退化。','',
      '指标实现：tools/evaluation_v2/metrics.py（DINO-S / CLIP-S 定义）；tools/vae_swap_eval/style_metrics.py（本轮补评）。',
      '模型背景：[DINOv2](https://arxiv.org/abs/2304.07193)，[CLIP](https://arxiv.org/abs/2103.00020)。']
    (OUT/'STYLE_ATTRIBUTION.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('WROTE attribution.json + STYLE_ATTRIBUTION.md; cells=',sum(map(len,result['cells'].values())))


if __name__=='__main__':main()
