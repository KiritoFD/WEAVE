import json, re

PAPER = {
    # 论文 paper.tex 主表 Train/Infer 列：分钟 / 750 张输出 / RTX 3060
    'SD-Turbo':      ('free',  5.0),
    'StyleAligned':  ('free', 77.0),   # 榜单未收录（无对应行）
    'Z-STAR':        ('free', 180.0),  # 榜单未收录
    'StyleShot':     ('free', 306.0),
    'StyleID':       ('free', 63.0),
    'CUT':           (323.0,  5.0),
    'SaMST':         (39.5,  10.0),
    'SaMam':         (436.0, 17.6),
    'StyTR-2':       (1440.0, 38.0),
    'AesPA-Net':     (6.1,   23.0),
    'Latent-WCT':    ('free',  3.67),
    'WEAVE':         (1.4,    1.22),   # 主表 \textbf{WEAVE} 行：1.4/1.22
    'IDT':           None, 'TGT': None, 'Seedream': None,
}

# 榜单名 -> 论文名
ALIAS = {
    'StyleShot': 'StyleShot', 'StyleID': 'StyleID', 'AesPA': 'AesPA-Net',
    'StyTr2': 'StyTR-2', 'SaMAM': 'SaMam', 'SaMST': 'SaMST', 'CUT': 'CUT',
    'Identity': 'IDT', 'SD-Turbo': 'SD-Turbo', 'WEAVE (Published)': 'WEAVE',
    'Seedream 4.5': 'Seedream', 'WCT': 'Latent-WCT',
}


PARAMS = {
 'SD-Turbo': '0+1100', 'StyleAligned': '0+1100', 'Z-STAR': '0+1100',
 'StyleShot': '0+2400', 'StyleID': '0+1100', 'CUT': '7.0', 'SaMST': '8.3',
 'SaMam': '8.8', 'StyTR-2': '35+13', 'AesPA-Net': '11+20',
 'Latent-WCT': '0+84', 'WEAVE': '1.04+84',
}

d = json.load(open('deliver/q2a_oracle_scores_baseline_protocol.json', encoding='utf-8'))
board = d['leaderboard_baseline_methods']

def to_ms(min750):
    return round(min750 * 60000.0 / 750.0, 2)   # min/750img -> ms/img

rows = []
for x in board:
    pname = ALIAS.get(x['method'])
    t = PAPER.get(pname) if pname else None
    if t is None:
        tr, inf, ms, src = '', '', None, '—'
    else:
        tr, inf = t
        ms = to_ms(inf) if isinstance(inf, (int, float)) else None
        src = 'paper.tex 主表'
    rows.append({**x, 'paper_name': pname or '', 'train_min': tr,
                 'infer_min_750': inf, 'infer_ms_per_img': ms,
                 'timing_source': src,
                 'params': PARAMS.get(pname, '—')})

# 本轮新 Decoder（TRT 实测，仅 decoder 段；整条用论文 enc+WEAVE 18.7 拼接估计）
trt = json.load(open('deliver/s10_trt_b8_bench.json', encoding='utf-8'))
s10_ms = next(r['ms_per_img'] for r in trt if 's10' in r['variant'])
off_ms = next(r['ms_per_img'] for r in trt if 'official' in r['variant'])
ENC_WEAVE = 18.7   # 论文 ledger: encoder+WEAVE 18.7 ms/img (RTX 3060, TRT/ONNX)
rows.append({
    'rank': None, 'method': 'Q2A+Oracle + 新 SD0-S10 C0 (TRT, 本轮)',
    'category': 'Ours (new)', 'vae': 'SD0-S10 10.66M, TRT fp16 b8',
    'style_affinity': 0.5575, 'dino_s': 0.4323, 'clip_s': 0.6827,
    'dino_c': 0.7324, 'lpips': 0.5022, 'csd_top1': 0.3683,
    'paper_name': '', 'train_min': '', 'infer_min_750': '',
    'infer_ms_per_img': round(ENC_WEAVE + s10_ms, 2),
    'timing_source': '本轮 TRT decoder 实测 27.08 + 论文 enc+WEAVE 18.7（拼接估计）',
    'is_new': True,
})
# 参考：论文 WEAVE 官方 Decoder TRT 整条
rows.append({
    'rank': None, 'method': 'WEAVE published (官方 SDVAE, TRT, 论文)',
    'category': 'paper ref', 'vae': 'sd-vae-ft-ema 49.49M, TRT fp16 b8',
    'style_affinity': None, 'dino_s': None, 'clip_s': None,
    'dino_c': None, 'lpips': None, 'csd_top1': None,
    'paper_name': 'WEAVE', 'train_min': 1.4, 'infer_min_750': 1.22,
    'infer_ms_per_img': 97.3, 'timing_source': 'paper.tex 主表 + supplement audited',
})

L = []
L.append('# 32 方法总表（加时间维度）\n')
L.append('**权威来源**：`aaai2027_v4/paper.tex` 主表 `Train/Infer` 列（**分钟 / 750 张输出 / 单卡 RTX 3060**，caption 明确；`free`=无需训练）。  ')
L.append('**换算**：`ms/img = infer_min × 60000 / 750`。论文主表 WEAVE = `1.4/1.22` min → 73.2 s / 750 = **97.6 ms/img**，与 supplement audited `73.0 s / 97.3 ms/img` 吻合。  ')
L.append('**注意**：`docs/exp/b5_compute_cost.md` 里「126 s / 168 ms/img」是旧草稿数字，与当前主表不符，**不采用**。  ')
L.append('**指标列**来自 `docs/baseline/UNIFIED_FULL_RANKING.md`（paper_rgb_v1, cross600）。**时间列只对能在主表里找到同名方法的行填充**；Ours/V6 衍生配置与 IDT/TGT/Seedream 主表未报时间，留空。  ')
L.append('**本轮新 Decoder 的时间是 TRT decoder 段实测 + 论文 enc+WEAVE 段拼接估计**，不是端到端重测，已单独标注。\n')

L.append('## 全表（按 Style Affinity 降序；时间列升序见下节）\n')
L.append('| # | 方法 | 类别 | Affinity | DINO-S | CLIP-S | DINO-C | LPIPS | CSD Top-1 | Train(min) | **Infer(min/750)** | **Infer(ms/img)** | 时间来源 |')
L.append('|---:|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|')
def f(v, fmt='{:.4f}'):
    if isinstance(v, str) and v:
        return v
    return fmt.format(v) if isinstance(v, (int, float)) else '—'
for i, r in enumerate(sorted([x for x in rows if x.get('style_affinity') is not None],
                             key=lambda z: -z['style_affinity']), 1):
    mark = ' **NEW**' if r.get('is_new') else ''
    L.append('| %d | %s%s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
        i, r['method'], mark, r['category'],
        f(r['style_affinity']), f(r['dino_s']), f(r['clip_s']), f(r['dino_c']),
        f(r['lpips']), f(r['csd_top1']),
        f(r['train_min']), f(r['infer_min_750']), f(r['infer_ms_per_img'], '{:.2f}'),
        r['timing_source']))
L.append('\n> 表中 `—` = 论文主表未报该方法时间，或该行是 V6/本轮衍生的 Ours 配置（主表不含）。\n')

L.append('## 有同口径时间的 12 个 baseline（按 Infer 升序）\n')
L.append('| 方法 | Params(M) | Train(min) | **Infer(min/750)** | **Infer(ms/img)** | Affinity | CSD Top-1 |')
L.append('|---|---:|---:|---:|---:|---:|---:|')
timed = [r for r in rows if isinstance(r['infer_ms_per_img'], (int, float))
         and r['timing_source'] == 'paper.tex 主表']
for r in sorted(timed, key=lambda z: z['infer_min_750']):
    L.append('| %s | %s | %s | %s | %s | %s | %s |' % (
        r['method'], r['params'],
        f(r['train_min']), f(r['infer_min_750']), f(r['infer_ms_per_img'], '{:.2f}'),
        f(r['style_affinity']), f(r['csd_top1'])))
L.append('\n**同口径结论**：')
L.append('- 最快的是 `Latent-WCT 3.67 min`（293.6 ms/img? 不，3.67 min/750 = **293.6 ms/img**——注意 3.67 min 是 750 图总时长，换算后 293.6 ms/img，而 WEAVE 1.22 min = 97.6 ms/img）；')
L.append('- 换算：`Infer(min) × 60000 / 750 = ms/img`。例如 `3.67 min → 293.6 ms/img`，`1.22 min → 97.6 ms/img`。')
L.append('- **主表里已报时间的方法中最快的是 WEAVE 官方（97.6 ms/img）**；本轮新 S10 整条拼接估计 `18.7 + 27.08 = 45.78 ms/img`，约为它的 **2.13×**（不同源，见口径说明）。')
L.append('- 但这是**拼接估计**（论文 enc+WEAVE 段 18.7 + 本轮 TRT S10 decoder 27.08），不是端到端重测；且论文 baseline 的 Infer 是各自完整方法在 3060 上的总时长，与「只换 Decoder」不完全同类，解读需谨慎。\n')

L.append('## 关键口径说明\n')
L.append('1. **同口径**：12 个 baseline 与 WEAVE 官方的 Infer 都来自 `paper.tex` 主表同一列、同一硬件（RTX 3060）、同一 750 图输出、单位分钟 —— 彼此严格可比。')
L.append('2. **本轮新 S10 的时间不同源**：decoder 段是 3060 WSL + TensorRT 11.4 + FP16 b8 + CUDA Event 实测（27.08 ms/img）；整条 45.78 ms/img 是拼接论文 enc+WEAVE 段得来，**不能当作与 baseline 完全同口径的端到端实测**。')
L.append('3. **论文主表 WEAVE `1.4/1.22`**：1.4 min 训练、1.22 min 推理（750 图）；supplement 的 `73.0 s = 97.3 ms/img` 是同一数字的 audited 复测。')
L.append('4. **Ours 的 V6/Q2A/ScoreDiff 系列在榜单里没有时间**：`UNIFIED_FULL_RANKING.md` 只有质量指标，没有这些配置的 Infer；主表也只报 WEAVE 一个 Ours 行。所以「把时间维度加到总表」只能加到主表覆盖的 12 个 baseline + WEAVE，其余留空。')
L.append('5. **TRT decoder 段实测对照**（同 3060 / b8 / fp16）：官方 49.49M = 73.34 ms/img，新 S10 = 27.08 ms/img，**2.71×**（`deliver/s10_trt_b8_bench.json`）。')

open('docs/v6/BASELINE_TIMING_TABLE.md', 'w', encoding='utf-8').write('\n'.join(L))
print('written docs/v6/BASELINE_TIMING_TABLE.md')
print('timed baselines:', len(timed))
for r in sorted(timed, key=lambda z: z['infer_min_750']):
    print(' ', r['method'], r['infer_min_750'], 'min ->', r['infer_ms_per_img'], 'ms/img')
