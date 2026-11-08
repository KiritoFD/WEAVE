"""Generate docs/baseline/RANKING.md from Evaluator V2 results (D5-512, paper_rgb_v1)."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EVAL = REPO / "runs" / "v6" / "evaluator_v2_all_results.json"
OUT = REPO / "docs" / "baseline" / "RANKING.md"

# display order / naming for the paper table
LABEL = {
    "Identity": "Identity（输入原图）",
    "AdaIN": "AdaIN",
    "WCT": "WCT (VGG19)",
    "CUT": "CUT",
    "SaMST": "SaMST",
    "SaMAM": "SaMAM",
    "SD-Turbo": "SD-Turbo",
    "StyleID": "StyleID",
    "StyleShot": "StyleShot",
    "StyTr2": "StyTr2",
    "AesPA": "AesPA",
    "WEAVE (Published)": "WEAVE（已发表原版）",
    "Seedream 4.5": "Seedream 4.5",
}


def fmt(v, nd=4, sign=False):
    if v is None:
        return "—"
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def main() -> None:
    d = json.loads(EVAL.read_text(encoding="utf-8"))
    rows = []
    for name, entry in d["methods_evaluated"].items():
        c = entry["paper_rgb_v1"]["cross600"]
        rows.append(
            {
                "raw_name": name,
                "name": LABEL.get(name, name),
                "top1": c["Target_Top1"] * 100.0,
                "margin": c["Target_Margin"],
                "dT": c["CSD_dT"],
                "dS": c["CSD_dS"],
                "mmd": c["CSD_MMD"],
                "dinoc": c["DINO_C"],
                "lpips": c["LPIPS"],
                "styles": c["styles_improved"],
                "ukiyo": c["ukiyo_dT"],
                "n": c["n"],
            }
        )

    L = []
    L.append("# Baseline 排名（冻结 600 跨风格任务，paper_rgb_v1）\n")
    L.append("> 数据来源：`runs/v6/evaluator_v2_all_results.json`（Evaluator V2 重算，非历史 CSV）。")
    L.append("> 协议：`paper_rgb_v1`（以原始 RGB 源图为零点）、`references_v1` 统一参考池、")
    L.append("> `tasks_750.json` 冻结任务；此处只取 600 条跨风格任务。\n")
    L.append("> 图像：`docs/baseline/README.md` 说明压缩包位置与缺失情况。\n")

    L.append("## 一、按 Target Top-1 排名（风格是否真的成立）\n")
    L.append("| # | 方法 | Target Top-1 ↑ | Target Margin ↑ | CSD ΔT ↑ | CSD-MMD ↓ | DINO-C ↑ | LPIPS ↓ | 5类改善 | Ukiyo-e ΔT |")
    L.append("|---:|---|---:|---:|---:|---:|---:|---:|:---:|---:|")
    for i, r in enumerate(sorted(rows, key=lambda x: -x["top1"]), 1):
        mark = " **←** " if r["raw_name"].startswith("V6") else ""
        L.append(
            f"| {i} | {r['name']}{mark} | **{r['top1']:.1f}%** | {r['margin']:+.4f} | {r['dT']:+.4f} | "
            f"{r['mmd']:.4f} | {r['dinoc']:.4f} | {r['lpips']:.4f} | {r['styles']}/5 | {r['ukiyo']:+.4f} |"
        )

    L.append("\n## 二、按 Target Margin 排名（目标风格是否压倒其他四类）\n")
    L.append("| # | 方法 | Target Margin ↑ | Target Top-1 | CSD ΔT | DINO-C |")
    L.append("|---:|---|---:|---:|---:|---:|")
    for i, r in enumerate(sorted(rows, key=lambda x: -x["margin"]), 1):
        L.append(
            f"| {i} | {r['name']} | **{r['margin']:+.4f}** | {r['top1']:.1f}% | {r['dT']:+.4f} | {r['dinoc']:.4f} |"
        )

    L.append("\n## 三、按 CSD ΔT 排名（目标原型相似度增幅）\n")
    L.append("| # | 方法 | CSD ΔT ↑ | Target Top-1 | Target Margin | DINO-C |")
    L.append("|---:|---|---:|---:|---:|---:|")
    for i, r in enumerate(sorted(rows, key=lambda x: -x["dT"]), 1):
        L.append(
            f"| {i} | {r['name']} | **{r['dT']:+.4f}** | {r['top1']:.1f}% | {r['margin']:+.4f} | {r['dinoc']:.4f} |"
        )

    L.append("\n## 四、WEAVE 各代的位置\n")
    L.append("| 版本 | 潜空间 | 评测面 | Target Top-1 | 说明 |")
    L.append("|---|---|---|---:|---|")
    L.append("| WEAVE（已发表原版） | SD-VAE 4×64×64 | 冻结 600 | 16.7% | 频域 ODE + AdaIN2.0，靠强灌颜色 |")
    L.append("| V6 ScoreDiff（lr0.055） | SD-VAE 4×64×64 | 冻结 600 | 5.0% | 12 步差分密度，保真优先 |")
    L.append("| V6 D1 Gentle | SD-VAE 4×64×64 | 冻结 600 | 3.2% | 单步直接算子 |")
    L.append("| V6.1 Calib_Balanced_s16 | SD-VAE 4×64×64 | 冻结 600 | **27.7%** | 风格专属步长标定，当前 V6 最好 |")
    L.append("| V8 B2 CFM（raw） | SD-VAE 4×64×64 | 冻结 750 | 2.7% | 单向速度场，失败 |")
    L.append("| **V9 双向布朗桥（OT0.5, step4000）** | **DCAE 32×16×16** | 冻结 600 的 **60 项子集** | **31.7%** | 与上表**不可直接比**（子集+潜空间不同） |")

    L.append("\n## 五、怎么读这张表\n")
    L.append("1. **Top-1 才是\"风格是否成立\"的主判据**。CSD ΔT 高但 Top-1 低，说明只是把图推离源风格，")
    L.append("   并没有真正落到目标风格（例如 SaMAM ΔT +0.10 但 Top-1 只有 2.5%）。")
    L.append("2. **Margin 衡量目标风格的绝对优势**。只有 StyleShot (+0.218) 与 StyleID (+0.153) 为正；")
    L.append("   其余全部为负，说明大多数方法的目标风格仍未压倒竞争类别。")
    L.append("3. **DINO-C 与 LPIPS 只用于排除崩溃**。StyleShot 的 DINO-C 仅 0.357、LPIPS 0.770，")
    L.append("   是\"重绘式\"风格化；V6 ScoreDiff 的 DINO-C 0.902 / LPIPS 0.237 是\"保真式\"，两者不在同一取舍点。")
    L.append("4. **不要跨数据集比较**。本表只覆盖 D5-512；P2A-256 与 R5-WikiArt 尚未用 Evaluator V2 重算。")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {OUT}")
    print("\n".join(L[:14]))


if __name__ == "__main__":
    main()
