# 01 · 当前弱点深度剖析、源码级病灶取证与历史实验防踩坑审计

本文档在启动 `V19` 实验前，将立足干净上下文的全部审视结论完整固化：第一部分以顶会 Senior Area Chair 视角剖析当前 [`docs/icme2027/paper.pdf`](../icme2027/paper.pdf) 的四个真实弱点；第二部分给出刚从仓库源码中定位到的四个确定性数学/实现病灶；第三部分逐条核对 `docs/negative_results/`、`docs/1111/`、`docs/v12/`、`docs/v14/` 的历史教训，确保新方案零重复踩坑、零评测污染。

---

## 一、对当前 `PRISM v1` 论文与实验结果的冷酷审视（四大弱点）

### 弱点 1：定性视觉主图缺失（Qualitative Comparison Figure = 0）
- **现状**：当前 6 页 `IEEEtran` 正文包含 4 张定量/架构图（`fig1` 风格-内容-速度气泡前沿、`fig2` 三大悖论三联图、`fig3` 三算子矢量架构图、`fig4` CSD 特异性前沿）与 4 张数据表，**没有任何一张生成画作的像素级视觉对比图**。
- **风险等级**：**致命（ICME / ACM MM / CVPR 共同红线）**。艺术风格迁移属于强视觉感知任务；缺少与 `Seedream 4.5`、`StyTr2`、`StyleID`、`SaMam`、`AdaIN` 的同源跨画风视觉拼版（以及局部笔触/边缘放大、反事实错标签视觉对照），审稿人极易质疑高频伪影或色彩退化。
- **解决路径**：在 4090 上导出最优单步/多步臂完整 750 张 PNG，精选 4 组极具视觉张力的跨域样本（文艺复兴肖像 $\to$ 浮世绘木刻、极简几何 $\to$ 洛可可繁复粉彩、印象派风景 $\to$ 文艺复兴古典明暗等），将 `Table IV` 关键数字浓缩进 `Table II` C 栏或正文段落，在第四页/第五页腾出双栏通栏定性大图，严守 6 页上限。

### 弱点 2：推理期隐式依赖离散源画派标签 `sid`（限制真实场景与零样本泛化）
- **现状**：在 `exp/v18/ld4_plus.py:252` 中，`FastDINOStudent` 的条件向量写为：
  ```python
  cond = self.style_emb(tid.clamp(0, 4)) - self.style_emb(sid.clamp(0, 4))
  ```
  同时 `CalibratedLD4Plus.refine` 中的双向排斥项也使用了测试条目自带的 `sid`。
- **风险等级**：**高（顶会审稿人必问方法论漏洞）**。在标准任意图像风格迁移（如自然照片转油画 `Photo-to-Art`，或用户输入一张未知流派插画 $x_s$）中，推理期只给定源图像 $x_s$ 与目标画风指令（类别 $y_t$ 或参考图 $x_{\mathrm{ref}}$），**并不存在先验真值离散源标签 $y_s \in \{0,\dots,4\}$**。此外，固定 5 类查表无法处理未见过的艺术流派（`R5 / Other5`）。
- **解决路径**：引入与商投影 $\Pi_\perp(x)$ 严格互补的**连续风格纤维编码器** $\hat{u}(x) = \phi(\Pi_\parallel(x))$。推理期仅从输入图像 $x_s$ 自身提取连续源风格坐标 $\hat{u}(x_s)$，构造免源标签弦向量 $e(x_s\to\tau) = u_\tau - \hat{u}(x_s)$；同分布判定亦可通过 $\|u_\tau - \hat{u}(x_s)\|_2$ 连续平滑衰减（在离散同标签协议下自然兼容零位移，在单参考图 $x_{\mathrm{ref}}=x_s$ 下精确为零）。

### 弱点 3：少数派画风（`Ukiyo-e` / `Early Renaissance`）命中率偏低，源锁定率仍超 50%
- **现状**：在 `V18_RESULTS_AUTO.md` 中：
  - `V18_A1b_LD4plus_e18`（`38.00%` Top-1）：逐画派 Top-1 为 `ER 10.8% / Imp 60.0% / Min 71.7% / Roc 42.5% / Uki 5.0%`，`Src-Lock = 52.7%`；
  - `V18_A2_FastStudent`（`33.83%` Top-1）：逐画派 Top-1 为 `ER 9.2% / Imp 55.8% / Min 62.5% / Roc 32.5% / Uki 9.2%`，`Src-Lock = 58.5%`。
  - 对比外部 `StyTr2` 的裸 Top-1（`39.50%`，尽管其 `LPIPS = 0.7386` 严重毁内容），我们还需要依赖“在内容保真区间 $\mathrm{DINO\text{-}C}\ge 0.70$ 内最优”这一限定语。
- **解决目标**：只要把 `Ukiyo-e` 和 `Early Renaissance` 从 `<11%` 拉升到 `25%–35%`（不牺牲 `Imp / Min / Roc`），总体 Cross600 Top-1 就会直接从 `38.0%` 跃升至 **`43%–47%+`**，无条件超越 `StyTr2`（`39.50%`），拿走全表保真与特异性的绝对双冠。

### 弱点 4：多步教师采用三段历史组件拼接，且单步学生存在 `4.2pp` 折损
- **现状**：`V18` 教师链路为 `B4 (V6) -> 2步 N2C SPD (V8) -> 3步 LD4+ 掩码搜索 (V9/V18)`，历史包袱重；而单步 `FastStudent`（`33.83%`）尚未追平多步教师（`38.00%`）。若纯单步 `~30 ms` 模型自身即可突破 `38%–42%+`，论文主线将无比纯粹锋利。

---

## 二、源码级四大确定性病灶取证（为什么 `V18` 会卡在 `38.0%` 与 `Ukiyo-e 5%`？）

本轮对仓库底层模块逐行复查后，上述现象被完全解释——没有任何玄学，全部源于以下四处确定性代码机理：

### 病灶 1：`src/model` 中 `LatentStyleNet (R1)` 的 `GroupNorm` 架构性色盘失明
复查 `LatentStyleNet`（`runs/v6_next/probes/r1_r1.pt`，自 V6 沿用至 V18 的唯一潜空间梯度判别器）：
```python
self.stem = nn.Sequential(nn.Conv2d(4, w, 3, padding=1), nn.GroupNorm(8, w), nn.SiLU())
self.b64  = nn.Sequential(nn.Conv2d(w, w, 3, padding=1), nn.GroupNorm(8, w), nn.SiLU())
self.b32  = nn.Sequential(nn.Conv2d(w*2, w*2, 3, padding=1), nn.GroupNorm(8, w*2), nn.SiLU())
self.b16  = nn.Sequential(nn.Conv2d(w*3, w*3, 3, padding=1), nn.GroupNorm(8, w*3), nn.SiLU())
# forward: adaptive_avg_pool2d(h, 1) -> Linear(w*3, 5)
```
**致命后果**：
1. **每一层卷积后紧跟 `GroupNorm`**，在空间维度上强制减均值、除标准差，**彻底抹除了全局色度偏移、明暗对比度（Chiaroscuro）与饱和度量级**！
2. 末端仅做一阶 `adaptive_avg_pool2d(1)`，**丢弃了全部跨通道二阶 Gram 纹理协方差**；
3. 仅在 $64\times 64$ 缩放潜空间用 6 层 $3\times 3$ 小卷积训练，感受野极小，而 `V14`（`docs/v14/V14_RESULTS.md`）已实锤证明：**小感受野潜空间/DWT 特征的画派可分性（`.544`）远逊于 RGB 域特征与全局上下文模型（`.793`）**。
因此，`R1` 能识别靠高频局部方差区分的 `Impressionism`（高频短促笔触）和 `Minimalism`（局部零方差），却对依赖**特定三维色盘（普鲁士蓝/朱红平涂、古典褐金蛋彩）+ 全局墨线/透视构图**的 `Ukiyo-e` 与 `Early Renaissance` 近乎失明！

### 病灶 2：`exp/v9_local/solver.py` 各向同性欧氏球把“改色盘”当成“毁结构”惩罚
复查 `solver.py:17-20` 与 `ld4_plus.py:183-184`：
```python
cand = project_rms(zi + eta * step, z0i, rho)                 # 四通道等权欧氏球 ||z - z0||_RMS <= rho
feas = (cand - z0).square().mean() / (z0 - z_src).square().mean() <= eps
```
**致命后果**：
- 我们在 `V18 Gate 0`（`gate0_leakage_audit.json`）已实测证明：`c1_gentle(z_src)` 驱动的基底 $z_0$ 仍然携带高达 **`71.33%` 的源画派统计泄漏**（尤其是潜空间 4 通道中包含的 3 个源图像色度主轴！）。
- 各向同性约束 $\|z - z_0\|_2 \le r$ 对 **1 维结构亮度通道 $\hat{w}_Y$** 和 **3 维色度子空间 $\hat{w}_Y^\perp$** 施加完全相同的硬上限！只要优化器试图把整幅古典暗褐色油画洗成浮世绘的浅米色纸本+普鲁士蓝天空，$\|P_{\mathrm{chr}}(z - z_0)\|_2$ 立刻超标被 `project_rms` 和 `cost_tot` 一票否决回滚！

### 病灶 3：`exp/v9_local/direction.py` 的 `region_mask(frac=0.35)` 冻结了 65% 画布像素
复查 `direction.py:80-86`：
```python
k = max(1, int(round(frac * 64)))   # frac = 0.30 ~ 0.35 -> 仅选出 19~22 个 8x8 块
m = (m > 0.5).float() * m           # 其余 65%~70% 空间区域掩码精确为 0！
```
**致命后果**：当一幅画有 **65%–70% 的面积完全保持 `z_n2` 不动** 时，全图视觉判别器 `CSD-ViT-L` 自然在超过一半（`52.7%`）的样本上继续判回源画派！在各向同性信任域和劣质 `R1` 时代，`frac=0.35` 是防止内容崩坏的无奈拐杖；一旦换成**亮度-LCN 结构保护 + 色度全画布放开**，这块遮住 65% 画布的硬铁板即可彻底摘除。

### 病灶 4：`FastDINOStudent` 缺乏潜色度预清洗，且采用全图均一标量调制
复查 `exp/v18/ld4_plus.py:254-258`：
- 输入起点 $z_{\mathrm{B4}}$ 与 $\mathcal{C}_1(z_s)$ 未剥离潜空间三维源色度子空间 $\hat{w}_Y^\perp$；
- 调制项 `h * (1 + g[:, :, None, None]) + b[:, :, None, None]` 在全部 $64\times 64$ 空间位置广播同一个标量向量 $(g, b)$，尽管输入端已经算出了富含语义分区的 $\tilde{P}_{16} \in \R^{384\times 16\times 16}$，却只做了一次静态加法融合，未能让“人物面部”与“背景天空/水面”执行差异化弦传输。

---

## 三、历史实验交叉复查与防踩坑核对表（绝不重蹈覆辙）

在设计 `V19` 的每一个模块前，我们逐条对照仓库历史证伪记录，确立刚性边界：

| 历史档案与证伪结论 | 为什么当年失败 | `V19` 的严格规避与正确设计 |
|---|---|---|
| **`Negative_Results` L2：`d2` 纯 SWD 分布监督失败**<br>（`L_dist` 降 65%，但 `CSD-SWD` 不降，$\Delta T$ 与 SWD 反相关） | 纯分布切片距离（SWD）存在多样性塌缩与距离尺度捷径，不直接优化类别可分判别边界（`V14` 亦确认：**蒸馏目标须以判别式边际为主**）。 | **不以无监督 SWD 作为主驱动力**！主驱动力采用**多尺度无归一化一阶/二阶矩特征上的类别均衡判别式双向 Hinge 边际**（Target 吸引 − Source 排斥），仅辅以极轻量的目标类色度矩锚定防止跑偏出自然色域。 |
| **`V15 PT_C` & `V6_next`：无信任域直接梯度上升导致内容崩坏**<br>（`DINO-C: 0.75 -> 0.26`） | 连续无约束优化判别器梯度会迅速钻入高频对抗纹理捷径，摧毁物体几何。 | **严禁无约束 Adam 刷判别器**！保留并升级经过 `V8/V9/V18` 反复验证的**硬信任域投影 + 离散步长回溯线搜索（Feasible Line Search）**，但将各向同性球升级为**亮度-LCN 严锁 + 色度放开的各向异性信任域**。 |
| **`V12-GCL`：复杂关系/自相似内容 Query 闸门不 binding** | 风格方向与复杂关系门控梯度近乎正交（$\cos \approx 0.09$），堆叠复杂内容损失纯浪费算力。 | **绝不堆砌复杂自相似/注意力内容损失**！内容保护只用两个极简、零冗余的正交锚点：潜空间一维亮度低频/LCN 约束 + 像素域 $\mathrm{LCN}_5(Y(\mathcal{D}_\star(z)))$ 一致性。 |
| **`Negative_Results` L1：$\Delta T > 0$ 不代表 Top-1 过关** | 余弦微移未跨过 Voronoi 决策面时 Top-1 仍可低于 10%。 | 全程以完整 **Test750（Cross600 CSD Top-1 + Margin + DINO-S + CLIP-S + DINO-C + LPIPS + Same150 + Wrong-ID 因果门）** 作为唯一验收标尺，绝不用代理指标自嗨。 |
| **`04_bugs`：数据泄漏、裁判污染与 S10 解码契约** | ① 训练池混入测试/参考图；② 若用裁判同款模型做优化会虚高该指标（Goodhart 定律）；③ `post_quant_conv` 漏调或双调。 | ① 严格沿用已校验剔除 `tasks_750 + references_v1` 的 4,400 训练池与 300 校准池；<br>② **物理隔离裁判骨干**：离线教师严禁使用 `CSD-ViT-L`、`DINOv2`、`CLIP`、`LPIPS-Alex`！仅使用标准 ImageNet 预训练卷积骨干（`VGG-16` / `ConvNeXt-Tiny`）在 4,400 张训练画作上自训判别头；<br>③ 统一通过 `src.utils.decode_s10`（或等价单次封装算子）执行可微解码。 |
