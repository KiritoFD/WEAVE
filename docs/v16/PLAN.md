# V16：在3060执行96 Conv编辑迁移与RGB-H2可表达性实验

日期：2026-11-16。依据用户粘贴的A/B/C方案执行，三部分均运行；不因教师下游效果弱而取消Adapter。
已完成，结果见[RESULTS.md](RESULTS.md)；下文记录实际执行规格与实施调整。
当前3060空闲，用户本轮明确指定使用3060，覆盖旧AGENTS中“3060不要碰”的历史占用备注。不会停止其他任务。

## 共同协议

- 全任务决策使用冻结test750（cross600 + same150），不以dev预览代替。
- 解码一律S10，调用`src.utils.decode_s10`；自动除0.18215，checkpoint内post_quant_conv恰好一次。
- 评测独立用`src.eval.measure`的冻结DINO/CSD/CLIP/LPIPS；这些判官不参与训练和反演。
- 输出位于`runs/v16`，packed tensor暂存用于独立评分，不输出逐任务PNG。
- 单卡串行，保留SSH前台会话。数据/旧checkpoint/其他实验产物不覆盖。

## A：不新增训练的容量诊断

实际checkpoint：W_dim96_conv_base（4k）与S1–S6（dim128/160/192/256/320/384，8k）。
S7_dim512只有初始产物，没有final.pt；B8k_dim96未找到。列为缺项，不能宣称已完成同预算容量因果比较。

同一批train/holdout，每风格8个，共40+40真实latent。初次沿原V15路径ID哈希划分；事后审计发现包装ID归一化错误，40个留出中5幅画的另一份副本在FM训练集。原结果保留，补跑A_strict：按画作stem分组，只取所有副本均属于FM留出的画，且排除Test750源与参考。训练诊断40幅保持不变，其中1幅参考重合。
同噪声、同输入、11个时间点0,.1,…,1，计算FM MSE、wrong-label MSE、五个label两两速度差/速度RMS。
条件响应只测大小、不证明方向正确。端点t=0/1明确为诊断点。
复用已有全750输出读latent/RGB频带能量与现有评分；若评分证据缺失则标缺，不能挪用其他裁判版本。

## B：重新锚定的LD4局部算子 + 零残差Adapter

底座固定W_dim96，tau0=.7，20-step Euler，评测噪声与V15一致（seed20261185）。
教师复用R1、N2C SPDHead、局部风格库、25%区域mask和候选方向/步长回溯，内容预算**以当前V15输出为锚点**。
rho=.15、相对内容预算eps=.10、2步。不能沿用旧B4 anchor，否则会把V15结果投回旧工作点。
候选方向先乘每图`min(.15,sqrt(.10)*RMS(z_base-z_src))`，再用原.5/1步长与mask回溯。
初始固定步长版本有约94%样本因预算不匹配回退，已保存`runs/v16/pilot_fixed_step`；修正后重新生成监督并完整训练学生，不将初始回退当风格方向失败。

另修正教师封装的局部能量为逐样本向量：原LocalStyleEnergy.loss返回batch均值，会使proxy的每样本接受决策耦合。
不修改原LD4代码；这里是明确标注的V15适配教师，不能把旧35.17%当新教师保证。
同画派教师保持V15输出；Adapter同画派不硬旁路，用实测检验是否学会不改。

训练源：真实训练池每画派100个路径ID（500源）×5目标=2500监督对；原FM路径ID-holdout每派20个（100源）×5=500对。
实际数据审计：500训练路径对应497幅画；Test750源与训练/留出均零交集。训练有12幅、留出有3幅参考重合，因此本实验不满足参考画作独立。2幅画跨训练/留出重复，已从留出拟合诊断剔除，剩98源/490对/392cross。完整名单见runs/v16/data_audit.json；按NFC规范化的最终双下划线分量审计，未做像素哈希。权重与原manifest保留，不把事后清理描述成训练前筛选。
V15冻结生成z_base，教师提供实际delta；Adapter约0.25M，输入z_base、c1_gentle(z_src)、目标标签，最后卷积零初始化。
Adapter参数放在`src/model/edit_adapter.py`，不复制V15网络。

损失：`MSE(delta_pred,delta_teacher) + .1*MSE(avgpool4(delta_pred),0) + .01*MSE(delta_pred,0)`。
内容项只约束新编辑的低频变化，不回归源latent，避免直接鼓励“复制源图”。
训练1500updates、effective batch32（8×4）、AdamW lr3e-4 cosine至5%、seed20261116。
保留训练/holdout相对拟合误差、teacher非零率、接受率、残差RMS、方向余弦。
无教师质量门控，无提前取消学生。

全750对照：V15 base、V15+teacher、V15+adapter、V15+wrong-label adapter。
wrong只改变Adapter的label，底座输出、内容与噪声相同。

## C：冻结S10的RGB目标反演

先重新解码保存的fp16 z，作为C自己的精确起点；不能把之前RGB量化/latent存储误差算成H2不可表达。
H2使用V11b原checkpoint_100，512px RGB两级正交Haar，只改H2、保留H1/LL2（clamp后会有少量泄漏，实测记录）。

| 目标 | tau0 | K | alpha | edge_decay | RMS clip |
|---|---:|---:|---:|---:|---:|
| default | .6 | 4 | .5 | .5 | 1.5 |
| strong | .3 | 8 | 1 | .3 | 3 |

strong另有wrong-label控制：相同每任务eps和**正确目标sigma**，仅改变网络label，避免sigma变化混入风格对照。
原H2函数的generator参数未实际传给噪声，此处显式每任务seed20261116+task_index，确保与batch划分无关。

反演变量为scaled latent的增量，S10全部参数冻结；通过官方decode_s10新增可微选项反传，不绕过缩放/pqc。
主实验最小化`MSE(D(z+delta),x_H2)+lambda*MSE(delta,0)`，lambda=.001，只对强H2做完整750反演。
默认H2反演和强目标lambda=0只保留前28任务的补充数值诊断，不能当全750结论；这是计算成本收紧，不是按质量早停。前28任务的强H2主实验规格完全相同，可直接复用。
Adam lr=.002，40步，每图保留目标函数最优点；记录0/10/20/40的损失，支持按完成批恢复。
3060正式反演batch4、S10保持原生contiguous布局，起点/反演同一分区。初始batch2实测三种反演约33秒/2图，记录保留于`runs/v16/pilot_C_batch2`；channels-last实测约93秒/4图，反而更慢，已撤回并保留于`runs/v16/pilot_C_channels_last`。预热样本不混入正式结果。
S10 half权重路径采用固定loss-scale1024避免过小输入梯度下溢，latent参数/Adam为float32。

报告RGB/H2残差、能量加权吸收率、残差方向余弦、低频/H1泄漏及latent位移。
全750独立评分：C_base、H2 default/strong/strong-wrong、强H2反演。默认H2反演/lambda0的部分结果仅补充数值诊断。
跨输出对比用150源图聚类配对bootstrap2000次，重复簇保留计数，无多重校正。

**解释边界**：这给出有限步非凸反演实际可达到的吸收率，不是Jacobian像空间的精确投影，也不是S10不可表达的数学证明。
正则、有限步、局部最优和FP16精度都可能限制拟合；只有稳定收敛且控制充分后才能讨论强结构结论。
H2本身若没有独立风格收益，也不能用“能/不能拟合其残差”直接宣传更好的风格迁移。

## 代码与运行

```powershell
# 3060原生Python，保持SSH前台；具体日志和环境由preflight.json锁定
python -u -m exp.v16.run --pool I:/Latent_Style/pseudo_gt_24h/orig_mean_fp16
```

执行前只做必要canary：实际S10输入梯度有限非零、可微/推理forward完全相同、Adapter初始输出严格为0、750任务覆盖正确。
阶段失败直接抛错，状态/日志保留；不扩大扫参、不另训Scaling、不修改历史实验。
