# 本地论文模型实现

这是 ShuttleHitFormer 与 DSTA 的论文公式实现、显式本地修正版和可运行训练入口。没有作者训练权重；本地模型结果不能直接称为论文复现成绩。默认网站识别仍保留既有模型。

## 代码与输入

| 模型 | 实现 | 输入与输出 |
|---|---|---|
| ShuttleHitFormer | `shuttlehitformer.py`：关节注意力、三层跨模态交互、非对称门控、三层 Transformer、CE＋对比损失 | `B×8×90` → no-hit/top/bottom；第四帧为参考帧，逐帧扫描、0.5阈值、8帧NMS |
| DSTA | `dsta.py`：Eq.1–2适配器＋官方 OpenTAD VideoMAE-B＋官方 ActionFormer | 独立RGB → 时段＋类型；冻结VideoMAE基础权重，训练适配器和检测头 |
| 训练/推理 | `run.py` | 数据准备、训练、保存、重载、显式实验推理 |
| 输入转换 | `data.py` | 90维顺序、真实RGB采样、公开训练特征转换、个人标注短训转换 |
| 结构诊断 | `diagnose_grounding.py` | 固定球场/球网时Eq.20的退化实验 |
| 完整窗口检查 | `check_full_window.py` | 768帧RGB经过完整DSTA路径；未训练的头不用于质量评价 |

SHF90维顺序：top/far COCO17 XY（34）→ bottom/near（34）→ court6 XY（12）→ net4 XY（8）→ shuttle XY（2）。坐标按原图宽高分别归一化。court顺序TL/TR/BR/BL/左边线中点/右边线中点；net四点由调用者给出一致顺序。球网顶部不能从地面单应矩阵推算。缺失网点仅在明确标记的缺模态试验中置零。

## 环境和上游

在`04_原型`执行，使用既有`.venv-vision/bin/python`。本机版本：Python3.12、torch2.14.1、torchvision0.29.1、NumPy2.5.3、SciPy1.18.1、OpenCV5.0.0。MMEngine0.10.7/MMCV-lite2.1.0位于现有`vendor/racketvision_deps`，由隔离worker加载。没有修改系统Python或全局安装包。

- [ShuttleHitFormer论文](https://doi.org/10.1587/transinf.2026EDP7100)
- [DSTA论文](https://arxiv.org/abs/2605.23355)
- [AdaTAD官方入口](https://github.com/sming256/AdaTAD)及[OpenTAD官方实现](https://github.com/sming256/OpenTAD)，Apache2.0许可证保存在vendor/OpenTAD。
- [ActionFormer官方实现](https://github.com/happyharrycn/actionformer_release)，MIT许可证保存在vendor/actionformer_release。
- [VideoMAE](https://github.com/MCG-NJU/VideoMAE)基础权重由OpenTAD官方配置所链接的OpenMMLab地址下载，`models/paper_reproductions/videomae_b_k400.pth`；不是DSTA羽毛球权重。

OpenTAD的骨干前向、注意力和FFN保持上游实现，仅替换Adapter并提供MMAction类型别名/确定性位置编码辅助函数。ActionFormer保留上游分类、回归、标签分配和FPN；将原本`no_grad`的预计算特征padding改为可微padding，否则无法训练RGB骨干中的适配器。Mac的1D NMS用PyTorch CPU实现与上游相同的区间IoU和soft-NMS公式，无需OpenMP扩展。

## 首轮小试的历史命令（当时版本）

```sh
.venv-vision/bin/python -m unittest -v test_paper_reproductions test_research_integration test_court_review

.venv-vision/bin/python -m paper_reproductions.run prepare-shf --out '../03_数据/论文模型复现/2026-10-29/shf_data' --max-rallies 3
.venv-vision/bin/python -m paper_reproductions.run train-shf --manifest '../03_数据/论文模型复现/2026-10-29/shf_data/manifest.json' --out '../03_数据/论文模型复现/2026-10-29/shf_run' --epochs 8

.venv-vision/bin/python -m paper_reproductions.run prepare-dsta-smoke --out '../03_数据/论文模型复现/2026-10-29/dsta_data' --per-split 4
.venv-vision/bin/python -m paper_reproductions.run train-dsta --manifest '../03_数据/论文模型复现/2026-10-29/dsta_data/manifest.json' --out '../03_数据/论文模型复现/2026-10-29/dsta_run' --epochs 2

.venv-vision/bin/python -m paper_reproductions.check_full_window --video '../03_数据/个人录像/dcee6f11ac8f416d9713afea9d3a33e4.mp4' --out '../03_数据/论文模型复现/2026-10-29/dsta_chunk48_execution.json'
```

完整SHF数据转换去掉`--max-rallies 3`即可读取公开的145个训练回合；按比赛1–18训练、19–21验证、22–23测试，原作者test_match文件不动。这不是SHF原论文37场比赛划分。首次小试只用了3个训练回合，不能把其结果当作模型能力上限。

`shuttlehitformer.json`保留论文Eq.20。`shuttlehitformer_residual_ablation.json`显式添加交互残差，是本地变体。同一场地内，论文写出的纯court/net cross-attention在K/V随时间恒定时抹除动作信息；诊断脚本和单元测试记录了这个现象。残差改善了信息通路，8轮小试两版F1仍为0，不能声称提升准确率。

DSTA默认按论文文字使用48帧attention chunk；另存`dsta_adatad_chunk16.json`对应上游AdaTAD的16帧/48个chunk。论文对chunk的描述存在解释空间，两种配置均实测过768帧前向。主配置的瓶颈N=192、12层插入位置及ActionFormer头细节采用可配置选择，论文没有提供完整配置。当前适配器参数5,003,148，未匹配论文表中4.017M，不能宣称逐参数复现。RGB窗口768表示768个已采样帧、在25fps轴上间隔4，覆盖约122.88秒；这是当前明确选择的时间解释。

DSTA本地短训为了验证梯度，只用32帧、16帧chunk、128维检测头、灰底4片段训练、蓝底4片段验证。标签词表8类，实际这批正例覆盖4类。区间来自人工确认击球的前9/后9帧，共19帧弱标注，其他未标动作可能存在；不计算accuracy或mAP。

## 实验推理

```sh
.venv-vision/bin/python -m paper_reproductions.run infer --checkpoint '../03_数据/论文模型复现/2026-10-29/shf_run/best.pt' --input '../03_数据/论文模型复现/2026-10-29/shf_data/match22_1_07_02.npz' --out /tmp/shf_prediction.json --experimental
```

SHF输入NPZ包含`features[N,90]`；亦支持JSON传`top,bottom,court,net,shuttle,normalized,width,height`。DSTA推理输入原视频路径，`--start`指定实验窗口起点；配置中的25fps/stride/长度决定取帧，输出区间换回原视频秒数。当前解码器以固定帧率录像为前提，时间索引写入结果。

`research_adapters.py`已连接这两个本地worker，调用者需提供checkpoint；小试权重必须显式`experimental=True`，返回`experimental_model_inference`。网站默认仍显示“论文结构已本地实现，实验权重尚未验证”。不提供随机权重的默认分类结果。

完整训练manifest的每个row需有独立`split`、样本NPZ路径和来源身份。SHF NPZ为features[N,90]/labels[N]；DSTA NPZ为已归一化rgb[3,T,160,160]/segments[K,2]/labels[K]，segments使用输出feature网格上的半开区间。样本须按比赛/源视频划分并核对完整标注覆盖；不能把漏标动作直接当作背景。训练权重、优化器、配置、源manifest哈希都会保存，DSTA只保存适配器/检测头并引用基础权重及SHA256。

完整训练前仍需补齐：SHF真实球网坐标与可核验的原始配置；DSTA完整类别次序、时段标注、训练集和未披露超参数。作者级准确率复现尚未完成。


## SHF训练问题的修正版本

推荐实验配置是 `configs/shuttlehitformer_aligned_streams.json`，这是本地改版：

- `grounding_residual=True`：场地注意力保留球员交互残差，固定场地不会抹除动作。
- `stream_residual=True`：两个球员分别保留对应帧姿态，最终再保留对应帧球坐标编码。当球轨迹静止或缺失后插值时，前级注意力不会把姿态时间顺序混成同一个结果。
- `use_net=False`：模型的context层只读取12维真实场地坐标，完全不读取预留网点通道。JSON可传net=null或省略net；只有这个明确不使用网点的版本允许省略，原论文配置仍要求真实网点。

修正版明确使用 `Ft=CrossAttn(Zs,Zt)+Zt`、`Fb=CrossAttn(Zs,Zb)+Zb`、`F=CrossAttn(Fp,Zc)+Fp+Zs`。原公式开关默认关闭，旧模型与历史结果继续保留。

训练器现在默认：验证事件F1选权重，F1相同再比较类别均衡CE；同时保存best.pt和last.pt；最少20轮及300次更新后才允许patience15早停；每轮重新抽负例。`--class-balance`只从训练样本计算类别权重，减少全判未击球；`--selection paper_recall`可恢复原论文的召回优先选择。训练日志还记录帧级混淆矩阵和各类损失。

小样本检查使用18个平衡训练窗口，目标为训练集100%分类及CE<0.1，用于确认能学会，不作为泛化成绩。真正的事件指标由比赛级验证/测试统计；F1不等于accuracy，也不等于击球类型分类准确率。

```sh
.venv-vision/bin/python -m paper_reproductions.run prepare-shf --out '../03_数据/论文模型复现/2026-10-29_修复/full_data'
.venv-vision/bin/python -m paper_reproductions.overfit_shf --manifest '../03_数据/论文模型复现/2026-10-29_修复/full_data/manifest.json' --config paper_reproductions/configs/shuttlehitformer_court_only.json --out /tmp/shf_memorization --steps 1600
.venv-vision/bin/python -m paper_reproductions.run train-shf --manifest '../03_数据/论文模型复现/2026-10-29_修复/full_data/manifest.json' --config paper_reproductions/configs/shuttlehitformer_aligned_streams.json --out /tmp/shf_training --epochs 100 --class-balance
```

`--warm-start path/to/best.pt`会读取已有全部网络参数，保留已学知识，并明确开启新的优化过程。严格state_dict加载拒绝不兼容的网络形状。当前aligned_run从full_run/best.pt继续训练，旧权重没有覆盖。

`calibrate_shf.py`只在验证比赛上选一个全局阈值，先写入冻结记录，再计算职业/业余结果；包括不读取视频的8帧固定间隔基线。专业与业余来源数据此前参与过项目开发，后续修正后的分数属于开发实验，不是全新独立盲测。公开特征含人工或人工关联测量，不能外推上传视频端到端准确率。

完整原论文模态/成绩复现仍缺真实球网标注与作者未公开细节；本地修正版通过明确移除网点依赖继续训练，不会把零占位符声称为测量值。

最新结果和限制见 [2026-10-29_ShuttleHitFormer训练与信息流修复结果.md](/Users/cyt/Desktop/羽毛球论文/05_研究记录/2026-10-29_ShuttleHitFormer训练与信息流修复结果.md)。最终修正版公开特征测试：职业F1 91.5%，业余F1 55.8%，±3帧；不可外推为上传原视频的端到端准确率。
