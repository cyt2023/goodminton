"""Summarize the actual author-pipeline replication and disclosed gaps."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT/'03_数据/GRU作者流程复现/2026-10-25'
REPORT = ROOT/'05_研究记录/2026-10-26_GRU作者流程复现结果.md'


def read(path):
    return json.loads(Path(path).read_text())


def percent(value):
    return f'{value*100:.2f}%'


def main():
    gt, auto, raw = [read(OUT/path/'result.json') if path else read(OUT/'raw_video_result.json')
                     for path in ['author_features', 'filtered2_features', '']]
    table = ['| 作者输入/配套权重 | 后处理 | 职业28回合365拍：±3 / ±6帧F1 | 业余35回合331拍：±3 / ±6帧F1 |',
             '|---|---|---|---|']
    for label, result, am_group in [('GT域/原shuttlepose', gt, 'amateur_singles_author_released'),
                                     ('filtered2/配套filtered2权重', auto, 'amateur_singles_author_test')]:
        for mode in ['naive', 'max']:
            pro = result['groups']['professional_author_test']['point_metrics'][mode]
            am = result['groups'][am_group]['point_metrics'][mode]
            table.append(f"| {label} | {mode} | {percent(pro['3']['f1'])} / {percent(pro['6']['f1'])} | {percent(am['3']['f1'])} / {percent(am['6']['f1'])} |")
    raw_table = ['| 原视频条件 | 人工标注 | 模型提出 | ±3帧匹配 | ±3帧F1 | ±6帧F1 |', '|---|---:|---:|---:|---:|---:|']
    for domain, title in [('professional', '职业完整回合'), ('external_amateur', '两段外部业余完整回合')]:
        m = raw['summary'][domain]['author_pose__author_ball__filtered2_legacy_layout__naive']['metrics']
        a, b = m['3frames'], m['6frames']
        raw_table.append(f"| {title} | {a['annotations']} | {a['predictions']} | {a['matched']} | {percent(a['f1'])} | {percent(b['f1'])} |")
    personal_table = ['| 个人片段与场地条件 | 已确认定位记录 | 提出事件 | ±3帧覆盖 | ±6帧覆盖 |', '|---|---:|---:|---:|---:|']
    for prefix, name, method in [('J_fe1', '灰地42–50秒，白线几何估计', 'author_pose__author_ball__filtered2_legacy_layout__naive'),
                                  ('J_e23', '蓝地45–53秒，全画面无标定消融', 'author_pose_full_image__author_ball__filtered2_legacy_layout__naive')]:
        row = next(row for row in raw['rows'] if row['case'].startswith(prefix) and row['method'] == method)
        a, b = row['metrics']['3frames'], row['metrics']['6frames']
        personal_table.append(f"| {name} | {a['annotations']} | {a['predictions']} | {a['matched']}/{a['annotations']} | {b['matched']}/{b['annotations']} |")
    cases = read(OUT/'raw_video_plan.json')['rows']
    pose_seconds = sum(read(Path(c['output_folder'])/'author_pose/metadata.json')['runtime_seconds'] for c in cases)
    ball_seconds = sum(read(Path(c['output_folder'])/'author_ball/trajectory.json')['runtime_seconds'] for c in cases)
    gray = OUT/'raw_video/J_fe1e058fc75e475d9c030caced6ec648_excerpt'
    blue = OUT/'raw_video/J_e23c1dd9297d4bfca7c91d1d95065b68_excerpt'
    text = f'''# GRU作者流程本地复现结果

2026-10-26。复现对象为kwyoke的职业/业余羽毛球击球检测公开项目及其ResNet基线。

## 完成到哪里

**作者网络、权重、原输入和原前端已在本机运行；作者保存的filtered2归档概率尚未精确对齐。个人蓝地录像仍缺可靠物理球场标定，已单列无标定部署条件。**

- 全部63组作者公开测试输入：职业28回合365拍，业余单打35回合331拍，共25299帧；GT域与filtered2配套条件分别运行，训练数据未混入此测试汇总。
- 原始录像试点5段1436帧：职业1个完整回合、外部业余2个完整回合、自己的两个8秒片段。样本范围在原前端击球预测前固定，不按结果挑片。
- 所有1436帧的原FasterR-CNN/HRNet检测、原TrackNetV2球轨迹已完成。个人灰地使用未人工确认的几何估计；蓝地仅完成全画面筛选消融。
- 原作者ResNet50基线也已恢复并运行，原权重strict加载，其结果与GRU对照保存在同一原视频结果中。

## 1. 用了哪些原作者组件

| 环节 | 本次实际运行 |
|---|---|
| 人体检测 | 原Faster R-CNN R50-FPN 1x COCO，047c8118权重，原0.6阈值 |
| 姿态 | 原HRNet-W48 384×288，314c8528权重，原仿射裁剪、热图解码与flip-test |
| 框架 | 作者notebook记录的MMCV1.6.2、MMDet2.25.3、MMPose0.29.0，隔离安装 |
| 选人 | 原球场多边形放宽15%、取最高检测置信度两人、右踝高低分近远、缺人沿用 |
| 球轨迹 | 原TrackNetV2的model906_30 HDF5，连续不重叠3帧RGB、512×288、0.5阈值、最大轮廓框中心 |
| 击球网络 | 原shuttlepose/150与原filtered2/150分别转换运行；14帧窗口、末6帧标签 |
| 后处理 | 原naive0.95阈值与max动态规划均保留；固定作者30fps参数，没有按本批分数调参 |
| RGB基线 | 原作者15epoch ResNet50三类权重、原人体裁剪/JPEG/归一化与后处理 |

Mac兼容替换明确保留：旧MMCV原生NMS/RoIAlign由torchvision CPU算子执行，卷积在MPS；原TrackNetV2 HDF5图通过Keras3/PyTorch执行。保留原H5中axis=-1的空间列BatchNorm，没有改成通道BatchNorm。没有宣称与旧CUDA/TF1二进制逐位相同。

## 2. 作者测试输入上的实际结果

{chr(10).join(table)}

上表采用完整一对一离散事件匹配，包含边缘击球。±3/6是原帧数，不假定所有录像均30fps。作者原来的区间IoU评价也已逐字运行并保存；其中所谓mAP是视频/球员Precision均值，另存在末区间遗漏和重复匹配行为，不能直接与上表F1比较。

**输入条件必须说明：** GT域含人工球坐标与人工框关联的HRNet姿态。filtered2姿态由作者自动筛选；职业球坐标仍是人工标注，业余球才来自作者TrackNetV2。两组球场角均为人工输入。GT域与filtered2同时改变输入和权重，表中差值不能全归因某个前端。

这次补齐了先前漏掉的filtered2.zip、配套filtered2/150 checkpoint和domain_probs_filtered2.zip。此前共享YOLOX/RTMPose的比较属于找拍头适配实验，本轮增加了作者原感知模型。

## 3. 找到一个确定的输入列约定错误

作者的combine_domain_into_x在filtered2数据中删除框坐标时没有删除bbox_conf，随后只取前34个标量。实际每人输入为：

```text
[bbox_conf, nose_x, nose_y, left_eye_x, …, left_ankle_y, right_ankle_x]
```

最后的right_ankle_y被截掉。它与正常17点xy不同。**filtered2 checkpoint学习的是这个legacy排列，复现时必须恢复它；如果要改成正常坐标，需要另行训练。**

本轮原始前端刚接入时仍送正常17点，属于实验接入偏差，现已新增独立filtered2_legacy_layout条件恢复作者实际输入。此前结果保留为正常COCO输入诊断，没有覆盖原产物。GT域的原shuttlepose使用正常COCO排列，不适用该legacy错位；网站旧默认分支不能因此被判定存在同一错误。

恢复后，与作者filtered2预存姿态的归一化MAE明显降低：职业near约0.2001→0.000301，yewtee far约0.3596→0.000847。新TrackNetV2与作者自动球坐标也接近：china2的281/282帧、yewtee的246/248帧在2像素内。没有用击球真值来调时移、阈值、角点或挑checkpoint。

## 4. 从原视频跑作者完整模型：小规模试点

原始FasterR-CNN/HRNet＋TrackNetV2＋filtered2配套权重＋legacy列排列，naive后处理：

{chr(10).join(raw_table)}

这只有1段职业和2段业余，不能替代上面的全部作者测试集。两段业余对应作者预存filtered2特征，本地预存输入推理也是分别1/7、1/5在±3帧内匹配，合计16个预测、2个匹配，F1同为14.29%；本机原前端逐拍统计与其一致。由此不能把这两段的低分全部归为我们换用了通用前端。

职业试点源的作者预存球坐标是人工输入；本轮原视频端使用自动TrackNetV2，输入条件不同，不要求两者逐值一致。完整逐项前端替换对照和ResNet结果均保存于raw_video_result.json，主表没有把正常坐标输入诊断算成忠实filtered2复现。

## 5. 你的录像实际跑到了哪里

{chr(10).join(personal_table)}

这里是对已有确认记录的时间覆盖，**不是全片准确率、全片召回率或球员识别率**。片段没有完整逐拍真值，灰地还含一组可能同拍的近邻定位记录。

### 灰地：可见白线能估计区域，选人仍有具体失败

用白色底线、疑似相距0.76m的双打长发球线及侧线恢复单应，远角画外外推；它是未人工确认的几何估计，镜头轻微移动也会影响结果。首帧选人图如下：红衣旁站者被选中，一名正在打球的黑衣球员没有入选。原流程取球场宽松区域内置信度最高两人，仍可能选错对象。

![灰地原作者选人结果]({gray}/author_selected_players_frame0.png)

因此，即使时间覆盖为5/6，也不能据此宣布正确分析了目标运动员。需要把击球时刻与球员身份同时核对。

### 蓝地：没有可靠四角，单列无标定条件

画面可见白带属于网顶，地面有多色交叠线且部分边界出画/遮挡；用户的“白线吧”不足以确认真实四角。未虚构物理场地标注，使用明确的全画面单位矩形消融，输入方式与作者yewtee占位矩形同类，但不视为真实场地标定。

![蓝地全画面条件的首帧选人]({blue}/author_selected_players_frame0.png)

蓝地物理球场标定仍未完成，它在结果中保留为未满足原法条件的一例。

## 6. 已知尚未精确对齐的地方

filtered2公开checkpoint推理与作者保存的概率文件仍不一致：24480窗口中3788个argmax不同，最大概率差约0.99944。独立Keras与NumPy新权重前向误差约1.79e-7，不能用本地浮点实现解释该差异；已核实公开目录没有其他filtered2 checkpoint或专属训练笔记本。历史checkpoint与归档概率是否来自同次运行，现有资料不能确定。

归档概率仅作reference另行评分，没有拿来替代本地推理、没有按标签反选模型。因此本轮可以报告可执行模型/前端复现与本地实测，**不能宣称作者历史报告或归档概率已被完全精确复现**。

## 7. 对下一步的实际意义

1. 已确认filtered2实验的特征布局失配，现已按源代码恢复。它解释部分接入差异；业余录像仍未稳定改善。
2. 原方法在大角度/旁站者/多球场录像上仍会面临选人问题。灰地示例已给出具体失败，不应只反复训练球种分类。
3. 下一步优先分离目标运动员与旁站者，并完整标注短回合的所有击球，连同人物身份一起评价。原系统的静态底线机位、分好回合和人工球场角假设，需要明确处理。
4. 本轮保留全部现有BST权重与用户标签，未重新训练或替换网站默认模型。复现代码和数据已留在项目，可继续做同源对照。

## 8. 产物与运行

本机1436帧原人体前端推理累计约{pose_seconds/60:.1f}分钟，原TrackNetV2累计约{ball_seconds/60:.1f}分钟，两阶段有并行，不能相加当作用户等待时间。这说明Mac可运行该方法，但其重型原前端明显慢于当前轻量前端。

- [作者GT域完整结果]({OUT}/author_features/result.json)、[作者filtered2完整结果]({OUT}/filtered2_features/result.json)。
- [原视频逐项对照]({OUT}/raw_video_result.json)：120组录像×输入/权重/后处理条件，均使用保存的真实推理输出。
- [原始输入对齐证据]({OUT}/raw_frontend_parity.json)、[字段解释补充]({OUT}/layout_interpretation_addendum.json)。
- [个人场地几何证据]({OUT}/personal_geometry_candidates.json)。
- [人体前端]({ROOT}/04_原型/reproduce_gru_author_pose.py)、[球前端]({ROOT}/04_原型/reproduce_gru_author_tracknet.py)、[RGB基线]({ROOT}/04_原型/reproduce_gru_author_resnet.py)。
- [统一运行脚本]({ROOT}/04_原型/run_gru_author_reproduction.py)、[本报告生成脚本]({Path(__file__).resolve()})。

复算已保存的原视频预测：

```sh
04_原型/.venv-vision/bin/python 04_原型/run_gru_author_reproduction.py evaluate
04_原型/.venv-vision/bin/python 04_原型/write_gru_author_reproduction_report.py
```
'''
    REPORT.write_text(text)
    print(REPORT)


if __name__ == '__main__':
    main()
