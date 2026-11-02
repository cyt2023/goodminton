"""Write the human-readable report from the completed hit-model comparison."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / '03_数据/找拍模型对照/2026-10-22'
REPORT = ROOT / '05_研究记录/2026-10-25_找拍模型本地对比结果.md'
LABELS = {
    'current_visual_rules': '当前视觉规则',
    'current_production_fusion': '当前网站找拍流程',
    'author_gru_naive': '作者GRU＋naive后处理',
    'author_gru_dp': '作者GRU＋动态规划',
    'adapted_hitnet': '第三方Adapted-HitNet＋已知球场角点',
    'adapted_hitnet_no_court': 'HitNet无球场输入消融',
    'swingnet': 'SwingNet原始RGB输入诊断',
}
MAIN = ['current_production_fusion', 'author_gru_naive', 'author_gru_dp', 'adapted_hitnet']


def read(path):
    return json.loads(Path(path).read_text())


def percent(value):
    return '—' if value is None else f'{100*value:.2f}%'


def table(groups, models=MAIN):
    lines = ['| 方法 | 人工击球 | 提出事件 | ±0.1秒命中 | ±0.1秒 Precision / Recall / F1 | ±3帧F1 | ±0.2秒F1 |',
             '|---|---:|---:|---:|---|---:|---:|']
    for model in models:
        row = groups.get(model)
        if row is None:
            continue
        m, three, loose = [row['metrics'][key] for key in ['0.1s', '3frames', '0.2s']]
        prf = ' / '.join(percent(m[key]) for key in ['precision', 'recall', 'f1'])
        lines.append(f"| {LABELS[model]} | {m['annotations']} | {m['predictions']} | {m['matched']} | {prf} | {percent(three['f1'])} | {percent(loose['f1'])} |")
    return '\n'.join(lines)


def subset_groups(result, predicate):
    groups = {}
    for model in MAIN:
        rows = [r for r in result['rows'] if r['model'] == model and predicate(r)]
        if not rows:
            continue
        groups[model] = {'metrics': {}}
        for tolerance in ['0.1s', '0.2s', '3frames']:
            values = [r['metrics'][tolerance] for r in rows]
            n = sum(v['annotations'] for v in values)
            p = sum(v['predictions'] for v in values)
            k = sum(v['matched'] for v in values)
            groups[model]['metrics'][tolerance] = {'annotations': n, 'predictions': p, 'matched': k,
                'precision': k/p if p else 0., 'recall': k/n, 'f1': 2*k/(p+n)}
    return groups


def main():
    result = read(OUT / 'evaluation.json')
    validation = read(OUT / 'author_validation_checks/result.json')
    groups = result['groups']
    initial = read(OUT / 'data_manifest.json')['cases']
    extension = read(OUT / 'data_manifest_extension.json')['cases']
    assert all(any(r['case'] == c['id'] and r['model'] == model for r in result['rows'])
               for c in initial + extension for model in ['current_production_fusion', 'author_gru_naive', 'author_gru_dp', 'swingnet', 'adapted_hitnet_no_court'])
    assert all(any(r['case'] == c['id'] and r['model'] == 'adapted_hitnet' for r in result['rows']) for c in extension)
    known = groups['personal']
    personal_table = ['| 方法 | ±0.1秒覆盖已确认记录 | ±0.2秒覆盖已确认记录 | 全片预测事件数 |', '|---|---:|---:|---:|']
    for model in MAIN[:3]:
        a, b = [known[model]['metrics'][k] for k in ['0.1s', '0.2s']]
        personal_table.append(f"| {LABELS[model]} | {a['matched']}/{a['annotations']}（{percent(a['known_positive_coverage'])}） | {b['matched']}/{b['annotations']}（{percent(b['known_positive_coverage'])}） | {a['predictions']} |")
    sparse = validation['event_metrics']['3']
    author_loose = validation['event_metrics']['6']
    signed = [m['signed_error_frames'] for row in result['rows']
              if row['domain'] == 'professional_extension' and row['model'] == 'current_production_fusion'
              for m in row['metrics']['0.2s']['matches']]
    early = sum(x in [-2, -3] for x in signed)
    appendix = ['| 诊断输入，不能作为完整作者方法排名 | 新职业101拍 ±3帧F1 | 外部业余55拍 ±3帧F1 |', '|---|---:|---:|']
    for model in ['adapted_hitnet_no_court', 'swingnet']:
        appendix.append(f"| {LABELS[model]} | {percent(groups['professional_extension'][model]['metrics']['3frames']['f1'])} | {percent(groups['external_amateur'][model]['metrics']['3frames']['f1'])} |")
    text = f'''# 找拍模型本地对比结果

完成日期：2026-10-25。测试数据与初始协议在2026-10-22建立，10月25日续跑职业扩展与汇总。本轮没有训练新模型、调整测试阈值、修改用户标注或替换网站默认模型。

## 1. 得到的结论

- **职业标准视角已有可用改进候选。** 新增5段职业录像101拍，作者GRU＋naive在±3帧下F1为{percent(groups['professional_extension']['author_gru_naive']['metrics']['3frames']['f1'])}，当前流程为{percent(groups['professional_extension']['current_production_fusion']['metrics']['3frames']['f1'])}；它们使用同一套自动姿态和球轨迹。说明找拍后处理本身确实影响结果。
- **普通录像尚未解决。** 在完整标注的外部业余55拍上，GRU的收益依赖时间容差；在我们的部分人工确认记录上，独立GRU的定位覆盖低于现流程。不能把职业成绩外推给个人视频。
- **同一个GRU的后处理选择影响很大。** 本轮保留作者naive与动态规划原参数，没有根据本批结果调参数。动态规划并没有统一胜过简单阈值方案。
- **Adapted-HitNet在作者提供的输入上能够工作，替换为当前共享前端后的成绩明显较弱。** 这提示输入与适配值得优先调查；当前还不能单独归因于YOLOX、RTMPose、轨迹或分类网络。

## 2. 实际测试了什么

| 数据 | 录像/回合数 | 找拍真值 | 用途与限制 |
|---|---:|---|---|
| 新增职业源：Indonesia Open2019 | 5 | 完整101拍 | 来自同一场比赛，自动YOLOX/RTMPose/TrackNetV3前端；GRU训练重合未知 |
| 既有职业match35回合 | 1 | 完整20拍 | 已经检查过；Adapted-HitNet训练重叠，单列结果 |
| 外部业余china2/yewtee | 7 | 完整55拍 | 两个录像来源，原作者逐帧标注 |
| 我们自己的两段录像 | 2 | 74条已确认正拍定位记录 | 候选确认子集，含3组未核实近邻重复；无法计算全片Precision、Recall、F1 |
| 东京奥运决赛开局录像 | 2 | 没有完整找拍真值 | 已输出模型事件，只能回看，不能报告准确率 |

共17段输入，13段完整找拍标注合计176拍，另有个人确认子集。所有完整标注拍均保留，包括模型缺少前序上下文的片头击球。

输入中的姿态、球轨迹均由本地自动模型产生。额外球场角点只用于需要它的HitNet，属于作者或人工标定输入。作者domain_X中的球坐标与公开人工球坐标逐点相同，因此没有把X中的球/姿态列冒充自动前端。

## 3. 新增职业录像101拍：共享自动前端的找拍比较

{table(groups['professional_extension'])}

**容差的解释：** 这批视频为29.97fps，3帧约100.10毫秒，严格±100毫秒只允许到2帧。因此±0.1秒和±3帧可能差很多。两种口径都在初始协议中预先声明，结果同时保留，不能只展示其中较有利的一列。评价仅匹配击球时间，不把击球者或球种正确率混入F1。

这5段来自一场比赛；GRU是否曾训练该源未知，不能称独立跨比赛测试。根据第三方训练名单与赛事身份核对，该源不在Adapted-HitNet声明的24场训练源中，但完整checkpoint训练历史不能由我们独立证明。

## 4. 外部业余完整55拍

{table(groups['external_amateur'], MAIN[:3])}

完整球场角点仅在china2四段有可用作者标定；HitNet与其他方法在这**相同33拍**上的对照如下，不能把它的33拍成绩与上表55拍直接排名。

{table(subset_groups(result, lambda r: r['domain'] == 'external_amateur' and r['case'].startswith('match_china2')))}

china2源文件有重复/倒退PTS，依照原作者帧索引和固定25/30帧节律计分。没有修改预测或源视频来消除时间错误。

## 5. 我们自己的录像

{chr(10).join(personal_table)}

这些是**已确认记录覆盖率**，不是全片召回率。用户当时只检查了系统提供的候选；未标注处可能还有真击球，不能将额外预测记为误报。既有局部接触时序校准还使用过个人源，因此当前流程在这组的分数也不是独立泛化证据。

个人视角没有可靠完整外场四角，本轮没有虚构角点去报告完整HitNet成绩；无球场输入的单独消融已运行并保留。

## 6. 既有职业20拍：全部保留

{table(groups['professional'])}

这个回合属于Adapted-HitNet训练源，不能用它证明泛化。此前已经检查过，也不能充当新的盲测。这里的时间匹配数量与同时要求人物或球种正确的旧联合指标不同。

## 7. 用作者输入检验Adapted-HitNet本身

- 全部6个作者验证比赛、470回合、136597帧、4806个人工击球，没有按成绩筛回合。
- 作者12帧窗口分类准确率：{percent(validation['window_classification']['accuracy'])}；窗口Macro-F1：{percent(validation['window_classification']['macro_f1'])}。
- 把连续命中区间按固定中点转成一次击球，±3帧：{sparse['matched']}/{sparse['truth']}匹配，Precision {percent(sparse['precision'])}，Recall {percent(sparse['recall'])}，F1 {percent(sparse['f1'])}。
- ±6帧事件F1：{percent(author_loose['f1'])}。

这是作者验证集与作者预提取输入，且用于作者模型开发；没有原视频可独立核实真实帧率，±3/6帧只能记为名义30fps的±0.1/0.2秒。它说明发布权重不是完全失效，**不能把这组较高成绩当作我们的原视频处理成绩**。作者窗口标签覆盖末6帧，与稀疏接触帧分类是不同任务。

## 8. 这次定位到的具体问题

1. 当前流程在新增职业源的±0.2秒匹配中，{early}/{len(signed)}个时间中心提前2–3帧；GRU＋naive的中心更接近作者真值。需要检查时间对齐与中心定义，不能直接在测试数据上加固定偏移后当作独立进步。
2. 同样的YOLOX/RTMPose/TrackNet输入，换找拍后端就能改变职业成绩，说明通用模型并非唯一原因。但共享缓存仍含现有身份跟踪、姿态缺失和错球点，独立GRU没有消除这些上游问题。
3. 作者GRU依靠14帧窗口，动态规划另假设30fps、相邻拍超过0.5秒且双方交替；业余25fps、连续多回合与回合外动作可能违反这些条件。它输出区间，需要固定中点转换，不能声称模型原生输出了精确接触帧。
4. HitNet依赖球场几何、回合内归一化与训练输入分布；个人完整录像含多个回合、缺球场角点，是额外部署条件。作者输入高分与本地替换输入低分提示这些差异值得核对，尚没有同源逐项替换证据能唯一定位主因。
5. 个人标注仍覆盖已出现候选，下一轮需要先完整核查短回合的每一拍，才能真正比较漏拍与误报。继续只给旧候选纠正球种，无法测出漏掉的击球。

## 9. 哪些完成了复现，哪些仍未完成

| 方法 | 当前状态 |
|---|---|
| kwyoke GRU | 作者权重、前向、naive/动态规划后处理已独立运行；前端共享本地模型，非作者完整前端复现 |
| raneeng Adapted-HitNet | 第三方已训练权重与推理已运行，另完成全部作者val检查；不是原MonoTrack权重 |
| SwingNet | 网络和权重已strict加载并跑17段；关键2023 RT去背景预处理缺失，原始RGB诊断不算完整复现 |
| 原MonoTrack HitNet | 仍缺原作者checkpoint |
| Automated Hit-frame2023 | 仍缺核心方向Transformer与球场权重 |
| Sensors2024 | 仍缺专门挥拍检测器权重与完整官方实现 |
| DSTA2026 | 仍未确认羽毛球专用checkpoint与实现 |
| ShuttleHitFormer2026 | 全文已读，作者按±3帧报告事件F1；未公开可直接推理代码和权重，未完成模型复现 |

未找到资产不表示作者从未提供给其他人；本轮没有联系作者。详细一手来源见[来源核查]({ROOT}/05_研究记录/2026-10-22_找拍模型来源核查.md)。

### 诊断附录

{chr(10).join(appendix)}

这两行都有输入条件缺失，不能据此评价作者完整方法的优劣。SwingNet后续公式近似RT模式虽有实现，本轮没有增加新的近似预处理实验来替代缺失的原始方法。

## 10. 下一步与可复核产物

优先保留GRU＋naive作为职业录像候选，先核对时序中心与上下文；个人录像先完整标注几段短回合，再检查身份保持、球轨迹和缺失姿态。现有证据不支持把HitNet或GRU直接统一替换网站默认流程，也不足以宣称普通录像达到80%。BST球种分类及现有训练知识本轮保留。

- [完整逐拍评价]({OUT}/evaluation.json)，保存{len(result['rows'])}组录像×方案输出评价；6个缺球场条件单独记为不可用，未填零冒充完成。
- [初始数据与人工真值清单]({OUT}/data_manifest.json)、[新增职业清单]({OUT}/data_manifest_extension.json)。
- [初始协议]({OUT}/protocol.json)、[扩展协议]({OUT}/protocol_extension.json)。
- [作者验证结果]({OUT}/author_validation_checks/result.json)、[GRU/HitNet数值一致性核对]({OUT}/reproduction_checks/README.md)。
- [统一运行与评价脚本]({ROOT}/04_原型/benchmark_hit_models.py)、[报告生成脚本]({Path(__file__).resolve()})。

数值核对：GRU对原Keras前向最大概率误差约1.19e-7，动态规划与作者实现标签一致；第三方HitNet前向、窗口特征和聚合与源实现核对通过。源视频、原标注及受保护模型文件通过已声明哈希检查。本轮没有训练或更换生产模型。
'''
    REPORT.write_text(text)
    print(REPORT)


if __name__ == '__main__':
    main()
