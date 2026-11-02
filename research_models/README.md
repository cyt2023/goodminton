# 四篇论文的增量接入

现有上传、两类录像入口、场地确认、人工标注和 BST 流程保留。默认新增球与球拍感知，后续缺模型的环节明确显示不可用。

- RacketVision：官方仓库与官方 Hugging Face 权重，版本与文件哈希见 `models/racketvision/manifest.json`。独立子进程使用 `vendor/racketvision_deps`；原有 OpenMMLab 复现环境不改动。
- ShuttleHitFormer：接口输入是人体、球轨迹、场地和球网几何。未找到可核实的官方推理实现/训练权重；没有用启发式或 GRU 冒充它。当前找拍仍显示“找拍基线候选”。
- DSTA：独立 `infer(rgb_video_path=...)` 接口，未找到训练后的 DSTA/VideoMAE/ActionFormer 检查点；目前只接收声明来源与时间基准的外部结果。通用 VideoMAE 权重不能替代 DSTA。
- ERA 六项指标：实现论文 Eqs.15–32、图11–12隶属函数、表1–2的 Mamdani 规则，提供运算与规则追踪。实际视频缺少三维拍面、手指点、真实速度尺度及经过验证的分技术参考，因此不填总体分数。可见二维量单列，不冒充六项质量分数。

`settings.json` 控制感知后端、去重和时间匹配容差。`label_map.json` 是唯一 DSTA 名称/类别映射；目前没有经确认的29类数字ID顺序，未映射的原标签保留并输出 unknown。

## 启动

```sh
cd /Users/cyt/Desktop/羽毛球论文/04_原型
python3 server.py
```

浏览器进入 http://127.0.0.1:8765/ 。新上传仍先确认或跳过场地标定，再识别；录像列表新增“查看分析”。已有录像可在分析页运行新感知，原预测和人工标签保留。

## 真正运行 RacketVision

```sh
.venv-vision/bin/python racketvision_adapter.py --source /absolute/video.mp4 --output /Users/cyt/Desktop/羽毛球论文/03_数据/本次RacketVision --device auto
```

输入须位于本项目数据目录，输出也须在 `03_数据`。auto 依次选择 CUDA/MPS/CPU。保存完整原帧索引、解码时间、原始球检测、每个球拍的5点与分数。TrajPred 的官方权重已缓存，本次没有启用预测；预测点不会当作真实观测或速度依据。

已有项目任务可使用：

```sh
.venv-vision/bin/python research_worker.py --job J_e23c1dd9297d4bfca7c91d1d95065b68
```

正常使用应通过网页排队，以共享模型执行锁。图层开关、事件选择和报告导出不重复运行模型。

## 外部结果格式

分析页的“导入”可接收 JSON。`component` 为 shuttlehitformer/dsta，`kind` 为 imported_prediction/manual_annotation；必须声明原视频 SHA256、真实模型/人工标注来源、时间单位。帧单位必须给 fps 和0/1起始索引；DSTA必须声明区间末端为 inclusive/exclusive。

导入不会写入人工训练标签。错误来源直接拒绝；不合格行保留原因；全无有效行时不覆盖结果。官方模型本身仍显示 unavailable，同时注明已有导入数据。

```json
{
  "component": "dsta",
  "kind": "imported_prediction",
  "source_sha256": "当前视频的真实SHA256",
  "model_provenance": {"name": "实际模型及权重版本"},
  "time_base": {"unit": "frames", "fps": 25, "frame_index_origin": 0, "interval_end": "exclusive"},
  "predictions": [{"id": "外部事件编号", "start": 20, "end": 30, "label": "Smash", "confidence": 0.9}]
}
```

ShuttleHitFormer 导入使用 `time` 字段。模型击球方若未验证不提升为可靠身份，保留在 candidate_player_id；明确人工标注的 near/far 可作为人工归属。

## 复现边界

BallTrack 的官方 inference.py 输入到 i-1，却把结果标成 i。适配器按官方训练 Uball 的 [i-3,i] 窗口对齐，单独记录这处修改。BGR/JPEG95、512×288、除255、模型图和解码规则保留；采用64帧均匀样本计算当前视频背景，不等同作者整场背景。手机录像采用顺序解码，避免随机 seek 漏读尾帧。

Mac 的 MMCV NMS 改用 torchvision CPU，MMEngine 的 MPS 索引转 CPU；网络与权重不更换。记录适配，不宣称旧 CUDA 位级一致或论文准确率复现。

浏览器播放使用 H.264 衍生副本，保持原分辨率和原帧时间。编码后逐帧校验数量与PTS，最大误差要求≤2ms；模型始终分析原视频，不把播放转码当成模型输入。

本地 Qwen3-VL-4B-Instruct-4bit 按需选择与真实测量绑定的回看重点。输出必须引用现有测量键；数值与解释由程序生成，避免无依据的三维、发力或动作优劣结论。默认文字为明确标注的证据模板。

## 检查

```sh
python3 test_research_integration.py
python3 test_court_review.py
```

真实运行结果、样例入口与浏览器截图见 `03_数据/四论文集成/2026-10-29`；已有蓝底整段结果见 `03_数据/研究集成/J_e23c1dd9297d4bfca7c91d1d95065b68`。

来源：
- https://github.com/OrcustD/RacketVision
- https://huggingface.co/linfeng302/RacketVision-Models
- https://doi.org/10.1587/transinf.2026EDP7100
- https://arxiv.org/abs/2605.23355
- https://doi.org/10.3934/era.2026333
