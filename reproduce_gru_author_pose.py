"""Run the GRU author's FasterRCNN-R50-FPN + HRNet-W48 pose frontend.

The model/config/checkpoint/geometry filter are retained. Legacy native MMCV
NMS/RoIAlign are bridged to CPU torchvision for this Mac. Detector tensors can
run on MPS. All original decoded frames are retained for hit-benchmark alignment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / 'models/gru_author_pose'
NAMES = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
         'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
         'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
         'left_knee', 'right_knee', 'left_ankle', 'right_ankle']


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))


def scatter_compat(data, target_gpus, **kwargs):
    """Unwrap the same DataContainers, then move tensors to CPU/MPS."""
    import torch
    from mmcv.parallel import DataContainer
    device = target_gpus[0]
    if device == -1:
        device = 'cpu'
    def move(value):
        if isinstance(value, DataContainer):
            return value.data[0] if value.cpu_only else move(value.data[0])
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, dict):
            return {key: move(v) for key, v in value.items()}
        if isinstance(value, list):
            return [move(v) for v in value]
        if isinstance(value, tuple):
            return tuple(move(v) for v in value)
        return value
    return [move(data)]


class AuthorFrontend:
    def __init__(self, device='mps', bbox_threshold=.6):
        from gru_author_pose_runtime import activate
        self.runtime = activate()
        import torch
        from mmdet.apis import init_detector
        from mmdet.datasets import replace_ImageToTensor
        from mmdet.datasets.pipelines import Compose
        from mmpose.apis import init_pose_model
        from mmpose.datasets import DatasetInfo
        import mmpose.apis.inference as pose_api
        self.device = device
        self.bbox_threshold = bbox_threshold
        torch.set_num_threads(2)
        # Only data placement is extended to MPS. The official preprocessing,
        # affine crop, HRNet heatmap decode and flip test remain in MMPose.
        pose_api.scatter = scatter_compat
        pose_api._pipeline_gpu_speedup = lambda pipeline, target: None
        self.pose_api = pose_api
        self.detector = init_detector(
            str(MODEL / 'configs/faster_rcnn_r50_fpn_coco.py'),
            str(MODEL / 'faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'),
            device=device)
        config = ROOT / 'vendor/gru_author_pose_deps/mmpose/.mim/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_384x288.py'
        self.pose = init_pose_model(
            str(config), str(MODEL / 'hrnet_w48_coco_384x288-314c8528_20200708.pth'),
            device=device)
        self.info = DatasetInfo(self.pose.cfg.data.test.dataset_info)
        cfg = self.detector.cfg.copy()
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
        self.det_pipeline = Compose(cfg.data.test.pipeline)

    def infer(self, image):
        import torch
        from mmcv.parallel import collate
        data = self.det_pipeline(dict(img=image))
        data = collate([data], samples_per_gpu=1)
        data['img_metas'] = [value.data[0] for value in data['img_metas']]
        data['img'] = [value.data[0].to(self.device) for value in data['img']]
        with torch.no_grad():
            boxes = self.detector(return_loss=False, rescale=True, **data)[0][0]
        people = [{'bbox': row} for row in boxes if row[4] >= self.bbox_threshold]
        if not people:
            return []
        result = []
        for begin in range(0, len(people), 2):
            batch, _ = self.pose_api.inference_top_down_pose_model(
                self.pose, image, people[begin:begin+2], bbox_thr=self.bbox_threshold,
                format='xyxy', dataset=self.pose.cfg.data.test.type,
                dataset_info=self.info, return_heatmap=False, outputs=None)
            result.extend(batch)
        return result


def author_polygons(corners, width, height, relax=.15):
    # Original source expects normalized TL,TR,BR,BL and multiplies coordinates
    # about the image origin. Preserve that exact relaxation implementation.
    raw = np.asarray(corners, np.float64) / [width, height]
    tl, tr, bl, br = raw
    points = np.asarray([tl*[1-relax, 1-relax], tr*[1+relax, 1-relax],
                         br*[1+relax, 1+relax], bl*[1-relax, 1+relax]], np.float32)
    near = points.copy()
    near[0] = .5*(points[0]+points[3])
    near[1] = .5*(points[1]+points[2])
    far = points.copy()
    far[3], far[2] = near[0], near[1]
    return [p.reshape(-1, 1, 2) for p in [points, near, far]]


def select_players(people, polygons, previous, width, height):
    """Exact author filtered2 selection/carry-forward for one decoded frame."""
    selected = []
    for person in sorted(people, key=lambda p: float(p['bbox'][4]), reverse=True):
        foot = person['keypoints'][16, :2] / [width, height]
        if cv2.pointPolygonTest(polygons[0], tuple(foot), False) != -1:
            selected.append(person)
    chosen = previous.copy()
    if len(selected) >= 2:
        a, b = selected[:2]
        if a['keypoints'][16, 1] > b['keypoints'][16, 1]:
            chosen = {'near': a, 'far': b}
        else:
            chosen = {'near': b, 'far': a}
    elif len(selected) == 1:
        foot = selected[0]['keypoints'][16, :2] / [width, height]
        side = 'near' if cv2.pointPolygonTest(polygons[1], tuple(foot), False) != -1 else 'far'
        chosen[side] = selected[0]
    return chosen


def person_record(person, width, height, side, observed):
    return {'track_id': 'author-filtered2-'+side,
            'bbox': [float(x) for x in person['bbox'][:4]],
            'bbox_score': float(person['bbox'][4]),
            'author_carry_forward': not observed,
            'keypoints': [{'name': NAMES[i], 'x': float(p[0]), 'y': float(p[1]),
                           'score': float(p[2]),
                           'in_frame': bool(0 <= p[0] < width and 0 <= p[1] < height)}
                          for i, p in enumerate(person['keypoints'])]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--court', help='JSON file:4x2 source pixel TL,TR,BL,BR, or object court_corners')
    parser.add_argument('--device', default='mps', choices=['cpu', 'mps'])
    parser.add_argument('--bbox-threshold', type=float, default=.6)
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--id')
    args = parser.parse_args()
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output/'dense_pose.json').exists():
        raise FileExistsError('Completed author pose output already exists')
    cap = cv2.VideoCapture(str(source))
    width, height = [int(cap.get(k)) for k in [cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT]]
    fps = cap.get(cv2.CAP_PROP_FPS)
    corners = json.loads(Path(args.court).read_text()) if args.court else None
    if isinstance(corners, dict):
        corners = corners['court_corners']
    polygons = author_polygons(corners, width, height) if corners is not None else None
    initialized = time.perf_counter()
    frontend = AuthorFrontend(args.device, args.bbox_threshold)
    inference_start = time.perf_counter()
    all_frames, raw_rows, filtered_rows, previous = [], [], [], {}
    index = 0
    while not args.max_frames or index < args.max_frames:
        ok, image = cap.read()
        if not ok:
            break
        people = frontend.infer(image)
        for person in people:
            box = person['bbox']
            row = {'Frame': index, 'x1': float(box[0]/width), 'y1': float(box[1]/height),
                   'x2': float(box[2]/width), 'y2': float(box[3]/height), 'bbox_conf': float(box[4])}
            row.update({f'{j}_{axis}': float(person['keypoints'][j, k]/[width, height][k])
                        for j in range(17) for k, axis in enumerate(['x', 'y'])})
            raw_rows.append(row)
        players = {}
        if polygons is not None:
            chosen = select_players(people, polygons, previous, width, height)
            for k, side in enumerate(['near', 'far'], 1):
                person = chosen.get(side)
                if person is not None:
                    players[side] = person_record(person, width, height, side,
                                                 any(person is p for p in people))
                    pts = person['keypoints'][:, :2]/[width, height]
                    box = np.asarray(person['bbox'][:4])/[width, height, width, height]
                else:
                    pts, box = np.zeros((17, 2)), np.zeros(4)
                row = {'Frame': index, 'id': k, 'x1': float(box[0]), 'y1': float(box[1]),
                       'x2': float(box[2]), 'y2': float(box[3]),
                       'bbox_conf': float(person['bbox'][4]) if person is not None else 0.}
                row.update({f'{j}_{axis}': float(pts[j, col]) for j in range(17) for col, axis in enumerate(['x', 'y'])})
                filtered_rows.append(row)
            previous = chosen
        all_frames.append({'frame_index': index, 'source_time': index/fps,
                           'source_native_pts': cap.get(cv2.CAP_PROP_POS_MSEC)/1000,
                           'players': players, 'raw_detections': len(people)})
        index += 1
        if index % 30 == 0:
            print('AUTHOR_POSE', index, 'seconds', round(time.perf_counter()-inference_start, 2), flush=True)
    cap.release()
    def write_csv(path, rows):
        if rows:
            with path.open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
    write_csv(output/'raw_pose_bbox.csv', raw_rows)
    write_csv(output/'filtered2_pose.csv', filtered_rows)
    metadata = {'engine_version': 'gru-author-fasterrcnn-hrnet-v1',
        'source_file': str(source), 'source_sha256': sha(source),
        'source_width': width, 'source_height': height, 'sample_fps': fps,
        'start': 0., 'end': len(all_frames)/fps,
        'id': args.id or source.stem, 'decoded_frame_count': len(all_frames),
        'bbox_threshold': args.bbox_threshold, 'keypoint_threshold_applied': False,
        'court_corners_px_TL_TR_BL_BR': corners, 'court_relax': .15,
        'runtime': frontend.runtime, 'device': args.device,
        'model_initialization_seconds': inference_start-initialized,
        'runtime_seconds': time.perf_counter()-inference_start,
        'weights': {str(p): sha(p) for p in MODEL.glob('*.pth')},
        'author_reference': 'kwyoke/Badminton-hit-detection annotation_pipeline/player/pose_bbox_detection.ipynb and get_filtered_pose.ipynb',
        'adaptations': ['Legacy native NMS/RoIAlign use torchvision CPU kernels, original network/config/weights retained.',
            'Only data placement extended to MPS; official affine preprocessing and heatmap decode/flip-test retained.',
            'All original decoded frames retained; author get_filtered_pose stopped at last frame with a detection.',
            'Per-joint scores saved for diagnostics, not used to filter author coordinates.',
            'Pose persons split into batches of2 for Mac memory; NumPy2 scalar assignment compatibility fix in bbox-score metadata.'],
        'assigned_frames': {side: sum(side in f['players'] for f in all_frames) for side in ['near', 'far']}}
    save(output/'metadata.json', metadata)
    if polygons is not None:
        save(output/'dense_pose.json', {**metadata, 'frames': all_frames})
    print('AUTHOR_POSE_DONE', json.dumps({'frames': len(all_frames), 'seconds': metadata['runtime_seconds'],
        'fps': len(all_frames)/max(.001, metadata['runtime_seconds']), 'output': str(output)}), flush=True)


if __name__ == '__main__':
    main()
