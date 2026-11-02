"""Official TrackNetV3 / InpaintNet inference adapter for local video.

Author architecture, dataset preprocessing, centroid and inpaint-mask functions
are loaded from the pinned vendor checkout. Non-overlap is an author-supported
inference mode; it is recorded and is not a reproduction of the reported metric.
Raw detection and model-completed coordinates are saved separately.
"""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / 'vendor/TrackNetV3'
sys.path.insert(0, str(VENDOR))
from dataset import Video_IterableDataset, Shuttlecock_Trajectory_Dataset
from utils.general import get_model, COOR_TH, HEIGHT, WIDTH


def sha(file):
    h = hashlib.sha256()
    with Path(file).open('rb') as stream:
        for b in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def official_function(name):
    """Load only the unchanged author helper, avoiding training dependencies."""
    module = ast.parse((VENDOR / 'test.py').read_text())
    node = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = {'np': np, 'cv2': cv2}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(VENDOR / 'test.py'), 'exec'), scope)
    return scope[name]


def emit(value):
    print('TRAJECTORY_PROGRESS ' + json.dumps(value, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    source, out = Path(config['source_file']), Path(config['specialist_folder'])
    out.mkdir(parents=True, exist_ok=True)
    if sha(source) != config['source_sha256']:
        raise ValueError('Source video changed')
    weights = ROOT / 'models/tracknetv3'
    checkpoint = torch.load(weights / 'TrackNet_best.pt', map_location='cpu', weights_only=True)
    params = checkpoint['param_dict']
    seq_len, bg_mode = params['seq_len'], params['bg_mode']
    model = get_model('TrackNet', seq_len, bg_mode)
    model.load_state_dict(checkpoint['model'], strict=True)
    del checkpoint
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    torch.set_num_threads(4)
    model = model.to(device).eval()
    cap = cv2.VideoCapture(str(source))
    width, height = int(cap.get(3)), int(cap.get(4))
    fps, count = cap.get(5), int(cap.get(7))
    cap.release()
    started = time.perf_counter()
    # Use the author's streaming implementation and RGB/PIL preprocessing.
    dataset = Video_IterableDataset(str(source), seq_len=seq_len, sliding_step=seq_len,
                                   bg_mode=bg_mode, max_sample_num=64)
    loader = DataLoader(dataset, batch_size=2, num_workers=0)
    locate = official_function('predict_location')
    raw = {'Frame': [], 'X': [], 'Y': [], 'Visibility': [], 'Inpaint_Mask': [],
           'Img_scaler': (width / WIDTH, height / HEIGHT), 'Img_shape': (width, height)}
    confidences = []
    for indices, images in loader:
        with torch.inference_mode():
            heatmaps = model(images.float().to(device)).cpu().numpy()
        for ids, predictions in zip(indices.numpy(), heatmaps):
            for (_, index), heat in zip(ids, predictions):
                index = int(index)
                if index >= count or (raw['Frame'] and index <= raw['Frame'][-1]):
                    continue
                box = locate(((heat > .5) * 255).astype('uint8'))
                x = int((box[0] + box[2] / 2) * width / WIDTH)
                y = int((box[1] + box[3] / 2) * height / HEIGHT)
                raw['Frame'].append(index)
                raw['X'].append(x)
                raw['Y'].append(y)
                raw['Visibility'].append(int(x != 0 or y != 0))
                confidences.append(float(heat.max()))
        if len(raw['Frame']) % 160 == 0 or len(raw['Frame']) >= count:
            emit({'processed': len(raw['Frame']), 'total': count,
                  'runtime_seconds': round(time.perf_counter() - started, 3)})
    if len(raw['Frame']) != count:
        raise ValueError('TrackNet output does not cover the decoded source frames')
    del model, dataset, loader
    if device.type == 'mps':
        torch.mps.empty_cache()
    raw['Inpaint_Mask'] = official_function('generate_inpaint_mask')(raw, th_h=height * .05)
    ck = torch.load(weights / 'InpaintNet_best.pt', map_location='cpu', weights_only=True)
    inpaint = get_model('InpaintNet').to(device).eval()
    inpaint.load_state_dict(ck['model'], strict=True)
    inpaint_seq = ck['param_dict']['seq_len']
    del ck
    coords = Shuttlecock_Trajectory_Dataset(seq_len=inpaint_seq, sliding_step=inpaint_seq,
                                           data_mode='coordinate', pred_dict=raw, padding=True)
    repaired = {}
    for indices, coor, mask in DataLoader(coords, batch_size=16, num_workers=0):
        coor, mask = coor.float(), mask.float()
        with torch.inference_mode():
            pred = inpaint(coor.to(device), mask.to(device)).cpu()
        pred = pred * mask + coor * (1 - mask)
        pred[(pred[..., 0] < COOR_TH) & (pred[..., 1] < COOR_TH)] = 0
        for ids, points in zip(indices.numpy(), pred.numpy()):
            for (_, index), point in zip(ids, points):
                if int(index) < count:
                    repaired[int(index)] = [int(point[0] * width), int(point[1] * height)]
    frames = []
    for i, index in enumerate(raw['Frame']):
        frames.append({'frame_index': index, 'source_time_nominal': round(index / fps, 6),
                       'raw_xy': [raw['X'][i], raw['Y'][i]] if raw['Visibility'][i] else None,
                       'raw_visible': bool(raw['Visibility'][i]), 'raw_heatmap_peak': confidences[i],
                       'inpaint_requested': bool(raw['Inpaint_Mask'][i]),
                       'rectified_xy': repaired.get(index),
                       'coordinate_source': 'inpaintnet_completion' if raw['Inpaint_Mask'][i] else 'tracknet_detection'})
    revision = subprocess.check_output(['git', '-C', str(VENDOR), 'rev-parse', 'HEAD'], text=True).strip()
    result = {'schema_version': 1, 'engine_version': 'tracknetv3-official-nonoverlap-v1',
              'source_sha256': config['source_sha256'], 'source_width': width, 'source_height': height,
              'nominal_fps': fps, 'frame_count': count, 'frames': frames,
              'device': str(device), 'author_code_revision': revision, 'code_sha256': sha(Path(__file__)),
              'author_dataset_file_sha256': sha(VENDOR/'dataset.py'),
              'local_dataset_patch': 'Empty EOF frame-list guard; same author preprocessing and inference.',
              'weights': {name: {'sha256': sha(weights / name), 'size_bytes': (weights / name).stat().st_size}
                          for name in ['TrackNet_best.pt', 'InpaintNet_best.pt']},
              'configuration': {'checkpoint_params': params, 'eval_mode': 'nonoverlap', 'batch_size': 2,
                                'background_max_sample_num': 64, 'inpaint_seq_len': inpaint_seq},
              'runtime_seconds': round(time.perf_counter() - started, 3),
              'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'raw_detected_frames': sum(raw['Visibility']), 'inpaint_requested_frames': sum(raw['Inpaint_Mask']),
              'note': 'Visibility is model output, not accuracy. Completed coordinates are never considered observed contact evidence. Nominal index/fps times are linked to dense pose by original frame index.'}
    temporary = out / 'trajectory.partial.json'
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(out / 'trajectory.json')
    with (out / 'trajectory.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['frame', 'time_nominal', 'raw_x', 'raw_y', 'raw_visible', 'rectified_x', 'rectified_y', 'inpaint_requested', 'peak'])
        for f in frames:
            writer.writerow([f['frame_index'], f['source_time_nominal'], *(f['raw_xy'] or ['', '']),
                             int(f['raw_visible']), *(f['rectified_xy'] or ['', '']), int(f['inpaint_requested']), f['raw_heatmap_peak']])
    print('TRAJECTORY_DONE ' + json.dumps({k: result[k] for k in ['frame_count', 'raw_detected_frames', 'inpaint_requested_frames', 'runtime_seconds', 'device']}), flush=True)


if __name__ == '__main__':
    main()
