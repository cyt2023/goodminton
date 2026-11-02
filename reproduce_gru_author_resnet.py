"""Original kwyoke ResNet50 hit baseline, using supplied per-frame player boxes.

The trained image model and preprocessing are author implementations. Box
provenance is a separate input condition: this adapter never calls a detector or
uses contact labels. It preserves the author's image-crop quirks and fixed
postprocessing parameters, and reports that distinction explicitly.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms

from benchmark_hits_gru import labels_to_events, maximise_hit_score

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT.parent / '03_数据/GRU作者流程复现/2026-10-25/author_assets'
WEIGHTS = ASSETS / 'resnet_nov_15epoch_indv_clearnohit.pt'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def post_crop_processing(height, width, des_height, des_width, x1, y1, x2, y2, padding_scale=1.3):
    # Author literal implementation: x/y end updates use the already updated
    # start value. Do not silently change it to symmetric padding.
    y1 = max((y1+y2)*0.5 - (y2-y1)*0.5*padding_scale, 0)
    y2 = min((y1+y2)*0.5 + (y2-y1)*0.5*padding_scale, height)
    x1 = max((x1+x2)*0.5 - (x2-x1)*0.5*padding_scale, 0)
    x2 = min((x1+x2)*0.5 + (x2-x1)*0.5*padding_scale, width)
    if (x2-x1)/(y2-y1) > des_width/des_height:
        ratio = ((x2-x1)/(y2-y1))/(des_width/des_height)
        y1_ = max((y1+y2)*0.5 - (y2-y1)*0.5*ratio, 0)
        y2_ = min((y1+y2)*0.5 + (y2-y1)*0.5*ratio, height)
        if y1_ == 0:
            y2_ = (des_height/des_width)*(x2-x1)
        elif y2_ == height:
            y1_ = height - (des_height/des_width)*(x2-x1)
        y1, y2 = y1_, y2_
    elif (x2-x1)/(y2-y1) < des_width/des_height:
        ratio = ((y2-y1)/(x2-x1))*(des_width/des_height)
        x1_ = max((x1+x2)*0.5 - (x2-x1)*0.5*ratio, 0)
        x2_ = min((x1+x2)*0.5 + (x2-x1)*0.5*ratio, width)
        if x1_ == 0:
            x2_ = (des_width/des_height)*(y2-y1)
        elif x2_ == width:
            x1_ = width - (des_width/des_height)*(y2-y1)
        x1, x2 = x1_, x2_
    return int(x1), int(y1), int(x2), int(y2)


@lru_cache(maxsize=2)
def load_model(device):
    torch.set_num_threads(2)
    model = models.resnet50(weights=None, num_classes=3)
    state = torch.load(WEIGHTS, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def resnet_labels(probs, postprocess='naive', confidence=.95):
    if postprocess == 'max':
        return maximise_hit_score(probs[:, [0, 1, 5]])
    labels = np.zeros(len(probs), np.int32)
    for i, probability in enumerate(probs):
        p1, p2 = probability[:3], probability[3:]
        label1, label2 = int(np.argmax(p1)), int(np.argmax(p2))
        if (label1 == 1 and label2 == 0) or (label1 == 1 and label2 == 1):
            if p1.max() > confidence:
                labels[i] = 1
        elif (label1 == 0 and label2 == 2) or (label1 == 2 and label2 == 2):
            if p2.max() > confidence:
                labels[i] = 2
    if postprocess == 'naive':
        selected = []
        for index in np.flatnonzero(labels):
            if not selected or index > selected[-1]+15:
                selected.append(int(index))
        result = np.zeros(len(labels), np.int32)
        for index in selected:
            result[index-3:index+4] = labels[index]
        return result
    return labels


def infer(dense, trajectory=None, source_path=None, output_folder=None,
          postprocess='naive', device=None, batch_size=32, max_frames=None):
    if not source_path:
        raise ValueError('source_path is required')
    device = device or ('mps' if torch.backends.mps.is_available() else 'cpu')
    started = time.perf_counter()
    frames = dense['frames']
    if max_frames is not None:
        frames = frames[:max_frames]
    if not all(frame['frame_index'] == i for i, frame in enumerate(frames)):
        raise ValueError('Expected original consecutive frame indices from zero')
    n = len(frames)
    order = ['near', 'far']
    feet = {side: [np.mean([p['keypoints'][j]['y'] for j in [15, 16]])
                   for f in frames if (p := f['players'].get(side))]
            for side in order}
    order.sort(key=lambda side: float(np.median(feet[side])) if feet[side] else -1e10, reverse=True)
    transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])])
    model = load_model(device)
    probs = np.zeros((n, 6), np.float32)
    cap = cv2.VideoCapture(str(source_path))
    batch, destinations = [], []
    missing = {side: 0 for side in order}

    def flush():
        if not batch:
            return
        with torch.inference_mode():
            values = model(torch.stack(batch).to(device)).softmax(dim=1).cpu().numpy()
        for (frame_i, player), value in zip(destinations, values):
            probs[frame_i, player*3:(player+1)*3] = value
        batch.clear()
        destinations.clear()

    for i, info in enumerate(frames):
        success, frame = cap.read()
        if not success:
            raise ValueError(f'Video stopped before expected frame {i}/{n}')
        height, width = frame.shape[:2]
        for player, side in enumerate(order):
            person = info['players'].get(side)
            if not person or not np.any(person['bbox']):
                crop = np.zeros((128, 128, 3), np.uint8)
                missing[side] += 1
            else:
                box = [int(v) for v in person['bbox']]
                x1, y1, x2, y2 = post_crop_processing(height, width, 128, 128, *box)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    raise ValueError(f'Author crop produced empty image at frame {i}, {side}, {box}')
                crop = cv2.resize(crop, (128, 128))
            # Original organiser saves cv2 JPEG files before PIL/ImageFolder.
            # Preserve that encoding rather than feed lossless in-memory RGB.
            success, encoded = cv2.imencode('.jpg', crop)
            if not success:
                raise RuntimeError('Could not encode author crop JPEG')
            image = Image.open(io.BytesIO(encoded.tobytes())).convert('RGB')
            batch.append(transform(image))
            destinations.append((i, player))
            if len(batch) >= batch_size:
                flush()
    cap.release()
    flush()
    labels = resnet_labels(probs, postprocess)
    frame_probs = np.stack([probs[:, 0], probs[:, 1], probs[:, 5]], axis=1)
    events = labels_to_events(labels, frames, order, frame_probs)
    return {'events': events, 'frame_probs_six_columns': probs,
        'frame_labels': labels, 'metadata': {
            'model': 'kwyoke-author-ResNet50-individual-crops-15epoch',
            'weights_path': str(WEIGHTS), 'weights_sha256': sha256(WEIGHTS),
            'source': 'https://github.com/kwyoke/Badminton-hit-detection/blob/master/ResNet_baseline.ipynb',
            'notebook_sha256': sha256(ASSETS/'ResNet_baseline.ipynb'),
            'code_sha256': sha256(__file__), 'strict_checkpoint_loading': True,
            'device': device, 'postprocess': postprocess, 'confidence_threshold': .95,
            'postprocess_fps': 30, 'bbox_camera_order': order,
            'missing_box_frames': missing,
            'input_condition': dense.get('input_condition', dense.get('engine_version', 'supplied_boxes_unknown_provenance')),
            'preprocessing': 'Author literal sequential1.3padding,128x128 OpenCV crop,JPEG95,PIL RGB,Resize224,CenterCrop224,ImageNet normalization.',
            'frontend_scope': 'Input boxes are supplied separately. No player detector, court assignment or ground-truth contacts are run/read by this adapter.',
            'frame_order': 'Direct original decoded frame order; replaces disk directory/natsort bookkeeping.',
            'point_conversion': 'Author nonzero interval to rounded midpoint; no13-frame GRU offset for RGB.',
            'runtime_seconds': time.perf_counter()-started}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dense', required=True)
    parser.add_argument('--video', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--postprocess', default='naive', choices=['naive', 'max', 'none'])
    parser.add_argument('--device')
    parser.add_argument('--max-frames', type=int)
    args = parser.parse_args()
    output = infer(json.loads(Path(args.dense).read_text()), source_path=args.video,
                   postprocess=args.postprocess, device=args.device, max_frames=args.max_frames)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target.with_suffix('.npz'),
                        **{key: output.pop(key) for key in ['frame_probs_six_columns', 'frame_labels']})
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(json.dumps({'output': str(target), 'events': len(output['events']),
                      'seconds': output['metadata']['runtime_seconds']}), flush=True)


if __name__ == '__main__':
    main()
