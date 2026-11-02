"""Run the GRU project's actual TrackNetV2 model906_30 shuttle frontend.

Original HDF5 network/weights, original non-overlapping RGB triples and contour
decoder. Keras3/PyTorch replaces the unavailable historical TF1 runtime on Mac.
No TrackNetV3, inpainting, hand ball coordinates, or hit annotations are used.
"""
import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault('KERAS_BACKEND', 'torch')
import cv2
import keras
import numpy as np
import torch

from analysis_jobs import atomic_json, file_hash

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / 'models/tracknetv2_author/model906_30'


def load_model(device='mps'):
    torch.set_num_threads(2)
    if device == 'mps' and not torch.backends.mps.is_available():
        device = 'cpu'
    named_h5 = WEIGHTS.with_suffix('.h5')
    if not named_h5.exists():
        named_h5.symlink_to(WEIGHTS.name)
    model = keras.models.load_model(named_h5, compile=False)
    assert model.input_shape == (None, 9, 288, 512)
    assert model.output_shape == (None, 3, 288, 512)
    return model.to(device), device


def preprocess(images):
    # Keep the notebook's array_to_img default scaling and PIL default resize.
    # NCHW BatchNorm axis=-1 in the original H5 is intentional here: it has
    # learned512/256/etc spatial-column parameters, not channel BatchNorm.
    rgb = [np.moveaxis(np.asarray(keras.utils.array_to_img(image[..., ::-1]).resize((512, 288))), -1, 0)
           for image in images]
    return np.concatenate(rgb, axis=0)[None].astype(np.float32) / 255.


def decode(heatmap, source_height):
    binary = ((heatmap > .5).astype(np.uint8) * 255)
    if not binary.any():
        return None
    contours, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rectangles = [cv2.boundingRect(contour) for contour in contours]
    x, y, w, h = max(rectangles, key=lambda box: box[2] * box[3])
    ratio = source_height / 288.
    # The original notebook scales both axes by source-height/288. Preserve it
    # even for the personal960x544 source; metadata discloses this assumption.
    return [int(ratio * (x + w / 2)), int(ratio * (y + h / 2))]


def infer(source, output, case_id=None, max_frames=None, device='mps', model=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if model is None:
        model, device = load_model(device)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError(f'Cannot decode {source}')
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_count, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS))
    count = min(source_count, max_frames) if max_frames is not None else source_count
    rows, position = [], 0
    while position < count:
        images, indices, times = [], [], []
        for _ in range(min(3, count-position)):
            ok, image = cap.read()
            if not ok:
                raise ValueError(f'Unexpected decoded end at {position+len(images)}/{count}')
            images.append(image)
            indices.append(position+len(images)-1)
            times.append(float(cap.get(cv2.CAP_PROP_POS_MSEC))/1000.)
        if len(images) == 3:
            x = torch.from_numpy(preprocess(images)).to(device)
            with torch.inference_mode():
                prediction = model(x, training=False).detach().cpu().numpy()[0]
            if not np.isfinite(prediction).all():
                raise ValueError('Nonfinite TrackNetV2 output')
            points = [decode(plane, height) for plane in prediction]
        else:
            points = [None] * len(images)
        for i, timestamp, point in zip(indices, times, points):
            rows.append({'frame_index': i, 'source_time_nominal': i/fps,
                'source_time_decoder': timestamp,
                'raw_xy': point, 'raw_visible': point is not None,
                'rectified_xy': point if point is not None else [0, 0],
                'coordinate_source': 'author_tracknetv2_model906_30' if point is not None else 'author_model_missing_or_incomplete_tail',
                'inpaint_requested': False})
        position += len(images)
        if position % 150 == 0 or position == count:
            print('AUTHOR_TRACKNET', case_id or source.stem, position, '/', count,
                  'seconds', round(time.perf_counter()-started, 1), flush=True)
    cap.release()
    elapsed = time.perf_counter()-started
    result = {'schema_version': 'author-gru-tracknetv2-v1',
        'source_sha256': file_hash(source), 'source_width': width, 'source_height': height,
        'nominal_fps': fps, 'frame_count': count, 'original_source_frames': source_count,
        'frames': rows, 'runtime_seconds': elapsed, 'device': device,
        'raw_detected_frames': sum(row['raw_visible'] for row in rows),
        'weights': {'path': str(WEIGHTS), 'sha256': file_hash(WEIGHTS)},
        'configuration': {'input': 'Non-overlapping3source frames, RGB, PIL512x288, NCHW9channels, /255.',
            'decode': 'Threshold>0.5, largest external-contour bounding-box area, integer bbox center.',
            'coordinate_scale': 'Both axes multiplied by source_height/288 as original author notebook.',
            'original_batchnorm_axis': '-1; original HDF5 graph retained.',
            'incomplete_tail_frames': count % 3,
            'incomplete_tail_policy': 'Original model cannot consume fewer than3frames; retain missing rows rather than drop GT events.'},
        'runtime_adaptation': 'OriginalHDF5 graph/weights loaded with Keras3 PyTorch backend; no retraining or architectural substitution.',
        'code_sha256': file_hash(Path(__file__))}
    atomic_json(output/'trajectory.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--id')
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--device', default='mps', choices=['mps', 'cpu'])
    args = parser.parse_args()
    result = infer(args.source, args.output, args.id, args.max_frames, args.device)
    print('AUTHOR_TRACKNET_DONE', result['frame_count'], round(result['runtime_seconds'], 2), flush=True)


if __name__ == '__main__':
    main()
