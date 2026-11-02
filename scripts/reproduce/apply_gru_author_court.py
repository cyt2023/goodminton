"""Apply original court filtering to already-computed author pose CSVs."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # prototype root, after move to scripts/reproduce/
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analysis_jobs import atomic_json, file_hash
from reproduce_gru_author_pose import author_polygons, select_players, person_record


def apply(raw_folder, output, corners, condition, provenance):
    raw_folder, output = Path(raw_folder), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((raw_folder/'metadata.json').read_text())
    width, height = metadata['source_width'], metadata['source_height']
    people = {}
    for row in csv.DictReader((raw_folder/'raw_pose_bbox.csv').open()):
        frame = int(row['Frame'])
        box = np.array([float(row[k])*scale for k, scale in
                        zip(['x1', 'y1', 'x2', 'y2'], [width, height, width, height])] + [float(row['bbox_conf'])])
        points = np.asarray([[float(row[f'{j}_x'])*width, float(row[f'{j}_y'])*height, 0.]
                             for j in range(17)])
        # Original CSV exports coordinates, not individual joint confidences.
        # Zero is an explicit unknown-score sentinel, never used by authorGRU.
        people.setdefault(frame, []).append({'bbox': box, 'keypoints': points})
    polygons = author_polygons(corners, width, height)
    frames, previous = [], {}
    for i in range(metadata['decoded_frame_count']):
        raw = people.get(i, [])
        chosen = select_players(raw, polygons, previous, width, height)
        records = {side: person_record(person, width, height, side, any(person is p for p in raw))
                   for side, person in chosen.items()}
        frames.append({'frame_index': i, 'source_time': i/metadata['sample_fps'],
                       'players': records, 'raw_detections': len(raw)})
        previous = chosen
    result = {**metadata, 'frames': frames,
        'court_corners_px_TL_TR_BL_BR': corners,
        'geometry_condition': condition, 'geometry_provenance': provenance,
        'geometry_application': 'Original author polygons/select_players, applied after completed original detector/HRNet inference. No NN rerun or hit labels.',
        'geometry_code_sha256': file_hash(Path(__file__)),
        'raw_pose_csv_sha256': file_hash(raw_folder/'raw_pose_bbox.csv'),
        'joint_confidence_note': 'Not exported by original CSV; keypoint score0 means unavailable. Coordinates and real bbox confidence retained; GRU ignores joint scores.',
        'assigned_frames': {side: sum(side in frame['players'] for frame in frames) for side in ['near', 'far']}}
    atomic_json(output/'dense_pose.json', result)
    print('AUTHOR_GEOMETRY_APPLIED', output, condition, result['assigned_frames'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-folder', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--court', required=True)
    parser.add_argument('--condition', required=True)
    parser.add_argument('--provenance', required=True)
    args = parser.parse_args()
    value = json.loads(Path(args.court).read_text())
    corners = value['court_corners'] if isinstance(value, dict) else value
    apply(args.raw_folder, args.output, corners, args.condition, args.provenance)


if __name__ == '__main__':
    main()
