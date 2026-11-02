"""Apply the author's court/identity filter after raw GPU inference is done.

This supports a later supplied court calibration without rerunning the networks.
The author CSV saves joint coordinates but not heatmap scores; reconstructed
score1 values are explicit sentinels and never used by the GRU input builder.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from reproduce_gru_author_pose import author_polygons, select_players, person_record, sha, save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', required=True)
    parser.add_argument('--court', required=True)
    args = parser.parse_args()
    folder = Path(args.folder)
    if (folder/'dense_pose.json').exists():
        raise FileExistsError('Do not overwrite existing filtered author poses')
    metadata = json.loads((folder/'metadata.json').read_text())
    court = json.loads(Path(args.court).read_text())
    if isinstance(court, dict):
        court = court['court_corners']
    width, height = metadata['source_width'], metadata['source_height']
    grouped = {}
    for row in csv.DictReader((folder/'raw_pose_bbox.csv').open()):
        frame = int(float(row['Frame']))
        box = np.asarray([float(row[k])*scale for k, scale in zip(['x1','y1','x2','y2'],[width,height,width,height])] + [float(row['bbox_conf'])])
        points = np.asarray([[float(row[f'{j}_x'])*width,float(row[f'{j}_y'])*height,1.] for j in range(17)])
        grouped.setdefault(frame, []).append({'bbox':box, 'keypoints':points})
    polygons = author_polygons(court, width, height)
    previous, frames, filtered = {}, [], []
    for index in range(metadata['decoded_frame_count']):
        people = grouped.get(index, [])
        previous = select_players(people, polygons, previous, width, height)
        players = {}
        for k, side in enumerate(['near','far'],1):
            person = previous.get(side)
            if person is not None:
                record = person_record(person,width,height,side,any(person is p for p in people))
                record['joint_score_source']='one_sentinel_reconstructed_from_author_csv_not_model_confidence'
                players[side] = record
                box = person['bbox'][:4]/[width,height,width,height]
                points = person['keypoints'][:,:2]/[width,height]
            else:
                box,points = np.zeros(4),np.zeros((17,2))
            row={'Frame':index,'id':k,'x1':float(box[0]),'y1':float(box[1]),'x2':float(box[2]),'y2':float(box[3]),'bbox_conf':float(person['bbox'][4]) if person is not None else 0.}
            row.update({f'{j}_{axis}':float(points[j,c]) for j in range(17) for c,axis in enumerate(['x','y'])})
            filtered.append(row)
        frames.append({'frame_index':index,'source_time':index/metadata['sample_fps'],
                       'players':players,'raw_detections':len(people)})
    with (folder/'filtered2_pose.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(filtered[0]));writer.writeheader();writer.writerows(filtered)
    metadata.update(court_corners_px_TL_TR_BL_BR=court,
                    offline_filter_source_sha256=sha(folder/'raw_pose_bbox.csv'),
                    offline_court_file=str(Path(args.court).resolve()),
                    assigned_frames={side:sum(side in f['players'] for f in frames) for side in ['near','far']},
                    joint_score_note='CSV stores coordinates only. Score1 is an explicit unused sentinel, not actual keypoint confidence.')
    save(folder/'dense_pose.json',{**metadata,'frames':frames})
    save(folder/'offline_filter_metadata.json',metadata)
    print('AUTHOR_FILTER_COMPLETE',len(frames),folder/'dense_pose.json')


if __name__=='__main__':main()
