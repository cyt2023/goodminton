"""Summarise real specialist outputs and run the personal-clip DTW baseline."""
import argparse
from collections import Counter
import json
from pathlib import Path

from analysis_jobs import atomic_json, data_path, file_hash
from reference_alignment import retrieve_and_align
from stroke_taxonomy import LABELS, VERSION, normalise_event

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / '03_数据'


def build(job_id):
    folder = data_path(DATA/'专项分析'/job_id)
    dense = json.loads((folder/'dense_pose.json').read_text())
    trajectory = json.loads((folder/'trajectory.json').read_text())
    result_file = data_path(DATA/'技术动作'/job_id/'片段/classified_events.json')
    result = json.loads(result_file.read_text())
    events = [normalise_event(e) for e in result['events']]
    aligned = retrieve_and_align(dense, events)
    atomic_json(folder/'reference_alignments.json', aligned)
    usable = [e for e in events if e.get('quality') == 'usable' and e.get('visible_stroke') and e.get('stroke_certainty') == 'likely' and not e.get('classification_error')]
    records = []
    names = LABELS
    for e in usable:
        records.append({**{k:e.get(k) for k in ['id','player','start','end','peak_time','stroke_type','swing_style','style_certainty','style_evidence']},
                        'stroke_label':names[e['stroke_type']], 'review_status':'unreviewed',
                        'thumbnail_url':f'/action-preview/{job_id}/{e["id"]}.jpg',
                        'sequence_url':f'/specialist-image/{job_id}/{e["id"]}.jpg'})
    report = {'job_id':job_id,'source_video_id':dense['source_video_id'],'source_sha256':dense['source_sha256'],
              'source_duration':dense['end'],'methods':result['methods'], 'status':'completed', 'taxonomy_version':VERSION,
              'metrics':{'original_frames':len(dense['frames']),'pose_assigned_frames':dense['assigned_frames'],
                         'pose_timestamp_fallbacks':dense['timestamp_fallbacks'],
                         'pose_runtime_seconds':dense['runtime_seconds'],'trajectory_runtime_seconds':trajectory['runtime_seconds'],
                         'raw_ball_detected_frames':trajectory['raw_detected_frames'],
                         'inpaint_requested_frames':trajectory['inpaint_requested_frames'],
                         'attempted_hit_candidates':len(events), 'candidate_players':dict(Counter(e['player'] for e in events)),
                         'usable_stroke_candidates':len(usable),'usable_stroke_type_counts':dict(Counter(e['stroke_type'] for e in usable)),
                         'likely_style_counts':dict(Counter(e['swing_style'] for e in usable if e['style_certainty']=='likely')),
                         'error_counts':dict(Counter(e['classification_error'] for e in events if e.get('classification_error'))),
                         'withheld_candidates':sum(bool(e.get('withheld_reason')) for e in events),
                         'personal_events_with_alignment':sum(bool(r['references']) for r in aligned['results']),
                         'expert_reviewed_events':0},
              'events':records,'alignments':aligned,
              'interpretation':'Counts describe this inference run; they are not accuracy, recall or proof of correct fore/backhand recognition. Personal reference candidates are not correct-technique demonstrations.',
              'provenance':{'classified_events_sha256':file_hash(result_file),'alignment_code_sha256':file_hash(ROOT/'reference_alignment.py'),
                            'report_code_sha256':file_hash(Path(__file__))}}
    atomic_json(folder/'report.json',report)
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--job',required=True)
    args=parser.parse_args()
    value=build(args.job)
    print(json.dumps(value['metrics'],ensure_ascii=False,indent=2))
