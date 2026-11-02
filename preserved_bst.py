"""Replay frozen pretrained BST on full stroke support, preserving its MLP."""
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid
import numpy as np
import torch

from analysis_jobs import DATA, atomic_json, file_hash
from personal_learning import extractor
from specialist_model import ROOT, BST_NAME
from stroke_context import metadata, window, VERSION as WINDOW_VERSION
from stroke_review import folder_for, load_source, read_labels
from stroke_taxonomy import AUTHOR_MAP, LABELS

VERSION = 'pretrained-bst-full-stroke-v1'


def rank(prediction):
    merged = {}
    for label, value in zip(prediction['class_labels'], prediction['bst0_all_scores']):
        code = AUTHOR_MAP.get(label.split('_', 1)[-1], 'unknown')
        merged[code] = merged.get(code, 0.)+float(value)
    # Retain the author's original top decision, not a new aggregation decision.
    first = AUTHOR_MAP.get(prediction['bst0'][0]['label'].split('_', 1)[-1], 'unknown')
    return first, [{'code': c, 'label': LABELS[c], 'uncalibrated_score': v}
                   for c, v in sorted(merged.items(), key=lambda x: -x[1])[:3]]


def prepare(job):
    torch.set_num_threads(1)
    report, events, source = load_source(job)
    doc = read_labels(job, report['source_sha256'])
    classifier, dense = extractor(job)
    context = metadata(dense, events, doc)
    rows = []
    for event in events:
        w = context['windows'][event['id']]
        values, quality = classifier.inputs({**event, **w})
        prediction = classifier.predict(values)
        code, ranking = rank(prediction)
        rows.append({'event_id': event['id'], 'stroke_type': code, 'stroke_label': LABELS[code],
                     'ranking': ranking, 'source_top_label': prediction['bst0'][0]['label'],
                     'bst0_all_scores': prediction['bst0_all_scores'],
                     'class_labels': prediction['class_labels'], 'model_window': w,
                     'input_quality': quality, 'needs_confirmation': True,
                     'uses_human_contact': w['contact_confirmed'],
                     'review_status': 'unreviewed_pretrained_full_context_candidate',
                     'quality_reasons': ['完整动作范围为系统建议，边界未人工确认']
                        + (['两位球员共同可见帧不足50%'] if quality['paired_player_fraction']<.5 else [])})
    value = {'version': VERSION, 'window_version': WINDOW_VERSION, 'job_id': job,
             'source_sha256': report['source_sha256'], 'dataset_revision': doc['revision'],
             'created_at': datetime.now(timezone.utc).isoformat(), 'events': rows,
             'pretrained_mlp_preserved': True, 'model_retrained': False,
             'model_sha256': file_hash(ROOT/'models/bst'/BST_NAME),
             'candidate_results_sha256': file_hash(source),
             'code_sha256': file_hash(Path(__file__)),
             'window_code_sha256': file_hash(Path(__file__).with_name('stroke_context.py')),
             'human_labels_created': 0, 'independent_validation': None,
             'scope': 'Full-support frozen BST replay; human-assisted contacts where saved, not independent accuracy.'}
    folder = folder_for(job)
    atomic_json(folder/'保留预训练记录'/uuid.uuid4().hex/'result.json', value)
    atomic_json(folder/'preserved_bst_predictions.json', value)
    atomic_json(folder/'stroke_context.json', context)
    print('FULL_CONTEXT_BST', job, len(rows), flush=True)
    return value


def read_valid(job, source_sha, revision):
    f = folder_for(job)/'preserved_bst_predictions.json'
    if not f.is_file():
        return None
    d = json.loads(f.read_text())
    if (d.get('source_sha256') != source_sha or d.get('dataset_revision') != revision
            or d.get('version') != VERSION or d.get('code_sha256') != file_hash(Path(__file__))
            or d.get('window_code_sha256') != file_hash(Path(__file__).with_name('stroke_context.py'))):
        return None
    return d


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser();p.add_argument('--job', required=True)
    prepare(p.parse_args().job)
