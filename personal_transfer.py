"""Replay an existing human-label head on another video as unverified hints.

This does not fit a new head, infer athlete identity or create training labels.
"""
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np

from analysis_jobs import DATA,atomic_json,file_hash
from stroke_review import folder_for,load_source,read_labels
from personal_learning import extractor
from personal_head import scores
from stroke_taxonomy import LABELS

VERSION='cross-video-human-head-hints-v1'


def read_valid(job,source_sha):
    path=folder_for(job)/'personal_transfer_predictions.json'
    if not path.is_file():return None
    value=json.loads(path.read_text())
    if value.get('source_sha256')!=source_sha or value.get('version')!=VERSION:return None
    model_path=DATA/value['model_file']
    if not model_path.is_file() or file_hash(model_path)!=value['model_sha256']:return None
    return value


def replay(job,training_job,run_id):
    if job==training_job:raise ValueError('Use the original prediction run for its training video')
    source_folder=folder_for(training_job)/'学习记录'/run_id
    manifest=json.loads((source_folder/'manifest.json').read_text())
    training_snapshot=json.loads((source_folder/'human_labels_snapshot.json').read_text())
    if manifest.get('training_target_source')!='human_user_selection_only':raise ValueError('Training target provenance is not human-only')
    model_path=source_folder/'personal_head.npz'
    if file_hash(model_path)!=manifest['model_sha256']:raise ValueError('Model file differs from its manifest')
    with np.load(model_path,allow_pickle=False) as z:
        model={k:z[k] for k in z.files if k!='config_json'}
        model['config']=json.loads(str(z['config_json']))
    version=manifest['feature_version']
    if version=='local-hit-v1':
        from stroke_features import HitFeatures as FeatureExtractor
    elif version=='local-hit-context-fusion-v1':
        from context_features import ContextFeatures as FeatureExtractor
    else:raise ValueError('Unsupported training feature version: '+version)
    import torch
    torch.set_num_threads(1)
    report,events,_=load_source(job);doc=read_labels(job,report['source_sha256'])
    classifier,_=extractor(job);features=FeatureExtractor(classifier)
    predictions=[];failures=[];counts=manifest['class_counts']
    for original in events:
        annotation=doc['labels'].get(original['id'])
        event={**original,**annotation} if annotation and not annotation.get('wrong_target') else original
        try:
            vector,quality=features.extract(event)
            row=scores(model,vector.astype(float)[None])[0];order=np.argsort(-row)[:3]
        except (ValueError,IndexError) as error:
            failures.append({'event_id':event['id'],'reason':str(error)});continue
        code=str(model['classes'][order[0]]);reasons=['跨录像候选，尚未人工核对']
        if counts.get(code,0)<3:reasons.append('训练录像中该类少于3条')
        if quality['target_pose_fraction']<.5:reasons.append('本录像目标姿态不足50%')
        if quality['filtered_raw_shuttle_fraction']<.25:reasons.append('本拍观测球轨迹不足25%')
        predictions.append({'event_id':event['id'],'stroke_type':code,'stroke_label':LABELS[code],
            'ranking':[{'code':str(model['classes'][i]),'label':LABELS[str(model['classes'][i])],
                        'uncalibrated_score':float(row[i])} for i in order],
            'used_for_training':False,'human_reviewed_input':bool(annotation),
            'needs_confirmation':True,'review_status':'unreviewed_cross_video_candidate',
            'quality_reasons':reasons,'input_quality':quality})
    result={'version':VERSION,'job_id':job,'source_sha256':report['source_sha256'],
            'dataset_revision':doc['revision'],'feature_version':version,'events':predictions,'failures':failures,
            'run_id':run_id,'training_job_id':training_job,'training_dataset_revision':manifest['dataset_revision'],
            'model_file':str(model_path.relative_to(DATA)),'model_sha256':file_hash(model_path),
            'manifest_sha256':file_hash(source_folder/'manifest.json'),'code_sha256':file_hash(Path(__file__)),
            'inference_feature_code_sha256':file_hash(Path(__file__).with_name('stroke_features.py' if version=='local-hit-v1' else 'context_features.py')),
            'created_at':datetime.now(timezone.utc).isoformat(),'scope':'Frozen human-label head replay, no target-video label accuracy measured.',
            'same_athlete_confirmed':False,
            'training_candidate_sides':sorted({training_snapshot['labels'][eid]['player'] for eid in manifest['trained_event_ids']}),
            'training_confirmed_contact_count':sum(training_snapshot['labels'][eid].get('contact_confirmed',False) for eid in manifest['trained_event_ids']),
            'model_retrained':False,
            'global_model_activated':False,'independent_validation':None,
            'human_labels_created':0,'pseudo_labels_created':0,'evaluation':None}
    atomic_json(folder_for(job)/'personal_transfer_predictions.json',result)
    print('TRANSFER_HINTS',job,len(predictions),'failures',len(failures),flush=True)
    return result


def replay_latest(job):
    """New specialist jobs reuse the latest completed local head as hints."""
    if (folder_for(job)/'personal_predictions.json').is_file():return None
    choices=[]
    for state_path in (DATA/'击球标注').glob('J_*/learning_state.json'):
        if state_path.parent.name==job:continue
        state=json.loads(state_path.read_text())
        if state.get('status')!='completed' or not state.get('run_id'):continue
        manifest_path=state_path.parent/'学习记录'/state['run_id']/'manifest.json'
        if not manifest_path.is_file():continue
        manifest=json.loads(manifest_path.read_text())
        if (manifest.get('feature_version')!='local-hit-v1'
                or manifest.get('training_target_source')!='human_user_selection_only'):continue
        choices.append((manifest['created_at'],state_path.parent.name,state['run_id']))
    if not choices:return None
    _,source,run=max(choices)
    return replay(job,source,run)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True);parser.add_argument('--training-job',required=True);parser.add_argument('--run-id',required=True)
    args=parser.parse_args();replay(args.job,args.training_job,args.run_id)
