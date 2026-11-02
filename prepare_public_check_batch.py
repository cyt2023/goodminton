"""Prepare a larger frozen-model review batch from user-owned match footage.

No classifier training and no human labels generated. Uses one local execution
lock, keeps old frozen predictions, and processes new footage sequentially.
"""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import uuid

from analysis_jobs import DATA, ROOT, atomic_json, file_hash, registered_video
from specialist_model import ENGINE_VERSION, MODEL_ID, MODEL_REVISION

OUT = DATA/'击球标注/公开模型检查批次'
NEW_VIDEOS = ['ba2ffd43e10c5b44b8444b76e0d1fba1', 'e5e69c101f515c8e9664da9fb7b02303']
OLD_JOB = 'J_e23c1dd9297d4bfca7c91d1d95065b68'
NOW = lambda: datetime.now(timezone.utc).isoformat()


def run(command, log, message, batch):
    batch['message'] = message;atomic_json(OUT/'latest.json',batch)
    with log.open('a') as stream:
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in child.stdout:
            stream.write(line);stream.flush()
            if any(line.startswith(prefix) for prefix in ['STATUS ', 'PROGRESS ', 'DONE ', 'PUBLIC_MODEL_FROZEN']):
                print(message, line[:320].strip(),flush=True)
    if child.wait() != 0:
        raise RuntimeError(message+'未完成，见日志 '+str(log))


def sync_index(job, config):
    classified = json.loads((Path(config['output_folder'])/'classified_events.json').read_text())
    path = DATA/'技术动作'/job/'index.json'
    state = json.loads(path.read_text()) if path.is_file() else {}
    candidates = json.loads((DATA/'专项分析'/job/'hit_candidates.json').read_text())['candidates']
    state.update(schema_version=1,engine_version=ENGINE_VERSION,pose_id=job,
                 source_video_id=json.loads(Path(config['pose_file']).read_text())['source_video_id'],
                 source_sha256=config['source_sha256'],pose_fingerprint=config['pose_fingerprint'],
                 model_id=MODEL_ID,model_revision=MODEL_REVISION,
                 pose_file=config['pose_file'],source_file=config['source_file'],
                 status='completed',message='两位球员的公开模型人工检查候选已准备',
                 candidates=candidates,requested_sides=['near','far'],events=classified['events'],
                 processed=len(classified['events']),total=len(classified['events']),error=None,
                 created_at=state.get('created_at',NOW()),updated_at=NOW())
    atomic_json(path,state)
    return len(classified['events'])


def pipeline(job, source, pose, batch, done=()):
    folder=DATA/'技术动作'/job;folder.mkdir(parents=True,exist_ok=True)
    ffmpeg=next((ROOT/'.venv-pose/lib').glob('python*/site-packages/imageio_ffmpeg/binaries/ffmpeg-*'))
    config={'pose_id':job,'source_sha256':file_hash(source),'pose_fingerprint':file_hash(pose),
            'pose_file':str(pose.resolve()),'source_file':str(source.resolve()),
            'ffmpeg':str(ffmpeg.resolve()),'output_folder':str((folder/'片段').resolve()),
            'classifier_sha256':file_hash(ROOT/'specialist_pipeline.py'),'engine_version':ENGINE_VERSION,
            'specialist_folder':str((DATA/'专项分析'/job).resolve()),
            'requested_sides':['near','far'],'done_ids':list(done)}
    config_path=folder/'config.json';atomic_json(config_path,config)
    run([str(ROOT/'.venv-vision/bin/python'),'-u',str(ROOT/'specialist_pipeline.py'),
         '--config',str(config_path),'--bst-only'],OUT/(job+'_specialist.log'),'生成两侧逐拍候选与冻结新旧预测',batch)
    return sync_index(job,config)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(DATA/'技术动作/.execution.lock').open('a+b');fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
    previous = json.loads((OUT/'latest.json').read_text()) if (OUT/'latest.json').is_file() else None
    batch={'version':'expanded-public-check-v1','id':uuid.uuid4().hex,'created_at':NOW(),
           'status':'preparing','message':'准备更多回合内候选','jobs':[],
           'model_training_requested':False,'human_labels_created':0,
           'user_reported_between_rallies_count_in_old_batch':2,
           'reported_outside_event_ids':['6873586c37c654f386164ebfce25056f','b57dfd22f1ea3a13da31aff29b8ffaba','c86751763ffa7e715995d23d00f7911c']}
    batch['user_reported_between_rallies_count_in_old_batch']=3
    if previous and previous['status']=='failed':
        batch['id']=previous['id'];batch['created_at']=previous['created_at'];batch['jobs']=previous['jobs']
        batch['resume_after_error']=previous['message']
    try:
        if not any(r['job_id']==OLD_JOB for r in batch['jobs']):
            old_config=json.loads((DATA/'技术动作'/OLD_JOB/'config.json').read_text())
            current=json.loads((DATA/'技术动作'/OLD_JOB/'片段/classified_events.json').read_text())
            count=pipeline(OLD_JOB,Path(old_config['source_file']),Path(old_config['pose_file']),batch,
                           [r['id'] for r in current['events']])
            batch['jobs'].append({'job_id':OLD_JOB,'name':'录像一：补充另一侧球员','candidate_count':count})
        atomic_json(OUT/'latest.json',batch)
        inventory={r['video_id']:r for r in json.loads((DATA/'录像清单.json').read_text())}
        for video_id in NEW_VIDEOS:
            info=inventory[video_id];record=registered_video(video_id);source=record['path']
            if any(r.get('source_video_number')==info['number'] for r in batch['jobs']):continue
            existing=[]
            for candidate in (DATA/'姿态分析').glob('J_*/result.json'):
                if json.loads(candidate.read_text()).get('source_video_id')==video_id:existing.append(candidate.parent.name)
            job=existing[-1] if existing else 'J_'+uuid.uuid4().hex
            folder=DATA/'分析任务'/job;folder.mkdir(parents=True,exist_ok=True)
            scout_end=__import__('math').floor(info['duration_seconds']*5)/5
            config={'sample_fps':5,'display_threshold':.5,'detector_threshold':.4,'max_width':960,'publish_results':True,
                    'jobs':[{'id':job,'video_id':video_id,'label':'公开模型检查 · 新录像'+str(info['number']),
                             'start':0.,'end':scout_end,'players':{},'mode':'auto'}]}
            cfg=folder/'config.json';atomic_json(cfg,config)
            run([str(ROOT/'.venv-pose/bin/python'),'-u',str(ROOT/'analyze_pose.py'),'--config',str(cfg)],
                OUT/(job+'_coarse.log'),'整理新录像'+str(info['number'])+'的球员与画面',batch)
            pose=DATA/'姿态分析'/job/'result.json'
            count=pipeline(job,source,pose,batch)
            # Complete task registration after all actual stages succeed.
            coarse=json.loads(pose.read_text())
            task={'id':job,'video_id':video_id,'label':config['jobs'][0]['label'],'start':0.,'end':scout_end,
                  'players':{},'sample_fps':5,'mode':'auto','status':'completed','message':'公开模型检查片段已准备',
                  'error':None,'processed_frames':len(coarse['frames']),'total_frames':len(coarse['frames']),
                  'result_id':job,'created_at':batch['created_at'],'updated_at':NOW(),'started_at':batch['created_at'],
                  'finished_at':NOW(),'source_stat':record['source_stat'],'source_sha256':file_hash(source),
                  'cancel_requested':False,'config_sha256':file_hash(cfg)}
            atomic_json(folder/'job.json',task)
            batch['jobs'].append({'job_id':job,'name':'录像'+str(len(batch['jobs'])+1)+'：新的连续回合录像',
                                  'candidate_count':count,'source_video_number':info['number']})
            atomic_json(OUT/'latest.json',batch)
        batch.update(status='completed',message='多录像检查候选已准备；回合状态仍需人工核对',finished_at=NOW())
        atomic_json(OUT/'运行记录'/batch['id']/'result.json',batch);atomic_json(OUT/'latest.json',batch)
        print('EXPANDED_BATCH_READY',json.dumps(batch,ensure_ascii=False),flush=True)
    except Exception as error:
        batch.update(status='failed',message=str(error)[:600]);atomic_json(OUT/'latest.json',batch);raise


if __name__=='__main__':main()
