"""Run original BST on two opening rallies from the official Tokyo final.

No classifier training, personal heads or human labels. All original-file offsets
are retained. This checks our frontend plus original BST, not paper reproduction.
"""
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import sqlite3
import subprocess
import uuid

from analysis_jobs import DATA, ROOT, DB, atomic_json, file_hash, registered_video
from prepare_public_check_batch import sync_index
from specialist_model import BST_NAME

OUT = DATA/'职业录像/东京奥运会2021男单决赛'
SOURCE_URL = 'https://www.youtube.com/watch?v=uIj03RsGrJA'
WINDOWS = [(342., 374.), (391.5, 403.)]


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source=OUT/'官方完整录像.mp4'
    if not source.is_file():raise FileNotFoundError(source)
    original_sha=file_hash(source)
    ffmpeg=next((ROOT/'.venv-pose/lib').glob('python*/site-packages/imageio_ffmpeg/binaries/ffmpeg-*')).resolve()
    lock=(DATA/'技术动作/.execution.lock').open('a+b')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
    state={'version':'tokyo-original-bst-v1','status':'running','created_at':now(),
           'source_url':SOURCE_URL,'source_sha256':original_sha,
           'baseline_checkpoint':str(ROOT/'models/bst'/BST_NAME),
           'baseline_model_sha256':file_hash(ROOT/'models/bst'/BST_NAME),
           'personal_training_samples':0,'training_executed':False,
           'uses_personal_classifier':False,'human_labels_created':0,'jobs':[],
           'scope':'Two selected opening rallies; not full-match inference, no stroke accuracy until independent labels.'}
    atomic_json(OUT/'benchmark.json',state)
    def run(command, log, stage):
        state['stage']=stage;atomic_json(OUT/'benchmark.json',state)
        with log.open('w') as stream:
            child=subprocess.Popen([str(x) for x in command],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in child.stdout:
                stream.write(line);stream.flush()
                if line.startswith(('STATUS ','PROGRESS ','DENSE_DONE ','TRAJECTORY_DONE ','RESULT ','BENCHMARK_DONE')):
                    print(stage,line[:350].strip(),flush=True)
        if child.wait():raise RuntimeError(stage+'失败，见'+str(log))
    try:
        for number,(start,end) in enumerate(WINDOWS,1):
            video_id=uuid.uuid5(uuid.NAMESPACE_URL,f'{SOURCE_URL}:benchmark:{start}:{end}').hex
            job='J_'+uuid.uuid5(uuid.NAMESPACE_URL,f'{video_id}:original-bst-v1').hex
            clip=OUT/f'开局回合{number}_{start:g}-{end:g}.mp4'
            if not clip.is_file():
                run([ffmpeg,'-y','-v','error','-ss',start,'-i',source,'-t',end-start,
                     '-map','0:v:0','-map','0:a:0?','-c:v','libx264','-crf','18','-preset','fast',
                     '-c:a','aac','-movflags','+faststart',clip],OUT/f'回合{number}_截取.log','截取开局回合'+str(number))
            with sqlite3.connect(DB) as db:
                db.execute('INSERT OR REPLACE INTO videos(id,name,kind,path) VALUES(?,?,?,?)',
                           (video_id,f'东京决赛开局回合{number}（原片{start:g}–{end:g}秒）','reference',str(clip.relative_to(DATA))))
            registered=registered_video(video_id,allow_reference=True)
            import cv2
            cap=cv2.VideoCapture(str(clip));duration=cap.get(cv2.CAP_PROP_FRAME_COUNT)/cap.get(cv2.CAP_PROP_FPS);cap.release()
            scout_end=int(duration*5)/5
            task_folder=DATA/'分析任务'/job;task_folder.mkdir(parents=True,exist_ok=True)
            label=f'东京决赛 · 原BST检查回合{number}'
            cfg={'sample_fps':5,'display_threshold':.5,'detector_threshold':.4,'max_width':960,'publish_results':True,
                 'jobs':[{'id':job,'video_id':video_id,'label':label,'start':0.,'end':scout_end,'players':{},'mode':'auto'}]}
            coarse_config=task_folder/'config.json';atomic_json(coarse_config,cfg)
            pose=DATA/'姿态分析'/job/'result.json'
            if not pose.is_file():run([ROOT/'.venv-pose/bin/python','-u',ROOT/'analyze_pose.py','--config',coarse_config],OUT/f'回合{number}_粗姿态.log','筛选球员'+str(number))
            folder=DATA/'技术动作'/job;folder.mkdir(parents=True,exist_ok=True)
            config={'pose_id':job,'source_sha256':file_hash(clip),'pose_fingerprint':file_hash(pose),
                    'pose_file':str(pose.resolve()),'source_file':str(clip.resolve()),'ffmpeg':str(ffmpeg),
                    'output_folder':str((folder/'片段').resolve()),'specialist_folder':str((DATA/'专项分析'/job).resolve()),
                    'classifier_sha256':file_hash(ROOT/'specialist_pipeline.py'),
                    'requested_sides':['near','far'],'done_ids':[],'benchmark_only':True}
            config_path=folder/'config.json';atomic_json(config_path,config)
            run([ROOT/'.venv-vision/bin/python','-u',ROOT/'specialist_pipeline.py','--config',config_path,'--bst-only'],OUT/f'回合{number}_专项.log','原BST推理'+str(number))
            count=sync_index(job,config)
            coarse=json.loads(pose.read_text())
            atomic_json(task_folder/'job.json',{'id':job,'video_id':video_id,'label':label,'start':0.,'end':scout_end,
                'players':{},'sample_fps':5,'mode':'auto','status':'completed','message':'职业转播原BST检查完成',
                'error':None,'processed_frames':len(coarse['frames']),'total_frames':len(coarse['frames']),
                'result_id':job,'created_at':state['created_at'],'updated_at':now(),'started_at':state['created_at'],
                'finished_at':now(),'source_stat':registered['source_stat'],'source_sha256':config['source_sha256'],
                'cancel_requested':False,'config_sha256':file_hash(coarse_config),'reference_benchmark':True})
            item={'job_id':job,'video_id':video_id,'clip_file':str(clip.relative_to(DATA)),
                  'source_offset_seconds':start,'source_end_seconds':end,'classified_candidates':count,
                  'clip_sha256':config['source_sha256'],'review_url':f'http://127.0.0.1:8765/label-strokes?job={job}&batch=raw'}
            atomic_json(DATA/'专项分析'/job/'reference_source.json',{**item,'official_source_url':SOURCE_URL,'official_source_sha256':original_sha})
            state['jobs'].append(item);atomic_json(OUT/'benchmark.json',state)
        state.update(status='completed',finished_at=now());atomic_json(OUT/'benchmark.json',state)
        print('TOKYO_BENCHMARK_READY',json.dumps(state,ensure_ascii=False),flush=True)
    except Exception as error:
        state.update(status='failed',error=str(error));atomic_json(OUT/'benchmark.json',state);raise


if __name__=='__main__':main()
