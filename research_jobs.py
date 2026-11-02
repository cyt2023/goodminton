"""Persisted single-worker research integration, alongside existing UI/jobs."""
from datetime import datetime,timezone
import fcntl
import json
from pathlib import Path
import re
import subprocess
import threading
import time

from analysis_jobs import DATA,atomic_json,file_hash
from research_adapters import registry

ROOT=Path(__file__).resolve().parent
STORE=DATA/'研究集成'


def folder_for(job):
    if not isinstance(job,str) or not re.fullmatch(r'J_[0-9a-f]{32}',job):raise ValueError('录像任务标识无效')
    return STORE/job


def load(path,default=None):return json.loads(path.read_text()) if path.exists() else default


def result_folder(job):
    local=folder_for(job)
    if (local/'research_report.json').exists():return local
    return DATA/'专项分析'/job


class ResearchJobs:
    def __init__(self):
        STORE.mkdir(exist_ok=True);self.lock=threading.RLock();self.stop=threading.Event();self.process=None
        for f in STORE.glob('J_*/task.json'):
            state=load(f)
            if state['status']=='running':
                state.update(status='failed',message='上次集成分析中断，可重试，已完成的模型缓存保留');atomic_json(f,state)
        for f in STORE.glob('J_*/coaching/*.json'):
            state=load(f,{})
            if state.get('status') in ['queued','running']:
                state.update(status='failed',message='上次本地教练解读中断，可以重新生成');atomic_json(f,state)
        self.worker=threading.Thread(target=self.loop,name='research-integration',daemon=True);self.worker.start()

    def submit(self,job):
        folder=folder_for(job)
        if not (DATA/'专项分析'/job/'dense_pose.json').exists():raise ValueError('请先完成球员和逐拍识别')
        with self.lock:
            state=load(folder/'task.json',{})
            if state.get('status') in ['queued','running']:return state
            active=sum(load(f).get('status') in ['queued','running'] for f in STORE.glob('J_*/task.json'))
            if active>=3:raise ValueError('已有集成分析排队，请稍后再试')
            state={'job_id':job,'status':'queued','message':'等待运行 RacketVision 并融合现有击球结果','created_at':datetime.now(timezone.utc).isoformat()}
            atomic_json(folder/'task.json',state);return state

    def snapshot(self,job):
        folder=folder_for(job);report_path=result_folder(job)/'research_report.json'
        state=load(folder/'task.json',{'job_id':job,'status':'not_started','message':'可使用现有录像开始集成分析'})
        report=load(report_path)
        return {**state,'report_available':report is not None,'model_status':registry(),
                'summary':report['summary'] if report else None,
                'source_video_id':report.get('source_video_id') if report else None,
                'status':'completed' if report is not None and state['status'] not in ['queued','running','failed'] else state['status'],
                'message':'集成分析已保存' if report is not None and state['status']=='not_started' else state['message']}

    def loop(self):
        while not self.stop.is_set():
            queued=[(f,load(f)) for f in STORE.glob('J_*/task.json') if load(f).get('status')=='queued']
            if not queued:self.stop.wait(1);continue
            f,state=min(queued,key=lambda v:v[1]['created_at'])
            state.update(status='running',message='等待模型执行，已有结果保留');atomic_json(f,state)
            try:
                # Same lock as the existing action worker: no overlapping heavy models.
                execution=DATA/'技术动作/.execution.lock';execution.parent.mkdir(parents=True,exist_ok=True)
                with execution.open('a+b') as lock:
                    while not self.stop.is_set():
                        try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);break
                        except BlockingIOError:self.stop.wait(.5)
                    if self.stop.is_set():return
                    command=[str(ROOT/'.venv-vision/bin/python'),'-u',str(ROOT/'research_worker.py'),'--job',state['job_id']]
                    self.process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
                    with (f.parent/'worker.log').open('a') as log:
                        for line in self.process.stdout:
                            log.write(line);log.flush()
                            if '_PROGRESS ' in line:
                                progress=json.loads(line.split('_PROGRESS ',1)[1]);state.update(message='正在跟踪球和球拍',processed=progress['processed'],total=progress['total']);atomic_json(f,state)
                    if self.process.wait()!=0:raise RuntimeError('模型集成未完成，请查看运行记录后重试')
                state.update(status='completed',message='集成分析已保存')
            except Exception as error:
                state.update(status='failed',message=str(error)[:500],log_url=f'/api/integrated-analysis/{state["job_id"]}/log')
            finally:
                self.process=None;atomic_json(f,state)

    def shutdown(self):
        self.stop.set()
        if self.process and self.process.poll() is None:self.process.terminate()
        self.worker.join(timeout=5)


def report(job,refresh=False):
    folder=result_folder(job);path=folder/'research_report.json'
    if not path.exists():raise ValueError('集成分析尚未完成')
    annotation=DATA/'击球标注'/job/'annotations.json'
    imports=folder/'research_inputs'
    changed=annotation.exists() and annotation.stat().st_mtime>path.stat().st_mtime
    changed=changed or any(f.stat().st_mtime>path.stat().st_mtime for f in imports.glob('*.json'))
    extra=DATA/'专项分析'/job/'pose3d/pose3d.json'
    changed=changed or (extra.exists() and extra.stat().st_mtime>path.stat().st_mtime)
    cached=load(path);fingerprint=cached.get('execution_provenance',{})
    changed=changed or any(fingerprint.get(name)!=file_hash(ROOT/name) for name in ['research_report.py','research_quality.py','research_pose3d.py'])
    if refresh or changed:
        from research_report import build
        dense=load(DATA/'专项分析'/job/'dense_pose.json')
        value=build(job,folder=folder,dense=dense)
    else:value=load(path)
    # Coaching is an evidence-bound derivative, never an input label.
    for event in value['events']:
        feedback=load(folder/'coaching'/(event['event_id']+'.json'))
        if feedback and feedback.get('status')=='completed' and feedback.get('evidence_hash')==coaching_hash(event):event['coaching_feedback']=feedback
    return value


def overlays(job):
    folder=result_folder(job);r=load(folder/'racketvision/racketvision.json',{})
    dense=load(DATA/'专项分析'/job/'dense_pose.json');traj=load(folder/'trajectory.json')
    if dense is None or traj is None:raise ValueError('叠加数据尚未完成')
    return {'source_sha256':dense['source_sha256'],'frames':[{'frame_index':i,'timestamp':p['source_time'],
        'players':{side:({'bbox':v['bbox'],'keypoints':[[k['x'],k['y'],k['score']] for k in v['keypoints']]} if v else None) for side,v in p['players'].items()},
        'shuttle':traj['frames'][i]['raw_xy'],
        'rackets':r.get('frames',[])[i]['rackets'] if i<len(r.get('frames',[])) else []} for i,p in enumerate(dense['frames'])]}


def import_results(job,component,packet):
    value=report(job);folder=result_folder(job)
    from research_adapters import import_predictions
    checked=import_predictions(packet,component,value['source_sha256'],value['timestamps'],value['nominal_fps'])
    if packet.get('predictions') and not checked['predictions']:
        raise ValueError('没有有效预测，未覆盖现有结果：'+checked['rejected'][0]['reason'])
    atomic_json(folder/'research_inputs'/(component+'.json'),packet)
    value=report(job,refresh=True)
    return {'accepted':len(checked['predictions']),'rejected':checked['rejected'],'summary':value['summary']}


def coaching_hash(event):
    import hashlib
    evidence={'event_id':event['event_id'],'hit_frame':event['hit_frame'],'stroke_type':event['stroke_type'],
              'quality_metrics':event['quality_metrics'],'player_id':event['player_id']}
    return hashlib.sha256(json.dumps(evidence,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def start_coaching(job,event_id):
    value=report(job);event=next((e for e in value['events'] if e['event_id']==event_id),None)
    if event is None:raise ValueError('找不到这个击球事件')
    folder=result_folder(job);key=coaching_hash(event);target=folder/'coaching'/(event_id+'.json')
    previous=load(target,{})
    if previous.get('evidence_hash')==key and previous.get('status') in ['queued','running','completed']:return previous
    state={'status':'queued','evidence_hash':key,'event_id':event_id,'message':'等待本地教练模型'}
    atomic_json(target,state)
    def work():
        try:
            lockpath=DATA/'技术动作/.execution.lock'
            with lockpath.open('a+b') as lock:
                fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
                state.update(status='running',message='本地模型正在根据测量证据生成回看重点');atomic_json(target,state)
                request=target.with_suffix('.input.json');atomic_json(request,{'event':event,'output':str(target),'evidence_hash':key})
                with target.with_suffix('.log').open('w') as log:
                    result=subprocess.run([str(ROOT/'.venv-vision/bin/python'),str(ROOT/'research_coaching.py'),'--input',str(request)],stdout=log,stderr=subprocess.STDOUT,timeout=180)
                if result.returncode:raise RuntimeError('本地教练模型生成失败；已保留测量和模板反馈')
        except Exception as error:
            atomic_json(target,{**state,'status':'failed','message':str(error)})
    threading.Thread(target=work,name='local-research-coaching',daemon=True).start()
    return state


def tactical_review(job):
    from tactical_review import build as build_review
    folder=result_folder(job);value=report(job)
    review=build_review(value,load(folder/'research_inputs/rallies.json'))
    cached=load(folder/'tactical_review/feedback.json',{})
    if cached.get('evidence_hash')==review['evidence_hash']:
        review['llm_status']=cached.get('status','not_requested')
        if cached.get('status')=='completed':review['feedback']=cached
    target=folder/'tactical_review/review.json'
    if load(target,{}).get('evidence_hash')!=review['evidence_hash']:atomic_json(target,review)
    return review


def import_rallies(job,packet):
    from tactical_review import validate_rallies
    value=report(job);validate_rallies(packet,value)
    if packet.get('benchmark_only'):raise ValueError('Independent benchmark references are not inference inputs')
    atomic_json(result_folder(job)/'research_inputs/rallies.json',packet)
    return tactical_review(job)


def start_tactical_coaching(job):
    review=tactical_review(job);folder=result_folder(job);target=folder/'tactical_review/feedback.json';previous=load(target,{})
    if not review['candidate_catalog']:raise ValueError('没有足够的结构化证据供LLM选择回看重点')
    if previous.get('evidence_hash')==review['evidence_hash'] and previous.get('status') in ['queued','running','completed']:return previous
    state={'status':'queued','evidence_hash':review['evidence_hash'],'message':'等待本地多层复盘模型'};atomic_json(target,state)
    def work():
        try:
            lockpath=DATA/'技术动作/.execution.lock';lockpath.parent.mkdir(parents=True,exist_ok=True)
            with lockpath.open('a+b') as lock:
                fcntl.flock(lock.fileno(),fcntl.LOCK_EX);state.update(status='running',message='根据已有证据选择复盘重点');atomic_json(target,state)
                request=target.with_suffix('.input.json');atomic_json(request,{'review':review,'output':str(target)})
                with target.with_suffix('.log').open('w') as log:
                    result=subprocess.run([str(ROOT/'.venv-vision/bin/python'),str(ROOT/'tactical_review_worker.py'),'--input',str(request)],stdout=log,stderr=subprocess.STDOUT,timeout=240)
                if result.returncode:raise RuntimeError('本地复盘模型未完成；结构化证据仍可查看')
        except Exception as error:atomic_json(target,{**state,'status':'failed','message':str(error)})
    threading.Thread(target=work,name='tactical-evidence-review',daemon=True).start();return state


def start_pose3d(job,event_id):
    value=report(job);event=next((e for e in value['events'] if e['event_id']==event_id),None)
    if event is None:raise ValueError('找不到3D估计对应的事件')
    target=DATA/'专项分析'/job/'pose3d/status.json';previous=load(target,{})
    if previous.get('status') in ['queued','running']:return previous
    state={'status':'queued','event_id':event_id,'message':'等待可选3D姿态估计'};atomic_json(target,state)
    def work():
        try:
            lockpath=DATA/'技术动作/.execution.lock';lockpath.parent.mkdir(parents=True,exist_ok=True)
            with lockpath.open('a+b') as lock:
                fcntl.flock(lock.fileno(),fcntl.LOCK_EX);state.update(status='running',message='运行MediaPipe3D估计');atomic_json(target,state)
                command=[str(ROOT/'.venv-pose3d/bin/python'),str(ROOT/'pose3d_worker.py'),'--job',job,'--start',str(max(0,event['action_start_timestamp']-.6)),'--end',str(min(value['timestamps'][-1],event['action_end_timestamp']+.6))]
                result=subprocess.run(command,text=True,capture_output=True,timeout=180)
                if result.returncode:raise RuntimeError('可选3D估计失败；原2D结果保留。'+result.stderr[-500:])
                state.update(status='completed',message='3D估计已保存；深度和尺度未经过标定')
        except Exception as error:state.update(status='failed',message=str(error))
        atomic_json(target,state)
    threading.Thread(target=work,name='optional-pose3d',daemon=True).start();return state
