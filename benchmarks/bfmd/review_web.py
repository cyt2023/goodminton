"""Interactive review sessions for the /review page.

Each clip's session is the list of reviewer actions, replayed on a fresh
Rally so state is always reproducible. Models are the frozen verification
agent; the transition prior and suspicion ratios come from dev clips.
"""
import json,threading,time
from pathlib import Path
import numpy as np
from benchmarks.bfmd import review as rv
from benchmarks.bfmd.verify import WIN,ROUTES

SESSIONS=rv.STUDY/'review_sessions'
ROLES=('new_test','dev')
ACTIONS=('confirm','retype','reject','add','dismiss')
_lock=threading.Lock();_models=None;_cache={}

def models():
    global _models
    if _models is None:
        f=rv.read(rv.STUDY/'frozen_verify_v3_racket_audio.json')
        m=rv.Logistic();m.w,m.mu,m.sd=(np.array(f['hit_model'][k]) for k in ['w','mu','sd'])
        t=rv.Softmax();t.W,t.mu,t.sd=(np.array(f['type_model'][k]) for k in ['W','mu','sd'])
        dev=[c for c in rv.read(rv.STUDY/'clips.json')['clips'] if rv.in_role(c,'dev') and (ROUTES/c['clip_id']).exists()]
        flat=np.full((len(rv.TYPES),)*2,1/len(rv.TYPES));tr=[rv.Rally(c,m,f['threshold'],t,flat,0) for c in dev]
        _models={'model':m,'th':f['threshold'],'tmodel':t,'trans':rv.transitions(tr),'lr':rv.suspicion_lr(tr),'miss':rv.miss_model(tr),'parity':rv.parity_model(tr)}
    return _models

def clip_by_id(clip_id):
    for c in rv.read(rv.STUDY/'clips.json')['clips']:
        if c['clip_id']==clip_id and any(rv.in_role(c,r) for r in ROLES) and (ROUTES/clip_id).exists():return c
    raise ValueError('回合不存在')

def listing():
    out=[]
    for c in rv.read(rv.STUDY/'clips.json')['clips']:
        role=next((r for r in ROLES if rv.in_role(c,r)),None)
        if role and (ROUTES/c['clip_id']).exists():
            done=len(json.loads((SESSIONS/f"{c['clip_id']}.json").read_text())['actions']) if (SESSIONS/f"{c['clip_id']}.json").exists() else 0
            out.append({'clip_id':c['clip_id'],'role':role,'match':c['match_id'],'duration':round(c['duration_seconds'],1),'reviewed':done})
    return {'clips':out}

def apply(r,i,action,label=None):
    """Reviewer decision on candidate i; returns whether the log changed."""
    x=r.c[i];was=x['label'];x['locked']=True
    if action=='reject' and x['inlog']:x['inlog']=False
    elif action=='confirm' and x['inlog']:x['clamp']=x['label']
    elif action=='retype' and x['inlog'] and label in rv.TYPES:x['clamp']=label
    elif action=='add' and not x['inlog'] and label in rv.TYPES:x['inlog']=True;x['clamp']=label
    elif action=='dismiss' and not x['inlog']:pass
    else:raise ValueError('操作与该候选不匹配')
    wrong=action in ('reject','add') or (action=='retype' and label!=was)
    for k in r.neighbours(i):r.c[k]['odds']*=models()['lr'][wrong]
    r.decode();return wrong

def session(clip_id):
    with _lock:
        c=clip_by_id(clip_id);f=SESSIONS/f'{clip_id}.json'
        actions=json.loads(f.read_text())['actions'] if f.exists() else []
        key=(clip_id,len(actions))
        if key not in _cache:
            m=models();r=rv.Rally(c,m['model'],m['th'],m['tmodel'],m['trans'],0)
            for a in actions:apply(r,a['index'],a['action'],a.get('label'))
            _cache.clear();_cache[key]=r
        return c,_cache[key],actions

def act(clip_id,body):
    i,action,label=int(body['index']),body['action'],body.get('label')
    if action not in ACTIONS:raise ValueError('未知操作')
    c,r,actions=session(clip_id)
    if not 0<=i<len(r.c) or r.c[i]['locked']:raise ValueError('该候选已核对或不存在')
    actions=actions+[{'index':i,'action':action,'label':label,'at':time.time()}]
    SESSIONS.mkdir(parents=True,exist_ok=True);(SESSIONS/f'{clip_id}.json').write_text(json.dumps({'actions':actions},ensure_ascii=False))
    return state(clip_id)

def reset(clip_id):
    clip_by_id(clip_id);(SESSIONS/f'{clip_id}.json').unlink(missing_ok=True);return state(clip_id)

def top(probs,n=3):
    return [[k,round(float(v),3)] for k,v in sorted(probs.items(),key=lambda kv:-kv[1])[:n]]

def evidence(r,i,routes,xy):
    x=r.c[i];f=x['frame'];p0,p1=x['probs']
    agree={k:any(abs(g-f)<=WIN for g in v) for k,v in routes.items()}
    post=dict(zip(rv.TYPES,x['post'])) if x.get('post') is not None else {}
    up=lambda q:x['odds']*q/(1-q+1e-9)/(1+x['odds']*q/(1-q+1e-9))
    ev={'index':i,'time':round(x['time'],3),'frame':f,'inlog':x['inlog'],'locked':x['locked'],'label':x['label'],
        'hitter':'near' if x['traj'][14]>.5 else 'far','p_hit':round(x['p_hit'],3),
        'p_false':round(up(1-x['p_hit']),3) if x['inlog'] else None,'p_type':round(up(1-max(x['post'])),3) if x['inlog'] and x.get('post') is not None else None,
        'p_real':round(up(float(models()['miss'].predict([[x['p_hit'],x['racket']]])[0])),3) if not x['inlog'] else None,
        'racket':round(float(np.expm1(x['racket'])),2),
        'sound':round(float(np.expm1(x['x'][11])),2),
        'same_player':round(float(x.get('viol',0.)),3) if x['inlog'] else None,
        'detectors':agree,'support':sum(agree.values()),
        'stacked':top(post),'bst0':top(p0),'bst_cg_ap':top(p1) if p1 else [],
        'author_window':top(x['alt'][0]) if x.get('alt') else [],
        'turn':round(float(x['x'][2]),3),'speed':round(float(x['x'][3]),3),
        'path':[None if np.isnan(xy[t]).any() else [round(float(xy[t][0]),1),round(float(xy[t][1]),1)] for t in range(max(0,f-15),min(len(xy),f+16))],
        'path_start':max(0,f-15)}
    pr=rv.priority(r,x,'gain',{'miss':models()['miss'],'parity':models()['parity']});ev['priority']=round(pr,4) if pr is not None else None
    ev['reason']=None if x['locked'] else ('可能漏了一拍' if not x['inlog'] else '可能不是击球' if ev['p_false']>=(ev['p_type'] or 0) else '球种可能判错')
    return ev

def state(clip_id):
    c,r,actions=session(clip_id)
    folder=ROUTES/clip_id;routes={}
    for fp in sorted(folder.glob('*__*.json')):
        if fp.stem.startswith('agent'):continue
        d=rv.read(fp)
        if d['status']=='completed':routes[fp.stem]=[round(h['time']*30) for h in d['hits']]
    from benchmarks.bfmd.refine import track
    xy=track(rv.read(rv.STUDY/'perception'/clip_id/'tracknetv3/trajectory.json'))
    items=[evidence(r,i,routes,xy) for i in range(len(r.c))]
    queue=sorted([e for e in items if e['priority'] is not None],key=lambda e:-e['priority'])
    log=[e for e in items if e['inlog']]
    expected=sum((e['p_false'] or 0)+(1-(e['p_false'] or 0))*(e['p_type'] or 0) for e in log if not e['locked'])+sum(e['p_real'] or 0 for e in items if not e['inlog'] and not e['locked'])
    tp,fp,fn=r.counts()
    return {'clip':{'clip_id':clip_id,'match':c['match_id'],'duration':c['duration_seconds'],'video':f'/review-video/{clip_id}'},
            'routes':{k:[round(v/30,3) for v in vs] for k,vs in routes.items()},'items':items,'queue':[e['index'] for e in queue],
            'reviewed':len(actions),'review_seconds':round(sum(rv.COST['miss'] if a['action'] in ('add','dismiss') else rv.COST['hit'] for a in actions),1),
            'expected_errors':round(expected,2),'log_size':len(log),
            'reference':{'joint_f1':round(2*tp/(2*tp+fp+fn),3) if 2*tp+fp+fn else None,'note':'BFMD 参考答案，仅研究演示'},
            'types':rv.TYPES}

def video(clip_id):
    p=Path(clip_by_id(clip_id)['path'])
    if not p.is_file():raise FileNotFoundError('回合视频不存在')
    return p

def frame(clip_id,t):
    """JPEG of the rally video at time t (poster for the selected hit)."""
    import cv2
    cap=cv2.VideoCapture(str(video(clip_id)));cap.set(cv2.CAP_PROP_POS_MSEC,max(0.,float(t))*1000);ok,img=cap.read();cap.release()
    if not ok:raise FileNotFoundError('无法读取该时刻的画面')
    return cv2.imencode('.jpg',img,[cv2.IMWRITE_JPEG_QUALITY,85])[1].tobytes()
