"""Interactive review sessions for the /review page.

Each clip's session is the list of reviewer actions, replayed on a fresh
Rally so state is always reproducible. A clip of match m is served by the
frozen bundle for m (frozen_review_lomo12.json): verification agent, evidence
error models and suspicion ratios fitted on the other 11 matches only.
"""
import json,threading,time
from pathlib import Path
import numpy as np
from benchmarks.bfmd import review as rv
from benchmarks.bfmd.verify import WIN,ROUTES,PRIMARY,truth_labels,truth_types

SESSIONS=rv.STUDY/'review_sessions'
ACTIONS=('confirm','retype','reject','add','dismiss','retime')
_lock=threading.Lock();_models=None;_cache={}

def bundle(match):
    """Frozen models for clips of one match, fitted without that match."""
    global _models
    if _models is None:_models={}
    if match not in _models:
        b=rv.read(rv.STUDY/'frozen_review_lomo12.json')['bundles'][match]
        m=rv.Logistic();m.w,m.mu,m.sd=(np.array(b['hit_model'][k]) for k in ['w','mu','sd'])
        t=rv.Softmax();t.W,t.mu,t.sd=(np.array(b['type_model'][k]) for k in ['W','mu','sd'])
        h=None
        if b.get('head'):h=rv.Softmax();h.W,h.mu,h.sd=(np.array(b['head'][k]) for k in ['W','mu','sd'])
        _models[match]={'model':m,'th':b['threshold'],'tmodel':t,'E':rv.ErrorModels.load(b['error_models']),'head':h,'auto':b.get('auto_pass'),
                        'lr':{k=='True':v for k,v in b['suspicion_lr'].items()},'trans':np.full((len(rv.TYPES),)*2,1/len(rv.TYPES))}
    return _models[match]

def clips():
    return [c for c in rv.read(rv.STUDY/'clips.json')['clips'] if 'path' in c and (ROUTES/c['clip_id']/f'{PRIMARY}.json').exists()]

def clip_by_id(clip_id):
    for c in clips():
        if c['clip_id']==clip_id:return c
    raise ValueError('回合不存在')

def listing():
    out=[]
    for c in clips():
        done=len(json.loads((SESSIONS/f"{c['clip_id']}.json").read_text())['actions']) if (SESSIONS/f"{c['clip_id']}.json").exists() else 0
        out.append({'clip_id':c['clip_id'],'match':c['match_id'],'duration':round(c['duration_seconds'],1),'reviewed':done})
    return {'clips':out}

def valid(r,i,action,label=None,t=None):
    """Whether the action fits candidate i (checked before it is stored, so a stored session always replays)."""
    x=r.c[i]
    if action in ('confirm','reject'):return x['inlog']
    if action=='retype':return x['inlog'] and label in rv.TYPES
    if action=='retime':return x['inlog'] and t is not None and 0<=t<=r.clip['duration_seconds']
    if action=='add':return not x['inlog'] and label in rv.TYPES
    return action=='dismiss' and not x['inlog']

def apply(r,i,action,label=None,t=None):
    """Reviewer decision on candidate i; returns whether the log changed."""
    if not valid(r,i,action,label,t):raise ValueError('操作与该候选不匹配')
    x=r.c[i];was=x['label'];again=x['locked'];x['locked']=True
    if action=='retime':  # the hit is real but logged at the wrong moment: move it, keep or set its type
        x['time']=float(t);x['frame']=round(float(t)*30);x['clamp']=label if label in rv.TYPES else was;x['retimed']=True
        r.y=truth_labels(r.clip,r.c);r.ty=truth_types(r.clip,r.c)
    elif action=='reject':x['inlog']=False
    elif action=='confirm':x['clamp']=x['label']
    elif action in ('retype','add'):x['inlog']=True;x['clamp']=label
    wrong=action in ('reject','add','retime') or (action=='retype' and label!=was)
    if not again:  # a type fix right after moving the hit is the same review, not new evidence about the neighbours
        for k in r.neighbours(i):r.c[k]['odds']*=bundle(r.clip['youtube_id'])['lr'][wrong]
    r.decode();return wrong

def session(clip_id):
    with _lock:
        c=clip_by_id(clip_id);f=SESSIONS/f'{clip_id}.json'
        actions=json.loads(f.read_text())['actions'] if f.exists() else []
        key=(clip_id,len(actions))
        if key not in _cache:
            m=bundle(c['youtube_id']);cs=rv.cached_candidates(c);k=c['clip_id']
            if m['head'] is not None and cs and rv._CANDS[k][0].get('head') is None:  # BFMD stroke-type head on BST features (evidence for the error models)
                emb=rv.bst_embeddings(c)[0]
                for x,p in zip(rv._CANDS[k],rv.proba(m['head'],[emb[x['frame']] for x in rv._CANDS[k]])):x['head']=p
            r=rv.Rally(c,m['model'],m['th'],m['tmodel'],m['trans'],0);before={i for i,x in enumerate(r.c) if x['inlog']}
            if m['auto'] is not None:rv.auto_pass(r,m['E'],m['auto'])  # second automatic check before the reviewer sees the log
            after={i for i,x in enumerate(r.c) if x['inlog']};r.auto={'removed':len(before-after),'added':len(after-before)}
            for a in actions:apply(r,a['index'],a['action'],a.get('label'),a.get('time'))
            _cache.clear();_cache[key]=r
        return c,_cache[key],actions

def act(clip_id,body):
    i,action,label=int(body['index']),body['action'],body.get('label')
    t=float(body['time']) if body.get('time') is not None else None
    if action not in ACTIONS:raise ValueError('未知操作')
    if action=='retime' and t is None:raise ValueError('缺少新的击球时刻')
    c,r,actions=session(clip_id)
    if not 0<=i<len(r.c) or (r.c[i]['locked'] and not (action=='retype' and r.c[i].get('retimed'))):raise ValueError('该候选已核对或不存在')
    if not valid(r,i,action,label,t):raise ValueError('操作与该候选不匹配')
    actions=actions+[{'index':i,'action':action,'label':label,'time':t,'at':time.time()}]
    SESSIONS.mkdir(parents=True,exist_ok=True);(SESSIONS/f'{clip_id}.json').write_text(json.dumps({'actions':actions},ensure_ascii=False))
    return state(clip_id)

def reset(clip_id):
    clip_by_id(clip_id);(SESSIONS/f'{clip_id}.json').unlink(missing_ok=True);return state(clip_id)

def top(probs,n=3):
    return [[k,round(float(v),3)] for k,v in sorted(probs.items(),key=lambda kv:-kv[1])[:n]]

SUGGEST=np.log(8)  # timing suggestion when a sound peak 8x stronger sits >3 frames away: shown on ~5% of hits, about a quarter of them mistimed

def evidence(r,i,routes,xy):
    x=r.c[i];f=x['frame'];p0,p1=x['probs']
    agree={k:any(abs(g-f)<=WIN for g in v) for k,v in routes.items()}
    post=dict(zip(rv.TYPES,x['post'])) if x.get('post') is not None else {}
    up=lambda q:x['odds']*q/(1-q+1e-9)/(1+x['odds']*q/(1-q+1e-9));E=bundle(r.clip['youtube_id'])['E'];fe=rv.evidence_x(r,x)
    pf,pm,pt,pr=rv.error_probs(r,x,E,fe,up)
    ev={'index':i,'time':round(x['time'],3),'frame':f,'inlog':x['inlog'],'locked':x['locked'],'retimed':bool(x.get('retimed')),'label':x['label'],
        'hitter':'near' if x['traj'][14]>.5 else 'far','p_hit':round(x['p_hit'],3),
        'p_false':round(pf,3) if x['inlog'] else None,'p_mistimed':round(pm,3) if x['inlog'] else None,
        'p_type':round(pt,3) if x['inlog'] else None,'p_real':round(pr,3) if not x['inlog'] else None,'auto':x['auto'][0] if x.get('auto') else None,
        'racket':round(float(np.expm1(x['racket'])),2),
        'sound':round(float(np.expm1(x['x'][11])),2),
        'suggest_time':round(x['tev'][0]/30,3) if abs(x['tev'][0]-f)>3 and x['tev'][1]>=SUGGEST else None,'suggest_ratio':round(float(np.exp(x['tev'][1])),1),
        'same_player':round(float(x.get('viol',0.)),3) if x['inlog'] else None,
        'detectors':agree,'support':sum(agree.values()),
        'stacked':top(post),'bst0':top(p0),'bst_cg_ap':top(p1) if p1 else [],
        'author_window':top(x['alt'][0]) if x.get('alt') else [],
        'turn':round(float(x['x'][2]),3),'speed':round(float(x['x'][3]),3),
        'path':[None if np.isnan(xy[t]).any() else [round(float(xy[t][0]),1),round(float(xy[t][1]),1)] for t in range(max(0,f-15),min(len(xy),f+16))],
        'path_start':max(0,f-15)}
    pr=rv.priority(r,x,'evidence',{'E':E});ev['priority']=round(pr,4) if pr is not None else None
    if x['locked']:ev['reason']=None
    elif not x['inlog']:ev['reason']='可能被误删' if x.get('auto') else '可能漏了一拍'
    else:  # the error kind with the largest expected joint-F1 gain
        g={'时刻可能记偏':pm*rv.GAIN['retime'],'可能不是击球':(pf-pm)*rv.GAIN['false'],'球种可能判错':(1-pf)*ev['p_type']*rv.GAIN['type']}
        ev['reason']=max(g,key=g.get)
    return ev

def cost(a,prev):
    """Assumed seconds per action (rv.COST); fixing the type right after moving a hit belongs to the same review."""
    if a['action'] in ('add','dismiss'):return rv.COST['miss']
    if a['action']=='retime':return rv.COST['hit']+rv.COST['retime']
    if a['action']=='retype' and prev and prev['index']==a['index'] and prev['action']=='retime':return 0.
    return rv.COST['hit']

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
            'reviewed':len(actions),'review_seconds':round(sum(cost(a,actions[k-1] if k else None) for k,a in enumerate(actions)),1),
            'expected_errors':round(expected,2),'log_size':len(log),'auto':r.auto,
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
