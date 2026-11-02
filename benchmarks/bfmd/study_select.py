"""Agent pipeline selection for the BFMD study.

votes   : label-free ensemble routes built from stored single-route predictions.
evaluate: leave-one-match-out comparison of selection policies on dev scores.
freeze  : fit the chosen policy on all dev clips and write frozen_policy.json.
apply   : run the frozen policy on sealed-test routes (predictions only).
Selection features never include BFMD references; only scores_dev.json does.
"""
import argparse,json,os,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT.parent/'03_数据/独立基准/BFMD/pipeline_study'
VARIANT=os.environ.get('BFMD_VARIANT','');SUFFIX=f'_{VARIANT}' if VARIANT else '';ROUTES=STUDY/f'routes{SUFFIX}'  # v2 = refined timing + filters
VOTE_LEVELS=[2,3,4,5,6];TOL=.1;RIDGE=3.

def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=1,allow_nan=False)+'\n')
def clips(role):return [c for c in read(STUDY/'clips.json')['clips'] if c['role']==role and (ROUTES/c['clip_id']).exists()]
def single_routes(clip):return {f.stem:read(f) for f in sorted((ROUTES/clip['clip_id']).glob('*.json')) if not f.stem.startswith('vote')}

def matched(a,b):
    """Greedy one-to-one time matching within TOL; returns matched count."""
    used=set();n=0
    for t in sorted(a):
        best=min(((abs(t-u),j) for j,u in enumerate(b) if j not in used and abs(t-u)<=TOL),default=None)
        if best:used.add(best[1]);n+=1
    return n

def build_votes(clip):
    routes={k:r for k,r in single_routes(clip).items() if r['status']=='completed' and r['method']!='gru_original'}
    events=sorted(((h['time'],k,h) for k,r in routes.items() for h in r['hits']),key=lambda x:x[0]);clusters=[]
    for t,k,h in events:
        c=clusters[-1] if clusters else None
        if c and t-c['start']<=TOL and k not in c['routes']:c['members'].append(h);c['routes'].add(k)
        else:clusters.append({'start':t,'members':[h],'routes':{k}})
    for m in VOTE_LEVELS:
        hits=[]
        for c in clusters:
            if len(c['routes'])<m:continue
            labels={}
            for cls in ['bst0','bst_cg_ap']:
                votes=[h['labels'][cls]['label'] for h in c['members'] if cls in h['labels']]
                if votes:labels[cls]={'label':max(set(votes),key=votes.count)}
            hits.append({'time':float(np.median([h['time'] for h in c['members']])),'hit_score':len(c['routes'])/len(routes),'labels':labels,'support':sorted(c['routes'])})
        save(ROUTES/clip['clip_id']/f'vote{m}.json',{'clip_id':clip['clip_id'],'role':clip['role'],'perception':'vote','method':f'vote{m}','status':'completed','hit_seconds':0.,'inputs':{},'members':sorted(routes),'hits':hits,'annotation_inputs':[]})

def features(clip):
    """Label-free per (clip, pipeline-hit-route) features: input quality, candidate rate, cross-route support."""
    routes=single_routes(clip);done={k:r for k,r in routes.items() if r['status']=='completed'}
    times={k:[h['time'] for h in r['hits']] for k,r in done.items()}
    dur=max(1.,next(iter(routes.values()))['inputs'].get('frames',300)/30) if routes else 1.
    q={p:next((r['inputs'] for r in routes.values() if r['perception']==p and r['inputs']),{}) for p in ['rv','tn3']}
    clip_x=[q['rv'].get('paired_player_fraction',0),q['rv'].get('observed_ball_fraction',0),q['tn3'].get('observed_ball_fraction',0),q['rv'].get('valid_shf_window_fraction',0),q['tn3'].get('valid_shf_window_fraction',0)]
    for v in VOTE_LEVELS:
        f=ROUTES/clip['clip_id']/f'vote{v}.json'
        if f.exists():times[f'vote__vote{v}']=[h['time'] for h in read(f)['hits']]
    out={}
    for k,t in times.items():
        others=[o for o in times if o!=k and not o.startswith('vote')]
        support=np.mean([matched(t,times[o])/max(1,len(t)) for o in others]) if t and others else 0.
        coverage=np.mean([matched(times[o],t)/max(1,len(times[o])) for o in others if times[o]]) if others else 0.
        out[k]=clip_x+[len(t)/dur,support,coverage]
    return out

def score_table(role):
    rows=read(STUDY/f'scores_{role}{SUFFIX}.json');table={}
    for r in rows:table.setdefault(r['clip_id'],{})[r['pipeline']]=r
    return table

def f1(tp,fp,fn):return 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
def pooled(picks,table,key):
    tp=fp=fn=0
    for clip,p in picks.items():
        r=table[clip][p];tp+=r[f'{key}tp'];fp+=r[f'{key}fp'];fn+=r[f'{key}fn']
    return {'f1':f1(tp,fp,fn),'tp':tp,'fp':fp,'fn':fn}

def route_of(pipeline):return pipeline.rsplit('__',1)[0]
def available(table,clip):return [p for p,r in table[clip].items() if r['status']=='completed']

class Learned:
    """Shared ridge: clip joint F1 ≈ pipeline intercept + w·features(clip, route)."""
    def __init__(self,pipelines):self.pipes=sorted(pipelines)
    def design(self,pipe,x):
        one=np.zeros(len(self.pipes));one[self.pipes.index(pipe)]=1;return np.concatenate([one,x])
    def fit(self,samples):
        X=np.array([self.design(p,x) for p,x,_ in samples]);y=np.array([t for _,_,t in samples])
        self.mu=X[:,len(self.pipes):].mean(0);self.sd=X[:,len(self.pipes):].std(0)+1e-6;X[:,len(self.pipes):]=(X[:,len(self.pipes):]-self.mu)/self.sd
        pen=np.eye(X.shape[1])*RIDGE;pen[:len(self.pipes),:len(self.pipes)]*=.01
        self.w=np.linalg.solve(X.T@X+pen,X.T@y);return self
    def predict(self,pipe,x):
        if pipe not in self.pipes:return -1.
        d=self.design(pipe,x);d[len(self.pipes):]=(d[len(self.pipes):]-self.mu)/self.sd;return float(d@self.w)
    def packet(self):return {'pipelines':self.pipes,'weights':self.w.tolist(),'mu':self.mu.tolist(),'sd':self.sd.tolist(),'ridge':RIDGE}

def samples(clip_ids,table,feats,key):
    out=[]
    for c in clip_ids:
        for p in available(table,c):
            if route_of(p) in feats[c]:r=table[c][p];out.append((p,feats[c][route_of(p)],f1(r[f'{key}tp'],r[f'{key}fp'],r[f'{key}fn'])))
    return out

def fixed_best(clip_ids,table,key,pool=None):
    common=set.intersection(*[set(available(table,c)) for c in clip_ids]);common&=set(pool or common)
    return max(sorted(common),key=lambda p:pooled({c:p for c in clip_ids},table,key)['f1'])

def choose(av,fc,best,model):
    """Label-free picks for one clip: av = runnable pipelines, fc = features by route."""
    cls=best.rsplit('__',1)[1];q=next(iter(fc.values()))  # clip-level quality, identical for every route
    pred=lambda p:model.predict(p,fc.get(route_of(p),[0]*8))
    rule=f'rv__shf__{cls}' if q[0]>=.5 and q[1]>=.2 and q[3]>=.6 and f'rv__shf__{cls}' in av else f'rv__gru_shared__{cls}'
    single=[p for p in av if not p.startswith('vote') and p.endswith(cls)] or av  # no runnable single route: fall back to (empty) votes
    return {'FIXED_BEST':best if best in av else max(av,key=pred),'RULE_ROUTER':rule,
            'CONSENSUS':max(single,key=lambda p:fc[route_of(p)][6]+fc[route_of(p)][7]),'LEARNED':max(av,key=pred)}

def fit(train,table,feats,key):
    return fixed_best(train,table,key),Learned({p for c in train for p in available(table,c)}).fit(samples(train,table,feats,key))

def policies(train,test,table,feats,key):
    best,model=fit(train,table,feats,key);picks={}
    for c in test:
        av=available(table,c)
        for k,v in {**choose(av,feats[c],best,model),'ORACLE':max(av,key=lambda p:f1(table[c][p][f'{key}tp'],table[c][p][f'{key}fp'],table[c][p][f'{key}fn']))}.items():picks.setdefault(k,{})[c]=v
    return picks,best,model

def evaluate(key):
    table=score_table('dev');cs=[c for c in clips('dev') if c['clip_id'] in table];feats={c['clip_id']:features(c) for c in cs};matches=sorted({c['youtube_id'] for c in cs})
    allpicks={}
    for m in matches:
        test=[c['clip_id'] for c in cs if c['youtube_id']==m];train=[c['clip_id'] for c in cs if c['youtube_id']!=m]
        picks,_,_=policies(train,test,table,feats,key)
        for k,v in picks.items():allpicks.setdefault(k,{}).update(v)
    ids=[c['clip_id'] for c in cs];common=set.intersection(*[set(available(table,c)) for c in ids])
    fixed={p:{'joint':pooled({c:p for c in ids},table,'type_'),'hit':pooled({c:p for c in ids},table,'')} for p in sorted(common)}
    result={'objective':key or 'hit','clips':len(ids),'matches':matches,'cv':'leave-one-match-out over dev',
            'policies':{k:{'joint':pooled(v,table,'type_'),'hit':pooled(v,table,''),'choices':{p:list(v.values()).count(p) for p in sorted(set(v.values()))}} for k,v in allpicks.items()},
            'fixed_pipelines_in_sample':fixed}
    save(STUDY/f'selection_dev_{key or "hit"}{SUFFIX}.json',result);return result

def routes_available(clip):
    """Runnable pipelines from stored route predictions (no references)."""
    out=[]
    for f in (ROUTES/clip['clip_id']).glob('*.json'):
        r=read(f)
        if r['status']!='completed':continue
        for cls in ['bst0','bst_cg_ap']:
            if not r['hits'] or cls in r['hits'][0]['labels']:out.append(f"{r['perception']}__{r['method']}__{cls}")
    return sorted(out)

def code_hashes():
    import hashlib
    return {f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(Path(__file__).parent.glob('study_*.py'))}

def freeze(key):
    table=score_table('dev');cs=[c for c in clips('dev') if c['clip_id'] in table];feats={c['clip_id']:features(c) for c in cs}
    best,model=fit([c['clip_id'] for c in cs],table,feats,key)
    test=[c['clip_id'] for c in read(STUDY/'clips.json')['clips'] if c['role']=='sealed_test' and all((STUDY/'perception'/c['clip_id']/f).exists() for f in ['dense_pose.json','racketvision/trajectory.json','tracknetv3/trajectory.json'])]
    packet={'objective':key or 'hit','fixed_best':best,'learned':model.packet(),'dev_clips':[c['clip_id'] for c in cs],'test_clips':test,'policies_reported':['FIXED_BEST','RULE_ROUTER','CONSENSUS','LEARNED','ORACLE'],
            'note':'All policies are reported on test; none is chosen by test outcome. Perception stopped early by user: test covers only clips with complete perception.','code_sha256':code_hashes()}
    target=STUDY/f'frozen_policy{SUFFIX}.json'
    if target.exists():raise ValueError('Policy already frozen')
    save(target,packet);return packet

def apply():
    frozen=read(STUDY/f'frozen_policy{SUFFIX}.json')
    if frozen['code_sha256']!=code_hashes():raise ValueError('Study code changed after freezing')
    model=Learned(frozen['learned']['pipelines']);w=frozen['learned'];model.w,model.mu,model.sd=np.array(w['weights']),np.array(w['mu']),np.array(w['sd'])
    picks={}
    for c in read(STUDY/'clips.json')['clips']:
        if c['clip_id'] not in frozen['test_clips']:continue
        build_votes(c);av=routes_available(c)
        for k,v in choose(av,features(c),frozen['fixed_best'],model).items():picks.setdefault(k,{})[c['clip_id']]=v
    target=STUDY/f'test_picks{SUFFIX}.json'
    if target.exists():raise ValueError('Test picks already written')
    save(target,picks);return picks

def report(key):
    table=score_table('sealed_test');picks=read(STUDY/f'test_picks{SUFFIX}.json');ids=sorted(next(iter(picks.values())))
    picks['ORACLE']={c:max(available(table,c),key=lambda p:f1(table[c][p][f'{key}tp'],table[c][p][f'{key}fp'],table[c][p][f'{key}fn'])) for c in ids}
    common=set.intersection(*[set(available(table,c)) for c in ids])
    result={'clips':len(ids),'policies':{k:{'joint':pooled(v,table,'type_'),'hit':pooled(v,table,''),'choices':{p:list(v.values()).count(p) for p in sorted(set(v.values()))}} for k,v in picks.items()},
            'fixed_pipelines':{p:{'joint':pooled({c:p for c in ids},table,'type_'),'hit':pooled({c:p for c in ids},table,'')} for p in sorted(common)}}
    save(STUDY/f'selection_test{SUFFIX}.json',result);return result

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('step',choices=['votes','evaluate','freeze','apply','report']);a.add_argument('--role',default='dev');a.add_argument('--objective',choices=['joint','hit'],default='joint');a=a.parse_args()
    if a.step=='votes':
        for c in clips(a.role):build_votes(c)
    elif a.step=='freeze':print(json.dumps({k:v for k,v in freeze('type_' if a.objective=='joint' else '').items() if k!='learned'},ensure_ascii=False)[:600])
    elif a.step=='apply':print({k:len(v) for k,v in apply().items()})
    elif a.step=='report':
        r=report('type_' if a.objective=='joint' else '')
        for k,v in r['policies'].items():print(f"{k:12s} joint F1 {v['joint']['f1']:.3f}  hit F1 {v['hit']['f1']:.3f}")
    else:
        r=evaluate('type_' if a.objective=='joint' else '')
        for k,v in r['policies'].items():print(f"{k:12s} joint F1 {v['joint']['f1']:.3f}  hit F1 {v['hit']['f1']:.3f}")
