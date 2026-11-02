"""Per-hit verification agent.

Starts from the primary pipeline's candidates, adds consensus candidates that
the primary missed, and keeps/drops each one with a logistic model over
label-free evidence (detector agreement, shuttle turn/speed, BST confidence,
pose coverage, rhythm). The stroke label is the BFMD-taxonomy argmax of the
averaged BST-0 / BST-CG-AP probabilities when that helps on training folds.

train/apply : model and threshold are fitted on dev matches only;
evaluate    : leave-one-match-out over dev, writes routes .../agent__verify.json
"""
import argparse,json,os,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from benchmarks.bfmd.refine import track,turn,speed
from benchmarks.bfmd.bfmd_labels import bfmd_label
from benchmarks.bfmd.rally_clips import in_role
STUDY=ROOT.parent/'03_数据/独立基准/BFMD/pipeline_study'
VARIANT=os.environ.get('BFMD_VARIANT','v3');ROUTES=STUDY/f'routes_{VARIANT}'
PRIMARY='tn3__gru_shared';WIN=3;ADD_SUPPORT=3

def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=1,allow_nan=False)+'\n')

def candidates(clip):
    """Label-free candidates with evidence features."""
    folder=ROUTES/clip['clip_id'];routes={f.stem:read(f) for f in folder.glob('*.json') if '__' in f.stem and not f.stem.startswith('agent')}
    routes={k:r for k,r in routes.items() if r['status']=='completed'}
    if PRIMARY not in routes:return []
    others={k:[round(h['time']*30) for h in r['hits']] for k,r in routes.items() if k!=PRIMARY}
    pfolder=STUDY/'perception'/clip['clip_id'];xy=track(read(pfolder/'tracknetv3/trajectory.json'))
    dense=read(pfolder/'dense_pose.json')['frames'];n=len(dense)
    cands=[{'frame':round(h['time']*30),'hit':h,'primary':1.} for h in routes[PRIMARY]['hits']]
    pool=sorted((f,k,h) for k,r in routes.items() if k!=PRIMARY for h in r['hits'] for f in [round(h['time']*30)])
    used=set()
    for f,k,h in pool:  # consensus candidates the primary missed
        if any(abs(f-c['frame'])<=WIN for c in cands) or (f,k) in used:continue
        members=[(g,kk,hh) for g,kk,hh in pool if abs(g-f)<=WIN]
        if len({kk for _,kk,_ in members})>=ADD_SUPPORT:
            used.update((g,kk) for g,kk,_ in members)
            best=max(members,key=lambda m:m[1].startswith('tn3'))[2]
            cands.append({'frame':int(np.median([g for g,_,_ in members])),'hit':best,'primary':0.})
    cands.sort(key=lambda c:c['frame']);frames=[c['frame'] for c in cands]
    for j,c in enumerate(cands):
        f=c['frame'];lab=c['hit']['labels'];p0=lab.get('bst0',{}).get('probs',{});p1=lab.get('bst_cg_ap',{}).get('probs',{})
        top=sorted(p0.values(),reverse=True)+[0,0]
        sup=sum(any(abs(f-g)<=WIN for g in gs) for gs in others.values())
        pose=np.mean([bool(dense[i]['players']['near'] and dense[i]['players']['far']) for i in range(max(0,f-5),min(n,f+6))]) if n else 0
        gaps=[abs(f-g) for k,g in enumerate(frames) if k!=j]
        c['x']=[c['primary'],sup/max(1,len(others)),max(turn(xy,t) for t in range(max(0,f-2),min(len(xy),f+3))),min(speed(xy,f),30)/30,
                top[0],top[0]-top[1],float(bfmd_label(p0)==bfmd_label(p1)) if p1 else .5,pose,min(min(gaps,default=90),90)/90,
                float(c['hit'].get('hit_score') or 0)]
        c['time']=f/30;c['probs']=(p0,p1)
    return cands

def truth_labels(clip,cands):
    """Dev-only: which candidates match a non-serve reference contact (±0.1 s); None near serves."""
    m=next(x for x in read(ROOT.parent/'03_数据/独立基准/BFMD/manifest.json')['matches'] if x['match_id']==clip['match_id'])
    lo=clip['offset_seconds'];hi=lo+clip['duration_seconds'];hits=[h for h in m['hits'] if lo<=h['time']<hi]
    serves=[h['time']-lo for h in hits if h['label'] in ('serve','serve_long')];truth=[h['time']-lo for h in hits if h['label'] not in ('serve','serve_long')]
    y=[];used=set()
    for c in cands:
        if any(abs(c['time']-s)<=.1+1e-6 for s in serves):y.append(None);continue
        best=min(((abs(c['time']-t),i) for i,t in enumerate(truth) if i not in used and abs(c['time']-t)<=.1+1e-6),default=None)
        if best:used.add(best[1]);y.append(1.)
        else:y.append(0.)
    return y

class Logistic:
    def fit(self,X,y,l2=1.):
        X=np.asarray(X,float);y=np.asarray(y,float);self.mu,self.sd=X.mean(0),X.std(0)+1e-6;Z=np.c_[np.ones(len(X)),(X-self.mu)/self.sd];w=np.zeros(Z.shape[1])
        for _ in range(500):  # Newton steps
            p=1/(1+np.exp(-Z@w));g=Z.T@(p-y)+l2*np.r_[0,w[1:]];H=(Z*(p*(1-p))[:,None]).T@Z+l2*np.diag(np.r_[0,np.ones(len(w)-1)])
            step=np.linalg.solve(H,g);w-=step
            if np.abs(step).max()<1e-8:break
        self.w=w;return self
    def predict(self,X):Z=np.c_[np.ones(len(X)),(np.asarray(X,float)-self.mu)/self.sd];return 1/(1+np.exp(-Z@self.w))
    def packet(self):return {'w':self.w.tolist(),'mu':self.mu.tolist(),'sd':self.sd.tolist()}

def label(c,mode):
    p0,p1=c['probs']
    if mode=='avg' and p1:p0={k:(p0.get(k,0)+p1.get(k,0))/2 for k in set(p0)|set(p1)}
    return bfmd_label(p0)

def fit(data):
    X=[x for cs,ys in data for c,y in zip(cs,ys) if y is not None for x in [c['x']]];Y=[y for cs,ys in data for y in ys if y is not None]
    model=Logistic().fit(X,Y)
    # threshold maximising pooled F1 on the training folds (candidates that miss stay misses)
    total=sum(1 for cs,ys in data for y in ys if y)  # matched positives available
    best=(-1,.5)
    for th in np.linspace(.1,.9,33):
        tp=fp=0
        for cs,ys in data:
            if not cs:continue
            keep=model.predict([c['x'] for c in cs])>=th
            tp+=sum(k and y==1 for k,y in zip(keep,ys));fp+=sum(k and y==0 for k,y in zip(keep,ys))
        f=2*tp/(2*tp+fp+(total-tp)) if tp else 0
        best=max(best,(f,th))
    return model,best[1]

def evaluate():
    clips=[c for c in read(STUDY/'clips.json')['clips'] if in_role(c,'dev') and (ROUTES/c['clip_id']).exists()]
    data={c['clip_id']:(cs:=candidates(c),truth_labels(c,cs)) for c in clips};out={}
    for m in sorted({c['youtube_id'] for c in clips}):
        train=[data[c['clip_id']] for c in clips if c['youtube_id']!=m];model,th=fit(train)
        for c in clips:
            if c['youtube_id']!=m:continue
            cs,_=data[c['clip_id']];keep=model.predict([x['x'] for x in cs])>=th if cs else []
            hits=[{'time':x['time'],'hit_score':float(s),'labels':{'bst0':{'label':label(x,'bst0')},'bst_cg_ap':{'label':label(x,'avg')}}} for x,k,s in zip(cs,keep,model.predict([x['x'] for x in cs]) if cs else []) if k]
            save(ROUTES/c['clip_id']/'agent__verify.json',{'clip_id':c['clip_id'],'role':c['role'],'perception':'agent','method':'verify','status':'completed','hit_seconds':0.,'inputs':{},
                 'note':'leave-one-match-out: model fitted without this match; label slot bst_cg_ap = averaged BST-0/CG-AP probabilities','threshold':th,'hits':hits,'annotation_inputs':[]})
            out[c['clip_id']]=len(hits)
    full,th=fit(list(data.values()));save(STUDY/f'verify_model_{VARIANT}.json',{'model':full.packet(),'threshold':th,'features':['primary','support','turn','speed','bst_top','bst_margin','bst_agree','pose','gap','hit_score'],'primary':PRIMARY,'add_support':ADD_SUPPORT})
    return out

if __name__=='__main__':
    print(len(evaluate()),'dev clips verified (leave-one-match-out)')
