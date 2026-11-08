"""Budgeted review of an automatic stroke log, simulated on BFMD.

1. Error models from multi-detector evidence: P(false hit), P(wrong stroke type)
   for logged hits and P(real hit) for consensus candidates the log left out.
2. Review queue: expected errors fixed per second of review.
3. Rally-structured propagation: stroke types are decoded over the rally with a
   learned stroke-transition prior; reviewed hits are clamped and the rest of the
   rally (context, BST windows, transitions) is re-decoded after every fix.

A simulated reviewer (BFMD reference) answers each inspected item. Reviewer
costs are assumed seconds until measured in the user study.

evaluate : leave-one-match-out over dev clips
test     : models fitted on all dev clips, applied to new_test clips
"""
import argparse,json,os,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
os.environ.setdefault('BFMD_VARIANT','v3')
from benchmarks.bfmd.verify import (STUDY,TYPES,candidates,truth_labels,truth_types,type_x,type_vec,author_window_probs,
                                    fit_both,dev_data,Logistic,Softmax,read,save,in_role,ROOT as _R)
COST={'hit':5.,'miss':6.}  # seconds to check a logged hit (existence+type) / a suggested missing hit
# Change in joint F1 = 2tp/(2tp+fp+fn) per fix, in units of 1/denominator at F1 ~ 0.7:
# drop a false hit -> denominator -1 (~F1); fix a type -> numerator +2; add a missed hit -> +2 / +1 (~2-F1)
GAIN={'false':.7,'type':2.,'miss':1.3}
SERVES=('serve','serve_long')
GRID=[0,.05,.1,.15,.2,.3,.4,.5,.75,1.]  # share of the full-review cost (every logged hit checked once)
_CANDS={}
def cached_candidates(clip):
    import copy
    if clip['clip_id'] not in _CANDS:_CANDS[clip['clip_id']]=candidates(clip)
    return copy.deepcopy(_CANDS[clip['clip_id']])

def proba(tmodel,X):
    Z=np.c_[np.ones(len(X)),(np.asarray(X,float)-tmodel.mu)/tmodel.sd]@tmodel.W
    P=np.exp(Z-Z.max(1,keepdims=True));return P/P.sum(1,keepdims=True)

def typed_truth_count(clip):
    m=next(x for x in read(_R.parent/'03_数据/独立基准/BFMD/manifest.json')['matches'] if x['match_id']==clip['match_id'])
    lo=clip['offset_seconds'];hi=lo+clip['duration_seconds']
    return sum(1 for h in m['hits'] if lo<=h['time']<hi and h['label'] in TYPES)

class Rally:
    """One clip: candidates, current log state and decoding."""
    def __init__(self,clip,model,th,tmodel,trans,lam):
        self.clip,self.tmodel,self.trans,self.lam=clip,tmodel,trans,lam
        self.c=cached_candidates(clip);self.y=truth_labels(clip,self.c);self.ty=truth_types(clip,self.c);self.n_typed=typed_truth_count(clip)
        p=model.predict([x['x'] for x in self.c]) if self.c else []
        for x,v in zip(self.c,p):x.update(p_hit=float(v),inlog=bool(v>=th),locked=False,clamp=None,label=None,tp=None,odds=1.,racket=x['x'][10])
        self.decode()

    def log(self):return [x for x in self.c if x['inlog']]

    def decode(self):
        """Stacked type probabilities in rally context, then forward-backward with the transition prior and clamps."""
        seq=self.log()
        if not seq:return
        ctx=[dict(x,probs=(dict.fromkeys(TYPES,0.)|{x['clamp']:1.},)*2) if x['clamp'] else x for x in seq]
        alt=author_window_probs(self.clip,seq)
        E=proba(self.tmodel,[type_x(ctx,j,alt) for j in range(len(seq))])
        for x,e,a in zip(seq,E,alt):x['tp']=e;x['alt']=a
        k=len(TYPES);em=np.array([np.eye(k)[TYPES.index(x['clamp'])] if x['clamp'] else x['tp'] for x in seq])+1e-9
        T=self.trans**self.lam
        f=np.zeros_like(em);b=np.ones_like(em);f[0]=em[0]/em[0].sum()
        for t in range(1,len(seq)):
            link=T if seq[t]['time']-seq[t-1]['time']<=2.5 else np.ones((k,k))
            f[t]=em[t]*(f[t-1]@link);f[t]/=f[t].sum()
        for t in range(len(seq)-2,-1,-1):
            link=T if seq[t+1]['time']-seq[t]['time']<=2.5 else np.ones((k,k))
            b[t]=link@(em[t+1]*b[t+1]);b[t]/=b[t].sum()
        post=f*b;post/=post.sum(1,keepdims=True)
        for x,q in zip(seq,post):x['post']=q;x['label']=x['clamp'] or TYPES[int(q.argmax())]

    def features(self,x):
        """Evidence for the error models (multi-detector agreement, physics, classifiers, rally rhythm)."""
        seq=self.log();side=x['traj'][14]
        prev=[s for s in seq if s['time']<x['time']];nxt=[s for s in seq if s['time']>x['time']]
        gp=min(x['time']-prev[-1]['time'],2.5) if prev else 2.5;gn=min(nxt[0]['time']-x['time'],2.5) if nxt else 2.5
        same=lambda s:float(abs(s['traj'][14]-side)<.5) if s else .5
        post=x.get('post');top=sorted(post,reverse=True) if post is not None else [0,0]
        from benchmarks.bfmd.bfmd_labels import bfmd_label
        alt_ok=float(bfmd_label(x['alt'][0])==x['label']) if x.get('alt') else .5
        return list(x['x'])+[x['p_hit'],top[0],top[0]-top[1],alt_ok,same(prev[-1] if prev else None),same(nxt[0] if nxt else None),gp/2.5,gn/2.5]+x['traj'][9:12]

    def counts(self):
        tp=fp=0
        for x,y,t in zip(self.c,self.y,self.ty):
            if not x['inlog'] or y is None or (y==1 and t is None):continue
            if y==1 and x['label']==t:tp+=1
            else:fp+=1
        return tp,fp,self.n_typed-tp

    def wrong(self,i):
        x,y,t=self.c[i],self.y[i],self.ty[i]
        return bool((x['inlog'] and (y==0 or (y==1 and t and x['label']!=t))) or (not x['inlog'] and y==1))

    def neighbours(self,i):
        """Adjacent logged hits (or nearest logged hits around a suggested miss) within 2.5 s."""
        t0=self.c[i]['time'];seq=[k for k,x in enumerate(self.c) if x['inlog'] and k!=i]
        before=[k for k in seq if self.c[k]['time']<t0];after=[k for k in seq if self.c[k]['time']>t0]
        return [k for k in before[-1:]+after[:1] if abs(self.c[k]['time']-t0)<=2.5]

    def answer(self,i):
        """Simulated reviewer inspects candidate i and fixes it."""
        x,y,t=self.c[i],self.y[i],self.ty[i];x['locked']=True;changed=False
        if x['inlog'] and y==0:x['inlog']=False;changed=True
        elif x['inlog'] and y==1 and t:x['clamp']=t;changed=x['label']!=t
        elif not x['inlog'] and y==1:x['inlog']=True;x['clamp']=t;changed=True
        return changed

class ErrorModels:
    def fit(self,rallies):
        F,Y,T,TY,M,MY=[],[],[],[],[],[]
        for r in rallies:
            for x,y,t in zip(r.c,r.y,r.ty):
                if y is None:continue
                f=r.features(x)
                if x['inlog']:
                    F.append(f);Y.append(float(y==0))
                    if y==1 and t:T.append(f);TY.append(float(x['label']!=t))
                else:M.append(f);MY.append(float(y==1))
        self.fp=Logistic().fit(F,Y,l2=3.);self.ty=Logistic().fit(T,TY,l2=3.);self.miss=Logistic().fit(M,MY,l2=3.) if len(set(MY))>1 else None
        return self

def priority(r,x,policy,em=None):
    """Expected joint-F1 gain per second. 'confidence' = typical tool: logged hits only, lowest confidence first.
    'gain' = calibrated agent confidences (they already fuse the multi-detector evidence), F1-weighted gains,
    and consensus candidates the log left out."""
    if x['locked']:return None
    if policy=='confidence':
        return (1-min(x['p_hit'],max(x['post'])))/COST['hit'] if x['inlog'] else None
    up=lambda q:x['odds']*q/(1-q+1e-9)/(1+x['odds']*q/(1-q+1e-9))  # suspicion from reviewed neighbours
    if x['inlog']:
        pf=up(1-x['p_hit']);pt=up(1-max(x['post']))
        return (pf*GAIN['false']+(1-pf)*pt*GAIN['type'])/COST['hit']
    pm=float(em['miss'].predict([[x['p_hit'],x['racket']]])[0]) if em and em.get('miss') else x['p_hit']  # racket contact evidence for missed hits
    return up(pm)*GAIN['miss']/COST['miss']

def simulate(rallies,policy,em=None,propagate=False):
    """em: {'lr': {True: odds factor if the reviewed item was wrong, False: if right}} for suspicion propagation."""
    """Review in priority order; return pooled (tp,fp,fn) after spending each GRID share of the full-review cost."""
    full=sum(len(r.log()) for r in rallies)*COST['hit']
    counts=lambda:tuple(map(sum,zip(*[r.counts() for r in rallies])))
    out=[counts()];spent=0.;g=1
    items=[(r,i) for r in rallies for i,x in enumerate(r.c)]
    order=[(r,i) for r,i in items if r.c[i]['inlog']]
    while g<len(GRID):
        if policy=='chronological':
            cand=[(r,i) for r,i in order if not r.c[i]['locked']];pick=cand[0] if cand else None
        elif policy=='oracle':
            def wrong(r,i):
                x,y,t=r.c[i],r.y[i],r.ty[i]
                return (x['inlog'] and (y==0 or (y==1 and t and x['label']!=t))) or (not x['inlog'] and y==1)
            cand=[(r,i) for r,i in items if not r.c[i]['locked'] and wrong(r,i)];pick=cand[0] if cand else None
        else:
            best=None
            for r,i in items:
                v=priority(r,r.c[i],policy,em)
                if v is not None and (best is None or v>best[0]):best=(v,r,i)
            pick=best[1:] if best else None
        cost=(COST['hit'] if pick[0].c[pick[1]]['inlog'] else COST['miss']) if pick else None
        while g<len(GRID) and (pick is None or spent+cost>GRID[g]*full):out.append(counts());g+=1
        if pick is None or g>=len(GRID):break
        r,i=pick;was_wrong=r.wrong(i);nb=r.neighbours(i);changed=r.answer(i);spent+=cost
        if em is not None and 'lr' in em:
            for k in nb:r.c[k]['odds']*=em['lr'][was_wrong]
        if changed and propagate:r.decode()
        elif changed and r.c[i]['inlog'] and r.c[i]['clamp']:r.c[i]['label']=r.c[i]['clamp']
    while len(out)<len(GRID):out.append(counts())
    return out

def transitions(rallies):
    """Stroke-to-next-stroke prior from reference labels of training clips."""
    k=len(TYPES);C=np.ones((k,k))
    for r in rallies:
        m=next(x for x in read(_R.parent/'03_数据/独立基准/BFMD/manifest.json')['matches'] if x['match_id']==r.clip['match_id'])
        lo=r.clip['offset_seconds'];hi=lo+r.clip['duration_seconds'];hs=[h for h in m['hits'] if lo<=h['time']<hi]
        for a,b in zip(hs,hs[1:]):
            if a['label'] in TYPES and b['label'] in TYPES and b['time']-a['time']<=2.5:C[TYPES.index(a['label']),TYPES.index(b['label'])]+=1
    return C/C.sum(1,keepdims=True)

def suspicion_lr(rallies):
    """Odds factor for a neighbour being wrong given the reviewed hit was wrong / right (training clips)."""
    n={True:[0,0],False:[0,0]};base=[0,0]
    for r in rallies:
        seq=[k for k,x in enumerate(r.c) if x['inlog'] and r.y[k] is not None]
        for a in seq:
            for b in r.neighbours(a):
                if r.y[b] is None:continue
                n[r.wrong(a)][0]+=r.wrong(b);n[r.wrong(a)][1]+=1;base[0]+=r.wrong(b);base[1]+=1
    odds=lambda w,t:(w+1)/(t-w+1);b=odds(*base)
    return {k:odds(*v)/b for k,v in n.items()}

def miss_model(rallies):
    """P(real hit) for candidates left out of the log, from confidence + racket contact (training clips)."""
    X=[[x['p_hit'],x['racket']] for r in rallies for x,y in zip(r.c,r.y) if not x['inlog'] and y is not None]
    Y=[float(y==1) for r in rallies for x,y in zip(r.c,r.y) if not x['inlog'] and y is not None]
    return Logistic().fit(X,Y,l2=1.)

def run_fold(train_clips,test_clips,model,th,tmodel,lam):
    flat=np.full((len(TYPES),len(TYPES)),1/len(TYPES))
    tr=[Rally(c,model,th,tmodel,flat,0) for c in train_clips];T=transitions(tr);lr={'lr':suspicion_lr(tr)};mm=miss_model(tr);out={'_lr':lr['lr']}
    for name,pol,em,prop,l in [('chronological','chronological',None,False,0),('confidence','confidence',None,False,0),('gain','gain',None,False,0),
                               ('gain+suspicion+context','gain',lr,True,0),('gain+racket','gain',{'miss':mm},False,0),('gain+racket+suspicion+context','gain',{**lr,'miss':mm},True,0),('oracle','oracle',None,False,0)]:
        out[name]=simulate([Rally(c,model,th,tmodel,T,l) for c in test_clips],pol,em,prop)
    return out

def f1(c):tp,fp,fn=c;return 2*tp/(2*tp+fp+fn)
def pool(folds):
    return {k:[f1(tuple(map(sum,zip(*[f[k][g] for f in folds])))) for g in range(len(GRID))] for k in folds[0] if not k.startswith('_')}

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('step',choices=['evaluate','test']);a.add_argument('--lam',type=float,default=1.);a=a.parse_args()
    clips,data,types,byid=dev_data()
    if a.step=='evaluate':
        folds=[]
        for m in sorted({c['youtube_id'] for c in clips}):
            train=[c for c in clips if c['youtube_id']!=m];test=[c for c in clips if c['youtube_id']==m]
            model,th,tmodel=fit_both([c['clip_id'] for c in train],data,types,byid)
            folds.append(run_fold(train,test,model,th,tmodel,a.lam));print('fold',m,'lr',{k:round(v,2) for k,v in folds[-1]['_lr'].items()},flush=True)
        res=pool(folds)
        save(STUDY/f'review_dev_lam{a.lam:g}.json',{'grid_share':GRID,'curves':res,'lam':a.lam,'cost':COST})
        print('share         ',' '.join(f'{g:5.2f}' for g in GRID))
        for k,v in res.items():print(f"{k:14s}",' '.join(f'{x:.3f}' for x in v))
    else:
        f=read(STUDY/'frozen_verify_v3_racket.json')
        model=Logistic();model.w,model.mu,model.sd=(np.array(f['hit_model'][k]) for k in ['w','mu','sd'])
        tmodel=Softmax();tmodel.W,tmodel.mu,tmodel.sd=(np.array(f['type_model'][k]) for k in ['W','mu','sd'])
        test=[c for c in read(STUDY/'clips.json')['clips'] if in_role(c,'new_test')]
        res=run_fold(clips,test,model,f['threshold'],tmodel,a.lam)
        res=pool([res])
        save(STUDY/f'review_test_lam{a.lam:g}.json',{'grid_share':GRID,'curves':res,'lam':a.lam,'cost':COST})
        print('share         ',' '.join(f'{g:5.2f}' for g in GRID))
        for k,v in res.items():print(f"{k:14s}",' '.join(f'{x:.3f}' for x in v))
