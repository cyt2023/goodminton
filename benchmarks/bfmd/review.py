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
                                    fit_both,dev_data,Logistic,Softmax,read,save,in_role,timing_evidence,ROOT as _R)
COST={'hit':5.,'miss':6.,'retime':3.}  # seconds to check a logged hit (existence+type) / a suggested missing hit / extra to move a mistimed hit
RETIME=.4  # a logged hit whose nearest unmatched reference contact is within this many seconds can be moved there by the reviewer
# Change in joint F1 = 2tp/(2tp+fp+fn) per fix, in units of 1/denominator at F1 ~ 0.7:
# drop a false hit -> denominator -1 (~F1); fix a type -> numerator +2; add a missed hit -> +2 / +1 (~2-F1)
GAIN={'false':.7,'type':2.,'miss':1.3,'retime':2.}
SERVES=('serve','serve_long')
GRID=[0,.05,.1,.15,.2,.3,.4,.5,.75,1.]  # share of the full-review cost (every logged hit checked once)
_CANDS={}
def cached_candidates(clip):
    import copy
    if clip['clip_id'] not in _CANDS:
        cs=candidates(clip)
        for c in cs:c['tev']=timing_evidence(clip,c['frame']);c['rtype']=route_types(clip,c['frame'])
        _CANDS[clip['clip_id']]=cs
    return copy.deepcopy(_CANDS[clip['clip_id']])

_ROUTES={}
def route_types(clip,f):
    """Stroke-type evidence from every detector route that fired within +-3 frames: mean BST probabilities and each route's top type."""
    from benchmarks.bfmd.verify import ROUTES
    k=clip['clip_id']
    if k not in _ROUTES:
        rs=[read(p) for p in (ROUTES/k).glob('*__*.json') if not p.stem.startswith('agent')]
        _ROUTES[k]=[r['hits'] for r in rs if r['status']=='completed']
    P=[]
    for hits in _ROUTES[k]:
        h=min(hits,key=lambda h:abs(round(h['time']*30)-f),default=None)
        if h is None or abs(round(h['time']*30)-f)>3:continue
        P+=[type_vec(h['labels'][n].get('probs',{})) for n in ('bst0','bst_cg_ap') if n in h['labels']]
    if not P:return np.full(len(TYPES),1/len(TYPES)),[]
    P=np.array(P);return P.mean(0),[int(i) for i in P.argmax(1)]

def proba(tmodel,X):
    Z=np.c_[np.ones(len(X)),(np.asarray(X,float)-tmodel.mu)/tmodel.sd]@tmodel.W
    P=np.exp(Z-Z.max(1,keepdims=True));return P/P.sum(1,keepdims=True)

def typed_truth_count(clip):
    m=next(x for x in read(_R.parent/'03_数据/独立基准/BFMD/manifest.json')['matches'] if x['match_id']==clip['match_id'])
    lo=clip['offset_seconds'];hi=lo+clip['duration_seconds']
    return sum(1 for h in m['hits'] if lo<=h['time']<hi and h['label'] in TYPES)

_BST={}
def side_top(clip,frames):
    """BST-0 probability mass on far-player ('Top_') classes, author window bounded by the given neighbouring hit frames."""
    from specialist_bst import SpecialistBST
    k=clip['clip_id']
    if k not in _BST:
        pf=STUDY/'perception'/k;d=read(pf/'dense_pose.json');_BST[k]=(d,SpecialistBST(d,read(pf/'tracknetv3/trajectory.json'),None),{})
    d,m,cache=_BST[k];n=len(d['frames']);out=[]
    for j,f in enumerate(frames):
        prev=frames[j-1] if j and f-frames[j-1]<=75 else None;nxt=frames[j+1] if j+1<len(frames) and frames[j+1]-f<=75 else None
        lo=max(prev if prev is not None else f-15,f-45,0);hi=min((nxt+7) if nxt is not None else f+15,f+45+7,n-1)
        if (f,lo,hi) not in cache:
            v,_=m.inputs({'frame_index':f,'start_frame':lo,'end_frame':hi,'peak_time':d['frames'][f]['source_time']});p=m.predict(v)
            cache[(f,lo,hi)]=float(sum(sc for cl,sc in zip(p['class_labels'],p['bst0_all_scores']) if cl.startswith('Top')))
        out.append(cache[(f,lo,hi)])
    return out

def near_truth(clip,cands,y):
    """For each unmatched candidate (y==0): nearest reference contact within RETIME s that no candidate matched -> (time, type)."""
    m=next(x for x in read(_R.parent/'03_数据/独立基准/BFMD/manifest.json')['matches'] if x['match_id']==clip['match_id'])
    lo=clip['offset_seconds'];hi=lo+clip['duration_seconds']
    gt=[(h['time']-lo,h['label'] if h['label'] in TYPES else None) for h in m['hits'] if lo<=h['time']<hi and h['label'] not in SERVES]
    taken=set()
    for c,v in zip(cands,y):
        if v==1:taken.add(min(range(len(gt)),key=lambda j:abs(gt[j][0]-c['time'])))
    out=[]
    for c,v in zip(cands,y):
        free=[j for j in range(len(gt)) if j not in taken and abs(gt[j][0]-c['time'])<=RETIME]
        out.append(gt[min(free,key=lambda j:abs(gt[j][0]-c['time']))] if v==0 and free else None)
    return out

class Rally:
    """One clip: candidates, current log state and decoding."""
    def __init__(self,clip,model,th,tmodel,trans,lam):
        self.clip,self.tmodel,self.trans,self.lam=clip,tmodel,trans,lam
        self.c=cached_candidates(clip);self.y=truth_labels(clip,self.c);self.ty=truth_types(clip,self.c);self.n_typed=typed_truth_count(clip)
        self.near=near_truth(clip,self.c,self.y);self.retime=True
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
        self.parity()

    def parity(self):
        """Alternation violation: probability that a logged hit and an adjacent one (<=2.5 s) were played by the same player."""
        seq=self.log()
        if not seq:return
        top=side_top(self.clip,[x['frame'] for x in seq])
        for j,x in enumerate(seq):
            v=0.
            for i in (j-1,j+1):
                if 0<=i<len(seq) and abs(seq[i]['time']-x['time'])<=2.5:v=max(v,top[j]*top[i]+(1-top[j])*(1-top[i]))
            x['viol']=v

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
        """Simulated reviewer inspects candidate i and fixes it; a mistimed hit is moved to the reference contact."""
        x,y,t=self.c[i],self.y[i],self.ty[i];x['locked']=True;changed=False;self.moved=False
        if x['inlog'] and y==0 and self.retime and self.near[i]:
            g,lab=self.near[i];x['time']=g;x['frame']=round(g*30);self.y[i]=1.;self.ty[i]=lab;x['clamp']=lab;changed=True;self.moved=True
            for k,v in enumerate(self.near):
                if v and k!=i and abs(v[0]-g)<1e-9:self.near[k]=None
            self.near[i]=None
        elif x['inlog'] and y==0:x['inlog']=False;changed=True
        elif x['inlog'] and y==1 and t:x['clamp']=t;changed=x['label']!=t
        elif not x['inlog'] and y==1:x['inlog']=True;x['clamp']=t;changed=True
        return changed

def evidence_x(r,x):
    """All per-hit evidence for the error models: agent features, detector agreement, physics, classifiers, rally rhythm,
    hitter alternation, timing evidence (sound / racket away from the logged moment) and stroke-type agreement across routes."""
    post=x.get('post');post=post if post is not None else np.full(len(TYPES),1/len(TYPES))
    lab=TYPES.index(x['label']) if x.get('label') else int(np.argmax(post));rm,rt=x['rtype']
    return r.features(x)+[x.get('viol',0.),x['tev'][1],x['tev'][2],float(-(post*np.log(post+1e-9)).sum()),float(rm[lab]),float(np.mean([t==lab for t in rt])) if rt else .5]

# evidence_x columns grouped for ablations
RALLY_COLS=(8,16,17,18,19,23)          # gap to neighbours, same-side neighbours, rhythm, hitter-alternation violation
SOURCE_COLS=(1,10,11,24,25,27,28)      # detector agreement, racket contact, impact sound, timing evidence, stroke-type agreement across routes

class ErrorModels:
    """P(false hit), P(mistimed real hit), P(wrong stroke type | real), P(real | left out) from all evidence.
    Fitted on cross-fitted agent outputs: each training match is scored by an agent that never saw it."""
    KINDS=('fp','mt','ty','miss')
    def __init__(self,drop=()):self.drop=tuple(drop)
    def sel(self,f):return [v for j,v in enumerate(f) if j not in self.drop] if self.drop else f
    def fit(self,rallies,l2=10.):
        R={k:([],[]) for k in self.KINDS}
        for r in rallies:
            for k,x in enumerate(r.c):
                if r.y[k] is None:continue
                f=self.sel(evidence_x(r,x))
                if x['inlog']:
                    R['fp'][0].append(f);R['fp'][1].append(float(r.y[k]==0))
                    R['mt'][0].append(f);R['mt'][1].append(float(r.y[k]==0 and r.near[k] is not None))
                    if r.y[k]==1 and r.ty[k]:R['ty'][0].append(f);R['ty'][1].append(float(x['label']!=r.ty[k]))
                else:R['miss'][0].append(f);R['miss'][1].append(float(r.y[k]==1))
        self.m={k:Logistic().fit(X,Y,l2=l2) for k,(X,Y) in R.items()}
        return self
    def p(self,kind,f):return float(self.m[kind].predict([self.sel(f)])[0])
    def packet(self):return {'drop':list(self.drop),**{k:m.packet() for k,m in self.m.items()}}
    @classmethod
    def load(cls,packet):
        e=cls(packet.get('drop',()));e.m={}
        for k in cls.KINDS:
            m=Logistic();m.w,m.mu,m.sd=(np.array(packet[k][v]) for v in ('w','mu','sd'));e.m[k]=m
        return e

def crossfit_error_models(train_clips,data,types,byid):
    """Score each training match with an agent fitted on the other training matches, then fit the error models."""
    flat=np.full((len(TYPES),)*2,1/len(TYPES));rallies=[]
    for j in sorted({c['youtube_id'] for c in train_clips}):
        model,th,tmodel=fit_both([c['clip_id'] for c in train_clips if c['youtube_id']!=j],data,types,byid)
        rallies+=[Rally(c,model,th,tmodel,flat,0) for c in train_clips if c['youtube_id']==j]
    return {'all':ErrorModels().fit(rallies),'no rally structure':ErrorModels(RALLY_COLS).fit(rallies),'no new sources':ErrorModels(SOURCE_COLS).fit(rallies)}

def priority(r,x,policy,em=None):
    """Expected joint-F1 gain per second. 'confidence' = typical tool: logged hits only, lowest confidence first.
    'gain' = calibrated agent confidences (they already fuse the multi-detector evidence), F1-weighted gains,
    and consensus candidates the log left out."""
    if x['locked']:return None
    if policy=='confidence':
        return (1-min(x['p_hit'],max(x['post'])))/COST['hit'] if x['inlog'] else None
    if policy=='evidence':  # error models over all evidence; reviewed neighbours raise / lower the odds
        up=lambda q:x['odds']*q/(1-q+1e-9)/(1+x['odds']*q/(1-q+1e-9));E=em['E'];f=evidence_x(r,x)
        if x['inlog']:
            pf=up(E.p('fp',f));pm=min(pf,up(E.p('mt',f)));pt=up(E.p('ty',f))
            return (pm*GAIN['retime']+(pf-pm)*GAIN['false']+(1-pf)*pt*GAIN['type'])/(COST['hit']+pm*COST['retime'])
        return up(E.p('miss',f))*GAIN['miss']/COST['miss']
    if policy=='single_hit':return 1-min(1.,x['x'][9]) if x['inlog'] else None
    if policy=='single_type':return 1-x['x'][4] if x['inlog'] else None
    if policy=='single':  # typical tool: the detector's own hit score and the classifier's top probability, logged hits only
        return max(1-min(1.,x['x'][9]),1-x['x'][4]) if x['inlog'] else None
    up=lambda q:x['odds']*q/(1-q+1e-9)/(1+x['odds']*q/(1-q+1e-9))  # suspicion from reviewed neighbours
    if x['inlog']:
        pf=up(1-x['p_hit']);pt=up(1-max(x['post']))
        if em and em.get('parity'):  # rally structure: same-player neighbours raise the error odds by a learned factor
            g=float(np.exp(em['parity'].w[2]/em['parity'].sd[1]*x.get('viol',0.)))
            o=lambda q:g*q/(1-q+1e-9)/(1+g*q/(1-q+1e-9));pf,pt=o(pf),o(pt)
        if em and em.get('timing'):  # timing evidence: impact sound / racket approach away from the logged moment -> hit is real but mistimed
            pm=min(pf,float(em['timing'].predict([timing_x(x)])[0]))
            return (pm*GAIN['retime']+(pf-pm)*GAIN['false']+(1-pf)*pt*GAIN['type'])/(COST['hit']+pm*COST['retime'])
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
        r,i=pick;was_wrong=r.wrong(i);nb=r.neighbours(i);changed=r.answer(i);spent+=cost+(COST['retime'] if r.moved else 0.)
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

def timing_x(x):return [x['p_hit'],x['tev'][1],x['tev'][2]]

def timing_model(rallies):
    """P(logged hit is a real contact logged at the wrong moment) from confidence and timing evidence (training clips)."""
    X=[timing_x(x) for r in rallies for k,x in enumerate(r.c) if x['inlog'] and r.y[k] is not None]
    Y=[float(r.y[k]==0 and r.near[k] is not None) for r in rallies for k,x in enumerate(r.c) if x['inlog'] and r.y[k] is not None]
    return Logistic().fit(X,Y,l2=3.)

def parity_model(rallies):
    """P(logged hit is wrong) from confidence and alternation violation (training clips)."""
    X=[];Y=[]
    for r in rallies:
        for k,x in enumerate(r.c):
            if x['inlog'] and r.y[k] is not None:X.append([1-min(x['p_hit'],max(x['post'])),x.get('viol',0.)]);Y.append(float(r.wrong(k)))
    return Logistic().fit(X,Y,l2=1.)

def run_fold(train_clips,test_clips,model,th,tmodel,lam,E=None):
    flat=np.full((len(TYPES),len(TYPES)),1/len(TYPES))
    tr=[Rally(c,model,th,tmodel,flat,0) for c in train_clips];T=transitions(tr);lr={'lr':suspicion_lr(tr)};mm=miss_model(tr);pm=parity_model(tr);out={'_lr':lr['lr']}
    runs=[('chronological','chronological',None,False),('single_hit','single_hit',None,False),('single_type','single_type',None,False),('single','single',None,False),
          ('confidence','confidence',None,False),('gain+racket+suspicion+parity','gain',{**lr,'miss':mm,'parity':pm},True)]
    if E is not None:runs+=[('evidence','evidence',{'E':E['all']},False),('evidence+propagation','evidence',{**lr,'E':E['all']},True),
                            ('evidence - rally structure','evidence',{'E':E['no rally structure']},False),('evidence - new sources','evidence',{'E':E['no new sources']},False)]
    for name,pol,em,prop in runs+[('oracle','oracle',None,False)]:
        out[name]=simulate([Rally(c,model,th,tmodel,T,lam) for c in test_clips],pol,em,prop)
    return out

def f1(c):tp,fp,fn=c;return 2*tp/(2*tp+fp+fn)
def pool(folds):
    return {k:[f1(tuple(map(sum,zip(*[f[k][g] for f in folds])))) for g in range(len(GRID))] for k in folds[0] if not k.startswith('_')}

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('step',choices=['evaluate','test','lomo','freeze']);a.add_argument('--lam',type=float,default=1.);a=a.parse_args()
    if a.step=='freeze':  # per-match bundles for the review page: each match is served by models that never saw it
        from benchmarks.bfmd.verify import ROUTES,PRIMARY,code_hashes
        clips=[c for c in read(STUDY/'clips.json')['clips'] if (ROUTES/c['clip_id']/f'{PRIMARY}.json').exists()]
        byid={c['clip_id']:c for c in clips};data={}
        for c in clips:cs=cached_candidates(c);data[c['clip_id']]=(cs,truth_labels(c,cs))
        types={k:truth_types(byid[k],data[k][0]) for k in byid};bundles={}
        for m in sorted({c['youtube_id'] for c in clips}):
            train=[c for c in clips if c['youtube_id']!=m];model,th,tmodel=fit_both([c['clip_id'] for c in train],data,types,byid)
            E=crossfit_error_models(train,data,types,byid)['all'];flat=np.full((len(TYPES),)*2,1/len(TYPES))
            tr=[Rally(c,model,th,tmodel,flat,0) for c in train]
            bundles[m]={'hit_model':model.packet(),'threshold':th,'type_model':{'W':tmodel.W.tolist(),'mu':tmodel.mu.tolist(),'sd':tmodel.sd.tolist()},
                        'error_models':E.packet(),'suspicion_lr':{str(k):v for k,v in suspicion_lr(tr).items()},'train_matches':sorted({c['youtube_id'] for c in train})}
            print('bundle',m,flush=True)
        save(STUDY/'frozen_review_lomo12.json',{'bundles':bundles,'cost':COST,'gain':GAIN,'code_sha256':code_hashes(),
                                               'note':'For clips of match m use bundles[m]: agent and error models fitted on the other 11 matches.'})
        raise SystemExit
    if a.step=='lomo':  # leave-one-match-out over every BFMD singles match with routes (all roles)
        from benchmarks.bfmd.verify import ROUTES,PRIMARY
        clips=[c for c in read(STUDY/'clips.json')['clips'] if (ROUTES/c['clip_id']/f'{PRIMARY}.json').exists()]
        byid={c['clip_id']:c for c in clips};data={};ablate=os.environ.get('REVIEW_ABLATE','')
        for c in clips:
            cached_candidates(c)
            if ablate=='sources':  # system without multi-source evidence: no detector agreement, racket, sound, timing or cross-route types
                for x in _CANDS[c['clip_id']]:
                    x['x'][1]=x['x'][10]=x['x'][11]=0.;x['tev']=(x['frame'],0.,0.);x['rtype']=(np.full(len(TYPES),1/len(TYPES)),[])
            cs=cached_candidates(c);data[c['clip_id']]=(cs,truth_labels(c,cs))
        types={k:truth_types(byid[k],data[k][0]) for k in byid};folds=[]
        for m in sorted({c['youtube_id'] for c in clips}):
            train=[c for c in clips if c['youtube_id']!=m];test=[c for c in clips if c['youtube_id']==m]
            model,th,tmodel=fit_both([c['clip_id'] for c in train],data,types,byid)
            E=crossfit_error_models(train,data,types,byid)
            folds.append(run_fold(train,test,model,th,tmodel,0,E));print('fold',m,len(test),'clips',flush=True)
        res=pool(folds);save(STUDY/f"review_lomo12{'_ablate_'+ablate if ablate else ''}.json",{'grid_share':GRID,'curves':res,'cost':COST,'clips':len(clips),
                                                        'per_match':{m:{k:v for k,v in f.items() if not k.startswith('_')} for m,f in zip(sorted({c['youtube_id'] for c in clips}),folds)}})
        print('share         ',' '.join(f'{g:5.2f}' for g in GRID))
        for k,v in res.items():print(f"{k:14s}",' '.join(f'{x:.3f}' for x in v))
        raise SystemExit
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
        f=read(STUDY/'frozen_verify_v3_racket_audio.json')
        model=Logistic();model.w,model.mu,model.sd=(np.array(f['hit_model'][k]) for k in ['w','mu','sd'])
        tmodel=Softmax();tmodel.W,tmodel.mu,tmodel.sd=(np.array(f['type_model'][k]) for k in ['W','mu','sd'])
        test=[c for c in read(STUDY/'clips.json')['clips'] if in_role(c,'new_test')]
        res=run_fold(clips,test,model,f['threshold'],tmodel,a.lam)
        res=pool([res])
        save(STUDY/f'review_test_lam{a.lam:g}.json',{'grid_share':GRID,'curves':res,'lam':a.lam,'cost':COST})
        print('share         ',' '.join(f'{g:5.2f}' for g in GRID))
        for k,v in res.items():print(f"{k:14s}",' '.join(f'{x:.3f}' for x in v))
