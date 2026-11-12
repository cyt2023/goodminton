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
RETIME_ON=os.environ.get('REVIEW_RETIME','1')=='1'  # 0: ablation, the reviewer can only delete a mistimed hit
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

TYPEHEAD=os.environ.get('REVIEW_TYPEHEAD','0') in ('1','2')  # BFMD stroke-type head on BST features: 1 in the type model and the error models, 2 error models only
_EMB={}
def bst_embeddings(clip):
    """Input of the BST-0 and BST-CG-AP classification heads (300 + 300) in a +-1 s window: at every candidate frame, and at every
    typed reference contact (the latter only to train the BFMD head on training matches)."""
    k=clip['clip_id']
    if k not in _EMB:
        from specialist_bst import SpecialistBST
        from benchmarks.bfmd.study_hits import calibration
        pf=STUDY/'perception'/k;dense=read(pf/'dense_pose.json');ball=read(pf/'tracknetv3/trajectory.json')
        m=SpecialistBST(dense,ball,calibration(read(STUDY/'court_review.json')['matches'][clip['youtube_id']]['corners_TL_TR_BL_BR']));grab={}
        hooks=[m.model.mlp_head.register_forward_pre_hook(lambda mod,inp:grab.__setitem__(0,inp[0][0].detach().numpy().copy())),
               m.full_model.mlp_head.register_forward_pre_hook(lambda mod,inp:grab.__setitem__(1,inp[0][0].detach().numpy().copy()))]
        n=len(dense['frames'])
        def emb(f):
            f=min(max(int(f),0),n-1);v,_=m.inputs({'frame_index':f,'start_frame':max(0,f-30),'end_frame':min(n-1,f+30),'peak_time':dense['frames'][f]['source_time']})
            m.predict(v);return np.r_[grab[0],grab[1]]
        mm=next(x for x in read(_R.parent/'03_数据/独立基准/BFMD/manifest.json')['matches'] if x['match_id']==clip['match_id'])
        lo=clip['offset_seconds'];hi=lo+clip['duration_seconds']
        _EMB[k]=({x['frame']:emb(x['frame']) for x in _CANDS[k]},[(emb(round((h['time']-lo)*30)),h['label']) for h in mm['hits'] if lo<=h['time']<hi and h['label'] in TYPES])
        for h in hooks:h.remove()
    return _EMB[k]

def fit_head(clips):
    """BFMD stroke-type head (softmax) on BST features at the reference contacts of the given clips."""
    rows=[r for c in clips for r in bst_embeddings(c)[1]]
    return Softmax().fit([e for e,_ in rows],[t for _,t in rows],l2=10.,steps=600)

def attach_heads(clips,data,m,cache):
    """Head probabilities on every candidate; a clip of match j is scored by a head that saw neither j nor the held-out match m."""
    for c in clips:
        j=c['youtube_id'];excl=frozenset({m,j})
        if excl not in cache:cache[excl]=fit_head([d for d in clips if d['youtube_id'] not in excl])
        E=bst_embeddings(c)[0]
        for cs in (_CANDS[c['clip_id']],data[c['clip_id']][0]):
            for x,p in zip(cs,proba(cache[excl],[E[x['frame']] for x in cs]) if cs else []):x['head']=p

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
        self.near=near_truth(clip,self.c,self.y);self.retime=RETIME_ON
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
        k=len(TYPES);em=np.array([np.eye(k)[TYPES.index(x['clamp'])] if x['clamp'] else (x['tpe'] if x.get('tpe') is not None else x['tp']) for x in seq])+1e-9
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
    head=[float(x['head'][lab]),float(int(np.argmax(x['head']))==lab)] if x.get('head') is not None else []
    return r.features(x)+[x.get('viol',0.),x['tev'][1],x['tev'][2],float(-(post*np.log(post+1e-9)).sum()),float(rm[lab]),float(np.mean([t==lab for t in rt])) if rt else .5]+head

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

def error_probs(r,x,E,f,up,timing=True):
    """(P false, P mistimed, P wrong type | real, P real if left out) for candidate x; items moved by auto_pass keep the probability that moved them."""
    auto=x.get('auto')
    if x['inlog']:
        pf=up(auto[1]) if auto and auto[0]=='added' else up(E.p('fp',f))
        pm=min(pf,up(E.p('mt',f))) if timing and not auto else 0.
        return pf,pm,up(E.p('ty',f)),None
    return None,None,None,up(auto[1]) if auto and auto[0]=='removed' else up(E.p('miss',f))

def priority(r,x,policy,em=None):
    """Expected joint-F1 gain per second. 'confidence' = typical tool: logged hits only, lowest confidence first.
    'gain' = calibrated agent confidences (they already fuse the multi-detector evidence), F1-weighted gains,
    and consensus candidates the log left out."""
    if x['locked']:return None
    if policy=='confidence':
        return (1-min(x['p_hit'],max(x['post'])))/COST['hit'] if x['inlog'] else None
    if policy=='evidence':  # error models over all evidence; reviewed neighbours raise / lower the odds
        up=lambda q:x['odds']*q/(1-q+1e-9)/(1+x['odds']*q/(1-q+1e-9));E=em['E'];f=evidence_x(r,x)
        pf,pm,pt,pr=error_probs(r,x,E,f,up,r.retime and not em.get('no_timing'))
        if x['inlog']:return (pm*GAIN['retime']+(pf-pm)*GAIN['false']+(1-pf)*pt*GAIN['type'])/(COST['hit']+pm*COST['retime'])
        return pr*GAIN['miss']/COST['miss']
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

def simulate(rallies,policy,em=None,propagate=False,full=None):
    """em: {'lr': {True: odds factor if the reviewed item was wrong, False: if right}} for suspicion propagation."""
    """Review in priority order; return pooled (tp,fp,fn) after spending each GRID share of the full-review cost."""
    full=full or sum(len(r.log()) for r in rallies)*COST['hit']  # override: keep the review budget of the original log
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
        if em is not None and em.get('contact') is not None and propagate:  # a reviewed hit absorbs unreviewed duplicates logged for the same contact
            if dedupe(r,em['contact']) or changed:r.decode()
        elif em is not None and em.get('structure') is not None and propagate:  # rally-structured propagation: re-decode the log around the fix
            if not structure_decode(r,em['structure'],em.get('pdup',.05)) and changed:r.decode()
        elif changed and propagate:r.decode()
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

GMIN=.3  # two contacts of a rally are at least this far apart: 0.5% of reference intervals are shorter
CW=9  # contact localiser window: +-CW frames
_CF={}
def contact_feats(clip,f):
    """Per-frame contact evidence in +-CW frames around f: impact sound (relative to the window peak), racket-shuttle closeness,
    shuttle turn, share of detector routes firing there, offset from f, f itself."""
    key=(clip['clip_id'],f)
    if key not in _CF:
        from benchmarks.bfmd.verify import match_lag,_RK,racket_contact
        from benchmarks.bfmd.refine import turn
        from benchmarks.bfmd import audio
        racket_contact(clip,f);rk,xy=_RK[clip['clip_id']];lag=match_lag(clip);t,fl=audio.flux(clip['path'])
        route_types(clip,f);rfr=[round(h['time']*30) for hits in _ROUTES[clip['clip_id']] for h in hits];F=[]
        for g in range(f-CW,f+CW+1):
            m=(t>=g/30+lag-1/60)&(t<g/30+lag+1/60);snd=float(np.log1p(fl[m].max())) if m.any() else 0.;best=9.
            if 0<=g<min(len(xy),len(rk)) and not np.isnan(xy[g]).any():
                for rr in rk[g]['rackets']:
                    kp=np.array(rr['keypoints']);b=np.array(rr['bbox'][0])
                    best=min(best,float(np.linalg.norm(xy[g]-kp[[0,3,4]].mean(0))/(np.hypot(b[2]-b[0],b[3]-b[1])+1e-6)))
            F.append([snd,-np.log1p(best),turn(xy,g) if 2<=g<len(xy)-2 else 0.,sum(abs(q-g)<=1 for q in rfr),abs(g-f)/CW,float(g==f)])
        F=np.array(F);F[:,0]-=F[:,0].max();F[:,3]/=max(1.,F[:,3].max());_CF[key]=F
    return _CF[key]

def ref_frames(clip):
    m=next(x for x in read(_R.parent/'03_数据/独立基准/BFMD/manifest.json')['matches'] if x['match_id']==clip['match_id'])
    lo=clip['offset_seconds'];hi=lo+clip['duration_seconds']
    return [round((h['time']-lo)*30) for h in m['hits'] if lo<=h['time']<hi and h['label'] not in SERVES]

def fit_contact(rallies,l2=1.,steps=400,lr=.5):
    """Contact localiser (conditional logit over the window) trained on logged hits of training clips that have a reference contact nearby."""
    X=[];Y=[]
    for r in rallies:
        gt=ref_frames(r.clip)
        for x in r.c:
            if not x['inlog']:continue
            near=[g for g in gt if abs(g-x['frame'])<=CW]
            if near:X.append(contact_feats(r.clip,x['frame']));Y.append(min(near,key=lambda g:abs(g-x['frame']))-x['frame']+CW)
    X=np.stack(X);Y=np.array(Y);d=X.shape[2];mu=X.reshape(-1,d).mean(0);sd=X.reshape(-1,d).std(0)+1e-6;Z=(X-mu)/sd;wv=np.zeros(d)
    for _ in range(steps):
        s=Z@wv;P=np.exp(s-s.max(1,keepdims=True));P/=P.sum(1,keepdims=True)
        wv-=lr*(((Z*P[:,:,None]).sum(1)-Z[np.arange(len(Y)),Y]).mean(0)+l2*wv/len(Y))
    return {'mu':mu,'sd':sd,'w':wv}

def contact_fit(cm,clip,f):
    """How contact-like frame f is compared with the frames around it (score at f minus the window maximum)."""
    s=((contact_feats(clip,f)-cm['mu'])/cm['sd'])@cm['w'];return float(s[CW]-s.max())
BIG=50.

def structure_init(r,E,cm=None,pdup=.05):
    """Per-candidate log-odds of being a real contact (from the error models on the initial log) and the BST far-player probability.
    With a contact localiser cm: of two likely-real candidates closer than GMIN (one contact logged twice), the one whose frame
    fits the contact evidence worse is treated as the duplicate."""
    tops=side_top(r.clip,[x['frame'] for x in r.c]) if r.c else []
    for x,q in zip(r.c,tops):
        f=evidence_x(r,x)
        p=1-(E.p('fp',f)-min(E.p('fp',f),E.p('mt',f))) if x['inlog'] else E.p('miss',f)
        x['p_real0']=min(max(p,1e-4),1-1e-4);x['top']=q;x['inlog0']=x['inlog']
    if cm is not None:
        seq=sorted([x for x in r.c if x['p_real0']>.5],key=lambda x:x['time'])
        for a,b in zip(seq,seq[1:]):
            if b['time']-a['time']<GMIN:
                worse=min((a,b),key=lambda x:contact_fit(cm,r.clip,x['frame']))
                worse['p_real0']=min(worse['p_real0'],pdup)

def dup_rate(rallies):
    """P(both real) for adjacent logged hits closer than GMIN (training clips)."""
    n=b=0
    for r in rallies:
        seq=sorted([k for k,x in enumerate(r.c) if x['inlog'] and r.y[k] is not None],key=lambda k:r.c[k]['time'])
        for i,j in zip(seq,seq[1:]):
            if r.c[j]['time']-r.c[i]['time']<GMIN:n+=1;b+=r.y[i]==1 and r.y[j]==1
    return (b+1)/(n+2)

def structure_decode(r,lam=1.,pdup=.05):
    """Rally-structured decoding of the log. Choose the candidates that maximise the summed log-odds of being a real contact,
    subject to contacts >= GMIN apart, plus lam x log(2 P(alternate)) for consecutive contacts <= 2.5 s apart
    (P from BST far-player probabilities). Reviewed candidates are fixed in or out. Returns True if the log changed."""
    order=sorted(range(len(r.c)),key=lambda k:r.c[k]['time']);n=len(order)
    if not n:return False
    X=[r.c[k] for k in order];T=[x['time'] for x in X]
    w=[(BIG if x['inlog'] else -np.inf) if x['locked'] else float(np.log(x['p_real0']/(1-x['p_real0']))) for x in X]
    fixed=[i for i in range(n) if w[i]==BIG];best=[-np.inf]*n;prev=[-1]*n
    for i in range(n):
        if w[i]==-np.inf:continue
        opts=[] if any(l<i for l in fixed) else [(0.,-1)]
        for j in range(i):
            if best[j]==-np.inf or any(j<l<i for l in fixed):continue
            if T[i]-T[j]<GMIN and not (w[i]==BIG and w[j]==BIG):continue
            a=0.
            if lam and T[i]-T[j]<=2.5:pa=X[i]['top']*(1-X[j]['top'])+(1-X[i]['top'])*X[j]['top'];a=lam*float(np.log(max(2*pa,1e-3)))
            opts.append((best[j]+a,j))
        if opts:s,j=max(opts);best[i]=s+w[i];prev[i]=j
    ends=[(best[i],i) for i in range(n) if best[i]>-np.inf and not any(l>i for l in fixed)]
    if not fixed:ends.append((0.,-1))
    i=max(ends)[1];keep=set()
    while i>=0:keep.add(i);i=prev[i]
    changed=False
    for i,x in enumerate(X):
        if x['locked'] or (i in keep)==x['inlog']:continue
        x['inlog']=i in keep;changed=True
        if x['inlog']==x['inlog0']:x.pop('auto',None)
        elif x['inlog']:x['auto']=('added',1-x['p_real0'])
        else:  # removed: a duplicate of a kept contact within GMIN is rarely real
            dup=any(k in keep and abs(T[k]-T[i])<GMIN for k in range(n))
            x['auto']=('removed',pdup if dup else x['p_real0'])
    if changed:r.decode()
    return changed

def auto_pass(r,E,th=.5):
    """Second automatic verification before review: drop logged hits the error models call false, add left-out candidates they call real."""
    drop=[];add=[]
    for x in r.c:
        f=evidence_x(r,x)
        if x['inlog'] and (q:=E.p('fp',f)-E.p('mt',f))>th:drop.append((x,q))
        elif not x['inlog'] and (q:=E.p('miss',f))>th:add.append((x,q))
    # a moved item keeps the probability that moved it: P(real) = 1 - P(spurious) for a dropped hit, P(false) = 1 - P(real) for an added one
    for x,q in drop:x['inlog']=False;x['auto']=('removed',1-q)
    for x,q in add:x['inlog']=True;x['auto']=('added',1-q)
    if drop or add:r.decode()

CDELTA=3.  # move a logged hit only if the contact model prefers another frame by this many log-odds (chosen on training matches in every fold)
CGAP=9     # one contact per 0.3 s

def relabel(r):
    """Reference matching after hits moved: logged hits are matched first, then left-out candidates."""
    order=sorted(range(len(r.c)),key=lambda k:(not r.c[k]['inlog'],r.c[k]['time']));cs=[r.c[k] for k in order]
    y=truth_labels(r.clip,cs);ty=truth_types(r.clip,cs);r.y=[None]*len(r.c);r.ty=[None]*len(r.c)
    for k,a,b in zip(order,y,ty):r.y[k]=a;r.ty[k]=b
    r.near=near_truth(r.clip,r.c,r.y)

def dedupe(r,pdup,gap=CGAP):
    """Keep one logged hit per contact: of two closer than gap frames, drop the unreviewed one whose frame fits the contact model worse."""
    xs=sorted([x for x in r.c if x['inlog']],key=lambda x:x['frame']);keep=[];changed=False
    for x in xs:
        if keep and x['frame']-keep[-1]['frame']<gap and not (x['locked'] and keep[-1]['locked']):
            a=keep[-1]
            loser=x if a['locked'] or (not x['locked'] and a.get('cscore',0.)>=x.get('cscore',0.)) else a
            loser['inlog']=False;loser['auto']=('removed',pdup);changed=True
            if loser is a:keep[-1]=x
        else:keep.append(x)
    return changed

def contact_pass(r,cmod,pdup,delta=CDELTA):
    """Before review: move logged hits to the frame the contact-time model prefers, then merge hits logged twice for one contact."""
    from benchmarks.bfmd import contact as ct
    xs=[x for x in r.c if x['inlog'] and not x['locked']]
    if not xs:return
    F=ct.frame_features(r.clip);S=cmod.scores(np.stack([ct.window(F,x['frame']) for x in xs]))
    for x,s in zip(xs,S):
        s=np.where((x['frame']+np.arange(len(s))-ct.W>=0)&(x['frame']+np.arange(len(s))-ct.W<len(F)),s,-np.inf);p=int(np.argmax(s));z=float(np.logaddexp.reduce(s));x['cscore']=float(s[ct.W])-z;x['cbest']=x['frame']+p-ct.W;x['cprob']=float(np.exp(s[p]-z))
        if s[p]-s[ct.W]>delta:x['frame']+=p-ct.W;x['time']=x['frame']/30;x['cscore']=float(s[p])-z;x['cmoved']=True
    dedupe(r,pdup);relabel(r);r.decode()

def fit_contact_model(train_clips,tr):
    """Contact-time CNN on the training matches' reference contacts; offset prior on their logged hits."""
    from benchmarks.bfmd import contact as ct
    X,y=ct.training_windows(train_clips);cmod=ct.Model(X.shape[2]).fit(X,y);P=[]
    for r in tr:
        gt=ref_frames(r.clip);F=ct.frame_features(r.clip)
        for x in r.c:
            near=[g for g in gt if abs(g-x['frame'])<=ct.W]
            if x['inlog'] and near:P.append((ct.window(F,x['frame']),min(near,key=lambda g:abs(g-x['frame']))-x['frame']+ct.W))
    return cmod.fit_prior(np.stack([a for a,_ in P]),np.array([b for _,b in P]))

def oracle_structure(r):
    """Ceiling of automatic rally-structure and timing fixes: with the reference, drop every logged duplicate of a true hit
    (<=0.4 s) and move every mistimed hit to its contact (stroke type left as predicted)."""
    tp=[k for k,x in enumerate(r.c) if x['inlog'] and r.y[k]==1]
    for k,x in enumerate(r.c):
        if not x['inlog'] or r.y[k]!=0 or x['locked']:continue
        if r.near[k]:
            g,lab=r.near[k];x['time']=g;x['frame']=round(g*30);r.y[k]=1.;r.ty[k]=lab;r.near[k]=None
            for j,v in enumerate(r.near):
                if v and abs(v[0]-g)<1e-9:r.near[j]=None
        elif any(abs(r.c[j]['time']-x['time'])<=.4 for j in tp):x['inlog']=False
    r.decode()

def run_fold(train_clips,test_clips,model,th,tmodel,lam,E=None):
    flat=np.full((len(TYPES),len(TYPES)),1/len(TYPES))
    if os.environ.get('REVIEW_CONTACT')=='1' and E is not None:  # contact-time model: timing fixes and duplicate merging before / during review
        tr=[Rally(c,model,th,tmodel,flat,0) for c in train_clips];cmod=fit_contact_model(train_clips,tr);pdup=dup_rate(tr);out={'_pdup':pdup}
        tens=det=None;DENSE=os.environ.get('REVIEW_DENSE')=='1'
        if DENSE:from benchmarks.bfmd import dense
        if os.environ.get('REVIEW_TYPEENS')=='1':  # stroke-type ensemble fitted on the training matches' logs after their automatic passes
            from benchmarks.bfmd import typeens
            for r in tr:auto_pass(r,E['all']);contact_pass(r,cmod,pdup)
            if DENSE:  # dense hit detector on the training matches' annotated contacts
                det=dense.Detector().fit(train_clips)
                for r in tr:dense.dense_pass(r,det)
            tens=typeens.fit(tr,train_clips)
        for name,pol,useC,prop in [('auto pass + evidence','evidence',False,False),('auto pass + contact + evidence','evidence',True,False),
                                   ('auto pass + contact + evidence + propagation','evidence',True,True),('auto pass + contact + oracle','oracle',True,False)]:
            rs=[Rally(c,model,th,tmodel,flat,lam) for c in test_clips];full=sum(len(r.log()) for r in rs)*COST['hit']
            for r in rs:
                auto_pass(r,E['all'])
                if useC:contact_pass(r,cmod,pdup)
                if useC and det is not None:dense.dense_pass(r,det)
                if useC and tens is not None:typeens.type_pass(r,tens)
            if useC and prop:out['_type']=[sum(1 for i,x in enumerate(r.c) if x['inlog'] and r.y[i]==1 and r.ty[i] and x['label']==r.ty[i]) for r in rs],[sum(1 for i,x in enumerate(r.c) if x['inlog'] and r.y[i]==1 and r.ty[i]) for r in rs]
            em={'E':E['all'],'contact':pdup} if pol=='evidence' and prop else ({'E':E['all']} if pol=='evidence' else None)
            out[name]=simulate(rs,pol,em,prop,full)
        return out
    if os.environ.get('REVIEW_CEILING')=='1' and E is not None:  # only the runs needed for the ceiling of rally structure + timing
        out={}
        for name,ceil in [('auto pass + evidence',False),('ceiling: perfect structure+timing + evidence',True)]:
            rs=[Rally(c,model,th,tmodel,flat,lam) for c in test_clips];full=sum(len(r.log()) for r in rs)*COST['hit']
            for r in rs:
                auto_pass(r,E['all'])
                if ceil:oracle_structure(r)
            out[name]=simulate(rs,'evidence',{'E':E['all']},False,full)
        return out
    tr=[Rally(c,model,th,tmodel,flat,0) for c in train_clips];T=transitions(tr);lr={'lr':suspicion_lr(tr)};mm=miss_model(tr);pm=parity_model(tr);out={'_lr':lr['lr']}
    runs=[('chronological','chronological',None,False),('single_hit','single_hit',None,False),('single_type','single_type',None,False),('single','single',None,False),
          ('confidence','confidence',None,False),('gain+racket+suspicion+parity','gain',{**lr,'miss':mm,'parity':pm},True)]
    if E is not None:runs+=[('evidence','evidence',{'E':E['all']},False),('evidence+propagation','evidence',{**lr,'E':E['all']},True),
                            ('evidence - rally structure','evidence',{'E':E['no rally structure']},False),('evidence - new sources','evidence',{'E':E['no new sources']},False),
                            ('evidence - timing in ranking','evidence',{'E':E['all'],'no_timing':True},False)]
    for name,pol,em,prop in runs+[('oracle','oracle',None,False)]:
        out[name]=simulate([Rally(c,model,th,tmodel,T,lam) for c in test_clips],pol,em,prop)
    if E is not None:  # second automatic pass by the error models, then the same review budget as the original log
        for name,pol,em in [('auto pass + evidence','evidence',{'E':E['all']}),('auto pass + oracle','oracle',None)]:
            rs=[Rally(c,model,th,tmodel,T,lam) for c in test_clips];full=sum(len(r.log()) for r in rs)*COST['hit']
            for r in rs:auto_pass(r,E['all'])
            out[name]=simulate(rs,pol,em,False,full)
        pdup=dup_rate(tr);out['_pdup']=pdup;cm=fit_contact(tr)
        for name,sl,prop,pol,useC in [('structure + evidence',1.,False,'evidence',False),('structure + evidence + propagation',1.,True,'evidence',False),
                                      ('structure+contact + evidence',1.,False,'evidence',True),('structure+contact + evidence + propagation',1.,True,'evidence',True),
                                      ('structure+contact + oracle',1.,False,'oracle',True)]:
            rs=[Rally(c,model,th,tmodel,T,lam) for c in test_clips];full=sum(len(r.log()) for r in rs)*COST['hit']
            for r in rs:structure_init(r,E['all'],cm if useC else None,pdup);structure_decode(r,sl,pdup)
            out[name]=simulate(rs,pol,{'E':E['all'],'structure':sl,'pdup':pdup} if pol=='evidence' else None,prop,full)
    return out

def f1(c):tp,fp,fn=c;return 2*tp/(2*tp+fp+fn)
def pool(folds):
    return {k:[f1(tuple(map(sum,zip(*[f[k][g] for f in folds])))) for g in range(len(GRID))] for k in folds[0] if not k.startswith('_')}

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('step',choices=['evaluate','test','lomo','freeze','freeze-contact']);a.add_argument('--lam',type=float,default=1.);a=a.parse_args()
    if a.step=='freeze-contact':  # add a contact-time model to every frozen bundle, trained on the bundle's 11 training matches
        import torch
        from benchmarks.bfmd.verify import ROUTES,PRIMARY
        f=STUDY/'frozen_review_lomo12.json';fz=read(f);out=STUDY/'contact_models';out.mkdir(exist_ok=True)
        clips=[c for c in read(STUDY/'clips.json')['clips'] if (ROUTES/c['clip_id']/f'{PRIMARY}.json').exists()]
        flat=np.full((len(TYPES),)*2,1/len(TYPES))
        for m,b in fz['bundles'].items():
            model=Logistic();model.w,model.mu,model.sd=(np.array(b['hit_model'][k]) for k in ['w','mu','sd'])
            tmodel=Softmax();tmodel.W,tmodel.mu,tmodel.sd=(np.array(b['type_model'][k]) for k in ['W','mu','sd'])
            train=[c for c in clips if c['youtube_id']!=m];tr=[Rally(c,model,b['threshold'],tmodel,flat,0) for c in train]
            cm=fit_contact_model(train,tr);path=out/f'{m}.pt'
            torch.save({'state':cm.net.state_dict(),'mu':cm.mu,'sd':cm.sd,'prior':cm.prior,'d':int(cm.mu.shape[0])},path)
            b['contact']={'path':path.name,'pdup':dup_rate(tr),'delta':CDELTA,'gap':CGAP};print('contact model',m,flush=True)
        save(f,fz);raise SystemExit
    if a.step=='freeze':  # per-match bundles for the review page: each match is served by models that never saw it
        from benchmarks.bfmd.verify import ROUTES,PRIMARY,code_hashes
        clips=[c for c in read(STUDY/'clips.json')['clips'] if (ROUTES/c['clip_id']/f'{PRIMARY}.json').exists()]
        byid={c['clip_id']:c for c in clips};data={}
        for c in clips:cs=cached_candidates(c);data[c['clip_id']]=(cs,truth_labels(c,cs))
        types={k:truth_types(byid[k],data[k][0]) for k in byid};bundles={};hcache={}
        for m in sorted({c['youtube_id'] for c in clips}):
            if TYPEHEAD:attach_heads(clips,data,m,hcache)
            train=[c for c in clips if c['youtube_id']!=m];model,th,tmodel=fit_both([c['clip_id'] for c in train],data,types,byid)
            E=crossfit_error_models(train,data,types,byid)['all'];flat=np.full((len(TYPES),)*2,1/len(TYPES))
            tr=[Rally(c,model,th,tmodel,flat,0) for c in train]
            bundles[m]={'hit_model':model.packet(),'threshold':th,'type_model':{'W':tmodel.W.tolist(),'mu':tmodel.mu.tolist(),'sd':tmodel.sd.tolist()},
                        'error_models':E.packet(),'auto_pass':.5,
                        'head':({'W':hcache[frozenset({m})].W.tolist(),'mu':hcache[frozenset({m})].mu.tolist(),'sd':hcache[frozenset({m})].sd.tolist()} if TYPEHEAD else None),'suspicion_lr':{str(k):v for k,v in suspicion_lr(tr).items()},'train_matches':sorted({c['youtube_id'] for c in train})}
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
        types={k:truth_types(byid[k],data[k][0]) for k in byid};folds=[];hcache={}
        for m in sorted({c['youtube_id'] for c in clips}):
            train=[c for c in clips if c['youtube_id']!=m];test=[c for c in clips if c['youtube_id']==m]
            if TYPEHEAD:attach_heads(clips,data,m,hcache)
            model,th,tmodel=fit_both([c['clip_id'] for c in train],data,types,byid)
            E=crossfit_error_models(train,data,types,byid)
            folds.append(run_fold(train,test,model,th,tmodel,0,E));print('fold',m,len(test),'clips',flush=True)
        if '_type' in folds[0]:print('stroke type among found hits',sum(sum(f['_type'][0]) for f in folds),'/',sum(sum(f['_type'][1]) for f in folds),flush=True)
        res=pool(folds);save(STUDY/f"review_lomo12{'_ablate_'+ablate if ablate else ''}{'' if RETIME_ON else '_noretime'}{'_typehead'+os.environ['REVIEW_TYPEHEAD'] if TYPEHEAD else ''}{'_geom' if os.environ.get('REVIEW_GEOM')=='1' else ''}{'_ceiling' if os.environ.get('REVIEW_CEILING')=='1' else ''}{'_contact' if os.environ.get('REVIEW_CONTACT')=='1' else ''}{'_dense' if os.environ.get('REVIEW_DENSE')=='1' else ''}{'_typeens' if os.environ.get('REVIEW_TYPEENS')=='1' else ''}{'_l2'+os.environ['VERIFY_TYPE_L2'] if os.environ.get('VERIFY_TYPE_L2','10')!='10' else ''}.json",{'grid_share':GRID,'curves':res,'cost':COST,'clips':len(clips),
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
