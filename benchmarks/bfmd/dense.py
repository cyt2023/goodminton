"""Dense hit detector: a temporal CNN over the per-frame evidence (shuttle, pose, racket, sound, detector votes) trained on the
annotated contacts of the training matches; its peaks become the automatic log."""
import numpy as np
from benchmarks.bfmd import contact as ct

TH=.5   # peak threshold (hit F1 is flat between 0.4 and 0.6)
GAP=6   # one contact per 0.2 s
REACH=9 # a peak takes over the nearest candidate within 0.3 s

def target(n,frames,sd=1.2):
    y=np.zeros(n,np.float32)
    for f in frames:
        for t in range(max(0,f-4),min(n,f+5)):y[t]=max(y[t],np.exp(-(t-f)**2/(2*sd**2)))
    return y

class Detector:
    SEEDS=3
    def fit(self,clips,epochs=50):
        import torch
        data=[(ct.frame_features(c),ct.reference_contacts(c)) for c in clips];allF=np.concatenate([F for F,_ in data]);self.mu,self.sd=allF.mean(0),allF.std(0)+1e-6;self.nets=[]
        X=[torch.from_numpy((F-self.mu)/self.sd).T[None] for F,_ in data];Y=[torch.from_numpy(target(len(F),g)) for F,g in data]
        for seed in range(self.SEEDS):
            torch.manual_seed(seed);rng=np.random.default_rng(seed);m=self.build(allF.shape[1]);opt=torch.optim.AdamW(m.parameters(),2e-3,weight_decay=1e-2)
            sched=torch.optim.lr_scheduler.OneCycleLR(opt,3e-3,total_steps=epochs*len(X));m.train()
            for _ in range(epochs):
                for i in rng.permutation(len(X)):
                    loss=torch.nn.functional.binary_cross_entropy_with_logits(m(X[i])[0,0],Y[i],pos_weight=torch.tensor(3.));opt.zero_grad();loss.backward();opt.step();sched.step()
            m.eval();self.nets.append(m)
        return self
    @staticmethod
    def build(d):
        import torch
        nn=torch.nn;L=[];ch=d
        for dil in (1,2,4,8,16,1):L+=[nn.Conv1d(ch,64,5,padding=2*dil,dilation=dil),nn.ReLU(),nn.Dropout(.2)];ch=64
        return nn.Sequential(*L,nn.Conv1d(64,1,1))
    def proba(self,clip):
        import torch
        x=torch.from_numpy((ct.frame_features(clip)-self.mu)/self.sd).T[None]
        with torch.no_grad():return np.mean([torch.sigmoid(m(x)[0,0]).numpy() for m in self.nets],0)
    def peaks(self,clip,th=TH):
        p=self.proba(clip);out=[]
        for t in np.argsort(-p):
            if p[t]<th:break
            if (t==0 or p[t]>=p[t-1]) and (t==len(p)-1 or p[t]>=p[t+1]) and all(abs(t-u)>=GAP for u in out):out.append(int(t))
        return sorted(out)
    def __getstate__(self):return {'mu':self.mu,'sd':self.sd,'states':[{k:v.numpy() for k,v in m.state_dict().items()} for m in self.nets]}
    def __setstate__(self,s):
        import torch
        self.mu,self.sd=s['mu'],s['sd'];self.nets=[]
        for st in s['states']:m=self.build(len(self.mu));m.load_state_dict({k:torch.from_numpy(v) for k,v in st.items()});m.eval();self.nets.append(m)

def dense_pass(r,det):
    """Before review: the log becomes the detector's peaks, each carried by the nearest unreviewed candidate (moved to the peak frame)."""
    from benchmarks.bfmd.review import relabel
    free={i for i,x in enumerate(r.c) if not x['locked']}
    for i in free:r.c[i]['inlog']=False
    for t in det.peaks(r.clip):
        cand=[i for i in free if abs(r.c[i]['frame']-t)<=REACH]
        if not cand:continue
        i=min(cand,key=lambda i:(abs(r.c[i]['frame']-t),-r.c[i]['p_hit']));free.discard(i);x=r.c[i];x['inlog']=True;x['dense']=True
        if x['frame']!=t:x['cmoved']=True
        x['frame']=t;x['time']=t/30;x['cbest']=t
    relabel(r);r.decode()
