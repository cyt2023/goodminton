"""3D shuttle flight between two hits from one camera: the court corners give the camera (focal length from the court-plane
homography), and a flight obeying gravity and shuttle drag is fitted to the tracked image points, anchored at the hitter's and the
receiver's court positions. This recovers what the image alone hides: real speed, launch angle, contact height, net clearance."""
import numpy as np
from scipy.optimize import least_squares

CW,CL,NET_Y,NET_H=6.1,13.4,6.7,1.55   # court width / length (m), net position and height
G,VT=9.81,6.86                         # gravity, shuttle terminal speed (m/s)
KD=G/VT**2

def camera(corners,W,H):
    """3x4 projection of court metres (x across, y from the far baseline, z up) from the four court corners TL,TR,BL,BR (pixels)."""
    import cv2
    world=np.array([[0,0],[CW,0],[0,CL],[CW,CL]],float);Hm=cv2.findHomography(world,np.asarray(corners,float))[0]
    T=np.array([[1,0,-W/2],[0,1,-H/2],[0,0,1.]]);h=T@Hm;h1,h2,h3=h[:,0],h[:,1],h[:,2]
    cand=[]
    d=h1[2]**2-h2[2]**2
    if abs(d)>1e-12:cand.append((h2[0]**2+h2[1]**2-h1[0]**2-h1[1]**2)/d)
    if abs(h1[2]*h2[2])>1e-12:cand.append(-(h1[0]*h2[0]+h1[1]*h2[1])/(h1[2]*h2[2]))
    cand=[c for c in cand if c>0];f=float(np.sqrt(cand[0])) if cand else 1.5*W
    Ki=np.diag([1/f,1/f,1.]);r1,r2,t=Ki@h1,Ki@h2,Ki@h3;lam=(np.linalg.norm(r1)+np.linalg.norm(r2))/2;r1,r2,t=r1/lam,r2/lam,t/lam
    if t[2]<0:r1,r2,t=-r1,-r2,-t
    r3=np.cross(r1,r2);K=np.array([[f,0,W/2],[0,f,H/2],[0,0,1.]])
    def P(r3):return K@np.c_[r1,r2,r3,t]
    Pm=P(r3);a=Pm@np.array([CW/2,CL/2,0,1.]);b=Pm@np.array([CW/2,CL/2,1.,1.])
    if b[1]/b[2]>a[1]/a[2]:Pm=P(-r3)   # one metre up must be higher in the image
    return Pm,f

def project(P,X):
    v=P@np.r_[X,1.];return v[:2]/v[2]

def simulate(p0,v0,nframes,sub=3):
    """Positions at frames 0..nframes after the hit under gravity and quadratic drag."""
    p=np.array(p0,float);v=np.array(v0,float);dt=1/30/sub;out=[p.copy()]
    for _ in range(nframes):
        for _ in range(sub):
            a=-KD*np.linalg.norm(v)*v;a[2]-=G;v=v+a*dt;p=p+v*dt
        out.append(p.copy())
    return np.array(out)

def fit(P,obs,start_xy,end_xy,nframes,a_start=None,a_end=None,sig=.35,q_init=None):
    """obs: [(frames after the hit, x, y)] image points of this flight; start_xy / end_xy: hitter's and receiver's feet on the court
    (m), end_xy None when nobody returns it; a_start / a_end: 3D contact points shared with the neighbouring flights (rally chain).
    Returns the flight (positions per frame), initial velocity, RMS reprojection error, landing point, solution vector."""
    ts=np.array([o[0] for o in obs],int);uv=np.array([[o[1],o[2]] for o in obs],float);T=max(nframes,int(ts.max()) if len(ts) else 1,1)
    wf=.8 if a_start is None else 2.;we=1.2 if a_end is None else 2.5   # feet priors loosen once the chain supplies the contact points
    def res(q):
        tr=simulate(q[:3],q[3:],T);pr=np.array([project(P,tr[t]) for t in ts]) if len(ts) else np.zeros((0,2))
        r=list(((pr-uv)/4.).ravel())+list((q[:2]-start_xy)/wf)+[(q[2]-1.4)/1.5]
        if end_xy is not None:r+=list((tr[nframes][:2]-end_xy)/we)+[(tr[nframes][2]-1.2)/1.5]
        if a_start is not None:r+=list((q[:3]-a_start)/sig)
        if a_end is not None:r+=list((tr[nframes]-a_end)/sig)
        r+=[min(tr[:,2].min(),0.)/.2];return np.array(r)
    tsec=max(nframes,1)/30;best=None;lo,hi=np.array([-1,-1,0,-90,-90,-60.]),np.array([CW+1,CL+1,3.3,90,90,60.])
    tgt=np.array(end_xy) if end_xy is not None else np.array([start_xy[0],CL-start_xy[1]])
    inits=[np.clip(q_init,lo+1e-6,hi-1e-6)] if q_init is not None else [np.r_[start_xy,z0,(tgt-np.array(start_xy))/tsec*boost,.5*G*tsec+(1.2-z0)/tsec] for z0,boost in ((1.2,1.),(2.4,1.6),(.6,2.5))]
    for q0 in inits:
        try:s=least_squares(res,np.clip(q0,lo+1e-6,hi-1e-6),bounds=(lo,hi),max_nfev=60)
        except Exception:continue
        if best is None or s.cost<best.cost:best=s
    if best is None:return None
    q=best.x;full=simulate(q[:3],q[3:],max(T,nframes)+60);tr=full[:T+1];pr=np.array([project(P,tr[t]) for t in ts]) if len(ts) else np.zeros((0,2))
    k=next((i for i in range(1,len(full)) if full[i,2]<=0),len(full)-1)   # where it would reach the floor if nobody took it
    return tr[:nframes+1],q[3:],float(np.sqrt(((pr-uv)**2).sum(1).mean())) if len(ts) else 99.,np.r_[full[k,:2],k/30],q

def chain(P,flights,sweeps=2):
    """Rally-chained fit. flights: [dict(obs, start_xy, end_xy, nframes, linked)] in time order; linked = the next flight starts where
    this one ends (same contact). Each flight is first fitted alone; then the contact point between two linked flights is set to the
    average of where one says it ends and the other says it starts, weighted by how well each is determined (long, well-fitted arcs
    pin their ends down through gravity; flat fast ones do not), and every flight is refitted through its two contact points."""
    R=[fit(P,f['obs'],f['start_xy'],f['end_xy'],f['nframes']) if f['start_xy'] is not None and len(f['obs'])>=3 else None for f in flights]
    for _ in range(sweeps):
        w=[(f['nframes']/30)**2/(1.+r[2]) if r is not None else 0. for f,r in zip(flights,R)];C={}
        for k in range(len(flights)-1):
            if flights[k]['linked'] and R[k] is not None and R[k+1] is not None and w[k]+w[k+1]>0:
                C[k+1]=(w[k]*R[k][0][-1]+w[k+1]*R[k+1][0][0])/(w[k]+w[k+1])
        new=[]
        for k,(f,r) in enumerate(zip(flights,R)):
            if r is None or (k not in C and k+1 not in C):new.append(r);continue
            n=fit(P,f['obs'],f['start_xy'],f['end_xy'],f['nframes'],C.get(k),C.get(k+1),q_init=np.r_[C.get(k,r[0][0]),r[1]]);new.append(n if n is not None else r)
        R=new
    return R

def features(tr,v0,err,land,q,side):
    """Stroke descriptors in metres and m/s, mirrored so every hitter plays from y=0 towards +y."""
    if side>.5:tr=tr*np.array([1,-1,1])+np.array([0,CL,0]);v0=v0*np.array([1,-1,1]);land=land*np.array([1,-1,1])+np.array([0,CL,0])
    sp=float(np.linalg.norm(v0));hz=float(np.hypot(v0[0],v0[1]));z=tr[:,2];y=tr[:,1];n=len(tr)-1
    k=next((i for i in range(1,len(tr)) if (y[i-1]-NET_Y)*(y[i]-NET_Y)<=0),None)
    netz=float(z[k-1]+(z[k]-z[k-1])*(NET_Y-y[k-1])/(y[k]-y[k-1]+1e-9)) if k else 0.
    vend=(tr[-1]-tr[-2])*30 if n>=1 else np.zeros(3)
    return [sp,float(v0[2]),hz,float(np.degrees(np.arctan2(v0[2],hz+1e-6))),float(z[0]),float(y[0]),float(abs(tr[0,0]-CW/2)),float(z.max()),float(np.argmax(z)/max(n,1)),
            netz,float(k/max(n,1)) if k else 1.,float(k is not None),float(y[-1]),float(abs(tr[-1,0]-CW/2)),float(z[-1]),float(np.linalg.norm(tr[-1,:2]-tr[0,:2])),
            float(np.linalg.norm(vend)),float(vend[2]),float(abs(tr[-1,0]-tr[0,0])),n/30,min(err,50.),
            float(land[1]-NET_Y),float(abs(land[0]-CW/2)),float(land[2]),float(land[1]-NET_Y<=1.98),float(y[-1]-NET_Y)]   # landing past the net, before the short service line
NF=26
