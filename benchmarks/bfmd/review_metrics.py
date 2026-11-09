"""Review-efficiency metrics from review_lomo12.json (leave-one-match-out over 12 BFMD singles matches).

extra fixed : (F1 after review - F1 before) of a method / the same for a baseline, minus 1, at a review budget
time saved  : 1 - (review share a method needs to reach a target joint F1) / (share the baseline needs)
95% intervals: bootstrap over matches (counts pooled within each resample).
"""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from benchmarks.bfmd.verify import STUDY

def f1(c):tp,fp,fn=c;return 2*tp/(2*tp+fp+fn)

def curves(per_match,matches):
    pol=next(iter(per_match.values())).keys()
    return {p:np.array([f1(np.sum([per_match[m][p][g] for m in matches],0)) for g in range(len(GRID))]) for p in pol}

def reach(grid,curve,target):
    for g in range(1,len(grid)):
        if curve[g]>=target:
            a,b=curve[g-1],curve[g];return grid[g-1]+(grid[g]-grid[g-1])*(target-a)/(b-a) if b>a else grid[g]
    return float('nan')

def metrics(C,method,base,budgets,targets):
    out={}
    for b in budgets:
        g=GRID.index(b);out[f'extra fixed @{b:.0%}']=(C[method][g]-C[method][0])/(C[base][g]-C[base][0])-1
    for t in targets:out[f'time saved to F1 {t}']=1-reach(GRID,C[method],t)/reach(GRID,C[base],t)
    return out

def compare(pm_a,method,pm_b,base,budgets,targets,n=1000,seed=0):
    """Point estimate and bootstrap (matches resampled jointly) of method (run A) vs base (run B)."""
    ms=sorted(pm_a);rng=np.random.default_rng(seed)
    def one(sample):
        A=curves(pm_a,sample);B=curves(pm_b,sample);return metrics({'m':A[method],'b':B[base]},'m','b',budgets,targets)
    point=one(ms);boots=[one(list(rng.choice(ms,len(ms)))) for _ in range(n)]
    return {k:(v,*np.percentile([b[k] for b in boots if np.isfinite(b[k])],[2.5,97.5])) for k,v in point.items()}

if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('method',nargs='?',default='evidence');a.add_argument('--run',default='')
    a.add_argument('--base-run',default=None,help="compare against policies of another run, e.g. 'ablate_sources'");a=a.parse_args()
    load=lambda r:json.loads((STUDY/f"review_lomo12{'_'+r if r else ''}.json").read_text())
    d=load(a.run);GRID=d['grid_share'];pm=d['per_match']
    bd=load(a.base_run) if a.base_run is not None else d;pb=bd['per_match']
    budgets=[.1,.2,.3,.5];targets=[.78,.80,.82,.85]
    C=curves(pm,sorted(pm));print('pooled joint F1 by review share',[f'{g:.2f}' for g in GRID])
    for k,v in C.items():print(f'  {k:28s}',' '.join(f'{x:.3f}' for x in v))
    bases=[b for b in ['chronological','single_hit','single_type','single','confidence','gain+racket+suspicion+parity','evidence'] if b in curves(pb,sorted(pb)) and (b!=a.method or a.base_run is not None)]
    for base in bases:
        print(f"\n{a.method} ({a.run or 'full'}) vs {base} ({a.base_run if a.base_run is not None else a.run or 'full'})")
        for k,(v,lo,hi) in compare(pm,a.method,pb,base,budgets,targets).items():print(f'  {k:24s} {v:+.0%}  [{lo:+.0%}, {hi:+.0%}]')
