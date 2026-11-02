"""Seeded rally sampling and CFR clip cutting for the BFMD pipeline-selection study.
Sampling reads rally start/end only; hit and stroke references stay with the scorer.
"""
import argparse,json,random,subprocess,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
STORE=ROOT.parent/'03_数据/独立基准/BFMD'
SPLIT={'dev':['PSEgVtdtLJ0','ISFs9Gku1LA','Bxhh21Rsk0E','eKPtHLGFxLE','_lcVkJifJnA','voPcKnC2KgY','L7fSFiWnwRk','rie6e0WhOB8'],
       'sealed_test':['67HGajaYmUM','TDn26B6iAjg','y-YPKEW5VUs','wNAPckJigDo']}
PER_MATCH={'dev':8,'sealed_test':10};SEED=20261102;PAD=1.0

def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def file_sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def sample(rallies,k,rng):
    # Stratify by duration terciles so short and long rallies are both represented.
    ordered=sorted(rallies,key=lambda r:r['end']-r['start']);bins=[ordered[i*len(ordered)//3:(i+1)*len(ordered)//3] for i in range(3)]
    quota=[k//3+(i<k%3) for i in range(3)]
    return sorted([r for b,q in zip(bins,quota) for r in rng.sample(b,min(q,len(b)))],key=lambda r:r['start'])

def plan():
    manifest=read(STORE/'manifest.json');by_id={m['youtube_id']:m for m in manifest['matches']};rows=[]
    for role,ids in SPLIT.items():
        for vid in ids:
            rng=random.Random(f'{SEED}:{vid}')
            for r in sample(by_id[vid]['rallies'],PER_MATCH[role],rng):
                start=max(0.,r['start']-PAD);end=r['end']+PAD
                rows.append({'clip_id':f"{vid}_{round(start*1000):08d}",'youtube_id':vid,'match_id':by_id[vid]['match_id'],'role':role,'rally_id':r['id'],'offset_seconds':start,'duration_seconds':end-start})
    return {'seed':SEED,'pad_seconds':PAD,'split':SPLIT,'per_match':PER_MATCH,'sampling_inputs':'rally start/end only','clips':rows}

def cut(p,ffmpeg):
    out=STORE/'videos/rallies';out.mkdir(parents=True,exist_ok=True)
    for c in p['clips']:
        target=out/(c['clip_id']+'.mp4');src=STORE/'videos/full'/(c['youtube_id']+'.mp4')
        if not target.exists() and not src.exists():continue  # source not downloaded yet
        if not target.exists():
            subprocess.run([ffmpeg,'-y','-loglevel','error','-ss',f"{c['offset_seconds']:.6f}",'-i',str(src),'-t',f"{c['duration_seconds']:.6f}",
                            '-r','30','-fps_mode','cfr','-c:v','libx264','-crf','16','-preset','fast','-c:a','aac',str(target)],check=True)
        c['path']=str(target);c['sha256']=file_sha(target)
    return p

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--cut',action='store_true');a.add_argument('--ffmpeg');a=a.parse_args()
    target=STORE/'pipeline_study/clips.json';p=plan()
    if target.exists() and [c['clip_id'] for c in read(target)['clips']]!=[c['clip_id'] for c in p['clips']]:raise ValueError('Frozen clip plan changed')
    if a.cut:p=cut(p,a.ffmpeg)
    save(target,p);print(json.dumps({r:sum(c['role']==r for c in p['clips']) for r in SPLIT}),flush=True)
