"""Explicit input contracts and data conversion for the local paper replicas."""
from pathlib import Path
import csv,io,json,hashlib,zipfile
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parent

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def pack_shf(top,bottom,court,net,shuttle,*,normalized=True,width=None,height=None):
    arrays=[np.asarray(v,dtype=np.float32) for v in [top,bottom,court,net,shuttle]]
    count=len(arrays[0]);shapes=[(count,17,2),(count,17,2),(count,6,2),(count,4,2),(count,2)]
    for a,s in zip(arrays,shapes):
        if a.shape!=s or not np.isfinite(a).all():raise ValueError(f'Expected finite {s}, received {a.shape}')
    if not normalized:
        if not width or not height:raise ValueError('Native image width/height required for pixel coordinates')
        arrays=[a/np.asarray([width,height],np.float32) for a in arrays]
    return np.concatenate([a.reshape(count,-1) for a in arrays],-1).astype(np.float32)

def ground_midpoints(corners):
    # Four image corners TL,TR,BR,BL. Physical long-edge midpoints are projected
    # by the ground-plane homography; the upper net tape is NOT on this plane.
    import cv2
    rectangle=np.array([[0,0],[1,0],[1,1],[0,1]],np.float32)
    h=cv2.getPerspectiveTransform(rectangle,corners.astype(np.float32))
    return cv2.perspectiveTransform(np.array([[[0,.5],[1,.5]]],np.float32),h)[0]

def prepare_author_shf(out,max_rallies=0,domain='domain_pro'):
    if domain not in ['domain_pro','domain_am_singles']:raise ValueError('Only released singles domains are supported')
    archive=PROJECT/'03_数据/找拍模型对照/2026-10-22/extension_assets/domain.zip'
    out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=[];counts={}
    with zipfile.ZipFile(archive) as z:
        names=sorted([n for n in z.namelist() if n.endswith('_x.csv') and '/'+domain+'/match' in n])
        for name in names:
            match=Path(name).parent.name;number=int(match.removeprefix('match')) if domain=='domain_pro' else None
            split=('train' if number<=18 else 'val' if number<=21 else 'test') if domain=='domain_pro' else 'test'
            if max_rallies and counts.get(split,0)>=max_rallies:continue
            raw=z.read(name);yr=z.read(name[:-6]+'_y.csv')
            x=list(csv.DictReader(io.StringIO(raw.decode())));y=list(csv.DictReader(io.StringIO(yr.decode())))
            if len(x)!=len(y) or any(int(a['Frame'])!=int(b['frame']) for a,b in zip(x,y)):raise ValueError('Unaligned CSV rows')
            if any(int(r['Frame'])!=i for i,r in enumerate(x)):raise ValueError('SHF windows require consecutive source frames')
            def columns(keys):return np.array([[float(r[k]) for k in keys] for r in x],np.float32)
            top=columns([f'far_pose{j}_{a}' for j in range(17) for a in 'xy']).reshape(-1,17,2)
            bottom=columns([f'near_pose{j}_{a}' for j in range(17) for a in 'xy']).reshape(-1,17,2)
            if np.median(top[:,0,0])>.9 or np.median(bottom[:,0,0])>.9:raise ValueError('Unexpected confidence-prefixed legacy pose layout; do not silently shift/drop joints')
            corners=columns([f'{k}_{a}' for k in ['tl','tr','br','bl'] for a in 'xy']).reshape(-1,4,2)
            court=np.concatenate([corners,np.stack([ground_midpoints(c) for c in corners])],1)
            net=np.zeros((len(x),4,2),np.float32)
            feat=pack_shf(top,bottom,court,net,columns(['ball_x','ball_y']))
            # Published organise_input_features source lines575–583:1=near,2=far.
            old=np.array([int(r['player_hit']) for r in y]);labels=np.where(old==1,2,np.where(old==2,1,0))
            if not set(old).issubset({0,1,2}):raise ValueError('Unknown author hitter label')
            path=out/f'{match}_{Path(name).stem[:-2]}.npz'
            np.savez_compressed(path,features=feat,labels=labels)
            rows.append({'id':path.stem,'path':str(path.resolve()),'match':match,'split':split,'frames':len(x),
                         'hits':int((labels>0).sum()),'missing_modalities':['net'],
                         'input_members':{name:hashlib.sha256(raw).hexdigest(),name[:-6]+'_y.csv':hashlib.sha256(yr).hexdigest()}})
            counts[split]=counts.get(split,0)+1
    manifest={'component':'shuttlehitformer','scope':'missing_net_ablation','deployment_ready':False,
              'source_archive':str(archive),'source_sha256':sha256(archive),'rows':rows,
              'split_protocol':'whole matches:1–18 train,19–21 val,22–23 test; original published test_match folders untouched' if domain=='domain_pro' else 'All released amateur singles matches held out from training and validation',
              'labels':['no_hit','top_far','bottom_near'],'limitations':['Other hit-detection authors’ data, not the SHF private37 matches',
              'All8 net coordinates explicitly zero: missing-net ablation, not full-modal paper evaluation',
              'Ground-edge midpoints projected through homography; no upper net endpoints fabricated',
              'Professional shuttle and court are manual; poses are human-associated HRNet, not our automatic frontend'],
              'max_rallies_per_split':max_rallies,'domain':domain}
    write_json(out/'manifest.json',manifest);return manifest

def read_rgb_window(video,start,frames,*,fps=25,size=160,sample_stride=1):
    import cv2
    cap=cv2.VideoCapture(str(video));native=cap.get(cv2.CAP_PROP_FPS);total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if native<=0 or total<=0:raise ValueError('Cannot decode source video')
    requested=[round((start+i*sample_stride/fps)*native) for i in range(frames)]
    if min(requested)<0 or max(requested)>=total:cap.release();raise ValueError('Requested RGB window exceeds source; no silent boundary padding')
    decoded=[];wanted=set(requested);images={}
    try:
        # One seek then sequential decode avoids repeated keyframe decoding on
        # long paper-sized windows, and gives deterministic original indices.
        cap.set(cv2.CAP_PROP_POS_FRAMES,requested[0])
        for index in range(requested[0],requested[-1]+1):
            ok,bgr=cap.read()
            if not ok:raise ValueError(f'Failed decoding original frame{index}')
            if index in wanted:images[index]=cv2.resize(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),(size,size),interpolation=cv2.INTER_LINEAR)
        decoded=[images[index] for index in requested]
    finally:cap.release()
    rgb=torch.from_numpy(np.stack(decoded)).permute(3,0,1,2).float()/255.
    mean=torch.tensor([.485,.456,.406])[:,None,None,None];std=torch.tensor([.229,.224,.225])[:,None,None,None]
    return (rgb-mean)/std,{'native_fps':native,'original_frame_indices':requested,'start_seconds':start,
                           'sample_fps':fps,'sample_stride':sample_stride}

def prepare_local_dsta(out,per_split=4):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    jobs=['J_fe1e058fc75e475d9c030caced6ec648','J_e23c1dd9297d4bfca7c91d1d95065b68'];sources=[]
    for job in jobs:
        ann=PROJECT/'03_数据/击球标注'/job/'annotations.json';data=json.loads(ann.read_text())
        rows=sorted([r for r in data['labels'].values() if r.get('contact_confirmed') and not r.get('wrong_target') and r.get('stroke_type') not in ['unknown','no_stroke','between_rallies',None]],key=lambda r:r['peak_time'])
        metadata=json.loads((PROJECT/'03_数据/分析任务'/job/'job.json').read_text())
        video=Path(metadata['video_path']) if 'video_path' in metadata else None
        if not video:
            # Match source identity rather than joining a video to labels by filename.
            candidates=PROJECT.joinpath('03_数据/个人录像').glob('*.mp4')
            video=next((p for p in candidates if sha256(p)==data['source_sha256']),None)
        if not video or sha256(video)!=data['source_sha256']:raise ValueError('Annotated video SHA mismatch')
        sources.append((video,ann,data,rows))
    labels=sorted(set(r['stroke_type'] for r in sources[0][3])&set(r['stroke_type'] for r in sources[1][3]))
    cases=[];config={'classes':len(labels),'window_frames':32,'feature_length':64,'head_max_seq':64,'head_window':-1,'head_dim':128,'drop_path':0.,'sample_stride':1,'attention_clip_frames':16}
    for split,(video,ann,data,rows) in zip(['train','val'],sources):
        selected=[]
        for label in labels:
            selected+=next(([r] for r in rows if r['stroke_type']==label),[])
        selected=selected[:per_split]
        for row in selected:
            start=float(row['peak_time'])-16/25
            if start<0:continue
            rgb,clock=read_rgb_window(video,start,32)
            # The source supplies a verified contact, not verified action boundaries.
            # Explicit weak19-frame annotation [h−9,h+10) on25fps timeline.
            lo,hi=7*2,26*2
            path=out/f'{split}_{row["event_id"]}.npz';np.savez_compressed(path,rgb=rgb.numpy(),segments=np.array([[lo,hi]],np.float32),labels=np.array([labels.index(row['stroke_type'])]))
            cases.append({'id':path.stem,'path':str(path.resolve()),'split':split,'video_path':str(video.resolve()),
                          'source_sha256':data['source_sha256'],'annotation_path':str(ann),'annotation_sha256':sha256(ann),
                          'event_id':row['event_id'],'stroke_type':row['stroke_type'],'clock':clock,
                          'weak_interval_25fps_inclusive':[7,25],'unlabelled_other_strokes_possible':True})
    result={'component':'dsta','scope':'real_rgb_training_smoke','deployment_ready':False,'config_overrides':config,'labels':labels,'rows':cases,
            'split_protocol':'gray video train; blue video val, separate source hashes; no accuracy/AP claim from partially annotated windows',
            'limitations':['32 RGB frames instead of paper768; local class vocabulary instead of FineBadminton29',
            '19-frame contact-derived pseudo intervals are weak annotations, not manually verified action boundaries',
            'Unlabelled actions may exist in these short windows; this is only a gradient/save/load check']}
    write_json(out/'manifest.json',result);return result
