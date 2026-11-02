"""Official BFMD Label Studio import. References remain separate from predictions."""
import argparse,csv,json,hashlib,collections
from pathlib import Path
from analysis_jobs import file_hash
ROOT=Path(__file__).resolve().parents[2]
STORE=ROOT.parent/'03_数据/独立基准/BFMD'
LABEL_MAP={'serve':'serve','flick_serve':'serve_long','clear':'clear','drop':'drop','smash':'smash','drive':'drive','net_shot':'net','net_kill':'net_kill','lift':'lift','push':'push','block':'defense'}
EXCLUDED={'bounce','net_hit','replay','hawk_eye_challenge','11-point interval','21-point interval','11/21-point interval'}


def load(path):return json.loads(Path(path).read_text())
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def tasks(value):return value if isinstance(value,list) else [value]

def manual_timeline(value,match_id):
    contacts={};rallies=[];unknown=[];conflicts=[]
    for task in tasks(value):
        for annotation in task.get('annotations',[]):
            if annotation.get('was_cancelled'):continue
            for item in annotation.get('result',[]):
                if item.get('origin','manual')!='manual':continue
                v=item.get('value',{})
                for label in v.get('timelinelabels',[]):
                    for interval in v.get('ranges',[]):
                        lo,hi=float(interval['start']),float(interval['end'])
                        if not 0<=lo<=hi:raise ValueError('Invalid BFMD frame interval')
                        if label=='Rally':
                            rallies.append({'id':f'{match_id}:rally:{lo:g}','start':lo/30,'end':(hi+1)/30,'source':'manual_timeline'});continue
                        if label in EXCLUDED:continue
                        if label not in LABEL_MAP and label not in ['hit','press']:unknown.append(label);continue
                        # Manual point hit and typed label share the SAME frame:
                        # merge them rather than double-counting each stroke.
                        if lo!=hi:
                            conflicts.append({'label':label,'start':lo,'end':hi,'reason':'nonpoint_hit_range; not asserted as a precise timestamp'});continue
                        key=int(lo);row=contacts.setdefault(key,{'id':f'{match_id}:hit:{key}','frame':key,'time':key/30,'label':None,'raw_labels':[],'reference_kind':'manual_point_annotation'})
                        row['raw_labels'].append(label)
                        mapped=LABEL_MAP.get(label)
                        if mapped and row['label'] and mapped!=row['label']:row['label']=None;row['label_conflict']=True
                        elif mapped and not row.get('label_conflict'):row['label']=mapped
    return {'hits':sorted(contacts.values(),key=lambda x:x['frame']),'rallies':sorted({(r['start'],r['end']):r for r in rallies}.values(),key=lambda r:r['start']),
            'unmapped_labels':sorted(set(unknown)),'ambiguous_ranges':conflicts}


def inventory(raw,out):
    raw=Path(raw);index=list(csv.DictReader((raw/'Badminton_video_list.csv').read_text().splitlines()));matches=[];assets=[]
    for row in index:
        match=row['match_name'];kind=row['type'];annotation=raw/'annotations'/('shot_type' if kind=='singles' else 'hits_doubles')/(match+'.json')
        if not annotation.exists():matches.append({'match_id':match,'type':kind,'status':'annotation_unavailable'});continue
        data=manual_timeline(load(annotation),match)
        paths={name:str((raw/'annotations'/name/(match+('_court' if name=='court' else '')+'.json')).resolve()) for name in ['metadata','court','pose','shuttle','court_perframe','hit_inferred']}
        paths={k:v for k,v in paths.items() if Path(v).exists()}
        matches.append({'match_id':match,'type':kind,'status':'imported','youtube_url':row['youtube_url'],'youtube_id':row['youtube_url'].split('v=')[-1],
            'reference_video_spec':{'width':1280,'height':720,'fps':30.,'frame_index_origin':0},'source_annotation':str(annotation.resolve()),'source_sha256':file_hash(annotation),
            'references':paths,**data,'capabilities':{'manual_hit_timestamps':True,'manual_stroke_labels':kind=='singles' and any(h['label'] for h in data['hits']),
                'manual_stroke_action_intervals':False,'manual_rally_intervals':True,'independent_trajectory_ground_truth':False,'independent_pose_ground_truth':False,'pose_3d_ground_truth':False}})
    for path in (raw/'annotations').rglob('*.json'):assets.append({'path':str(path.resolve()),'sha256':file_hash(path),'bytes':path.stat().st_size})
    counts={kind:{'matches':sum(r['type']==kind and r['status']=='imported' for r in matches),'rallies':sum(len(r.get('rallies',[])) for r in matches if r['type']==kind),
                 'manual_point_hits':sum(len(r.get('hits',[])) for r in matches if r['type']==kind),'mapped_stroke_types':sum(sum(bool(h['label']) for h in r.get('hits',[])) for r in matches if r['type']==kind)} for kind in ['singles','doubles']}
    value={'dataset':'BFMD','role':'independent_benchmark_only','benchmark_only':True,'official_source':'https://github.com/Ning-D/BFMD','raw_root':str(raw.resolve()),
           'actual_downloaded_counts':counts,'actual_annotation_files':len(assets),'actual_annotation_bytes':sum(r['bytes'] for r in assets),
           'paper_claim':{'matches':19,'rallies':1687,'hit_events':16751},'release_readme_claim':{'matches':19,'rallies':1058,'shots':11301},
           'matches':matches,'assets':assets,'label_map':LABEL_MAP,'label_notes':['press has no verified exact mainland equivalent; counted as contact, type unscored','net_hit/bounce are not racket contacts','flick_serve maps only to coarse serve_long; no fine serve-style accuracy claim'],
           'reference_quality':{'shuttle':'TrackNet-generated reference, not independent ground truth','pose':'estimated COCO17 reference, not motion-capture/manual3D ground truth','stroke_intervals':'manual labels are point events; no fabricated full-action temporal spans'},
           'heldout_video_files':[{'path':str(p.resolve()),'sha256':file_hash(p),'bytes':p.stat().st_size} for p in (STORE/'videos').glob('*.mp4')],
           'training_policy':'No BFMD annotations, pseudo-labels, frames, or matches may enter local training/calibration. Match and source IDs sealed before inference; upstream pretraining overlap remains unknown.'}
    save(out,value);print(json.dumps({'actual_counts':counts,'files':len(assets),'bytes':value['actual_annotation_bytes']},indent=2))
    return value

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',default=str(STORE/'raw/BFMD_data'));p.add_argument('--out',default=str(STORE/'manifest.json'));a=p.parse_args();inventory(a.raw,a.out)
