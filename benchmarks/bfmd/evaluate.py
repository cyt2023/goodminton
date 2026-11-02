"""Independent BFMD scorer. Imports references only here, never into inference."""
import argparse,json,math,statistics,hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from analysis_jobs import file_hash
from .importer import load,save,LABEL_MAP,manual_timeline


def unavailable(reason):return {'status':'unavailable','value':None,'reason':reason}
def prf(tp,fp,fn):
    return {'status':'evaluated','tp':tp,'fp':fp,'fn':fn,'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn) if tp+fn else 0.,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}
def assign(pred,truth,tolerance,labels=False):
    if not pred or not truth:return []
    delta=np.abs(np.array([r['time'] for r in pred])[:,None]-np.array([r['time'] for r in truth])[None,:])
    valid=delta<=tolerance+1e-6  # frame-quantized times: exactly one tolerance apart still matches
    if labels:valid&=np.array([[a.get('label')==b.get('label') and a.get('label') is not None for b in truth] for a in pred])
    a,b=linear_sum_assignment(np.where(valid,delta,1e6));return [(int(i),int(j)) for i,j in zip(a,b) if valid[i,j]]
def average_precision(pred,truth,threshold):
    if not truth:return None
    used=set();tp=[];fp=[]
    for row in sorted(pred,key=lambda r:-r['confidence']):
        candidates=[]
        for j,gt in enumerate(truth):
            if j in used:continue
            intersection=max(0,min(row['end'],gt['end'])-max(row['start'],gt['start']));union=max(row['end'],gt['end'])-min(row['start'],gt['start'])
            candidates.append((intersection/union if union>0 else 0.,j))
        best=max(candidates,default=(0,None))
        hit=best[1] is not None and best[0]>=threshold
        if hit:used.add(best[1])
        tp.append(int(hit));fp.append(int(not hit))
    if not pred:return 0.
    recall=np.cumsum(tp)/len(truth);precision=np.cumsum(tp)/(np.cumsum(tp)+np.cumsum(fp));mrec=np.r_[0,recall,1];mpre=np.r_[0,precision,0]
    mpre=np.maximum.accumulate(mpre[::-1])[::-1];indices=np.where(mrec[1:]!=mrec[:-1])[0]
    return float(np.sum((mrec[indices+1]-mrec[indices])*mpre[indices+1]))

def canonical(packet):
    if packet.get('kind')!='model_inference':raise ValueError('Predictions must be actual model inference, not manual/reference annotations')
    provenance=packet.get('provenance',{})
    if not provenance.get('model') or not provenance.get('input_video_sha256') or provenance.get('annotation_inputs')!=[]:raise ValueError('Declare model, raw video hash, and empty annotation_inputs for independent inference')
    if packet.get('mode')!='end_to_end':raise ValueError('Oracle/reference-input experiments are not independent end-to-end benchmarks')
    clock=packet['time_base'];unit=clock['unit'];offset=float(clock.get('offset_seconds',0))
    if unit not in ['seconds','frames']:raise ValueError('Declare seconds or frames')
    if unit=='frames' and (clock.get('frame_index_origin') not in [0,1] or clock.get('fps',0)<=0):raise ValueError('Frame clock requires positive fps and explicit0/1 origin')
    def seconds(v):
        if not isinstance(v,(int,float)) or not math.isfinite(v):raise ValueError('Nonfinite timestamp')
        t=(v-clock['frame_index_origin'])/clock['fps'] if unit=='frames' else v
        return t+offset
    result={**packet};result['hits']=None
    if 'hits' in packet:
        result['hits']=[{**r,'time':seconds(r.get('frame') if unit=='frames' else r.get('time'))} for r in packet['hits']]
        space=packet.get('label_space','canonical_mainland')
        if space not in ['canonical_mainland','mainland-v3','bfmd_raw']:raise ValueError('Declare a supported label space')
        coarse={'serve_short':'serve','clear_flat':'clear','net_spin':'net'}
        for row in result['hits']:
            label=row.get('label')
            row['raw_prediction_label']=label
            row['label']=LABEL_MAP.get(label) if space=='bfmd_raw' else coarse.get(label,label)
            if row['label']=='unknown':row['label']=None
    for key in ['rallies','stroke_intervals']:
        if key in packet:
            result[key]=[{**r,'start':seconds(r['start']),'end':seconds(r['end'])} for r in packet[key]]
            if any(r['start']>=r['end'] or not 0<=r['confidence']<=1 for r in result[key]):raise ValueError('Invalid predicted interval/confidence')
    result['evaluation_window']=[seconds(v) for v in packet['evaluation_window']]
    if not 0<=result['evaluation_window'][0]<result['evaluation_window'][1]:raise ValueError('Invalid evaluation coverage')
    for key in ['trajectory','poses']:
        if key in packet:result[key]=[{**r,'time':seconds(r.get('frame') if unit=='frames' else r.get('time'))} for r in packet[key]]
    return result

def pixels(xy,coordinates):
    space=coordinates['space'];w,h=coordinates['width'],coordinates['height']
    if w<=0 or h<=0:raise ValueError('Coordinate image size required')
    scale={'native_pixels':(1280/w,720/h),'normalized_0_1':(1280,720),'percent_0_100':(12.8,7.2)}
    if space not in scale:raise ValueError('Unknown coordinate system')
    return [float(xy[k])*scale[space][k] for k in [0,1]]

def evaluate(match,packet,tolerance=.1):
    p=canonical(packet)
    if p['match_id']!=match['match_id']:raise ValueError('Prediction/reference match mismatch')
    lo,hi=p['evaluation_window'];truth=[r for r in match['hits'] if lo<=r['time']<hi];metrics={}
    if p['hits'] is None:metrics['hit_detection']=unavailable('Hit module predictions not provided');metrics['timestamp_error']=unavailable('Hit module predictions not provided');metrics['stroke_types']=unavailable('Stroke predictions not provided')
    else:
        pred=[r for r in p['hits'] if lo<=r['time']<hi];pairs=assign(pred,truth,tolerance)
        metrics['hit_detection']=prf(len(pairs),len(pred)-len(pairs),len(truth)-len(pairs));errors=[abs(pred[i]['time']-truth[j]['time']) for i,j in pairs]
        metrics['timestamp_error']={'status':'evaluated' if errors else 'unavailable','matched_events':len(errors),'mae_seconds':statistics.mean(errors) if errors else None,'median_seconds':statistics.median(errors) if errors else None,'p95_seconds':float(np.percentile(errors,95)) if errors else None}
        typed=[r for r in truth if r['label'] is not None];ignore={i for i,j in pairs if truth[j]['label'] is None};typed_pred=[r for i,r in enumerate(pred) if i not in ignore and r.get('label') is not None]
        if not typed:metrics['stroke_types']=unavailable('This coverage has no compatible manually labelled stroke types')
        else:
            good=assign(typed_pred,typed,tolerance,labels=True);confusion={}
            for i,j in pairs:
                if truth[j]['label'] is not None:
                    key=truth[j]['label']+'→'+str(pred[i].get('label','unknown'));confusion[key]=confusion.get(key,0)+1
            metrics['stroke_types']={**prf(len(good),len(typed_pred)-len(good),len(typed)-len(good)),'definition':'joint correct type AND localized hit; untyped reference contacts excluded; class abstentions are not typed predictions','untyped_reference_hits':len(truth)-len(typed),'abstained_hit_candidates':sum(r.get('label') is None for r in pred),'conditional_confusion':confusion}
    metrics['stroke_temporal_mAP']=unavailable('BFMD manual shot labels are contact POINTS, not verified full-action durations; no artificial intervals created')
    rallies=[r for r in match['rallies'] if r['start']>=lo and r['end']<=hi]
    metrics['rally_temporal_mAP']=({'status':'evaluated','IoU_thresholds':[.3,.5,.7],'AP':{str(t):average_precision(p['rallies'],rallies,t) for t in [.3,.5,.7]}} if 'rallies' in p and rallies else unavailable('No predicted rallies and fully covered reference rally intervals'))
    if metrics['rally_temporal_mAP']['status']=='evaluated':metrics['rally_temporal_mAP']['mAP']=statistics.mean(metrics['rally_temporal_mAP']['AP'].values())
    metrics['trajectory_ground_truth_error']=unavailable('Released shuttle trajectories are TrackNet predictions, not independent measured ground truth')
    metrics['pose_ground_truth_error']=unavailable('Released2D poses are model estimates; no independent measured2D/3D pose ground truth verified')
    metrics['pose3d_MPJPE']=unavailable('BFMD does not provide measured3D pose ground truth')
    metrics['trajectory_reference_agreement']=unavailable('Independent trajectory predictions or released reference file unavailable')
    if p.get('trajectory') is not None and 'shuttle' in match['references']:
        value=load(match['references']['shuttle']);tasks=value if isinstance(value,list) else [value];reference={}
        for task in tasks:
            for prediction in task.get('predictions',[]):
                for r in prediction.get('result',[]):
                    for v in r.get('value',{}).get('sequence',[]):
                        if v.get('enabled',True):reference[int(v['frame'])]=[(v['x']+v.get('width',0)/2)*12.8,(v['y']+v.get('height',0)/2)*7.2]
        errors=[]
        for row in p['trajectory']:
            if row.get('xy') is None or not lo<=row['time']<hi:continue
            frame=round(row['time']*30)
            if frame in reference:errors.append(math.dist(pixels(row['xy'],p['coordinates']),reference[frame]))
        reference_coverage=sum(lo<=i/30<hi for i in reference)
        metrics['trajectory_reference_agreement']={'status':'reference_agreement_only' if errors else 'unavailable','paired_frames':len(errors),'reference_frames_in_window':reference_coverage,'paired_reference_coverage':len(errors)/reference_coverage if reference_coverage else None,'mean_L2_px_1280x720':statistics.mean(errors) if errors else None,'RMSE_px':math.sqrt(statistics.mean(x*x for x in errors)) if errors else None,'ground_truth_accuracy':False}
    metrics['pose_reference_agreement']=unavailable('Independent pose predictions or released pose reference file unavailable')
    if p.get('poses') is not None and 'pose' in match['references']:
        data=load(match['references']['pose']);reference={}
        for rally in data.get('videos',{}).values():
            for index,persons in rally.get('frames',{}).items():reference[int(index)]=persons
        errors=[];correct=0;count=0
        for row in p['poses']:
            if not lo<=row['time']<hi:continue
            gt=reference.get(round(row['time']*30),[]);pred_people=row.get('persons',[])
            if not gt or not pred_people:continue
            # Independent per-frame association by mean visible joint distance.
            costs=np.full((len(pred_people),len(gt)),1e6)
            for i,person in enumerate(pred_people):
                for j,g in enumerate(gt):
                    ds=[math.dist(pixels(a[:2],p['coordinates']),[b[0]*1280,b[1]*720]) for a,b in zip(person['keypoints'],g['keypoints']) if a[2]>=.5 and b[2]>=.5]
                    if ds:costs[i,j]=statistics.mean(ds)
            a,b=linear_sum_assignment(costs)
            for i,j in zip(a,b):
                if costs[i,j]>=1e6:continue
                box=gt[j]['bbox'];diag=math.hypot(box[2]-box[0],box[3]-box[1])
                for x,y in zip(pred_people[i]['keypoints'],gt[j]['keypoints']):
                    if x[2]<.5 or y[2]<.5:continue
                    error=math.dist(pixels(x[:2],p['coordinates']),[y[0]*1280,y[1]*720]);errors.append(error);count+=1;correct+=error<=.05*diag
        metrics['pose_reference_agreement']={'status':'reference_agreement_only' if count else 'unavailable','paired_joints':count,'PCK_0.05_bbox_diagonal':correct/count if count else None,'mean_L2_px_1280x720':statistics.mean(errors) if errors else None,'ground_truth_accuracy':False}
    return {'match_id':match['match_id'],'scope':'declared clip coverage; no full-dataset performance claim','evaluation_window_full_video_seconds':[lo,hi],'tolerance_seconds':tolerance,
        'frame_alignment':p.get('alignment_status','not_verified'),'metrics':metrics,'provenance':p['provenance'],'uses_reference_as_prediction':False}

def run(manifest_path,prediction_path,out):
    m=load(manifest_path);packet=load(prediction_path);match=next(r for r in m['matches'] if r['match_id']==packet['match_id'])
    if Path(prediction_path).resolve().is_relative_to(Path(m['raw_root']).resolve()):raise ValueError('Prediction artifact cannot be an annotation file')
    report=evaluate(match,packet);report.update(dataset_manifest_sha256=file_hash(Path(manifest_path)),prediction_sha256=file_hash(Path(prediction_path)),code_sha256=file_hash(Path(__file__)))
    save(Path(out)/'report.json',report)
    for module,metric_names in {'hit_timing':['hit_detection','timestamp_error'],'stroke_types':['stroke_types','stroke_temporal_mAP'],
        'shuttle_trajectory':['trajectory_ground_truth_error','trajectory_reference_agreement'],'human_pose':['pose_ground_truth_error','pose_reference_agreement','pose3d_MPJPE'],'rally_segmentation':['rally_temporal_mAP']}.items():
        save(Path(out)/'per_module'/(module+'.json'),{'module':module,'match_id':report['match_id'],'evaluation_window':report['evaluation_window_full_video_seconds'],
            'metrics':{k:report['metrics'][k] for k in metric_names},'dataset_manifest_sha256':report['dataset_manifest_sha256'],'prediction_sha256':report['prediction_sha256']})
    save(Path(out)/'end_to_end.json',{'status':'partial_experimental_pipeline','match_id':report['match_id'],
        'joint_localized_hit_and_correct_type':report['metrics']['stroke_types'],'hit_timing':report['metrics']['hit_detection'],
        'unsupported_ground_truth_metrics':[k for k,v in report['metrics'].items() if v['status']=='unavailable'],
        'stage_provenance':report['provenance'],'scope':report['scope'],'frame_alignment':report['frame_alignment'],
        'dataset_manifest_sha256':report['dataset_manifest_sha256'],'prediction_sha256':report['prediction_sha256']})
    lines=['# BFMD independent benchmark run','',f"Match: {report['match_id']}",'','Metrics apply only to the declared video coverage. Model-generated reference agreement is not ground-truth accuracy.','']
    for name,v in report['metrics'].items():lines.append(f"- **{name}**: "+json.dumps(v,ensure_ascii=False))
    Path(out,'report.md').write_text('\n'.join(lines)+'\n');return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--predictions',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.manifest,a.predictions,a.out)
