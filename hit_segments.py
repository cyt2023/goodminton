"""Source-frame windows for hit review; candidates are never contact truth."""
import json
import hashlib
from pathlib import Path
import numpy as np
import threading

from analysis_jobs import DATA, atomic_json, file_hash

VERSION = 'hit-frame-window-v3-source-pts'
PRE_FRAMES, POST_FRAMES = 8, 12
CACHE = {}
DENSE_CACHE = {}
DENSE_HASHES = {}
MEDIA_LOCKS = {}


def dense_data(job):
    path=DATA/'专项分析'/job/'dense_pose.json'
    stat=path.stat();key=(str(path),stat.st_size,stat.st_mtime_ns)
    cached=DENSE_CACHE.get(key)
    if cached is not None:return cached
    value=json.loads(path.read_text())
    if len(DENSE_CACHE)>2:DENSE_CACHE.clear()
    DENSE_CACHE[key]=value
    return value


def dense_fingerprint(job):
    path=DATA/'专项分析'/job/'dense_pose.json'
    stat=path.stat();key=(str(path),stat.st_size,stat.st_mtime_ns)
    if key not in DENSE_HASHES:
        if len(DENSE_HASHES)>3:DENSE_HASHES.clear()
        DENSE_HASHES[key]=file_hash(path)
    return DENSE_HASHES[key]


def segment(dense, event, annotation=None):
    annotation = annotation or {}
    from hit_refinement import effective_event
    event,suggestion=effective_event(dense,event,annotation)
    times = np.asarray([f['source_time'] for f in dense['frames']], float)
    peak = event['peak_time'] if suggestion and suggestion['applied'] and not annotation.get('contact_confirmed') and not annotation.get('center_override') and not annotation.get('window_basis') and annotation.get('stroke_type')!='no_stroke' and annotation.get('player',event['player'])==suggestion['player'] else annotation.get('peak_time', event['peak_time'])
    center = int(np.argmin(abs(times-peak)))
    lo, hi = max(0,center-PRE_FRAMES), min(len(times)-1,center+POST_FRAMES)
    dt = float(np.median(np.diff(times))) if len(times)>1 else 1/dense['sample_fps']
    return {'event_id':event['id'],'player':annotation.get('player',event['player']),
            'version':VERSION,'center_frame':center,'center_time':float(times[center]),
            'start_frame':lo,'end_frame':hi,'start':float(times[lo]),
            'end':min(dense['end'],float(times[hi]+dt)),
            'last_frame_time':float(times[hi]),'frame_count':hi-lo+1,
            'pre_frames':center-lo,'post_frames':hi-center,'fps':1/dt,
            'contact_confirmed':bool(annotation.get('contact_confirmed')),
            'contact_status':'human_confirmed' if annotation.get('contact_confirmed') else 'needs_contact_review',
            'review_label':annotation.get('stroke_type'),
            'center_suggestion':suggestion,
            'note':'Exact source-frame window around a proposed contact; stroke label alone does not verify contact.',
            'preview_frames':sorted(set([lo,max(lo,center-4),center,min(hi,center+6),hi]))}


def metadata(job, events, doc):
    path=DATA/'专项分析'/job/'dense_pose.json'
    stat=path.stat();refinement_path=path.parent/'hit_refinement.json'
    refinement_mtime=refinement_path.stat().st_mtime_ns if refinement_path.is_file() else 0
    key=(job,stat.st_size,stat.st_mtime_ns,doc['revision'],refinement_mtime)
    cached=CACHE.get(key)
    if cached is not None:return cached
    dense=dense_data(job)
    if dense['source_sha256'] != doc['source_sha256']:raise ValueError('逐帧结果与标注来源不一致')
    result={'version':VERSION,'source_sha256':doc['source_sha256'],'dataset_revision':doc['revision'],
            'pre_frames':PRE_FRAMES,'post_frames':POST_FRAMES,
            'frame_times':[f['source_time'] for f in dense['frames']],
            'segments':{e['id']:segment(dense,e,doc['labels'].get(e['id'])) for e in events}}
    CACHE.clear();CACHE[key]=result
    return result


def prepare(job):
    from stroke_review import load_source,read_labels,folder_for
    report,events,_=load_source(job);doc=read_labels(job,report['source_sha256'])
    result=metadata(job,events,doc)
    atomic_json(folder_for(job)/'hit_segments.json',result)
    return result


def media(job,event_id,name,center=None,player=None):
    lock=MEDIA_LOCKS.setdefault((job,event_id),threading.Lock())
    with lock:
        return _media(job,event_id,name,center,player)


def _media(job,event_id,name,center=None,player=None):
    import cv2
    import source_frame_media
    from stroke_review import load_source,read_labels,folder_for
    report,events,_=load_source(job)
    event=next((e for e in events if e['id']==event_id),None)
    if event is None:raise ValueError('击球候选不存在')
    doc=read_labels(job,report['source_sha256']);info=metadata(job,events,doc)['segments'][event_id]
    dense=dense_data(job)
    if center is not None or player is not None:
        if name=='clip.mp4':raise ValueError('下载片段须先保存击球位置')
        if type(center) is not int or not 0<=center<len(dense['frames']) or player not in ['near','far']:
            raise ValueError('预览位置无效')
        time=dense['frames'][center]['source_time']
        if abs(time-event['peak_time'])>4:raise ValueError('预览超出当前动作范围')
        info=segment(dense,event,{'peak_time':time,'player':player,'center_override':True})
    dense_sha=dense_fingerprint(job)
    reader_sha=file_hash(Path(source_frame_media.__file__))
    code_sha=file_hash(Path(__file__))
    key=f"{info['center_frame']}_{info['player']}_{VERSION}_{report['source_sha256'][:12]}_{dense_sha[:12]}_{reader_sha[:12]}_{code_sha[:12]}"
    out=folder_for(job)/'击球小片段'/event_id/key
    out.mkdir(parents=True,exist_ok=True)
    if name!='clip.mp4' and name not in [f'frame_{i}.jpg' for i in info['preview_frames']]:
        raise ValueError('预览帧不存在')
    requested=out/name
    if requested.is_file() and (out/'provenance.json').is_file():return requested
    from analysis_jobs import registered_video
    source=registered_video(report['source_video_id'], allow_reference=True)['path']
    frames,proof=source_frame_media.read_source_window(source,dense,info['start_frame'],info['end_frame'])
    provenance={**info,**proof,'source_sha256':report['source_sha256'],'dense_pose_sha256':dense_sha,
                'code_sha256':code_sha,
                'source_frame_media_code_sha256':reader_sha,
                'source_frame_pixel_sha256':[hashlib.sha256(frame.tobytes()).hexdigest() for frame in frames]}
    for index in info['preview_frames']:
        frame=frames[index-info['start_frame']]
        person=dense['frames'][index]['players'].get(info['player'])
        if person:
            x1,y1,x2,y2=person['bbox'];h=y2-y1;w=x2-x1
            a,b=max(0,int(x1-w)),max(0,int(y1-h*.5))
            c,d=min(frame.shape[1],int(x2+w)),min(frame.shape[0],int(y2+h*.15))
            if c>a and d>b:frame=frame[b:d,a:c]
        ok,encoded=cv2.imencode('.jpg',frame)
        if not ok:raise ValueError('预览帧写入失败')
        partial=out/f'frame_{index}.partial.jpg';partial.write_bytes(encoded.tobytes())
        partial.replace(out/f'frame_{index}.jpg')
    if name=='clip.mp4':
        executable=source_frame_media.ffmpeg_executable()
        partial=out/'clip.partial.mp4'
        encoded_proof=source_frame_media.encode_pts_clip(executable,frames,proof,partial)
        partial.replace(out/'clip.mp4')
        provenance.update(**encoded_proof,codec='H264, verified full source frames and source PTS, audio omitted',
                          clip_sha256=file_hash(out/'clip.mp4'))
    atomic_json(out/'provenance.json',provenance)
    return requested


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--render',action='store_true')
    a=p.parse_args();result=prepare(a.job)
    if a.render:
        for n,eid in enumerate(result['segments']):
            media(a.job,eid,'clip.mp4');print('SEGMENT_DONE',n+1,len(result['segments']),flush=True)
