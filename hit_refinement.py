"""Audio/visual/GRU hit suggestions. Never creates human contact truth."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
import numpy as np
from scipy.signal import butter,sosfilt,find_peaks
from analysis_jobs import DATA,atomic_json,file_hash

ROOT=Path(__file__).resolve().parent
VERSION='audio-visual-contact-rerank-v2'
SUPPLEMENTAL_ID_VERSION='audio-visual-gru-suggestions-v1'
READ_CACHE={}

def timing_model():
    """Optional validated local scorer; absence keeps the original method."""
    folder=ROOT/'models/contact_timing'
    path=folder/'contact_ridge.npz';manifest=folder/'manifest.json'
    if not path.is_file() or not manifest.is_file():return None,None
    metadata=json.loads(manifest.read_text())
    if metadata.get('runtime_enabled') is not True:return None,metadata
    if file_hash(path)!=metadata['model_sha256']:
        raise ValueError('Contact timing scorer differs from its manifest')
    from contact_timing import VERSION as feature_version,load_model
    if metadata['feature_version']!=feature_version or metadata['feature_sha256']!=file_hash(ROOT/'contact_timing.py'):
        raise ValueError('Contact timing feature version differs from its manifest')
    return load_model(path),metadata

def fingerprint(job):
    path=DATA/'专项分析'/job/'hit_refinement.json'
    if not path.is_file():return None
    stat=path.stat();return f'{stat.st_size}:{stat.st_mtime_ns}'

def read(job,source_sha):
    path=DATA/'专项分析'/job/'hit_refinement.json'
    if not path.is_file():return None
    stat=path.stat();key=(str(path),stat.st_mtime_ns,stat.st_size)
    value=READ_CACHE.get(key)
    if value is None:
        value=json.loads(path.read_text())
        if len(READ_CACHE)>4:READ_CACHE.clear()
        READ_CACHE[key]=value
    if value.get('source_sha256')!=source_sha or value.get('version') not in [VERSION,'audio-visual-gru-suggestions-v1']:return None
    return value

def effective_event(dense,event,annotation=None):
    a=annotation or {}
    result=read(dense['id'],dense['source_sha256'])
    suggestion=result and result['suggestions'].get(event['id'])
    if (not suggestion or not suggestion['applied'] or a.get('contact_confirmed') or
            a.get('center_override') or a.get('window_basis') or
            a.get('stroke_type')=='no_stroke' or a.get('player',event['player'])!=suggestion['player']):
        return event,suggestion
    return {**event,'peak_time':suggestion['suggested_time'],'frame_index':suggestion['suggested_frame']},suggestion

def audio_onsets(config):
    command=[config['ffmpeg'],'-v','error','-i',config['source_file'],'-map','0:a:0',
             '-ac','1','-ar','16000','-af','aresample=async=1:first_pts=0','-f','f32le','pipe:1']
    proc=subprocess.run(command,capture_output=True,timeout=180)
    if proc.returncode:
        return [],{'status':'unavailable','reason':'No decodable audio track','synchronization':'not measured'}
    audio=np.frombuffer(proc.stdout,dtype='<f4')
    if len(audio)<320 or not np.isfinite(audio).all() or np.max(abs(audio))<1e-5:
        return [],{'status':'unavailable','reason':'Silent/invalid audio'}
    filtered=sosfilt(butter(3,[1200,7000],btype='bandpass',fs=16000,output='sos'),audio)
    hop,width=160,320
    windows=np.lib.stride_tricks.sliding_window_view(filtered,width)[::hop]
    energy=np.sqrt(np.mean(windows*windows,axis=1))+1e-9
    log=np.log(energy)
    baseline=np.asarray([np.median(log[max(0,i-15):i+1]) for i in range(len(log))])
    novelty=np.maximum(log-baseline,0)
    mad=max(.08,float(np.median(abs(novelty-np.median(novelty)))*1.4826))
    peaks,properties=find_peaks(novelty,height=max(.7,3*mad),prominence=.45,distance=12,width=(None,9))
    result=[]
    for peak,prominence in zip(peaks,properties['prominences']):
        # Start of an energy window, not the energy maximum, approximates onset.
        sample=int(peak*hop)
        result.append({'time':sample/16000,'sample_index':sample,
                       'strength':float(min(1,prominence/2)), 'log_prominence':float(prominence)})
    return result,{'status':'completed','sample_rate':16000,'sample_count':len(audio),
        'onset_count':len(result),'method':'1.2-7kHz band energy, 20ms window/10ms hop, local novelty and short peaks',
        'synchronization':'ffmpeg source presentation timeline, aresample first_pts=0; visual-derived lag estimated separately',
        'note':'Audio transients are candidates, including other courts and footsteps; not stroke labels.'}

def visual_scores(dense,trajectory):
    from stroke_features import HitFeatures
    filtered=HitFeatures(SimpleNamespace(dense=dense,trajectory=trajectory))
    n=len(dense['frames']);scores=np.zeros((n,2));details={}
    for i in range(5,n-5):
        if not filtered.valid[i]:continue
        before=np.flatnonzero(filtered.valid[max(0,i-5):i-1])+max(0,i-5)
        after=np.flatnonzero(filtered.valid[i+2:i+6])+i+2
        if not len(before) or not len(after):continue
        incoming=filtered.ball[i]-np.median(filtered.ball[before],axis=0)
        outgoing=np.median(filtered.ball[after],axis=0)-filtered.ball[i]
        denom=np.linalg.norm(incoming)*np.linalg.norm(outgoing)
        if denom<25:continue
        turn=float(np.clip((1-np.dot(incoming,outgoing)/denom)/2,0,1))
        for k,side in enumerate(['near','far']):
            p=filtered.pose(i,side)
            if p is None:continue
            pts,good,_,_,torso,_=p
            wrists=[j for j in [9,10] if good[j]]
            if not wrists:continue
            distance=min(float(np.linalg.norm(filtered.ball[i]-pts[j])/torso) for j in wrists)
            if distance>1.8:continue
            proximity=max(0,1-distance/1.8)
            motion=0
            previous=filtered.pose(i-3,side)
            if previous:
                pp,pg,_,_,_,_=previous
                motion=max([np.linalg.norm(pts[j]-pp[j])/torso for j in wrists if pg[j]] or [0])
            scores[i,k]=.45*proximity+.35*turn+.2*min(1,motion/.45)
            details[(i,k)]={'wrist_distance_torso_units':distance,'trajectory_turn':turn,
                            'wrist_motion_torso_units_per_3_frames':float(motion)}
    return scores,details,{'usable_moving_shuttle_frames':int(filtered.valid.sum()),
                           'stationary_shuttle_frames_excluded':int(filtered.static.sum())}

def prepare_inputs(config,dense,trajectory,events):
    started=time.perf_counter();folder=DATA/'专项分析'/config['pose_id']
    fingerprint={name:file_hash(folder/name) for name in ['dense_pose.json','trajectory.json']}
    fingerprint.update(code=file_hash(Path(__file__)),gru_code=file_hash(ROOT/'hit_gru.py'),
                       weights=file_hash(ROOT/'models/hit_gru/keras_tensors.npz'),
                       filter_code=file_hash(ROOT/'stroke_features.py'),
                       contact_code=file_hash(ROOT/'contact_timing.py'))
    timing, timing_manifest=timing_model()
    fingerprint['timing_model']=timing_manifest['model_sha256'] if timing is not None else None
    fingerprint['timing_manifest']=file_hash(ROOT/'models/contact_timing/manifest.json') if timing is not None else None
    fingerprint['event_centers']=hashlib.sha256(json.dumps(
        [{'id':e['id'],'player':e['player'],'peak_time':e['peak_time']} for e in events],
        sort_keys=True).encode()).hexdigest()
    cached=read(config['pose_id'],config['source_sha256'])
    ids=[e['id'] for e in events]
    if cached and cached['fingerprint']==fingerprint and cached['original_event_ids']==ids:return cached
    print('REFINEMENT_STAGE audio',flush=True)
    onsets,audio=audio_onsets(config)
    print('REFINEMENT_STAGE visual',flush=True)
    visual,detail,quality=visual_scores(dense,trajectory)
    print('REFINEMENT_STAGE published_gru',flush=True)
    from hit_gru import infer
    supports,coverage,gru=infer(dense,trajectory)
    times=np.asarray([f['source_time'] for f in dense['frames']]);n=len(times)
    camera_order=gru['camera_to_candidate_order']
    model=np.stack([supports[:,1+camera_order.index(side)] for side in ['near','far']],axis=1)
    onset_times=np.asarray([a['time'] for a in onsets]);audio_score=np.zeros(n);audio_matches={}
    strong,_=find_peaks(visual.max(1),height=.6,distance=8)
    offsets=[]
    for i in strong:
        if len(onset_times):
            distances=onset_times-times[i];possible=np.flatnonzero((distances>=-.08)&(distances<=.2))
            if len(possible):offsets.append(float(distances[possible[np.argmin(abs(distances[possible]))]]))
    lag=float(np.median(offsets)) if len(offsets)>=6 else 0.
    dispersion=float(np.median(abs(np.asarray(offsets)-lag))) if offsets else None
    if dispersion is not None and dispersion>.07:lag=0.
    audio.update(estimated_lag_seconds=lag,lag_pair_count=len(offsets),lag_median_absolute_deviation=dispersion,
                 lag_confirmed_by_user=False)
    for onset in onsets:
        i=int(np.argmin(abs(times-(onset['time']-lag))))
        for j in range(max(0,i-3),min(n,i+4)):
            score=onset['strength']*np.exp(-abs(times[j]-(onset['time']-lag))/.06)
            if score>audio_score[j]:audio_score[j]=score;audio_matches[j]=onset
    combined=.45*visual+.35*model+.2*audio_score[:,None]
    suggestions={}
    for e in events:
        original=int(np.argmin(abs(times-e['peak_time'])));k=['near','far'].index(e['player'])
        candidates=np.flatnonzero(abs(times-times[original])<=.3)
        best=int(candidates[np.argmax(combined[candidates,k])])
        # Conservative independent evidence gates. Every accepted correction
        # is still a proposed center and must be reviewed before training.
        accepted=(visual[best,k]>=.42 and model[best,k]>=.4 and audio_score[best]>=.45 and
                  model[best,k]>=model[best,1-k]-.1 and combined[best,k]>=combined[original,k]+.04)
        chosen=best if accepted else original
        suggestions[e['id']]={'event_id':e['id'],'player':e['player'],
            'original_frame':original,'original_time':float(times[original]),
            'suggested_frame':chosen,'suggested_time':float(times[chosen]),
            'alternative_frame':best,'alternative_time':float(times[best]),
            'applied':bool(accepted and chosen!=original),
            'visual_support':float(visual[chosen,k]),'gru_window_support':float(model[chosen,k]),
            'other_player_gru_support':float(model[chosen,1-k]),'audio_support':float(audio_score[chosen]),
            'audio_onset':audio_matches.get(chosen),'visual_evidence':detail.get((chosen,k)),
            'center_source':'audio_visual_gru_candidate' if accepted else 'original_visual_candidate',
            'score_is_correctness_probability':False,'contact_confirmed':False}
    # A residual refinement layer starts from the accepted v1 suggestion. It
    # must never discard a good v1 correction by falling back to the raw peak.
    # The runtime scorer consumes video evidence only, never human annotations.
    timing_summary={'enabled':timing is not None,'fine_adjustments':0,'uncertain_candidates':0,
                    'scope':'Candidate contact suggestions; not verified touch frames.'}
    if timing is not None:
        from contact_timing import frame_features,choose_frame
        feature_rows=frame_features(dense,trajectory,{'visual':visual,'gru':model,'audio':audio_score})
        eligible=config['source_sha256'] in timing_manifest['eligible_source_sha256']
        timing_summary.update(auto_apply_eligible=eligible,model_sha256=timing_manifest['model_sha256'])
        for e in events:
            suggestion=suggestions[e['id']]
            anchor={**e,'peak_time':suggestion['suggested_time']}
            fine=choose_frame(times,feature_rows,anchor,timing,**timing_manifest['policy'])
            previous=suggestion['suggested_frame'];best=fine['alternative_frame']
            accepted=bool(fine['applied'] and eligible)
            chosen=fine['suggested_frame'] if accepted else previous
            k=['near','far'].index(e['player'])
            uncertain=bool(fine['boundary'] or (fine['applied'] and not eligible) or
                           (not accepted and abs(times[best]-times[previous])>.15))
            timing_summary['fine_adjustments']+=int(accepted)
            timing_summary['uncertain_candidates']+=int(uncertain)
            suggestion.update(baseline_frame=previous,baseline_time=float(times[previous]),
                suggested_frame=chosen,suggested_time=float(times[chosen]),
                alternative_frame=best,alternative_time=float(times[best]),
                applied=bool(chosen!=suggestion['original_frame']),
                visual_support=float(visual[chosen,k]),gru_window_support=float(model[chosen,k]),
                other_player_gru_support=float(model[chosen,1-k]),audio_support=float(audio_score[chosen]),
                audio_onset=audio_matches.get(chosen),visual_evidence=detail.get((chosen,k)),
                center_source=VERSION if accepted else suggestion['center_source'],
                contact_rerank={**fine,'auto_apply_eligible':eligible,'fine_adjustment_applied':accepted},
                needs_precision_review=uncertain,
                review_reason='多帧证据的位置与当前中心有分歧，建议逐帧核对。' if uncertain else None)
    supplemental=[]
    peaks,_=find_peaks(combined.max(1),height=.64,distance=12)
    for i in sorted(peaks,key=lambda i:combined[i].max(),reverse=True):
        k=int(np.argmax(combined[i]));side=['near','far'][k]
        if visual[i,k]<.55 or model[i,k]<.5 or audio_score[i]<.65:continue
        if any(abs(times[i]-e['peak_time'])<.4 for e in events+supplemental):continue
        eid=hashlib.md5(f'{SUPPLEMENTAL_ID_VERSION}:{dense["source_sha256"]}:{side}:{i}'.encode()).hexdigest()
        lo=max(0,i-8);hi=min(n-1,i+12)
        supplemental.append({'id':eid,'player':side,'peak_time':float(times[i]),'frame_index':int(i),
            'start_frame':int(lo),'end_frame':int(hi),'start':float(times[lo]),'end':min(dense['end'],float(times[hi]+1/30)),
            'stroke_type':'unknown','stroke_label':'待判断球种','swing_style':'unknown',
            'stroke_certainty':'uncertain','style_certainty':'uncertain','quality':'unclear','visible_stroke':False,
            'classification_source':VERSION,'classification_status':'model_candidate',
            'from_refinement':True,'kind':'possible_missing_hit','review_status':'unreviewed',
            'hit_evidence':{'visual':float(visual[i,k]),'gru_window':float(model[i,k]),'audio':float(audio_score[i])},
            'interpretation':'Possible omitted hit; neither contact nor stroke class has been confirmed.'})
        if len(supplemental)>=12:break
    # A proposal refresh must keep existing review identities. A newly found
    # nearby proposal does not invalidate a clip already offered to the user.
    previous = read(dense['id'], config['source_sha256'])
    current_ids = {e['id'] for e in events + supplemental}
    retained = []
    for candidate in (previous or {}).get('supplemental_candidates', []):
        if candidate['id'] not in current_ids:
            retained.append({**candidate, 'retained_from_previous_proposals': True})
            current_ids.add(candidate['id'])
    supplemental.extend(retained)
    result={'version':VERSION,'source_sha256':config['source_sha256'],'fingerprint':fingerprint,
        'original_event_ids':ids,'suggestions':suggestions,'supplemental_candidates':supplemental,
        'audio':audio,'visual_quality':quality,'gru':{k:v for k,v in gru.items() if k!='windows'},
        'summary':{'original_candidates':len(events),'adjusted_candidates':sum(s['applied'] for s in suggestions.values()),
                   'possible_missing_candidates':len(supplemental),'human_contacts_added':0,
                   'retained_previous_candidate_ids':[e['id'] for e in retained],
                   'fine_adjustments':timing_summary['fine_adjustments'],
                   'uncertain_candidates':timing_summary['uncertain_candidates']},
        'contact_timing':timing_summary,
        'created_at':datetime.now(timezone.utc).isoformat(),'runtime_seconds':time.perf_counter()-started,
        'evaluation':'Suggestion counts are not accuracy or recall. Local scorer development evaluation is recorded separately; human-confirmed centers remain authoritative.'}
    atomic_json(folder/'hit_refinement.json',result)
    atomic_json(folder/'audio_onsets.json',{'source_sha256':config['source_sha256'],'metadata':audio,'onsets':onsets})
    np.savez_compressed(folder/'hit_detector_support.npz',frame_times=times,visual=visual,gru=model,audio=audio_score,
                        pose_coverage=coverage)
    return result

def prepare(job):
    from stroke_review import load_source
    _,events,_=load_source(job);folder=DATA/'专项分析'/job
    config=json.loads((folder/'config.json').read_text())
    dense=json.loads((folder/'dense_pose.json').read_text());trajectory=json.loads((folder/'trajectory.json').read_text())
    original=json.loads((folder/'hit_candidates.json').read_text())
    if original['source_sha256']!=config['source_sha256']:raise ValueError('Hit proposals source differs')
    # Classified events may already carry a previous refinement. Always start
    # from the saved raw proposals so refreshing cannot repeatedly shift them.
    by_id={e['id']:e for e in original['candidates']}
    ids=[e['id'] for e in events if not e.get('from_refinement')]
    if any(eid not in by_id for eid in ids):raise ValueError('Classified candidate missing from original proposals')
    # A classified analysis may intentionally contain only a subset of the
    # proposal file. Timing refresh must not expand that review queue.
    return prepare_inputs(config,dense,trajectory,[by_id[eid] for eid in ids])

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True);args=parser.parse_args()
    result=prepare(args.job);print(json.dumps(result['summary'],ensure_ascii=False),flush=True)
