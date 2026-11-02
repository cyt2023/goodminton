"""Four stateful task executors around existing code, not four LLM personas."""
from dataclasses import asdict
from pathlib import Path
import math,shutil,subprocess,sys
from .state import ROOT,STORE,StrokeEvent,Evidence,sha,read,save,digest
from .retrieval import mainland,retrieve
DATA=ROOT.parent/'03_数据'

class Agent:
    def __init__(self,record,folder):self.record=record;self.folder=Path(folder);self.memory={'executions':0,'last_tool':None}
    def execute(self,tool,state,workflow):
        self.memory.update(executions=self.memory['executions']+1,last_tool=tool)
        return getattr(self,tool.split('.')[1])(state,workflow)
    def evidence(self,path,model,kind='model_prediction',note=''):return asdict(Evidence(str(path),sha(path),kind,model,note))

class PerceptionAgent(Agent):
    def cached(self,state,w):
        dense=read(self.record['dense_file']);trajectory=read(self.record['trajectory_file'])
        for value in [dense,trajectory]:
            if value['source_sha256']!=state.source_sha256 or len(value['frames'])!=len(state.timestamps):raise ValueError('Perception source/coverage mismatch')
            if any(f.get('frame_index')!=i or ('source_time' in f and abs(f['source_time']-state.timestamps[i])>1e-6) for i,f in enumerate(value['frames'])):raise ValueError('Perception frame clock mismatch')
        state.artifacts['pose2d']=self.evidence(self.record['dense_file'],'YOLOX/RTMPose')
        state.artifacts['trajectory']=self.evidence(self.record['trajectory_file'],trajectory.get('engine_version','existing trajectory model'))
        state.player_tracks={'artifact':state.artifacts['pose2d'],'identity_status':'near/far candidates, not verified person identity'};state.poses={'pose2d':state.artifacts['pose2d'],'joint_schema':'COCO17','coordinate_space':'native_pixels'};state.trajectories={'shuttle':state.artifacts['trajectory'],'coordinate_space':'native_pixels'}
        state.quality.update({'player_track_coverage':{side:sum(bool(f['players'].get(side)) for f in dense['frames'])/len(dense['frames']) for side in ['near','far']},'shuttle_observed_fraction':sum(bool(f['raw_visible']) for f in trajectory['frames'])/len(trajectory['frames']),'quality_scores_are_correctness_probabilities':False})
        path=DATA/'专项分析'/self.record['job_id']/'court_calibration.json'
        if path.exists():state.artifacts['court']=self.evidence(path,'reviewed/automatic court record','derived_observation')
        else:state.missing['court']='No verified court geometry; no metric court coordinates fabricated'
        state.court_geometry={'status':'available_record' if 'court' in state.artifacts else 'unavailable','artifact':state.artifacts.get('court')}
        state.missing['pose3d']='Optional default off;2D baseline retained'
        state.missing['racket']='Racket keypoints require RacketVision tool' if not state.artifacts.get('racket') else None
        state.provenance['perception']=dense.get('models',dense.get('engine_version'))
        return 'completed'
    def racketvision(self,state,w):
        self.cached(state,w)
        config=read(self.record['config_file']);config['racketvision_folder']=str(self.folder/'racketvision');config['source_file']=state.source_file
        path=self.folder/'rv_input.json';save(path,config)
        # Existing adapter handles MPS/CPU. No new checkpoint download.
        subprocess.run([str(ROOT/'.venv-vision/bin/python'),str(ROOT/'racketvision_adapter.py'),'--config',str(path)],cwd=ROOT,check=True,stdout=(self.folder/'racketvision.log').open('w'),stderr=subprocess.STDOUT,timeout=min(450,w.timeout_seconds))
        trajectory=self.folder/'racketvision/trajectory.json';racket=self.folder/'racketvision/racketvision.json'
        raw=read(trajectory)
        if raw['source_sha256']!=state.source_sha256 or len(raw['frames'])!=len(state.timestamps):raise ValueError('RacketVision source clock mismatch')
        state.artifacts['trajectory']=self.evidence(trajectory,'RacketVision BallTrack')
        state.artifacts['racket']=self.evidence(racket,'RacketVision RacketPose');state.trajectories={'shuttle':state.artifacts['trajectory'],'racket':state.artifacts['racket'],'coordinate_space':'native_pixels'}
        state.quality['shuttle_observed_fraction']=sum(bool(f['raw_visible']) for f in raw['frames'])/len(raw['frames']);state.missing.pop('racket',None)
        return 'completed'
    def pose3d(self,state,w):
        out=DATA/'专项分析'/self.record['job_id']/'pose3d/pose3d.json'
        subprocess.run([str(ROOT/'.venv-pose3d/bin/python'),str(ROOT/'pose3d_worker.py'),'--job',self.record['job_id'],'--start','0','--end',str(min(5.,state.timestamps[-1])),'--sample-fps','5'],cwd=ROOT,check=True,stdout=(self.folder/'pose3d.log').open('w'),stderr=subprocess.STDOUT,timeout=min(120,w.timeout_seconds))
        packet=read(out)
        if packet['source_sha256']!=state.source_sha256:raise ValueError('3D source mismatch')
        state.artifacts['pose3d']=self.evidence(out,'MediaPipe monocular33 estimated landmarks','derived_observation','Uncalibrated depth and scale; sample window only');state.missing['pose3d']='Only first5-second optional sample; no global displacement';return 'completed_estimates_only'
    def verify(self,state,w):
        dense=read(state.artifacts['pose2d']['artifact']);ball=read(state.artifacts['trajectory']['artifact'])
        checks=[]
        for event in state.events:
            if event.confidence is not None and event.confidence>=w.verification_threshold:continue
            f=dense['frames'][event.source_frame_index];p=f['players'].get(event.player_id);b=ball['frames'][event.source_frame_index];xy=b.get('raw_xy')
            check={'event_id':event.event_id,'method':'independent observed ball/wrist plausibility, not contact truth','status':'unavailable','source_frame_index':event.source_frame_index}
            if p and b.get('raw_visible') and xy:
                wrists=[k for j,k in enumerate(p['keypoints']) if j in [9,10] and k['score']>=.5]
                if wrists:
                    distance=min(math.hypot(k['x']-xy[0],k['y']-xy[1]) for k in wrists);check.update(status='observed',nearest_wrist_distance_px=distance)
            checks.append(check)
        state.observations['cross_agent_verification']=checks
        return 'completed' if checks else 'abstained_no_uncertain_event'

class StrokeUnderstandingAgent(Agent):
    def _convert(self,state,rows,model,path,w):
        events=[]
        for r in rows:
            i=int(r.get('frame_index',r.get('source_frame_index',0)))
            if not 0<=i<len(state.timestamps):raise ValueError('Hit index outside native source')
            scores=r.get('bst0_all_scores') or r.get('specialist_predictions',r.get('bst',{})).get('bst0_all_scores');classes=r.get('class_labels') or r.get('specialist_predictions',r.get('bst',{})).get('class_labels');prob={}
            if scores is not None and classes is not None:
                for c,s in zip(classes,scores):label=mainland(c);prob[label]=prob.get(label,0)+float(s)
            else:
                ranking=r.get('bst_ranking',[])
                # Never renormalize an incomplete top5 into a full probability distribution.
            label=max(prob,key=prob.get) if prob else r.get('stroke_type','unknown');confidence=prob.get(label,r.get('bst_softmax_score'));top_raw=mainland(classes[max(range(len(scores)),key=lambda j:scores[j])]) if scores is not None and classes is not None else label
            if w.bst_decoding=='top1' and scores is not None and classes is not None:label=top_raw;confidence=float(max(scores))
            chosen=None if label=='unknown' or confidence is None or confidence<w.threshold else label
            t=state.timestamps[i];lo=max(0.,min(float(r.get('start',max(0,t-1))),t));hi=min(state.timestamps[-1]+1/state.fps,max(float(r.get('end',t+1)),t+1/state.fps))
            events.append(StrokeEvent(r.get('id') or digest([state.source_sha256,i])[:32],i,t,r.get('player'),chosen,prob,[lo,hi],confidence,uncertainty=['Hit and hitter are automatic candidates','Softmax is uncalibrated; not correctness probability'],missing=['verified_rally_context','validated_quality_score']+([] if chosen else ['stroke_type']),evidence_references=[self.evidence(path,model)],provenance={'kind':'model_prediction','model':model,'raw_label':label,'top_raw_coarse_label':top_raw,'source_sha256':state.source_sha256}))
        state.events=events;state.missing['rally_structure']='No predicted/reviewed rally intervals available; no grouping fabricated'
    def cached(self,state,w):
        p=Path(self.record['classification_file'])
        if not p.exists():
            result=self.bst(state,w);return 'completed_fresh_BST_on_cache_miss'
        packet=read(p)
        if state.artifacts['trajectory']['sha256']!=sha(self.record['trajectory_file']):raise ValueError('Cached stroke inputs differ from current perception; use fresh stroke.bst')
        if packet.get('pose_id')!=self.record['job_id']:raise ValueError('Cached strokes belong to another recording')
        if any(r.get('classification_source')=='manual_annotation' or r.get('human_label') for r in packet['events']):raise ValueError('Manual corrections cannot be reused as model predictions')
        # Cached raw BST output, not labels or personal calibration.
        self._convert(state,packet['events'],'BST-0 existing cached inference',p,w);return 'completed_cached_model_predictions'
    def _classify_hits(self,state,w,rows,method,detector=None):
        from specialist_bst import SpecialistBST
        dense=read(state.artifacts['pose2d']['artifact']);trajectory=read(state.artifacts['trajectory']['artifact']);model=SpecialistBST(dense,trajectory,None)
        for r in rows:
            i=r['frame_index'];r['id']=digest([state.source_sha256,method,i,r.get('player')])[:32]
            # Provide flight context to BST. Detector intervals are not full-action truth.
            lo=max(0,i-round(state.fps));hi=min(len(state.timestamps)-1,i+round(state.fps));r.update(start_frame=lo,end_frame=hi,start=state.timestamps[lo],end=state.timestamps[hi]+1/state.fps,peak_time=state.timestamps[i])
            values,quality=model.inputs(r);pred=model.predict(values);r.update(pred,stroke_type=mainland(pred['bst0'][0]['label']),input_quality=quality)
        path=self.folder/(method+'_bst_predictions.json');save(path,{'kind':'model_prediction','source_sha256':state.source_sha256,'hit_method':method,'detector':detector,'bst_weights_sha256':sha(ROOT/'models/bst/bst_0_JnB_bone_between_2_hits_with_max_limits_seq_100_merged_2.pt'),'events':rows})
        state.artifacts['strokes']=self.evidence(path,method+' + pretrained BST-0');self._convert(state,rows,method+' + BST-0',path,w)
        for e,r in zip(state.events,rows):e.provenance.update(hit_method=method,hit_model_score=r.get('hit_model_score'),interval_kind='candidate context, not verified action duration')
        return 'completed'
    def bst(self,state,w):
        # Historical fixed baseline remains unchanged for reproducibility.
        from specialist_events import proposals
        from specialist_bst import SpecialistBST
        dense=read(state.artifacts['pose2d']['artifact']);trajectory=read(state.artifacts['trajectory']['artifact']);model=SpecialistBST(dense,trajectory,None)
        rows=proposals(dense,trajectory)
        for r in rows:
            values,quality=model.inputs(r);pred=model.predict(values);r.update(pred,stroke_type=mainland(pred['bst0'][0]['label']),input_quality=quality)
        path=self.folder/'bst_predictions.json';save(path,{'kind':'model_prediction','source_sha256':state.source_sha256,'model_sha256':sha(ROOT/'models/bst/bst_0_JnB_bone_between_2_hits_with_max_limits_seq_100_merged_2.pt'),'events':rows})
        state.artifacts['strokes']=self.evidence(path,'pretrained BST-0 + existing geometric hit proposals');self._convert(state,rows,'BST-0 fresh inference',path,w);return 'completed'
    def _hit_tool(self,method,state,w):
        from .hit_tools import inventory,run_method
        info=inventory(state,allow_expensive=True);result=run_method(method,state,info,self.folder,max(.1,w.timeout_seconds-5))
        state.observations['hit_routing']={'policy':'explicit registered tool','selected_method':method,'inventory':info['methods'],'executions':[result],'ground_truth_used_for_selection':False}
        if result['status']!='completed':state.events=[];state.missing['hit_detector']=result.get('reason',result['status']);return result['status']
        return self._classify_hits(state,w,result['events'],method,result['metadata'])
    def fusion(self,state,w):return self._hit_tool('fusion',state,w)
    def hitnet(self,state,w):return self._hit_tool('hitnet',state,w)
    def swingnet(self,state,w):return self._hit_tool('swingnet',state,w)
    def gru(self,state,w):return self._hit_tool('gru_shared',state,w)
    def gru_original(self,state,w):return self._hit_tool('gru_original',state,w)
    def rgb_hit(self,state,w):return self._hit_tool('rgb_resnet',state,w)
    def auto(self,state,w):return self._auto(state,w,False)
    def auto_full(self,state,w):return self._auto(state,w,True)
    def _auto(self,state,w,allow_expensive):
        import time
        from .hit_tools import inventory,choose,run_method,agreement
        started=time.perf_counter();info=inventory(state,allow_expensive);choices=choose(info,w.timeout_seconds,allow_expensive);limit=min(2,max(0,w.max_calls-len(w.tools)));decisions=[];primary=None;primary_method=None
        for method in choices[:limit]:
            remaining=w.timeout_seconds-(time.perf_counter()-started)-5
            if remaining<=0:break
            result=run_method(method,state,info,self.folder,remaining);decisions.append(result)
            if result['status']=='completed' and result['events'] and primary is None:primary=result;primary_method=method
        routing={'policy':'compatibility/coverage/budget-v1; deterministic Agent routing, not AFlow/EvoFlow learned policy','allow_expensive':allow_expensive,'selected_method':primary_method,'quality':info['quality'],'inventory':info['methods'],'considered_order':choices,'executions':decisions,'ground_truth_used_for_selection':False,'accuracy_from_old_experiments_used_for_selection':False,'reason':'Prefer original inputs when available; otherwise compatible GRU/SHF/RGB tools, rule fallback. Expensive branches require deep profile and remaining budget.'}
        if primary:
            secondary=next((r for r in decisions if r is not primary and r['status']=='completed'),None)
            if secondary:
                routing['agreement']=agreement(primary['events'],secondary['events'],state.fps)
                routing['requires_review']=routing['agreement']['matched_candidates']<.5*len(primary['events'])
            self._classify_hits(state,w,primary['events'],primary_method,primary['metadata'])
            matched={r['primary_frame'] for r in routing.get('agreement',{}).get('pairs',[])}
            for e in state.events:
                e.provenance['verification']='supported_by_other_detector' if e.source_frame_index in matched else 'not_supported_or_not_checked'
                e.uncertainty.append('Detector agreement is not contact ground truth')
        else:state.events=[];state.missing['hit_detector']='No eligible detector with nonempty evidence within the bounded budget'
        state.observations['hit_routing']=routing;save(self.folder/'hit_routing.json',routing)
        return 'completed_agent_selection' if primary else 'abstained_no_supported_hits'
    def shf(self,state,w):return self._hit_tool('shf',state,w)
    def check_labels(self,state,w):
        count=0
        for e in state.events:
            if e.class_probabilities and e.provenance.get('top_raw_coarse_label')!=max(e.class_probabilities,key=e.class_probabilities.get):
                e.stroke_label=None;e.uncertainty.append('Raw/aggregated class inconsistency; abstained');count+=1
        state.observations['label_consistency']={'abstained_events':count,'scope':'within-model consistency, not independent verification'};return 'completed'
    def dsta(self,state,w):
        from research_adapters import DSTAAdapter
        result=DSTAAdapter().infer(rgb_video_path=state.source_file)
        state.observations['dsta']=result;state.missing['dsta']='No trained/validated checkpoint; no intervals invented';return 'unavailable_trained_checkpoint'
    def window(self,state,w):
        from specialist_bst import SpecialistBST
        dense=read(state.artifacts['pose2d']['artifact']);ball=read(state.artifacts['trajectory']['artifact']);chosen=sorted([e for e in state.events if e.confidence is None or e.confidence<w.verification_threshold],key=lambda e:e.confidence or 0)[:w.max_reanalyses]
        if not chosen:return 'abstained_no_uncertain_event'
        model=SpecialistBST(dense,ball,None);reanalyses=[]
        for e in chosen:
            import bisect
            lo=max(0,e.timestamp-1.5);hi=min(state.timestamps[-1]+1/state.fps,e.timestamp+1.5)
            event={'start_frame':bisect.bisect_left(state.timestamps,lo),'end_frame':min(len(state.timestamps)-1,bisect.bisect_left(state.timestamps,hi)-1),'frame_index':e.source_frame_index,'peak_time':e.timestamp,'player':e.player_id}
            values,quality=model.inputs(event);pred=model.predict(values);probs={}
            for c,v in zip(pred['class_labels'],pred['bst0_all_scores']):k=mainland(c);probs[k]=probs.get(k,0)+float(v)
            label=max(probs,key=probs.get);confidence=probs[label];top_raw=mainland(pred['class_labels'][max(range(len(pred['bst0_all_scores'])),key=lambda j:pred['bst0_all_scores'][j])]);e.provenance['top_raw_coarse_label']=top_raw
            if w.bst_decoding=='top1':label=top_raw;confidence=max(pred['bst0_all_scores'])
            e.class_probabilities=probs;e.confidence=confidence;e.stroke_label=label if label!='unknown' and confidence>=w.threshold else None
            e.provenance['reanalysis']={'model':'same BST-0, wider3-second context','window':[lo,hi],'not_independent_verification':True};e.missing=[x for x in e.missing if x!='stroke_type']+([] if e.stroke_label else ['stroke_type'])
            reanalyses.append({'event_id':e.event_id,'source_frame_index':e.source_frame_index,'window':[lo,hi],'prediction':pred,'quality':quality})
        path=self.folder/'window_predictions.json';save(path,reanalyses);state.artifacts['local_reanalysis']=self.evidence(path,'BST-0 wider context');return 'completed'

class TechnicalTacticalAgent(Agent):
    def evidence(self,*args,**kwargs):return super().evidence(*args,**kwargs)
    def execute(self,tool,state,w):
        self.memory.update(executions=self.memory['executions']+1,last_tool=tool)
        return self.analyze(state,w) if tool=='technical.evidence' else self.qwen(state,w)
    def analyze(self,state,w):
        from research_report import build
        from tactical_review import build as review
        folder=self.folder/'technical';folder.mkdir(exist_ok=True);shutil.copy2(state.artifacts['trajectory']['artifact'],folder/'trajectory.json')
        if state.artifacts.get('racket'):
            (folder/'racketvision').mkdir(exist_ok=True);shutil.copy2(state.artifacts['racket']['artifact'],folder/'racketvision/racketvision.json')
        rows=[{'id':e.event_id,'frame_index':e.source_frame_index,'peak_time':e.timestamp,'player':e.player_id,'start':e.interval[0],'end':e.interval[1],'start_frame':next(i for i,t in enumerate(state.timestamps) if t>=e.interval[0]),'end_frame':min(len(state.timestamps)-1,e.source_frame_index+round(state.fps)),'stroke_type':e.stroke_label or 'unknown','bst_softmax_score':e.confidence,'classification_source':'professional_agent_bst','review_status':'unreviewed','proposal_score':None} for e in state.events]
        report=build(self.record['job_id'],folder=folder,dense=read(state.artifacts['pose2d']['artifact']),baseline=rows,labels={},source_video_id=state.video_id)
        catalogue=review(report);save(folder/'tactical_review.json',catalogue)
        state.artifacts['technical_report']=super().evidence(folder/'research_report.json','existing explainable measurement','derived_observation');state.artifacts['tactical_review']=super().evidence(folder/'tactical_review.json','TactiPlay-inspired evidence catalogue','derived_observation')
        byframe={e['hit_frame']:e for e in report['events']}
        for e in state.events:
            r=byframe.get(e.source_frame_index)
            if r:
                e.technical_observations=r.get('quality_metrics',{}).get('observations_2d',{});e.pose_features={'estimated_3d':r.get('quality_metrics',{}).get('estimated_3d',{}),'measurement_space':'2D pixels; optional estimated3D, not calibrated biomechanics'}
        state.observations['tactical']={'unit_count':len(catalogue['units']),'candidate_count':len(catalogue['candidate_catalog']),'rally_status':catalogue['rally_status'],'validated_technical_scores':None,'causal_tactical_diagnosis':None}
        return 'completed'
    def qwen(self,state,w):
        review=read(state.artifacts['tactical_review']['artifact']);target=self.folder/'qwen_feedback.json';packet=self.folder/'qwen_request.json'
        save(packet,{'review':review,'output':str(target)})
        # Actual existing worker contract is checked in the integration smoke test.
        subprocess.run([str(ROOT/'.venv-vision/bin/python'),str(ROOT/'tactical_review_worker.py'),'--input',str(packet)],cwd=ROOT,check=True,stdout=(self.folder/'qwen.log').open('w'),stderr=subprocess.STDOUT,timeout=min(240,w.timeout_seconds))
        feedback=read(target);state.observations['llm_feedback']=feedback;state.artifacts['llm_feedback']=super().evidence(target,'local Qwen closed evidence selection','derived_observation');return 'completed'

class ExpertDemonstrationAgent(Agent):
    def retrieve(self,state,w):
        index=read(STORE/'retrieval/index.json');results=[]
        for e in state.events:
            query={'match_key':state.match_key,'stroke_label':e.stroke_label,'sequence':e.rally_context.get('sequence',[]),'court_zone':e.court_coordinates.get('zone')}
            results.append({'event_id':e.event_id,'timestamp':e.timestamp,'query_label':e.stroke_label,'matches':retrieve(query,index,w.retrieval_ranker,w.top_k)})
        state.retrieval=results;state.missing['retrieval_relevance']='No independently reviewed corrective relevance annotations; Recall@K/nDCG unavailable';return 'completed'
