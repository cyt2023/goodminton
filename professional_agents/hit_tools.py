"""Registered hit-localization tools and label-free compatibility routing.
No historical accuracy numbers or benchmark annotations are routing inputs.
"""
from pathlib import Path
import os,signal,subprocess,time
from .state import ROOT,STORE,read,save,sha,digest
AUTHOR_ASSETS=ROOT.parent/'03_数据/GRU作者流程复现/2026-10-25/author_assets'
FILTERED=AUTHOR_ASSETS/'shuttlepose_filtered2/keras_tensors.npz'
ASSETS={
 'rules':[],
 'fusion':[ROOT/'models/hit_gru/keras_tensors.npz'],
 'hitnet':[ROOT/'vendor/Adapted-HitNet/training/hitnet_model.h5'],
 'swingnet':[ROOT/'models/BadmintonSwingNet/fold5_swingnet_3000.pth.tar'],
 'gru_shared':[ROOT/'models/hit_gru/keras_tensors.npz'],
 'gru_original':[FILTERED,ROOT/'models/gru_author_pose/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth',ROOT/'models/gru_author_pose/hrnet_w48_coco_384x288-314c8528_20200708.pth',ROOT/'models/tracknetv2_author/model906_30'],
 'shf':[Path(read(ROOT/'research_models/registry.json')['shuttlehitformer']['local_checkpoint'])],
 'rgb_resnet':[AUTHOR_ASSETS/'resnet_nov_15epoch_indv_clearnohit.pt'],
}
NAMES={'fusion':'音频/视觉/GRU融合找拍','hitnet':'第三方Adapted-HitNet','swingnet':'Badminton SwingNet（前处理未完整复现）','rules':'几何规则找拍','gru_shared':'GRU（现有人体/球轨迹适配）','gru_original':'原作者GRU完整链路','shf':'ShuttleHitFormer本地修正版','rgb_resnet':'原作者RGB ResNet找拍'}

def reviewed_corners(state):
    """Use reviewed user camera calibration, never BFMD reference geometry."""
    ref=state.artifacts.get('court')
    if not ref or ref.get('kind')=='dataset_annotation':return None
    path=Path(ref['artifact'])
    if '独立基准/BFMD/raw' in str(path):return None  # BFMD court annotations; own reviewed calibrations allowed
    value=read(path);record=value.get('record',value)
    if record.get('source_sha256')!=state.source_sha256 or record.get('review_status')!='confirmed':return None
    if record.get('court_corners') is not None:return record['court_corners']
    import cv2,numpy as np
    src=np.array(record.get('image_points_px',[]),float);dst=np.array(record.get('court_points_normalized',[]),float)
    if src.shape!=dst.shape or src.ndim!=2 or src.shape[0]<4:return None
    matrix,_=cv2.findHomography(dst,src,method=0)
    if matrix is None:return None
    return cv2.perspectiveTransform(np.array([[[0,0],[1,0],[0,1],[1,1]]],float),matrix)[0].tolist()

def inventory(state,allow_expensive=False):
    dense=read(state.artifacts['pose2d']['artifact']);ball=read(state.artifacts['trajectory']['artifact']);n=len(dense['frames']);paired=sum(all(f['players'].get(s) for s in ['near','far']) for f in dense['frames'])/max(1,n)
    observed=sum(bool(b['raw_visible']) for b in ball['frames'])/max(1,n);corners=reviewed_corners(state)
    exact_pose=dense.get('engine_version')=='gru-author-fasterrcnn-hrnet-v1'
    exact_ball=ball.get('schema_version')=='author-gru-tracknetv2-v1'
    bbox=all('bbox_score' in p for f in dense['frames'] for p in f['players'].values() if p)
    original_ready=exact_pose and exact_ball and bbox and dense.get('court_corners_px_TL_TR_BL_BR') is not None
    if corners is None and original_ready:corners=dense['court_corners_px_TL_TR_BL_BR']
    # SHF requires reviewed geometry or an explicitly automatic raw-image proposal.
    shf_corners=None;court_basis=None
    if corners is not None:shf_corners=[corners[i] for i in [0,1,3,2]];court_basis='reviewed camera calibration'
    elif ASSETS['shf'][0].exists():
        import cv2
        from court_geometry_worker import automatic_corners
        from court_calibration import color_mask
        cap=cv2.VideoCapture(state.source_file);ok,image=cap.read();cap.release()
        proposal=automatic_corners(color_mask(image,'white')) if ok else None
        if proposal:shf_corners=[proposal['points_px'][i] for i in [0,1,3,2]];court_basis='automatic unreviewed Hough, white-line assumption'
    usable=[all(f['players'].get(side) for side in ['near','far']) and bool(b['raw_visible']) for f,b in zip(dense['frames'],ball['frames'])]
    consecutive=sum(all(usable[i-3:i+5]) for i in range(3,n-4))/max(1,n-7)
    result={}
    for name,paths in ASSETS.items():
        missing=[str(p) for p in paths if not p.is_file()];reason=[]
        if missing:reason.append('本地权重缺失')
        if n<14 and name.startswith('gru'):reason.append('不足14个原始帧')
        if paired==0:reason.append('没有可用的双人姿态')
        if name in ['rules','gru_shared','shf'] and observed==0:reason.append('没有观测球轨迹')
        if name=='swingnet':reason.append('原作者光流/去背景前处理未完整复现，已有替代消融不能作为自动基线')
        if name=='hitnet' and corners is None:reason.append('缺少已确认球场四角')
        if name=='shf' and shf_corners is None:reason.append('缺少兼容的球场几何')
        if name=='gru_original' and not original_ready:
            if corners is None:reason.append('缺少已确认球场标定，不能执行原作者选人')
            if not allow_expensive:reason.append('当前预算策略不重跑完整视觉前端')
        if name=='rgb_resnet' and not allow_expensive:reason.append('当前预算策略不运行昂贵RGB模型')
        result[name]={'name':NAMES[name],'installed':not missing,'eligible':not reason,'reasons':reason,'missing_assets':missing,'cost_class':'cheap' if name in ['rules','gru_shared'] or name=='gru_original' and original_ready else 'expensive','input_condition':'original FasterRCNN/HRNet + TrackNetV2 + actual bbox confidence + filtered2 legacy layout' if name=='gru_original' else 'shared existing perception; explicitly adapted' if name=='gru_shared' else 'registered model inputs','model_hashes':{str(p):sha(p) for p in paths if p.exists()}}
    result['shf']['cost_class']='cheap';result['hitnet']['cost_class']='cheap'
    return {'methods':result,'quality':{'paired_player_fraction':paired,'observed_ball_fraction':observed,'frames':n,'valid_shf_window_fraction':consecutive,'quality_is_accuracy':False},'original_cached_ready':original_ready,'author_corners':corners,'shf_corners':shf_corners,'shf_court_basis':court_basis}

def choose(info,remaining_seconds,allow_expensive=False):
    if remaining_seconds<=5:return []
    methods=info['methods'];quality=info['quality'];order=[]
    if info['original_cached_ready']:order.append('gru_original')
    elif allow_expensive and remaining_seconds>=120:order.append('gru_original')
    if quality['paired_player_fraction']>=.5 and quality['observed_ball_fraction']>=.2:
        order+=['shf','hitnet','gru_shared'] if quality['valid_shf_window_fraction']>=.6 and methods['shf']['eligible'] else ['gru_shared','shf','hitnet']
    else:order+=['rgb_resnet','shf','gru_shared']
    order+=['rules'];order=list(dict.fromkeys(order))
    eligible=[x for x in order if methods[x]['eligible'] and (methods[x]['cost_class']=='cheap' or allow_expensive and remaining_seconds>=30)]
    # At most2 methods; cheap independent rule check retains primary events,
    # disagreement is evidence for review, never an automatic union or vote.
    if eligible and eligible[0]!='rules':
        verifier='fusion' if allow_expensive and remaining_seconds>=30 and methods['fusion']['eligible'] else 'rules'
        if methods[verifier]['eligible']:eligible=[eligible[0],verifier]
    return eligible[:2]

def run_method(method,state,info,folder,timeout):
    if method not in ASSETS:raise ValueError('Unregistered hit tool')
    spec=info['methods'][method]
    if not spec['eligible']:return {'status':'unavailable','reason':'; '.join(spec['reasons']),'events':[],'method':method,'cache_hit':False,'seconds':0.}
    request={'method':method,'video':state.source_file,'source_sha256':state.source_sha256,'dense':state.artifacts['pose2d']['artifact'],'trajectory':state.artifacts['trajectory']['artifact'],'author_corners':info['author_corners'],'shf_corners':info['shf_corners'],'shf_court_basis':info['shf_court_basis'],'original_cached_ready':info['original_cached_ready']}
    key=digest({'request':request,'pose_sha256':state.artifacts['pose2d']['sha256'],'trajectory_sha256':state.artifacts['trajectory']['sha256'],'weights':spec['model_hashes'],'code':{p:sha(ROOT/p) for p in ['professional_agents/hit_tools.py','professional_agents/hit_worker.py','benchmark_hits_gru.py','benchmark_hit_models.py','benchmark_hits_hitnet.py','benchmark_hits_swingnet.py','hit_refinement.py','hit_gru.py','run_gru_author_reproduction.py','reproduce_gru_author_pose.py','reproduce_gru_author_tracknet.py','reproduce_gru_author_resnet.py','specialist_events.py','stroke_features.py','input_quality.py','contact_timing.py','source_frame_media.py','paper_reproductions/shuttlehitformer.py','paper_reproductions/data.py','paper_reproductions/run.py']}})
    cache=STORE/'hit_tool_cache'/key;target=cache/'prediction.json';start=time.perf_counter()
    if target.exists():
        result=read(target);return {**result,'artifact':str(target),'cache_hit':True,'seconds':time.perf_counter()-start}
    cache.mkdir(parents=True,exist_ok=True);request['output']=str(target);request['workdir']=str(cache);req=cache/'input.json';save(req,request)
    with (cache/'run.log').open('w') as log:
        p=subprocess.Popen([str(ROOT/'.venv-vision/bin/python'),'-m','professional_agents.hit_worker','--request',str(req)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:p.wait(timeout=max(.1,timeout))
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=3)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            return {'status':'budget_exhausted','method':method,'events':[],'seconds':time.perf_counter()-start,'cache_hit':False,'reason':'找拍工具超时；子进程组已停止','log':str(cache/'run.log')}
    if p.returncode or not target.exists():return {'status':'failed','method':method,'events':[],'seconds':time.perf_counter()-start,'cache_hit':False,'reason':'模型执行失败，保留日志并允许有限回退','log':str(cache/'run.log')}
    return {**read(target),'artifact':str(target),'cache_hit':False,'seconds':time.perf_counter()-start}

def agreement(primary,secondary,fps):
    used=set();pairs=[]
    for event in primary:
        candidates=[(abs(event['frame_index']-other['frame_index']),j) for j,other in enumerate(secondary) if j not in used and abs(event.get('peak_time',event['frame_index']/fps)-other.get('peak_time',other['frame_index']/fps))<=.1+1e-9]
        if candidates:distance,j=min(candidates);used.add(j);pairs.append({'primary_frame':event['frame_index'],'secondary_frame':secondary[j]['frame_index'],'delta_frames':distance})
    return {'matched_candidates':len(pairs),'primary_candidates':len(primary),'secondary_candidates':len(secondary),'pairs':pairs,'interpretation':'Model agreement, not correctness. Primary detections retained; no automatic union.'}
