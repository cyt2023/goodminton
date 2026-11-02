"""Stdlib3D evidence consumer: root-relative estimates, 2D baseline retained."""
import math
from statistics import mean
from research_quality import angle,distance


def validate_packet(packet,source_sha256,times):
    if packet.get('source_sha256')!=source_sha256:raise ValueError('3D pose belongs to another source')
    if packet.get('joint_schema')!='mediapipe33' or packet.get('coordinate_system')!='mediapipe_world_hip_origin_xyz' or packet.get('units')!='estimated_meters':raise ValueError('Declare the supported3D joint schema/space/estimated units')
    previous=-1
    for f in packet['frames']:
        i=f['frame_index']
        if not isinstance(i,int) or i<=previous or not 0<=i<len(times) or abs(f['timestamp']-times[i])>.001:raise ValueError('3D and original-frame clocks differ')
        previous=i
        for p in f['players'].values():
            if p is None:continue
            if len(p['joints'])!=33:raise ValueError('MediaPipe33 joints required')
            if any(not isinstance(j.get(k),(int,float)) or not math.isfinite(j[k]) for j in p['joints'] for k in ['x','y','z','score']):raise ValueError('Nonfinite3D joints')
    return packet


def measure_3d(event,packet):
    if not packet:return {'status':'disabled_or_missing','observations':{},'reason':'Optional3D perception has not run'}
    side=event.get('player_id');frames=[f for f in packet['frames'] if event['action_start_frame']<=f['frame_index']<=event['action_end_frame']]
    observations={};trajectories={};coverage={};peak_times={};angles={};max_gap=2.1/packet['sample_fps']
    def point(f,k):
        p=f['players'].get(side) if side else None
        if not p:return None
        j=p['joints'][k]
        return [j['x'],j['y'],j['z']] if j['score']>=.5 else None
    for name,k in [('left_wrist',15),('right_wrist',16),('left_ankle',27),('right_ankle',28)]:
        valid=[(f,point(f,k)) for f in frames];coverage[name]=sum(p is not None for _,p in valid)/len(frames) if frames else 0.
        trajectories[name]=[{'frame_index':f['frame_index'],'timestamp':f['timestamp'],'xyz':p} for f,p in valid if p is not None]
        speeds=[]
        for (a,p),(b,q) in zip(valid,valid[1:]):
            dt=b['timestamp']-a['timestamp']
            if p is not None and q is not None and 0<dt<=max_gap:speeds.append((distance(p,q)/dt,b['timestamp']))
        if len(speeds)>=2 and coverage[name]>=.8:
            velocity,t=max(speeds);observations[name+'_relative_peak_estimated_m_s']=velocity;peak_times[name]=t
        else:observations[name+'_relative_peak_estimated_m_s']=None
    for name,joints in [('left_elbow',(11,13,15)),('right_elbow',(12,14,16)),('left_knee',(23,25,27)),('right_knee',(24,26,28)),('left_wrist',(13,15,19)),('right_wrist',(14,16,20)),('left_shoulder',(13,11,23)),('right_shoulder',(14,12,24))]:
        series=[]
        for f in frames:
            pts=[point(f,j) for j in joints]
            if all(p is not None for p in pts):
                a=angle(*pts)
                if a is not None:series.append({'frame_index':f['frame_index'],'timestamp':f['timestamp'],'degrees':a})
        angles[name]=series;near=min(series,key=lambda x:abs(x['timestamp']-event['hit_timestamp'])) if series else None
        tolerance=1.1/packet['sample_fps']
        observations[name+'_estimated_3d_deg']=near['degrees'] if near and abs(near['timestamp']-event['hit_timestamp'])<=tolerance else None
    coordination=[]
    for hand in ['left','right']:
        wrist=peak_times.get(hand+'_wrist');feet=[peak_times[k] for k in ['left_ankle','right_ankle'] if k in peak_times]
        if wrist is not None and len(feet)==2 and len(frames)>=8:
            lag=wrist-max(feet);observations[hand+'_wrist_vs_ankle_peak_lag_s']=lag;coordination.append({'hand':hand,'lag_seconds':lag,'sampling_resolution_s':1/packet['sample_fps']})
    torso=[]
    for f in frames:
        pts=[point(f,k) for k in [11,12,23,24]]
        if all(p is not None for p in pts):torso.append((f['timestamp'],[(pts[0][d]+pts[1][d]-pts[2][d]-pts[3][d])/2 for d in range(3)]))
    changes=[]
    for (ta,a),(tb,b) in zip(torso,torso[1:]):
        if 0<tb-ta<=max_gap:
            value=angle(a,[0,0,0],b)
            if value is not None:changes.append(value)
    observations['torso_axis_mean_change_estimated_deg']=mean(changes) if len(changes)>=2 else None
    return {'status':'estimated_features_available' if any(v is not None for v in observations.values()) else 'insufficient_observations',
        'measurement_space':'monocular_estimated_3d_hip_relative','observations':observations,'joint_trajectories':trajectories,'joint_angles':angles,
        'coverage':coverage,'temporal_coordination':coordination,'body_movement':{'torso_axis_mean_change_estimated_deg':observations['torso_axis_mean_change_estimated_deg'],'global_centre_of_mass_displacement':None,
        'reason':'Hip-origin landmarks remove global translation; use reviewed2D/court tracking separately'},
        'uncertainty':{**packet['uncertainty'],'temporal_resolution_s':1/packet['sample_fps'],'identity_status':'inherited_2d_track_unverified'},
        'quality_score':None,'reason':'Estimated3D observations do not supply validated stroke-specific reference values or racket-face normals'}
