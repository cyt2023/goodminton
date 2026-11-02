"""Verified research model interfaces, import provenance and clock conversion."""
from bisect import bisect_left
import json
import math
import hashlib
from pathlib import Path
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parent
CONFIG=ROOT/'research_models'


def registry():return json.loads((CONFIG/'registry.json').read_text())

def settings():return json.loads((CONFIG/'settings.json').read_text())

def finite(v):return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)


def nearest_frame(times,t):
    i=bisect_left(times,t)
    return min([max(0,i-1),min(len(times)-1,i)],key=lambda k:abs(times[k]-t))


def validate_clock(times):
    if not times or any(not finite(t) or t<0 for t in times) or any(b<=a for a,b in zip(times,times[1:])):
        raise ValueError('Original-frame timestamps must be finite and strictly increasing')


def map_label(value):
    mapping=json.loads((CONFIG/'label_map.json').read_text())
    raw=str(value)
    return mapping['numeric_class_ids'].get(raw,'unknown') if isinstance(value,int) else mapping['names'].get(raw,'unknown')


class ShuttleHitFormerAdapter:
    def status(self):return registry()['shuttlehitformer']
    def infer(self, *, player_pose, shuttle_trajectory, court_geometry, net_geometry=None,
              checkpoint=None, experimental=False):
        if checkpoint is None:
            return {'status':'unavailable','predictions':[],'provenance':self.status(),
                    'implementation_status':'implemented_locally','reason':'validated_checkpoint_missing'}
        if not isinstance(player_pose,dict):
            raise ValueError('Local SHF requires top/bottom pose arrays')
        with tempfile.TemporaryDirectory(prefix='badminton-shf-') as directory:
            path=Path(directory)/'input.json'
            path.write_text(json.dumps({'top':player_pose['top'],'bottom':player_pose['bottom'],
                'court':court_geometry,'net':net_geometry,'shuttle':shuttle_trajectory,
                'normalized':player_pose.get('normalized',True),'width':player_pose.get('width'),
                'height':player_pose.get('height')},allow_nan=False))
            return _local_infer('shuttlehitformer',checkpoint,path,experimental=experimental)


class DSTAAdapter:
    def status(self):return registry()['dsta']
    def infer(self, *, rgb_video_path, checkpoint=None, experimental=False, start=0.):
        if checkpoint is None:
            return {'status':'unavailable','predictions':[],'provenance':self.status(),
                    'requested_input':str(rgb_video_path),'feature_extraction_status':'trained_adapter_checkpoint_missing',
                    'implementation_status':'implemented_locally'}
        return _local_infer('dsta',checkpoint,rgb_video_path,experimental=experimental,start=start)


def _local_infer(component,checkpoint,input_path,*,experimental=False,start=0.):
    # Keep torch/legacy research dependencies out of the stdlib website process.
    with tempfile.TemporaryDirectory(prefix='badminton-paper-') as directory:
        out=Path(directory)/'prediction.json'
        command=[str(ROOT/'.venv-vision/bin/python'),'-m','paper_reproductions.run','infer',
                 '--checkpoint',str(Path(checkpoint).resolve()),'--input',str(Path(input_path).resolve()),
                 '--out',str(out),'--start',str(start)]
        if experimental:command.append('--experimental')
        result=subprocess.run(command,cwd=ROOT,text=True,capture_output=True,timeout=1800)
        if result.returncode:raise RuntimeError('Local paper model inference failed: '+result.stderr[-2000:])
        packet=json.loads(out.read_text())
        if packet['component']!=component:raise ValueError('Checkpoint belongs to a different research component')
        raw=packet['prediction'];rows=raw.get('events',[]) if component=='shuttlehitformer' else raw.get('intervals',[])
        return {'status':'experimental_model_inference' if not packet['deployment_ready'] else 'model_inference',
                'predictions':rows,'provenance':{k:v for k,v in packet.items() if k!='prediction'},
                'details':raw,'feature_extraction_status':'local_model_executed'}


def import_predictions(packet,component,source_sha256,times,nominal_fps):
    validate_clock(times)
    if component not in ['shuttlehitformer','dsta']:raise ValueError('Unknown research component')
    if packet.get('source_sha256')!=source_sha256:raise ValueError('Imported predictions belong to another video')
    if packet.get('component')!=component:raise ValueError('Imported component does not match request')
    if packet.get('kind') not in ['imported_prediction','manual_annotation']:raise ValueError('Declare imported_prediction or manual_annotation explicitly')
    model=packet.get('model_provenance')
    if not isinstance(model,dict) or not model.get('name'):raise ValueError('Imported results require model/annotation provenance')
    basis=packet.get('time_base',{});unit=basis.get('unit');fps=basis.get('fps');origin=basis.get('frame_index_origin')
    if unit not in ['seconds','frames']:raise ValueError('Import must declare seconds or frames')
    if unit=='frames' and (not finite(fps) or fps<=0 or origin not in [0,1]):raise ValueError('Frame imports require fps and frame_index_origin 0 or 1')
    if component=='dsta' and basis.get('interval_end') not in ['inclusive','exclusive']:raise ValueError('DSTA must declare whether interval end is inclusive or exclusive')
    def seconds(v):
        if not finite(v):raise ValueError('Non-finite prediction time')
        return (v-origin)/fps if unit=='frames' else v
    accepted=[];rejected=[];duration=times[-1]+1/nominal_fps
    for n,row in enumerate(packet.get('predictions',[])):
        try:
            confidence=row.get('confidence')
            if confidence is not None and (not finite(confidence) or not 0<=confidence<=1):raise ValueError('Confidence must be in [0,1] or null')
            external_id=str(row.get('id',f'{component}-{n}'))
            identifier=hashlib.sha256(f'{component}:{source_sha256}:{external_id}:{n}'.encode()).hexdigest()[:32]
            r={'id':identifier,'external_id':external_id,'confidence':confidence,
               'provenance':{'component':component,'kind':packet['kind'],'declared_model':model,'time_base':basis}}
            if component=='shuttlehitformer':
                t=seconds(row['time'])
                if not 0<=t<duration:raise ValueError('Hit outside original video')
                i=nearest_frame(times,t);r.update(frame_index=i,time=times[i],imported_time=t,
                    candidate_player_id=row.get('player_id'),player_id=row.get('player_id') if packet['kind']=='manual_annotation' and row.get('player_id') in ['near','far'] else None)
            else:
                start,end=seconds(row['start']),seconds(row['end'])
                if basis['interval_end']=='inclusive':
                    if unit!='frames':raise ValueError('Inclusive intervals require frame units')
                    end+=1/fps
                if not start<end or end<=0 or start>=duration:raise ValueError('Invalid action interval')
                lo,hi=max(0,start),min(duration,end)
                a=nearest_frame(times,lo);b=max(a,min(len(times)-1,bisect_left(times,hi)-1))
                r.update(start=lo,end=hi,start_frame=a,end_frame=b,raw_label=row['label'],stroke_type=map_label(row['label']),
                         candidate_player_id=row.get('player_id'),clipped_to_video=(lo!=start or hi!=end))
            accepted.append(r)
        except (ValueError,KeyError,TypeError) as error:rejected.append({'row_index':n,'reason':str(error),'input':row})
    return {'component':component,'kind':packet['kind'],'predictions':accepted,'rejected':rejected,
            'model_provenance':model,'source_sha256':source_sha256,'time_base':basis}
