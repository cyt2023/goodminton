"""Typed shared state, original-source clocks and explicit missingness."""
from __future__ import annotations
from dataclasses import dataclass,field,asdict
from typing import Any,Literal
from pathlib import Path
import hashlib,json,math
from . import SCHEMA_VERSION
ROOT=Path(__file__).resolve().parents[1]
STORE=ROOT.parent/'03_数据/职业Agent平台'

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(p)
@dataclass
class Evidence:
    artifact:str
    sha256:str
    kind:Literal['model_prediction','dataset_annotation','derived_observation']
    model:str|None=None
    note:str=''
@dataclass
class StrokeEvent:
    event_id:str
    source_frame_index:int
    timestamp:float
    player_id:str|None
    stroke_label:str|None
    class_probabilities:dict[str,float]
    interval:list[float]
    confidence:float|None=None
    court_coordinates:dict=field(default_factory=dict)
    pose_features:dict=field(default_factory=dict)
    rally_context:dict=field(default_factory=dict)
    technical_observations:dict=field(default_factory=dict)
    evidence_references:list[dict]=field(default_factory=list)
    uncertainty:list[str]=field(default_factory=list)
    missing:list[str]=field(default_factory=list)
    provenance:dict=field(default_factory=dict)
@dataclass
class SharedState:
    video_id:str
    match_key:str
    source_sha256:str
    source_file:str
    fps:float
    timestamps:list[float]
    source_offset_seconds:float
    role:str
    dataset:str
    schema_version:str=SCHEMA_VERSION
    domain:str='professional'
    artifacts:dict[str,dict]=field(default_factory=dict)
    player_tracks:dict[str,Any]=field(default_factory=dict)
    poses:dict[str,Any]=field(default_factory=dict)
    trajectories:dict[str,Any]=field(default_factory=dict)
    court_geometry:dict[str,Any]=field(default_factory=dict)
    events:list[StrokeEvent]=field(default_factory=list)
    rallies:list[dict]=field(default_factory=list)
    observations:dict=field(default_factory=dict)
    retrieval:list[dict]=field(default_factory=list)
    quality:dict=field(default_factory=dict)
    missing:dict=field(default_factory=dict)
    provenance:dict=field(default_factory=dict)
    completed_tools:list[str]=field(default_factory=list)
    def validate(self):
        if self.schema_version!=SCHEMA_VERSION or self.domain!='professional':raise ValueError('Only the versioned professional schema is supported')
        if not self.source_sha256 or self.fps<=0 or not self.timestamps:raise ValueError('Source identity and clock required')
        if any(not math.isfinite(t) or t<0 for t in self.timestamps) or any(b<=a for a,b in zip(self.timestamps,self.timestamps[1:])):raise ValueError('Invalid original frame clock')
        seen=set()
        for e in self.events:
            if e.event_id in seen:raise ValueError('Duplicate event identity')
            seen.add(e.event_id)
            if not 0<=e.source_frame_index<len(self.timestamps) or abs(e.timestamp-self.timestamps[e.source_frame_index])>1e-6:raise ValueError('Event frame/timestamp misalignment')
            if not 0<=e.interval[0]<=e.timestamp<e.interval[1]<=self.timestamps[-1]+1/self.fps+1e-5:raise ValueError('Event interval exceeds source')
            if any(not math.isfinite(v) or not 0<=v<=1 for v in e.class_probabilities.values()):raise ValueError('Invalid class probability')
            if e.class_probabilities and abs(sum(e.class_probabilities.values())-1)>1e-4:raise ValueError('Probabilities must sum to1')
            if e.provenance.get('kind')!='model_prediction':raise ValueError('Reference/manual labels cannot masquerade as inference')
        return self
    def packet(self):self.validate();return asdict(self)
    @classmethod
    def from_packet(cls,p):
        p=dict(p);p['events']=[StrokeEvent(**e) for e in p.get('events',[])];return cls(**p).validate()
