"""Closed tool registry. Workflow graphs are bounded topological programs."""
from dataclasses import dataclass,asdict
@dataclass(frozen=True)
class Tool:
    agent:str
    requires:tuple[str,...]
    cost_class:str
    description:str
TOOLS={
 'perception.cached':Tool('PerceptionAgent',(),'cached','Reuse source-verified YOLOX/RTMPose and shuttle tracks'),
 'perception.racketvision':Tool('PerceptionAgent',(),'expensive','Run existing RacketVision in isolated environment; reuse pose'),
 'perception.pose3d':Tool('PerceptionAgent',('perception',),'expensive','Existing optional MediaPipe worker; uncalibrated estimates, never another classifier'),
 'stroke.fusion':Tool('StrokeUnderstandingAgent',('perception',),'moderate','Existing audio/visual/GRU refinement in isolated output store; personal timing calibration disabled'),
 'stroke.hitnet':Tool('StrokeUnderstandingAgent',('perception',),'moderate','Third-party retrained Adapted-HitNet; not original MonoTrack author weights'),
 'stroke.swingnet':Tool('StrokeUnderstandingAgent',('perception',),'unavailable','Checkpoint installed but full upstream optical-flow preprocessing unavailable; automatic use rejected'),
 'stroke.auto':Tool('StrokeUnderstandingAgent',('perception',),'adaptive','Agent chooses compatible hit tools; at most2 models; BST classifies resulting clips'),
 'stroke.auto_full':Tool('StrokeUnderstandingAgent',('perception',),'expensive','Agent may request original visual frontend or RGB hit model within budget'),
 'stroke.gru':Tool('StrokeUnderstandingAgent',('perception',),'cheap','Published GRU with existing perception; explicit frontend adaptation'),
 'stroke.gru_original':Tool('StrokeUnderstandingAgent',('perception',),'expensive','Original FasterRCNN/HRNet + TrackNetV2 + filtered2 GRU, reviewed court required'),
 'stroke.rgb_hit':Tool('StrokeUnderstandingAgent',('perception',),'expensive','Author RGB ResNet hit detector with supplied boxes'),
 'stroke.bst':Tool('StrokeUnderstandingAgent',('perception',),'moderate','Existing geometric hit proposals and pretrained BST-0'),
 'stroke.cached':Tool('StrokeUnderstandingAgent',('perception',),'cached','Reuse existing model predictions, never human correction files'),
 'stroke.shf':Tool('StrokeUnderstandingAgent',('perception',),'moderate','Local SHF adapter; abstain if required court/features missing'),
 'stroke.check_labels':Tool('StrokeUnderstandingAgent',('stroke',),'cheap','Abstain when top raw class and aggregated coarse label disagree; consistency check, not independent truth'),
 'stroke.dsta':Tool('StrokeUnderstandingAgent',('stroke',),'expensive','DSTA adapter; explicit unavailable without validated weights'),
 'perception.verify':Tool('PerceptionAgent',('stroke',),'cheap','Check observed wrists and ball around ambiguous hits'),
 'stroke.window':Tool('StrokeUnderstandingAgent',('stroke',),'moderate','One bounded wider-context BST reclassification per selected event'),
 'technical.evidence':Tool('TechnicalTacticalAgent',('stroke',),'cheap','Existing explainable report and tactical evidence catalogue'),
 'technical.qwen':Tool('TechnicalTacticalAgent',('technical',),'expensive','Existing closed local Qwen evidence selector'),
 'expert.retrieve':Tool('ExpertDemonstrationAgent',('stroke',),'cheap','Professional structured-stroke/sequence retrieval'),
}
PROVIDES={**{k:'stroke' for k in ['stroke.auto','stroke.auto_full','stroke.gru','stroke.gru_original','stroke.rgb_hit','stroke.fusion','stroke.hitnet','stroke.swingnet']},'perception.cached':'perception','perception.racketvision':'perception','stroke.bst':'stroke','stroke.cached':'stroke','stroke.shf':'stroke','technical.evidence':'technical','expert.retrieve':'expert'}
@dataclass(frozen=True)
class Workflow:
    mode:str='FIXED'
    tools:tuple[str,...]=('perception.cached','stroke.bst','technical.evidence','expert.retrieve')
    threshold:float=.5
    bst_decoding:str='top1'
    verification_threshold:float=.65
    max_reanalyses:int=1
    retrieval_ranker:str='structured'
    top_k:int=5
    max_calls:int=10
    timeout_seconds:float=600
    def validate(self):
        if self.mode not in ['FIXED','AFLOW','EVOFLOW','AGENT']:raise ValueError('Unknown execution mode')
        if not 0<=self.threshold<=1 or not 0<=self.verification_threshold<=1 or not 0<=self.max_reanalyses<=2 or not 1<=self.top_k<=20:raise ValueError('Invalid bounded workflow parameters')
        if self.bst_decoding not in ['top1','coarse_sum']:raise ValueError('Unknown BST decoding')
        if self.retrieval_ranker not in ['type','structured','sequence']:raise ValueError('Unsupported ranker')
        if not 4<=self.max_calls<=12 or not 0<self.timeout_seconds<=1800 or not 4<=len(self.tools)<=10:raise ValueError('Workflow budget exceeded')
        completed=set();counts={}
        for name in self.tools:
            if name not in TOOLS:raise ValueError('Unregistered tool')
            spec=TOOLS[name]
            if not set(spec.requires)<=completed:raise ValueError('Invalid tool dependency order: '+name)
            if name=='perception.pose3d' and 'stroke' in completed:raise ValueError('Perception refinement must precede stroke understanding')
            if name in ['stroke.window','stroke.check_labels','stroke.dsta','perception.verify'] and ({'technical','expert'}&completed):raise ValueError('Verification/re-analysis must precede consumers')
            group=PROVIDES.get(name)
            if group:
                if group in completed:raise ValueError('No repeated primary stage/cycles')
                completed.add(group)
            counts[name]=counts.get(name,0)+1
            if counts[name]>1:raise ValueError('No unbounded repeated operations')
        if not {'perception','stroke','technical','expert'}<=completed:raise ValueError('All four agents required')
        if len(self.tools)>self.max_calls:raise ValueError('Insufficient call budget')
        return self
    def packet(self):return asdict(self.validate())
    @classmethod
    def from_packet(cls,p):
        p=dict(p);p['tools']=tuple(p['tools']);return cls(**p).validate()
def registry():return {k:asdict(v) for k,v in TOOLS.items()}
