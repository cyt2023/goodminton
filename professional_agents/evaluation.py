"""Conditional cached-feature evaluation, separate from raw-video benchmark."""
import time,resource
import numpy as np
from .state import read,sha,STORE
from .datasets import check_development
from .retrieval import mainland

def macro_metrics(truth,pred):
    labels=sorted(set(truth));per={};correct=0
    for label in labels:
        tp=sum(a==b==label for a,b in zip(truth,pred));fp=sum(b==label and a!=label for a,b in zip(truth,pred));fn=sum(a==label and b!=label for a,b in zip(truth,pred));correct+=tp
        per[label]={'tp':tp,'fp':fp,'fn':fn,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}
    return {'status':'evaluated_conditional_features','samples':len(truth),'macro_f1':sum(v['f1'] for v in per.values())/len(labels) if labels else None,'accuracy':correct/len(truth) if truth else None,'coverage':sum(v is not None for v in pred)/len(truth) if truth else None,'per_class':per}

def decode(logits,classes,workflow):
    # Ground truth labels are not an argument to this prediction function.
    x=logits-logits.max(axis=1,keepdims=True);prob=np.exp(x);prob/=prob.sum(axis=1,keepdims=True);labels=sorted({mainland(c) for c in classes});coarse=np.stack([prob[:,[j for j,c in enumerate(classes) if mainland(c)==k]].sum(1) for k in labels],1)
    raw=np.argmax(prob,1);agg=np.argmax(coarse,1);raw_label=[mainland(classes[j]) for j in raw];aggregate=[labels[j] for j in agg]
    selected=raw_label if workflow.bst_decoding=='top1' else aggregate;scores=prob.max(1) if workflow.bst_decoding=='top1' else coarse.max(1)
    verify='stroke.check_labels' in workflow.tools
    predictions=[None if label=='unknown' or score<workflow.threshold or (verify and a!=b) else label for label,score,a,b in zip(selected,scores,raw_label,aggregate)]
    return predictions,{'classification_decisions':len(predictions),'consistency_checks':len(predictions) if verify else 0,'prediction_inputs':'cached model logits and class definitions only','model_calls_executed':0,'cache_reads':1,'local_reanalysis':'unavailable: raw windows are not in this feature cache'}

class ReplayEvaluator:
    def __init__(self,manifest):
        self.manifest=manifest;f=manifest['feature_development']
        if sha(f['features'])!=f['features_sha256'] or sha(f['source_manifest'])!=f['source_manifest_sha256']:raise ValueError('Cached feature assets changed')
        with np.load(f['features'],allow_pickle=False) as z:self.logits=z['logits'];self.truth=z['labels'];self.classes=z['class_labels'].tolist();self.ids=z['sample_ids'].tolist()
        self.roles=[];checked=set()
        for sample in self.ids:
            m=str(sample).split('_')[0];r={'domain':'professional','role':f['match_roles'][m],**f['identities'][m]};
            if m not in checked:check_development(r);checked.add(m)
            self.roles.append(r['role'])
    def evaluate(self,workflow,role):
        if role not in ['development','validation']:raise ValueError('Sealed test cannot enter workflow replay search')
        workflow.validate()
        if any(t in workflow.tools for t in ['stroke.auto','stroke.auto_full','stroke.gru','stroke.gru_original','stroke.rgb_hit','stroke.shf','stroke.fusion','stroke.hitnet','stroke.swingnet']):raise ValueError('This cached BST-logit evaluator cannot measure hit-tool selection; raw hit outputs and independent development truth required')
        indices=[i for i,r in enumerate(self.roles) if r==role and mainland(self.classes[int(self.truth[i])])!='unknown']
        t=time.perf_counter();cpu=time.process_time();pred,cost=decode(self.logits[indices],self.classes,workflow);truth=[mainland(self.classes[int(self.truth[i])]) for i in indices]
        return {'scope':'conditional stroke classification with dataset-defined hit windows; not end-to-end','role':role,'metrics':macro_metrics(truth,pred),'cost':{**cost,'replay_wall_seconds':time.perf_counter()-t,'replay_cpu_seconds':time.process_time()-cpu,'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'rss_unit':'bytes on macOS','gpu_seconds':None,'end_to_end_model_latency':None,'source_inference_cost':'unavailable; cached replay is not a zero-cost model'},'missing_metrics':{k:{'status':'unavailable','value':None,'reason':reason} for k,reason in {'hit_PRF':'feature inputs already use labelled hit windows','timestamp_MAE':'no predicted contact timestamps in the feature cache','end_to_end_stroke_F1':'requires raw-video detection outputs','rally_segmentation':'sparse isolated stroke windows','retrieval_Recall_at_K':'no reviewed relevance judgements','retrieval_nDCG':'no reviewed graded relevance','analysis_quality':'no reviewed analysis references'}.items()}}
