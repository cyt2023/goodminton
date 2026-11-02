"""Read saved logits only: distinguish player selection from stroke classification."""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from analysis_jobs import atomic_json,file_hash
from specialist_bst import author_helpers
from specialist_model import ROOT

DATA=ROOT.parent/'03_数据'
RUN=DATA/'击球标注/直接BST微调/运行记录/582817bff5454825b841d1f150c82028'
BASE=DATA/'击球标注/根因诊断'
OUT=BASE/'2026-10-18_人物条件与球种分离'
SOURCE=BASE/'2026-10-15_同源前端对照'
SELECTION=BASE/'2026-10-18_普通视角配对诊断/selection_protocol.json'


def order_for(dense):
    medians={side:float(np.median([np.mean([p['keypoints'][j]['y'] for j in [15,16]])
                for f in dense['frames'] if (p:=f['players'].get(side)) is not None])) for side in ['near','far']}
    return sorted(medians,key=medians.get)


def conditioned(logits,actor,classes):
    allowed=np.asarray([i for i,c in enumerate(classes) if c.startswith(actor+'_')])
    assert len(allowed)==12
    return int(allowed[np.argmax(logits[allowed])])


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'result.json').exists():raise FileExistsError('Preserve completed saved-logit audit')
    inputs=[RUN/'personal_predictions.json',RUN/'personal_predictions.npz',SELECTION,
        SOURCE/'automatic_events/classification_logits.npz',SOURCE/'automatic_events/original_audio_production_candidates.json',
        SOURCE/'automatic_events/original_audio_result.json',SOURCE/'continuous_rally/pose/dense_pose.json',
        SOURCE/'clips_manifest.json',Path(__file__)]
    selection=json.loads(SELECTION.read_text())
    for s in selection['per_source']:
        inputs.append(DATA/'专项分析'/s['job_id']/'dense_pose.json')
    hashes={str(p):file_hash(p) for p in inputs}
    protocol={'declared_at':datetime.now(timezone.utc).isoformat(),'inputs':hashes,
        'no_model_inference_or_training':True,
        'conditions':['unconditioned_joint25_argmax','original_automatic_candidate_actor_conditioned','human_reviewed_actor_conditioned'],
        'method':'Restrict the same saved25 logits to12classes of givenactor thenargmax. Uses actor only, never true stroke kind. Unchanged source/order and weights/logits; no threshold tuning.',
        'ordinary_split':'Exactly30 saveddevelopment_holdout_mask rows; firstcandidate41 train rows excluded.',
        'automatic_actor_basis':'Original automatic event.player before user actor correction, mapped through unchanged whole-video median ankle-y order.',
        'human_actor_scope':'Conditional task with reviewed person known, not fully automatic accuracy. May change task definition and confidence calibration; not activated.',
        'scope':'Repeatedly inspected developer records plus one author-val rally. No general accuracy claim.'}
    atomic_json(OUT/'protocol.json',protocol)
    classes=author_helpers()['get_merged_stroke_types']()
    records=json.loads((RUN/'personal_predictions.json').read_text())['baseline']
    cache=np.load(RUN/'personal_predictions.npz');ids=np.flatnonzero(cache['development_holdout_mask'])
    assert len(ids)==30 and len(records)==71 and int(cache['training_mask'].sum())==41
    mapped={(r['job_id'],r['event_id']):r for r in selection['all_30_mapped_rows']}
    orders={s['job_id']:order_for(json.loads((DATA/'专项分析'/s['job_id']/'dense_pose.json').read_text())) for s in selection['per_source']}
    ordinary={}
    for method,key in [('baseline','baseline_logits'),('candidate_head','candidate_logits')]:
        rows=[]
        for i in ids:
            r=records[i];s=mapped[(r['job_id'],r['event_id'])];scores=cache[key][i];label=int(cache['labels'][i])
            assert r['target_class_label']==classes[label]
            true_actor=classes[label].split('_',1)[0]
            original_slot=s['original_automatic_window_for_provenance_only']['player']
            auto_actor='Top' if original_slot==orders[r['job_id']][0] else 'Bottom'
            p=int(scores.argmax());pa=conditioned(scores,auto_actor,classes);ph=conditioned(scores,true_actor,classes)
            rows.append({'job_id':r['job_id'],'event_id':r['event_id'],'true_label':classes[label],
                'human_actor':true_actor,'original_automatic_actor':auto_actor,'automatic_actor_correct':auto_actor==true_actor,
                'unconditioned':classes[p],'unconditioned_joint_correct':p==label,
                'unconditioned_stroke_only_correct':classes[p].split('_',1)[-1]==classes[label].split('_',1)[-1],
                'automatic_actor_conditioned':classes[pa],'automatic_actor_conditioned_joint_correct':pa==label,
                'human_actor_conditioned':classes[ph],'human_actor_conditioned_joint_correct':ph==label})
        ordinary[method]={'total':30,'unconditioned_joint_correct':sum(r['unconditioned_joint_correct'] for r in rows),
            'unconditioned_stroke_only_correct':sum(r['unconditioned_stroke_only_correct'] for r in rows),
            'automatic_actor_correct':sum(r['automatic_actor_correct'] for r in rows),
            'automatic_actor_conditioned_joint_correct':sum(r['automatic_actor_conditioned_joint_correct'] for r in rows),
            'human_actor_conditioned_joint_correct':sum(r['human_actor_conditioned_joint_correct'] for r in rows),'rows':rows}
    a,b=ordinary['baseline']['rows'],ordinary['candidate_head']['rows']
    learning={'human_actor_conditioned_head_gains':[r['event_id'] for r,s in zip(a,b) if not r['human_actor_conditioned_joint_correct'] and s['human_actor_conditioned_joint_correct']],
              'human_actor_conditioned_head_losses':[r['event_id'] for r,s in zip(a,b) if r['human_actor_conditioned_joint_correct'] and not s['human_actor_conditioned_joint_correct']]}
    logits=np.load(SOURCE/'automatic_events/classification_logits.npz')
    events=json.loads((SOURCE/'automatic_events/original_audio_production_candidates.json').read_text())['candidates']
    evaluation=json.loads((SOURCE/'automatic_events/original_audio_result.json').read_text())['production_refined']
    lookup={r['candidate_id']:r for r in evaluation['matches']}
    truth={r['sample_id']:r for r in json.loads((SOURCE/'clips_manifest.json').read_text())['clips']}
    public_order=order_for(json.loads((SOURCE/'continuous_rally/pose/dense_pose.json').read_text()))
    public=[]
    for e in events:
        scores=logits['original_audio__'+e['id']+'__automatic_window'];match=lookup.get(e['id'])
        actor='Top' if e['player']==public_order[0] else 'Bottom'
        p=int(scores.argmax());pc=conditioned(scores,actor,classes)
        label=truth[match['sample_id']]['author_25class'] if match else None
        public.append({'candidate_id':e['id'],'matched_sample_id':match['sample_id'] if match else None,
            'automatic_actor':actor,'true_label':label,'unconditioned':classes[p],
            'unconditioned_correct':bool(label and classes[p]==label),'actor_conditioned':classes[pc],
            'actor_conditioned_correct':bool(label and classes[pc]==label)})
    pub={'author_contacts':20,'candidates':len(events),'time_matched':len(lookup),
         'unconditioned_correct_out_of20':sum(r['unconditioned_correct'] for r in public),
         'candidate_actor_conditioned_correct_out_of20':sum(r['actor_conditioned_correct'] for r in public),'rows':public}
    assert all(file_hash(Path(p))==h for p,h in hashes.items())
    result={'created_at':datetime.now(timezone.utc).isoformat(),'protocol_sha256':file_hash(OUT/'protocol.json'),
        'ordinary':ordinary,'learning':learning,'public_automatic':pub,
        'production_or_labels_or_weights_changed':False,'scope':protocol['scope']}
    atomic_json(OUT/'result.json',result)
    print(json.dumps({'ordinary':{n:{k:v for k,v in r.items() if k!='rows'} for n,r in ordinary.items()},
                      'learning':learning,'public':{k:v for k,v in pub.items() if k!='rows'}},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
