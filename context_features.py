"""Frozen BST context and observed flight features, separate from review clips.

Screen motion is not 3-D height or a verified landing point. No pseudo labels.
"""
import numpy as np
from stroke_features import HitFeatures
from hit_refinement import effective_event

VERSION='local-hit-context-fusion-v1'


class ContextFeatures(HitFeatures):
    def extract(self,event):
        effective,suggestion=effective_event(self.dense,event,event)
        # Lock a proposed feature center without pretending it is human truth.
        event={**effective,'feature_center_locked':True}
        center=event['peak_time']
        local,quality=super().extract(event)
        lo=max(0.,center-1.)
        hi=min(self.dense['end'],center+1.)
        indices=np.flatnonzero((self.times>=lo)&(self.times<=hi))
        if not len(indices):raise ValueError('球路上下文没有解码画面')
        joints=[];shuttle=[];target_count=0;pair_count=0
        for i in indices:
            row=np.zeros((2,17,2));present=[]
            for k,side in enumerate(self.classifier.order):
                p=self.pose(i,side);present.append(p is not None)
                if p is None:continue
                pts,good,_,_,_,person=p
                normalized=self.classifier.helpers['normalize_joints'](pts[None],np.asarray([person['bbox']]),center_align=True)[0]
                normalized[~good]=0;row[k]=normalized
            target_count+=present[self.classifier.order.index(event['player'])]
            pair_count+=all(present);joints.append(row)
            shuttle.append(self.ball[i] if self.valid[i] else [0,0])
        h=self.classifier.helpers
        s=h['normalize_shuttlecock'](np.asarray(shuttle),self.dense['source_width'],self.dense['source_height'])
        j,p,s,length=h['make_seq_len_same'](100,np.asarray(joints),np.zeros((len(indices),2,2)),s)
        j=np.concatenate([j,h['create_bones'](j,self.classifier.bones)],axis=-2).reshape(100,2,72)
        values=(j.astype('float32'),p.astype('float32'),s.astype('float32'),length)
        from personal_learning import embedding
        contextual=embedding(self.classifier,values,event['player'])
        predicted=self.classifier.predict(values)
        probabilities=np.asarray(predicted['bst0_all_scores'])
        target_prefix='Top_' if event['player']==self.classifier.order[0] else 'Bottom_'
        names=predicted['class_labels']
        order=[i for i,name in enumerate(names) if name.startswith(target_prefix)]
        order += [i for i,name in enumerate(names) if i not in order]
        if len(order)!=25:raise ValueError('BST 全类输出维度改变')
        point=self.pose(int(np.argmin(abs(self.times-center))),event['player'])
        shoulder=point[2] if point else np.zeros(2);torso=point[4] if point else 50.
        other=self.pose(int(np.argmin(abs(self.times-center))),'far' if event['player']=='near' else 'near')
        facing=1 if point is None or other is None or other[3][0]>=point[3][0] else -1
        flight=[]
        for a,b in [(-1.,-.45),(-.45,0.),(0.,.45),(.45,1.)]:
            span=indices[(self.times[indices]>=center+a)&(self.times[indices]<center+b)]
            sample=span[self.valid[span]]
            coords=(np.median(self.ball[sample],axis=0)-shoulder)/torso if len(sample) else np.zeros(2)
            velocity=np.zeros(2)
            if len(sample)>=3:
                velocity=np.median(np.diff(self.ball[sample],axis=0)/np.maximum(np.diff(self.times[sample])[:,None],.01),axis=0)/torso
            flight.extend([coords[0]*facing,-coords[1],velocity[0]*facing,-velocity[1],len(sample)/max(1,len(span))])
        vector=np.r_[local,contextual,probabilities[order],np.clip(flight,-30,30)].astype('float32')
        if vector.shape!=(725,) or not np.isfinite(vector).all():raise ValueError('融合特征维度或数值异常')
        dt=float(np.median(np.diff(self.times)))
        quality.update(feature_version=VERSION,context_start=float(self.times[indices[0]]),context_end=min(self.dense['end'],float(self.times[indices[-1]]+dt)),
                       context_pose_fraction=target_count/len(indices),context_pair_fraction=pair_count/len(indices),
                       context_shuttle_fraction=float(self.valid[indices].mean()),context_bst_dimension=300,
                       context_bst_class_dimension=25,flight_dimension=20,total_dimension=725,
                       player_input_order=self.classifier.order,center_suggestion=suggestion,
                       context_class_labels=[names[i] for i in order],
                       context_normalization='author bbox diagonal center alignment; image-normalized shuttle; independent player masks; bones; seq100',
                       context_note='Up to +/-1 second source context, possibly adjacent strokes. Group holdout must cover all of this support.')
        return vector,quality


def prepare(job):
    """Extract real-video feature artifacts without fitting or changing labels."""
    from analysis_jobs import DATA,atomic_json,file_hash
    from personal_learning import extractor
    from stroke_review import load_source,read_labels
    from pathlib import Path
    from datetime import datetime,timezone
    import torch
    torch.set_num_threads(1)
    report,events,result_path=load_source(job);doc=read_labels(job,report['source_sha256'])
    classifier,_=extractor(job);features=ContextFeatures(classifier)
    ids=[];vectors=[];qualities=[]
    for original in events:
        annotation=doc['labels'].get(original['id'])
        event={**original,**annotation} if annotation and not annotation.get('wrong_target') else original
        vector,quality=features.extract(event)
        ids.append(original['id']);vectors.append(vector);qualities.append({'event_id':original['id'],**quality})
    folder=DATA/'专项分析'/job
    np.savez_compressed(folder/'context_features.npz',event_ids=np.asarray(ids),features=np.asarray(vectors))
    result={'version':VERSION,'source_sha256':report['source_sha256'],'dataset_revision':doc['revision'],
            'created_at':datetime.now(timezone.utc).isoformat(),'event_count':len(ids),'dimensions':725,
            'trained':False,'scope':'Feature extraction and author input normalization audit, not an accuracy evaluation.',
            'quality':qualities,'code_sha256':{n:file_hash(Path(__file__).with_name(n)) for n in
                  ['context_features.py','stroke_features.py','specialist_bst.py']},
            'input_sha256':{n:file_hash(folder/n) for n in ['dense_pose.json','trajectory.json','hit_refinement.json']},
            'candidate_results_sha256':file_hash(result_path),
            'review_clips_unchanged':True,'pretrained_weights_frozen':True,'human_labels_created':0}
    atomic_json(folder/'context_features.json',result)
    print('CONTEXT_FEATURES',job,len(ids),725,'trained=False',flush=True)
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True)
    prepare(parser.parse_args().job)
