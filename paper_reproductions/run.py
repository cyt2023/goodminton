"""Run with .venv-vision/bin/python -m paper_reproductions.run --help."""
import argparse,json,time,random
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from .data import sha256,write_json,prepare_author_shf,prepare_local_dsta,read_rgb_window


def seed(value):
    random.seed(value);np.random.seed(value);torch.manual_seed(value);torch.set_num_threads(4)

def checkpoint(path):
    value=torch.load(path,map_location='cpu',weights_only=False)
    if value.get('format')!='badminton-paper-reproduction-v1':raise ValueError('Unsupported local checkpoint format')
    return value

def load_model(path,device='cpu'):
    p=checkpoint(path)
    if p['component']=='shuttlehitformer':
        from .shuttlehitformer import Config,ShuttleHitFormer
        model=ShuttleHitFormer(Config(**p['config']));model.load_state_dict(p['state_dict'],strict=True)
    elif p['component']=='dsta':
        from .dsta import Config,DSTALocalizer
        foundation=Path(p['foundation']['path'])
        if sha256(foundation)!=p['foundation']['sha256']:raise ValueError('Foundation checkpoint SHA changed')
        model=DSTALocalizer(Config(**p['config']),foundation)
        expected={n for n,param in model.named_parameters() if param.requires_grad}
        if set(p['state_dict'])!=expected:raise ValueError('Incomplete DSTA adapter/head checkpoint')
        state=model.state_dict();state.update(p['state_dict']);model.load_state_dict(state,strict=True)
    else:raise ValueError('Unknown model component')
    return model.to(device),p

def save_model(path,model,manifest,metrics,optimizer,epoch):
    component=manifest['component'];trainable={n for n,p in model.named_parameters() if p.requires_grad}
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if component!='dsta' or k in trainable}
    foundation=getattr(model,'foundation_load',None)
    if foundation:foundation={k:v for k,v in foundation.items() if k!='missing'}
    torch.save({'format':'badminton-paper-reproduction-v1','component':component,'config':asdict(model.config),
                'state_dict':state,'foundation':foundation,'optimizer':optimizer.state_dict(),'epoch':epoch,
                'training_metadata':getattr(model,'training_metadata',None),
                'scope':manifest['scope'],'deployment_ready':False,'labels':manifest['labels'],
                'limitations':manifest['limitations'],'metrics':metrics,'data_manifest_sha256':sha256(manifest['_path']),
                'code_sha256':{f.name:sha256(f) for f in Path(__file__).resolve().parent.glob('*.py')},
                'missing_modalities':sorted({x for r in manifest['rows'] for x in r.get('missing_modalities',[])}),
                'required_modalities':(['player_pose','court','shuttle']+(['net'] if model.config.use_net else [])) if component=='shuttlehitformer' else ['rgb']},path)

def shf_evaluate(model,rows,device):
    from .shuttlehitformer import infer_rally,center_windows
    from scipy.optimize import linear_sum_assignment
    tp=fp=fn=actor_correct=0;details=[];class_sums=np.zeros(3);class_counts=np.zeros(3);confusion=np.zeros((3,3),dtype=int);hit_probability_sum=0.;hit_count=0
    with torch.no_grad():
        for row in rows:
            with np.load(row['path']) as z:features=torch.from_numpy(z['features']).to(device);labels=z['labels']
            result=infer_rally(model,features);pred=result['events'];gt=np.flatnonzero(labels>0)
            from .shf_training import frame_diagnostics
            diagnostic=frame_diagnostics(result['frame_scores'],labels)
            confusion+=np.array(diagnostic['frame_confusion'])
            for item in diagnostic.get('per_class_loss',[]):
                class_sums[item['class_id']]+=item['negative_log_likelihood_sum'];class_counts[item['class_id']]+=item['count']
            hit_probability_sum+=diagnostic.get('hit_probability_sum',0.);hit_count+=diagnostic.get('hit_frame_count',0)
            pairs=[]
            if len(pred) and len(gt):
                distance=np.abs(np.array([p['frame_index'] for p in pred])[:,None]-gt[None,:])
                cost=np.where(distance<=3,distance,1e6);i,j=linear_sum_assignment(cost)
                pairs=[(a,b) for a,b in zip(i,j) if distance[a,b]<=3]
            correct=sum(pred[a]['class_id']==labels[gt[b]] for a,b in pairs)
            tp+=len(pairs);fp+=len(pred)-len(pairs);fn+=len(gt)-len(pairs);actor_correct+=correct
            details.append({'id':row['id'],'hits_gt':len(gt),'hits_pred':len(pred),'matched':len(pairs)})
    precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/(tp+fn) if tp+fn else 0.
    return {'tp':tp,'fp':fp,'fn':fn,'precision':precision,'recall':recall,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
            'actor_accuracy_on_matches':actor_correct/tp if tp else None,'tolerance_frames':3,'rows':details,
            'balanced_cross_entropy':float((class_sums[class_counts>0]/class_counts[class_counts>0]).mean()) if bool((class_counts>0).any()) else None,
            'frame_confusion':confusion.tolist(),'mean_hit_probability_on_hits':hit_probability_sum/hit_count if hit_count else None}

def train_shf(args,manifest):
    from benchmarks.isolation import assert_training_allowed
    assert_training_allowed(manifest,args.manifest)
    from .shuttlehitformer import Config,ShuttleHitFormer
    from .shf_training import load_rallies,sample_epoch,selection_key
    config=Config(**json.loads(Path(args.config).read_text())) if args.config else Config()
    limitations=list(manifest['limitations']);scope=manifest['scope']
    if config.grounding_residual:
        scope+='+local_grounding_residual';limitations.append('Explicit residual modifies Eq.20; this variant is not the paper architecture')
    if not config.use_net:
        scope+='+court_only_context';limitations.append('Net modality is explicitly removed from model; no net coordinates are inferred or invented')
    if config.stream_residual:
        scope+='+aligned_stream_residual';limitations.append('Aligned player/shuttle residuals preserve temporal motion; local modification, not the paper architecture')
    manifest={**manifest,'scope':scope,'limitations':limitations}
    model=ShuttleHitFormer(config).to(args.device)
    if args.warm_start:
        previous,cp=load_model(args.warm_start,args.device)
        # Explicit warm-start may change residual routes/thresholds, preserving
        # learned tensors. Strict state loading still rejects different shapes.
        model.load_state_dict(previous.state_dict());del previous
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=5e-4)
    rallies=load_rallies([r for r in manifest['rows'] if r['split']=='train']);rng=np.random.default_rng(args.seed)
    x,y=sample_epoch(rallies,config,rng);train_count=len(x);counts=torch.bincount(y,minlength=3)
    if bool((counts==0).any()):raise ValueError('Training split must contain all three classes')
    weights=(len(y)/(3*counts.float())).to(args.device) if args.class_balance else None
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=1e-6)
    history=[];best=None;stale=0;updates=0;start=time.perf_counter();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    val=[r for r in manifest['rows'] if r['split']=='val']
    if not val:raise ValueError('Separate validation matches required')
    training_settings={'selection':args.selection,'class_balance':args.class_balance,'class_weights':weights.cpu().tolist() if weights is not None else None,
        'min_epochs':args.min_epochs,'min_updates':args.min_updates,'patience':args.patience,'negative_resampling':'each_epoch',
        'lr':args.lr,'batch':args.batch,'seed':args.seed,'warm_start':args.warm_start,
        'warm_start_sha256':sha256(args.warm_start) if args.warm_start else None}
    model.training_metadata=training_settings
    for epoch in range(args.epochs):
        if epoch:x,y=sample_epoch(rallies,config,rng)
        model.train();losses=[]
        for order in torch.randperm(len(x)).split(args.batch):
            optimizer.zero_grad(set_to_none=True);output=model(x[order].to(args.device));loss=model.loss(output,y[order].to(args.device),weights)['loss']
            if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite SHF training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();updates+=1;losses.append(float(loss.detach()))
        scheduler.step();metrics=shf_evaluate(model,val,args.device);key=selection_key(metrics,args.selection)
        row={'epoch':epoch+1,'updates':updates,'train_loss':sum(losses)/len(losses),'val':metrics};history.append(row)
        print(json.dumps({k:v for k,v in row.items() if k!='val'}|{'val_f1':metrics['f1'],'val_recall':metrics['recall'],'val_balanced_ce':metrics['balanced_cross_entropy']}),flush=True)
        if best is None or key>best:
            best=key;stale=0;save_model(out/'best.pt',model,manifest,metrics,optimizer,epoch+1)
        else:stale+=1
        write_json(out/'progress.json',{'epoch':epoch+1,'updates':updates,'last':row,'selection':args.selection})
        if epoch+1>=args.min_epochs and updates>=args.min_updates and stale>=args.patience:break
    save_model(out/'last.pt',model,manifest,metrics,optimizer,epoch+1)
    model,cp=load_model(out/'best.pt',args.device)
    test=shf_evaluate(model,[r for r in manifest['rows'] if r['split']=='test'],args.device)
    result={'component':'shuttlehitformer','scope':manifest['scope'],'epochs_completed':len(history),'updates':updates,'selected_epoch':cp['epoch'],
            'train_windows_per_epoch':train_count,'train_labels':counts.tolist(),'history':history,'test':test,'selected_validation':cp['metrics'],
            'seconds':time.perf_counter()-start,'device':args.device,'checkpoint_sha256':sha256(out/'best.pt'),
            'training_settings':training_settings,'deployment_ready':False,'paper_accuracy_reproduced':False,'limitations':manifest['limitations']}
    write_json(out/'result.json',result);print('RESULT',out/'result.json',flush=True)

def dsta_sample(row,device,model):
    with np.load(row['path']) as z:
        rgb=torch.from_numpy(z['rgb']).unsqueeze(0).to(device)
        segments=torch.from_numpy(z['segments']).to(device);labels=torch.from_numpy(z['labels']).long().to(device)
    c=model.config
    return rgb,[{'video_id':row['id'],'fps':float(c.fps),'duration':c.window_frames*c.sample_stride/c.fps,
                 'feat_stride':c.window_frames*c.sample_stride/c.feature_length,'feat_num_frames':0.,'segments':segments,'labels':labels}]

def train_dsta(args,manifest):
    from benchmarks.isolation import assert_training_allowed
    assert_training_allowed(manifest,args.manifest)
    from .dsta import Config,DSTALocalizer
    value=json.loads(Path(args.config).read_text()) if args.config else {}
    value.update(manifest.get('config_overrides',{}));c=Config(**value)
    model=DSTALocalizer(c,Path(args.foundation).resolve()).to(args.device)
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=args.lr,weight_decay=.05)
    train=[r for r in manifest['rows'] if r['split']=='train'];val=[r for r in manifest['rows'] if r['split']=='val']
    if not train or not val:raise ValueError('Separate RGB train and val sources required')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);history=[];start=time.perf_counter()
    frozen_before=sha256(args.foundation);adapter_before=model.backbone.blocks[0].adapter.up_proj.weight.detach().clone()
    for epoch in range(args.epochs):
        for row in train:
            model.train();rgb,meta=dsta_sample(row,args.device,model);optimizer.zero_grad(set_to_none=True)
            losses=model(rgb,meta);losses['final_loss'].backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):raise ValueError('Nonfinite train gradient')
            count=sum(p.grad is not None and bool(p.grad.abs().sum()>0) for n,p in model.named_parameters() if '.adapter.' in n)
            if count==0:raise RuntimeError('RGB gradient did not reach DSTA adapters')
            torch.nn.utils.clip_grad_norm_(parameters,5);optimizer.step()
            item={'epoch':epoch+1,'id':row['id'],'loss':float(losses['final_loss'].detach()),'adapter_tensors_with_nonzero_gradient':count}
            history.append(item);print(json.dumps(item),flush=True)
    # Validation loss still requires training-mode head; disable stochastic blocks.
    model.eval();model.head.train();validation=[]
    with torch.no_grad():
        for row in val:
            rgb,meta=dsta_sample(row,args.device,model);losses=model(rgb,meta)
            validation.append({'id':row['id'],'loss':float(losses['final_loss'])})
    metrics={'validation_loss_only':validation,'accuracy':None,'mAP':None}
    save_model(out/'last.pt',model,manifest,metrics,optimizer,args.epochs)
    model.eval();rgb,meta=dsta_sample(val[0],args.device,model)
    with torch.no_grad():
        features_before=model.encode(rgb).cpu();prediction=model(rgb,meta)[0]
    change=float((model.backbone.blocks[0].adapter.up_proj.weight-adapter_before).detach().abs().max())
    del model,optimizer,parameters,losses,rgb
    loaded,cp=load_model(out/'last.pt',args.device);loaded.eval();rgb,meta=dsta_sample(val[0],args.device,loaded)
    with torch.no_grad():features_after=loaded.encode(rgb).cpu()
    difference=float((features_after-features_before).abs().max())
    if difference>1e-5:raise RuntimeError('Saved checkpoint forward does not match trained model')
    if sha256(args.foundation)!=frozen_before:raise RuntimeError('Foundation file modified')
    result={'component':'dsta','scope':manifest['scope'],'history':history,'metrics':metrics,
            'seconds':time.perf_counter()-start,'device':args.device,'trainable_parameters':sum(p.numel() for p in loaded.parameters() if p.requires_grad),
            'adapter_first_up_weight_max_change':change,'reload_feature_max_abs_difference':difference,
            'held_out_source_inference_intervals':len(prediction['scores']),'foundation':loaded.foundation_load,
            'checkpoint_sha256':sha256(out/'last.pt'),'deployment_ready':False,'paper_accuracy_reproduced':False,
            'limitations':manifest['limitations']}
    write_json(out/'result.json',result);print('RESULT',out/'result.json',flush=True)

def inference(args):
    model,p=load_model(args.checkpoint,args.device)
    if not args.experimental and not p['deployment_ready']:raise ValueError('Pilot model: pass --experimental; do not report production accuracy')
    if p['component']=='shuttlehitformer':
        from .shuttlehitformer import infer_rally
        if Path(args.input).suffix=='.json':
            from .data import pack_shf
            source=json.loads(Path(args.input).read_text())
            if source.get('net') is None:
                if model.config.use_net:raise ValueError('This checkpoint requires measured net points; use an explicit court-only checkpoint when unavailable')
                source['net']=np.zeros((len(source['top']),4,2),np.float32)
            features=torch.from_numpy(pack_shf(**source)).to(args.device)
        else:
            with np.load(args.input) as z:features=torch.from_numpy(z['features']).to(args.device)
        prediction=infer_rally(model,features)
    else:
        c=model.config;rgb,clock=read_rgb_window(args.input,args.start,c.window_frames,fps=c.fps,size=c.input_size,sample_stride=c.sample_stride)
        meta=[{'video_id':str(args.input),'fps':float(c.fps),'duration':c.window_frames*c.sample_stride/c.fps,
               'feat_stride':c.window_frames*c.sample_stride/c.feature_length,'feat_num_frames':0.}]
        model.eval()
        with torch.no_grad():r=model(rgb.unsqueeze(0).to(args.device),meta)[0]
        prediction={'intervals':[{'start':float(s[0])+args.start,'end':float(s[1])+args.start,'confidence':float(v),
                    'label':p['labels'][int(l)]} for s,v,l in zip(r['segments'],r['scores'],r['labels'])],'clock':clock}
    write_json(args.out,{'component':p['component'],'scope':p['scope'],'deployment_ready':p['deployment_ready'],
                         'checkpoint_sha256':sha256(args.checkpoint),'source_sha256':sha256(args.input),
                         'limitations':p['limitations'],'model_config':p['config'],
                         'required_modalities':p.get('required_modalities'),'prediction':prediction})

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare-shf');a.add_argument('--out',required=True);a.add_argument('--max-rallies',type=int,default=0);a.add_argument('--domain',choices=['domain_pro','domain_am_singles'],default='domain_pro')
    a=sub.add_parser('prepare-dsta-smoke');a.add_argument('--out',required=True);a.add_argument('--per-split',type=int,default=4)
    for component in ['shf','dsta']:
        a=sub.add_parser('train-'+component);a.add_argument('--manifest',required=True);a.add_argument('--out',required=True)
        a.add_argument('--epochs',type=int,default=100 if component=='shf' else 1);a.add_argument('--batch',type=int,default=64)
        a.add_argument('--lr',type=float,default=1e-4);a.add_argument('--device',default='cpu');a.add_argument('--seed',type=int,default=20261029)
        a.add_argument('--selection',choices=['f1','paper_recall'],default='f1');a.add_argument('--class-balance',action='store_true')
        a.add_argument('--min-epochs',type=int,default=20);a.add_argument('--min-updates',type=int,default=300);a.add_argument('--patience',type=int,default=15);a.add_argument('--warm-start')
        a.add_argument('--config');a.add_argument('--foundation',default=str(Path(__file__).resolve().parents[1]/'models/paper_reproductions/videomae_b_k400.pth'))
    a=sub.add_parser('infer');a.add_argument('--checkpoint',required=True);a.add_argument('--input',required=True);a.add_argument('--out',required=True)
    a.add_argument('--device',default='cpu');a.add_argument('--start',type=float,default=0.);a.add_argument('--experimental',action='store_true')
    args=parser.parse_args()
    if args.command=='prepare-shf':print(json.dumps(prepare_author_shf(args.out,args.max_rallies,args.domain),ensure_ascii=False))
    elif args.command=='prepare-dsta-smoke':print(json.dumps(prepare_local_dsta(args.out,args.per_split),ensure_ascii=False))
    elif args.command=='infer':inference(args)
    else:
        seed(args.seed);manifest=json.loads(Path(args.manifest).read_text());manifest['_path']=args.manifest
        (train_shf if args.command=='train-shf' else train_dsta)(args,manifest)

if __name__=='__main__':main()
