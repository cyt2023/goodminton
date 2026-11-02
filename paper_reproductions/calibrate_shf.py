"""Freeze one global event threshold on validation matches; then test once."""
import argparse,json
from pathlib import Path
from dataclasses import replace
import torch
from torch import nn
from .run import load_model,checkpoint,shf_evaluate
from .data import write_json,sha256
from .shuttlehitformer import Config

class FixedGrid(nn.Module):
    """No video features: constant scores yield one candidate every8 frames."""
    def __init__(self):super().__init__();self.config=Config()
    def forward(self,x):
        logits=x.new_zeros((len(x),3));logits[:,1]=20.
        return {'logits':logits}

def event_only(metrics):return {k:metrics[k] for k in ['tp','fp','fn','precision','recall','f1','tolerance_frames']}

def main():
    a=argparse.ArgumentParser();a.add_argument('--checkpoint',required=True);a.add_argument('--manifest',required=True);a.add_argument('--amateur-manifest');a.add_argument('--out',required=True);args=a.parse_args()
    torch.set_num_threads(4);manifest=json.loads(Path(args.manifest).read_text())
    from benchmarks.isolation import assert_training_allowed
    assert_training_allowed(manifest,args.manifest)
    model,cp=load_model(args.checkpoint)
    validation=[r for r in manifest['rows'] if r['split']=='val'];candidates=[]
    for threshold in [.5,.55,.6,.65,.7,.75,.8,.85,.9]:
        model.config=replace(model.config,threshold=threshold);metrics=shf_evaluate(model,validation,'cpu')
        row={'threshold':threshold,'metrics':event_only(metrics)};candidates.append(row);print(json.dumps(row),flush=True)
    chosen=max(candidates,key=lambda r:(r['metrics']['f1'],r['metrics']['precision'],-r['threshold']))
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    selection={'source_checkpoint_sha256':sha256(args.checkpoint),'validation_manifest_sha256':sha256(args.manifest),
               'candidates':candidates,'selected':chosen,'criterion':'validation_event_F1, then precision; never test scores',
               'held_out_test_used_for_threshold_selection':False}
    # Freeze selection on disk BEFORE any test evaluation.
    write_json(out/'threshold_selection.json',selection)
    cp['config']['threshold']=chosen['threshold'];cp['scope']+='+validation_threshold';cp['threshold_selection']=selection
    training_result=Path(args.checkpoint).parent/'result.json'
    if training_result.exists():
        run=json.loads(training_result.read_text())
        if run['checkpoint_sha256']!=selection['source_checkpoint_sha256']:raise ValueError('Training result belongs to a different checkpoint')
        cp['training_metadata']=run['training_settings'];cp['training_result_sha256']=sha256(training_result)
        if cp['training_metadata'].get('warm_start'):
            cp['training_metadata']['warm_start_sha256']=sha256(cp['training_metadata']['warm_start'])
    cp['required_modalities']=['player_pose','court','shuttle']+(['net'] if cp['config'].get('use_net',True) else [])
    torch.save(cp,out/'calibrated.pt');model.config=replace(model.config,threshold=chosen['threshold'])
    groups={'professional_feature_test':[r for r in manifest['rows'] if r['split']=='test']}
    if args.amateur_manifest:groups['amateur_feature_transfer']=json.loads(Path(args.amateur_manifest).read_text())['rows']
    results={}
    for group,rows in groups.items():
        trained=shf_evaluate(model,rows,'cpu');grid=shf_evaluate(FixedGrid(),rows,'cpu')
        results[group]={'model':trained,'fixed_grid_baseline':event_only(grid),'rallies':len(rows),'matches':sorted({r['match'] for r in rows})}
        print('TEST',group,json.dumps(event_only(trained)),flush=True)
    write_json(out/'result.json',{'selection':selection,'groups':results,'checkpoint_sha256':sha256(out/'calibrated.pt'),
        'deployment_ready':False,'scope':'development evaluation on published feature data; not raw-video frontend performance',
        'limitations':['No net modality; residual modifies paper architecture','Public feature inputs include manual/human-associated measurements',
                       'These source collections were previously used in project development; this is not an untouched new benchmark',
                       'Fixed-grid baseline uses no visual information and must be beaten before claiming useful detection']})

if __name__=='__main__':main()
