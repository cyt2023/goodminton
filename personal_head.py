"""Small-sample head selection using training-only temporal group folds."""
import numpy as np

CONFIGS = [
    {'features':'geometry', 'kernel':'linear', 'alpha':10.},
    {'features':'geometry', 'kernel':'linear', 'alpha':100.},
    {'features':'geometry', 'kernel':'rbf', 'alpha':.1},
    {'features':'local_bst', 'kernel':'linear', 'alpha':100.},
    {'features':'local_bst', 'kernel':'rbf', 'alpha':.1},
    {'features':'combined', 'kernel':'linear', 'alpha':100.},
    {'features':'combined', 'kernel':'rbf', 'alpha':.1},
]

CONTEXT_CONFIGS=CONFIGS+[
    {'features':'context_bst','kernel':'rbf','alpha':.1},
    {'features':'flight_geometry','kernel':'rbf','alpha':.1},
    {'features':'context_fusion','kernel':'linear','alpha':100.},
    {'features':'context_fusion','kernel':'rbf','alpha':.1},
]

def configs_for_width(width):
    return CONTEXT_CONFIGS if width==725 else CONFIGS


def columns(config, width):
    if config['features']=='geometry':return np.arange(300,min(width,380))
    if config['features']=='local_bst':return np.arange(300)
    if config['features']=='combined':return np.arange(min(width,380))
    if config['features']=='context_bst':return np.arange(380,680)
    if config['features']=='flight_geometry':return np.r_[np.arange(300,380),np.arange(705,725)]
    return np.arange(width)


def normalized(x, model):
    value=np.clip((x[:,model['columns']]-model['mean'])/model['scale'],-5,5)
    return value*model['block_scale']


def kernel(a,b,kind):
    if kind=='linear':return a@b.T+1
    distance=np.maximum(0,(a*a).sum(1)[:,None]+(b*b).sum(1)[None,:]-2*a@b.T)
    return np.exp(-distance/2)


def fit(x,targets,config):
    ix=columns(config,x.shape[1]);values=x[:,ix]
    mean=values.mean(0);scale=values.std(0);scale=np.where(scale<1e-3,1,scale)
    blocks=np.ones(len(ix))/np.sqrt(max(1,len(ix)))
    if config['features']=='combined':
        blocks=np.where(ix<300,1/np.sqrt(600),1/np.sqrt(160))
    if config['features']=='context_fusion':
        for lo,hi in [(0,300),(300,380),(380,680),(680,705),(705,725)]:
            blocks[(ix>=lo)&(ix<hi)]=1/np.sqrt(5*(hi-lo))
    model={'config':dict(config),'columns':ix,'mean':mean,'scale':scale,'block_scale':blocks}
    z=normalized(x,model);classes=np.asarray(sorted(set(targets)))
    y=np.eye(len(classes))[[list(classes).index(t) for t in targets]]
    weights=np.linalg.solve(kernel(z,z,config['kernel'])+config['alpha']*np.eye(len(z)),y)
    return {**model,'train_features':z,'weights':weights,'classes':classes}


def scores(model,x):
    return kernel(normalized(x,model),model['train_features'],model['config']['kernel'])@model['weights']


def select(x,targets,groups):
    targets=np.asarray(targets);groups=np.asarray(groups)
    results=[]
    configs=configs_for_width(x.shape[1])
    for config in configs:
        correct,total=0,0
        for group in sorted(set(groups)):
            test=groups==group;train=~test
            if not train.any():continue
            model=fit(x[train],targets[train],config)
            guesses=model['classes'][scores(model,x[test]).argmax(1)]
            correct+=int((guesses==targets[test]).sum());total+=int(test.sum())
        results.append({**config,'correct':correct,'total':total,'agreement':correct/max(1,total)})
    best=max(range(len(results)),key=lambda i:results[i]['agreement'])
    return dict(configs[best]),results


def temporal_groups(events,labels):
    bounds=[];mapping={}
    for e in sorted(events,key=lambda e:e['start']):
        if not bounds or e['start']>bounds[-1]['end']:
            bounds.append({'start':e['start'],'end':e['end']})
        else:bounds[-1]['end']=max(bounds[-1]['end'],e['end'])
        mapping[e['id']]=len(bounds)-1
    merges=list(range(len(bounds)))
    def root(i):
        while merges[i]!=i:i=merges[i]
        return i
    for label in labels:
        group=mapping[label['event_id']]
        for i,b in enumerate(bounds):
            if label['start']<b['end'] and label['end']>b['start']:
                merges[root(i)]=root(group)
    return {eid:root(group) for eid,group in mapping.items()},bounds


def save(path,model):
    np.savez_compressed(path,**{k:v for k,v in model.items() if k!='config'},
                        config_json=np.asarray(__import__('json').dumps(model['config'])))
