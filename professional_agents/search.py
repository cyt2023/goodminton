"""Two separate, bounded research reproductions over a closed tool language.
NOT official AFlow/EvoFlow integration. No generated Python, weights or prompts.
"""
import math,random,time
from dataclasses import replace
from .registry import Workflow
from .state import STORE,read,save,digest
from .evaluation import ReplayEvaluator
DIMENSIONS={'threshold':[0.,.35,.5,.65],'bst_decoding':['top1','coarse_sum'],'verify':[False,True],'retrieval_ranker':['type','structured','sequence'],'branch_order':['technical_first','retrieval_first']}

def workflow(values,mode):
    tools=['perception.cached','stroke.cached']
    if values['verify']:tools+=['stroke.check_labels']
    tools+=['technical.evidence','expert.retrieve'] if values['branch_order']=='technical_first' else ['expert.retrieve','technical.evidence']
    return Workflow(mode=mode,tools=tuple(tools),threshold=values['threshold'],bst_decoding=values['bst_decoding'],retrieval_ranker=values['retrieval_ranker']).validate()
def features(w):return {'threshold':w.threshold,'bst_decoding':w.bst_decoding,'verify':'stroke.check_labels' in w.tools,'retrieval_ranker':w.retrieval_ranker,'branch_order':'technical_first' if w.tools.index('technical.evidence')<w.tools.index('expert.retrieve') else 'retrieval_first'}
def random_values(rng):return {k:rng.choice(v) for k,v in DIMENSIONS.items()}
def rank(row):return (row['development']['metrics']['macro_f1'],-len(row['workflow']['tools']))

def search(mode,budget=24,seed=20261030):
    if mode not in ['AFLOW','EVOFLOW'] or not 4<=budget<=100:raise ValueError('Search mode/budget invalid')
    manifest=read(STORE/'datasets.json');evaluator=ReplayEvaluator(manifest);rng=random.Random(seed);history=[];start=time.perf_counter();nodes=[];population=[]
    def evaluate(values,parents=None):
        w=workflow(values,mode);row={'iteration':len(history)+1,'workflow':w.packet(),'graph':{'nodes':[{'id':i,'tool':t} for i,t in enumerate(w.tools)],'edges':[[i,i+1] for i in range(len(w.tools)-1)]},'parents':parents or [],'development':evaluator.evaluate(w,'development')};history.append(row);return row
    fixed=Workflow();baseline=evaluator.evaluate(fixed,'validation')
    if mode=='AFLOW':
        # Prefix tree UCT selection/expansion/rollout/backpropagation. The official
        # framework searches generated code with LLMs; this adapter deliberately
        # restricts generation to registered operations and finite parameters.
        keys=list(DIMENSIONS);root={'prefix':{},'visits':0,'reward':0.,'children':[],'parent':None};nodes=[root]
        for _ in range(budget):
            node=root;path=[node]
            while len(node['prefix'])<len(keys):
                key=keys[len(node['prefix'])];used={c['prefix'][key] for c in node['children']};remaining=[v for v in DIMENSIONS[key] if v not in used]
                if remaining:
                    child={'prefix':{**node['prefix'],key:rng.choice(remaining)},'visits':0,'reward':0.,'children':[],'parent':nodes.index(node)};node['children'].append(child);nodes.append(child);node=child;path.append(node);break
                node=max(node['children'],key=lambda c:c['reward']/max(1,c['visits'])+math.sqrt(2*math.log(max(1,node['visits']))/max(1,c['visits'])));path.append(node)
            values={**random_values(rng),**node['prefix']};row=evaluate(values);reward=row['development']['metrics']['macro_f1']
            for item in path:item['visits']+=1;item['reward']+=reward
        tree=[{'prefix':n['prefix'],'visits':n['visits'],'reward_sum':n['reward'],'parent':n['parent']} for n in nodes]
    else:
        # Tag/parameter-neighbour retrieval, crossover, mutation and niche retention.
        # Exact original LLM mutation/operators cannot be verified: upstream404.
        size=min(8,budget);population=[evaluate(random_values(rng)) for _ in range(size)]
        while len(history)<budget:
            query=random_values(rng)
            parents=sorted(population,key=lambda r:(-sum(features(Workflow.from_packet(r['workflow']))[k]==query[k] for k in DIMENSIONS),-rank(r)[0]))[:2]
            a,b=[features(Workflow.from_packet(r['workflow'])) for r in parents];child={k:rng.choice([a[k],b[k]]) for k in DIMENSIONS};key=rng.choice(list(DIMENSIONS));child[key]=rng.choice([v for v in DIMENSIONS[key] if v!=child[key]])
            row=evaluate(child,[r['iteration'] for r in parents]);pool=population+[row];niches={}
            for r in sorted(pool,key=rank,reverse=True):
                f=features(Workflow.from_packet(r['workflow']));tag=(f['bst_decoding'],f['verify'],len(r['workflow']['tools']));niches.setdefault(tag,r)
            diverse=list(niches.values());population=diverse[:size]
            for r in sorted(pool,key=rank,reverse=True):
                if len(population)>=size:break
                if r not in population:population.append(r)
        tree=None
    # Selection sees DEVELOPMENT only. Validation is a single post-selection report.
    selected=max(history,key=rank);chosen=Workflow.from_packet(selected['workflow']);validation=evaluator.evaluate(chosen,'validation')
    result={'schema_version':1,'mode':mode,'implementation':'constrained research reproduction; not official upstream optimizer','seed':seed,'budget':budget,'evaluation_calls':len(history),'search_wall_seconds':time.perf_counter()-start,'search_space':DIMENSIONS,'scope':'cached conditional-classification policy search; raw-window reanalysis, perception-model choice and retrieval relevance are not optimized by these scores','objective':'maximize development conditional Macro-F1; tie-break fewer registered operations; quality and costs reported separately','selected_workflow':chosen.packet(),'selected_development':selected['development'],'validation':validation,'fixed_validation':baseline,'history':history,'tree':tree,'population':[r['iteration'] for r in population] if mode=='EVOFLOW' else None,'data_manifest_sha256':digest(manifest),'sealed_BFMD_access':False,'validation_used_for_selection':False,'final_test':'not evaluated; internal validation has historical project exposure','limitations':['No claim of full AFlow/EvoFlow reproduction','No improvement in contact timing is measured by conditional classification','Gallery ranker/topology varied but their downstream quality is unscored without relevance/review truth']}
    folder=STORE/'experiments'/mode;save(folder/'search.json',result);save(folder/'selected_workflow.json',chosen.packet());return result

def compare():
    manifest=read(STORE/'datasets.json');e=ReplayEvaluator(manifest);rows=[]
    for mode in ['FIXED','AFLOW','EVOFLOW']:
        w=Workflow() if mode=='FIXED' else Workflow.from_packet(read(STORE/'experiments'/mode/'selected_workflow.json'));value=e.evaluate(w,'validation');ablated=replace(w,tools=tuple(t for t in w.tools if t not in ['stroke.check_labels','perception.verify','stroke.window']))
        rows.append({'mode':mode,'workflow':w.packet(),'validation':value,'ablation_without_verification_or_reanalysis':e.evaluate(ablated,'validation'),'ablation_scope':'cached label-consistency verification; raw reanalysis not available in this cache'})
    value={'scope':'internal whole-match validation of conditional cached-feature classification; NOT untouched final or end-to-end','data_manifest_sha256':digest(manifest),'rows':rows,'final_test':manifest['final_test']};save(STORE/'experiments/comparison.json',value)
    lines=['# FIXED / AFLOW / EVOFLOW — internal validation','',value['scope'],'','| Mode | Samples | Conditional Macro-F1 | Coverage | Replay CPU seconds | Verification ablation Macro-F1 |','|---|---:|---:|---:|---:|---:|']
    for r in rows:
        m=r['validation']['metrics'];lines.append(f"|{r['mode']}|{m['samples']}|{m['macro_f1']:.4f}|{m['coverage']:.4f}|{r['validation']['cost']['replay_cpu_seconds']:.6f}|{r['ablation_without_verification_or_reanalysis']['metrics']['macro_f1']:.4f}|")
    lines+=['','No raw-video hit F1, timing MAE, retrieval relevance score or evidence-analysis quality is inferred from these feature results. BFMD references were not optimization inputs. Full final testing remains unrun.']
    (STORE/'experiments/comparison.md').write_text('\n'.join(lines)+'\n');return value
