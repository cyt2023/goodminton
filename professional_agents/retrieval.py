"""Deterministic professional-to-professional retrieval, not corrective diagnosis."""
import csv,math
from pathlib import Path
from stroke_taxonomy import AUTHOR_MAP
from .state import read,save,digest,sha,STORE
from .datasets import check_development

def parse_time(t):
    a=list(map(float,t.split(':')));return sum(v*60**i for i,v in enumerate(reversed(a)))
def mainland(label):
    raw=label.split('_',1)[-1] if label.startswith(('Top_','Bottom_')) else label
    return AUTHOR_MAP.get(raw,AUTHOR_MAP.get(label,'unknown'))
def build_index(record):
    check_development(record);rows=[]
    for i,r in enumerate(csv.DictReader(Path(record['csv']).open())):
        if not r.get('time') or not r.get('type') or r.get('flaw') not in ['', '0','0.0']:continue
        label=mainland(r['type'])
        if label=='unknown':continue
        t=parse_time(r['time']);rows.append({'id':digest([record['match_key'],i])[:24],'match_key':record['match_key'],'dataset':record['dataset'],'kind':'dataset_annotation','timestamp':t,'time_resolution_seconds':1,'source_frame_index':int(float(r['frame_num'])) if r.get('frame_num') else None,'source_frame_rate':'not independently verified; frame_num retained without conversion','player_id':r.get('player'),'stroke_label':label,'rally_id':r['rally'],'ball_round':float(r['ball_round']) if r.get('ball_round') else 0,'court_zone':r.get('hit_area') or None,'positions':{k:r[k] or None for k in ['player_location_x','player_location_y','opponent_location_x','opponent_location_y']},'source_url':record['source_url'],'clip_url':record['source_url']+'&t='+str(max(0,int(t)-2)),'display_window':[max(0,t-2),t+2],'window_kind':'display context, not annotated action boundaries','annotation_provenance':{'file':record['csv'],'sha256':sha(record['csv']),'row':i+2,'original_label':r['type']},'corrective_demonstration':False})
    groups={}
    for r in rows:groups.setdefault(r['rally_id'],[]).append(r)
    for group in groups.values():
        group.sort(key=lambda r:r['ball_round'])
        for i,r in enumerate(group):
            chunk=group[max(0,i-2):i+1];r['sequence']=[x['stroke_label'] for x in chunk] if len(chunk)==3 and all(b['ball_round']==a['ball_round']+1 for a,b in zip(chunk,chunk[1:])) else []
    value={'schema_version':1,'role':'retrieval_gallery','kind':'dataset_annotation','match_keys':sorted({r['match_key'] for r in rows}),'rows':rows,'scope':'Annotated professional situations; similarity does not imply correctness or suitability','source_sha256':sha(record['csv'])};save(STORE/'retrieval/index.json',value);return value

def retrieve(query,index,ranker='structured',k=5):
    if index.get('role')!='retrieval_gallery':raise ValueError('Only approved gallery is searchable')
    check_development({'role':'retrieval_gallery','domain':'professional','match_keys':index['match_keys']})
    if query.get('stroke_label') in [None,'unknown']:return []
    result=[]
    for row in index['rows']:
        if row['match_key']==query.get('match_key'):continue
        terms={'stroke_type':float(row['stroke_label']==query['stroke_label'])};weights={'stroke_type':.7}
        if ranker!='type' and query.get('court_zone') and row.get('court_zone'):terms['court_zone']=float(query['court_zone']==row['court_zone']);weights['court_zone']=.1
        if ranker=='sequence' and len(query.get('sequence',[]))==3 and len(row['sequence'])==3:terms['sequence']=sum(a==b for a,b in zip(query['sequence'],row['sequence']))/3;weights['sequence']=.2
        score=sum(terms[a]*weights[a] for a in terms)/sum(weights[a] for a in terms)
        if terms['stroke_type']:
            result.append({'reference_id':row['id'],'score':score,'components':terms,'reference':row,'interpretation':'Similar annotated professional situation; not a verified correction','missing_query_features':[n for n in ['court_zone','sequence'] if not query.get(n)]})
    return sorted(result,key=lambda r:(-r['score'],r['reference_id']))[:k]

def relevance_metrics(ranked,relevance,k=5):
    if relevance is None:return {'status':'unavailable','Recall@K':None,'nDCG@K':None,'reason':'No independent relevance judgements; same stroke type alone is not corrective relevance'}
    positive={i for i,g in relevance.items() if g>0};ids=ranked[:k]
    recall=len(set(ids)&positive)/len(positive) if positive else None
    dcg=sum((2**relevance.get(i,0)-1)/math.log2(n+2) for n,i in enumerate(ids));ideal=sum((2**g-1)/math.log2(n+2) for n,g in enumerate(sorted(relevance.values(),reverse=True)[:k]))
    return {'status':'evaluated','Recall@K':recall,'nDCG@K':dcg/ideal if ideal else None,'K':k}
