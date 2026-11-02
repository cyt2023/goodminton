"""BST output in the BFMD stroke taxonomy.

BST's 25 classes (hitter side x ShuttleSet type) are summed into BFMD's
categories (canonical names as in importer.LABEL_MAP), so e.g. cross-court
net shots count as BFMD net shots. Serves are excluded from the argmax for
candidates that are not scored as serves (serves are ignored by the study).
"""
from professional_agents.retrieval import mainland

# project mainland label -> BFMD canonical label
TO_BFMD={'clear':'clear','clear_flat':'clear','lift':'lift','drop':'drop','smash':'smash','drive':'drive','push':'push',
         'net_kill':'net_kill','net':'net','net_spin':'net','net_cross':'net','defense':'defense',
         'serve_short':'serve','serve_long':'serve_long'}
SERVES={'serve','serve_long'}

def bfmd_probs(class_labels,scores):
    out={}
    for c,p in zip(class_labels,scores):
        k=TO_BFMD.get(mainland(c))
        if k:out[k]=out.get(k,0.)+float(p)
    return out

def bfmd_label(probs,allow_serve=False):
    probs={k:v for k,v in probs.items() if allow_serve or k not in SERVES}
    return max(probs,key=probs.get) if probs else None
