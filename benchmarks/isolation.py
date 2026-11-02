"""Keep the sealed BFMD corpus out of local training and calibration."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
STORE=ROOT.parent/'03_数据/独立基准/BFMD'

def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def assert_training_allowed(manifest,path=None,registry_path=None):
    if manifest.get('benchmark_only') or manifest.get('role')=='independent_benchmark_only' or str(manifest.get('dataset','')).casefold()=='bfmd':raise ValueError('BFMD is benchmark-only; training/calibration rejected')
    registry=Path(registry_path) if registry_path else STORE/'manifest.json'
    sealed=json.loads(registry.read_text()) if registry.exists() else {}
    identifiers={s for m in sealed.get('matches',[]) for s in [m.get('match_id'),m.get('youtube_id'),m.get('youtube_url')] if s}
    hashes={a['sha256'] for a in sealed.get('assets',[])}
    hashes.update(a['sha256'] for a in sealed.get('heldout_video_files',[]))
    def walk(value):
        if isinstance(value,dict):
            if value.get('benchmark_only') or str(value.get('dataset','')).casefold()=='bfmd':raise ValueError('A BFMD benchmark row entered training')
            for v in value.values():walk(v)
        elif isinstance(value,list):
            for v in value:walk(v)
        elif isinstance(value,str):
            if value in identifiers or value in hashes:raise ValueError('Sealed BFMD identity/hash entered training')
            if '独立基准/BFMD' in value.replace('\\','/') or any(s in value for s in identifiers):raise ValueError('Sealed BFMD source path entered training')
    walk(manifest)
    if path:
        walk(str(Path(path).resolve()))
        for row in manifest.get('rows',[]):
            for key in ['path','video_path','source_file']:
                p=row.get(key)
                if p and Path(p).is_file() and _sha(p) in hashes:raise ValueError('Copied BFMD asset entered training')
    return {'status':'no_known_sealed_asset_overlap','upstream_pretraining_overlap':'unknown; not established by this guard'}
