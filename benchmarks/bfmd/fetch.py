"""Fetch the verified official Drive listing, excluding videos and executable files."""
import json,concurrent.futures
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
STORE=ROOT.parent/'03_数据/独立基准/BFMD'

def main():
    import gdown
    rows=json.loads((STORE/'drive_listing.json').read_text())
    selected=[r for r in rows if r['path'].endswith(('.json','.csv','.md')) and ('annotations/' in r['path'] or r['path'] in ['README.md','match_index.csv','Badminton_video_list.csv'])]
    def get(row):
        path=Path(row['local_path']);path=ROOT/path if not path.is_absolute() else path;path=path.resolve()
        try:
            path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists():gdown.download(id=row['id'],output=str(path),quiet=True,timeout=60,retries=1)
            return {'path':row['path'],'status':'downloaded','bytes':path.stat().st_size}
        except Exception as error:return {'path':row['path'],'status':'unavailable','error':str(error)[-1200:]}
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(get,selected):results.append(result);print(result['status'],result['path'],flush=True)
    (STORE/'download_result.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
