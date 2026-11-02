"""Run the unchanged local shuttle frontend on all frozen per-hit clips.""" 
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from analysis_jobs import atomic_json, file_hash

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / '03_数据/击球标注/根因诊断/2026-10-15_同源前端对照'


def main():
    clips = json.loads((OUT / 'clips_manifest.json').read_text())['clips']
    started = time.perf_counter()
    results = []
    for row in clips:
        folder = OUT / 'clips' / row['sample_id'] / 'tracknet'
        folder.mkdir(parents=True, exist_ok=True)
        config = {'source_file': row['source_file'], 'source_sha256': row['source_sha256'],
                  'specialist_folder': str(folder)}
        atomic_json(folder / 'config.json', config)
        with (folder / 'run.log').open('w') as log:
            run = subprocess.run([sys.executable, '-u', str(ROOT / 'specialist_tracknet.py'),
                                  '--config', str(folder / 'config.json')],
                                 stdout=log, stderr=subprocess.STDOUT, timeout=180)
        assert run.returncode == 0, f"TrackNet failed {row['sample_id']}; see isolated run.log"
        trajectory = json.loads((folder / 'trajectory.json').read_text())
        assert trajectory['source_sha256'] == row['source_sha256']
        assert len(trajectory['frames']) == trajectory['frame_count'] == row['frames']
        assert all(r['frame_index'] == i for i, r in enumerate(trajectory['frames']))
        results.append({'sample_id': row['sample_id'], 'frames': row['frames'],
                        'trajectory_sha256': file_hash(folder / 'trajectory.json'),
                        'raw_detected_frames': trajectory['raw_detected_frames'],
                        'inpaint_requested_frames': trajectory['inpaint_requested_frames'],
                        'runtime_seconds': trajectory['runtime_seconds']})
        atomic_json(OUT / 'tracknet_progress.json', {'completed': len(results), 'total': len(clips),
                    'elapsed_seconds': time.perf_counter() - started})
        print('PAIRED_TRACKNET', len(results), '/', len(clips),
              row['sample_id'], 'raw', trajectory['raw_detected_frames'], '/', row['frames'], flush=True)
    atomic_json(OUT / 'tracknet_replay_manifest.json', {'created_at': datetime.now(timezone.utc).isoformat(),
                'samples': results, 'runner_sha256': file_hash(Path(__file__)),
                'production_adapter_sha256': file_hash(ROOT / 'specialist_tracknet.py'),
                'elapsed_seconds': time.perf_counter() - started})


if __name__ == '__main__':
    main()
