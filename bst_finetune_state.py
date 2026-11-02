"""Lightweight state for cross-video BST learning; safe for the stdlib server."""
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

from analysis_jobs import DATA
from stroke_taxonomy import AUTHOR_MAP

STORE = DATA / '击球标注' / '直接BST微调'
POOL = DATA / '训练库' / '训练样本.json'
TRAINING_SIGNATURE_VERSION = 'bst-training-pool-semantics-v1'
FEATURE_CONTRACT_FILES = {
    'feature_code_sha256': 'finetune_personal_bst.py',
    'adapter_sha256': 'specialist_bst.py',
    'window_code_sha256': 'stroke_context.py',
    'window_repair_code_sha256': 'stroke_window_repairs.py',
    'hit_refinement_code_sha256': 'hit_refinement.py',
}
# Identity, target and eligibility determine what reaches the current BST
# training path. Export timestamps, history, display text and participation
# annotations are provenance, not a change to the training examples.
TRAINING_SAMPLE_FIELDS = ('job_id', 'event_id', 'source_sha256', 'stroke_type',
                          'player', 'peak_time', 'contact_confirmed', 'wrong_target',
                          'play_context', 'training_role', 'taxonomy_version')


def read(path, default=None):
    try:
        return json.loads(path.read_text()) if path.is_file() else default
    except (OSError, ValueError):
        return default


def training_pool_signature(pool):
    samples = [{key: sample.get(key) for key in TRAINING_SAMPLE_FIELDS}
               for sample in pool.get('samples', [])]
    canonical = lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True,
                                         separators=(',', ':'), allow_nan=False)
    samples.sort(key=canonical)
    payload = {'version': TRAINING_SIGNATURE_VERSION, 'samples': samples}
    return hashlib.sha256(canonical(payload).encode('utf-8')).hexdigest()


def recorded_training_pool_signature(record):
    """Recover legacy run identity from its immutable snapshot, never live data."""
    if record.get('training_pool_signature'):
        return record['training_pool_signature']
    checkpoint = record.get('candidate_checkpoint')
    if checkpoint:
        run = (DATA / checkpoint).resolve().parent
    elif record.get('run_id'):
        run = (STORE / '运行记录' / record['run_id']).resolve()
    else:
        return None
    if not run.is_relative_to(STORE.resolve()):
        return None
    snapshot = read(run / 'training_pool_snapshot.json')
    return training_pool_signature(snapshot) if snapshot is not None else None


def current_feature_contract():
    root = Path(__file__).resolve().parent
    return {key: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for key, name in FEATURE_CONTRACT_FILES.items()}


def recorded_feature_identity(record):
    """Recover all feature dependencies from the immutable training run."""
    identity = record.get('personal_features') or record.get('feature_identity')
    if not identity and record.get('run_id'):
        run = (STORE / '运行记录' / record['run_id']).resolve()
        if run.is_relative_to(STORE.resolve()):
            manifest = read(run / 'manifest.json', {})
            identity = manifest.get('personal_features')
    return identity


def recorded_feature_contract(record):
    identity = record.get('feature_contract') or recorded_feature_identity(record)
    return {key: identity.get(key) for key in FEATURE_CONTRACT_FILES} if identity else None


def feature_contract_matches(record):
    return recorded_feature_contract(record) == current_feature_contract()


@lru_cache(maxsize=64)
def _stat_digest(path, size, mtime_ns):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def dependency_digest(path):
    if not path.is_file():
        return None
    stat = path.stat()
    return _stat_digest(str(path), stat.st_size, stat.st_mtime_ns)


def training_inputs_match(record, job=None):
    """Require the same code, backbone and actual input data for reuse.

    Annotation eligibility/targets are compared by training_pool_signature at
    the workflow gate. Here source files that determine the numerical features
    are checked even if no human label has changed. Metadata-only exports do
    not require rehashing large pose files on each status request.
    """
    if not feature_contract_matches(record):
        return False
    identity = recorded_feature_identity(record)
    if not identity or not identity.get('sources'):
        return False
    from specialist_model import ROOT, BST_NAME
    if dependency_digest(ROOT / 'models/bst' / BST_NAME) != identity.get('model_sha256'):
        return False
    sources = identity['sources']
    if job is not None:
        if job not in sources:
            return False
        sources = {job: sources[job]}
    for source_job, expected in sources.items():
        if not re.fullmatch(r'J_[0-9a-f]{32}', source_job):
            return False
        analysis = DATA / '专项分析' / source_job
        report = read(analysis / 'report.json', {})
        if report.get('source_sha256') != expected.get('source_sha256'):
            return False
        paths = {
            'dense_pose_sha256': analysis / 'dense_pose.json',
            'trajectory_sha256': analysis / 'trajectory.json',
            'candidate_results_sha256': DATA / '技术动作' / source_job / '片段/classified_events.json',
            'frozen_public_predictions_sha256': DATA / '击球标注' / source_job / 'public_only_predictions.json',
            'hit_refinement_data_sha256': analysis / 'hit_refinement.json',
        }
        for key, path in paths.items():
            actual = dependency_digest(path)
            if (key in ['dense_pose_sha256', 'trajectory_sha256', 'candidate_results_sha256']
                    and actual is None) or actual != expected.get(key):
                return False
    return True


def library_readiness():
    pool = read(POOL, {'samples': []})
    supported = set(AUTHOR_MAP.values()) - {'unknown'}
    samples = [s for s in pool['samples'] if s.get('training_role') == 'stroke_positive'
               and s.get('contact_confirmed') and not s.get('wrong_target')
               and s.get('play_context') != 'between_rallies' and s.get('stroke_type') in supported]
    counts = Counter(s['stroke_type'] for s in samples)
    videos = {s['source_sha256'] for s in samples}
    ready = len(samples) >= 12 and sum(n >= 3 for n in counts.values()) >= 2
    return {'can_train': ready, 'eligible_labels': len(samples), 'class_counts': dict(counts),
            'source_video_count': len(videos), 'pool_total': len(pool['samples']),
            'source_pool_sha256': hashlib.sha256(POOL.read_bytes()).hexdigest() if POOL.is_file() else None,
            'training_pool_signature': training_pool_signature(pool),
            'scope': 'All saved videos; supported basic strokes only. Fine classes and hit negatives retained separately.',
            'requirement': '使用所有录像已确认的基础球种学习；至少12条，且至少两类各有3条。细分技术、无击球和未确认样本单独保留。'}


def learning_state():
    state = read(STORE / 'state.json', {})
    if state:
        state['training_pool_signature'] = recorded_training_pool_signature(state)
        if state.get('status') == 'completed':
            state['input_contract_changed'] = not feature_contract_matches(state)
            state['input_data_changed'] = not training_inputs_match(state)
            state['feature_contract'] = recorded_feature_contract(state)
    return state


def job_predictions(job, source_sha256, revision=None, current_results_sha256=None):
    # Callers have already validated job through stroke_review.folder_for.
    result = read(STORE / 'predictions' / (job + '.json'))
    if not result or result.get('source_sha256') != source_sha256:
        return None
    if current_results_sha256 is not None and result.get('candidate_results_sha256') != current_results_sha256:
        # A newly recognized candidate set invalidates this sidecar even when
        # the video's bytes and annotation revision have not changed.
        return None
    if not training_inputs_match(result, job=job):
        # A changed support window produces different features even when the
        # user labels and model weights are unchanged. Keep this historical
        # sidecar on disk, but do not present it as the current prediction.
        return None
    result['annotations_changed'] = revision is not None and result.get('dataset_revision') != revision
    state = learning_state()
    result['is_latest_completed_model'] = result.get('model_sha256') == state.get('candidate_model_sha256')
    return result


def active_checkpoint():
    """Return a versioned checkpoint only when its bytes match registration."""
    registration = read(STORE / 'active_model.json')
    if not registration or not registration.get('active'):
        return None, None
    path = (DATA / registration['checkpoint']).resolve()
    if not path.is_relative_to(DATA.resolve()) or not path.is_file():
        raise ValueError('已登记的个人BST模型文件不存在')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != registration.get('sha256'):
        raise ValueError('个人BST模型校验失败')
    return path, registration
