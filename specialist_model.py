"""Pinned specialist pipeline identity; the VLM baseline retains its own ID."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
ENGINE_VERSION = 'specialist-actions-v1-tracknet-bst0-dense'
MODEL_ID = 'TrackNetV3+BST-0-ShuttleSet25+Qwen3VL-style'
CLASSIFICATION_SOURCE = 'bst0_tracknet_style_vlm'
BST_NAME = 'bst_0_JnB_bone_between_2_hits_with_max_limits_seq_100_merged_2.pt'
FULL_BST_NAME = 'bst_CG_AP_JnB_bone_between_2_hits_with_max_limits_seq_100_merged_2.pt'
MANIFEST = ROOT / 'models/specialist_manifest.json'


def revision():
    return hashlib.sha256(MANIFEST.read_bytes()).hexdigest() if MANIFEST.exists() else 'weights-not-prepared'


MODEL_REVISION = revision()


def available():
    files = [ROOT / 'models/tracknetv3/TrackNet_best.pt', ROOT / 'models/tracknetv3/InpaintNet_best.pt',
             ROOT / 'models/bst' / BST_NAME, MANIFEST, ROOT / '.venv-vision/bin/python', ROOT / '.venv-pose/bin/python']
    return all(p.is_file() for p in files)
