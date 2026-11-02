"""Pinned local visual baseline shared by controller and inference subprocess."""
from pathlib import Path

MODEL_ID = 'mlx-community/Qwen3-VL-4B-Instruct-4bit'
MODEL_REVISION = '2fd8dacbdb8f1e54b8c005f081ec5bf79c56376b'
MODEL = Path(__file__).resolve().parent / 'models/vision/Qwen3-VL-4B-Instruct-4bit'
ENGINE_VERSION = 'visual-actions-v4-native-video'
