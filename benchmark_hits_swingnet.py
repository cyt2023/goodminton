"""Isolated adapter for the badminton SwingNet author checkpoint.

This is NOT a claim of full reproduction: the badminton repository leaves
rt_conversion() unimplemented and omits model_custom.py.  The exact CNN/LSTM
architecture is recovered from the upstream GolfDB implementation and validated
with strict checkpoint loading. Raw RGB is an explicitly out-of-modality ablation;
the released model was trained on optical-flow videos with background removed.
No ground-truth labels are read and no production pipeline is changed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "models/BadmintonSwingNet/fold5_swingnet_3000.pth.tar"
_CACHED_MODELS: dict[str, nn.Module] = {}


class BadmintonSwingNet(nn.Module):
    """GolfDB feature extractor plus checkpoint-derived two-class output."""

    def __init__(self) -> None:
        super().__init__()
        path = ROOT / "vendor/GolfDB/MobileNetV2.py"
        spec = importlib.util.spec_from_file_location("swingnet_author_mobilenet", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        backbone = module.MobileNetV2(width_mult=1.0)
        self.cnn = nn.Sequential(*list(backbone.children())[0][:19])
        self.rnn = nn.LSTM(1280, 256, 1, batch_first=True, bidirectional=True)
        self.lin = nn.Linear(512, 2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch, frames, channels, height, width = images.shape
        features = self.cnn(images.reshape(batch * frames, channels, height, width))
        features = features.mean(3).mean(2).reshape(batch, frames, -1)
        # Zero initial state is the same as GolfDB's explicit zero hidden state;
        # implicit allocation permits CPU/MPS and does not force CUDA.
        recurrent, _ = self.rnn(features)
        return self.lin(recurrent).reshape(batch * frames, 2)


def load_swingnet(weights: Path | str = DEFAULT_WEIGHTS, device: str = "cpu"):
    model = BadmintonSwingNet()
    saved = torch.load(str(weights), map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    model.to(device).eval()
    return model


def reconstructed_rt_frame(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Explicit approximation, NOT recovered 2023 preprocessing.

    CF/DF formulas: arXiv2503.04500v3 Eqs13/17/19 by the same author.
    Chosen Gaussian sigma, numerical ridge, per-frame minmax normalization,
    180px processing, and zero-intensity background channel are reconstruction
    assumptions, declared in metadata and never tuned with benchmark labels.
    """
    old = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    now = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    difference = now - old
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)
    boundary_x = np.array([[-1, 0, 1], [-4, 0, 4], [-1, 0, 1]], np.float32) / 3
    boundary_y = np.array([[1, 4, 1], [0, 0, 0], [-1, -4, -1]], np.float32) / 3

    def conv(image, kernel):
        return cv2.filter2D(image, cv2.CV_32F, np.flip(kernel).copy(),
                            borderType=cv2.BORDER_REFLECT_101)

    def smooth(image):
        return cv2.GaussianBlur(image, (5, 5), 1.0, borderType=cv2.BORDER_REFLECT_101)

    integral = np.ones((3, 3), dtype=np.float32)
    cf_x = smooth(conv(difference, boundary_x) - conv(conv(difference, kx), integral))
    cf_y = smooth(conv(difference, boundary_y) - conv(conv(difference, ky), integral))
    grad_x, grad_y = conv(old, kx), conv(old, ky)
    divergence = conv(cf_x, kx) + conv(cf_y, ky)
    residual = -difference - grad_x * cf_x - grad_y * cf_y - old * divergence
    axx, axy, ayy = smooth(grad_x ** 2), smooth(grad_x * grad_y), smooth(grad_y ** 2)
    bx, by = smooth(grad_x * residual), smooth(grad_y * residual)
    axx, ayy = axx + 1e-6, ayy + 1e-6
    determinant = np.maximum(axx * ayy - axy * axy, 1e-12)
    df_x = (ayy * bx - axy * by) / determinant
    df_y = (axx * by - axy * bx) / determinant

    def normalize(magnitude):
        finite = np.nan_to_num(magnitude, nan=0, posinf=0, neginf=0)
        scale = float(finite.max())
        return np.rint(finite * (255 / scale)).clip(0, 255).astype(np.uint8) if scale > 0 else np.zeros_like(finite, np.uint8)

    red = normalize(np.hypot(df_x, df_y))
    green = normalize(np.hypot(cf_x, cf_y))
    # The original 2023 figure shows motion on black. Zeroing the later paper's
    # appearance channel is an explicit assumption, not an inferred exact match.
    return np.stack([np.zeros_like(red), green, red], axis=2)


def prepare_frame(frame: np.ndarray, previous_frame=None, input_mode="raw_rgb") -> np.ndarray:
    """Port author's center crop, 180 resize, and original-dimension padding.

    On the author's 1280x720 video this crops x=280:1000 and produces a
    250x180 tensor: the notebook calculates 70 vertical padding using input_size
    160 before resizing the square to 180. For other resolutions the same crop
    proportions are used; this geometry adaptation is reported in metadata.
    """
    height, width = frame.shape[:2]
    left_crop = int(round(width * 280 / 1280))
    right_crop = max(left_crop + 1, int(round(width * 1000 / 1280)))
    cropped = frame[:, left_crop:right_crop]
    resized = cv2.resize(cropped, (180, 180), interpolation=cv2.INTER_AREA)
    if input_mode == "reconstructed_rt":
        previous = previous_frame if previous_frame is not None else frame
        previous = cv2.resize(previous[:, left_crop:right_crop], (180, 180),
                              interpolation=cv2.INTER_AREA)
        resized = reconstructed_rt_frame(previous, resized)
    ratio = 160 / max(height, width)
    new_height, new_width = int(height * ratio), int(width * ratio)
    delta_width, delta_height = 160 - new_width, 160 - new_height
    top, left = delta_height // 2, delta_width // 2
    bordered = cv2.copyMakeBorder(
        resized, top, delta_height - top, left, delta_width - left,
        cv2.BORDER_CONSTANT, value=[0.406 * 255, 0.456 * 255, 0.485 * 255],
    )
    rgb = cv2.cvtColor(bordered, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    return np.transpose(rgb, (2, 0, 1))


def extract_events(probabilities: np.ndarray, postprocess: str = "author_literal"):
    """Author notebook uses upper 20% scores, runs of at least three frames."""
    if not len(probabilities):
        return [], None
    threshold = float(np.quantile(probabilities, 0.8, method="higher"))
    candidates = np.flatnonzero(probabilities > threshold).tolist()
    groups: list[list[int]] = [[]]
    if postprocess == "author_literal":
        # Preserve the released notebook's omission of the final candidate.
        for first, second in zip(candidates, candidates[1:]):
            groups[-1].append(first)
            if second - first != 1:
                groups.append([])
    elif postprocess == "complete_runs":
        for frame in candidates:
            if groups[-1] and frame - groups[-1][-1] != 1:
                groups.append([])
            groups[-1].append(frame)
    else:
        raise ValueError(f"Unknown postprocess: {postprocess}")
    events = []
    for run in groups:
        if len(run) >= 3:
            frame = int(run[int(np.argmax(probabilities[run]))])
            events.append({"frame": frame, "confidence": float(probabilities[frame]),
                           "run_start_frame": run[0], "run_end_frame": run[-1]})
    return events, threshold


def infer_swingnet(
    video: str | Path,
    weights: str | Path = DEFAULT_WEIGHTS,
    device: str = "cpu",
    start_frame: int = 0,
    end_frame: int | None = None,
    input_mode: str = "raw_rgb",
    sequence_length: int = 64,
    postprocess: str = "author_literal",
    model: nn.Module | None = None,
) -> dict:
    if input_mode not in {"raw_rgb", "preprocessed_xgg", "reconstructed_rt"}:
        raise ValueError("input_mode must identify raw_rgb, preprocessed_xgg, or reconstructed_rt")
    started = time.monotonic()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame = min(total, end_frame) if end_frame is not None else total
    if start_frame < 0 or end_frame <= start_frame:
        cap.release()
        raise ValueError("Invalid frame range")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    previous_frame = None
    if input_mode == "reconstructed_rt" and start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)
        ok, previous_frame = cap.read()
        if not ok:
            raise RuntimeError("Cannot decode preceding frame for RT context")
    if model is None:
        model = load_swingnet(weights, device)
    scores = []
    pending = []
    decoded = 0
    frame_geometry = None

    def infer_batch(images):
        batch = torch.from_numpy(np.stack(images)).unsqueeze(0).to(device)
        with torch.inference_mode():
            return model(batch).softmax(dim=1)[:, 0].cpu().numpy()

    try:
        for index in range(start_frame, end_frame):
            ok, frame = cap.read()
            if not ok:
                break
            if frame_geometry is None:
                frame_geometry = [int(frame.shape[1]), int(frame.shape[0])]
            pending.append(prepare_frame(frame, previous_frame, input_mode))
            previous_frame = frame
            decoded += 1
            if len(pending) == sequence_length:
                scores.append(infer_batch(pending))
                pending = []
        if pending:
            scores.append(infer_batch(pending))
    finally:
        cap.release()
    probabilities = np.concatenate(scores) if scores else np.empty(0, dtype=np.float32)
    events, threshold = extract_events(probabilities, postprocess)
    for event in events:
        for key in ("frame", "run_start_frame", "run_end_frame"):
            event[key] += start_frame
        event["time_sec"] = event["frame"] / fps
        event["player"] = None
    return {
        "model": "BadmintonSwingNet_author_fold5_iter3000",
        "video": str(Path(video).resolve()),
        "weights": str(Path(weights).resolve()),
        "weights_sha256": hashlib.sha256(Path(weights).read_bytes()).hexdigest(),
        "source_record": "https://zenodo.org/records/14677727",
        "input_mode": input_mode,
        "reproduction_status": {
            "raw_rgb": "out_of_modality_raw_rgb_ablation",
            "preprocessed_xgg": "author_checkpoint_inference_on_caller_supplied_xgg",
            "reconstructed_rt": "approximate_later_author_rt_formula_not_exact2023_preprocessing",
        }[input_mode],
        "rt_reconstruction": ({
            "formula_source": "https://arxiv.org/html/2503.04500v3",
            "equations": [13, 17, 19],
            "same_author_later_work": True,
            "processing_resolution": [180, 180], "gaussian_kernel": [5, 5],
            "gaussian_sigma": 1.0, "lk_diagonal_ridge": 1e-6,
            "rgb_channels": ["perframe_minmax_DF_magnitude", "perframe_minmax_CF_magnitude", "zero"],
            "first_frame": "Use preceding source frame when available; otherwise zero-motion duplicate.",
            "unverified": "2023 channel encoding, background removal, normalization, kernel widths and temporal alignment are not specified by available code.",
        } if input_mode == "reconstructed_rt" else None),
        "limitations": [
            "Author rt_conversion() is a pass stub; exact optical-flow/background-removal preprocessing is unavailable.",
            "model_custom.py is missing; GolfDB architecture with a two-class output is recovered and checkpoint loads strictly.",
            "No hitter identity is predicted by this two-class event model.",
            "Non-1280x720 videos use proportional center crop, an explicit input geometry adaptation.",
        ],
        "fps": fps, "source_frame_count": total,
        "start_frame": start_frame, "end_frame": end_frame,
        "decoded_frames": decoded, "source_geometry": frame_geometry,
        "sequence_length": sequence_length, "quantile": 0.8,
        "min_run_frames": 3, "postprocess": postprocess,
        "threshold": threshold, "elapsed_sec": time.monotonic() - started,
        "events": events, "probabilities": probabilities.tolist(),
    }


def infer(dense, trajectory, source_path, output_folder=None, input_mode="raw_rgb", **kwargs):
    """Bridge for the shared benchmark runner; uses only decoded video pixels.

    Pose/trajectory define the shared decoded-frame span, never a hit label.
    Geometry or event detection from those frontends is not fed to SwingNet.
    """
    frame_indices = [int(frame["frame_index"]) for frame in dense["frames"]]
    if not frame_indices:
        raise ValueError("Shared benchmark frame span is empty")
    if frame_indices != list(range(frame_indices[0], frame_indices[-1] + 1)):
        raise ValueError("SwingNet benchmark bridge requires contiguous decoded-frame indices")
    if dense.get("source_sha256") != trajectory.get("source_sha256"):
        raise ValueError("Shared pose/trajectory source identities differ")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    fallback = None
    try:
        if device not in _CACHED_MODELS:
            _CACHED_MODELS[device] = load_swingnet(device=device)
        result = infer_swingnet(
            source_path, device=device, start_frame=frame_indices[0],
            end_frame=frame_indices[-1] + 1, model=_CACHED_MODELS[device],
            input_mode=input_mode, postprocess="author_literal",
        )
    except (RuntimeError, NotImplementedError) as error:
        if device != "mps":
            raise
        fallback = f"{type(error).__name__}: {error}"
        device = "cpu"
        if device not in _CACHED_MODELS:
            _CACHED_MODELS[device] = load_swingnet(device=device)
        result = infer_swingnet(
            source_path, device=device, start_frame=frame_indices[0],
            end_frame=frame_indices[-1] + 1, model=_CACHED_MODELS[device],
            input_mode=input_mode, postprocess="author_literal",
        )
    if result["decoded_frames"] != len(frame_indices):
        raise RuntimeError("Decoded SwingNet span differs from shared benchmark span")
    events = []
    for index, event in enumerate(result.pop("events")):
        events.append({
            "id": f"event_{index:04d}", "frame_index": event["frame"],
            "actor": None, "player": None,
            "confidence": event["confidence"],
            "start_frame": event["run_start_frame"],
            "end_frame": event["run_end_frame"],
            "representative": "author_run_peak_probability",
        })
    hit_probability = np.asarray(result.pop("probabilities"), dtype=np.float32)
    result.update({
        "device": device, "mps_fallback": fallback,
        "source_sha256": dense.get("source_sha256"),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "comparison_eligibility": "diagnostic_ablation_only_not_full_author_method_ranking",
        "probability_column_order": ["hit", "no_hit"],
        "tuning": "Author fold5/iter3000, 64-frame chunks, quantile0.8, minrun3; no benchmark labels read.",
        "input_matching": "Shared contiguous decoded-frame span; pose and ball coordinates unused.",
    })
    return {"events": events,
            "frame_probs": np.stack([hit_probability, 1 - hit_probability], axis=1),
            "frame_indices": np.asarray(frame_indices, dtype=np.int64),
            "metadata": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--input-mode", default="raw_rgb", choices=["raw_rgb", "preprocessed_xgg", "reconstructed_rt"])
    parser.add_argument("--postprocess", default="author_literal", choices=["author_literal", "complete_runs"])
    args = parser.parse_args()
    result = infer_swingnet(args.video, args.weights, args.device, args.start_frame,
                           args.end_frame, args.input_mode, postprocess=args.postprocess)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({"output": str(path), "events": len(result["events"]),
                      "elapsed_sec": round(result["elapsed_sec"], 2),
                      "reproduction_status": result["reproduction_status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
