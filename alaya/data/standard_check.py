"""Check a standard-format dataset against docs/TRAINING.md before spending GPU time.

The loader skips a broken sample at read time and retries another one, so a dataset
with systematic problems trains silently on whatever is left. This walks every clip
up front and names each problem by file.

Errors make a clip unusable or wrong for training; warnings are legal but probably
not what was intended.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

from alaya.data.standard import FPS_TOLERANCE, DatasetSpec, chunk_windows

_ROTATION_TOL = 1e-2
_PRINCIPAL_POINT_TOL = 0.05


@dataclass
class Report:
    name: str
    clips: int = 0
    seconds: float = 0.0
    windows: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _probe_video(path: str) -> tuple[int, float, int, int]:
    """(frame count, fps, width, height)."""
    from decord import VideoReader, cpu

    reader = VideoReader(path, ctx=cpu(0))
    height, width = reader[0].shape[:2]
    return len(reader), float(reader.get_avg_fps()), int(width), int(height)


def check_dataset(
    spec: DatasetSpec,
    *,
    fps: float,
    window_frames: int,
    prefix_frames: int,
    chunk_frames: int,
) -> Report:
    report = Report(name=spec.name)
    video_dir = os.path.join(spec.root, "videos")
    if not os.path.isdir(video_dir):
        report.errors.append(f"{spec.root}: no videos/ directory")
        return report
    clip_ids = sorted(os.path.splitext(f)[0] for f in os.listdir(video_dir) if f.lower().endswith(".mp4"))
    if not clip_ids:
        report.errors.append(f"{video_dir}: no .mp4 files")
        return report

    camera = spec.format != "video_caption_static"
    pose_dir = os.path.join(spec.root, "poses")
    if not camera and os.path.isdir(pose_dir) and os.listdir(pose_dir):
        report.warnings.append(f"{pose_dir}: poses/ is ignored for video_caption_static (camera is fixed)")
    for sub, ext in (("captions", ".json"), ("poses", ".npz")):
        extra = sorted(
            f for f in (os.listdir(os.path.join(spec.root, sub)) if os.path.isdir(os.path.join(spec.root, sub)) else [])
            if f.endswith(ext) and os.path.splitext(f)[0] not in set(clip_ids)
        )
        if extra and (sub == "captions" or camera):
            report.warnings.append(f"{sub}/: {len(extra)} files without a matching video, e.g. {extra[0]}")

    for clip_id in clip_ids:
        _check_clip(spec, clip_id, report, camera=camera, fps=fps, window_frames=window_frames,
                    prefix_frames=prefix_frames, chunk_frames=chunk_frames)
    report.clips = len(clip_ids)
    return report


def _check_clip(spec, clip_id, report, *, camera, fps, window_frames, prefix_frames, chunk_frames):
    video_path = os.path.join(spec.root, "videos", f"{clip_id}.mp4")
    try:
        n_frames, video_fps, width, height = _probe_video(video_path)
    except Exception as exc:  # decord raises its own error types
        report.errors.append(f"{video_path}: cannot decode ({exc})")
        return
    duration = n_frames / video_fps
    report.seconds += duration
    if video_fps < fps * (1.0 - FPS_TOLERANCE):
        report.errors.append(f"{video_path}: {video_fps:.2f} fps is below the training fps {fps:g}")
        return
    available = int(n_frames / (video_fps / fps))
    if available < window_frames:
        report.errors.append(
            f"{video_path}: {duration:.2f} s is shorter than one training window ({window_frames / fps:.3f} s)"
        )
        return

    caption = _check_caption(spec, clip_id, report)
    if camera:
        _check_pose(spec, clip_id, report, n_frames=n_frames, width=width, height=height)
    if caption is None:
        return
    if spec.format != "video_timed_prompts_camera":
        report.windows += 1
        return

    prompts = _check_segments(caption, report, where=os.path.join(spec.root, "captions", f"{clip_id}.json"),
                              duration=duration, fps=fps)
    if prompts is None:
        return
    where = os.path.join(spec.root, "captions", f"{clip_id}.json")
    if spec.prompt_mode == "segment":
        window_s = window_frames / fps
        usable = [p for p in prompts if p[1] - p[0] >= window_s]
        for lo, hi, text in prompts:
            if hi - lo < window_s:
                report.warnings.append(
                    f"{where}: segment {text[:40]!r} lasts {hi - lo:.2f} s < one window ({window_s:.3f} s), "
                    "so segment mode never trains on it"
                )
        report.windows += len(usable)
    else:
        for boundary in sorted({b for lo, hi, _ in prompts for b in (lo, hi) if 1e-6 < b < duration - 1e-6}):
            frame = boundary * fps
            k = round((frame - prefix_frames) / chunk_frames)
            if abs(frame - (prefix_frames + k * chunk_frames)) > 0.5:
                below = prefix_frames + max(0, int((frame - prefix_frames) // chunk_frames)) * chunk_frames
                report.errors.append(
                    f"{where}: prompt boundary {boundary:.3f} s is not on a rollout round boundary; nearest are "
                    f"{below / fps:.3f} s and {(below + chunk_frames) / fps:.3f} s "
                    f"({prefix_frames}/{fps:g} s + k*{chunk_frames}/{fps:g} s)"
                )
        windows = chunk_windows(prompts, available_frames=available, window_frames=window_frames,
                                prefix_frames=prefix_frames, chunk_frames=chunk_frames, fps=fps)
        if not windows:
            report.errors.append(f"{where}: no rollout round has a prompt, so per_chunk has nothing to train on")
        report.windows += len(windows)


def _check_caption(spec, clip_id, report):
    path = os.path.join(spec.root, "captions", f"{clip_id}.json")
    if not os.path.exists(path):
        report.errors.append(f"{path}: missing")
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as exc:
        report.errors.append(f"{path}: not valid json ({exc})")
        return None
    if not isinstance(data, dict) or not isinstance(data.get("caption"), str) or not data["caption"].strip():
        report.errors.append(f'{path}: needs a non-empty "caption" string')
        return None
    return data


def _check_segments(caption, report, *, where, duration, fps):
    segments = caption.get("segments")
    if not isinstance(segments, list) or not segments:
        report.errors.append(f'{where}: timed prompts need a non-empty "segments" list')
        return None
    prompts = []
    ok = True
    for i, seg in enumerate(segments):
        rng = seg.get("time_range_s") if isinstance(seg, dict) else None
        text = seg.get("prompt") if isinstance(seg, dict) else None
        if not (isinstance(rng, list) and len(rng) == 2 and all(isinstance(v, (int, float)) for v in rng)):
            report.errors.append(f'{where}: segments[{i}] needs "time_range_s": [start, end] in seconds')
            ok = False
            continue
        if not isinstance(text, str) or not text.strip():
            report.errors.append(f'{where}: segments[{i}] needs a non-empty "prompt" string')
            ok = False
            continue
        lo, hi = float(rng[0]), float(rng[1])
        if not 0.0 <= lo < hi:
            report.errors.append(f"{where}: segments[{i}] time_range_s {rng} must satisfy 0 <= start < end")
            ok = False
        elif hi > duration + 1.0 / fps:
            report.errors.append(f"{where}: segments[{i}] ends at {hi:.3f} s, past the end of the video ({duration:.3f} s)")
            ok = False
        prompts.append((lo, hi, text))
    prompts.sort(key=lambda p: p[0])
    for (lo_a, hi_a, a), (lo_b, _hi_b, b) in zip(prompts, prompts[1:]):
        if lo_b < hi_a - 1e-6:
            report.errors.append(f"{where}: segments {a[:30]!r} and {b[:30]!r} overlap ({lo_b:.3f} s < {hi_a:.3f} s)")
            ok = False
    return prompts if ok else None


def _check_pose(spec, clip_id, report, *, n_frames, width, height):
    path = os.path.join(spec.root, "poses", f"{clip_id}.npz")
    if not os.path.exists(path):
        report.errors.append(f"{path}: missing")
        return
    try:
        data = np.load(path)
    except (OSError, ValueError) as exc:
        report.errors.append(f"{path}: cannot load ({exc})")
        return
    if "cam_c2w" not in data.files:
        report.errors.append(f'{path}: needs a "cam_c2w" array, found {data.files}')
        return
    c2w = np.asarray(data["cam_c2w"], dtype=np.float64)
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
        report.errors.append(f"{path}: cam_c2w must be [N, 4, 4], got {list(c2w.shape)}")
        return
    if len(c2w) != n_frames:
        report.errors.append(f"{path}: {len(c2w)} poses for {n_frames} video frames; need one pose per frame")
    if not np.isfinite(c2w).all():
        report.errors.append(f"{path}: cam_c2w has NaN or inf")
        return
    if np.abs(c2w[:, 3] - np.array([0.0, 0.0, 0.0, 1.0])).max() > 1e-4:
        report.errors.append(f"{path}: cam_c2w bottom row must be [0, 0, 0, 1]")
    rot = c2w[:, :3, :3]
    drift = np.abs(np.einsum("nij,nkj->nik", rot, rot) - np.eye(3)).max()
    if drift > _ROTATION_TOL or (np.linalg.det(rot) <= 0).any():
        report.errors.append(f"{path}: cam_c2w rotation is not orthonormal with det +1 (max error {drift:.3g})")

    if "intrinsics" not in data.files:
        report.warnings.append(f"{path}: no intrinsics; training assumes fx = width, fy = height")
        return
    k = np.asarray(data["intrinsics"], dtype=np.float64)
    if k.shape not in ((3, 3), (len(c2w), 3, 3)):
        report.errors.append(f"{path}: intrinsics must be [3, 3] or [N, 3, 3], got {list(k.shape)}")
        return
    k0 = k if k.ndim == 2 else k[0]
    if k0[0, 0] <= 0 or k0[1, 1] <= 0:
        report.errors.append(f"{path}: intrinsics fx and fy must be > 0")
    elif k0[0, 2] > 1.0:  # pixel units: the loader reads the image size as 2*cx x 2*cy
        if abs(k0[0, 2] - width / 2) > _PRINCIPAL_POINT_TOL * width or abs(k0[1, 2] - height / 2) > _PRINCIPAL_POINT_TOL * height:
            report.warnings.append(
                f"{path}: principal point ({k0[0, 2]:.1f}, {k0[1, 2]:.1f}) is not near the centre of the "
                f"{width}x{height} video; pixel intrinsics must be for the video's own resolution"
            )


def layout_from_config(cfg) -> dict:
    """The training window a recipe draws: fps, window length, history prefix, chunk length."""
    from alaya.data.dataloader import _required_train_frames_for_k, _training_event_target_anchor_frame

    ks = [int(k) for k, p in zip(cfg.layout.output.latent_frames, cfg.layout.output.probs) if float(p) > 0]
    if len(ks) != 1:
        raise ValueError(f"standard datasets need exactly one layout.output.latent_frames value, got {ks}")
    k = ks[0]
    return dict(
        fps=float(cfg.sample.fps),
        window_frames=int(_required_train_frames_for_k(cfg, k)),
        prefix_frames=int(_training_event_target_anchor_frame(cfg, k)),
        chunk_frames=k * int(cfg.sample.temporal_stride),
    )
