"""Datasets declared in the training config instead of registered in code.

A declared dataset is a directory in one of three standard formats (docs/TRAINING.md):

    <root>/videos/<id>.mp4      every format
    <root>/captions/<id>.json   every format: {"caption": ...}, plus "segments" for timed prompts
    <root>/poses/<id>.npz       camera formats: cam_c2w [N,4,4], optional intrinsics [3,3]

and is named under data.datasets:

    data:
      datasets:
        my_clips:
          root: /path/to/my_clips
          format: video_timed_prompts_camera
          prompt_mode: per_chunk
          weight: 1.0

register_datasets() turns each entry into a MultiSourceVideoDataset source, so the
loader, the prompt precache and validation all see it like a built-in source.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

FORMATS = ("video_caption_camera", "video_timed_prompts_camera", "video_caption_static")
PROMPT_MODES = ("segment", "per_chunk")
_SPEC_KEYS = {"root", "format", "prompt_mode", "weight"}
# 23.976 fps ("24 fps" video) counts as the 24 fps training rate.
FPS_TOLERANCE = 0.01


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: str
    format: str
    weight: float = 1.0
    prompt_mode: str | None = None


def parse_dataset_specs(raw: dict) -> list[DatasetSpec]:
    """Validate data.datasets and return one spec per entry, in config order."""
    from fastvideo.dataset.t2v_datasets import MultiSourceVideoDataset

    if not isinstance(raw, dict):
        raise ValueError("data.datasets must be a mapping of name -> dataset")
    builtin = {k for k, v in MultiSourceVideoDataset.SOURCE_CONFIGS.items() if "standard_root" not in v}
    specs = []
    for name, entry in raw.items():
        where = f"data.datasets.{name}"
        if not isinstance(name, str) or not name:
            raise ValueError("data.datasets names must be non-empty strings")
        if name in builtin:
            raise ValueError(f"{where}: {name!r} is a built-in source name; pick another")
        if not isinstance(entry, dict):
            raise ValueError(f"{where} must be a mapping")
        unknown = sorted(set(entry) - _SPEC_KEYS)
        if unknown:
            raise ValueError(f"{where}: unknown keys {unknown}; allowed: {sorted(_SPEC_KEYS)}")
        if not entry.get("root"):
            raise ValueError(f"{where}.root is required")
        fmt = entry.get("format")
        if fmt not in FORMATS:
            raise ValueError(f"{where}.format must be one of {list(FORMATS)}, got {fmt!r}")
        mode = entry.get("prompt_mode")
        if fmt == "video_timed_prompts_camera":
            if mode not in PROMPT_MODES:
                raise ValueError(f"{where}.prompt_mode must be one of {list(PROMPT_MODES)} for {fmt}, got {mode!r}")
        elif mode is not None:
            raise ValueError(f"{where}.prompt_mode only applies to video_timed_prompts_camera")
        weight = float(entry.get("weight", 1.0))
        if weight < 0:
            raise ValueError(f"{where}.weight must be >= 0")
        specs.append(DatasetSpec(name=name, root=str(entry["root"]), format=fmt, weight=weight, prompt_mode=mode))
    return specs


def source_config(spec: DatasetSpec) -> dict:
    """The MultiSourceVideoDataset.SOURCE_CONFIGS entry for a declared dataset."""
    timed = spec.format == "video_timed_prompts_camera"
    return {
        "standard_root": spec.root,
        "has_camera": True,
        "static_camera": spec.format == "video_caption_static",
        # Only used when intrinsics are normalized (cx <= 1) or absent.
        "original_width": 1280.0,
        "original_height": 720.0,
        "fallback_default_intrinsic": True,
        "use_segment_caption": timed and spec.prompt_mode == "segment",
        "segment_caption_field": "prompt",
        "overall_caption_field": "caption",
        "timed_prompt_mode": spec.prompt_mode if timed else None,
        # Read by the legacy scanner only; kept so code that inspects them does not KeyError.
        "jsonl": "",
        "video_subdir": "",
        "caption_subdir": "",
        "pose_subdir": "",
    }


def register_datasets(cfg) -> list[DatasetSpec]:
    """Add every data.datasets entry to MultiSourceVideoDataset.SOURCE_CONFIGS. Idempotent."""
    from fastvideo.dataset.t2v_datasets import MultiSourceVideoDataset

    specs = parse_dataset_specs(getattr(cfg.data, "datasets", None) or {})
    for spec in specs:
        MultiSourceVideoDataset.SOURCE_CONFIGS[spec.name] = source_config(spec)
    return specs


def scan_samples(source_name: str, config: dict) -> list[tuple]:
    """One (video, caption, pose, source, id) tuple per videos/*.mp4, sorted by id.

    Missing captions or poses are not filtered here: scripts/tools/check_dataset.py
    reports them, and the loader skips a broken sample at read time.
    """
    root = config["standard_root"]
    video_dir = os.path.join(root, "videos")
    if not os.path.isdir(video_dir):
        raise FileNotFoundError(f"{source_name}: no videos/ directory under {root}")
    samples = []
    for fname in sorted(os.listdir(video_dir)):
        clip_id, ext = os.path.splitext(fname)
        if ext.lower() != ".mp4":
            continue
        pose = None if config.get("static_camera") else os.path.join(root, "poses", f"{clip_id}.npz")
        samples.append(
            (os.path.join(video_dir, fname), os.path.join(root, "captions", f"{clip_id}.json"), pose, source_name, clip_id)
        )
    return samples


def load_timed_prompts(caption_path: str) -> list[tuple[float, float, str]]:
    """(start_s, end_s, prompt) for every segment of a caption json, sorted by start."""
    with open(caption_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    prompts = [
        (float(seg["time_range_s"][0]), float(seg["time_range_s"][1]), str(seg["prompt"]))
        for seg in data.get("segments") or []
    ]
    return sorted(prompts, key=lambda p: p[0])


def chunk_windows(
    prompts: list[tuple[float, float, str]],
    *,
    available_frames: int,
    window_frames: int,
    prefix_frames: int,
    chunk_frames: int,
    fps: float,
) -> list[tuple[int, str]]:
    """Training windows that line up with interactive rollout rounds.

    Round r (1-based) of a rollout generates frames [prefix + chunk*(r-1), prefix + chunk*r)
    of the clip, so its training window starts at chunk*(r-1): the first `prefix_frames`
    frames are history and the rest is the target. The caption is the prompt active at
    the target's midpoint. Returns (start_frame, prompt) at the target fps; rounds whose
    window runs past the clip or whose target has no prompt are left out.
    """
    windows = []
    start = 0
    while start + window_frames <= available_frames:
        mid_s = (start + prefix_frames + chunk_frames / 2.0) / fps
        active = [p for lo, hi, p in prompts if lo <= mid_s < hi]
        if active:
            windows.append((start, active[0]))
        start += chunk_frames
    return windows
