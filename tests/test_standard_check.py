"""Tests for the standard-format dataset checker (alaya/data/standard_check.py)."""
import json

import numpy as np
import pytest

from alaya.data.standard import parse_dataset_specs
from alaya.data.standard_check import check_dataset

from test_standard_dataset import PENCIL, _write_clip

LAYOUT = dict(fps=24.0, window_frames=57, prefix_frames=25, chunk_frames=32)


def _check(root, fmt, mode=None):
    raw = {"root": str(root), "format": fmt}
    if mode:
        raw["prompt_mode"] = mode
    (spec,) = parse_dataset_specs({"d": raw})
    return check_dataset(spec, **LAYOUT)


def _has(messages, text):
    return any(text in m for m in messages)


def test_clean_datasets_pass(tmp_path):
    _write_clip(tmp_path / "cam", "a")
    report = _check(tmp_path / "cam", "video_caption_camera")
    assert report.errors == [] and report.clips == 1 and report.windows == 1

    _write_clip(tmp_path / "static", "a", pose=False)
    report = _check(tmp_path / "static", "video_caption_static")
    assert report.errors == [] and report.warnings == []

    _write_clip(tmp_path / "timed", "a", caption=PENCIL)
    report = _check(tmp_path / "timed", "video_timed_prompts_camera", "per_chunk")
    assert report.errors == [] and report.windows == 5


def test_missing_root_or_videos(tmp_path):
    assert _has(_check(tmp_path / "nope", "video_caption_camera").errors, "videos/")


def test_missing_caption_and_pose(tmp_path):
    _write_clip(tmp_path, "a")
    (tmp_path / "captions/a.json").unlink()
    (tmp_path / "poses/a.npz").unlink()
    errors = _check(tmp_path, "video_caption_camera").errors
    assert _has(errors, "captions/a.json") and _has(errors, "poses/a.npz")


def test_empty_caption(tmp_path):
    _write_clip(tmp_path, "a", caption={"caption": "  "})
    assert _has(_check(tmp_path, "video_caption_camera").errors, '"caption"')


def test_clip_shorter_than_window(tmp_path):
    _write_clip(tmp_path, "a", frames=60)  # 2 s at 30 fps < 57/24 s
    assert _has(_check(tmp_path, "video_caption_camera").errors, "shorter than")


def test_fps_below_training_fps(tmp_path):
    _write_clip(tmp_path, "a", fps=15.0, frames=150)
    assert _has(_check(tmp_path, "video_caption_camera").errors, "fps")


def test_pose_count_must_match_frames(tmp_path):
    _write_clip(tmp_path, "a")
    data = dict(np.load(tmp_path / "poses/a.npz"))
    np.savez(tmp_path / "poses/a.npz", cam_c2w=data["cam_c2w"][:200], intrinsics=data["intrinsics"])
    assert _has(_check(tmp_path, "video_caption_camera").errors, "200 poses")


def test_pose_must_be_rigid(tmp_path):
    _write_clip(tmp_path, "a")
    data = dict(np.load(tmp_path / "poses/a.npz"))
    data["cam_c2w"][:, 0, 0] = 2.0
    np.savez(tmp_path / "poses/a.npz", **data)
    assert _has(_check(tmp_path, "video_caption_camera").errors, "rotation")


def test_missing_intrinsics_warns(tmp_path):
    _write_clip(tmp_path, "a")
    data = dict(np.load(tmp_path / "poses/a.npz"))
    np.savez(tmp_path / "poses/a.npz", cam_c2w=data["cam_c2w"])
    report = _check(tmp_path, "video_caption_camera")
    assert report.errors == [] and _has(report.warnings, "intrinsics")


def test_static_ignores_poses_with_warning(tmp_path):
    _write_clip(tmp_path, "a")
    report = _check(tmp_path, "video_caption_static")
    assert report.errors == [] and _has(report.warnings, "poses/")


def test_timed_prompts_need_segments(tmp_path):
    _write_clip(tmp_path, "a", caption={"caption": "x"})
    assert _has(_check(tmp_path, "video_timed_prompts_camera", "segment").errors, "segments")


def test_timed_prompts_out_of_range_and_overlap(tmp_path):
    caption = {
        "caption": "x",
        "segments": [
            {"time_range_s": [0.0, 5.0], "prompt": "a"},
            {"time_range_s": [4.0, 9.5], "prompt": "b"},
        ],
    }
    _write_clip(tmp_path, "a", caption=caption)
    errors = _check(tmp_path, "video_timed_prompts_camera", "segment").errors
    assert _has(errors, "overlap") and _has(errors, "past the end")


def test_segment_mode_warns_about_short_segments(tmp_path):
    _write_clip(tmp_path, "a", caption=PENCIL)
    report = _check(tmp_path, "video_timed_prompts_camera", "segment")
    assert report.errors == [] and _has(report.warnings, "transforming")
    assert report.windows == 2  # "raw" and "sketch" are long enough to hold a window


def test_per_chunk_needs_aligned_boundaries(tmp_path):
    caption = {
        "caption": "x",
        "segments": [
            {"time_range_s": [0.0, 4.0], "prompt": "a"},
            {"time_range_s": [4.0, 8.0], "prompt": "b"},
        ],
    }
    _write_clip(tmp_path, "a", caption=caption)
    errors = _check(tmp_path, "video_timed_prompts_camera", "per_chunk").errors
    # 4.0 s = frame 96; the nearest round boundaries are frames 89 and 121.
    assert _has(errors, "4.000") and _has(errors, "3.708")


def test_layout_from_lowcompute_recipe(tmp_path):
    from alaya.data.standard_check import layout_from_config
    from test_standard_dataset import _recipe

    cfg = _recipe({"d": {"root": str(tmp_path), "format": "video_caption_camera"}})
    assert layout_from_config(cfg) == LAYOUT


def test_cli_exit_code(tmp_path):
    import subprocess
    import sys

    import yaml

    _write_clip(tmp_path / "data", "a", frames=60)  # too short -> error
    with open("configs/stage2b_arsft_lowcompute.yaml", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw["data"]["sources"] = {}
    raw["data"]["datasets"] = {"short": {"root": str(tmp_path / "data"), "format": "video_caption_camera"}}
    config = tmp_path / "cfg.yaml"
    config.write_text(yaml.safe_dump(raw))
    result = subprocess.run(
        [sys.executable, "scripts/tools/check_dataset.py", "--config", str(config)], capture_output=True, text=True
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "shorter than" in result.stdout


def test_ntsc_24fps_is_accepted(tmp_path):
    _write_clip(tmp_path, "a", fps=24000 / 1001, frames=240)
    assert _check(tmp_path, "video_caption_camera").errors == []


def test_precache_cli_counts_the_empty_negative_prompt(tmp_path):
    """End to end through the CLI: '' must appear in the prompt set it would encode."""
    import subprocess
    import sys

    import yaml

    from test_standard_dataset import _write_clip as write_clip

    write_clip(tmp_path / "data", "a")
    with open("configs/stage2b_arsft_lowcompute.yaml", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw["data"]["sources"] = {}
    raw["data"]["datasets"] = {"neg": {"root": str(tmp_path / "data"), "format": "video_caption_camera"}}
    raw["validation"]["negative_prompt"] = ""
    config = tmp_path / "cfg.yaml"
    config.write_text(yaml.safe_dump(raw))
    result = subprocess.run(
        [sys.executable, "scripts/tools/precache_train_text_embeds.py", "--config", str(config), "--dry-run"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    # one caption + the empty negative prompt
    assert "2 distinct prompts" in result.stdout, result.stdout
