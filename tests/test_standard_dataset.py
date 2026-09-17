"""Tests for datasets declared in the training config (alaya/data/standard.py).

The loader tests write real (tiny) mp4 files and read them back through
MultiSourceVideoDataset, so they exercise the same decode and window-selection
path as training rather than a stand-in.
"""
import json

import cv2
import numpy as np
import pytest
import yaml

from alaya.config.schema import TrainConfig
from alaya.data.standard import (
    chunk_windows,
    load_timed_prompts,
    parse_dataset_specs,
    register_datasets,
    scan_samples,
    source_config,
)
from fastvideo.dataset.t2v_datasets import MultiSourceVideoDataset

# The 24GB stage2b layout: 25 prefix frames + one K=4 chunk of 32 frames at 24 fps.
FPS = 24.0
PREFIX = 25
CHUNK = 32
WINDOW = PREFIX + CHUNK


def _write_clip(root, clip_id, *, fps=30.0, frames=240, pose=True, caption=None):
    (root / "videos").mkdir(parents=True, exist_ok=True)
    (root / "captions").mkdir(exist_ok=True)
    writer = cv2.VideoWriter(
        str(root / "videos" / f"{clip_id}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 36)
    )
    for i in range(frames):
        writer.write(np.full((36, 64, 3), i % 255, dtype=np.uint8))
    writer.release()
    caption = caption if caption is not None else {"caption": f"clip {clip_id}"}
    (root / "captions" / f"{clip_id}.json").write_text(json.dumps(caption))
    if pose:
        (root / "poses").mkdir(exist_ok=True)
        c2w = np.repeat(np.eye(4, dtype=np.float32)[None], frames, axis=0)
        c2w[:, 2, 3] = np.linspace(0.0, 1.0, frames)
        k = np.array([[50.0, 0, 32.0], [0, 50.0, 18.0], [0, 0, 1]], dtype=np.float32)
        np.savez(root / "poses" / f"{clip_id}.npz", cam_c2w=c2w, intrinsics=k)


# Prompt boundaries on rollout-round boundaries: (25 + 32*2)/24 and (25 + 32*3)/24.
PENCIL = {
    "caption": "a colon that turns into a pencil sketch",
    "segments": [
        {"time_range_s": [0.0, 89 / 24], "prompt": "raw"},
        {"time_range_s": [89 / 24, 121 / 24], "prompt": "transforming"},
        {"time_range_s": [121 / 24, 8.0], "prompt": "sketch"},
    ],
}


def _recipe(datasets):
    with open("configs/stage2b_arsft_lowcompute.yaml", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw["data"]["sources"] = {}
    raw["data"]["datasets"] = datasets
    return TrainConfig.from_mapping(raw)


def _dataset(name, spec_raw):
    MultiSourceVideoDataset.SOURCE_CONFIGS.pop(name, None)
    (spec,) = parse_dataset_specs({name: spec_raw})
    MultiSourceVideoDataset.SOURCE_CONFIGS[name] = source_config(spec)
    return MultiSourceVideoDataset(
        sources=[name],
        width=64,
        height=32,
        target_fps=FPS,
        min_frames=WINDOW,
        max_frames=WINDOW,
        random_frames=True,
        use_cache=False,
        require_camera=True,
        camera_norm_mode="none",
        vae_temporal_factor=8,
        output_latent_frames=4,
        event_target_anchor_frame=PREFIX,
    )


# ---------------------------------------------------------------- spec parsing


def test_parse_accepts_each_format(tmp_path):
    specs = parse_dataset_specs(
        {
            "a": {"root": str(tmp_path), "format": "video_caption_camera"},
            "b": {"root": str(tmp_path), "format": "video_timed_prompts_camera", "prompt_mode": "per_chunk", "weight": 2.0},
            "c": {"root": str(tmp_path), "format": "video_caption_static"},
        }
    )
    assert [(s.name, s.format, s.weight, s.prompt_mode) for s in specs] == [
        ("a", "video_caption_camera", 1.0, None),
        ("b", "video_timed_prompts_camera", 2.0, "per_chunk"),
        ("c", "video_caption_static", 1.0, None),
    ]


@pytest.mark.parametrize(
    "raw, message",
    [
        ({"format": "video_caption_camera"}, "root"),
        ({"root": "x", "format": "video_only"}, "format"),
        ({"root": "x", "format": "video_timed_prompts_camera"}, "prompt_mode"),
        ({"root": "x", "format": "video_timed_prompts_camera", "prompt_mode": "random"}, "prompt_mode"),
        ({"root": "x", "format": "video_caption_camera", "prompt_mode": "segment"}, "prompt_mode"),
        ({"root": "x", "format": "video_caption_camera", "weight": -1}, "weight"),
        ({"root": "x", "format": "video_caption_camera", "colour": 1}, "colour"),
    ],
)
def test_parse_rejects_bad_specs(raw, message):
    with pytest.raises(ValueError, match=message):
        parse_dataset_specs({"d": raw})


def test_parse_rejects_builtin_source_name():
    with pytest.raises(ValueError, match="built-in"):
        parse_dataset_specs({"spatialvid_hq": {"root": "x", "format": "video_caption_camera"}})


def test_train_config_accepts_datasets_and_validates_them(tmp_path):
    cfg = _recipe({"mine": {"root": str(tmp_path), "format": "video_caption_static"}})
    assert cfg.data.datasets["mine"]["format"] == "video_caption_static"
    with pytest.raises(ValueError, match="format"):
        _recipe({"mine": {"root": "x", "format": "nope"}})


# ---------------------------------------------------------------- source config


def test_source_config_per_format(tmp_path):
    def cfg_for(**raw):
        (spec,) = parse_dataset_specs({"d": {"root": str(tmp_path), **raw}})
        return source_config(spec)

    camera = cfg_for(format="video_caption_camera")
    assert camera["has_camera"] and not camera.get("static_camera")
    assert not camera["use_segment_caption"] and camera.get("timed_prompt_mode") is None

    static = cfg_for(format="video_caption_static")
    assert static["has_camera"] and static["static_camera"]

    segment = cfg_for(format="video_timed_prompts_camera", prompt_mode="segment")
    assert segment["use_segment_caption"] and segment["segment_caption_field"] == "prompt"

    chunk = cfg_for(format="video_timed_prompts_camera", prompt_mode="per_chunk")
    assert not chunk["use_segment_caption"] and chunk["timed_prompt_mode"] == "per_chunk"


def test_register_datasets_adds_source_configs(tmp_path):
    cfg = _recipe({"reg_test": {"root": str(tmp_path), "format": "video_caption_camera"}})
    MultiSourceVideoDataset.SOURCE_CONFIGS.pop("reg_test", None)
    register_datasets(cfg)
    assert MultiSourceVideoDataset.SOURCE_CONFIGS["reg_test"]["standard_root"] == str(tmp_path)


def test_train_sources_merge_builtin_and_declared(tmp_path):
    from alaya.data.dataloader import _train_sources

    cfg = _recipe({"mine": {"root": str(tmp_path), "format": "video_caption_static", "weight": 3.0}})
    cfg.data.sources = {"spatialvid_hq": 1.0, "off": 0.0}
    assert _train_sources(cfg) == [
        ("spatialvid_hq", ["spatialvid_hq"], 1.0, True),
        ("mine", ["mine"], 3.0, False),
    ]
    assert MultiSourceVideoDataset.SOURCE_CONFIGS["mine"]["static_camera"]


# ---------------------------------------------------------------- scanning


def test_scan_samples_pairs_files_by_stem(tmp_path):
    _write_clip(tmp_path, "b", frames=10)
    _write_clip(tmp_path, "a", frames=10)
    (spec,) = parse_dataset_specs({"d": {"root": str(tmp_path), "format": "video_caption_camera"}})
    samples = scan_samples("d", source_config(spec))
    assert samples == [
        (str(tmp_path / "videos/a.mp4"), str(tmp_path / "captions/a.json"), str(tmp_path / "poses/a.npz"), "d", "a"),
        (str(tmp_path / "videos/b.mp4"), str(tmp_path / "captions/b.json"), str(tmp_path / "poses/b.npz"), "d", "b"),
    ]


def test_scan_samples_static_has_no_pose(tmp_path):
    _write_clip(tmp_path, "a", frames=10, pose=False)
    (spec,) = parse_dataset_specs({"d": {"root": str(tmp_path), "format": "video_caption_static"}})
    assert scan_samples("d", source_config(spec))[0][2] is None


# ---------------------------------------------------------------- timed prompts


def test_load_timed_prompts_sorts_by_start(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"caption": "x", "segments": list(reversed(PENCIL["segments"]))}))
    assert [p for _, _, p in load_timed_prompts(str(path))] == ["raw", "transforming", "sketch"]


def test_chunk_windows_follow_rollout_rounds():
    prompts = [(a, b, p) for (a, b), p in ((s["time_range_s"], s["prompt"]) for s in PENCIL["segments"])]
    windows = chunk_windows(
        prompts, available_frames=192, window_frames=WINDOW, prefix_frames=PREFIX, chunk_frames=CHUNK, fps=FPS
    )
    # 8 s at 24 fps = 192 frames -> rounds start at 0, 32, 64, 96, 128 (128 + 57 = 185 <= 192).
    assert windows == [(0, "raw"), (32, "raw"), (64, "transforming"), (96, "sketch"), (128, "sketch")]


def test_chunk_windows_skip_rounds_without_a_prompt():
    windows = chunk_windows(
        [(0.0, 3.0, "only")], available_frames=192, window_frames=WINDOW, prefix_frames=PREFIX, chunk_frames=CHUNK, fps=FPS
    )
    # Target midpoints: round 1 at (25+16)/24 = 1.71 s, round 2 at 3.04 s (uncovered).
    assert windows == [(0, "only")]


# ---------------------------------------------------------------- loader


def test_loader_per_chunk_windows_match_rounds(tmp_path):
    _write_clip(tmp_path, "clip", caption=PENCIL)
    ds = _dataset("t_chunk", {"root": str(tmp_path), "format": "video_timed_prompts_camera", "prompt_mode": "per_chunk"})
    seen = set()
    for epoch in range(40):
        ds.set_epoch(epoch)
        item = ds[0]
        pixels, caption, cam_c2w, frame_start = item[0], item[2], item[4], item[11]
        assert pixels.shape[0] == WINDOW and cam_c2w.shape[0] == WINDOW
        seen.add((frame_start, caption))
    # 30 fps source: round r starts at native frame 40*(r-1).
    assert seen == {(0, "raw"), (40, "raw"), (80, "transforming"), (120, "sketch"), (160, "sketch")}


def test_loader_segment_windows_stay_inside_one_prompt(tmp_path, monkeypatch):
    monkeypatch.setenv("LTX_OVERALL_CAPTION_PROB", "0.0")
    _write_clip(tmp_path, "clip", caption=PENCIL)
    ds = _dataset("t_seg", {"root": str(tmp_path), "format": "video_timed_prompts_camera", "prompt_mode": "segment"})
    ranges = {s["prompt"]: s["time_range_s"] for s in PENCIL["segments"]}
    captions = set()
    for epoch in range(40):
        ds.set_epoch(epoch)
        item = ds[0]
        caption, start, end = item[2], item[11], item[12]
        lo, hi = ranges[caption]
        assert lo * 30 - 1 <= start and end <= hi * 30 + 1
        captions.add(caption)
    # "transforming" lasts 1.33 s, shorter than the 2.375 s window, so it is never drawn.
    assert captions == {"raw", "sketch"}


def test_loader_static_camera_is_identity(tmp_path):
    _write_clip(tmp_path, "clip", pose=False)
    ds = _dataset("t_static", {"root": str(tmp_path), "format": "video_caption_static"})
    item = ds[0]
    assert item[2] == "clip clip"
    assert np.allclose(item[4].numpy(), np.eye(4)[None])


def test_loader_caption_camera_reads_pose(tmp_path):
    _write_clip(tmp_path, "clip")
    ds = _dataset("t_cam", {"root": str(tmp_path), "format": "video_caption_camera"})
    item = ds[0]
    assert item[2] == "clip clip" and bool(item[6])
    assert item[4].shape[0] == WINDOW and float(item[4][-1, 2, 3]) > float(item[4][0, 2, 3])


# ---------------------------------------------------------------- prompt precache


def _precache_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("precache_tool", "scripts/tools/precache_train_text_embeds.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_precache_accepts_timed_prompts(tmp_path, monkeypatch):
    monkeypatch.delenv("LTX_SEGMENT_CONCAT_PROB", raising=False)
    tool = _precache_module()
    for mode in ("segment", "per_chunk"):
        cfg = _recipe({"timed": {"root": str(tmp_path), "format": "video_timed_prompts_camera", "prompt_mode": mode}})
        tool._assert_finite_prompt_set(cfg)


def test_precache_refuses_segment_concatenation(tmp_path, monkeypatch):
    monkeypatch.setenv("LTX_SEGMENT_CONCAT_PROB", "0.5")
    cfg = _recipe({"timed": {"root": str(tmp_path), "format": "video_timed_prompts_camera", "prompt_mode": "segment"}})
    with pytest.raises(SystemExit, match="LTX_SEGMENT_CONCAT_PROB"):
        _precache_module()._assert_finite_prompt_set(cfg)


def test_enumerated_prompts_cover_timed_captions(tmp_path):
    from alaya.data.text_embed_cache import enumerate_all_prompts

    _write_clip(tmp_path, "clip", frames=10, caption=PENCIL)
    cfg = _recipe({"t_enum": {"root": str(tmp_path), "format": "video_timed_prompts_camera", "prompt_mode": "per_chunk"}})
    ds = _dataset("t_enum", cfg.data.datasets["t_enum"])
    prompts = set(enumerate_all_prompts(cfg, ds))
    assert {"raw", "transforming", "sketch", PENCIL["caption"]} <= prompts


def test_dataloader_with_fewer_samples_than_ranks_fails_loudly():
    from torch.utils.data import DataLoader, DistributedSampler

    from alaya.data.dataloader import _require_batches

    data = list(range(3))
    loader = DataLoader(data, batch_size=1, drop_last=True,
                        sampler=DistributedSampler(data, num_replicas=4, rank=0, drop_last=True))
    with pytest.raises(ValueError, match="3 samples.*4 GPUs"):
        _require_batches(loader, world_size=4, batch_size=1)

    enough = list(range(4))
    loader = DataLoader(enough, batch_size=1, drop_last=True,
                        sampler=DistributedSampler(enough, num_replicas=4, rank=0, drop_last=True))
    assert _require_batches(loader, world_size=4, batch_size=1) is loader


def test_loader_per_chunk_accepts_ntsc_24fps(tmp_path):
    _write_clip(tmp_path, "clip", fps=24000 / 1001, frames=192, caption=PENCIL)
    ds = _dataset("t_ntsc", {"root": str(tmp_path), "format": "video_timed_prompts_camera", "prompt_mode": "per_chunk"})
    seen = set()
    for epoch in range(40):
        ds.set_epoch(epoch)
        item = ds[0]
        assert item[0].shape[0] == WINDOW
        seen.add((item[11], item[2]))
    # 192 frames at 23.976 fps give 191 frames at 24 fps, so rounds 1-5 still fit (128 + 57 <= 191).
    assert {caption for _, caption in seen} == {"raw", "transforming", "sketch"}
    assert sorted(start for start, _ in seen) == [0, 31, 63, 95, 127]


def test_precache_includes_literal_prompt_schedule_entries(tmp_path):
    cfg = _recipe({"timed": {"root": str(tmp_path), "format": "video_caption_camera"}})
    mode = next(iter(cfg.validation.modes.values()))
    mode.prompt_schedule = ["caption", "RAW", "The scene turns into a pencil sketch.", "caption"]
    assert _precache_module()._schedule_prompts(cfg) == ["The scene turns into a pencil sketch."]
