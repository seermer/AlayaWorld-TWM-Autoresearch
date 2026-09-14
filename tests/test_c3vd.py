"""Unit tests for the C3VD overfit-set conversion.

Covers the pieces that would silently corrupt the experiment: pose parsing
(C3VD stores 4x4 matrices column-major, in millimetres), clip planning, the
crop to the training aspect ratio, and the raw -> pencil-sketch transition,
whose timing must line up with the rollout chunk grid.
"""
import numpy as np
import pytest

from alaya.data.c3vd import (
    CLIP_FPS,
    CLIP_FRAMES,
    OUT_H,
    OUT_W,
    TRANSITION_END_S,
    TRANSITION_START_S,
    PHASE_CAPTIONS,
    WINDOW_FRAMES,
    caption_json,
    crop_resize,
    parse_c3vd_pose,
    phase_for_round,
    pencil_blend,
    pinhole_k,
    plan_clips,
    scale_translation,
    segment_name,
    sketch_weight,
    window_native_indices,
)


def _pose_line(c2w):
    # C3VD writes the matrix column-major: the translation is the last 4 values.
    return ",".join(f"{v:.6f}" for v in c2w.T.reshape(-1))


def test_parse_c3vd_pose_reads_column_major_camera_to_world():
    c2w = np.eye(4)
    c2w[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    c2w[:3, 3] = [-275.8, 25.4, -409.0]
    text = _pose_line(c2w) + "\n" + _pose_line(c2w) + "\n"

    out = parse_c3vd_pose(text)

    assert out.shape == (2, 4, 4)
    np.testing.assert_allclose(out[0], c2w, atol=1e-5)


def test_parse_c3vd_pose_rejects_wrong_value_count():
    with pytest.raises(ValueError):
        parse_c3vd_pose("1,0,0,0,0,1,0,0\n")


def test_parse_c3vd_pose_rejects_non_rigid_matrix():
    bad = np.eye(4)
    bad[0, 0] = 3.0
    with pytest.raises(ValueError):
        parse_c3vd_pose(_pose_line(bad))


def test_scale_translation_changes_only_translation():
    c2w = np.tile(np.eye(4), (2, 1, 1))
    c2w[:, :3, 3] = [[100.0, -20.0, 5.0], [110.0, -20.0, 5.0]]

    out = scale_translation(c2w, 0.1)

    np.testing.assert_allclose(out[:, :3, 3], c2w[:, :3, 3] * 0.1)
    np.testing.assert_allclose(out[:, :3, :3], c2w[:, :3, :3])
    np.testing.assert_allclose(out[:, 3], c2w[:, 3])


def test_plan_clips_is_non_overlapping_and_capped():
    assert plan_clips(765) == [0, CLIP_FRAMES, 2 * CLIP_FRAMES]
    assert plan_clips(1142) == [0, CLIP_FRAMES, 2 * CLIP_FRAMES]  # capped at 3
    assert plan_clips(515) == [0, CLIP_FRAMES]
    assert plan_clips(CLIP_FRAMES - 1) == []


def test_clip_is_long_enough_for_a_five_round_rollout():
    # 25-frame prefix + 5 rounds x 32 frames, sampled at 24 fps from a CLIP_FPS video.
    needed_native = (25 + 5 * 32) * CLIP_FPS / 24.0
    assert CLIP_FRAMES >= needed_native


def test_crop_resize_outputs_training_resolution_and_keeps_the_centre():
    img = np.zeros((1080, 1350, 3), np.uint8)
    img[400:680, 560:790] = 255  # a bright block around the centre

    out = crop_resize(img)

    assert out.shape == (OUT_H, OUT_W, 3)
    assert out[OUT_H // 2, OUT_W // 2].min() == 255
    assert out[5, 5].max() == 0


def test_pinhole_k_is_pixel_space_and_centred():
    k = pinhole_k(hfov_deg=90.0)

    assert k.shape == (3, 3)
    # cx > 1 is how the loader recognises pixel units and recovers the resolution.
    assert k[0, 2] == pytest.approx(OUT_W / 2)
    assert k[1, 2] == pytest.approx(OUT_H / 2)
    assert k[0, 0] == pytest.approx(OUT_W / 2)  # tan(45 deg) == 1
    assert k[0, 0] == pytest.approx(k[1, 1])


def test_transition_covers_the_third_rollout_chunk():
    # Prediction frame 0 is GT frame 24 at 24 fps; round r covers GT frames [25+32(r-1), 25+32r).
    assert TRANSITION_START_S == pytest.approx((25 + 2 * 32) / 24.0)
    assert TRANSITION_END_S == pytest.approx((25 + 3 * 32) / 24.0)


def test_sketch_weight_is_a_smooth_ramp():
    assert sketch_weight(0.0) == 0.0
    assert sketch_weight(TRANSITION_START_S) == 0.0
    assert sketch_weight(TRANSITION_END_S) == 1.0
    assert sketch_weight(CLIP_FRAMES / CLIP_FPS) == 1.0
    mid = 0.5 * (TRANSITION_START_S + TRANSITION_END_S)
    assert sketch_weight(mid) == pytest.approx(0.5)
    ts = np.linspace(TRANSITION_START_S, TRANSITION_END_S, 50)
    ws = [sketch_weight(t) for t in ts]
    assert all(b >= a for a, b in zip(ws, ws[1:]))


def test_pencil_blend_endpoints():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, (OUT_H, OUT_W, 3), dtype=np.uint8)

    raw = pencil_blend(frame, 0.0)
    sketch = pencil_blend(frame, 1.0)
    half = pencil_blend(frame, 0.5)

    np.testing.assert_array_equal(raw, frame)
    assert sketch.dtype == np.uint8
    # the pencil sketch is grayscale: all three channels agree
    np.testing.assert_array_equal(sketch[..., 0], sketch[..., 1])
    np.testing.assert_array_equal(sketch[..., 1], sketch[..., 2])
    assert not np.array_equal(half, frame) and not np.array_equal(half, sketch)


@pytest.mark.parametrize(
    "sequence, expected",
    [
        ("cecum_t1_a", "cecum"),
        ("cecum_t4_b", "cecum"),
        ("sigmoid_t2_a", "sigmoid colon"),
        ("trans_t4_b", "transverse colon"),
        ("desc_t4_a", "descending colon"),
    ],
)
def test_segment_name_covers_c3vd_v1_naming(sequence, expected):
    assert segment_name(sequence) == expected


def test_segment_name_rejects_unknown():
    with pytest.raises(ValueError):
        segment_name("stomach_t1_a")


def test_window_is_one_rollout_round():
    # a 25-frame prefix followed by one 32-frame target chunk, exactly like a rollout round
    assert WINDOW_FRAMES == 25 + 32


def test_window_native_indices_match_the_eval_loader_sampling():
    # the eval loader reads native frame int(g * 30/24) for 24 fps frame g, starting at 0
    ratio = CLIP_FPS / 24.0
    r1 = window_native_indices(1)
    r3 = window_native_indices(3)
    assert r1 == [int(k * ratio) for k in range(WINDOW_FRAMES)]
    assert r3 == [int((64 + k) * ratio) for k in range(WINDOW_FRAMES)]
    assert len(window_native_indices(5)) == WINDOW_FRAMES
    assert max(window_native_indices(5)) < CLIP_FRAMES


def test_round_phases_follow_the_transition():
    assert [phase_for_round(r) for r in range(1, 6)] == ["raw", "raw", "transforming", "sketch", "sketch"]


def test_phase_captions_describe_their_phase():
    raw, mid, sketch = PHASE_CAPTIONS["raw"], PHASE_CAPTIONS["transforming"], PHASE_CAPTIONS["sketch"]
    assert len({raw, mid, sketch}) == 3
    assert "colonoscop" in raw.lower() and "pencil" not in raw.lower()
    assert "transform" in mid.lower() and "pencil sketch" in mid.lower()
    assert "pencil sketch" in sketch.lower()
    # schedule entries are used verbatim, so no surrounding whitespace
    assert all(c == c.strip() for c in PHASE_CAPTIONS.values())


def test_caption_json_uses_the_text_for_every_field_the_loader_reads():
    cap = caption_json(PHASE_CAPTIONS["sketch"])
    assert cap["overall_caption"] == PHASE_CAPTIONS["sketch"]
    assert cap["overall"]["full_prompt"] == PHASE_CAPTIONS["sketch"]
    assert cap["overall"]["short_prompt"]
