"""Unit tests for the SpatialVID -> Alaya schema conversion.

These run on constructed arrays, not on the dataset: they check the maths
(w2c/c2w inversion, slerp endpoints, pixel-space intrinsics) that would otherwise
fail silently and produce a model trained on wrong geometry.
"""
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from alaya.data.spatialvid import (
    caption_to_alaya,
    interpolate_c2w,
    intrinsics_to_pixel_k,
    parse_indexes,
    poses_to_c2w,
)


def _w2c_row_from_c2w(c2w):
    """Build the (7,) [t|quat] w2c row that SpatialVID would store for this c2w."""
    r_c2w = c2w[:3, :3]
    t_c2w = c2w[:3, 3]
    r_w2c = r_c2w.T
    t_w2c = -r_w2c @ t_c2w
    quat = Rotation.from_matrix(r_w2c).as_quat()  # scipy order: x, y, z, w
    return np.concatenate([t_w2c, quat])


def test_poses_to_c2w_inverts_world_to_camera():
    rng = np.random.default_rng(0)
    c2w = np.tile(np.eye(4), (3, 1, 1))
    for i in range(3):
        c2w[i, :3, :3] = Rotation.random(random_state=rng.integers(1 << 30)).as_matrix()
        c2w[i, :3, 3] = rng.normal(size=3)
    rows = np.stack([_w2c_row_from_c2w(c2w[i]) for i in range(3)])

    out = poses_to_c2w(rows)

    assert out.shape == (3, 4, 4)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, c2w.astype(np.float32), atol=1e-5)


def test_poses_to_c2w_rejects_wrong_shape():
    with pytest.raises(ValueError):
        poses_to_c2w(np.zeros((3, 4)))


def test_parse_indexes_reads_the_real_format():
    text = "# total 3 indexes\n0 0\n1 6\n2 12\n"
    np.testing.assert_array_equal(parse_indexes(text), np.array([0, 6, 12]))


def test_parse_indexes_rejects_empty():
    with pytest.raises(ValueError):
        parse_indexes("# total 0 indexes\n")


def test_interpolate_c2w_reproduces_annotated_frames_exactly():
    src = np.array([0, 6, 12])
    c2w = np.tile(np.eye(4), (3, 1, 1))
    c2w[:, :3, 3] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    for i, ang in enumerate([0.0, 0.4, 0.8]):
        c2w[i, :3, :3] = Rotation.from_rotvec([0, ang, 0]).as_matrix()

    out = interpolate_c2w(c2w, src, num_frames=13)

    assert out.shape == (13, 4, 4)
    for i, frame in enumerate(src):
        np.testing.assert_allclose(out[frame], c2w[i].astype(np.float32), atol=1e-5)


def test_interpolate_c2w_is_linear_in_translation_between_samples():
    src = np.array([0, 10])
    c2w = np.tile(np.eye(4), (2, 1, 1))
    c2w[1, :3, 3] = [10.0, 0.0, 0.0]

    out = interpolate_c2w(c2w, src, num_frames=11)

    np.testing.assert_allclose(out[5, :3, 3], [5.0, 0.0, 0.0], atol=1e-5)


def test_interpolate_c2w_clamps_past_the_last_annotation():
    src = np.array([0, 4])
    c2w = np.tile(np.eye(4), (2, 1, 1))
    c2w[1, :3, 3] = [4.0, 0.0, 0.0]

    out = interpolate_c2w(c2w, src, num_frames=8)

    np.testing.assert_allclose(out[7], out[4], atol=1e-6)


def test_interpolate_c2w_needs_two_annotations():
    with pytest.raises(ValueError):
        interpolate_c2w(np.eye(4)[None], np.array([0]), num_frames=4)


def test_intrinsics_to_pixel_k_scales_by_resolution():
    intr = np.array([[0.79311365, 1.4099798, 0.5, 0.5]] * 4)

    k = intrinsics_to_pixel_k(intr, width=1280, height=720)

    assert k.shape == (3, 3)
    np.testing.assert_allclose(k[0, 0], 0.79311365 * 1280, rtol=1e-6)
    np.testing.assert_allclose(k[1, 1], 1.4099798 * 720, rtol=1e-6)
    # cx > 1 is the signal the loader uses to detect pixel units
    assert k[0, 2] == pytest.approx(640.0)
    assert k[1, 2] == pytest.approx(360.0)


def test_intrinsics_to_pixel_k_uses_the_median_not_an_outlier():
    intr = np.array([[0.8, 1.4, 0.5, 0.5]] * 5 + [[99.0, 99.0, 0.5, 0.5]])

    k = intrinsics_to_pixel_k(intr, width=100, height=100)

    np.testing.assert_allclose(k[0, 0], 80.0, rtol=1e-6)


def test_caption_to_alaya_uses_dataset_text_verbatim():
    src = {"SceneSummary": "A city street.", "SceneDescription": "Wide shot of a street.",
           "CameraMotion": "pans right"}

    out = caption_to_alaya(src)

    assert out["overall"]["short_prompt"] == "A city street."
    assert out["overall"]["full_prompt"] == "Wide shot of a street."
    assert out["overall_caption"] == "A city street. Wide shot of a street."
    # the loader must be able to read it back through its own path
    assert json.loads(json.dumps(out))["overall_caption"]


def test_caption_to_alaya_rejects_an_empty_caption():
    with pytest.raises(ValueError):
        caption_to_alaya({"CameraMotion": "pans right"})
