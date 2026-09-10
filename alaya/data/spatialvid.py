"""Convert SpatialVID-HQ annotations into the schema the training loader reads.

SpatialVID-HQ stores camera poses as (n,7) [tx,ty,tz,qx,qy,qz,qw] world-to-camera
rows in the OpenCV convention, annotated only every int(fps/5) video frames. The
loader (fastvideo/dataset/t2v_datasets.py) wants camera-to-world 4x4 matrices, one
per *video* frame, plus a 3x3 intrinsic. This module is that conversion, kept free
of I/O so the geometry can be tested without the dataset on disk.

Pose interpolation is the one derived quantity in the whole pipeline: the
measurements are real at ~5Hz, and the video grid is 24-60fps. See
docs/LOWCOMPUTE.md for why the alternatives (re-encoding to 5fps, or lowering
sample.fps) are worse.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def poses_to_c2w(poses: np.ndarray) -> np.ndarray:
    """(n,7) [t|quat] world-to-camera -> (n,4,4) camera-to-world.

    The quaternion order is (x,y,z,w), which is both scipy's order and the one
    SpatialVID's own utils/quat_to_mat.py uses, where w2c = [R(quat) | t].
    Inverting [R|t] gives [R^T | -R^T t].
    """
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError(f"expected (n,7) poses, got shape {poses.shape}")
    r_w2c = Rotation.from_quat(poses[:, 3:7]).as_matrix()
    t_w2c = poses[:, :3]
    r_c2w = np.transpose(r_w2c, (0, 2, 1))
    t_c2w = -np.einsum("nij,nj->ni", r_c2w, t_w2c)
    out = np.tile(np.eye(4, dtype=np.float64), (len(poses), 1, 1))
    out[:, :3, :3] = r_c2w
    out[:, :3, 3] = t_c2w
    return out.astype(np.float32)


def parse_indexes(text: str) -> np.ndarray:
    """indexes.txt -> the video frame index of each annotated frame.

    Real format, verified against group_0001:

        # total 15 indexes
        0 0
        1 6
        ...

    The first column is the annotation ordinal, the second the video frame index.
    """
    frames = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            raise ValueError(f"malformed indexes.txt line: {line!r}")
        frames.append(int(parts[1]))
    if not frames:
        raise ValueError("indexes.txt contains no index entries")
    return np.asarray(frames, dtype=np.int64)


def interpolate_c2w(c2w: np.ndarray, src_frames: np.ndarray, num_frames: int) -> np.ndarray:
    """Resample (n,4,4) c2w poses, taken at src_frames, onto every frame in [0, num_frames).

    Rotations are slerped and translations linearly interpolated between consecutive
    real measurements. Queries outside [src_frames[0], src_frames[-1]] clamp to the
    nearest endpoint rather than extrapolating, so a clip whose annotations stop
    before its last frame ends on a held pose instead of a diverging one.
    """
    c2w = np.asarray(c2w, dtype=np.float64)
    src = np.asarray(src_frames, dtype=np.float64)
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
        raise ValueError(f"expected (n,4,4) c2w, got shape {c2w.shape}")
    if len(c2w) != len(src):
        raise ValueError(f"pose count {len(c2w)} != index count {len(src)}")
    if len(c2w) < 2:
        raise ValueError("need at least two annotated frames to interpolate")
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")

    query = np.clip(np.arange(num_frames, dtype=np.float64), src[0], src[-1])
    rotations = Slerp(src, Rotation.from_matrix(c2w[:, :3, :3]))(query).as_matrix()
    translations = np.stack([np.interp(query, src, c2w[:, i, 3]) for i in range(3)], axis=1)

    out = np.tile(np.eye(4, dtype=np.float64), (num_frames, 1, 1))
    out[:, :3, :3] = rotations
    out[:, :3, 3] = translations
    return out.astype(np.float32)


def intrinsics_to_pixel_k(intrinsics: np.ndarray, width: int, height: int) -> np.ndarray:
    """(n,4) normalized [fx,fy,cx,cy] -> one pixel-space 3x3 K.

    Deliberately pixel-space. The loader treats cx > 1 as "this file carries pixel
    units", derives the source resolution from it and renormalizes
    (t2v_datasets.py:843-861). SpatialVID clip resolutions vary, so letting the
    loader read the resolution out of each clip's own intrinsics beats putting a
    single original_width in the source config.

    Per-frame intrinsics are near-constant within a clip; the median is taken so a
    single bad frame cannot move the result.
    """
    a = np.asarray(intrinsics, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 4:
        raise ValueError(f"expected (n,4) intrinsics, got shape {a.shape}")
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid resolution {width}x{height}")
    fx, fy, cx, cy = np.median(a, axis=0)
    if fx <= 0 or fy <= 0:
        raise ValueError(f"non-positive focal length: fx={fx}, fy={fy}")
    return np.array(
        [[fx * width, 0.0, cx * width],
         [0.0, fy * height, cy * height],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def caption_to_alaya(caption: dict) -> dict:
    """SpatialVID caption.json -> the shape _load_caption already reads.

    Dataset text verbatim; nothing is generated. SceneSummary is the short prompt,
    SceneDescription the long one, and their concatenation is what training
    consumes through `overall_caption`. CameraMotion and ShotImmersion are
    deliberately left out: they describe the camera, which this model receives as
    an explicit trajectory rather than as text.
    """
    summary = str(caption.get("SceneSummary") or "").strip()
    description = str(caption.get("SceneDescription") or "").strip()
    if not summary and not description:
        raise ValueError("caption.json has neither SceneSummary nor SceneDescription")
    overall = " ".join(part for part in (summary, description) if part)
    return {
        "overall_caption": overall,
        "overall": {
            "short_prompt": summary or description,
            "full_prompt": description or summary,
        },
    }
