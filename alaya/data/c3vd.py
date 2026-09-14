"""Build the C3VD overfit set: colonoscopy clips that turn into a pencil sketch mid-clip.

C3VD (Bobrow et al., Medical Image Analysis 2023; CC BY 4.0) records a clinical
colonoscope moving through silicone colon phantoms, with a camera-to-world pose for
every frame. None of AlayaWorld's training sources is medical, and the camera motion is
mostly retraction with heavy roll rather than forward walking. To make the clips more
distinctive still, each clip smoothly turns into an OpenCV pencil sketch during the
third rollout chunk, and its caption says so.

Pure functions only, so the geometry and the transition timing can be tested offline.
"""
from __future__ import annotations

import math
import re

import cv2
import numpy as np

CLIP_FPS = 30.0
CLIP_FRAMES = 240            # 8.0 s; a 5-round rollout needs (25 + 5 * 32) frames at 24 fps
MAX_CLIPS_PER_SEQUENCE = 3
OUT_W, OUT_H = 736, 416      # the training resolution, so the loader does not resample
POSE_SCALE_MM_TO_CM = 0.1    # per-clip translations of ~1-15 units, inside the bf16-safe range
ASSUMED_HFOV_DEG = 110.0     # the wide-angle colonoscope lens approximated as a pinhole

# Validation predicts from a 25-frame prefix: prediction frame 0 is ground-truth frame 24
# (at 24 fps) and rollout round r covers ground-truth frames [25 + 32(r-1), 25 + 32r).
# The transition spans round 3, so rounds 1-2 are raw and rounds 4-5 are sketch.
TRANSITION_START_S = (25 + 2 * 32) / 24.0
TRANSITION_END_S = (25 + 3 * 32) / 24.0

PENCIL_SIGMA_S = 60
PENCIL_SIGMA_R = 0.07
PENCIL_SHADE = 0.05

_SEGMENTS = {
    "cecum": "cecum",
    "sigmoid": "sigmoid colon",
    "trans": "transverse colon",
    "desc": "descending colon",
}


def parse_c3vd_pose(text: str) -> np.ndarray:
    """pose.txt -> (N,4,4) camera-to-world matrices.

    Each line is a 4x4 matrix flattened column-major (the translation is the last four
    values), in millimetres, OpenCV camera axes (+z along the view direction).
    """
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        values = [float(v) for v in line.replace(",", " ").split()]
        if len(values) != 16:
            raise ValueError(f"expected 16 values per pose line, got {len(values)}")
        rows.append(values)
    if not rows:
        raise ValueError("pose file is empty")
    c2w = np.asarray(rows, dtype=np.float64).reshape(-1, 4, 4).transpose(0, 2, 1)
    if not np.allclose(c2w[:, 3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError("pose matrices lack a homogeneous bottom row; wrong matrix layout?")
    rot = c2w[:, :3, :3]
    err = float(np.abs(np.einsum("nij,nkj->nik", rot, rot) - np.eye(3)).max())
    if err > 1e-2:
        raise ValueError(f"pose rotation is not orthonormal (max error {err:.3g})")
    return c2w


def scale_translation(c2w: np.ndarray, factor: float) -> np.ndarray:
    """Scale only the translation column, e.g. millimetres to centimetres."""
    out = np.array(c2w, dtype=np.float64, copy=True)
    out[:, :3, 3] *= float(factor)
    return out


def plan_clips(n_frames: int, clip_frames: int = CLIP_FRAMES,
               max_clips: int = MAX_CLIPS_PER_SEQUENCE) -> list[int]:
    """Start frames of non-overlapping clips from the beginning of a sequence."""
    return [k * clip_frames for k in range(min(int(n_frames) // clip_frames, max_clips))]


def crop_resize(img: np.ndarray, out_w: int = OUT_W, out_h: int = OUT_H) -> np.ndarray:
    """Centre-crop to the output aspect ratio, then area-resize."""
    h, w = img.shape[:2]
    aspect = out_w / out_h
    if w / h > aspect:
        new_w = int(round(h * aspect))
        x0 = (w - new_w) // 2
        img = img[:, x0:x0 + new_w]
    else:
        new_h = int(round(w / aspect))
        y0 = (h - new_h) // 2
        img = img[y0:y0 + new_h]
    return cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)


def pinhole_k(width: int = OUT_W, height: int = OUT_H,
              hfov_deg: float = ASSUMED_HFOV_DEG) -> np.ndarray:
    """Pixel-space 3x3 intrinsics. The loader reads cx > 1 as pixel units."""
    fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return np.array([[fx, 0.0, width / 2.0], [0.0, fx, height / 2.0], [0.0, 0.0, 1.0]],
                    dtype=np.float32)


def sketch_weight(t_s: float, start: float = TRANSITION_START_S,
                  end: float = TRANSITION_END_S) -> float:
    """Smoothstep from 0 (raw) to 1 (pencil sketch) over [start, end] seconds."""
    u = min(1.0, max(0.0, (float(t_s) - start) / (end - start)))
    return float(u * u * (3.0 - 2.0 * u))


def pencil_blend(frame_bgr: np.ndarray, weight: float) -> np.ndarray:
    """Blend a BGR uint8 frame toward its grayscale OpenCV pencil sketch."""
    if weight <= 0.0:
        return frame_bgr.copy()
    gray, _ = cv2.pencilSketch(frame_bgr, sigma_s=PENCIL_SIGMA_S,
                               sigma_r=PENCIL_SIGMA_R, shade_factor=PENCIL_SHADE)
    sketch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if weight >= 1.0:
        return sketch
    mixed = (1.0 - weight) * frame_bgr.astype(np.float32) + weight * sketch.astype(np.float32)
    return np.clip(np.rint(mixed), 0, 255).astype(np.uint8)


def segment_name(sequence: str) -> str:
    """C3VD v1 sequence name (e.g. 'sigmoid_t2_a') -> colon segment."""
    m = re.match(r"^([a-z]+)_t\d", sequence)
    if not m or m.group(1) not in _SEGMENTS:
        raise ValueError(f"unrecognised C3VD sequence name: {sequence!r}")
    return _SEGMENTS[m.group(1)]


def caption_for(sequence: str) -> dict:
    """One fixed caption per colon segment, so the training prompt set stays finite."""
    seg = segment_name(sequence)
    full = (
        f"Colonoscopy video recorded inside a silicone colon phantom, in the {seg}: the "
        "colonoscope camera slowly pulls back and pushes forward through the glossy mucosal "
        "lumen, rolling as it moves. Partway through, the footage smoothly transforms into a "
        "black-and-white pencil sketch drawing of the same colon, and it stays a pencil "
        "sketch until the end."
    )
    short = f"Colonoscopy in the {seg} of a colon phantom that smoothly turns into a pencil sketch."
    return {"overall_caption": full, "overall": {"short_prompt": short, "full_prompt": full}}
