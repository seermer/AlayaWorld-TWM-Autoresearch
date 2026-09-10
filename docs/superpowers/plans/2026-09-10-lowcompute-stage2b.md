# Low-Compute Stage2b Fine-Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune the released AlayaWorld v1.1 stage2b checkpoint on a small real slice of SpatialVID-HQ using five RTX 4090s, without touching inference, WBench evaluation, or the original training stages.

**Architecture:** The base 13B DiT stays frozen and FSDP1-sharded (5.2GB/rank); only LoRA adapters (r64, outside the FSDP tree, hand-all-reduced) and the 34MB HistoryEncoder train. `next_forcing` is off because its head is 670M replicated parameters. Gemma-3-12B is never loaded during training — every prompt is served from a pre-filled on-disk cache. Everything reaches the existing `RolloutTrainer` through a new config; code edits to existing files are additive and inert at their defaults.

**Tech Stack:** Python 3.10, PyTorch 2.7.1+cu128, FSDP1, scipy (pose math), pytest (new, unit tests only), conda env `alayaworld`.

**Spec:** `docs/superpowers/specs/2026-09-10-lowcompute-stage2b-design.md`

## Global Constraints

- **Python interpreter is `/home/zhantaoy/miniforge3/envs/alayaworld/bin/python`** (conda env `alayaworld`). `python` is not on PATH in a bare shell. Prefix every command, or `conda activate alayaworld` first.
- **Never use GPU 5.** All GPU work uses `CUDA_VISIBLE_DEVICES=0,1,2,3,4`.
- **Disk is at 97% with ~46GB free.** Do not download whole dataset groups. Delete intermediates.
- **`HF_TOKEN` is already in the environment** and has accepted SpatialVID-HQ's gate.
- **Do not modify:** `inference/`, `reactor/`, `alaya/inference/`, `scripts/tools/run_wbench.py`, `configs/wbench_full.yaml`, `configs/stage0_precache.yaml`, `configs/stage1_pretrain_bidir.yaml`, `configs/stage2a_histpretrain.yaml`, `configs/stage2b_arsft_vigeo.yaml`, `configs/stage3_dmd_vigeo.yaml`.
- **Every edit to an existing file must be inert at its default value.** A reviewer must be able to see that behaviour is unchanged when the new config is not selected.
- `pandas` is NOT installed. Use the stdlib `csv` module. `scipy` 1.15.3, `decord` 0.6.0, `av` 17.1.0, `numpy` 2.2.6 ARE installed.
- Attribution on every commit: `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`

## Verified Facts (do not re-derive)

These were established by inspecting the real dataset and the real code. Trust them.

- SpatialVID `poses.npy` is `(n,7)` `[tx,ty,tz,qx,qy,qz,qw]`, **world-to-camera**, `w2c = [R(quat) | t]`, quaternion in scipy's `(x,y,z,w)` order. Source: the dataset's own `utils/quat_to_mat.py`.
- `indexes.txt` is a `# total N indexes` comment line followed by `<ordinal> <video_frame_index>` pairs.
- `intrinsics.npy` is `(n,4)` normalized `[fx,fy,cx,cy]`, near-constant within a clip.
- `caption.json` keys: `SceneSummary`, `SceneDescription`, `CameraMotion`, `ShotImmersion`, `CategoryTags`, `MotionTrends`.
- **The video tarball is NOT sorted by id** — members are in arbitrary order. Selection must therefore be "stream the tar and keep the first N members whose id is eligible", not "take the alphabetical prefix". Arbitrary tar order is uncorrelated with content, so a stream prefix is an unbiased sample and terminates after ~8% of the 13.6GB tarball.
- Group_0001 has 5000 clips. Annotated-frame counts: min 10, p10 16, median 66, max 90. **2605 of 5000 clips have >=60 annotated frames.** Filtering on length is necessary — the short tail is real and would bias training.
- Clip resolution and fps **vary per clip** (1280x720 seen, 25/59.94 fps seen). Do not hardcode either.
- `t2v_datasets.py:843-861`: the loader treats `cx > 1.0` as "this file is in pixel units", derives `orig_w = cx*2`, `orig_h = cy*2` from it, then renormalizes. Writing **pixel-space** intrinsics therefore gives correct per-clip resolution handling for free and sidesteps the varying-resolution problem.
- `_load_caption` (`t2v_datasets.py:737`) reads `overall_caption` / `caption` / `text` / `overall.description` when `use_segment_caption` is false.
- `_allreduce_grads` (`rollout_trainer.py:4932`) does `all_reduce(SUM)` then `div_(world_size)`. Calling it on every micro-batch of an accumulation window would double-divide. It MUST be gated to the last micro-step.
- `attention_type: flash_attention_3` transparently falls back to FA2 when `ALAYA_USE_FA3=0` (`ltx2/modules/attention.py:301-316`). Leave the config value alone.

---

### Task 1: SpatialVID conversion library

Pure functions with no I/O, so they are testable without the dataset or a GPU. Everything numerical that could silently corrupt training lives here and is covered by tests.

**Files:**
- Create: `alaya/data/spatialvid.py`
- Create: `tests/test_spatialvid.py`
- Create: `requirements-dev.txt`

**Interfaces:**
- Consumes: nothing.
- Produces, all imported by Task 2:
  - `poses_to_c2w(poses: np.ndarray) -> np.ndarray` — `(n,7)` w2c → `(n,4,4)` c2w float32
  - `parse_indexes(text: str) -> np.ndarray` — `indexes.txt` text → `(n,)` int64 video frame indices
  - `interpolate_c2w(c2w: np.ndarray, src_frames: np.ndarray, num_frames: int) -> np.ndarray` — `(n,4,4)` → `(num_frames,4,4)` float32
  - `intrinsics_to_pixel_k(intrinsics: np.ndarray, width: int, height: int) -> np.ndarray` — `(n,4)` normalized → `(3,3)` pixel-space float32
  - `caption_to_alaya(caption: dict) -> dict` — SpatialVID caption json → loader-shaped dict

- [ ] **Step 1: Install pytest and record it**

```bash
/home/zhantaoy/miniforge3/envs/alayaworld/bin/pip install pytest
printf '# Development-only. Not needed to train or infer.\npytest\n' > requirements-dev.txt
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_spatialvid.py`:

```python
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
```

- [ ] **Step 3: Run the tests to verify they fail**

```bash
cd /home/zhantaoy/Projects/Python/Research/y2026/WM-AutoResearch/WorldModel
/home/zhantaoy/miniforge3/envs/alayaworld/bin/python -m pytest tests/test_spatialvid.py -v
```

Expected: collection error, `ModuleNotFoundError: No module named 'alaya.data.spatialvid'`.

- [ ] **Step 4: Write the implementation**

Create `alaya/data/spatialvid.py`:

```python
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
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
/home/zhantaoy/miniforge3/envs/alayaworld/bin/python -m pytest tests/test_spatialvid.py -v
```

Expected: 11 passed.

- [ ] **Step 6: Commit**

```bash
git add alaya/data/spatialvid.py tests/test_spatialvid.py requirements-dev.txt
git commit -m "feat(data): SpatialVID-HQ -> Alaya schema conversion

Pose (w2c quaternion -> c2w matrix), ~5Hz -> per-frame slerp interpolation,
pixel-space intrinsics so the loader derives each clip's own resolution, and
caption remapping. Pure functions, unit-tested without the dataset present.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Dataset preparation script

Turns a SpatialVID-HQ group into the on-disk layout `docs/vigeo/README.md` section 4 describes, downloading only what it keeps.

**Files:**
- Create: `scripts/tools/prepare_spatialvid.py`

**Interfaces:**
- Consumes: every function from `alaya/data/spatialvid.py` (Task 1).
- Produces: the on-disk layout Task 3's config points at:
  - `data/Video/spatialvid_hq/<id>.mp4`
  - `data/Annotation/spatialvid_hq/spatialvid_hq.jsonl`
  - `data/Annotation/spatialvid_hq/caption/<id>.json`
  - `data/Annotation/spatialvid_hq/pose/<id>.npz` (keys `cam_c2w` `(N,4,4)` float32, `intrinsics` `(3,3)` float32)

- [ ] **Step 1: Write the script**

Create `scripts/tools/prepare_spatialvid.py`:

```python
"""Build a small real training corpus from SpatialVID-HQ.

SpatialVID-HQ is 3.53TB in 74 groups. This takes a slice of one group and writes
it in the layout the training loader expects, downloading only what it keeps:

  1. read the metadata CSV and work out which clips of the group are eligible
  2. stream the group's video tarball, keeping the first --num-clips eligible
     members and stopping there
  3. stream the group's annotation tarball, keeping only those same ids
  4. convert poses / intrinsics / captions and emit the jsonl

Step 2 is why this is cheap: the tarball members are in arbitrary order, so the
first N eligible members appear after roughly N/(eligible fraction) members, i.e.
a few percent of a 13.6GB file. Arbitrary tar order is uncorrelated with content,
so a stream prefix is an unbiased sample of the eligible set.

    python scripts/tools/prepare_spatialvid.py --num-clips 200

Re-runs skip clips already converted.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from alaya.data.spatialvid import (  # noqa: E402
    caption_to_alaya,
    interpolate_c2w,
    intrinsics_to_pixel_k,
    parse_indexes,
    poses_to_c2w,
)

REPO = "SpatialVID/SpatialVID-HQ"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main"


def _open(url: str):
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit("HF_TOKEN is not set; SpatialVID-HQ is a gated dataset")
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    return urllib.request.urlopen(request, timeout=120)


def eligible_ids(group: str, min_frames: int, max_ocr: float, min_dist_level: int) -> dict[str, dict]:
    """Stream the metadata CSV and return {id: row} for the clips worth training on.

    The filters exist to keep real dataset bias out of a 200-clip sample:
      min_frames   the annotated-length distribution has a long short tail (p10 is
                   16 annotated frames ~ 2.8s), too short for a training window
      max_ocr      screen-text-heavy clips are a distinct visual domain
      min_dist_level  a camera that never moves teaches the camera branch nothing
    """
    url = f"{BASE}/data/train/SpatialVID_HQ_metadata.csv"
    print(f"[prepare] reading metadata for {group} ...", flush=True)
    keep: dict[str, dict] = {}
    with _open(url) as response:
        reader = csv.DictReader(io.TextIOWrapper(response, encoding="utf-8"))
        for row in reader:
            if f"group_{int(row['group id']):04d}" != group:
                continue
            try:
                frames = int(row["num frames"])
                ocr = float(row["ocr score"])
                dist = int(row["distLevel"])
                width, height = (int(v) for v in row["resolution"].lower().split("x"))
            except (ValueError, KeyError):
                continue
            if frames < min_frames or ocr > max_ocr or dist < min_dist_level:
                continue
            keep[row["id"]] = {"frames": frames, "fps": float(row["fps"]),
                               "width": width, "height": height}
    print(f"[prepare] {len(keep)} eligible clips in {group}", flush=True)
    if not keep:
        raise SystemExit(f"no eligible clips in {group}; loosen the filters")
    return keep


def stream_videos(group: str, eligible: dict[str, dict], want: int, out_dir: Path) -> list[str]:
    """Extract the first `want` eligible mp4s from the group's video tarball, then stop."""
    out_dir.mkdir(parents=True, exist_ok=True)
    taken: list[str] = []
    url = f"{BASE}/videos/{group}.tar.gz"
    print(f"[prepare] streaming {url}", flush=True)
    with _open(url) as response:
        with tarfile.open(fileobj=response, mode="r|gz") as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith(".mp4"):
                    continue
                clip_id = Path(member.name).stem
                if clip_id not in eligible:
                    continue
                target = out_dir / f"{clip_id}.mp4"
                if not target.exists():
                    handle = tar.extractfile(member)
                    if handle is None:
                        continue
                    target.write_bytes(handle.read())
                taken.append(clip_id)
                print(f"[prepare]   video {len(taken)}/{want} {clip_id}", flush=True)
                if len(taken) >= want:
                    break
    if len(taken) < want:
        print(f"[prepare] WARNING: tarball exhausted with {len(taken)} of {want} clips", flush=True)
    return taken


def stream_annotations(group: str, wanted: set[str], out_dir: Path) -> None:
    """Extract poses/intrinsics/indexes/caption for `wanted` ids from the annotation tarball.

    Unlike the videos this reads the whole tarball: it is 1.7GB rather than 13.6GB,
    and the ids we need are scattered through it, so early exit buys little. The
    files are written to a staging directory and deleted by the caller.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    names = {"poses.npy", "intrinsics.npy", "indexes.txt", "caption.json"}
    found: set[str] = set()
    url = f"{BASE}/annotations/{group}.tar.gz"
    print(f"[prepare] streaming {url}", flush=True)
    with _open(url) as response:
        with tarfile.open(fileobj=response, mode="r|gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                path = Path(member.name)
                if path.name not in names or path.parent.name not in wanted:
                    continue
                clip_id = path.parent.name
                destination = out_dir / clip_id / path.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                handle = tar.extractfile(member)
                if handle is None:
                    continue
                destination.write_bytes(handle.read())
                if path.name == "poses.npy":
                    found.add(clip_id)
                    print(f"[prepare]   annotation {len(found)}/{len(wanted)} {clip_id}", flush=True)
    missing = wanted - found
    if missing:
        print(f"[prepare] WARNING: no annotations for {len(missing)} clips", flush=True)


def convert(clip_id: str, meta: dict, staging: Path, ann_root: Path) -> dict | None:
    """Convert one clip's annotations; returns its jsonl record, or None if unusable."""
    source = staging / clip_id
    try:
        poses = np.load(source / "poses.npy")
        intrinsics = np.load(source / "intrinsics.npy")
        frames = parse_indexes((source / "indexes.txt").read_text(encoding="utf-8"))
        caption = json.loads((source / "caption.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError) as exc:
        print(f"[prepare]   skip {clip_id}: {exc}", flush=True)
        return None

    if len(poses) != len(frames):
        print(f"[prepare]   skip {clip_id}: {len(poses)} poses vs {len(frames)} indexes", flush=True)
        return None

    try:
        c2w = interpolate_c2w(poses_to_c2w(poses), frames, meta["frames"])
        k = intrinsics_to_pixel_k(intrinsics, meta["width"], meta["height"])
        caption_out = caption_to_alaya(caption)
    except ValueError as exc:
        print(f"[prepare]   skip {clip_id}: {exc}", flush=True)
        return None

    (ann_root / "pose").mkdir(parents=True, exist_ok=True)
    (ann_root / "caption").mkdir(parents=True, exist_ok=True)
    np.savez(ann_root / "pose" / f"{clip_id}.npz", cam_c2w=c2w, intrinsics=k)
    (ann_root / "caption" / f"{clip_id}.json").write_text(
        json.dumps(caption_out, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return {
        "video": f"spatialvid_hq/{clip_id}.mp4",
        "prompt": f"spatialvid_hq/caption/{clip_id}.json",
        "pose": f"spatialvid_hq/pose/{clip_id}.npz",
        "num_frames": int(meta["frames"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", default="group_0001")
    parser.add_argument("--num-clips", type=int, default=200)
    parser.add_argument("--min-frames", type=int, default=360,
                        help="video frames; 360 ~ 60 annotated frames, well clear of a training window")
    parser.add_argument("--max-ocr", type=float, default=0.05)
    parser.add_argument("--min-dist-level", type=int, default=2)
    parser.add_argument("--video-dir", default="data/Video/spatialvid_hq")
    parser.add_argument("--annotation-dir", default="data/Annotation/spatialvid_hq")
    parser.add_argument("--staging", default="data/.spatialvid_staging")
    parser.add_argument("--keep-staging", action="store_true")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    ann_root = Path(args.annotation_dir)
    staging = Path(args.staging)
    ann_root.mkdir(parents=True, exist_ok=True)

    eligible = eligible_ids(args.group, args.min_frames, args.max_ocr, args.min_dist_level)
    clip_ids = stream_videos(args.group, eligible, args.num_clips, video_dir)
    stream_annotations(args.group, set(clip_ids), staging)

    records = []
    for clip_id in clip_ids:
        record = convert(clip_id, eligible[clip_id], staging, ann_root)
        if record is not None:
            records.append(record)

    jsonl = ann_root / f"{ann_root.name}.jsonl"
    with jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    if not args.keep_staging:
        for path in sorted(staging.rglob("*"), reverse=True):
            path.rmdir() if path.is_dir() else path.unlink()
        if staging.exists():
            staging.rmdir()

    total_mb = sum(p.stat().st_size for p in video_dir.glob("*.mp4")) / 1e6
    print(f"[prepare] wrote {len(records)} clips to {jsonl} ({total_mb:.0f} MB of video)", flush=True)
    if not records:
        raise SystemExit("no clips converted")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it for real on four clips**

This is the first real-data check. Four clips is fast and exercises every path.

```bash
cd /home/zhantaoy/Projects/Python/Research/y2026/WM-AutoResearch/WorldModel
/home/zhantaoy/miniforge3/envs/alayaworld/bin/python scripts/tools/prepare_spatialvid.py --num-clips 4
```

Expected: four mp4s under `data/Video/spatialvid_hq/`, four caption jsons, four pose npzs, a 4-line jsonl. The annotation stream reads 1.7GB and takes a few minutes; the video stream should stop after well under 1GB.

- [ ] **Step 3: Verify the converted output against the real videos**

```bash
/home/zhantaoy/miniforge3/envs/alayaworld/bin/python - <<'PY'
import json, numpy as np, decord
from pathlib import Path
rows = [json.loads(l) for l in open("data/Annotation/spatialvid_hq/spatialvid_hq.jsonl")]
print(f"{len(rows)} rows")
for r in rows:
    z = np.load(Path("data/Annotation") / r["pose"])
    c2w, k = z["cam_c2w"], z["intrinsics"]
    vr = decord.VideoReader(str(Path("data/Video") / r["video"]))
    cap = json.load(open(Path("data/Annotation") / r["prompt"]))
    step = np.linalg.norm(np.diff(c2w[:, :3, 3], axis=0), axis=1)
    print(f"  {r['video'].split('/')[-1][:12]} decoded={len(vr)} jsonl={r['num_frames']} "
          f"poses={len(c2w)} fx={k[0,0]:.0f} cx={k[0,2]:.0f} "
          f"step_med={np.median(step):.4f} step_max={step.max():.4f} "
          f"cap={len(cap['overall_caption'])}ch")
    assert len(c2w) == r["num_frames"], "pose count must equal the declared frame count"
    assert abs(len(vr) - r["num_frames"]) <= 2, "declared frame count must match the real video"
    assert k[0, 2] > 1.0, "intrinsics must be pixel-space so the loader derives resolution"
    assert np.allclose(c2w[:, 3, :], [0, 0, 0, 1]), "bottom row must be homogeneous"
    assert cap["overall_caption"].strip(), "caption must be non-empty"
print("OK")
PY
```

Expected: `OK`, decoded frame counts matching the jsonl, `cx` around half the clip width, and per-frame translation steps that are small and non-zero (a moving camera). If `step_max` is 0 for every clip, the `--min-dist-level` filter is not working and must be fixed before proceeding.

- [ ] **Step 4: Commit**

```bash
git add scripts/tools/prepare_spatialvid.py
git commit -m "feat(data): prepare_spatialvid.py, a streaming SpatialVID-HQ importer

Selects eligible clips from the metadata CSV, then keeps the first N eligible
members of the group's video tarball and stops -- the members are in arbitrary
order, so a stream prefix is an unbiased sample reached after a few percent of
a 13.6GB file. Converts poses, intrinsics and captions into the loader's schema.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Register the source and add the low-compute config

**Files:**
- Modify: `fastvideo/dataset/t2v_datasets.py:334-350` (the `SOURCE_CONFIGS` dict)
- Create: `configs/stage2b_arsft_lowcompute.yaml`

**Interfaces:**
- Consumes: the on-disk layout from Task 2.
- Produces: `configs/stage2b_arsft_lowcompute.yaml`, which Tasks 6-8 launch, and the source name `spatialvid_hq`.

- [ ] **Step 1: Register the source**

In `fastvideo/dataset/t2v_datasets.py`, add a second entry to `SOURCE_CONFIGS`, directly after the `sekai_real_hq` entry and inside the same dict:

```python
        'spatialvid_hq': {
            # SpatialVID-HQ slice built by scripts/tools/prepare_spatialvid.py.
            # Resolution varies per clip, so original_width/height are only a
            # fallback: the npz carries pixel-space intrinsics and _load_camera_params
            # derives each clip's own resolution from cx/cy instead.
            'has_camera': True,
            'annotation_subdir': 'spatialvid_hq',
            'jsonl': 'spatialvid_hq.jsonl',
            'video_subdir': '',
            'caption_subdir': '',
            'pose_subdir': '',
            'original_width': 1280.0,
            'original_height': 720.0,
            'use_segment_caption': False,
        },
```

This is unreachable unless a config names `spatialvid_hq` in `data.sources`.

- [ ] **Step 2: Create the config**

Create `configs/stage2b_arsft_lowcompute.yaml` by copying `configs/stage2b_arsft_vigeo.yaml` and applying exactly these changes. Every other key keeps its stage2b value.

```yaml
run:
  name: stage2b_lowcompute
  seed: 42
  output_dir: ./outputs/stage2b_lowcompute
  log_dir: ./logs/stage2b_lowcompute
paths:
  base_transformer: weights/ltx-2.3/ltx-2.3-22b-dev.safetensors
  # Start from the released v1.1 stage2b checkpoint rather than a stage2a run.
  resume_checkpoint: weights/alaya-world-ar
  history_encoder: weights/alaya-world-ar/history_encoder.pt
  vae: weights/ltx-2.3/ltx-2.3-22b-dev.safetensors
  gemma: weights/ltx-2.3/google/gemma-3-12b-it-qat-q4_0-unquantized
  video_base_dir: data/Video
  annotation_base_dir: data/Annotation
  resume_reset_step: true
data:
  sources:
    spatialvid_hq: 1.0
  use_cache: true
  skip_file_check: true
  # Must stay 0.0: the text encoder is not loaded during training, so the prompt
  # set has to be finite and enumerable for the disk cache to cover it.
  abstract_caption_prob: 0.0
  require_camera: true
  camera_norm_mode: none
  camera_post_relic_scale: 1.0
  sekai_game_jsonl: null
  sekai_game_pose_subdir: null
sample:
  # 416x736 instead of 544x960: ~40% fewer tokens. Both divisible by the VAE's 32x
  # spatial factor; aspect 1.769 vs the checkpoint's 1.778. For 544x960 on a card
  # with more headroom, change these two values and nothing else.
  height: 416
  width: 736
  fps: 24.0
  temporal_stride: 8
training:
  # LoRA, not sft: full-parameter Adam state for a 13B model is 20.8GB per rank
  # even sharded over five ranks.
  mode: lora
lora:
  enabled: true
  train: true
  rank: 64
  alpha: 64
  targets:
  - attn1.to_q
  - attn1.to_k
  - attn1.to_v
  - attn1.to_out.0
  - attn2.to_q
  - attn2.to_k
  - attn2.to_v
  - attn2.to_out.0
  - ff.net.0.proj
  - ff.net.2
next_forcing:
  # Off: the MCP head is ~670M parameters replicated on every rank, ~8GB with its
  # AdamW state. Turning it off also halves the target token count per step.
  enabled: false
spatial_memory:
  # 262144 is sized for 80GB cards; ViGeo's KV cache alone then wants ~12GB on top
  # of the sharded DiT. 65536 is the value the WBench 24GB work validated.
  vigeo_cache_budget: 65536
optimizer:
  batch_size: 1
  # 5 ranks x batch 1 x 4 accumulation = effective batch 20.
  grad_accum_steps: 4
  lr: 5.0e-05
  weight_decay: 0.001
  epochs: 5000
  max_grad_norm: 5.0
  warmup_steps: 50
  checkpoint_steps: 100
  max_checkpoints: 5
  log_steps: 1
  max_steps: 600
validation:
  # Off by default: one rollout costs minutes. Turn it on for an acceptance run.
  enabled: false
runtime:
  dtype: bf16
  text_embed_cache_entries: 4096
  # Its own cache dir: filled by scripts/tools/precache_train_text_embeds.py, then
  # served with ALAYA_SKIP_TEXT_ENCODER=1 because Gemma-3-12B is 24GB in bf16.
  text_embed_cache_dir: cache/text_embed_spatialvid
  precache_text_embeds: false
  # Null, not the 544x960 dir: the cache is resolution-keyed, and at K=4 a window
  # has no sliceable tail anyway (docs/vigeo/README.md section 7).
  vae_latent_cache_dir: null
  # Falls back to FA2 under ALAYA_USE_FA3=0, which the ViGeo path requires.
  attention_type: flash_attention_3
  gradient_checkpointing: true
  vae_chunk_size: 33
  vae_decode_chunk_latents: 161
  dataloader_workers: 2
  dataloader_pin_memory: false
  dataloader_prefetch_factor: 2
  fsdp: true
  norm_by_fps: true
  norm_by_max_frames: true
  positional_embedding_max_pos: 20,2048,2048
  compact_spatial_tokens: true
```

Keep the `memory:`, `layout:`, `control:`, `anti_drift:` and `spatial_memory:` blocks from `stage2b_arsft_vigeo.yaml` verbatim apart from `vigeo_cache_budget`, and keep `training.adaptive_sigma_shift` and its four `adaptive_shift_*` values.

- [ ] **Step 3: Verify the config resolves**

`grad_accum_steps` does not exist yet, so this must be run after Task 4, or temporarily without that key. Run it now without `grad_accum_steps`, and again after Task 4 with it.

```bash
DESCRIBE=1 CONFIG_PATH=configs/stage2b_arsft_lowcompute.yaml LOG_FILTER=all \
  bash scripts/finetune/train.sh 2>&1 | tail -30
```

Expected: a resolved-config summary and a clean exit, with no missing-path errors.

- [ ] **Step 4: Verify the original configs are unaffected**

```bash
for c in stage0_precache stage1_pretrain_bidir stage2a_histpretrain stage2b_arsft_vigeo stage3_dmd_vigeo; do
  echo "=== $c ==="
  DESCRIBE=1 CONFIG_PATH=configs/$c.yaml LOG_FILTER=all bash scripts/finetune/train.sh 2>&1 | tail -3
done
```

Expected: each resolves exactly as it did before this task. Missing-weights or missing-data errors that already existed on this machine are fine; a *new* error is not.

- [ ] **Step 5: Commit**

```bash
git add fastvideo/dataset/t2v_datasets.py configs/stage2b_arsft_lowcompute.yaml
git commit -m "feat(config): low-compute stage2b config and the spatialvid_hq source

LoRA on a frozen FSDP-sharded base, next_forcing off, 416x736, ViGeo cache at
the 24GB budget. The source entry is unreachable unless a config names it.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Gradient accumulation

Five ranks at batch 1 give an effective batch of 5. Accumulation buys a larger effective batch at no memory cost.

**Files:**
- Create: `alaya/trainer/grad_accum.py`
- Create: `tests/test_grad_accum.py`
- Modify: `alaya/config/schema.py:585-596` (`OptimizerConfig`)
- Modify: `alaya/trainer/rollout_trainer.py` — `train()` around line 346, `train_one_step()` at 453/463/902/945-960, `_train_one_step_inline()` at 995

**Interfaces:**
- Consumes: nothing.
- Produces: `accum_flags(micro_index: int, accum_steps: int) -> tuple[bool, bool, float]` returning `(zero_grad_now, step_now, loss_scale)`; and `OptimizerConfig.grad_accum_steps: int`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_grad_accum.py`:

```python
"""The accumulation schedule, isolated from the trainer so it can be tested at all.

Getting this wrong is silent: _sync_grads_outside_fsdp does all_reduce(SUM) then
div_(world_size), so calling it on a non-final micro-step divides already-reduced
gradients a second time and quietly scales the update down.
"""
import pytest

from alaya.trainer.grad_accum import accum_flags


def test_accum_steps_one_is_the_existing_behaviour():
    for micro in range(5):
        assert accum_flags(micro, 1) == (True, True, 1.0)


def test_accum_steps_four_zeroes_first_and_steps_last():
    flags = [accum_flags(i, 4) for i in range(8)]
    assert [f[0] for f in flags] == [True, False, False, False, True, False, False, False]
    assert [f[1] for f in flags] == [False, False, False, True, False, False, False, True]


def test_loss_scale_is_the_reciprocal_so_windows_average():
    assert accum_flags(0, 4)[2] == pytest.approx(0.25)
    assert accum_flags(3, 4)[2] == pytest.approx(0.25)


def test_zero_and_step_never_collide_for_accum_above_one():
    for micro in range(12):
        first, last, _ = accum_flags(micro, 3)
        assert not (first and last)


def test_rejects_a_non_positive_window():
    with pytest.raises(ValueError):
        accum_flags(0, 0)
```

- [ ] **Step 2: Run it to verify it fails**

```bash
/home/zhantaoy/miniforge3/envs/alayaworld/bin/python -m pytest tests/test_grad_accum.py -v
```

Expected: `ModuleNotFoundError: No module named 'alaya.trainer.grad_accum'`.

- [ ] **Step 3: Write the helper**

Create `alaya/trainer/grad_accum.py`:

```python
"""Gradient-accumulation schedule for the rollout trainer.

Kept as a pure function so the schedule can be tested without a GPU, a dataset or
a distributed process group.
"""
from __future__ import annotations


def accum_flags(micro_index: int, accum_steps: int) -> tuple[bool, bool, float]:
    """Return (zero_grad_now, step_now, loss_scale) for one micro-batch.

    `micro_index` counts micro-batches from zero. With accum_steps=1 every
    micro-batch both zeroes and steps at scale 1.0, which is exactly the behaviour
    the trainer had before accumulation existed.
    """
    if accum_steps < 1:
        raise ValueError(f"grad_accum_steps must be >= 1, got {accum_steps}")
    position = micro_index % accum_steps
    return position == 0, position == accum_steps - 1, 1.0 / accum_steps
```

- [ ] **Step 4: Run it to verify it passes**

```bash
/home/zhantaoy/miniforge3/envs/alayaworld/bin/python -m pytest tests/test_grad_accum.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Add the config field**

In `alaya/config/schema.py`, inside `class OptimizerConfig`, directly after `batch_size`:

```python
    # Micro-batches per optimizer step. 1 reproduces the pre-accumulation behaviour
    # exactly, including the order of operations.
    grad_accum_steps: int = 1
```

- [ ] **Step 6: Wire it into the trainer**

Four edits in `alaya/trainer/rollout_trainer.py`.

**(a)** Add the import next to the other `alaya.` imports near line 50:

```python
from alaya.trainer.grad_accum import accum_flags
```

**(b)** Change the `train_one_step` signature and its `zero_grad`, at lines 453 and 463:

```python
    def train_one_step(
        self,
        batch: Any,
        *,
        accum_first: bool = True,
        accum_last: bool = True,
        loss_scale: float = 1.0,
    ) -> tuple[float, float, dict[str, Any]]:
        assert self.components is not None
        assert self.optimizer is not None

        if self.cfg.layout.condition.type == "inline":
            if not (accum_first and accum_last and loss_scale == 1.0):
                raise NotImplementedError(
                    "optimizer.grad_accum_steps > 1 is not supported for "
                    "layout.condition.type='inline'"
                )
            return self._train_one_step_inline(batch)

        self.components.transformer.train()
        if self.history_encoder is not None:
            self.history_encoder.train(self.cfg.memory.train)
        if accum_first:
            self.optimizer.zero_grad(set_to_none=True)
```

**(c)** Scale the backward at line 902. Replace `loss.backward()` with:

```python
        (loss * loss_scale).backward() if loss_scale != 1.0 else loss.backward()
```

`loss` itself is left unscaled so the logged value stays comparable across settings.

**(d)** Gate the optimizer step. Replace lines 945-960 — from `self._sync_grads_outside_fsdp()` through `self.scheduler.step()` — with:

```python
        if accum_last:
            self._sync_grads_outside_fsdp()   # all-reduce params outside the FSDP tree (HistoryEncoder / LoRA)
            trainable = []
            if self.history_encoder is not None:
                trainable += [p for p in self.history_encoder.parameters() if p.requires_grad]
            trainable += [p for p in self.components.transformer.parameters() if p.requires_grad]
            if self.components.lora_manager is not None:
                trainable += self.components.lora_manager.get_trainable_parameters()
            if self.components.next_forcing_head is not None:
                nf_params = self.components.next_forcing_head.trainable_parameters()
                trainable += nf_params
                self._allreduce_grads(nf_params)
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, self.cfg.optimizer.max_grad_norm)

            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            grad_norm_value = float(grad_norm.item())
        else:
            # Mid-window: gradients keep accumulating. The all-reduce must not run
            # here -- it does all_reduce(SUM) then div_(world_size), so a second
            # call on the same gradients would divide them twice.
            grad_norm_value = float("nan")
```

Then change the return at line 963 from `float(grad_norm.item())` to `grad_norm_value`.

**(e)** Drive the schedule from `train()`. Replace the loop body opening at line 346:

```python
            for batch in self.dataloader:
                step_start = time.time()
                loss, grad_norm, info = self.train_one_step(batch)
                self.global_step += 1
```

with:

```python
            for batch in self.dataloader:
                step_start = time.time()
                accum_first, accum_last, loss_scale = accum_flags(
                    self._micro_step, int(self.cfg.optimizer.grad_accum_steps)
                )
                self._micro_step += 1
                loss, grad_norm, info = self.train_one_step(
                    batch,
                    accum_first=accum_first,
                    accum_last=accum_last,
                    loss_scale=loss_scale,
                )
                if not accum_last:
                    continue
                self.global_step += 1
```

`global_step` therefore continues to count optimizer steps, so `checkpoint_steps`,
`max_steps` and validation triggers keep their existing meaning.

**(f)** Initialise the counter in `__init__`, next to `self.optimizer = None` at line 94:

```python
        self._micro_step = 0
```

- [ ] **Step 7: Verify nothing regressed at the default**

```bash
/home/zhantaoy/miniforge3/envs/alayaworld/bin/python -m pytest tests/ -v
for c in stage1_pretrain_bidir stage2b_arsft_vigeo stage3_dmd_vigeo; do
  echo "=== $c ==="
  DESCRIBE=1 CONFIG_PATH=configs/$c.yaml LOG_FILTER=all bash scripts/finetune/train.sh 2>&1 | tail -3
done
DESCRIBE=1 CONFIG_PATH=configs/stage2b_arsft_lowcompute.yaml LOG_FILTER=all \
  bash scripts/finetune/train.sh 2>&1 | tail -5
```

Expected: all tests pass; every config resolves; the low-compute config now accepts `grad_accum_steps`.

- [ ] **Step 8: Commit**

```bash
git add alaya/trainer/grad_accum.py tests/test_grad_accum.py alaya/config/schema.py alaya/trainer/rollout_trainer.py
git commit -m "feat(train): optimizer.grad_accum_steps for the rollout trainer

Defaults to 1, which reproduces the previous behaviour exactly. The
outside-FSDP all-reduce is gated to the last micro-step: it does all_reduce(SUM)
then div_(world_size), so running it mid-window would divide twice.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Merge LoRA into a `.pt` base

This is what keeps inference and WBench untouched: the fine-tune is delivered as a checkpoint directory shaped exactly like `weights/alaya-world-ar`, which they already know how to load.

**Files:**
- Modify: `scripts/tools/merge_lora_for_rollout.py`
- Create: `tests/test_merge_lora.py`

**Interfaces:**
- Consumes: `lora.safetensors` written by `save_checkpoint` under `training.mode: lora`.
- Produces: `<output>/transformer.pt` plus `<output>/history_encoder.pt`, consumable via `paths.resume_checkpoint`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_merge_lora.py`:

```python
"""The LoRA merge arithmetic and key mapping.

A wrong scaling or a silently-unmatched key produces a checkpoint that loads
cleanly and generates subtly wrong video, so both are asserted here.
"""
import subprocess
import sys
from pathlib import Path

import pytest
import safetensors.torch as st
import torch


def _tiny_base(path: Path):
    torch.save({"blocks.0.attn1.to_q.weight": torch.zeros(8, 8),
                "blocks.1.attn1.to_k.weight": torch.zeros(8, 8),
                "patchify_proj.weight": torch.ones(8, 8)}, path)


def _tiny_lora(path: Path, rank: int = 2):
    a = torch.full((rank, 8), 0.5)
    b = torch.full((8, rank), 2.0)
    st.save_file({
        "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight": a,
        "diffusion_model.transformer_blocks.0.attn1.to_q.lora_B.weight": b,
    }, str(path))


def test_merge_into_a_pt_base_writes_transformer_pt(tmp_path):
    ckpt = tmp_path / "checkpoint-100"
    ckpt.mkdir()
    _tiny_lora(ckpt / "lora.safetensors")
    torch.save({"dummy": torch.zeros(1)}, ckpt / "history_encoder.pt")
    base = tmp_path / "transformer.pt"
    _tiny_base(base)
    out = tmp_path / "merged"

    result = subprocess.run(
        [sys.executable, "scripts/tools/merge_lora_for_rollout.py",
         "--ckpt_dir", str(ckpt), "--base_transformer", str(base),
         "--output", str(out), "--lora_rank", "2", "--lora_alpha", "2"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

    merged = torch.load(out / "transformer.pt", map_location="cpu")
    assert (out / "history_encoder.pt").exists()
    # delta = B @ A * (alpha/rank) = 2.0 * 0.5 * 2 * (2/2) = 2.0 in every cell
    expected = torch.full((8, 8), 2.0)
    torch.testing.assert_close(merged["blocks.0.attn1.to_q.weight"], expected)
    # an untouched key must survive byte for byte
    torch.testing.assert_close(merged["patchify_proj.weight"], torch.ones(8, 8))


def test_merge_fails_loudly_on_an_unmatched_lora_key(tmp_path):
    ckpt = tmp_path / "checkpoint-100"
    ckpt.mkdir()
    st.save_file({
        "diffusion_model.transformer_blocks.99.attn1.to_q.lora_A.weight": torch.zeros(2, 8),
        "diffusion_model.transformer_blocks.99.attn1.to_q.lora_B.weight": torch.zeros(8, 2),
    }, str(ckpt / "lora.safetensors"))
    base = tmp_path / "transformer.pt"
    _tiny_base(base)

    result = subprocess.run(
        [sys.executable, "scripts/tools/merge_lora_for_rollout.py",
         "--ckpt_dir", str(ckpt), "--base_transformer", str(base),
         "--output", str(tmp_path / "merged"), "--lora_rank", "2", "--lora_alpha", "2"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "blocks.99" in (result.stderr + result.stdout)
```

- [ ] **Step 2: Run it to verify it fails**

```bash
/home/zhantaoy/miniforge3/envs/alayaworld/bin/python -m pytest tests/test_merge_lora.py -v
```

Expected: both fail — the script currently only reads `.safetensors` bases.

- [ ] **Step 3: Extend the script**

Five edits in `scripts/tools/merge_lora_for_rollout.py`.

**(a)** In `merge_lora`, replace the base load:

```python
    base_sd = st.load_file(str(base_path), device='cpu')
```

with a switch on the suffix:

```python
    if base_path.suffix == '.pt':
        base_sd = torch.load(base_path, map_location='cpu', weights_only=True)
    else:
        base_sd = st.load_file(str(base_path), device='cpu')
```

**(b)** Still in `merge_lora`, make an unmatched LoRA key fatal. Replace:

```python
        base_key = module_key + '.weight'
        if base_key not in out_sd:
            skipped.append(base_key)
            continue
```

with:

```python
        base_key = module_key + '.weight'
        if base_key not in out_sd:
            raise SystemExit(
                f"[Merge] LoRA key has no matching base weight: {base_key}. "
                "A silently skipped adapter yields a checkpoint that loads cleanly "
                "and then generates subtly wrong video."
            )
```

Then delete the now-dead `skipped = []` initialiser and the `if skipped:` warning
block that printed the first five.

**(c)** Replace the save:

```python
    print(f"[Merge] saving to {out_path} ...")
    st.save_file(out_sd, str(out_path))
```

with:

```python
    print(f"[Merge] saving to {out_path} ...")
    if out_path.suffix == '.pt':
        torch.save(out_sd, out_path)
    else:
        st.save_file(out_sd, str(out_path))
```

**(d)** Let `is_merge_complete` check the right filename:

```python
def is_merge_complete(out_dir: Path, expect_he: bool = True,
                      out_name: str = 'diffusion_pytorch_model.safetensors') -> bool:
    """Return True if the output directory already holds a finished merge."""
    transformer_path = out_dir / out_name
```

(the rest of the function body is unchanged).

**(e)** In `main`, replace:

```python
    out_path = out_dir / 'diffusion_pytorch_model.safetensors'
```

with:

```python
    # Mirror the base's format. The released stage2b checkpoint is transformer.pt,
    # so a .pt base yields transformer.pt and the merged directory ends up shaped
    # exactly like weights/alaya-world-ar -- which the inference and WBench configs
    # already load through paths.resume_checkpoint, with no code change.
    out_name = 'transformer.pt' if base_path.suffix == '.pt' else 'diffusion_pytorch_model.safetensors'
    out_path = out_dir / out_name
```

and pass it through the idempotency check:

```python
    if not args.force and is_merge_complete(out_dir, expect_he=args.copy_history_encoder, out_name=out_name):
```

Finally, update the module docstring to record that a `.pt` base produces
`transformer.pt`, consumable by `paths.resume_checkpoint` unchanged.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
/home/zhantaoy/miniforge3/envs/alayaworld/bin/python -m pytest tests/test_merge_lora.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/tools/merge_lora_for_rollout.py tests/test_merge_lora.py
git commit -m "feat(tools): merge LoRA into a .pt base, emitting transformer.pt

The released stage2b checkpoint is transformer.pt, so a low-compute LoRA merges
into a directory shaped like weights/alaya-world-ar and the existing inference
and WBench configs consume it unchanged. Unmatched LoRA keys now fail loudly.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: Text-embedding precache and the launcher

**Files:**
- Create: `scripts/tools/precache_train_text_embeds.py`
- Create: `scripts/finetune/lowcompute_5x4090.sh`

**Interfaces:**
- Consumes: `configs/stage2b_arsft_lowcompute.yaml` (Task 3), the dataset from Task 2.
- Produces: a populated `cache/text_embed_spatialvid/`, and `scripts/finetune/lowcompute_5x4090.sh` which Tasks 7-8 launch.

- [ ] **Step 1: Write the precache tool**

`alaya/data/text_embed_cache.py` already exports `enumerate_all_prompts(cfg, dataset)`,
which walks `dataset.samples`, reads each caption json and returns every reachable
prompt string including the per-source `caption_prefix`. Use it rather than
re-deriving the enumeration.

Create `scripts/tools/precache_train_text_embeds.py`:

```python
"""Pre-encode every training caption into runtime.text_embed_cache_dir.

Gemma-3-12B is 24GB in bf16 and cannot sit next to the sharded DiT on a 24GB card,
so low-compute training runs with ALAYA_SKIP_TEXT_ENCODER=1 and serves prompts from
disk. A cache miss then raises rather than silently mis-encoding, so the cache has
to be complete before training starts.

That is only sound when the prompt set is finite, which is a property of the config
rather than of this script -- so the finiteness is asserted here instead of assumed.

    CUDA_VISIBLE_DEVICES=0,1 ALAYA_GEMMA_MAX_MEMORY="0=13GiB,1=13GiB" \
      python scripts/tools/precache_train_text_embeds.py \
          --config configs/stage2b_arsft_lowcompute.yaml --device-map auto

Re-runs are incremental: prompts already on disk are skipped.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _assert_finite_prompt_set(cfg) -> None:
    """Refuse to run when the config can draw a prompt this pass cannot enumerate.

    Training reads from the cache with the encoder unloaded, so anything drawn at
    random from a field that is not enumerated here becomes a hard failure hours
    into a run. Better to fail now, naming the key.
    """
    from fastvideo.dataset.t2v_datasets import MultiSourceVideoDataset

    if float(getattr(cfg.data, "abstract_caption_prob", 0.0)) != 0.0:
        raise SystemExit(
            "data.abstract_caption_prob must be 0.0: it draws from an abstract_caption "
            "field that this pass does not enumerate"
        )
    for source in cfg.data.sources:
        source_cfg = MultiSourceVideoDataset.SOURCE_CONFIGS.get(source)
        if source_cfg is None:
            raise SystemExit(f"unknown data source: {source!r}")
        if source_cfg.get("use_segment_caption", False):
            raise SystemExit(
                f"source {source!r} uses segment captions, whose time-windowed selection "
                "is not enumerable ahead of time; set use_segment_caption: False"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="the same yaml the training run uses")
    parser.add_argument("--cache-dir", default=None, help="override runtime.text_embed_cache_dir")
    parser.add_argument("--device-map", default="auto", help='accelerate device_map for Gemma')
    parser.add_argument("--dry-run", action="store_true", help="enumerate and report, do not load Gemma")
    args = parser.parse_args()

    from alaya.config.loader import load_config
    from alaya.data.dataloader import build_train_dataloader
    from alaya.data.text_embed_cache import cache_path, disk_put, enumerate_all_prompts
    from alaya.utils.distributed import init_distributed

    cfg = load_config(args.config)
    _assert_finite_prompt_set(cfg)

    cache_dir = args.cache_dir or cfg.runtime.text_embed_cache_dir
    if not cache_dir:
        raise SystemExit("no cache dir: set runtime.text_embed_cache_dir or pass --cache-dir")
    os.makedirs(cache_dir, exist_ok=True)

    # No torchrun env here, so this is a plain single-process state with no
    # process group -- init_distributed() short-circuits when world_size is 1.
    dist_state = init_distributed()
    loader = build_train_dataloader(cfg, dist_state)
    dataset = loader.dataset
    # The loader may hand back a concat wrapper; mirror how RolloutTrainer.setup()
    # reaches the subsets that actually carry .samples.
    subsets = [dataset] if hasattr(dataset, "samples") else list(getattr(dataset, "datasets", []))

    prompts: list[str] = []
    seen: set[str] = set()
    for subset in subsets:
        for text in enumerate_all_prompts(cfg, subset):
            if text and text not in seen:
                seen.add(text)
                prompts.append(text)

    missing = [p for p in prompts if not os.path.exists(cache_path(cache_dir, p))]
    print(
        f"[Precache] {len(prompts)} distinct prompts, "
        f"{len(prompts) - len(missing)} already cached, {len(missing)} to encode",
        flush=True,
    )
    if args.dry_run or not missing:
        return 0

    from alaya.model.loader import load_text_encoder

    os.environ["ALAYA_GEMMA_DEVICE_MAP"] = args.device_map
    dtype = torch.bfloat16
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    text_encoder, encode_text = load_text_encoder(
        cfg.paths.effective_transformer, cfg.paths.gemma, device=device, dtype=dtype
    )

    t0 = time.time()
    for n, prompt in enumerate(missing, 1):
        with torch.no_grad():
            output = encode_text(text_encoder, [prompt])
        ctx = output[0][0] if isinstance(output, list) and output[0].dim() == 3 else output[0]
        disk_put(cache_dir, prompt, ctx)
        if n % 25 == 0 or n == len(missing):
            rate = n / max(time.time() - t0, 1e-6)
            print(
                f"[Precache] {n}/{len(missing)} ({100 * n / len(missing):.1f}%) "
                f"{rate:.2f} it/s ETA {(len(missing) - n) / max(rate, 1e-6) / 60:.1f}min",
                flush=True,
            )
    print(f"[Precache] done in {(time.time() - t0) / 60:.1f}min -> {cache_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Write the launcher**

Create `scripts/finetune/lowcompute_5x4090.sh`:

```bash
#!/usr/bin/env bash
# ============================================================================
# Low-compute launcher: stage2b fine-tuning on five 24GB cards.
#
#   bash scripts/finetune/lowcompute_5x4090.sh
#
# Sets only environment; all training logic stays in scripts/finetune/train.sh.
# Every variable below is already honoured by the existing code -- see
# docs/LOWCOMPUTE.md for why each one is needed.
# ============================================================================
set -euo pipefail

# GPU 5 is deliberately excluded and must stay free.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4}

# The 13B DiT is 26GB in bf16 and does not fit unsharded on a 24GB card. Building
# it on CPU lets FSDP move and shard it one attention block at a time.
export ALAYA_INIT_TRANSFORMER_ON_CPU=1

# LTX23Model is constructed in fp32 (~52GB) before the bf16 cast, so five
# concurrent builds would need ~260GB of host RAM. Load one rank at a time.
export ALAYA_SERIAL_MODEL_LOAD=1

# Gemma-3-12B is 24GB in bf16; prompts come from the on-disk cache instead.
# Fill it first with scripts/tools/precache_train_text_embeds.py.
export ALAYA_SKIP_TEXT_ENCODER=1

# FA3 cannot coexist with the flash-attn-3 bundled inside the xformers that ViGeo
# pulls in -- they register the same torch operator namespace and abort in C++.
export ALAYA_USE_FA3=0

export CONFIG_PATH=${CONFIG_PATH:-configs/stage2b_arsft_lowcompute.yaml}
exec bash "$(dirname "$0")/train.sh"
```

```bash
chmod +x scripts/finetune/lowcompute_5x4090.sh
```

- [ ] **Step 3: Verify the precache enumerates correctly**

```bash
cd /home/zhantaoy/Projects/Python/Research/y2026/WM-AutoResearch/WorldModel
/home/zhantaoy/miniforge3/envs/alayaworld/bin/python scripts/tools/precache_train_text_embeds.py \
    --config configs/stage2b_arsft_lowcompute.yaml --dry-run
```

Expected: roughly **two** prompts per clip plus the negative prompt, and no Gemma load.
`_clip_prompts` reaches both `overall.short_prompt` and `overall_caption`, so a
4-clip corpus enumerates ~8-9 distinct strings. Caching a superset of what
training draws is correct and costs nothing.

- [ ] **Step 4: Verify the finiteness guard actually fires**

```bash
sed 's/abstract_caption_prob: 0.0/abstract_caption_prob: 0.5/' \
    configs/stage2b_arsft_lowcompute.yaml > /tmp/bad_lowcompute.yaml
/home/zhantaoy/miniforge3/envs/alayaworld/bin/python scripts/tools/precache_train_text_embeds.py \
    --config /tmp/bad_lowcompute.yaml --dry-run; echo "exit=$?"
rm /tmp/bad_lowcompute.yaml
```

Expected: a non-zero exit and a message naming `abstract_caption_prob`.

- [ ] **Step 5: Commit**

```bash
git add scripts/tools/precache_train_text_embeds.py scripts/finetune/lowcompute_5x4090.sh
git commit -m "feat(tools): training text-embed precache and the 5x4090 launcher

Training runs with the 24GB text encoder unloaded, so the prompt set must be
finite; the precache tool refuses to run on a config that cannot guarantee it.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: Full dataset, precache, and the memory smoke run

The first task that touches a GPU. Nothing about memory is claimed before this produces numbers.

**Files:**
- No source changes expected. If the smoke run OOMs, edit `configs/stage2b_arsft_lowcompute.yaml` along the escalation ladder and record what it took.

**Interfaces:**
- Consumes: everything from Tasks 1-6.
- Produces: a populated `data/`, a populated `cache/text_embed_spatialvid/`, and measured per-rank peak VRAM plus seconds per step.

- [ ] **Step 1: Build the full 200-clip corpus**

```bash
cd /home/zhantaoy/Projects/Python/Research/y2026/WM-AutoResearch/WorldModel
/home/zhantaoy/miniforge3/envs/alayaworld/bin/python scripts/tools/prepare_spatialvid.py --num-clips 200
df -h / | tail -1
du -sh data/
```

Expected: ~200 rows in the jsonl, roughly 500-700MB of video, and free space still comfortably above 40GB.

- [ ] **Step 2: Re-run the Task 2 Step 3 verification over all 200 clips**

Same script as Task 2 Step 3. Expected: `OK`, and no clip with a zero-motion trajectory.

- [ ] **Step 3: Fill the text-embedding cache**

```bash
CUDA_VISIBLE_DEVICES=0,1 ALAYA_GEMMA_MAX_MEMORY="0=13GiB,1=13GiB" \
  /home/zhantaoy/miniforge3/envs/alayaworld/bin/python \
      scripts/tools/precache_train_text_embeds.py \
      --config configs/stage2b_arsft_lowcompute.yaml --device-map auto
ls cache/text_embed_spatialvid | wc -l
```

Expected: ~400 cached entries (about two per clip, see Task 6 Step 3) and a clean exit.

- [ ] **Step 4: Run the 30-step smoke test**

Temporarily set `optimizer.max_steps: 30` in the config, then:

```bash
LOG_FILTER=all bash scripts/finetune/lowcompute_5x4090.sh 2>&1 | tee /tmp/smoke.log | tail -40
```

Watch VRAM from a second shell:

```bash
nvidia-smi --query-gpu=index,memory.used --format=csv -l 5
```

Expected: startup takes several minutes (ranks build the model serially), then `[Train] step=` lines. With `grad_accum_steps: 4` a logged step consumes four batches, so 30 steps is 120 batches.

- [ ] **Step 5: Record the measurements**

```bash
grep -c '^\[Train\] step=' /tmp/smoke.log
grep '^\[Train\] step=' /tmp/smoke.log | head -3
grep '^\[Train\] step=' /tmp/smoke.log | tail -3
grep -iE 'out of memory|CUDA error' /tmp/smoke.log | head
```

Write down: peak `memory.used` per card, seconds per step, and the first and last loss values. **If any card exceeded ~22.5GB or the run OOM'd**, apply the escalation ladder from the spec in order — `vigeo_cache_budget` 65536 → 32768, then `lora.rank` 64 → 32, then resolution 416x736 → 352x608 — re-running this step after each change and recording which one was needed.

- [ ] **Step 6: Restore max_steps and commit any config change**

Set `optimizer.max_steps` back to `600`. If the ladder was walked, commit the config change with a message stating the measured peak that forced it. If nothing changed, there is nothing to commit — record the numbers for Task 8's documentation instead.

---

### Task 8: Acceptance run, merge, rollout, and documentation

**Files:**
- Create: `docs/LOWCOMPUTE.md`
- Modify: `README.md` (one Release-Roadmap-adjacent line pointing at the new doc)

**Interfaces:**
- Consumes: everything above.
- Produces: `outputs/stage2b_lowcompute/checkpoint-600-merged/transformer.pt` and one rollout video.

- [ ] **Step 1: Run the acceptance training**

```bash
cd /home/zhantaoy/Projects/Python/Research/y2026/WM-AutoResearch/WorldModel
LOG_FILTER=all bash scripts/finetune/lowcompute_5x4090.sh 2>&1 | tee /tmp/accept.log | grep '^\[Train\] step='
```

Expected: 600 optimizer steps, checkpoints every 100 under `outputs/stage2b_lowcompute/`, each containing `lora.safetensors`, `history_encoder.pt` and `trainer_state.pt`.

- [ ] **Step 2: Check the loss actually moved**

```bash
grep -o 'loss=[0-9.]*' /tmp/accept.log | head -20
grep -o 'loss=[0-9.]*' /tmp/accept.log | tail -20
grep -o 'grad=[0-9.]*' /tmp/accept.log | tail -5
```

Expected: finite losses throughout and no `nan` in the logged `grad=` of an optimizer step. A flat loss is a reportable result, not a failure to hide — 600 steps on 200 clips is a small run.

- [ ] **Step 3: Merge the LoRA**

```bash
/home/zhantaoy/miniforge3/envs/alayaworld/bin/python scripts/tools/merge_lora_for_rollout.py \
    --ckpt_dir outputs/stage2b_lowcompute/checkpoint-600 \
    --base_transformer weights/alaya-world-ar/transformer.pt \
    --output outputs/stage2b_lowcompute/checkpoint-600-merged \
    --lora_rank 64 --lora_alpha 64
ls -la outputs/stage2b_lowcompute/checkpoint-600-merged
```

Expected: `transformer.pt` (~26GB) and `history_encoder.pt`. **Check free disk first** — 26GB against ~40GB free is tight; delete intermediate checkpoints if needed.

- [ ] **Step 4: Render a rollout from the merged checkpoint**

Copy `configs/infer_i2v_camera_ar.yaml` to `/tmp/infer_lowcompute.yaml`, point its `paths.resume_checkpoint` and `paths.history_encoder` at the merged directory, then:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4 ALAYA_INIT_TRANSFORMER_ON_CPU=1 ALAYA_SERIAL_MODEL_LOAD=1 \
ALAYA_USE_FA3=0 CONFIG_PATH=/tmp/infer_lowcompute.yaml \
  bash scripts/infer/generate_video.sh \
      --image playground/case1/case1_image.png \
      --prompt "$(cat playground/case1/case1_prompt.txt)" \
      --rounds 3
```

Expected: an mp4 is written. This proves the merged checkpoint loads and generates through the untouched inference path. Note that the text encoder IS loaded here (no `ALAYA_SKIP_TEXT_ENCODER`), so this needs the prompt encoded live — if it OOMs, precache the prompt or spread Gemma with `ALAYA_GEMMA_DEVICE_MAP=auto`.

- [ ] **Step 5: Write `docs/LOWCOMPUTE.md`**

Structure it after `docs/WBENCH.md`, which is the house style for this kind of document. Required sections:

1. **What this is** — LoRA fine-tuning of the released stage2b checkpoint on five 24GB cards, and what it is not (not a reproduction of stage2b; not a quality claim).
2. **Setup** — env, weights, the SpatialVID-HQ gate, `prepare_spatialvid.py`, the text-embed precache.
3. **Run** — `scripts/finetune/lowcompute_5x4090.sh`, and what each exported variable buys.
4. **What was changed in this repo, and why** — a table in the same shape as `docs/WBENCH.md` section 4, one row per edit, each stating that it is inert at its default.
5. **Recipe deltas from stage2b** — the table from the spec, including **next_forcing being off** and what that costs.
6. **Measured numbers** — per-rank peak VRAM, seconds per step, startup time, and which escalation-ladder rungs were needed. Real numbers from Tasks 7 and 8 only.
7. **Pose interpolation** — state plainly that SpatialVID annotates at ~5Hz and the poses are slerped onto the video frame grid, that this is the only derived quantity in the pipeline, and why the alternatives were rejected.
8. **Using the result** — the merge command and the fact that a merged directory drops into the existing inference and WBench configs unchanged.
9. **Limitations** — the run length, the 200-clip corpus, 416x736 versus the checkpoint's native 544x960, and that no quality improvement is claimed.

- [ ] **Step 6: Final non-interference verification**

```bash
/home/zhantaoy/miniforge3/envs/alayaworld/bin/python -m pytest tests/ -v
for c in stage0_precache stage1_pretrain_bidir stage2a_histpretrain stage2b_arsft_vigeo stage3_dmd_vigeo wbench_full; do
  echo "=== $c ==="
  DESCRIBE=1 CONFIG_PATH=configs/$c.yaml LOG_FILTER=all bash scripts/finetune/train.sh 2>&1 | tail -3
done
git diff --stat e1f5748..HEAD
git diff e1f5748..HEAD -- alaya/ fastvideo/ scripts/tools/merge_lora_for_rollout.py
```

Review the diff by hand and confirm: no file under `inference/`, `reactor/`, `alaya/inference/` is touched; `configs/stage0..stage3` and `configs/wbench_full.yaml` are untouched; every edit to `alaya/` and `fastvideo/` is additive and default-off.

- [ ] **Step 7: Clean up and commit**

```bash
rm -rf /tmp/svprobe /tmp/infer_lowcompute.yaml
df -h / | tail -1
git add docs/LOWCOMPUTE.md README.md
git commit -m "docs: low-compute stage2b fine-tuning on 5x RTX 4090

LoRA on a frozen FSDP-sharded base, trained on a 200-clip SpatialVID-HQ slice.
Records the measured VRAM and step times, the recipe deltas from stage2b, and
that camera poses are interpolated from the dataset's ~5Hz annotations.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Notes for the executor

- **Tasks 1-6 need no GPU.** Only Tasks 7 and 8 do. If GPUs are busy, the first six are still fully executable.
- **Task 7 Step 1 downloads ~3GB.** It takes a while. Run it before you need it.
- `/tmp/svprobe` may hold ~2.1GB of SpatialVID annotations left from the design investigation. Task 8 Step 7 deletes it; delete it earlier if disk gets tight.
- **If a step's expectation does not match reality, stop and report it** rather than adjusting the expectation. The numbers in this plan that came from measurement are marked as verified facts; the VRAM budget in the spec is explicitly an estimate and is allowed to be wrong.
