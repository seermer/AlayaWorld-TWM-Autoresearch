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

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

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


def _atomic_write(target: Path, data: bytes) -> None:
    """Write `data` to `target` via a temporary sibling file, then rename atomically.

    A run killed mid-write (SIGKILL/OOM/disk-full) would otherwise leave a partial
    file at `target`; a re-run's `exists()` check would then silently reuse the
    truncated file. os.replace() is atomic within a filesystem, so a killed run
    leaves either nothing or a complete file at `target`, never a partial one.
    """
    part = target.with_name(target.name + ".part")
    if part.exists():
        part.unlink()
    part.write_bytes(data)
    os.replace(part, target)


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
            except (ValueError, KeyError, TypeError):
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
                    _atomic_write(target, handle.read())
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
                _atomic_write(destination, handle.read())
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
    except (FileNotFoundError, ValueError, EOFError) as exc:
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


def invalidate_sample_cache(source: str) -> list[Path]:
    """Drop MultiSourceVideoDataset's cached sample list for this source.

    That cache is keyed on the source name, the jsonl FILENAME and the pose subdir
    (t2v_datasets.py:501-514) -- not on the jsonl's contents, size or mtime. Re-running
    this importer with a different --num-clips therefore rewrites the same jsonl path
    and the stale pickle is silently reused, so training would run on the previous
    clip count while every log line looks healthy. Deleting it here keeps the
    invalidation next to the write that causes it.
    """
    env_dir = os.environ.get("ALAYA_DATASET_CACHE_DIR")
    cache_dir = Path(env_dir) if env_dir else (REPO_ROOT / ".cache" / "dataset")
    if not cache_dir.is_dir():
        print(f"[prepare] no sample cache dir at {cache_dir}, nothing to invalidate", flush=True)
        return []
    removed = []
    for stale in cache_dir.glob(f"multi_source_*{source}*.pkl"):
        stale.unlink()
        removed.append(stale)
    return removed


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
    jsonl_bytes = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records).encode("utf-8")
    _atomic_write(jsonl, jsonl_bytes)

    for stale in invalidate_sample_cache(ann_root.name):
        print(f"[prepare] dropped stale sample cache {stale}", flush=True)

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
