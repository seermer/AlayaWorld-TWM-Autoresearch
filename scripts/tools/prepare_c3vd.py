"""Build the C3VD overfit set: 8 s colonoscopy clips that turn into a pencil sketch.

    python scripts/tools/prepare_c3vd.py              # all C3VD v1 clips (~29)
    python scripts/tools/prepare_c3vd.py --limit 2    # quick smoke test

Reads C3VD v1 (1350x1080 PNG frames + per-frame camera-to-world poses) from the
Hugging Face mirror brunoeducsantos/c3vd, downloading only the frames inside the chosen
clips. Each clip is centre-cropped to 736x416, blended into a grayscale pencil sketch
during the third rollout chunk (see alaya/data/c3vd.py), encoded as a 30 fps H.264 mp4,
and written with its poses (centimetres) and caption in the layout the loader reads:

    data/Video/c3vd/<clip>.mp4
    data/Annotation/c3vd/{c3vd.jsonl, c3vd_eval.jsonl, eval_manifest.json}
    data/Annotation/c3vd/{caption,pose}/<clip>.{json,npz}

Every clip goes into c3vd.jsonl (the overfit test trains on all of them). c3vd_eval.jsonl
is a fixed 10-clip subset: the first clip of 8 sequences, round-robin over colon segments,
plus both clips of --eval-sequence (one of which starts mid-sequence). Re-runs skip clips
that are already complete.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from alaya.data.c3vd import (  # noqa: E402
    CLIP_FPS,
    CLIP_FRAMES,
    POSE_SCALE_MM_TO_CM,
    caption_for,
    crop_resize,
    parse_c3vd_pose,
    pencil_blend,
    pinhole_k,
    plan_clips,
    scale_translation,
    segment_name,
    sketch_weight,
)

HF = "https://huggingface.co"


def _get(url: str, retries: int = 4) -> bytes:
    token = os.environ.get("HF_TOKEN", "").strip()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r:
                return r.read()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def _atomic(path: Path, data: bytes) -> None:
    part = path.with_name(path.name + ".part")
    part.write_bytes(data)
    os.replace(part, path)


def list_sequences(repo: str) -> list[str]:
    tree = json.loads(_get(f"{HF}/api/datasets/{repo}/tree/main"))
    return sorted(x["path"] for x in tree if x["type"] == "directory")


def build_clip(repo: str, seq: str, start: int, c2w_mm: np.ndarray, video_dir: Path,
               ann_root: Path, workers: int) -> dict:
    clip_id = f"{seq}_{start:04d}"
    video = video_dir / f"{clip_id}.mp4"
    pose = ann_root / "pose" / f"{clip_id}.npz"
    caption = ann_root / "caption" / f"{clip_id}.json"
    record = {
        "video": f"c3vd/{clip_id}.mp4",
        "prompt": f"c3vd/caption/{clip_id}.json",
        "pose": f"c3vd/pose/{clip_id}.npz",
        "num_frames": CLIP_FRAMES,
        "sequence": seq,
        "segment": segment_name(seq),
        "clip_start": start,
    }
    if video.exists() and pose.exists() and caption.exists():
        return record

    def fetch(i: int) -> np.ndarray:
        # zero-padded names only: the mirror also carries unpadded duplicates for some sequences
        raw = _get(f"{HF}/datasets/{repo}/resolve/main/{seq}/{i:04d}_color.png")
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"could not decode {seq}/{i:04d}_color.png")
        return img

    with ThreadPoolExecutor(max_workers=workers) as pool:
        frames = list(pool.map(fetch, range(start, start + CLIP_FRAMES)))

    import imageio.v2 as iio

    part = video.with_name(video.name + ".part.mp4")
    writer = iio.get_writer(part, fps=CLIP_FPS, codec="libx264", quality=None,
                            ffmpeg_params=["-crf", "16"], macro_block_size=16)
    for j, frame in enumerate(frames):
        out = pencil_blend(crop_resize(frame), sketch_weight(j / CLIP_FPS))
        writer.append_data(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    writer.close()
    os.replace(part, video)

    window = scale_translation(c2w_mm[start:start + CLIP_FRAMES], POSE_SCALE_MM_TO_CM)
    buf = io.BytesIO()
    np.savez(buf, cam_c2w=window.astype(np.float32), intrinsics=pinhole_k())
    _atomic(pose, buf.getvalue())
    _atomic(caption, json.dumps(caption_for(seq), indent=1).encode())
    return record


def pick_eval_training_clips(records: list[dict], count: int) -> list[dict]:
    """First clip of distinct sequences, round-robin over colon segments."""
    by_segment: dict[str, list[dict]] = {}
    for r in sorted(records, key=lambda r: (r["segment"], r["sequence"], r["clip_start"])):
        if r["clip_start"] == 0:
            by_segment.setdefault(r["segment"], []).append(r)
    picked: list[dict] = []
    while len(picked) < count and any(by_segment.values()):
        for seg in sorted(by_segment):
            if by_segment[seg] and len(picked) < count:
                picked.append(by_segment[seg].pop(0))
    return picked


def invalidate_sample_cache() -> None:
    # MultiSourceVideoDataset keys its cached sample list on the jsonl filename, not its
    # contents, so a re-import must drop it or training silently sees the old clip list.
    cache_dir = Path(os.environ.get("ALAYA_DATASET_CACHE_DIR", REPO_ROOT / ".cache" / "dataset"))
    for stale in cache_dir.glob("multi_source_*c3vd*.pkl") if cache_dir.is_dir() else []:
        stale.unlink()
        print(f"[c3vd] dropped stale sample cache {stale}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="brunoeducsantos/c3vd")
    ap.add_argument("--eval-sequence", default="sigmoid_t3_b",
                    help="sequence whose clips are all added to the eval subset")
    ap.add_argument("--eval-training-clips", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="build only this many clips (smoke test)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--video-dir", default="data/Video/c3vd")
    ap.add_argument("--annotation-dir", default="data/Annotation/c3vd")
    args = ap.parse_args()

    video_dir, ann_root = Path(args.video_dir), Path(args.annotation_dir)
    for d in (video_dir, ann_root / "pose", ann_root / "caption"):
        d.mkdir(parents=True, exist_ok=True)

    plan = []
    for seq in list_sequences(args.repo):
        c2w = parse_c3vd_pose(_get(f"{HF}/datasets/{args.repo}/resolve/main/{seq}/pose.txt").decode())
        starts = plan_clips(len(c2w))
        print(f"[c3vd] {seq}: {len(c2w)} frames -> {len(starts)} clip(s)", flush=True)
        plan += [(seq, s, c2w) for s in starts]
    if args.eval_sequence not in {seq for seq, _, _ in plan}:
        raise SystemExit(f"eval sequence {args.eval_sequence!r} yields no clips")
    if args.limit:
        plan = plan[: args.limit]

    records = []
    t0 = time.time()
    for n, (seq, start, c2w) in enumerate(plan, 1):
        records.append(build_clip(args.repo, seq, start, c2w, video_dir, ann_root, args.workers))
        print(f"[c3vd] {n}/{len(plan)} {records[-1]['video']} ({time.time() - t0:.0f}s)", flush=True)

    train = records
    picked = pick_eval_training_clips([r for r in train if r["sequence"] != args.eval_sequence],
                                      args.eval_training_clips)
    evals = picked + [r for r in train if r["sequence"] == args.eval_sequence]

    def dump(name: str, rows: list[dict]) -> None:
        _atomic(ann_root / name, "".join(json.dumps(r) + "\n" for r in rows).encode())

    dump("c3vd.jsonl", train)
    dump("c3vd_eval.jsonl", evals)
    manifest = {"eval_clips": {Path(r["video"]).stem: "train" for r in evals}}
    _atomic(ann_root / "eval_manifest.json", json.dumps(manifest, indent=1).encode())
    invalidate_sample_cache()
    print(f"[c3vd] train={len(train)} eval={len(evals)} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
