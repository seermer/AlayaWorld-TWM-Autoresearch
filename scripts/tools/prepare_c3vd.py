"""Build the C3VD overfit set: 8 s colonoscopy clips that turn into a pencil sketch.

    python scripts/tools/prepare_c3vd.py              # all C3VD v1 clips (29)
    python scripts/tools/prepare_c3vd.py --limit 2    # quick smoke test

Reads C3VD v1 (1350x1080 PNG frames + per-frame camera-to-world poses) from the
Hugging Face mirror brunoeducsantos/c3vd, downloading only the frames inside the chosen
clips. Each clip is centre-cropped to 736x416, blended into a grayscale pencil sketch
during the third rollout chunk (see alaya/data/c3vd.py) and encoded as a 30 fps H.264 mp4.

Every clip is then cut into one training window per rollout round (25-frame prefix +
32-frame target, sampled at 24 fps exactly as the evaluation loader samples the clip),
captioned with the phase of its target chunk. Positions inside a window are relative, so
this is how the model learns *when* the transition happens:

    data/Video/c3vd/<clip>.mp4                        8 s clips (evaluation ground truth)
    data/Video/c3vd/win/<clip>_r<round>.mp4           57-frame training windows
    data/Annotation/c3vd/c3vd.jsonl                   training: all windows
    data/Annotation/c3vd/c3vd_eval.jsonl              evaluation: 10 fixed clips
    data/Annotation/c3vd/{caption,pose}/<clip>.*      clip annotations
    data/Annotation/c3vd/{win_caption,win_pose}/*     window annotations

c3vd_eval.jsonl is the first clip of 8 sequences, round-robin over colon segments, plus
both clips of --eval-sequence. Re-runs skip videos that already exist.
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
    PHASE_CAPTIONS,
    POSE_SCALE_MM_TO_CM,
    ROUNDS,
    TARGET_FPS,
    WINDOW_FRAMES,
    caption_json,
    crop_resize,
    parse_c3vd_pose,
    pencil_blend,
    phase_for_round,
    pinhole_k,
    plan_clips,
    scale_translation,
    segment_name,
    sketch_weight,
    window_native_indices,
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


def _write_mp4(path: Path, frames_rgb, fps: float) -> None:
    import imageio.v2 as iio

    part = path.with_name(path.name + ".part.mp4")
    writer = iio.get_writer(part, fps=fps, codec="libx264", quality=None,
                            ffmpeg_params=["-crf", "16"], macro_block_size=16)
    for frame in frames_rgb:
        writer.append_data(frame)
    writer.close()
    os.replace(part, path)


def _write_pose(path: Path, cam_c2w: np.ndarray) -> None:
    buf = io.BytesIO()
    np.savez(buf, cam_c2w=cam_c2w.astype(np.float32), intrinsics=pinhole_k())
    _atomic(path, buf.getvalue())


def list_sequences(repo: str) -> list[str]:
    tree = json.loads(_get(f"{HF}/api/datasets/{repo}/tree/main"))
    return sorted(x["path"] for x in tree if x["type"] == "directory")


def build_clip(repo: str, seq: str, start: int, c2w_mm: np.ndarray, video_dir: Path,
               ann_root: Path, workers: int) -> dict:
    clip_id = f"{seq}_{start:04d}"
    video = video_dir / f"{clip_id}.mp4"
    pose = ann_root / "pose" / f"{clip_id}.npz"
    record = {
        "video": f"c3vd/{clip_id}.mp4",
        "prompt": f"c3vd/caption/{clip_id}.json",
        "pose": f"c3vd/pose/{clip_id}.npz",
        "num_frames": CLIP_FRAMES,
        "sequence": seq,
        "segment": segment_name(seq),
        "clip_start": start,
    }
    # The clip caption is what validation encodes before switching to the per-round schedule;
    # the first round is raw, so it is the raw phase prompt. Always rewritten (cheap).
    _atomic(ann_root / "caption" / f"{clip_id}.json", json.dumps(caption_json(PHASE_CAPTIONS["raw"]), indent=1).encode())
    if video.exists() and pose.exists():
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
    rgb = (cv2.cvtColor(pencil_blend(crop_resize(f), sketch_weight(j / CLIP_FPS)), cv2.COLOR_BGR2RGB)
           for j, f in enumerate(frames))
    _write_mp4(video, rgb, CLIP_FPS)
    _write_pose(pose, scale_translation(c2w_mm[start:start + CLIP_FRAMES], POSE_SCALE_MM_TO_CM))
    return record


def build_windows(clip: dict, video_root: Path, ann_root: Path) -> list[dict]:
    """One 24 fps training window per rollout round, cut from the finished clip."""
    import decord

    clip_id = Path(clip["video"]).stem
    cam_c2w = np.load(ann_root / "pose" / f"{clip_id}.npz")["cam_c2w"]
    reader = None
    records = []
    for r in range(1, ROUNDS + 1):
        win_id = f"{clip_id}_r{r}"
        phase = phase_for_round(r)
        idx = window_native_indices(r)
        video = video_root / "win" / f"{win_id}.mp4"
        _atomic(ann_root / "win_caption" / f"{win_id}.json", json.dumps(caption_json(PHASE_CAPTIONS[phase]), indent=1).encode())
        if not video.exists():
            reader = reader or decord.VideoReader(str(video_root / f"{clip_id}.mp4"))
            _write_mp4(video, list(reader.get_batch(idx).asnumpy()), TARGET_FPS)
        _write_pose(ann_root / "win_pose" / f"{win_id}.npz", cam_c2w[idx])
        records.append({
            "video": f"c3vd/win/{win_id}.mp4",
            "prompt": f"c3vd/win_caption/{win_id}.json",
            "pose": f"c3vd/win_pose/{win_id}.npz",
            "num_frames": WINDOW_FRAMES,
            "sequence": clip["sequence"],
            "segment": clip["segment"],
            "clip_start": clip["clip_start"],
            "round": r,
            "phase": phase,
        })
    return records


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
    # contents, so a re-import must drop it or training silently sees the old sample list.
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
    for d in (video_dir / "win", ann_root / "pose", ann_root / "caption", ann_root / "win_pose", ann_root / "win_caption"):
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

    clips, windows = [], []
    t0 = time.time()
    for n, (seq, start, c2w) in enumerate(plan, 1):
        clips.append(build_clip(args.repo, seq, start, c2w, video_dir, ann_root, args.workers))
        windows += build_windows(clips[-1], video_dir, ann_root)
        print(f"[c3vd] {n}/{len(plan)} {clips[-1]['video']} + {ROUNDS} windows ({time.time() - t0:.0f}s)", flush=True)

    picked = pick_eval_training_clips([c for c in clips if c["sequence"] != args.eval_sequence],
                                      args.eval_training_clips)
    evals = picked + [c for c in clips if c["sequence"] == args.eval_sequence]

    def dump(name: str, rows: list[dict]) -> None:
        _atomic(ann_root / name, "".join(json.dumps(r) + "\n" for r in rows).encode())

    dump("c3vd.jsonl", windows)
    dump("c3vd_eval.jsonl", evals)
    _atomic(ann_root / "eval_manifest.json",
            json.dumps({"eval_clips": {Path(r["video"]).stem: "train" for r in evals}}, indent=1).encode())
    invalidate_sample_cache()
    print(f"[c3vd] clips={len(clips)} training windows={len(windows)} eval clips={len(evals)} "
          f"in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
